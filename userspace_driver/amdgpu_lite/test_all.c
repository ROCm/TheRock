#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/eventfd.h>
#include <stdint.h>
#include <string.h>

#include "amdgpu_lite.h"

static int test_get_info(int fd, struct amdgpu_lite_get_info *info) {
    printf("[1] GET_INFO...\n");
    memset(info, 0, sizeof(*info));
    if (ioctl(fd, AMDGPU_LITE_IOC_GET_INFO, info) < 0) {
        perror("  GET_INFO");
        return 1;
    }
    printf("  Vendor=0x%04x Device=0x%04x Rev=0x%02x\n",
        info->vendor_id, info->device_id, info->revision_id);
    printf("  VRAM=%llu MB, BARs=%u (MMIO=%u VRAM=%u Door=%u)\n",
        (unsigned long long)info->vram_size / (1024*1024),
        info->num_bars, info->mmio_bar_index, info->vram_bar_index,
        info->doorbell_bar_index);
    printf("  GART: bus=0x%llx size=%llu VA_start=0x%llx\n",
        (unsigned long long)info->gart_table_bus_addr,
        (unsigned long long)info->gart_table_size,
        (unsigned long long)info->gart_gpu_va_start);
    printf("  PASS\n\n");
    return 0;
}

static int test_mmio_mmap(int fd, struct amdgpu_lite_get_info *info) {
    printf("[2] MMIO BAR mmap...\n");
    struct amdgpu_lite_map_bar mb;
    memset(&mb, 0, sizeof(mb));
    mb.bar_index = info->mmio_bar_index;
    if (ioctl(fd, AMDGPU_LITE_IOC_MAP_BAR, &mb) < 0) {
        perror("  MAP_BAR");
        return 1;
    }
    uint64_t mmio_size = info->bars[info->mmio_bar_index].size;
    void *mmio = mmap(NULL, mmio_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, mb.mmap_offset);
    if (mmio == MAP_FAILED) { perror("  mmap MMIO"); return 1; }
    volatile uint32_t *r = (volatile uint32_t *)mmio;
    printf("  REG[0x0000]=0x%08x REG[0x0004]=0x%08x\n", r[0], r[1]);
    munmap(mmio, mmio_size);
    printf("  PASS\n\n");
    return 0;
}

static int test_gtt(int fd) {
    printf("[3] GTT alloc + mmap + read/write...\n");
    struct amdgpu_lite_alloc_gtt gtt;
    memset(&gtt, 0, sizeof(gtt));
    gtt.size = 4096;
    if (ioctl(fd, AMDGPU_LITE_IOC_ALLOC_GTT, &gtt) < 0) {
        perror("  ALLOC_GTT");
        return 1;
    }
    printf("  handle=%llu bus=0x%llx\n",
        (unsigned long long)gtt.handle, (unsigned long long)gtt.bus_addr);
    void *buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, gtt.mmap_offset);
    if (buf == MAP_FAILED) { perror("  mmap GTT"); return 1; }
    uint32_t *p = (uint32_t *)buf;
    p[0] = 0xDEADBEEF; p[1] = 0xCAFEBABE;
    if (p[0] != 0xDEADBEEF || p[1] != 0xCAFEBABE) {
        printf("  FAIL: readback mismatch\n");
        munmap(buf, 4096);
        return 1;
    }
    printf("  Write/read 0xDEADBEEF/0xCAFEBABE OK\n");
    munmap(buf, 4096);

    struct amdgpu_lite_free_gtt fg;
    memset(&fg, 0, sizeof(fg));
    fg.handle = gtt.handle;
    ioctl(fd, AMDGPU_LITE_IOC_FREE_GTT, &fg);
    printf("  PASS\n\n");
    return 0;
}

static int test_vram(int fd) {
    printf("[4] VRAM alloc + mmap + read/write...\n");
    struct amdgpu_lite_alloc_vram vram;
    memset(&vram, 0, sizeof(vram));
    vram.size = 4096;
    if (ioctl(fd, AMDGPU_LITE_IOC_ALLOC_VRAM, &vram) < 0) {
        perror("  ALLOC_VRAM");
        return 1;
    }
    printf("  handle=%llu gpu_addr=0x%llx\n",
        (unsigned long long)vram.handle, (unsigned long long)vram.gpu_addr);
    void *buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, vram.mmap_offset);
    if (buf == MAP_FAILED) {
        perror("  mmap VRAM");
        return 1;
    }
    uint32_t *p = (uint32_t *)buf;
    p[0] = 0x12345678; p[1] = 0xABCD0000;
    uint32_t r0 = p[0], r1 = p[1];
    printf("  Write 0x12345678/0xABCD0000, Read 0x%08x/0x%08x\n", r0, r1);
    if (r0 != 0x12345678 || r1 != 0xABCD0000) {
        printf("  FAIL: VRAM readback mismatch\n");
        munmap(buf, 4096);
        return 1;
    }
    munmap(buf, 4096);

    struct amdgpu_lite_free_vram fv;
    memset(&fv, 0, sizeof(fv));
    fv.handle = vram.handle;
    ioctl(fd, AMDGPU_LITE_IOC_FREE_VRAM, &fv);
    printf("  PASS\n\n");
    return 0;
}

static int test_map_gpu(int fd) {
    printf("[5] MAP_GPU (GART PTE)...\n");
    /* Allocate GTT buffer */
    struct amdgpu_lite_alloc_gtt gtt;
    memset(&gtt, 0, sizeof(gtt));
    gtt.size = 8192;  /* 2 pages */
    if (ioctl(fd, AMDGPU_LITE_IOC_ALLOC_GTT, &gtt) < 0) {
        perror("  ALLOC_GTT");
        return 1;
    }
    printf("  GTT: handle=%llu bus=0x%llx\n",
        (unsigned long long)gtt.handle, (unsigned long long)gtt.bus_addr);

    /* Map to GPU VA */
    struct amdgpu_lite_map_gpu mg;
    memset(&mg, 0, sizeof(mg));
    mg.handle = gtt.handle;
    mg.gpu_va = 0;  /* auto-assign */
    mg.size = 8192;
    if (ioctl(fd, AMDGPU_LITE_IOC_MAP_GPU, &mg) < 0) {
        perror("  MAP_GPU");
        struct amdgpu_lite_free_gtt fg;
        memset(&fg, 0, sizeof(fg));
        fg.handle = gtt.handle;
        ioctl(fd, AMDGPU_LITE_IOC_FREE_GTT, &fg);
        return 1;
    }
    printf("  Mapped to GPU VA 0x%llx\n", (unsigned long long)mg.gpu_va);

    /* Unmap */
    struct amdgpu_lite_unmap_gpu ug;
    memset(&ug, 0, sizeof(ug));
    ug.gpu_va = mg.gpu_va;
    ug.size = 8192;
    ioctl(fd, AMDGPU_LITE_IOC_UNMAP_GPU, &ug);

    struct amdgpu_lite_free_gtt fg;
    memset(&fg, 0, sizeof(fg));
    fg.handle = gtt.handle;
    ioctl(fd, AMDGPU_LITE_IOC_FREE_GTT, &fg);
    printf("  PASS\n\n");
    return 0;
}

static int test_eventfd(int fd) {
    printf("[6] IRQ eventfd registration...\n");
    int efd = eventfd(0, EFD_NONBLOCK);
    if (efd < 0) { perror("  eventfd"); return 1; }

    struct amdgpu_lite_setup_irq si;
    memset(&si, 0, sizeof(si));
    si.eventfd = efd;
    si.irq_source = 0xB5;  /* CP_EOP source ID */
    if (ioctl(fd, AMDGPU_LITE_IOC_SETUP_IRQ, &si) < 0) {
        perror("  SETUP_IRQ");
        close(efd);
        return 1;
    }
    printf("  Registered eventfd for source 0xB5, reg_id=%u\n", si.out_registration_id);

    /* Teardown */
    struct amdgpu_lite_setup_irq td;
    memset(&td, 0, sizeof(td));
    td.registration_id = si.out_registration_id;
    if (ioctl(fd, AMDGPU_LITE_IOC_SETUP_IRQ, &td) < 0) {
        perror("  SETUP_IRQ teardown");
        close(efd);
        return 1;
    }
    printf("  Teardown OK\n");

    close(efd);
    printf("  PASS\n\n");
    return 0;
}

int main(void) {
    int fd = open("/dev/amdgpu_lite0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    printf("=== amdgpu_lite full test suite ===\n\n");

    struct amdgpu_lite_get_info info;
    int fails = 0;
    fails += test_get_info(fd, &info);
    fails += test_mmio_mmap(fd, &info);
    fails += test_gtt(fd);
    fails += test_vram(fd);
    fails += test_map_gpu(fd);
    fails += test_eventfd(fd);

    close(fd);
    printf("=== Results: %d failures ===\n", fails);
    return fails;
}
