#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>

#include "amdgpu_lite.h"

int main(void) {
    int fd = open("/dev/amdgpu_lite0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/amdgpu_lite0");
        return 1;
    }
    printf("Opened /dev/amdgpu_lite0 (fd=%d)\n", fd);

    /* Test 1: GET_INFO */
    struct amdgpu_lite_get_info info;
    memset(&info, 0, sizeof(info));
    int ret = ioctl(fd, AMDGPU_LITE_IOC_GET_INFO, &info);
    if (ret < 0) {
        perror("GET_INFO ioctl");
        close(fd);
        return 1;
    }
    printf("\n[GET_INFO]\n");
    printf("  Vendor:    0x%04x\n", info.vendor_id);
    printf("  Device:    0x%04x\n", info.device_id);
    printf("  Revision:  0x%02x\n", info.revision_id);
    printf("  Num BARs:  %u\n", info.num_bars);
    printf("  VRAM:      %llu MB\n", (unsigned long long)info.vram_size / (1024*1024));
    printf("  MMIO BAR:  %u\n", info.mmio_bar_index);
    printf("  VRAM BAR:  %u\n", info.vram_bar_index);
    printf("  Door BAR:  %u\n", info.doorbell_bar_index);
    for (unsigned i = 0; i < info.num_bars && i < 6; i++) {
        if (info.bars[i].size > 0) {
            printf("  BAR[%u]: phys=0x%llx size=%llu MB\n", i,
                (unsigned long long)info.bars[i].phys_addr,
                (unsigned long long)info.bars[i].size / (1024*1024));
        }
    }

    /* Test 2: MAP_BAR + mmap MMIO */
    struct amdgpu_lite_map_bar map_bar;
    memset(&map_bar, 0, sizeof(map_bar));
    map_bar.bar_index = info.mmio_bar_index;
    ret = ioctl(fd, AMDGPU_LITE_IOC_MAP_BAR, &map_bar);
    if (ret < 0) {
        perror("MAP_BAR ioctl");
        close(fd);
        return 1;
    }
    /* Use the BAR size from info for mmap */
    __u64 mmio_size = info.bars[info.mmio_bar_index].size;
    printf("\n[MAP_BAR MMIO] offset=0x%llx size=%llu\n",
        (unsigned long long)map_bar.mmap_offset,
        (unsigned long long)mmio_size);

    void *mmio = mmap(NULL, mmio_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, map_bar.mmap_offset);
    if (mmio == MAP_FAILED) {
        perror("mmap MMIO");
        close(fd);
        return 1;
    }
    printf("  mmap'd MMIO at %p\n", mmio);

    /* Read a few registers */
    volatile uint32_t *regs = (volatile uint32_t *)mmio;
    printf("  REG[0x0000] = 0x%08x\n", regs[0]);
    printf("  REG[0x0004] = 0x%08x\n", regs[1]);
    printf("  REG[0x0008] = 0x%08x\n", regs[2]);
    printf("  REG[0x000C] = 0x%08x\n", regs[3]);

    munmap(mmio, mmio_size);

    /* Test 3: ALLOC_GTT */
    struct amdgpu_lite_alloc_gtt gtt;
    memset(&gtt, 0, sizeof(gtt));
    gtt.size = 4096;
    ret = ioctl(fd, AMDGPU_LITE_IOC_ALLOC_GTT, &gtt);
    if (ret < 0) {
        perror("ALLOC_GTT ioctl");
        close(fd);
        return 1;
    }
    printf("\n[ALLOC_GTT] handle=%llu bus_addr=0x%llx mmap_offset=0x%llx\n",
        (unsigned long long)gtt.handle,
        (unsigned long long)gtt.bus_addr,
        (unsigned long long)gtt.mmap_offset);

    /* mmap the GTT buffer and write/read */
    void *gtt_buf = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, gtt.mmap_offset);
    if (gtt_buf == MAP_FAILED) {
        perror("mmap GTT");
    } else {
        uint32_t *p = (uint32_t *)gtt_buf;
        p[0] = 0xDEADBEEF;
        p[1] = 0xCAFEBABE;
        printf("  GTT write: 0x%08x 0x%08x\n", p[0], p[1]);
        printf("  GTT read:  0x%08x 0x%08x\n", p[0], p[1]);
        if (p[0] == 0xDEADBEEF && p[1] == 0xCAFEBABE)
            printf("  GTT read-back OK!\n");
        munmap(gtt_buf, 4096);
    }

    /* Free GTT */
    struct amdgpu_lite_free_gtt free_gtt;
    memset(&free_gtt, 0, sizeof(free_gtt));
    free_gtt.handle = gtt.handle;
    ret = ioctl(fd, AMDGPU_LITE_IOC_FREE_GTT, &free_gtt);
    if (ret < 0)
        perror("FREE_GTT");
    else
        printf("  GTT freed OK\n");

    close(fd);
    printf("\nAll tests passed!\n");
    return 0;
}
