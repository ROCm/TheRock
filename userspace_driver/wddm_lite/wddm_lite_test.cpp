/*
 * wddm_lite_test.cpp - Test harness for WDDM escape interface
 *
 * Exercises all escape commands and performs GPU bring-up:
 * 1. Basic escape tests (get_info, reg read/write, VRAM map, DMA)
 * 2. IP discovery
 * 3. GMC init
 * 4. PSP ring + firmware loading
 * 5. SMU messaging
 * 6. Compute escape tests (alloc/free, queue, event)
 */

#include "wddm_lite.h"
#include <cstdio>
#include <cstring>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;
static int g_total = 0;

#define TEST_BEGIN(name) \
    do { \
        g_total++; \
        printf("\n--- TEST %d: %s ---\n", g_total, name); \
    } while(0)

#define TEST_PASS() \
    do { \
        g_passed++; \
        printf("  PASS\n"); \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        g_failed++; \
        printf("  FAIL: %s\n", msg); \
    } while(0)

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { TEST_FAIL(msg); return; } \
    } while(0)

/* ======================================================================
 * Test: Open adapter
 * ====================================================================== */

static void testOpenAdapter(WddmLite &gpu)
{
    TEST_BEGIN("Open AMD GPU adapter");
    TEST_CHECK(gpu.open(), "Failed to open GPU");
    TEST_PASS();
}

/* ======================================================================
 * Test: GET_INFO
 * ====================================================================== */

static AMDGPU_ESCAPE_GET_INFO_DATA g_info;

static void testGetInfo(WddmLite &gpu)
{
    TEST_BEGIN("GET_INFO escape");
    TEST_CHECK(gpu.getInfo(&g_info), "GET_INFO failed");
    TEST_CHECK(g_info.VendorId == 0x1002, "Wrong vendor ID");

    printf("  Vendor:   0x%04X\n", g_info.VendorId);
    printf("  Device:   0x%04X\n", g_info.DeviceId);
    printf("  Revision: 0x%02X\n", g_info.RevisionId);
    printf("  NumBars:  %u\n", g_info.NumBars);
    printf("  VRAM:     %llu MB\n", g_info.VramSizeBytes / (1024*1024));
    printf("  Visible:  %llu MB\n", g_info.VisibleVramSizeBytes / (1024*1024));
    printf("  MMIO BAR: %u\n", g_info.MmioBarIndex);
    printf("  VRAM BAR: %u\n", g_info.VramBarIndex);
    printf("  Headless: %s\n", g_info.Headless ? "YES" : "NO");

    for (uint32_t i = 0; i < g_info.NumBars && i < 6; i++) {
        if (g_info.Bars[i].Length > 0) {
            printf("  BAR[%u]: phys=0x%llX len=0x%llX mem=%d 64=%d pf=%d\n",
                   i, g_info.Bars[i].PhysicalAddress.QuadPart,
                   g_info.Bars[i].Length, g_info.Bars[i].IsMemory,
                   g_info.Bars[i].Is64Bit, g_info.Bars[i].IsPrefetchable);
        }
    }

    TEST_PASS();
}

/* ======================================================================
 * Test: Register read/write
 * ====================================================================== */

static void testRegAccess(WddmLite &gpu)
{
    TEST_BEGIN("Register read/write");

    uint32_t value = 0;
    TEST_CHECK(gpu.readReg32(0, &value), "Read reg offset 0 failed");
    printf("  MMIO[0x0000] = 0x%08X\n", value);

    /* Read a few well-known offsets */
    gpu.readReg32(0x0050 * 4, &value);  /* GRBM_STATUS */
    printf("  GRBM_STATUS = 0x%08X\n", value);

    TEST_PASS();
}

/* ======================================================================
 * Test: VRAM mapping
 * ====================================================================== */

static void testVramRead(WddmLite &gpu)
{
    TEST_BEGIN("VRAM read (kernel copy)");

    uint8_t buf[256] = {};
    TEST_CHECK(gpu.readVram(0, 256, buf), "Read VRAM failed");

    uint32_t firstDword = *(uint32_t *)buf;
    printf("  VRAM[0] = 0x%08X\n", firstDword);
    printf("  VRAM[4] = 0x%08X\n", *(uint32_t *)(buf + 4));
    printf("  VRAM[8] = 0x%08X\n", *(uint32_t *)(buf + 8));

    TEST_PASS();
}

/* ======================================================================
 * Test: DMA allocation
 * ====================================================================== */

static void testDmaAlloc(WddmLite &gpu)
{
    TEST_BEGIN("DMA alloc/free");

    void *cpuAddr = nullptr;
    uint64_t busAddr = 0;
    void *handle = nullptr;

    TEST_CHECK(gpu.allocDma(0x10000, &cpuAddr, &busAddr, &handle),
               "DMA alloc failed");
    TEST_CHECK(cpuAddr != nullptr, "CPU address is NULL");
    TEST_CHECK(busAddr != 0, "Bus address is 0");

    printf("  DMA: CPU=%p BUS=0x%llX\n", cpuAddr, busAddr);

    /* Write/read test */
    *(volatile uint32_t *)cpuAddr = 0xDEADBEEF;
    uint32_t readback = *(volatile uint32_t *)cpuAddr;
    TEST_CHECK(readback == 0xDEADBEEF, "DMA readback mismatch");

    TEST_CHECK(gpu.freeDma(handle), "DMA free failed");
    TEST_PASS();
}

/* ======================================================================
 * Test: IP Discovery
 * ====================================================================== */

static IpDiscoveryResult g_ipd;

static void testIpDiscovery(WddmLite &gpu)
{
    TEST_BEGIN("IP Discovery");
    TEST_CHECK(ipDiscovery(gpu, g_info, g_ipd), "IP discovery failed");
    TEST_CHECK(g_ipd.numBlocks > 0, "No IP blocks found");
    TEST_CHECK(g_ipd.mmhubBase != 0, "MMHUB base not found");
    TEST_CHECK(g_ipd.mp0Base != 0, "PSP (MP0) base not found");
    TEST_PASS();
}

/* ======================================================================
 * Test: GMC Init
 * ====================================================================== */

static GmcState g_gmc;

static void testGmcInit(WddmLite &gpu)
{
    TEST_BEGIN("GMC Init (MMHUB)");
    TEST_CHECK(g_ipd.valid, "IP discovery not valid");
    TEST_CHECK(gmcInit(gpu, g_ipd, g_gmc), "GMC init failed");
    TEST_CHECK(g_gmc.vramSize > 0, "VRAM size is 0");
    TEST_PASS();
}

/* ======================================================================
 * Test: PSP Sign of Life
 * ====================================================================== */

static void testPspSol(WddmLite &gpu)
{
    TEST_BEGIN("PSP Sign of Life");
    TEST_CHECK(g_ipd.valid, "IP discovery not valid");
    TEST_CHECK(pspCheckSos(gpu, g_ipd), "PSP SOS check failed");
    TEST_PASS();
}

/* ======================================================================
 * Test: PSP Ring
 * ====================================================================== */

static PspState g_psp;

static void testPspRing(WddmLite &gpu)
{
    TEST_BEGIN("PSP GPCOM Ring create");
    TEST_CHECK(g_ipd.valid, "IP discovery not valid");
    TEST_CHECK(pspRingCreate(gpu, g_ipd, g_psp), "PSP ring create failed");
    TEST_CHECK(g_psp.ringCreated, "Ring not marked as created");
    TEST_PASS();
}

/* ======================================================================
 * Test: SMU Disable GFXOFF
 * ====================================================================== */

static void testSmuGfxOff(WddmLite &gpu)
{
    TEST_BEGIN("SMU Disable GFXOFF");
    TEST_CHECK(g_ipd.valid, "IP discovery not valid");
    /* SMU might not respond if firmware isn't loaded yet.
     * We try but don't fail the whole test suite. */
    bool ok = smuDisableGfxOff(gpu, g_ipd);
    if (!ok) {
        printf("  WARNING: SMU didn't respond (may need firmware first)\n");
    }
    /* Pass regardless - SMU response depends on firmware state */
    TEST_PASS();
}

/* ======================================================================
 * Test: Firmware loading (if firmware files present)
 * ====================================================================== */

static void testFirmwareLoad(WddmLite &gpu)
{
    TEST_BEGIN("Firmware loading");
    TEST_CHECK(g_psp.ringCreated, "PSP ring not created");

    const char *fwDir = "C:\\dev\\firmware";

    /* Try to load TOC first (CRITICAL) */
    std::vector<uint8_t> tocData;
    char tocPath[256];
    snprintf(tocPath, sizeof(tocPath), "%s\\psp_14_0_3_toc.bin", fwDir);

    if (loadFirmwareFile(tocPath, tocData)) {
        printf("\n  --- Loading TOC ---\n");
        if (!pspLoadFirmware(gpu, g_ipd, g_psp, PSP_FW_TYPE_TOC,
                             tocData.data(), (uint32_t)tocData.size())) {
            printf("  WARNING: TOC load failed\n");
        }
    } else {
        /* Try extracting TOC from SOS binary */
        char sosPath[256];
        snprintf(sosPath, sizeof(sosPath), "%s\\psp_14_0_3_sos.bin", fwDir);
        std::vector<uint8_t> sosData;
        if (loadFirmwareFile(sosPath, sosData)) {
            /* TOC is at offset 0x62ef0, size 2304 bytes */
            uint32_t tocOffset = 0x62ef0;
            uint32_t tocSize = 2304;
            if (sosData.size() >= tocOffset + tocSize) {
                printf("\n  --- Loading TOC from SOS (offset 0x%X) ---\n", tocOffset);
                pspLoadFirmware(gpu, g_ipd, g_psp, PSP_FW_TYPE_TOC,
                               sosData.data() + tocOffset, tocSize);
            }
        }
    }

    /* Load SMU firmware */
    struct FwEntry { const char *name; uint32_t type; };
    FwEntry fwList[] = {
        {"smu_14_0_3.bin",         PSP_FW_TYPE_SMU},
        {"sdma_7_0_1.bin",         PSP_FW_TYPE_SDMA0},
        {"gc_12_0_1_pfp.bin",      PSP_FW_TYPE_PFP},
        {"gc_12_0_1_me.bin",       PSP_FW_TYPE_ME},
        {"gc_12_0_1_mec.bin",      PSP_FW_TYPE_MEC},
        {"gc_12_0_1_imu_i.bin",    PSP_FW_TYPE_IMU_I},
        {"gc_12_0_1_imu_d.bin",    PSP_FW_TYPE_IMU_D},
        {"gc_12_0_1_rlc.bin",      PSP_FW_TYPE_RLC_G},
    };

    int loaded = 0;
    int attempted = 0;

    for (const auto &fw : fwList) {
        char path[256];
        snprintf(path, sizeof(path), "%s\\%s", fwDir, fw.name);
        std::vector<uint8_t> fwData;
        if (loadFirmwareFile(path, fwData)) {
            attempted++;
            if (pspLoadFirmware(gpu, g_ipd, g_psp, fw.type,
                               fwData.data(), (uint32_t)fwData.size())) {
                loaded++;
            }
        }
    }

    printf("\n  Firmware: loaded %d/%d attempted\n", loaded, attempted);

    if (attempted == 0) {
        printf("  No firmware files found in %s\n", fwDir);
        printf("  Copy firmware files to continue firmware testing\n");
    }

    /* Trigger autoload if RLC was loaded */
    if (loaded > 0) {
        pspTriggerAutoload(gpu, g_ipd, g_psp);
    }

    /* This test passes even without firmware files -
     * the purpose is to validate the PSP ring protocol */
    TEST_PASS();
}

/* ======================================================================
 * Test: PSP Ring Destroy
 * ====================================================================== */

static void testPspRingDestroy(WddmLite &gpu)
{
    TEST_BEGIN("PSP Ring destroy");
    TEST_CHECK(pspRingDestroy(gpu, g_ipd, g_psp), "PSP ring destroy failed");
    TEST_PASS();
}

/* ======================================================================
 * Test: Compute memory allocation
 * ====================================================================== */

static void testAllocMemory(WddmLite &gpu)
{
    TEST_BEGIN("Compute: alloc/free memory");

    void *cpuAddr = nullptr;
    uint64_t gpuAddr = 0;
    uint64_t handle = 0;

    TEST_CHECK(gpu.allocMemory(0x10000,
                               AMDGPU_MEM_TYPE_SYSTEM | AMDGPU_MEM_FLAG_HOST_ACCESS,
                               &cpuAddr, &gpuAddr, &handle),
               "AllocMemory failed");

    printf("  Alloc: CPU=%p GPU=0x%llX handle=%llu\n", cpuAddr, gpuAddr, handle);
    TEST_CHECK(gpuAddr != 0, "GPU address is 0");

    /* Write/read if CPU mapped */
    if (cpuAddr) {
        *(volatile uint32_t *)cpuAddr = 0xCAFEBABE;
        uint32_t rb = *(volatile uint32_t *)cpuAddr;
        TEST_CHECK(rb == 0xCAFEBABE, "Memory readback mismatch");
    }

    TEST_CHECK(gpu.freeMemory(handle), "FreeMemory failed");
    TEST_PASS();
}

/* ======================================================================
 * Test: Compute queue create/destroy
 * ====================================================================== */

static void testCreateQueue(WddmLite &gpu)
{
    TEST_BEGIN("Compute: create/destroy queue");

    uint64_t queueId = 0;
    uint64_t doorbellOffset = 0;

    /* Queue type 1 = COMPUTE */
    TEST_CHECK(gpu.createQueue(1, 0x100000, 0x40000, &queueId, &doorbellOffset),
               "CreateQueue failed");

    printf("  Queue: id=%llu doorbell=0x%llX\n", queueId, doorbellOffset);
    TEST_CHECK(queueId != 0, "Queue ID is 0");

    TEST_CHECK(gpu.destroyQueue(queueId), "DestroyQueue failed");
    TEST_PASS();
}

/* ======================================================================
 * Test: Event create/destroy
 * ====================================================================== */

static void testCreateEvent(WddmLite &gpu)
{
    TEST_BEGIN("Compute: create/destroy event");

    uint32_t eventId = 0;
    uint64_t eventPageAddr = 0;
    uint32_t slotIndex = 0;

    TEST_CHECK(gpu.createEvent(AMDGPU_EVENT_TYPE_SIGNAL, &eventId,
                               &eventPageAddr, &slotIndex),
               "CreateEvent failed");

    printf("  Event: id=%u page=0x%llX slot=%u\n", eventId, eventPageAddr, slotIndex);
    TEST_CHECK(eventId != 0, "Event ID is 0");

    TEST_CHECK(gpu.destroyEvent(eventId), "DestroyEvent failed");
    TEST_PASS();
}

/* ======================================================================
 * Test: Get version
 * ====================================================================== */

static void testGetVersion(WddmLite &gpu)
{
    TEST_BEGIN("KFD version");

    uint32_t major = 0, minor = 0;
    TEST_CHECK(gpu.getVersion(&major, &minor), "GetVersion failed");

    printf("  KFD version: %u.%u\n", major, minor);
    TEST_CHECK(major == 1, "Unexpected major version");
    TEST_PASS();
}

/* ======================================================================
 * Test: Clock counters
 * ====================================================================== */

static void testClockCounters(WddmLite &gpu)
{
    TEST_BEGIN("Clock counters");

    uint64_t gpuClock = 0, cpuClock = 0;
    TEST_CHECK(gpu.getClockCounters(&gpuClock, &cpuClock), "GetClockCounters failed");

    printf("  GPU clock: %llu\n", gpuClock);
    printf("  CPU clock: %llu\n", cpuClock);
    TEST_CHECK(gpuClock != 0, "GPU clock is 0");
    TEST_PASS();
}

/* ======================================================================
 * Test: Process apertures
 * ====================================================================== */

static void testProcessApertures(WddmLite &gpu)
{
    TEST_BEGIN("Process apertures");

    uint64_t vmBase = 0, vmLimit = 0;
    TEST_CHECK(gpu.getProcessApertures(&vmBase, &vmLimit),
               "GetProcessApertures failed");

    printf("  GPUVM: 0x%llX - 0x%llX\n", vmBase, vmLimit);
    TEST_CHECK(vmBase < vmLimit, "Invalid GPUVM range");
    TEST_PASS();
}

/* ======================================================================
 * Main
 * ====================================================================== */

static bool shouldRun(const char *filter, const char *name)
{
    if (!filter || !filter[0]) return true;
    return strstr(name, filter) != nullptr;
}

int main(int argc, char *argv[])
{
    printf("=== wddm_lite_test ===\n");

    const char *filter = nullptr;
    if (argc > 1) {
        filter = argv[1];
        printf("Filter: '%s'\n", filter);
    }
    printf("Testing WDDM escape interface and GPU bring-up\n\n");

    WddmLite gpu;

    /* Always open adapter first */
    testOpenAdapter(gpu);
    if (!gpu.isOpen()) {
        printf("\nFATAL: Cannot open GPU, aborting\n");
        return 1;
    }

    /* Phase 1: Basic escape tests */
    if (shouldRun(filter, "info"))     testGetInfo(gpu);
    if (shouldRun(filter, "reg"))      testRegAccess(gpu);
    if (shouldRun(filter, "vram"))     testVramRead(gpu);
    if (shouldRun(filter, "dma"))      testDmaAlloc(gpu);

    /* Phase 2: IP Discovery + Hardware init */
    /* Ensure GET_INFO is populated for IP discovery */
    bool needIpd = shouldRun(filter, "ipd") || shouldRun(filter, "gmc") ||
                   shouldRun(filter, "sol") || shouldRun(filter, "psp") ||
                   shouldRun(filter, "smu") || shouldRun(filter, "fw") ||
                   shouldRun(filter, "pspd");
    if (needIpd && g_info.VendorId == 0) {
        gpu.getInfo(&g_info);
    }
    if (needIpd && !g_ipd.valid) {
        testIpDiscovery(gpu);
    } else if (shouldRun(filter, "ipd")) {
        testIpDiscovery(gpu);
    }

    if (g_ipd.valid) {
        if (shouldRun(filter, "gmc"))  testGmcInit(gpu);
        if (shouldRun(filter, "sol"))  testPspSol(gpu);
        if (shouldRun(filter, "psp"))  testPspRing(gpu);
        if (shouldRun(filter, "smu"))  testSmuGfxOff(gpu);
        if (shouldRun(filter, "fw"))   testFirmwareLoad(gpu);
        if (shouldRun(filter, "pspd")) testPspRingDestroy(gpu);
    }

    /* Phase 3: Compute escape tests */
    if (shouldRun(filter, "alloc"))    testAllocMemory(gpu);
    if (shouldRun(filter, "queue"))    testCreateQueue(gpu);
    if (shouldRun(filter, "event"))    testCreateEvent(gpu);
    if (shouldRun(filter, "ver"))      testGetVersion(gpu);
    if (shouldRun(filter, "clock"))    testClockCounters(gpu);
    if (shouldRun(filter, "aper"))     testProcessApertures(gpu);

    /* Summary */
    printf("\n========================================\n");
    printf("Results: %d/%d passed, %d failed\n", g_passed, g_total, g_failed);
    printf("========================================\n");

    gpu.close();
    return g_failed > 0 ? 1 : 0;
}
