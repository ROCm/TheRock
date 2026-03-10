/*
 * gpu_init.cpp - GPU initialization: IP discovery, GMC, PSP, SMU
 *
 * Performs hardware bring-up via the WDDM escape channel:
 * 1. IP discovery - Parse IP blocks from VRAM discovery table
 * 2. GMC init - Configure MMHUB/GFXHUB memory controllers
 * 3. PSP GPCOM ring - Create ring, load firmware
 * 4. SMU - Disable GFXOFF
 */

#include "wddm_lite.h"
#include <cstring>
#include <cstdlib>

/* ======================================================================
 * IP Discovery
 * ====================================================================== */

/*
 * IP discovery table lives at VRAM_SIZE - 64KB.
 * Binary format: signature + header + per-die tables with IP blocks.
 */

bool ipDiscovery(WddmLite &gpu, const AMDGPU_ESCAPE_GET_INFO_DATA &info,
                 IpDiscoveryResult &result)
{
    memset(&result, 0, sizeof(result));

    uint64_t vramSize = info.VramSizeBytes;
    if (vramSize == 0) {
        /* Use visible VRAM as fallback */
        vramSize = info.VisibleVramSizeBytes;
    }
    if (vramSize == 0) {
        /* Estimate from BAR size */
        for (uint32_t i = 0; i < info.NumBars; i++) {
            if (info.Bars[i].IsMemory && info.Bars[i].Length > vramSize)
                vramSize = info.Bars[i].Length;
        }
    }

    printf("VRAM size: 0x%llX (%llu MB)\n", vramSize, vramSize / (1024*1024));

    /* Discovery table is at VRAM_SIZE - 64KB.
     * Map the last 64KB + some extra of VRAM. */
    uint64_t discoveryOffset = vramSize - 0x10000;
    uint64_t mapLen = 0x10000;  /* 64KB */

    /* Read VRAM through driver escape (avoids user-mode mapping issues) */
    auto *vramBuf = new uint8_t[mapLen];
    if (!gpu.readVram(discoveryOffset, mapLen, vramBuf)) {
        printf("ERROR: Failed to read VRAM at offset 0x%llX\n", discoveryOffset);
        delete[] vramBuf;
        return false;
    }

    printf("Read VRAM at offset 0x%llX (%llu bytes)\n", discoveryOffset, mapLen);

    const uint8_t *data = vramBuf;

    /* Check for PSP header (256 bytes) */
    uint32_t tableOffset = 0;
    uint32_t sig = *(const uint32_t *)data;
    if (sig != IP_DISCOVERY_SIGNATURE) {
        /* Try after PSP header */
        sig = *(const uint32_t *)(data + 256);
        if (sig == IP_DISCOVERY_SIGNATURE) {
            tableOffset = 256;
        } else {
            printf("ERROR: IP discovery signature not found (got 0x%08X)\n", sig);
            delete[] vramBuf;
            return false;
        }
    }

    printf("IP discovery signature found at offset %u\n", tableOffset);

    /* Binary header: signature + version + checksum + size + table_list[6] */
    const uint8_t *hdr = data + tableOffset;
    uint16_t verMajor = *(const uint16_t *)(hdr + 4);
    uint16_t verMinor = *(const uint16_t *)(hdr + 6);
    printf("IP discovery version: %u.%u\n", verMajor, verMinor);

    /* Table list starts at offset 12, 6 tables x 8 bytes each.
     * Table 0 = IP discovery, offset is relative to binary start. */
    uint16_t ipTableOffset = *(const uint16_t *)(hdr + 12);

    if (ipTableOffset == 0 || ipTableOffset >= mapLen) {
        printf("ERROR: Invalid IP table offset: %u\n", ipTableOffset);
        delete[] vramBuf;
        return false;
    }

    const uint8_t *ipTable = hdr + ipTableOffset;

    /* IP discovery table header (ip_discovery_header):
     *   offset  0: signature  (uint32_t) - should be "IPDS" = 0x53445049
     *   offset  4: version    (uint16_t)
     *   offset  6: size       (uint16_t) - total table size
     *   offset  8: id         (uint32_t)
     *   offset 12: num_dies   (uint16_t)
     *   offset 14: die entries start
     */
    uint32_t ipSig = *(const uint32_t *)(ipTable + 0);
    uint16_t ipVer = *(const uint16_t *)(ipTable + 4);
    uint16_t ipSize = *(const uint16_t *)(ipTable + 6);
    uint32_t ipId = *(const uint32_t *)(ipTable + 8);
    uint16_t numDies = *(const uint16_t *)(ipTable + 12);
    printf("IP table: sig=0x%08X ver=%u size=%u id=0x%X numDies=%u\n",
           ipSig, ipVer, ipSize, ipId, numDies);

    if (numDies == 0 || numDies > 16) {
        printf("ERROR: Invalid numDies: %u\n", numDies);
        delete[] vramBuf;
        return false;
    }

    /* After ip_discovery_header (14 bytes), there's a die_info[] array.
     * Each die_info is 4 bytes: die_id(uint16) + die_offset(uint16).
     * die_offset is relative to the binary header start (hdr). */
    for (uint16_t dieIdx = 0; dieIdx < numDies && dieIdx < 16; dieIdx++) {
        const uint8_t *dieInfo = ipTable + 14 + dieIdx * 4;
        uint16_t dieId = *(const uint16_t *)(dieInfo + 0);
        uint16_t dieOffset = *(const uint16_t *)(dieInfo + 2);

        printf("Die info[%u]: die_id=%u die_offset=0x%04X\n",
               dieIdx, dieId, dieOffset);

        /* Jump to actual die data (offset relative to binary header) */
        const uint8_t *dieData = hdr + dieOffset;
        uint16_t dDieId = *(const uint16_t *)(dieData + 0);
        uint16_t numIps = *(const uint16_t *)(dieData + 2);
        printf("Die %u: %u IP blocks\n", dDieId, numIps);

        if (numIps > 256) {
            printf("WARNING: Suspiciously large numIps: %u, capping at 128\n", numIps);
            numIps = 128;
        }

        /* Parse IP entries starting at dieData + 4 */
        const uint8_t *ipData = dieData + 4;

        for (uint16_t i = 0; i < numIps && result.numBlocks < 64; i++) {
            /* Bounds check */
            if (ipData < data || ipData + 8 > data + mapLen) {
                printf("WARNING: IP data out of bounds at entry %u\n", i);
                break;
            }

            IpBlock &blk = result.blocks[result.numBlocks];
            blk.hwId = *(const uint16_t *)(ipData + 0);
            blk.instance = ipData[2];
            blk.numBaseAddrs = ipData[3];
            blk.majorVer = ipData[4];
            blk.minorVer = ipData[5];
            blk.revision = ipData[6];

            /* Base addresses start at offset 8 */
            uint8_t nBases = blk.numBaseAddrs;
            if (nBases > 8) nBases = 8;
            for (uint8_t j = 0; j < nBases; j++) {
                blk.baseAddrs[j] = *(const uint32_t *)(ipData + 8 + j * 4);
            }

            /* Advance past this IP entry: 8 + numBaseAddrs * 4 bytes */
            ipData += 8 + blk.numBaseAddrs * 4;
            result.numBlocks++;
        }
    }

    printf("\nDiscovered %u IP blocks:\n", result.numBlocks);
    for (uint32_t i = 0; i < result.numBlocks; i++) {
        const IpBlock &b = result.blocks[i];
        printf("  [%2u] HW_ID=%3u inst=%u v%u.%u.%u bases=%u",
               i, b.hwId, b.instance, b.majorVer, b.minorVer, b.revision,
               b.numBaseAddrs);
        if (b.numBaseAddrs > 0)
            printf(" [0x%04X", b.baseAddrs[0]);
        for (uint8_t j = 1; j < b.numBaseAddrs && j < 4; j++)
            printf(", 0x%04X", b.baseAddrs[j]);
        if (b.numBaseAddrs > 0) printf("]");
        printf("\n");
    }

    /* Resolve well-known IP bases */
    for (uint32_t i = 0; i < result.numBlocks; i++) {
        const IpBlock &b = result.blocks[i];
        switch (b.hwId) {
        case HWID_MMHUB:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.mmhubBase = b.baseAddrs[0];
            break;
        case HWID_GC:
            if (b.instance == 0) {
                if (b.numBaseAddrs > 0) result.gcBase = b.baseAddrs[0];
                if (b.numBaseAddrs > 1) result.gcBase1 = b.baseAddrs[1];
            }
            break;
        case HWID_MP0:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.mp0Base = b.baseAddrs[0];
            break;
        case HWID_MP1:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.mp1Base = b.baseAddrs[0];
            break;
        case HWID_SDMA0:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.sdma0Base = b.baseAddrs[0];
            break;
        case HWID_OSSSYS:
            if (b.instance == 0 && b.numBaseAddrs > 0)
                result.ihBase = b.baseAddrs[0];
            break;
        }
    }

    printf("\nResolved bases:\n");
    printf("  MMHUB  = 0x%04X\n", result.mmhubBase);
    printf("  GC[0]  = 0x%04X\n", result.gcBase);
    printf("  GC[1]  = 0x%04X\n", result.gcBase1);
    printf("  MP0    = 0x%04X (PSP)\n", result.mp0Base);
    printf("  MP1    = 0x%04X (SMU)\n", result.mp1Base);
    printf("  SDMA0  = 0x%04X\n", result.sdma0Base);
    printf("  IH     = 0x%04X\n", result.ihBase);

    delete[] vramBuf;

    result.valid = (result.mmhubBase != 0 && result.mp0Base != 0);
    return result.valid;
}

/* ======================================================================
 * GMC Init (MMHUB)
 * ====================================================================== */

/* MMHUB v4.1.0 register offsets (DWORD offsets, multiply by 4 for byte offset) */
#define regMMMC_VM_FB_LOCATION_BASE             0x0554
#define regMMMC_VM_FB_LOCATION_TOP              0x0555
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR     0x0559
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR    0x055A
#define regMMMC_VM_MX_L1_TLB_CNTL              0x055B
#define regMMVM_CONTEXT0_CNTL                   0x0564
#define regMMVM_L2_CNTL                         0x04E4

/* Helper: read MMHUB register (base + dword_offset * 4) */
static bool mmhubRead(WddmLite &gpu, const IpDiscoveryResult &ipd,
                      uint32_t dwOffset, uint32_t *value)
{
    uint32_t byteOffset = (ipd.mmhubBase + dwOffset) * 4;
    return gpu.readReg32(byteOffset, value);
}

static bool mmhubWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                       uint32_t dwOffset, uint32_t value)
{
    uint32_t byteOffset = (ipd.mmhubBase + dwOffset) * 4;
    return gpu.writeReg32(byteOffset, value);
}

bool gmcInit(WddmLite &gpu, const IpDiscoveryResult &ipd, GmcState &gmc)
{
    memset(&gmc, 0, sizeof(gmc));

    printf("\n=== GMC Init (MMHUB) ===\n");

    /* Read FB location to determine VRAM range.
     * Register defines are DWORD offsets relative to MMHUB base (BASE_IDX=0).
     * mmhubRead adds mmhubBase automatically. */
    uint32_t fbBase, fbTop;
    if (!mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_BASE, &fbBase) ||
        !mmhubRead(gpu, ipd, regMMMC_VM_FB_LOCATION_TOP, &fbTop)) {
        printf("ERROR: Cannot read FB_LOCATION registers\n");
        return false;
    }

    gmc.vramStart = (uint64_t)fbBase << 24;
    gmc.vramEnd = ((uint64_t)fbTop << 24) | 0xFFFFFF;
    gmc.vramSize = gmc.vramEnd - gmc.vramStart + 1;

    printf("FB_LOCATION_BASE = 0x%08X -> VRAM start 0x%llX\n", fbBase, gmc.vramStart);
    printf("FB_LOCATION_TOP  = 0x%08X -> VRAM end   0x%llX\n", fbTop, gmc.vramEnd);
    printf("VRAM size: %llu MB\n", gmc.vramSize / (1024*1024));

    /* GART sits at the end of VRAM range */
    gmc.gartSize = 512ULL * 1024 * 1024;  /* 512MB default */
    gmc.gartEnd = gmc.vramEnd;
    gmc.gartStart = gmc.gartEnd - gmc.gartSize + 1;

    printf("GART range: 0x%llX - 0x%llX (%llu MB)\n",
           gmc.gartStart, gmc.gartEnd, gmc.gartSize / (1024*1024));

    /* Read current MMHUB state */
    uint32_t ctx0Cntl, l1Cntl, l2Cntl;
    mmhubRead(gpu, ipd, regMMVM_CONTEXT0_CNTL, &ctx0Cntl);
    mmhubRead(gpu, ipd, regMMMC_VM_MX_L1_TLB_CNTL, &l1Cntl);
    mmhubRead(gpu, ipd, regMMVM_L2_CNTL, &l2Cntl);

    printf("Current MMHUB state:\n");
    printf("  CONTEXT0_CNTL = 0x%08X\n", ctx0Cntl);
    printf("  L1_TLB_CNTL   = 0x%08X\n", l1Cntl);
    printf("  L2_CNTL       = 0x%08X\n", l2Cntl);

    gmc.mmhubConfigured = (ctx0Cntl != 0);

    printf("MMHUB appears %s\n",
           gmc.mmhubConfigured ? "configured by VBIOS" : "unconfigured");

    return true;
}

/* ======================================================================
 * PSP - Sign of Life check
 * ====================================================================== */

/* PSP v14.0 register offsets (DWORD offsets from MP0 base) */
#define PSP_C2PMSG_64   0x0080  /* Ring command/status */
#define PSP_C2PMSG_67   0x0083  /* Ring write pointer */
#define PSP_C2PMSG_69   0x0085  /* Ring address low */
#define PSP_C2PMSG_70   0x0086  /* Ring address high */
#define PSP_C2PMSG_71   0x0087  /* Ring size */
#define PSP_C2PMSG_81   0x0091  /* SOS sign of life */
#define PSP_C2PMSG_35   0x0063  /* Bootloader mailbox */

static bool pspRead(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    uint32_t dwOffset, uint32_t *value)
{
    uint32_t byteOffset = (ipd.mp0Base + dwOffset) * 4;
    return gpu.readReg32(byteOffset, value);
}

static bool pspWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                     uint32_t dwOffset, uint32_t value)
{
    uint32_t byteOffset = (ipd.mp0Base + dwOffset) * 4;
    return gpu.writeReg32(byteOffset, value);
}

bool pspCheckSos(WddmLite &gpu, const IpDiscoveryResult &ipd)
{
    printf("\n=== PSP Sign of Life ===\n");
    printf("MP0 base = 0x%04X\n", ipd.mp0Base);

    /* Read bootloader status */
    uint32_t bootStatus = 0;
    pspRead(gpu, ipd, PSP_C2PMSG_35, &bootStatus);
    printf("C2PMSG_35 (bootloader) = 0x%08X%s\n", bootStatus,
           (bootStatus & 0x80000000) ? " [DONE]" : "");

    /* Read SOS sign of life */
    uint32_t sol = 0;
    pspRead(gpu, ipd, PSP_C2PMSG_81, &sol);
    printf("C2PMSG_81 (SOS sign of life) = 0x%08X\n", sol);

    /* Check ring status */
    uint32_t ringStatus = 0;
    pspRead(gpu, ipd, PSP_C2PMSG_64, &ringStatus);
    printf("C2PMSG_64 (ring status) = 0x%08X\n", ringStatus);

    bool tosReady = (ringStatus & 0x80000000) != 0;
    printf("TOS ready: %s\n", tosReady ? "YES" : "NO");

    if (sol == 0 && ringStatus == 0) {
        printf("WARNING: PSP GPCOM registers read 0\n");
        printf("  Bootloader done but ring not ready.\n");
        printf("  PSP GPCOM may require SMN indirect access on this GPU.\n");
        return false;
    }

    return true;
}

/* ======================================================================
 * PSP GPCOM Ring
 * ====================================================================== */

#define PSP_RING_SIZE   0x10000     /* 64KB */
#define PSP_FW_BUF_SIZE 0x100000   /* 1MB */
#define PSP_FENCE_SIZE  0x1000     /* 4KB */

bool pspRingCreate(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp)
{
    printf("\n=== PSP Ring Create ===\n");
    memset(&psp, 0, sizeof(psp));

    /* Allocate ring buffer via DMA */
    if (!gpu.allocDma(PSP_RING_SIZE, &psp.ringCpuAddr, &psp.ringBusAddr, &psp.ringDmaHandle)) {
        printf("ERROR: Failed to allocate PSP ring buffer\n");
        return false;
    }
    memset(psp.ringCpuAddr, 0, PSP_RING_SIZE);
    printf("Ring buffer: CPU=%p BUS=0x%llX\n", psp.ringCpuAddr, psp.ringBusAddr);

    /* Allocate firmware buffer */
    if (!gpu.allocDma(PSP_FW_BUF_SIZE, &psp.fwBufCpuAddr, &psp.fwBufBusAddr, &psp.fwBufDmaHandle)) {
        printf("ERROR: Failed to allocate firmware buffer\n");
        return false;
    }
    printf("FW buffer:   CPU=%p BUS=0x%llX\n", psp.fwBufCpuAddr, psp.fwBufBusAddr);

    /* Allocate fence buffer */
    void *fenceBuf = nullptr;
    if (!gpu.allocDma(PSP_FENCE_SIZE, &fenceBuf, &psp.fenceBusAddr, &psp.fenceDmaHandle)) {
        printf("ERROR: Failed to allocate fence buffer\n");
        return false;
    }
    psp.fenceCpuAddr = (volatile uint32_t *)fenceBuf;
    *psp.fenceCpuAddr = 0;
    psp.fenceValue = 0;
    printf("Fence:       CPU=%p BUS=0x%llX\n", fenceBuf, psp.fenceBusAddr);

    /* Wait for TOS ready */
    uint32_t status = 0;
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }
    if (!(status & 0x80000000)) {
        printf("ERROR: TOS not ready (C2PMSG_64 = 0x%08X)\n", status);
        return false;
    }
    printf("TOS ready (status = 0x%08X)\n", status);

    /* Destroy any existing ring first (cmd = 3 << 16 = 0x30000) */
    printf("Destroying existing ring...\n");
    pspWrite(gpu, ipd, PSP_C2PMSG_64, 0x30000);
    Sleep(50);

    /* Wait for response */
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }
    printf("After destroy: C2PMSG_64 = 0x%08X\n", status);

    /* Create new ring (cmd = PSP_RING_TYPE_KM << 16 = 0x20000) */
    uint32_t addrLo = (uint32_t)(psp.ringBusAddr & 0xFFFFFFFF);
    uint32_t addrHi = (uint32_t)(psp.ringBusAddr >> 32);

    pspWrite(gpu, ipd, PSP_C2PMSG_69, addrLo);
    pspWrite(gpu, ipd, PSP_C2PMSG_70, addrHi);
    pspWrite(gpu, ipd, PSP_C2PMSG_71, PSP_RING_SIZE);
    pspWrite(gpu, ipd, PSP_C2PMSG_64, PSP_RING_TYPE_KM << 16);

    Sleep(50);

    /* Wait for response */
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }

    uint16_t ringStatus = status & 0xFFFF;
    printf("Ring create response: 0x%08X (status=%u)\n", status, ringStatus);

    if (!(status & 0x80000000) || ringStatus != 0) {
        printf("ERROR: Ring create failed\n");
        return false;
    }

    psp.ringWptr = 0;
    psp.ringCreated = true;
    printf("PSP GPCOM ring created successfully\n");
    return true;
}

bool pspRingDestroy(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp)
{
    if (!psp.ringCreated) return true;

    printf("\n=== PSP Ring Destroy ===\n");
    pspWrite(gpu, ipd, PSP_C2PMSG_64, 0x30000);
    Sleep(50);

    uint32_t status = 0;
    for (int retry = 0; retry < 100; retry++) {
        pspRead(gpu, ipd, PSP_C2PMSG_64, &status);
        if (status & 0x80000000) break;
        Sleep(10);
    }
    printf("Destroy response: 0x%08X\n", status);

    /* Free DMA buffers */
    if (psp.ringDmaHandle) gpu.freeDma(psp.ringDmaHandle);
    if (psp.fwBufDmaHandle) gpu.freeDma(psp.fwBufDmaHandle);
    if (psp.fenceDmaHandle) gpu.freeDma(psp.fenceDmaHandle);

    memset(&psp, 0, sizeof(psp));
    return true;
}

/* ======================================================================
 * PSP Firmware Loading
 * ====================================================================== */

/*
 * PSP ring entry format (16 bytes = 4 DWORDs):
 *   DW[0] = fence_addr_lo
 *   DW[1] = fence_addr_hi
 *   DW[2] = fence_value
 *   DW[3] = cmd_id | (fw_type << 16)
 */

static bool pspRingSubmit(WddmLite &gpu, const IpDiscoveryResult &ipd,
                          PspState &psp, uint32_t cmdId, uint32_t fwType)
{
    uint32_t entryOffset = (psp.ringWptr * 16) % PSP_RING_SIZE;
    uint32_t *entry = (uint32_t *)((uint8_t *)psp.ringCpuAddr + entryOffset);

    psp.fenceValue++;

    entry[0] = (uint32_t)(psp.fenceBusAddr & 0xFFFFFFFF);
    entry[1] = (uint32_t)(psp.fenceBusAddr >> 32);
    entry[2] = psp.fenceValue;
    entry[3] = cmdId | (fwType << 16);

    /* Memory barrier */
    MemoryBarrier();

    psp.ringWptr++;
    pspWrite(gpu, ipd, PSP_C2PMSG_67, psp.ringWptr);

    /* Wait for fence */
    for (int retry = 0; retry < 500; retry++) {
        if (*psp.fenceCpuAddr >= psp.fenceValue)
            return true;
        Sleep(10);
    }

    printf("ERROR: PSP ring submission timed out (fence=%u, expected=%u)\n",
           *psp.fenceCpuAddr, psp.fenceValue);
    return false;
}

bool pspLoadFirmware(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp,
                     uint32_t fwType, const void *fwData, uint32_t fwSize)
{
    if (!psp.ringCreated) {
        printf("ERROR: PSP ring not created\n");
        return false;
    }

    if (fwSize > PSP_FW_BUF_SIZE) {
        printf("ERROR: Firmware too large (%u > %u)\n", fwSize, PSP_FW_BUF_SIZE);
        return false;
    }

    /* Copy firmware data to DMA buffer */
    memcpy(psp.fwBufCpuAddr, fwData, fwSize);
    MemoryBarrier();

    /* Write firmware buffer address to PSP bootloader mailbox
     * Address must be 1MB aligned, shifted >> 20 */
    uint32_t fwAddrShifted = (uint32_t)((psp.fwBufBusAddr >> 20) & 0xFFFFFFFF);
    pspWrite(gpu, ipd, PSP_C2PMSG_35 + 1, fwAddrShifted);

    /* Submit load command via ring */
    printf("  Loading FW type %u (%u bytes) at bus 0x%llX...",
           fwType, fwSize, psp.fwBufBusAddr);

    if (!pspRingSubmit(gpu, ipd, psp, GFX_CMD_ID_LOAD_IP_FW, fwType)) {
        printf(" FAILED\n");

        /* Check C2PMSG_64 for error details */
        uint32_t ringStatus = 0;
        pspRead(gpu, ipd, PSP_C2PMSG_64, &ringStatus);
        printf("  Ring status: 0x%08X\n", ringStatus);
        return false;
    }

    printf(" OK\n");
    return true;
}

bool pspTriggerAutoload(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp)
{
    printf("  Triggering RLC autoload...");
    if (!pspRingSubmit(gpu, ipd, psp, GFX_CMD_ID_AUTOLOAD_RLC, 0)) {
        printf(" FAILED\n");
        return false;
    }
    printf(" OK\n");
    return true;
}

/* ======================================================================
 * SMU Messaging
 * ====================================================================== */

/* SMU v11 mailbox offsets from MP1 base (DWORD offsets) */
#define SMU_MSG_REG     0x0282
#define SMU_PARAM_REG   0x0292
#define SMU_RESP_REG    0x029A

static bool smuRead(WddmLite &gpu, const IpDiscoveryResult &ipd,
                    uint32_t dwOffset, uint32_t *value)
{
    uint32_t byteOffset = (ipd.mp1Base + dwOffset) * 4;
    return gpu.readReg32(byteOffset, value);
}

static bool smuWrite(WddmLite &gpu, const IpDiscoveryResult &ipd,
                     uint32_t dwOffset, uint32_t value)
{
    uint32_t byteOffset = (ipd.mp1Base + dwOffset) * 4;
    return gpu.writeReg32(byteOffset, value);
}

bool smuDisableGfxOff(WddmLite &gpu, const IpDiscoveryResult &ipd)
{
    printf("\n=== SMU: Disable GFXOFF ===\n");

    /* Clear response register */
    smuWrite(gpu, ipd, SMU_RESP_REG, 0);

    /* Write parameter (0 = no param for DisallowGfxOff) */
    smuWrite(gpu, ipd, SMU_PARAM_REG, 0);

    /* Send message */
    smuWrite(gpu, ipd, SMU_MSG_REG, PPSMC_MSG_DisallowGfxOff);

    /* Wait for response */
    uint32_t resp = 0;
    for (int retry = 0; retry < 100; retry++) {
        smuRead(gpu, ipd, SMU_RESP_REG, &resp);
        if (resp != 0) break;
        Sleep(10);
    }

    printf("SMU response: 0x%08X (0 = pending, 1 = success)\n", resp);

    if (resp != 1) {
        printf("WARNING: SMU DisallowGfxOff may have failed\n");
        return false;
    }

    printf("GFXOFF disabled successfully\n");
    return true;
}

/* ======================================================================
 * Firmware file loading utility
 * ====================================================================== */

bool loadFirmwareFile(const char *path, std::vector<uint8_t> &data)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: Cannot open firmware file: %s\n", path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 16 * 1024 * 1024) {
        printf("ERROR: Invalid firmware size: %ld\n", size);
        fclose(f);
        return false;
    }

    data.resize(size);
    size_t read = fread(data.data(), 1, size, f);
    fclose(f);

    if ((long)read != size) {
        printf("ERROR: Short read: %zu/%ld\n", read, size);
        return false;
    }

    printf("Loaded %s (%ld bytes)\n", path, size);
    return true;
}
