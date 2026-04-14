/*
 * ROCmGPUShared.h - Shared definitions between DEXT and userspace.
 *
 * Defines the escape command protocol for communication between the
 * ROCmGPU DriverKit extension and the Python userspace driver via
 * IOConnectCallScalarMethod / IOConnectCallStructMethod.
 *
 * This file is included by both the C++ DEXT and referenced by the
 * Python ctypes bindings (iokit_client.py).
 */

#ifndef ROCMGPU_SHARED_H
#define ROCMGPU_SHARED_H

#include <stdint.h>

/* ====================================================================
 * External method selectors (IOUserClient dispatch table indices)
 *
 * Scalar methods use up to 16 uint64_t inputs / 16 uint64_t outputs.
 * Struct methods pass binary blobs for larger transfers.
 * ==================================================================== */

enum ROCmGPUSelector : uint32_t {
    /* Device info / lifecycle */
    kROCmGPU_GetInfo         = 0,   /* () -> DeviceInfo struct */
    kROCmGPU_Reset           = 1,   /* () -> () : Function-Level Reset */

    /* PCI configuration space */
    kROCmGPU_CfgRead         = 2,   /* (offset, width) -> (value) */
    kROCmGPU_CfgWrite        = 3,   /* (offset, width, value) -> () */

    /* MMIO register access (BAR-relative) */
    kROCmGPU_MMIORead32      = 4,   /* (barIndex, offset) -> (value) */
    kROCmGPU_MMIOWrite32     = 5,   /* (barIndex, offset, value) -> () */

    /* BAR mapping (maps PCI BAR into client process) */
    kROCmGPU_MapBAR          = 6,   /* (barIndex) -> (size) ; use IOConnectMapMemory64 */
    kROCmGPU_UnmapBAR        = 7,   /* (barIndex) -> () */

    /* DMA buffer management */
    kROCmGPU_AllocDMA        = 8,   /* (size, flags) -> (bufferID, physAddr) */
    kROCmGPU_FreeDMA         = 9,   /* (bufferID) -> () */
    kROCmGPU_MapDMA          = 10,  /* (bufferID) -> () ; then IOConnectMapMemory64 */

    /* Interrupt handling */
    kROCmGPU_EnableMSI       = 11,  /* (vectorIndex) -> () */
    kROCmGPU_WaitInterrupt   = 12,  /* (timeoutMS) -> (status) */

    kROCmGPU_SelectorCount   = 13
};

/* ====================================================================
 * Memory type constants for IOConnectMapMemory64()
 *
 * The 'type' parameter selects what memory region to map:
 *   0-5:     PCI BARs (BAR0 through BAR5)
 *   0x100+N: DMA buffer with ID=N
 * ==================================================================== */

enum ROCmGPUMemoryType : uint32_t {
    kROCmGPU_MemType_BAR0     = 0,
    kROCmGPU_MemType_BAR1     = 1,
    kROCmGPU_MemType_BAR2     = 2,
    kROCmGPU_MemType_BAR3     = 3,
    kROCmGPU_MemType_BAR4     = 4,
    kROCmGPU_MemType_BAR5     = 5,
    kROCmGPU_MemType_DMABase  = 0x100,  /* DMA buffers start at 0x100 */
};

/* ====================================================================
 * Device info structure (returned by kROCmGPU_GetInfo)
 * ==================================================================== */

struct ROCmGPUDeviceInfo {
    uint16_t vendorID;
    uint16_t deviceID;
    uint16_t subsystemVendorID;
    uint16_t subsystemDeviceID;
    uint8_t  revisionID;
    uint8_t  _pad[3];

    /* BAR information (up to 6 BARs) */
    struct {
        uint64_t size;          /* BAR size in bytes (0 = not present) */
        uint8_t  memoryIndex;   /* DriverKit memory index for this BAR */
        uint8_t  type;          /* 0=memory, 1=IO, 2=not present */
        uint8_t  is64bit;       /* 1 if this is a 64-bit BAR */
        uint8_t  prefetchable;  /* 1 if prefetchable */
        uint8_t  _pad2[4];
    } bars[6];

    uint64_t vramSize;          /* Total VRAM in bytes (from BAR sizing) */
};

/* ====================================================================
 * DMA allocation flags
 * ==================================================================== */

enum ROCmGPUDMAFlags : uint32_t {
    kROCmGPU_DMA_Contiguous   = (1 << 0),  /* Physically contiguous */
    kROCmGPU_DMA_Uncached     = (1 << 1),  /* Uncacheable mapping */
    kROCmGPU_DMA_ReadOnly     = (1 << 2),  /* Device reads only */
    kROCmGPU_DMA_WriteOnly    = (1 << 3),  /* Device writes only */
};

/* ====================================================================
 * DMA buffer info (returned by kROCmGPU_AllocDMA via struct method)
 * ==================================================================== */

struct ROCmGPUDMAInfo {
    uint64_t bufferID;          /* Opaque ID for subsequent calls */
    uint64_t size;              /* Actual allocation size */
    uint32_t segmentCount;      /* Number of physical segments */
    uint32_t _pad;
    struct {
        uint64_t address;       /* IOMMU-translated physical address */
        uint64_t length;        /* Segment length in bytes */
    } segments[64];             /* Scatter-gather list (max 64 segments) */
};

/* ====================================================================
 * Interrupt wait result
 * ==================================================================== */

enum ROCmGPUInterruptStatus : uint32_t {
    kROCmGPU_IntStatus_OK       = 0,  /* Interrupt received */
    kROCmGPU_IntStatus_Timeout  = 1,  /* Timed out waiting */
    kROCmGPU_IntStatus_Error    = 2,  /* Error occurred */
};

#endif /* ROCMGPU_SHARED_H */
