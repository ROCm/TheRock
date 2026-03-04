/*
 * amdgpu_mcdm.h - Shared header for AMD GPU MCDM miniport driver
 *
 * Defines escape command codes and structures shared between the
 * kernel-mode driver and the Python userspace driver.
 */

#pragma once

/*
 * Include order matters: wdm.h must come first in kernel mode to
 * bootstrap the full type system (UINT, ULONG, etc.) before d3dkmddi.h
 * and dispmprt.h are processed. If ntdef.h is included first, its
 * #pragma once prevents wdm.h from re-including it, which breaks the
 * type chain for d3dukmdt.h.
 */
#ifdef _KERNEL_MODE
/* ntddk.h includes wdm.h plus BUS_DATA_TYPE etc. needed by video.h */
#include <ntddk.h>
/*
 * dispmprt.h must come before d3dkmddi.h: it includes <windef.h> which
 * defines UINT, needed by d3dukmdt.h (pulled in by d3dkmddi.h chain).
 * dispmprt.h also includes d3dkmddi.h and d3dkmdt.h internally.
 */
#include <dispmprt.h>
#else
/* Non-kernel: basic NT types for escape structure definitions */
#include <ntdef.h>
#endif

/* ======================================================================
 * Escape command interface
 *
 * Python sends commands through D3DKMTEscape -> DxgkDdiEscape.
 * The escape buffer starts with an AMDGPU_ESCAPE_HEADER followed
 * by command-specific data.
 * ====================================================================== */

/* Escape command codes */
typedef enum _AMDGPU_ESCAPE_CODE {
    AMDGPU_ESCAPE_GET_INFO          = 0x0001,
    AMDGPU_ESCAPE_READ_REG32        = 0x0010,
    AMDGPU_ESCAPE_WRITE_REG32       = 0x0011,
    AMDGPU_ESCAPE_MAP_BAR           = 0x0020,
    AMDGPU_ESCAPE_UNMAP_BAR         = 0x0021,
    AMDGPU_ESCAPE_ALLOC_DMA         = 0x0030,
    AMDGPU_ESCAPE_FREE_DMA          = 0x0031,
    AMDGPU_ESCAPE_MAP_VRAM          = 0x0040,
    AMDGPU_ESCAPE_REGISTER_EVENT    = 0x0050,
    AMDGPU_ESCAPE_ENABLE_MSI        = 0x0051,
    AMDGPU_ESCAPE_GET_IOMMU_INFO    = 0x0060,
} AMDGPU_ESCAPE_CODE;

/* Header for all escape commands */
typedef struct _AMDGPU_ESCAPE_HEADER {
    AMDGPU_ESCAPE_CODE  Command;
    NTSTATUS            Status;     /* Filled by driver on return */
    ULONG               Size;       /* Total size including header */
} AMDGPU_ESCAPE_HEADER;

/* ---- GET_INFO ---- */
typedef struct _AMDGPU_ESCAPE_GET_INFO_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    /* Output */
    USHORT  VendorId;
    USHORT  DeviceId;
    USHORT  SubsystemVendorId;
    USHORT  SubsystemId;
    UCHAR   RevisionId;
    UCHAR   Reserved[3];
    ULONG   NumBars;
    struct {
        LARGE_INTEGER   PhysicalAddress;
        ULONGLONG       Length;
        BOOLEAN         IsMemory;       /* TRUE = memory, FALSE = I/O */
        BOOLEAN         Is64Bit;
        BOOLEAN         IsPrefetchable;
        UCHAR           Reserved;
    } Bars[6];
    ULONGLONG   VramSizeBytes;
    ULONGLONG   VisibleVramSizeBytes;
} AMDGPU_ESCAPE_GET_INFO_DATA;

/* ---- READ_REG32 / WRITE_REG32 ---- */
typedef struct _AMDGPU_ESCAPE_REG32_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG   BarIndex;       /* Which BAR (usually 0 for MMIO) */
    ULONG   Offset;         /* Byte offset within BAR */
    ULONG   Value;          /* Value read/written */
} AMDGPU_ESCAPE_REG32_DATA;

/* ---- MAP_BAR / UNMAP_BAR ---- */
typedef struct _AMDGPU_ESCAPE_MAP_BAR_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       BarIndex;       /* Input: which BAR to map */
    ULONGLONG   Offset;         /* Input: offset within BAR */
    ULONGLONG   Length;         /* Input: length to map (0 = entire BAR) */
    PVOID       MappedAddress;  /* Output: usermode VA of mapped region */
    PVOID       MappingHandle;  /* Output: opaque handle for UNMAP */
} AMDGPU_ESCAPE_MAP_BAR_DATA;

/* ---- ALLOC_DMA / FREE_DMA ---- */
typedef struct _AMDGPU_ESCAPE_ALLOC_DMA_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Size;           /* Input: size in bytes */
    PVOID       CpuAddress;     /* Output: usermode VA */
    ULONGLONG   BusAddress;     /* Output: DMA/IOMMU-translated address */
    PVOID       AllocationHandle; /* Output: opaque handle for FREE */
} AMDGPU_ESCAPE_ALLOC_DMA_DATA;

/* ---- MAP_VRAM ---- */
typedef struct _AMDGPU_ESCAPE_MAP_VRAM_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Offset;         /* Input: offset within VRAM (via BAR2) */
    ULONGLONG   Length;         /* Input: length to map */
    PVOID       MappedAddress;  /* Output: usermode VA */
    PVOID       MappingHandle;  /* Output: opaque handle for unmap */
} AMDGPU_ESCAPE_MAP_VRAM_DATA;

/* ---- REGISTER_EVENT ---- */
typedef struct _AMDGPU_ESCAPE_REGISTER_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    HANDLE      EventHandle;    /* Input: user-mode event handle */
    ULONG       InterruptSource; /* Input: which interrupt to watch */
    ULONG       RegistrationId; /* Output: ID for deregistration */
} AMDGPU_ESCAPE_REGISTER_EVENT_DATA;

/* ---- ENABLE_MSI (configure IH ring for interrupt handling) ---- */
typedef struct _AMDGPU_ESCAPE_ENABLE_MSI_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    /* Input: IH ring configuration from Python */
    PVOID       IhRingDmaHandle;    /* DMA allocation handle (from ALLOC_DMA) */
    ULONG       IhRingSize;         /* Ring size in bytes (must be power of 2) */
    ULONG       IhRptrRegOffset;    /* BAR0 byte offset of IH_RB_RPTR */
    ULONG       IhWptrRegOffset;    /* BAR0 byte offset of IH_RB_WPTR */
    /* Output */
    BOOLEAN     Enabled;
    UCHAR       Reserved[3];
    ULONG       NumVectors;
} AMDGPU_ESCAPE_ENABLE_MSI_DATA;

/* ---- GET_IOMMU_INFO ---- */
typedef struct _AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    BOOLEAN     IommuPresent;
    BOOLEAN     IommuEnabled;
    BOOLEAN     DmaRemappingActive;
    UCHAR       Reserved;
} AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA;

/* ======================================================================
 * Internal driver structures (kernel-mode only)
 * ====================================================================== */

#ifdef _KERNEL_MODE

/* Pool allocation tag: 'AMDG' */
#define AMDGPU_POOL_TAG     'GDMA'

/* Maximum number of PCI BARs */
#define AMDGPU_MAX_BARS     6

/* Maximum number of registered events */
#define AMDGPU_MAX_EVENTS   64

/* Maximum number of DMA allocations tracked */
#define AMDGPU_MAX_DMA_ALLOCS   256

/* Per-BAR resource info */
typedef struct _AMDGPU_BAR_INFO {
    PHYSICAL_ADDRESS    PhysicalAddress;
    ULONGLONG           Length;
    PVOID               KernelAddress;  /* Kernel VA from MmMapIoSpaceEx */
    BOOLEAN             Mapped;
    BOOLEAN             IsMemory;
    BOOLEAN             Is64Bit;
    BOOLEAN             IsPrefetchable;
} AMDGPU_BAR_INFO;

/* DMA allocation tracking */
typedef struct _AMDGPU_DMA_ALLOC {
    PVOID               KernelVa;
    PHYSICAL_ADDRESS    BusAddress;
    ULONGLONG           Size;
    PMDL                Mdl;
    PVOID               UserVa;
    BOOLEAN             InUse;
} AMDGPU_DMA_ALLOC;

/* IH (Interrupt Handler) ring state — configured by Python via ENABLE_MSI */
typedef struct _AMDGPU_IH_RING {
    PVOID               RingBuffer;     /* Kernel VA of IH ring (from DMA alloc) */
    ULONG               RingSize;       /* Size in bytes */
    ULONG               RingMask;       /* RingSize - 1, for wrapping */
    ULONG               RptrRegOffset;  /* BAR0 byte offset of IH_RB_RPTR */
    ULONG               WptrRegOffset;  /* BAR0 byte offset of IH_RB_WPTR */
    volatile ULONG      Rptr;           /* Current read pointer (bytes) */
    BOOLEAN             Configured;
} AMDGPU_IH_RING;

/* Registered interrupt event */
typedef struct _AMDGPU_EVENT_REG {
    PKEVENT             Event;          /* Referenced kernel event object */
    ULONG               SourceId;       /* IH source ID to match */
    BOOLEAN             InUse;
} AMDGPU_EVENT_REG;

/* Per-adapter (device) context */
typedef struct _AMDGPU_ADAPTER {
    /* DXGK handles */
    PVOID                   DxgkHandle;
    DXGK_START_INFO         DxgkStartInfo;
    DXGKRNL_INTERFACE       DxgkInterface;

    /* PCI info */
    USHORT                  VendorId;
    USHORT                  DeviceId;
    USHORT                  SubsystemVendorId;
    USHORT                  SubsystemId;
    UCHAR                   RevisionId;

    /* BAR resources */
    AMDGPU_BAR_INFO         Bars[AMDGPU_MAX_BARS];
    ULONG                   NumBars;

    /* Interrupt */
    BOOLEAN                 InterruptEnabled;
    ULONG                   NumMsiVectors;
    PKINTERRUPT             InterruptObject;

    /* IH ring (configured by Python via ENABLE_MSI escape) */
    AMDGPU_IH_RING          IhRing;
    volatile LONG           IhPending;      /* Set by ISR, cleared by DPC */

    /* Registered events (set by Python via REGISTER_EVENT escape) */
    AMDGPU_EVENT_REG        Events[AMDGPU_MAX_EVENTS];
    KSPIN_LOCK              EventsLock;

    /* DMA allocations */
    AMDGPU_DMA_ALLOC        DmaAllocs[AMDGPU_MAX_DMA_ALLOCS];
    KSPIN_LOCK              DmaAllocsLock;

    /* VRAM info */
    ULONGLONG               VramSize;
    ULONGLONG               VisibleVramSize;

    /* Flags */
    BOOLEAN                 Started;
} AMDGPU_ADAPTER;

/*
 * DDI function declarations live in driver_entry.c (using DDI typedefs
 * from dispmprt.h) and are implemented across ddi_*.c files.
 */

#endif /* _KERNEL_MODE */
