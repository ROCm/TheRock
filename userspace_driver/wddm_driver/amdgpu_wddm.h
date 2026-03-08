/*
 * amdgpu_wddm.h - Shared header for ROCm Display Driver
 *
 * WDDM display miniport with POST framebuffer passthrough.
 * Registers as Display class, acquires POST framebuffer from UEFI/GOP,
 * implements software Present for DWM desktop rendering, and provides
 * DxgkDdiEscape channel for Python compute driver.
 */

#pragma once

#ifdef _KERNEL_MODE
#include <ntddk.h>
#include <dispmprt.h>
#else
#include <ntdef.h>
#endif

/* ======================================================================
 * Escape command interface (shared with Python userspace driver)
 *
 * Identical to MCDM driver -- Python sends commands through
 * D3DKMTEscape -> DxgkDdiEscape.
 * ====================================================================== */

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

typedef struct _AMDGPU_ESCAPE_HEADER {
    AMDGPU_ESCAPE_CODE  Command;
    NTSTATUS            Status;
    ULONG               Size;
} AMDGPU_ESCAPE_HEADER;

typedef struct _AMDGPU_ESCAPE_GET_INFO_DATA {
    AMDGPU_ESCAPE_HEADER Header;
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
        BOOLEAN         IsMemory;
        BOOLEAN         Is64Bit;
        BOOLEAN         IsPrefetchable;
        UCHAR           Reserved;
    } Bars[6];
    ULONGLONG   VramSizeBytes;
    ULONGLONG   VisibleVramSizeBytes;
    ULONG       MmioBarIndex;       /* Which Bars[] entry is MMIO registers */
    ULONG       VramBarIndex;       /* Which Bars[] entry is VRAM aperture */
    BOOLEAN     Headless;           /* TRUE if compute-only (no display) */
    UCHAR       Reserved2[3];
} AMDGPU_ESCAPE_GET_INFO_DATA;

typedef struct _AMDGPU_ESCAPE_REG32_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG   BarIndex;
    ULONG   Offset;
    ULONG   Value;
} AMDGPU_ESCAPE_REG32_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_BAR_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       BarIndex;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    PVOID       MappedAddress;
    PVOID       MappingHandle;
} AMDGPU_ESCAPE_MAP_BAR_DATA;

typedef struct _AMDGPU_ESCAPE_ALLOC_DMA_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Size;
    PVOID       CpuAddress;
    ULONGLONG   BusAddress;
    PVOID       AllocationHandle;
} AMDGPU_ESCAPE_ALLOC_DMA_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_VRAM_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    PVOID       MappedAddress;
    PVOID       MappingHandle;
} AMDGPU_ESCAPE_MAP_VRAM_DATA;

typedef struct _AMDGPU_ESCAPE_REGISTER_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    HANDLE      EventHandle;
    ULONG       InterruptSource;
    ULONG       RegistrationId;
} AMDGPU_ESCAPE_REGISTER_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_ENABLE_MSI_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    PVOID       IhRingDmaHandle;
    ULONG       IhRingSize;
    ULONG       IhRptrRegOffset;
    ULONG       IhWptrRegOffset;
    BOOLEAN     Enabled;
    UCHAR       Reserved[3];
    ULONG       NumVectors;
} AMDGPU_ESCAPE_ENABLE_MSI_DATA;

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

/* Pool allocation tag: 'WDDM' */
#define AMDGPU_POOL_TAG     'MDDW'

#define AMDGPU_MAX_BARS     6
#define AMDGPU_MAX_EVENTS   64
#define AMDGPU_MAX_DMA_ALLOCS   256

/* Per-BAR resource info */
typedef struct _AMDGPU_BAR_INFO {
    PHYSICAL_ADDRESS    PhysicalAddress;
    ULONGLONG           Length;
    PVOID               KernelAddress;
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

/* IH ring state */
typedef struct _AMDGPU_IH_RING {
    PVOID               RingBuffer;
    ULONG               RingSize;
    ULONG               RingMask;
    ULONG               RptrRegOffset;
    ULONG               WptrRegOffset;
    volatile ULONG      Rptr;
    BOOLEAN             Configured;
} AMDGPU_IH_RING;

/* Registered interrupt event */
typedef struct _AMDGPU_EVENT_REG {
    PKEVENT             Event;
    ULONG               SourceId;
    BOOLEAN             InUse;
} AMDGPU_EVENT_REG;

/* ======================================================================
 * POST display state -- acquired from UEFI/GOP via
 * DxgkCbAcquirePostDisplayOwnership
 * ====================================================================== */

typedef struct _AMDGPU_POST_DISPLAY {
    BOOLEAN             Acquired;
    UINT                Width;
    UINT                Height;
    UINT                Pitch;
    D3DDDIFORMAT        ColorFormat;
    PHYSICAL_ADDRESS    FramebufferPhysAddr;
    PVOID               FramebufferKernelVa;
    ULONGLONG           FramebufferSize;
} AMDGPU_POST_DISPLAY;

/* VidPn runtime state */
typedef struct _AMDGPU_VIDPN_STATE {
    BOOLEAN             SourceVisible;
    BOOLEAN             PathActive;
    PHYSICAL_ADDRESS    PrimaryAddress;
} AMDGPU_VIDPN_STATE;

/* ======================================================================
 * DMA command structures for software Present
 * ====================================================================== */

#define AMDGPU_CMD_NOP       0
#define AMDGPU_CMD_BLT       1
#define AMDGPU_CMD_COLORFILL 2

typedef struct _AMDGPU_DMA_CMD {
    ULONG             Command;
    PHYSICAL_ADDRESS  SrcPhysAddr;
    PHYSICAL_ADDRESS  DstPhysAddr;
    ULONG             SrcPitch;
    ULONG             DstPitch;
    ULONG             SrcAllocIndex;
    ULONG             DstAllocIndex;
    RECT              SrcRect;
    RECT              DstRect;
    ULONG             BytesPerPixel;
    ULONG             FillColor;
    BOOLEAN           DstIsPrimary;
} AMDGPU_DMA_CMD;

/* Per-allocation private data */
typedef struct _AMDGPU_ALLOC_PRIVATE {
    UINT              Width;
    UINT              Height;
    UINT              Pitch;
    D3DDDIFORMAT      Format;
    BOOLEAN           IsPrimary;
} AMDGPU_ALLOC_PRIVATE;

/* ======================================================================
 * Per-allocation tracking (extended from MCDM)
 * ====================================================================== */

typedef struct _AMDGPU_ALLOCATION {
    ULONGLONG       Size;
    ULONG           SegmentId;
    BOOLEAN         InUse;
    /* WDDM display extensions */
    UINT            Width;
    UINT            Height;
    UINT            Pitch;
    D3DDDIFORMAT    Format;
    BOOLEAN         IsPrimary;
} AMDGPU_ALLOCATION;

/* ======================================================================
 * Per-adapter context (extended with POST display state)
 * ====================================================================== */

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

    /* IH ring */
    AMDGPU_IH_RING          IhRing;
    volatile LONG           IhPending;

    /* Registered events */
    AMDGPU_EVENT_REG        Events[AMDGPU_MAX_EVENTS];
    KSPIN_LOCK              EventsLock;

    /* DMA allocations */
    AMDGPU_DMA_ALLOC        DmaAllocs[AMDGPU_MAX_DMA_ALLOCS];
    KSPIN_LOCK              DmaAllocsLock;

    /* VRAM info */
    ULONGLONG               VramSize;
    ULONGLONG               VisibleVramSize;

    /* BAR classification (set by ClassifyBars after EnumerateBars) */
    ULONG                   MmioBarIndex;     /* MMIO register BAR (~512KB) */
    ULONG                   VramBarIndex;     /* VRAM aperture BAR (largest) */
    ULONG                   DoorbellBarIndex; /* Doorbell BAR */

    /* POST display state */
    AMDGPU_POST_DISPLAY     PostDisplay;

    /* VidPn state */
    AMDGPU_VIDPN_STATE      VidPnState;

    /* Flags */
    BOOLEAN                 Started;
    BOOLEAN                 Headless;       /* TRUE if no display output (compute-only GPU) */
} AMDGPU_ADAPTER;

/* Lazy MMIO BAR mapping */
NTSTATUS AmdGpuMapMmioIfNeeded(_Inout_ AMDGPU_ADAPTER *pAdapter);

#endif /* _KERNEL_MODE */
