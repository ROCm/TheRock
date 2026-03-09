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

    /* KFD-equivalent compute operations (Phase 2) */
    AMDGPU_ESCAPE_ALLOC_MEMORY      = 0x0100,
    AMDGPU_ESCAPE_FREE_MEMORY       = 0x0101,
    AMDGPU_ESCAPE_MAP_MEMORY        = 0x0102,
    AMDGPU_ESCAPE_UNMAP_MEMORY      = 0x0103,
    AMDGPU_ESCAPE_CREATE_QUEUE      = 0x0110,
    AMDGPU_ESCAPE_DESTROY_QUEUE     = 0x0111,
    AMDGPU_ESCAPE_UPDATE_QUEUE      = 0x0112,
    AMDGPU_ESCAPE_CREATE_EVENT      = 0x0120,
    AMDGPU_ESCAPE_DESTROY_EVENT     = 0x0121,
    AMDGPU_ESCAPE_SET_EVENT         = 0x0122,
    AMDGPU_ESCAPE_RESET_EVENT       = 0x0123,
    AMDGPU_ESCAPE_WAIT_EVENTS       = 0x0124,
    AMDGPU_ESCAPE_GET_PROCESS_APERTURES = 0x0130,
    AMDGPU_ESCAPE_SET_MEMORY_POLICY = 0x0131,
    AMDGPU_ESCAPE_SET_SCRATCH_BACKING = 0x0132,
    AMDGPU_ESCAPE_SET_TRAP_HANDLER  = 0x0133,
    AMDGPU_ESCAPE_GET_CLOCK_COUNTERS = 0x0140,
    AMDGPU_ESCAPE_GET_VERSION       = 0x0150,
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
 * KFD-equivalent compute escape structures (Phase 2)
 * ====================================================================== */

/* Memory type flags for ALLOC_MEMORY */
#define AMDGPU_MEM_TYPE_VRAM        0x0001
#define AMDGPU_MEM_TYPE_GTT         0x0002
#define AMDGPU_MEM_TYPE_SYSTEM      0x0004
#define AMDGPU_MEM_FLAG_USERPTR     0x0010
#define AMDGPU_MEM_FLAG_HOST_ACCESS 0x0020
#define AMDGPU_MEM_FLAG_NONPAGED    0x0040
#define AMDGPU_MEM_FLAG_READONLY    0x0080
#define AMDGPU_MEM_FLAG_EXECUTABLE  0x0100
#define AMDGPU_MEM_FLAG_AQL_QUEUE   0x0200
#define AMDGPU_MEM_FLAG_UNCACHED    0x0400
#define AMDGPU_MEM_FLAG_CONTIGUOUS  0x0800
#define AMDGPU_MEM_FLAG_NO_SUBSTITUTE 0x1000
#define AMDGPU_MEM_FLAG_SCRATCH     0x2000
#define AMDGPU_MEM_FLAG_GDS         0x4000
#define AMDGPU_MEM_FLAG_COHERENT    0x8000

typedef struct _AMDGPU_ESCAPE_ALLOC_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   SizeInBytes;
    ULONGLONG   Alignment;
    ULONG       Flags;
    ULONGLONG   VaAddress;
    /* Output */
    PVOID       CpuAddress;
    ULONGLONG   GpuAddress;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_ALLOC_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_FREE_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_FREE_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_MAP_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
    ULONG       GpuId;
    /* Output */
    ULONGLONG   GpuAddress;
} AMDGPU_ESCAPE_MAP_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_UNMAP_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
    ULONG       GpuId;
} AMDGPU_ESCAPE_UNMAP_MEMORY_DATA;

/* Queue types matching KFD */
#define AMDGPU_QUEUE_TYPE_COMPUTE     1
#define AMDGPU_QUEUE_TYPE_SDMA        2
#define AMDGPU_QUEUE_TYPE_COMPUTE_AQL 21

typedef struct _AMDGPU_ESCAPE_CREATE_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONG       QueueType;
    ULONG       QueuePercentage;
    LONG        Priority;
    ULONGLONG   QueueAddress;
    ULONGLONG   QueueSizeInBytes;
    ULONGLONG   WritePointerAddress;
    ULONGLONG   ReadPointerAddress;
    ULONGLONG   EopBufferAddress;
    ULONG       EopBufferSize;
    ULONGLONG   ContextSaveAddress;
    ULONG       ContextSaveSize;
    ULONG       SdmaEngineId;
    /* Output */
    ULONGLONG   QueueId;
    ULONGLONG   DoorbellOffset;
} AMDGPU_ESCAPE_CREATE_QUEUE_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   QueueId;
} AMDGPU_ESCAPE_DESTROY_QUEUE_DATA;

typedef struct _AMDGPU_ESCAPE_UPDATE_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   QueueId;
    ULONG       QueuePercentage;
    LONG        Priority;
    ULONGLONG   QueueAddress;
    ULONGLONG   QueueSizeInBytes;
} AMDGPU_ESCAPE_UPDATE_QUEUE_DATA;

/* Event types matching KFD HSA_EVENTTYPE */
#define AMDGPU_EVENT_TYPE_SIGNAL        0
#define AMDGPU_EVENT_TYPE_QUEUE         7
#define AMDGPU_EVENT_TYPE_MEMORY        8

typedef struct _AMDGPU_ESCAPE_CREATE_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventType;
    ULONG       GpuId;
    BOOLEAN     AutoReset;
    UCHAR       Reserved[3];
    /* Output */
    ULONG       EventId;
    ULONGLONG   EventPageAddress;
    ULONG       EventSlotIndex;
} AMDGPU_ESCAPE_CREATE_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_DESTROY_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_SET_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_SET_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_RESET_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_RESET_EVENT_DATA;

#define AMDGPU_MAX_WAIT_EVENTS 16

typedef struct _AMDGPU_ESCAPE_WAIT_EVENTS_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       NumEvents;
    ULONG       EventIds[AMDGPU_MAX_WAIT_EVENTS];
    BOOLEAN     WaitAll;
    UCHAR       Reserved[3];
    ULONG       TimeoutMs;
    /* Output */
    ULONG       SignaledIndex;
} AMDGPU_ESCAPE_WAIT_EVENTS_DATA;

typedef struct _AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    /* Output */
    ULONGLONG   LdsBase;
    ULONGLONG   LdsLimit;
    ULONGLONG   ScratchBase;
    ULONGLONG   ScratchLimit;
    ULONGLONG   GpuVmBase;
    ULONGLONG   GpuVmLimit;
} AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA;

typedef struct _AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONG       DefaultPolicy;
    ULONG       AlternatePolicy;
    ULONGLONG   AlternateApertureBase;
    ULONGLONG   AlternateApertureSize;
} AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA;

typedef struct _AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   ScratchBackingVa;
    ULONGLONG   ScratchBackingSize;
} AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA;

typedef struct _AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   TbaAddress;
    ULONGLONG   TbaSize;
    ULONGLONG   TmaAddress;
    ULONGLONG   TmaSize;
} AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA;

typedef struct _AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    /* Output */
    ULONGLONG   GpuClockCounter;
    ULONGLONG   CpuClockCounter;
    ULONGLONG   SystemClockCounter;
    ULONGLONG   SystemClockFrequencyHz;
    ULONGLONG   GpuClockFrequencyHz;
} AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA;

typedef struct _AMDGPU_ESCAPE_GET_VERSION_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    /* Output */
    ULONG       KfdMajorVersion;
    ULONG       KfdMinorVersion;
} AMDGPU_ESCAPE_GET_VERSION_DATA;

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
 * GPU memory allocation tracking (Phase 2)
 * ====================================================================== */

#define AMDGPU_MAX_GPU_ALLOCS   4096
#define AMDGPU_MAX_GPU_QUEUES   64
#define AMDGPU_MAX_GPU_EVENTS   256

/* GPU memory allocation */
typedef struct _AMDGPU_GPU_ALLOC {
    BOOLEAN         InUse;
    ULONG           Flags;              /* AMDGPU_MEM_TYPE_* | AMDGPU_MEM_FLAG_* */
    ULONGLONG       SizeInBytes;
    ULONGLONG       GpuVa;             /* GPU virtual address */
    PVOID           CpuVa;             /* CPU mapping (if host-accessible) */
    PHYSICAL_ADDRESS PhysAddr;          /* Physical/bus address */
    PMDL            Mdl;                /* MDL for user mapping */
    PVOID           KernelVa;           /* Kernel mapping */
} AMDGPU_GPU_ALLOC;

/* Compute queue */
typedef struct _AMDGPU_GPU_QUEUE {
    BOOLEAN         InUse;
    ULONG           QueueType;
    ULONG           QueueId;
    ULONGLONG       RingBufferGpuVa;
    ULONGLONG       RingSizeBytes;
    ULONGLONG       DoorbellOffset;
    LONG            Priority;
    /* MQD (Memory Queue Descriptor) for MES */
    PVOID           MqdKernelVa;
    PHYSICAL_ADDRESS MqdPhysAddr;
} AMDGPU_GPU_QUEUE;

/* Kernel event for signaling */
typedef struct _AMDGPU_GPU_EVENT {
    BOOLEAN         InUse;
    ULONG           EventType;
    ULONG           EventId;
    PKEVENT         KernelEvent;        /* KeSetEvent target */
    BOOLEAN         AutoReset;
    BOOLEAN         Signaled;
    ULONGLONG       EventPageSlot;      /* Offset in event page */
} AMDGPU_GPU_EVENT;

/* Event page - shared memory page for signal values */
typedef struct _AMDGPU_EVENT_PAGE {
    PVOID           KernelVa;
    PHYSICAL_ADDRESS PhysAddr;
    PMDL            Mdl;
    PVOID           UserVa;
    BOOLEAN         Allocated;
} AMDGPU_EVENT_PAGE;

/* Extended adapter context with compute support */
typedef struct _AMDGPU_COMPUTE_STATE {
    /* GPU memory allocations */
    AMDGPU_GPU_ALLOC    Allocs[AMDGPU_MAX_GPU_ALLOCS];
    KSPIN_LOCK          AllocsLock;
    ULONG               NextAllocHandle;

    /* Compute queues */
    AMDGPU_GPU_QUEUE    Queues[AMDGPU_MAX_GPU_QUEUES];
    KSPIN_LOCK          QueuesLock;
    ULONG               NextQueueId;

    /* Events */
    AMDGPU_GPU_EVENT    Events[AMDGPU_MAX_GPU_EVENTS];
    KSPIN_LOCK          EventsLock;
    ULONG               NextEventId;
    AMDGPU_EVENT_PAGE   EventPage;

    /* GPU VM apertures */
    ULONGLONG           GpuVmBase;      /* Start of GPUVM range */
    ULONGLONG           GpuVmLimit;     /* End of GPUVM range */
    ULONGLONG           LdsBase;
    ULONGLONG           LdsLimit;
    ULONGLONG           ScratchBase;
    ULONGLONG           ScratchLimit;

    /* Scratch backing memory */
    ULONGLONG           ScratchBackingVa;
    ULONGLONG           ScratchBackingSize;

    /* Trap handler */
    ULONGLONG           TbaAddress;
    ULONGLONG           TbaSize;
    ULONGLONG           TmaAddress;
    ULONGLONG           TmaSize;

    /* Memory policy */
    ULONG               DefaultCachePolicy;
    ULONG               AlternateCachePolicy;

    BOOLEAN             Initialized;
} AMDGPU_COMPUTE_STATE;

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

    /* Compute state (Phase 2) */
    AMDGPU_COMPUTE_STATE    Compute;

    /* Flags */
    BOOLEAN                 Started;
    BOOLEAN                 Headless;       /* TRUE if no display output (compute-only GPU) */
} AMDGPU_ADAPTER;

/* Lazy MMIO BAR mapping */
NTSTATUS AmdGpuMapMmioIfNeeded(_Inout_ AMDGPU_ADAPTER *pAdapter);

/* Compute state init/cleanup */
NTSTATUS AmdGpuComputeInit(_Inout_ AMDGPU_ADAPTER *pAdapter);
void AmdGpuComputeCleanup(_Inout_ AMDGPU_ADAPTER *pAdapter);

/* Compute escape handlers */
NTSTATUS EscapeAllocMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                           _Inout_ AMDGPU_ESCAPE_ALLOC_MEMORY_DATA *pData);
NTSTATUS EscapeFreeMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                          _Inout_ AMDGPU_ESCAPE_FREE_MEMORY_DATA *pData);
NTSTATUS EscapeMapMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                         _Inout_ AMDGPU_ESCAPE_MAP_MEMORY_DATA *pData);
NTSTATUS EscapeUnmapMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                           _Inout_ AMDGPU_ESCAPE_UNMAP_MEMORY_DATA *pData);
NTSTATUS EscapeCreateQueue(_In_ AMDGPU_ADAPTER *pAdapter,
                           _Inout_ AMDGPU_ESCAPE_CREATE_QUEUE_DATA *pData);
NTSTATUS EscapeDestroyQueue(_In_ AMDGPU_ADAPTER *pAdapter,
                            _Inout_ AMDGPU_ESCAPE_DESTROY_QUEUE_DATA *pData);
NTSTATUS EscapeUpdateQueue(_In_ AMDGPU_ADAPTER *pAdapter,
                           _Inout_ AMDGPU_ESCAPE_UPDATE_QUEUE_DATA *pData);
NTSTATUS EscapeCreateEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                           _Inout_ AMDGPU_ESCAPE_CREATE_EVENT_DATA *pData);
NTSTATUS EscapeDestroyEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                            _Inout_ AMDGPU_ESCAPE_DESTROY_EVENT_DATA *pData);
NTSTATUS EscapeSetEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                        _Inout_ AMDGPU_ESCAPE_SET_EVENT_DATA *pData);
NTSTATUS EscapeResetEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                          _Inout_ AMDGPU_ESCAPE_RESET_EVENT_DATA *pData);
NTSTATUS EscapeWaitEvents(_In_ AMDGPU_ADAPTER *pAdapter,
                          _Inout_ AMDGPU_ESCAPE_WAIT_EVENTS_DATA *pData);
NTSTATUS EscapeGetProcessApertures(_In_ AMDGPU_ADAPTER *pAdapter,
                                   _Inout_ AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA *pData);
NTSTATUS EscapeSetMemoryPolicy(_In_ AMDGPU_ADAPTER *pAdapter,
                               _Inout_ AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA *pData);
NTSTATUS EscapeSetScratchBacking(_In_ AMDGPU_ADAPTER *pAdapter,
                                 _Inout_ AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA *pData);
NTSTATUS EscapeSetTrapHandler(_In_ AMDGPU_ADAPTER *pAdapter,
                              _Inout_ AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA *pData);
NTSTATUS EscapeGetClockCounters(_In_ AMDGPU_ADAPTER *pAdapter,
                                _Inout_ AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA *pData);
NTSTATUS EscapeGetVersion(_In_ AMDGPU_ADAPTER *pAdapter,
                          _Inout_ AMDGPU_ESCAPE_GET_VERSION_DATA *pData);

#endif /* _KERNEL_MODE */
