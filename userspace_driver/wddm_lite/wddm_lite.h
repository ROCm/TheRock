/*
 * wddm_lite.h - Userspace interface to WDDM display driver escape channel
 *
 * Wraps D3DKMTEscape for communicating with the amdgpu_wddm.sys driver.
 * Provides register access, BAR mapping, DMA allocation, and compute
 * escape operations.
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <vector>

/*
 * We don't include d3dkmthk.h directly because it requires full WDK
 * include setup. Instead we define the minimal D3DKMT types we need
 * and load the functions dynamically from gdi32.dll.
 */

typedef UINT D3DKMT_HANDLE;

typedef struct _D3DKMT_ADAPTERINFO {
    D3DKMT_HANDLE hAdapter;
    LUID          AdapterLuid;
    ULONG         NumOfSources;
    BOOL          bPrecisePresentRegionsPreferred;
} D3DKMT_ADAPTERINFO;

typedef struct _D3DKMT_ENUMADAPTERS2 {
    ULONG               NumAdapters;
    D3DKMT_ADAPTERINFO *pAdapters;
} D3DKMT_ENUMADAPTERS2;

typedef struct _D3DKMT_OPENADAPTERFROMLUID {
    LUID          AdapterLuid;
    D3DKMT_HANDLE hAdapter;
} D3DKMT_OPENADAPTERFROMLUID;

/*
 * D3DKMT_CREATEDEVICE on x64:
 *   offset 0: union { hAdapter(UINT); pAdapter(PVOID); } = 8 bytes
 *   offset 8: Flags (UINT) = 4 bytes
 *   offset 12: hDevice (UINT) = 4 bytes
 *   offset 16: pCommandBuffer (PVOID) = 8 bytes
 *   ... more fields follow
 */
typedef struct _D3DKMT_CREATEDEVICE {
    union {
        D3DKMT_HANDLE hAdapter;
        PVOID         pAdapter;
    };
    UINT          Flags;
    D3DKMT_HANDLE hDevice;
    PVOID         pCommandBuffer;
    UINT          CommandBufferSize;
    UINT          _pad1;
    PVOID         pAllocationList;
    UINT          AllocationListSize;
    UINT          _pad2;
    PVOID         pPatchLocationList;
    UINT          PatchLocationListSize;
    UINT          _pad3;
} D3DKMT_CREATEDEVICE;

typedef struct _D3DKMT_DESTROYDEVICE {
    D3DKMT_HANDLE hDevice;
} D3DKMT_DESTROYDEVICE;

typedef struct _D3DKMT_CLOSEADAPTER {
    D3DKMT_HANDLE hAdapter;
} D3DKMT_CLOSEADAPTER;

/* D3DKMT_ESCAPETYPE */
#define D3DKMT_ESCAPE_DRIVERPRIVATE 0

/*
 * D3DKMT_ESCAPE on x64:
 *   offset 0: hAdapter (UINT)
 *   offset 4: hDevice (UINT)
 *   offset 8: Type (D3DKMT_ESCAPETYPE = enum = UINT)
 *   offset 12: Flags (D3DDDI_ESCAPEFLAGS = union of UINT = 4 bytes)
 *   offset 16: pPrivateDriverData (D3DKMT_PTR = PVOID, 8 bytes)
 *   offset 24: PrivateDriverDataSize (UINT)
 *   offset 28: hContext (UINT)
 */
typedef struct _D3DKMT_ESCAPE {
    D3DKMT_HANDLE hAdapter;
    D3DKMT_HANDLE hDevice;
    UINT          Type;
    UINT          Flags;
    PVOID         pPrivateDriverData;
    UINT          PrivateDriverDataSize;
    D3DKMT_HANDLE hContext;
} D3DKMT_ESCAPE;

/* NTSTATUS for userspace */
#ifndef NTSTATUS
typedef LONG NTSTATUS;
#endif

/* Pull in the shared escape structures from the driver header.
 * We redefine them here for userspace since the driver header
 * has kernel-mode dependencies. */

typedef enum _AMDGPU_ESCAPE_CODE {
    AMDGPU_ESCAPE_GET_INFO          = 0x0001,
    AMDGPU_ESCAPE_READ_REG32        = 0x0010,
    AMDGPU_ESCAPE_WRITE_REG32       = 0x0011,
    AMDGPU_ESCAPE_MAP_BAR           = 0x0020,
    AMDGPU_ESCAPE_UNMAP_BAR         = 0x0021,
    AMDGPU_ESCAPE_ALLOC_DMA         = 0x0030,
    AMDGPU_ESCAPE_FREE_DMA          = 0x0031,
    AMDGPU_ESCAPE_MAP_VRAM          = 0x0040,
    AMDGPU_ESCAPE_READ_VRAM         = 0x0041,
    AMDGPU_ESCAPE_REGISTER_EVENT    = 0x0050,
    AMDGPU_ESCAPE_ENABLE_MSI        = 0x0051,
    AMDGPU_ESCAPE_GET_IOMMU_INFO    = 0x0060,

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

/* No #pragma pack - must match driver's default packing */

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
    ULONG       MmioBarIndex;
    ULONG       VramBarIndex;
    BOOLEAN     Headless;
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

typedef struct _AMDGPU_ESCAPE_READ_VRAM_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Offset;
    ULONGLONG   Length;
    UCHAR       Data[1];    /* Flexible array - actual size = header + Length */
} AMDGPU_ESCAPE_READ_VRAM_DATA;

/* Memory flags */
#define AMDGPU_MEM_TYPE_VRAM        0x0001
#define AMDGPU_MEM_TYPE_GTT         0x0002
#define AMDGPU_MEM_TYPE_SYSTEM      0x0004
#define AMDGPU_MEM_FLAG_USERPTR     0x0010
#define AMDGPU_MEM_FLAG_HOST_ACCESS 0x0020
#define AMDGPU_MEM_FLAG_NONPAGED    0x0040
#define AMDGPU_MEM_FLAG_UNCACHED    0x0400
#define AMDGPU_MEM_FLAG_CONTIGUOUS  0x0800

typedef struct _AMDGPU_ESCAPE_ALLOC_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   SizeInBytes;
    ULONGLONG   Alignment;
    ULONG       Flags;
    ULONGLONG   VaAddress;
    PVOID       CpuAddress;
    ULONGLONG   GpuAddress;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_ALLOC_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_FREE_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   Handle;
} AMDGPU_ESCAPE_FREE_MEMORY_DATA;

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
    ULONGLONG   QueueId;
    ULONGLONG   DoorbellOffset;
} AMDGPU_ESCAPE_CREATE_QUEUE_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_QUEUE_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG   QueueId;
} AMDGPU_ESCAPE_DESTROY_QUEUE_DATA;

#define AMDGPU_EVENT_TYPE_SIGNAL    0
#define AMDGPU_EVENT_TYPE_QUEUE     7
#define AMDGPU_EVENT_TYPE_MEMORY    8

typedef struct _AMDGPU_ESCAPE_CREATE_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventType;
    ULONG       GpuId;
    BOOLEAN     AutoReset;
    UCHAR       Reserved[3];
    ULONG       EventId;
    ULONGLONG   EventPageAddress;
    ULONG       EventSlotIndex;
} AMDGPU_ESCAPE_CREATE_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_DESTROY_EVENT_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       EventId;
} AMDGPU_ESCAPE_DESTROY_EVENT_DATA;

typedef struct _AMDGPU_ESCAPE_GET_VERSION_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       KfdMajorVersion;
    ULONG       KfdMinorVersion;
} AMDGPU_ESCAPE_GET_VERSION_DATA;

typedef struct _AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   GpuClockCounter;
    ULONGLONG   CpuClockCounter;
    ULONGLONG   SystemClockCounter;
    ULONGLONG   SystemClockFrequencyHz;
    ULONGLONG   GpuClockFrequencyHz;
} AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA;

typedef struct _AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG       GpuId;
    ULONGLONG   LdsBase;
    ULONGLONG   LdsLimit;
    ULONGLONG   ScratchBase;
    ULONGLONG   ScratchLimit;
    ULONGLONG   GpuVmBase;
    ULONGLONG   GpuVmLimit;
} AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA;

/* ======================================================================
 * WddmLite - Userspace GPU access class
 * ====================================================================== */

class WddmLite {
public:
    WddmLite() : m_adapter(0), m_device(0), m_opened(false) {}
    ~WddmLite() { close(); }

    bool open();
    void close();
    bool isOpen() const { return m_opened; }

    /* Escape wrappers */
    bool getInfo(AMDGPU_ESCAPE_GET_INFO_DATA *info);
    bool readReg32(uint32_t offset, uint32_t *value, uint32_t barIndex = 0);
    bool writeReg32(uint32_t offset, uint32_t value, uint32_t barIndex = 0);
    bool mapVram(uint64_t offset, uint64_t length, void **addr, void **handle);
    bool unmapVram(void *addr, void *handle);
    bool readVram(uint64_t offset, uint64_t length, void *buffer);
    bool allocDma(uint64_t size, void **cpuAddr, uint64_t *busAddr, void **handle);
    bool freeDma(void *handle);

    /* Compute escapes */
    bool allocMemory(uint64_t size, uint32_t flags, void **cpuAddr,
                     uint64_t *gpuAddr, uint64_t *handle);
    bool freeMemory(uint64_t handle);
    bool createQueue(uint32_t queueType, uint64_t ringAddr, uint64_t ringSize,
                     uint64_t *queueId, uint64_t *doorbellOffset);
    bool destroyQueue(uint64_t queueId);
    bool createEvent(uint32_t eventType, uint32_t *eventId,
                     uint64_t *eventPageAddr, uint32_t *slotIndex);
    bool destroyEvent(uint32_t eventId);
    bool getVersion(uint32_t *major, uint32_t *minor);
    bool getClockCounters(uint64_t *gpuClock, uint64_t *cpuClock);
    bool getProcessApertures(uint64_t *gpuVmBase, uint64_t *gpuVmLimit);

    D3DKMT_HANDLE adapter() const { return m_adapter; }
    D3DKMT_HANDLE device() const { return m_device; }

private:
    bool escape(void *data, uint32_t size);

    D3DKMT_HANDLE m_adapter;
    D3DKMT_HANDLE m_device;
    bool m_opened;
};

/* ======================================================================
 * IP Discovery structures
 * ====================================================================== */

/* Hardware IDs */
#define HWID_MP0    255     /* PSP */
#define HWID_MP1    1       /* SMU */
#define HWID_GC     11      /* Graphics/Compute */
#define HWID_MMHUB  34
#define HWID_SDMA0  42
#define HWID_OSSSYS 40      /* IH */
#define HWID_NBIF   108

#define IP_DISCOVERY_SIGNATURE  0x28211407

struct IpBlock {
    uint16_t hwId;
    uint8_t  instance;
    uint8_t  numBaseAddrs;
    uint8_t  majorVer;
    uint8_t  minorVer;
    uint8_t  revision;
    uint32_t baseAddrs[8];  /* Up to 8 base address registers */
};

struct IpDiscoveryResult {
    IpBlock blocks[64];
    uint32_t numBlocks;

    /* Resolved base addresses */
    uint32_t mmhubBase;
    uint32_t gcBase;        /* GC BASE_IDX=0 */
    uint32_t gcBase1;       /* GC BASE_IDX=1 */
    uint32_t mp0Base;       /* PSP */
    uint32_t mp1Base;       /* SMU */
    uint32_t sdma0Base;
    uint32_t ihBase;        /* OSSSYS/IH */

    bool valid;
};

bool ipDiscovery(WddmLite &gpu, const AMDGPU_ESCAPE_GET_INFO_DATA &info,
                 IpDiscoveryResult &result);

/* ======================================================================
 * GMC Init
 * ====================================================================== */

struct GmcState {
    uint64_t vramStart;     /* FB_LOCATION_BASE << 24 */
    uint64_t vramEnd;       /* FB_LOCATION_TOP << 24 */
    uint64_t vramSize;
    uint64_t gartStart;
    uint64_t gartEnd;
    uint64_t gartSize;
    bool mmhubConfigured;
    bool gfxhubConfigured;
};

bool gmcInit(WddmLite &gpu, const IpDiscoveryResult &ipd, GmcState &gmc);

/* ======================================================================
 * PSP / SMU / Firmware
 * ====================================================================== */

/* PSP firmware types */
#define PSP_FW_TYPE_TOC         4
#define PSP_FW_TYPE_SMU         12  /* SMC/SMU */
#define PSP_FW_TYPE_SDMA0       0
#define PSP_FW_TYPE_PFP         33  /* RS64_PFP */
#define PSP_FW_TYPE_ME          30  /* RS64_ME */
#define PSP_FW_TYPE_MEC         36  /* RS64_MEC */
#define PSP_FW_TYPE_IMU_I       47
#define PSP_FW_TYPE_IMU_D       48
#define PSP_FW_TYPE_RLC_G       7
#define PSP_FW_TYPE_RLC_AUTO    57  /* RLC_AUTOLOAD */

/* PSP ring commands */
#define GFX_CMD_ID_LOAD_IP_FW   0x00006
#define GFX_CMD_ID_AUTOLOAD_RLC 0x00017

/* PSP ring type */
#define PSP_RING_TYPE_KM        2

/* SMU messages */
#define PPSMC_MSG_DisallowGfxOff 0x29

struct PspState {
    /* Ring buffer */
    void    *ringCpuAddr;
    uint64_t ringBusAddr;
    void    *ringDmaHandle;
    uint32_t ringWptr;

    /* Firmware buffer */
    void    *fwBufCpuAddr;
    uint64_t fwBufBusAddr;
    void    *fwBufDmaHandle;

    /* Fence */
    volatile uint32_t *fenceCpuAddr;
    uint64_t fenceBusAddr;
    void    *fenceDmaHandle;
    uint32_t fenceValue;

    bool ringCreated;
    bool sosAlive;
};

bool pspCheckSos(WddmLite &gpu, const IpDiscoveryResult &ipd);
bool pspRingCreate(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp);
bool pspRingDestroy(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp);
bool pspLoadFirmware(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp,
                     uint32_t fwType, const void *fwData, uint32_t fwSize);
bool pspTriggerAutoload(WddmLite &gpu, const IpDiscoveryResult &ipd, PspState &psp);
bool smuDisableGfxOff(WddmLite &gpu, const IpDiscoveryResult &ipd);

/* Firmware file loading */
bool loadFirmwareFile(const char *path, std::vector<uint8_t> &data);
