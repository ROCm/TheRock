/*
 * wddm_lite.cpp - D3DKMT escape wrapper implementation
 *
 * Opens the AMD GPU adapter via D3DKMTEnumAdapters2, creates a device,
 * and provides typed wrappers for each escape command.
 */

#include "wddm_lite.h"
#include <cstring>

#pragma comment(lib, "gdi32.lib")

/* D3DKMT function pointers - loaded from gdi32.dll */
typedef NTSTATUS(APIENTRY *PFN_D3DKMTEnumAdapters2)(D3DKMT_ENUMADAPTERS2 *);
typedef NTSTATUS(APIENTRY *PFN_D3DKMTOpenAdapterFromLuid)(D3DKMT_OPENADAPTERFROMLUID *);
typedef NTSTATUS(APIENTRY *PFN_D3DKMTCreateDevice)(D3DKMT_CREATEDEVICE *);
typedef NTSTATUS(APIENTRY *PFN_D3DKMTDestroyDevice)(D3DKMT_DESTROYDEVICE *);
typedef NTSTATUS(APIENTRY *PFN_D3DKMTCloseAdapter)(D3DKMT_CLOSEADAPTER *);
typedef NTSTATUS(APIENTRY *PFN_D3DKMTEscape)(D3DKMT_ESCAPE *);

static PFN_D3DKMTEnumAdapters2 pfnEnumAdapters2;
static PFN_D3DKMTOpenAdapterFromLuid pfnOpenAdapterFromLuid;
static PFN_D3DKMTCreateDevice pfnCreateDevice;
static PFN_D3DKMTDestroyDevice pfnDestroyDevice;
static PFN_D3DKMTCloseAdapter pfnCloseAdapter;
static PFN_D3DKMTEscape pfnEscape;

static bool loadD3DKMTFunctions()
{
    static bool loaded = false;
    static bool ok = false;
    if (loaded) return ok;
    loaded = true;

    HMODULE hGdi = LoadLibraryA("gdi32.dll");
    if (!hGdi) return false;

    pfnEnumAdapters2 = (PFN_D3DKMTEnumAdapters2)GetProcAddress(hGdi, "D3DKMTEnumAdapters2");
    pfnOpenAdapterFromLuid = (PFN_D3DKMTOpenAdapterFromLuid)GetProcAddress(hGdi, "D3DKMTOpenAdapterFromLuid");
    pfnCreateDevice = (PFN_D3DKMTCreateDevice)GetProcAddress(hGdi, "D3DKMTCreateDevice");
    pfnDestroyDevice = (PFN_D3DKMTDestroyDevice)GetProcAddress(hGdi, "D3DKMTDestroyDevice");
    pfnCloseAdapter = (PFN_D3DKMTCloseAdapter)GetProcAddress(hGdi, "D3DKMTCloseAdapter");
    pfnEscape = (PFN_D3DKMTEscape)GetProcAddress(hGdi, "D3DKMTEscape");

    ok = pfnEnumAdapters2 && pfnOpenAdapterFromLuid && pfnCreateDevice &&
         pfnDestroyDevice && pfnCloseAdapter && pfnEscape;
    return ok;
}

bool WddmLite::open()
{
    if (m_opened) return true;
    if (!loadD3DKMTFunctions()) {
        printf("ERROR: Failed to load D3DKMT functions\n");
        return false;
    }

    /* Enumerate adapters to find AMD GPU */
    printf("DEBUG: sizeof(D3DKMT_ADAPTERINFO)=%zu sizeof(D3DKMT_ENUMADAPTERS2)=%zu\n",
           sizeof(D3DKMT_ADAPTERINFO), sizeof(D3DKMT_ENUMADAPTERS2));
    printf("DEBUG: sizeof(D3DKMT_CREATEDEVICE)=%zu sizeof(D3DKMT_ESCAPE)=%zu\n",
           sizeof(D3DKMT_CREATEDEVICE), sizeof(D3DKMT_ESCAPE));

    D3DKMT_ENUMADAPTERS2 enumArgs = {};
    enumArgs.NumAdapters = 0;
    enumArgs.pAdapters = nullptr;

    NTSTATUS status = pfnEnumAdapters2(&enumArgs);
    printf("DEBUG: EnumAdapters2 count: status=0x%08X NumAdapters=%u\n",
           (unsigned)status, enumArgs.NumAdapters);
    if (status != 0 && enumArgs.NumAdapters == 0) {
        printf("ERROR: D3DKMTEnumAdapters2 count failed: 0x%08X\n", (unsigned)status);
        return false;
    }

    uint32_t numAdapters = enumArgs.NumAdapters;
    auto *adapters = new D3DKMT_ADAPTERINFO[numAdapters];
    enumArgs.pAdapters = adapters;

    status = pfnEnumAdapters2(&enumArgs);
    printf("DEBUG: EnumAdapters2 fill: status=0x%08X\n", (unsigned)status);
    if (status != 0) {
        printf("ERROR: D3DKMTEnumAdapters2 failed: 0x%08X\n", (unsigned)status);
        delete[] adapters;
        return false;
    }

    /* Find AMD adapter (vendor 0x1002) */
    bool found = false;
    LUID adapterLuid = {};
    for (uint32_t i = 0; i < enumArgs.NumAdapters; i++) {
        printf("DEBUG: Adapter %u: hAdapter=%u LUID=%08X:%08X NumSources=%u\n",
               i, adapters[i].hAdapter,
               adapters[i].AdapterLuid.HighPart, adapters[i].AdapterLuid.LowPart,
               adapters[i].NumOfSources);

        /* Open each adapter to check vendor ID */
        D3DKMT_OPENADAPTERFROMLUID openArgs = {};
        openArgs.AdapterLuid = adapters[i].AdapterLuid;
        NTSTATUS openStatus = pfnOpenAdapterFromLuid(&openArgs);
        printf("DEBUG: OpenAdapterFromLuid: status=0x%08X hAdapter=%u\n",
               (unsigned)openStatus, openArgs.hAdapter);
        if (openStatus == 0) {
            /* Try GET_INFO to see if it's our driver */
            D3DKMT_CREATEDEVICE createDev = {};
            createDev.hAdapter = openArgs.hAdapter;
            NTSTATUS devStatus = pfnCreateDevice(&createDev);
            printf("DEBUG: CreateDevice: status=0x%08X hDevice=%u\n",
                   (unsigned)devStatus, createDev.hDevice);
            if (devStatus == 0) {
                AMDGPU_ESCAPE_GET_INFO_DATA info = {};
                info.Header.Command = AMDGPU_ESCAPE_GET_INFO;
                info.Header.Size = sizeof(info);

                D3DKMT_ESCAPE esc = {};
                esc.hAdapter = openArgs.hAdapter;
                esc.hDevice = createDev.hDevice;
                esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
                esc.pPrivateDriverData = &info;
                esc.PrivateDriverDataSize = sizeof(info);

                NTSTATUS escStatus = pfnEscape(&esc);
                printf("DEBUG: Escape GET_INFO: status=0x%08X VendorId=0x%04X DeviceId=0x%04X\n",
                       (unsigned)escStatus, info.VendorId, info.DeviceId);

                if (escStatus == 0 && info.VendorId == 0x1002) {
                    m_adapter = openArgs.hAdapter;
                    m_device = createDev.hDevice;
                    adapterLuid = adapters[i].AdapterLuid;
                    found = true;
                    printf("Found AMD GPU: %04X:%04X (adapter %u)\n",
                           info.VendorId, info.DeviceId, i);
                    break;
                }

                D3DKMT_DESTROYDEVICE destroyDev = {};
                destroyDev.hDevice = createDev.hDevice;
                pfnDestroyDevice(&destroyDev);
            }
            D3DKMT_CLOSEADAPTER closeAdapt = {};
            closeAdapt.hAdapter = openArgs.hAdapter;
            pfnCloseAdapter(&closeAdapt);
        }
    }

    delete[] adapters;

    if (!found) {
        printf("ERROR: No AMD GPU adapter found\n");
        return false;
    }

    m_opened = true;
    return true;
}

void WddmLite::close()
{
    if (!m_opened) return;

    if (m_device) {
        D3DKMT_DESTROYDEVICE destroyDev = {};
        destroyDev.hDevice = m_device;
        pfnDestroyDevice(&destroyDev);
        m_device = 0;
    }

    if (m_adapter) {
        D3DKMT_CLOSEADAPTER closeAdapt = {};
        closeAdapt.hAdapter = m_adapter;
        pfnCloseAdapter(&closeAdapt);
        m_adapter = 0;
    }

    m_opened = false;
}

bool WddmLite::escape(void *data, uint32_t size)
{
    if (!m_opened) return false;

    D3DKMT_ESCAPE esc = {};
    esc.hAdapter = m_adapter;
    esc.hDevice = m_device;
    esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
    esc.pPrivateDriverData = data;
    esc.PrivateDriverDataSize = size;

    NTSTATUS status = pfnEscape(&esc);
    if (status != 0) {
        auto *hdr = (AMDGPU_ESCAPE_HEADER *)data;
        printf("ERROR: D3DKMTEscape cmd=0x%04X failed: NTSTATUS=0x%08X, DriverStatus=0x%08X\n",
               hdr->Command, (unsigned)status, (unsigned)hdr->Status);
        return false;
    }
    return true;
}

bool WddmLite::getInfo(AMDGPU_ESCAPE_GET_INFO_DATA *info)
{
    memset(info, 0, sizeof(*info));
    info->Header.Command = AMDGPU_ESCAPE_GET_INFO;
    info->Header.Size = sizeof(*info);
    return escape(info, sizeof(*info));
}

bool WddmLite::readReg32(uint32_t offset, uint32_t *value, uint32_t barIndex)
{
    AMDGPU_ESCAPE_REG32_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_READ_REG32;
    data.Header.Size = sizeof(data);
    data.BarIndex = barIndex;
    data.Offset = offset;

    if (!escape(&data, sizeof(data))) return false;
    *value = data.Value;
    return true;
}

bool WddmLite::writeReg32(uint32_t offset, uint32_t value, uint32_t barIndex)
{
    AMDGPU_ESCAPE_REG32_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_WRITE_REG32;
    data.Header.Size = sizeof(data);
    data.BarIndex = barIndex;
    data.Offset = offset;
    data.Value = value;

    return escape(&data, sizeof(data));
}

bool WddmLite::mapVram(uint64_t offset, uint64_t length, void **addr, void **handle)
{
    AMDGPU_ESCAPE_MAP_VRAM_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
    data.Header.Size = sizeof(data);
    data.Offset = offset;
    data.Length = length;

    if (!escape(&data, sizeof(data))) return false;
    *addr = data.MappedAddress;
    *handle = data.MappingHandle;
    return true;
}

bool WddmLite::readVram(uint64_t offset, uint64_t length, void *buffer)
{
    /* Allocate escape buffer: header fields + data */
    size_t dataOffset = offsetof(AMDGPU_ESCAPE_READ_VRAM_DATA, Data);
    size_t totalSize = dataOffset + (size_t)length;
    auto *data = (AMDGPU_ESCAPE_READ_VRAM_DATA *)calloc(1, totalSize);
    if (!data) return false;

    data->Header.Command = AMDGPU_ESCAPE_READ_VRAM;
    data->Header.Size = (ULONG)totalSize;
    data->Offset = offset;
    data->Length = length;

    bool ok = escape(data, (uint32_t)totalSize);
    if (ok) {
        memcpy(buffer, data->Data, (size_t)length);
    }
    free(data);
    return ok;
}

bool WddmLite::unmapVram(void *addr, void *handle)
{
    AMDGPU_ESCAPE_MAP_BAR_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
    data.Header.Size = sizeof(data);
    data.MappedAddress = addr;
    data.MappingHandle = handle;

    return escape(&data, sizeof(data));
}

bool WddmLite::allocDma(uint64_t size, void **cpuAddr, uint64_t *busAddr, void **handle)
{
    AMDGPU_ESCAPE_ALLOC_DMA_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_ALLOC_DMA;
    data.Header.Size = sizeof(data);
    data.Size = size;

    if (!escape(&data, sizeof(data))) return false;
    *cpuAddr = data.CpuAddress;
    *busAddr = data.BusAddress;
    *handle = data.AllocationHandle;
    return true;
}

bool WddmLite::freeDma(void *handle)
{
    AMDGPU_ESCAPE_ALLOC_DMA_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_FREE_DMA;
    data.Header.Size = sizeof(data);
    data.AllocationHandle = handle;

    return escape(&data, sizeof(data));
}

bool WddmLite::allocMemory(uint64_t size, uint32_t flags, void **cpuAddr,
                            uint64_t *gpuAddr, uint64_t *handle)
{
    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    data.Header.Size = sizeof(data);
    data.GpuId = 0;
    data.SizeInBytes = size;
    data.Flags = flags;

    if (!escape(&data, sizeof(data))) return false;
    if (cpuAddr) *cpuAddr = data.CpuAddress;
    if (gpuAddr) *gpuAddr = data.GpuAddress;
    if (handle) *handle = data.Handle;
    return true;
}

bool WddmLite::freeMemory(uint64_t handle)
{
    AMDGPU_ESCAPE_FREE_MEMORY_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_FREE_MEMORY;
    data.Header.Size = sizeof(data);
    data.Handle = handle;

    return escape(&data, sizeof(data));
}

bool WddmLite::createQueue(uint32_t queueType, uint64_t ringAddr, uint64_t ringSize,
                            uint64_t *queueId, uint64_t *doorbellOffset)
{
    AMDGPU_ESCAPE_CREATE_QUEUE_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_CREATE_QUEUE;
    data.Header.Size = sizeof(data);
    data.GpuId = 0;
    data.QueueType = queueType;
    data.QueuePercentage = 100;
    data.Priority = 0;
    data.QueueAddress = ringAddr;
    data.QueueSizeInBytes = ringSize;

    if (!escape(&data, sizeof(data))) return false;
    if (queueId) *queueId = data.QueueId;
    if (doorbellOffset) *doorbellOffset = data.DoorbellOffset;
    return true;
}

bool WddmLite::destroyQueue(uint64_t queueId)
{
    AMDGPU_ESCAPE_DESTROY_QUEUE_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_DESTROY_QUEUE;
    data.Header.Size = sizeof(data);
    data.QueueId = queueId;

    return escape(&data, sizeof(data));
}

bool WddmLite::createEvent(uint32_t eventType, uint32_t *eventId,
                            uint64_t *eventPageAddr, uint32_t *slotIndex)
{
    AMDGPU_ESCAPE_CREATE_EVENT_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_CREATE_EVENT;
    data.Header.Size = sizeof(data);
    data.EventType = eventType;
    data.GpuId = 0;
    data.AutoReset = FALSE;

    if (!escape(&data, sizeof(data))) return false;
    if (eventId) *eventId = data.EventId;
    if (eventPageAddr) *eventPageAddr = data.EventPageAddress;
    if (slotIndex) *slotIndex = data.EventSlotIndex;
    return true;
}

bool WddmLite::destroyEvent(uint32_t eventId)
{
    AMDGPU_ESCAPE_DESTROY_EVENT_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_DESTROY_EVENT;
    data.Header.Size = sizeof(data);
    data.EventId = eventId;

    return escape(&data, sizeof(data));
}

bool WddmLite::getVersion(uint32_t *major, uint32_t *minor)
{
    AMDGPU_ESCAPE_GET_VERSION_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_GET_VERSION;
    data.Header.Size = sizeof(data);

    if (!escape(&data, sizeof(data))) return false;
    if (major) *major = data.KfdMajorVersion;
    if (minor) *minor = data.KfdMinorVersion;
    return true;
}

bool WddmLite::getClockCounters(uint64_t *gpuClock, uint64_t *cpuClock)
{
    AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_GET_CLOCK_COUNTERS;
    data.Header.Size = sizeof(data);
    data.GpuId = 0;

    if (!escape(&data, sizeof(data))) return false;
    if (gpuClock) *gpuClock = data.GpuClockCounter;
    if (cpuClock) *cpuClock = data.CpuClockCounter;
    return true;
}

bool WddmLite::getProcessApertures(uint64_t *gpuVmBase, uint64_t *gpuVmLimit)
{
    AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA data = {};
    data.Header.Command = AMDGPU_ESCAPE_GET_PROCESS_APERTURES;
    data.Header.Size = sizeof(data);
    data.GpuId = 0;

    if (!escape(&data, sizeof(data))) return false;
    if (gpuVmBase) *gpuVmBase = data.GpuVmBase;
    if (gpuVmLimit) *gpuVmLimit = data.GpuVmLimit;
    return true;
}
