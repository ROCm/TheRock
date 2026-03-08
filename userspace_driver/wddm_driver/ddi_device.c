/*
 * ddi_device.c - Device lifecycle DDIs for ROCm Display Driver
 *
 * Key difference from MCDM: StartDevice acquires POST display ownership
 * and reports 1 video present source + 1 child device.
 */

#include "amdgpu_wddm.h"

typedef struct _AMDGPU_DEVICE_CONTEXT {
    PVOID                   AdapterContext;
    HANDLE                  hDevice;
} AMDGPU_DEVICE_CONTEXT;

typedef struct _AMDGPU_CONTEXT {
    PVOID                   DeviceContext;
    UINT                    NodeOrdinal;
    UINT                    EngineAffinity;
} AMDGPU_CONTEXT;

typedef struct _AMDGPU_PROCESS_CONTEXT {
    HANDLE                  hKmdProcess;
} AMDGPU_PROCESS_CONTEXT;

/* ======================================================================
 * AddDevice
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuAddDevice(
    IN_CONST_PDEVICE_OBJECT     PhysicalDeviceObject,
    OUT_PPVOID                  MiniportDeviceContext
    )
{
    AMDGPU_ADAPTER *pAdapter;

    UNREFERENCED_PARAMETER(PhysicalDeviceObject);

    KdPrint(("AmdGpuWddm: AddDevice called\n"));

    {
        UNICODE_STRING KeyPath;
        OBJECT_ATTRIBUTES ObjAttrs;
        HANDLE hKey;
        NTSTATUS s;
        UNICODE_STRING ValName;
        ULONG Val;

        RtlInitUnicodeString(&KeyPath,
            L"\\Registry\\Machine\\SOFTWARE\\AmdGpuWddm");
        InitializeObjectAttributes(&ObjAttrs, &KeyPath,
            OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, NULL, NULL);
        s = ZwCreateKey(&hKey, KEY_ALL_ACCESS, &ObjAttrs, 0, NULL,
            REG_OPTION_NON_VOLATILE, NULL);
        if (NT_SUCCESS(s)) {
            RtlInitUnicodeString(&ValName, L"AddDevice");
            Val = 1;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));
            ZwClose(hKey);
        }
    }

    pAdapter = (AMDGPU_ADAPTER *)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(AMDGPU_ADAPTER), AMDGPU_POOL_TAG);
    if (pAdapter == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(pAdapter, sizeof(*pAdapter));
    KeInitializeSpinLock(&pAdapter->DmaAllocsLock);
    KeInitializeSpinLock(&pAdapter->EventsLock);

    *MiniportDeviceContext = pAdapter;
    KdPrint(("AmdGpuWddm: AddDevice succeeded, ctx=%p\n", pAdapter));
    return STATUS_SUCCESS;
}

/* ======================================================================
 * PCI BAR enumeration (identical to MCDM)
 * ====================================================================== */

#define mmRCC_CONFIG_MEMSIZE        0xDE3
#define mmRCC_CONFIG_MEMSIZE_BYTE   (mmRCC_CONFIG_MEMSIZE * 4)

static NTSTATUS
EnumerateBars(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    DXGK_DEVICE_INFO DeviceInfo;
    NTSTATUS Status;
    PCM_RESOURCE_LIST ResList;
    PCM_FULL_RESOURCE_DESCRIPTOR FullDesc;
    PCM_PARTIAL_RESOURCE_LIST PartialList;
    PCM_PARTIAL_RESOURCE_DESCRIPTOR Desc;
    ULONG i;
    ULONG BarIndex = 0;

    Status = pAdapter->DxgkInterface.DxgkCbGetDeviceInformation(
        pAdapter->DxgkInterface.DeviceHandle, &DeviceInfo);
    if (!NT_SUCCESS(Status))
        return Status;

    ResList = DeviceInfo.TranslatedResourceList;
    if (ResList == NULL || ResList->Count == 0)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    FullDesc = &ResList->List[0];
    PartialList = &FullDesc->PartialResourceList;

    for (i = 0; i < PartialList->Count; i++) {
        Desc = &PartialList->PartialDescriptors[i];

        switch (Desc->Type) {
        case CmResourceTypeMemory:
        case CmResourceTypeMemoryLarge:
            if (BarIndex < AMDGPU_MAX_BARS) {
                PHYSICAL_ADDRESS PhysAddr;
                ULONGLONG Length;

                PhysAddr = Desc->u.Memory.Start;

                if (Desc->Type == CmResourceTypeMemoryLarge) {
                    if (Desc->Flags & CM_RESOURCE_MEMORY_LARGE_40) {
                        Length = (ULONGLONG)Desc->u.Memory.Length << 8;
                    } else if (Desc->Flags & CM_RESOURCE_MEMORY_LARGE_48) {
                        Length = (ULONGLONG)Desc->u.Memory.Length << 16;
                    } else if (Desc->Flags & CM_RESOURCE_MEMORY_LARGE_64) {
                        Length = (ULONGLONG)Desc->u.Memory.Length << 32;
                    } else {
                        Length = Desc->u.Memory.Length;
                    }
                } else {
                    Length = Desc->u.Memory.Length;
                }

                pAdapter->Bars[BarIndex].PhysicalAddress = PhysAddr;
                pAdapter->Bars[BarIndex].Length = Length;
                pAdapter->Bars[BarIndex].IsMemory = TRUE;
                pAdapter->Bars[BarIndex].IsPrefetchable =
                    (Desc->Flags & CM_RESOURCE_MEMORY_PREFETCHABLE) != 0;
                pAdapter->Bars[BarIndex].Is64Bit =
                    (Desc->Flags & CM_RESOURCE_MEMORY_BAR) != 0;
                pAdapter->Bars[BarIndex].Mapped = FALSE;
                pAdapter->Bars[BarIndex].KernelAddress = NULL;

                BarIndex++;
            }
            break;

        case CmResourceTypeInterrupt:
            pAdapter->InterruptEnabled = FALSE;
            break;

        default:
            break;
        }
    }

    pAdapter->NumBars = BarIndex;
    return STATUS_SUCCESS;
}

/*
 * ClassifyBars -- determine which BAR is MMIO, VRAM, and doorbell.
 *
 * AMD GPUs present BARs as (by PCI BAR index):
 *   BAR0/1: VRAM aperture (largest, prefetchable, 64-bit)
 *   BAR2/3: Doorbell aperture (medium, prefetchable, 64-bit, ~256MB)
 *   BAR5:   MMIO registers (smallest, non-prefetchable, 32-bit, ~512KB)
 *
 * CM resource list doesn't preserve PCI BAR indices, so we classify
 * by size: VRAM=largest, MMIO=smallest, Doorbell=middle.
 */
static void
ClassifyBars(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    ULONG i;
    ULONGLONG LargestSize = 0;
    ULONGLONG SmallestSize = (ULONGLONG)-1;
    ULONG LargestIdx = 0, SmallestIdx = 0;

    /* Default: all point to 0 */
    pAdapter->MmioBarIndex = 0;
    pAdapter->VramBarIndex = 0;
    pAdapter->DoorbellBarIndex = 0;

    if (pAdapter->NumBars < 2)
        return;

    /* Find largest and smallest memory BARs */
    for (i = 0; i < pAdapter->NumBars; i++) {
        if (!pAdapter->Bars[i].IsMemory || pAdapter->Bars[i].Length == 0)
            continue;
        if (pAdapter->Bars[i].Length > LargestSize) {
            LargestSize = pAdapter->Bars[i].Length;
            LargestIdx = i;
        }
        if (pAdapter->Bars[i].Length < SmallestSize) {
            SmallestSize = pAdapter->Bars[i].Length;
            SmallestIdx = i;
        }
    }

    /*
     * MMIO register BAR is the SMALLEST (BAR5, ~512KB).
     * VRAM is the LARGEST (BAR0, up to full VRAM with ReBAR).
     * Doorbell is the MIDDLE one (BAR2, ~256MB).
     */
    pAdapter->VramBarIndex = LargestIdx;
    pAdapter->MmioBarIndex = SmallestIdx;

    /* Doorbell is the one that's neither largest nor smallest */
    for (i = 0; i < pAdapter->NumBars; i++) {
        if (!pAdapter->Bars[i].IsMemory || pAdapter->Bars[i].Length == 0)
            continue;
        if (i != LargestIdx && i != SmallestIdx) {
            pAdapter->DoorbellBarIndex = i;
            break;
        }
    }

    /* If only 2 BARs, MMIO is the smaller one */
    if (pAdapter->NumBars == 2) {
        pAdapter->MmioBarIndex = SmallestIdx;
        pAdapter->DoorbellBarIndex = LargestIdx;
    }

    KdPrint(("AmdGpuWddm: BAR classification: MMIO=%u (%lluKB) VRAM=%u (%lluMB) Doorbell=%u (%lluMB)\n",
        pAdapter->MmioBarIndex,
        pAdapter->Bars[pAdapter->MmioBarIndex].Length / 1024,
        pAdapter->VramBarIndex,
        pAdapter->Bars[pAdapter->VramBarIndex].Length / (1024 * 1024),
        pAdapter->DoorbellBarIndex,
        pAdapter->Bars[pAdapter->DoorbellBarIndex].Length / (1024 * 1024)));
}

static NTSTATUS
MapMmioBar(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    PVOID MappedAddr;
    ULONG Idx = pAdapter->MmioBarIndex;

    if (pAdapter->NumBars == 0 || pAdapter->Bars[Idx].Length == 0)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    MappedAddr = MmMapIoSpaceEx(
        pAdapter->Bars[Idx].PhysicalAddress,
        (SIZE_T)pAdapter->Bars[Idx].Length,
        PAGE_READWRITE | PAGE_NOCACHE);

    if (MappedAddr == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    pAdapter->Bars[Idx].KernelAddress = MappedAddr;
    pAdapter->Bars[Idx].Mapped = TRUE;

    KdPrint(("AmdGpuWddm: Mapped MMIO BAR[%u] phys=0x%llX len=%lluMB kva=%p\n",
        Idx, pAdapter->Bars[Idx].PhysicalAddress.QuadPart,
        pAdapter->Bars[Idx].Length / (1024 * 1024), MappedAddr));

    return STATUS_SUCCESS;
}

static void
UnmapBars(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    ULONG i;
    for (i = 0; i < pAdapter->NumBars; i++) {
        if (pAdapter->Bars[i].Mapped && pAdapter->Bars[i].KernelAddress != NULL) {
            MmUnmapIoSpace(
                pAdapter->Bars[i].KernelAddress,
                (SIZE_T)pAdapter->Bars[i].Length);
            pAdapter->Bars[i].KernelAddress = NULL;
            pAdapter->Bars[i].Mapped = FALSE;
        }
    }
}

static void
DetectVramSize(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    ULONG MmioIdx = pAdapter->MmioBarIndex;
    ULONG VramIdx = pAdapter->VramBarIndex;
    ULONG MemSizeMB;

    if (!pAdapter->Bars[MmioIdx].Mapped ||
        pAdapter->Bars[MmioIdx].KernelAddress == NULL) {
        /* Fall back to VRAM BAR size as estimate */
        pAdapter->VramSize = pAdapter->Bars[VramIdx].Length;
        pAdapter->VisibleVramSize = pAdapter->VramSize;
        return;
    }

    if (mmRCC_CONFIG_MEMSIZE_BYTE + sizeof(ULONG) >
        pAdapter->Bars[MmioIdx].Length) {
        pAdapter->VramSize = pAdapter->Bars[VramIdx].Length;
        pAdapter->VisibleVramSize = pAdapter->VramSize;
        return;
    }

    MemSizeMB = READ_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[MmioIdx].KernelAddress +
                 mmRCC_CONFIG_MEMSIZE_BYTE));

    KdPrint(("AmdGpuWddm: RCC_CONFIG_MEMSIZE register = 0x%08X (%u MB)\n",
        MemSizeMB, MemSizeMB));

    if (MemSizeMB > 0 && MemSizeMB < 0x100000) {
        /* Register returned valid MB count */
        pAdapter->VramSize = (ULONGLONG)MemSizeMB * 1024ULL * 1024ULL;
    } else {
        /* Register read failed or returned garbage, use VRAM BAR size */
        pAdapter->VramSize = pAdapter->Bars[VramIdx].Length;
    }

    /* Visible VRAM = min(total VRAM, VRAM BAR size) */
    pAdapter->VisibleVramSize =
        (pAdapter->VramSize < pAdapter->Bars[VramIdx].Length)
        ? pAdapter->VramSize
        : pAdapter->Bars[VramIdx].Length;
}

/* PCI config offsets */
#define PCI_CFG_VENDOR_ID       0x00
#define PCI_CFG_DEVICE_ID       0x02
#define PCI_CFG_REVISION_ID     0x08
#define PCI_CFG_SUBSYS_VENDOR   0x2C
#define PCI_CFG_SUBSYS_ID       0x2E

/* Helper to compute bytes per pixel from D3DDDIFORMAT */
static ULONG
FormatBytesPerPixel(D3DDDIFORMAT Format)
{
    switch (Format) {
    case D3DDDIFMT_A8R8G8B8:
    case D3DDDIFMT_X8R8G8B8:
        return 4;
    case D3DDDIFMT_R5G6B5:
        return 2;
    default:
        return 4;  /* Default to 32bpp */
    }
}

/* ======================================================================
 * StartDevice -- acquire POST display, enumerate BARs
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuStartDevice(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN_PDXGK_START_INFO     DxgkStartInfo,
    IN_PDXGKRNL_INTERFACE   DxgkInterface,
    OUT_PULONG              NumberOfVideoPresentSources,
    OUT_PULONG              NumberOfChildren
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;
    NTSTATUS Status;

    KdPrint(("AmdGpuWddm: StartDevice called\n"));

    pAdapter->DxgkHandle = DxgkInterface->DeviceHandle;
    pAdapter->DxgkStartInfo = *DxgkStartInfo;
    RtlCopyMemory(&pAdapter->DxgkInterface, DxgkInterface,
                   sizeof(DXGKRNL_INTERFACE));

    /* ---- Read PCI config ---- */
    {
        ULONG BytesRead = 0;
        USHORT VendorId = 0, DeviceId = 0;
        USHORT SubsysVendor = 0, SubsysId = 0;
        UCHAR  RevisionId = 0;

        pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &VendorId, PCI_CFG_VENDOR_ID,
            sizeof(VendorId), &BytesRead);
        pAdapter->VendorId = VendorId;

        pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &DeviceId, PCI_CFG_DEVICE_ID,
            sizeof(DeviceId), &BytesRead);
        pAdapter->DeviceId = DeviceId;

        pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &RevisionId, PCI_CFG_REVISION_ID,
            sizeof(RevisionId), &BytesRead);
        pAdapter->RevisionId = RevisionId;

        pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &SubsysVendor, PCI_CFG_SUBSYS_VENDOR,
            sizeof(SubsysVendor), &BytesRead);
        pAdapter->SubsystemVendorId = SubsysVendor;

        pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &SubsysId, PCI_CFG_SUBSYS_ID,
            sizeof(SubsysId), &BytesRead);
        pAdapter->SubsystemId = SubsysId;
    }

    /* ---- Enumerate PCI BARs and classify ---- */
    Status = EnumerateBars(pAdapter);
    if (!NT_SUCCESS(Status))
        pAdapter->NumBars = 0;

    ClassifyBars(pAdapter);

    /* ---- Acquire POST display ownership ---- */
    {
        DXGK_DISPLAY_INFORMATION PostInfo;
        RtlZeroMemory(&PostInfo, sizeof(PostInfo));

        Status = pAdapter->DxgkInterface.DxgkCbAcquirePostDisplayOwnership(
            pAdapter->DxgkInterface.DeviceHandle, &PostInfo);

        if (NT_SUCCESS(Status) && PostInfo.Width > 0 && PostInfo.Height > 0) {
            pAdapter->PostDisplay.Acquired = TRUE;
            pAdapter->PostDisplay.Width = PostInfo.Width;
            pAdapter->PostDisplay.Height = PostInfo.Height;
            pAdapter->PostDisplay.Pitch = PostInfo.Pitch;
            pAdapter->PostDisplay.ColorFormat = PostInfo.ColorFormat;
            pAdapter->PostDisplay.FramebufferPhysAddr = PostInfo.PhysicAddress;
            pAdapter->PostDisplay.FramebufferSize =
                (ULONGLONG)PostInfo.Pitch * PostInfo.Height;

            /* Map POST framebuffer into kernel VA for CPU blits */
            pAdapter->PostDisplay.FramebufferKernelVa = MmMapIoSpaceEx(
                PostInfo.PhysicAddress,
                (SIZE_T)pAdapter->PostDisplay.FramebufferSize,
                PAGE_READWRITE | PAGE_NOCACHE);

            KdPrint(("AmdGpuWddm: POST display acquired: %ux%u pitch=%u fmt=%d phys=0x%llX kva=%p\n",
                PostInfo.Width, PostInfo.Height, PostInfo.Pitch,
                PostInfo.ColorFormat, PostInfo.PhysicAddress.QuadPart,
                pAdapter->PostDisplay.FramebufferKernelVa));
        } else {
            /* POST acquisition failed -- provide a fallback so driver still loads */
            KdPrint(("AmdGpuWddm: POST display NOT acquired (status=0x%08X, w=%u h=%u)\n",
                Status, PostInfo.Width, PostInfo.Height));
            pAdapter->PostDisplay.Acquired = FALSE;
            pAdapter->PostDisplay.Width = 1024;
            pAdapter->PostDisplay.Height = 768;
            pAdapter->PostDisplay.Pitch = 1024 * 4;
            pAdapter->PostDisplay.ColorFormat = D3DDDIFMT_X8R8G8B8;
        }
    }

    /*
     * Detect headless (compute-only) GPU: no POST display acquired means
     * no physical display output. This is normal for MI100, MI200, MI300,
     * and any GPU that isn't driving a monitor.
     */
    pAdapter->Headless = !pAdapter->PostDisplay.Acquired;

    /*
     * Always report 1 video present source and 1 child device.
     * WDDM Display class requires at least 1 child. For headless GPUs,
     * QueryChildStatus will report the child as disconnected so no
     * display path is established, but the driver still loads and the
     * escape channel works.
     */
    *NumberOfVideoPresentSources = 1;
    *NumberOfChildren = 1;

    pAdapter->Started = TRUE;

    KdPrint(("AmdGpuWddm: StartDevice succeeded, VendorId=0x%04X DeviceId=0x%04X NumBars=%u Headless=%u\n",
        pAdapter->VendorId, pAdapter->DeviceId, pAdapter->NumBars, pAdapter->Headless));

    /* Write StartDevice status to registry */
    {
        UNICODE_STRING KeyPath;
        OBJECT_ATTRIBUTES ObjAttrs;
        HANDLE hKey;
        NTSTATUS s;
        UNICODE_STRING ValName;
        ULONG Val;

        RtlInitUnicodeString(&KeyPath,
            L"\\Registry\\Machine\\SOFTWARE\\AmdGpuWddm");
        InitializeObjectAttributes(&ObjAttrs, &KeyPath,
            OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, NULL, NULL);
        s = ZwOpenKey(&hKey, KEY_SET_VALUE, &ObjAttrs);
        if (NT_SUCCESS(s)) {
            RtlInitUnicodeString(&ValName, L"StartDevice");
            Val = 1;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));

            RtlInitUnicodeString(&ValName, L"PostWidth");
            Val = pAdapter->PostDisplay.Width;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));

            RtlInitUnicodeString(&ValName, L"PostHeight");
            Val = pAdapter->PostDisplay.Height;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));

            RtlInitUnicodeString(&ValName, L"PostAcquired");
            Val = pAdapter->PostDisplay.Acquired ? 1 : 0;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));

            RtlInitUnicodeString(&ValName, L"Headless");
            Val = pAdapter->Headless ? 1 : 0;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));

            ZwClose(hKey);
        }
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * MapMmioIfNeeded -- lazy MMIO BAR mapping
 * ====================================================================== */

NTSTATUS
AmdGpuMapMmioIfNeeded(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    NTSTATUS Status;
    ULONG Idx = pAdapter->MmioBarIndex;

    if (pAdapter->Bars[Idx].Mapped)
        return STATUS_SUCCESS;

    if (pAdapter->NumBars == 0)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    Status = MapMmioBar(pAdapter);
    if (!NT_SUCCESS(Status))
        return Status;

    DetectVramSize(pAdapter);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * StopDevice
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuStopDevice(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

    pAdapter->Started = FALSE;
    pAdapter->IhRing.Configured = FALSE;

    /* Cleanup compute state (Phase 2) */
    AmdGpuComputeCleanup(pAdapter);

    /* Release registered events */
    {
        ULONG i;
        KIRQL OldIrql;

        KeAcquireSpinLock(&pAdapter->EventsLock, &OldIrql);
        for (i = 0; i < AMDGPU_MAX_EVENTS; i++) {
            if (pAdapter->Events[i].InUse) {
                ObDereferenceObject(pAdapter->Events[i].Event);
                pAdapter->Events[i].Event = NULL;
                pAdapter->Events[i].InUse = FALSE;
            }
        }
        KeReleaseSpinLock(&pAdapter->EventsLock, OldIrql);
    }

    /* Unmap POST framebuffer */
    if (pAdapter->PostDisplay.FramebufferKernelVa != NULL) {
        MmUnmapIoSpace(
            pAdapter->PostDisplay.FramebufferKernelVa,
            (SIZE_T)pAdapter->PostDisplay.FramebufferSize);
        pAdapter->PostDisplay.FramebufferKernelVa = NULL;
    }

    UnmapBars(pAdapter);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * RemoveDevice
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuRemoveDevice(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    if (MiniportDeviceContext != NULL)
        ExFreePoolWithTag(MiniportDeviceContext, AMDGPU_POOL_TAG);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * CreateDevice / DestroyDevice
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuCreateDevice(
    IN_CONST_HANDLE                 hAdapter,
    INOUT_PDXGKARG_CREATEDEVICE     pCreateDevice
    )
{
    AMDGPU_DEVICE_CONTEXT *pDevCtx;

    pDevCtx = (AMDGPU_DEVICE_CONTEXT *)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(AMDGPU_DEVICE_CONTEXT), AMDGPU_POOL_TAG);
    if (pDevCtx == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(pDevCtx, sizeof(*pDevCtx));
    pDevCtx->AdapterContext = (PVOID)hAdapter;

    pCreateDevice->hDevice = pDevCtx;
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuDestroyDevice(
    IN_CONST_HANDLE     hDevice
    )
{
    if (hDevice != NULL)
        ExFreePoolWithTag((PVOID)hDevice, AMDGPU_POOL_TAG);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * CreateContext / DestroyContext -- with DMA private data for Present
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuCreateContext(
    IN_CONST_HANDLE                 hDevice,
    INOUT_PDXGKARG_CREATECONTEXT    pCreateContext
    )
{
    AMDGPU_CONTEXT *pCtx;

    pCtx = (AMDGPU_CONTEXT *)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(AMDGPU_CONTEXT), AMDGPU_POOL_TAG);
    if (pCtx == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(pCtx, sizeof(*pCtx));
    pCtx->DeviceContext = (PVOID)hDevice;
    pCtx->NodeOrdinal = pCreateContext->NodeOrdinal;
    pCtx->EngineAffinity = pCreateContext->EngineAffinity;

    pCreateContext->hContext = pCtx;
    pCreateContext->ContextInfo.DmaBufferSize = 4096;
    pCreateContext->ContextInfo.DmaBufferSegmentSet = 0;     /* System memory */
    pCreateContext->ContextInfo.DmaBufferPrivateDataSize = sizeof(AMDGPU_DMA_CMD);
    pCreateContext->ContextInfo.AllocationListSize = 16;
    pCreateContext->ContextInfo.PatchLocationListSize = 16;

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuDestroyContext(
    IN_CONST_HANDLE     hContext
    )
{
    if (hContext != NULL)
        ExFreePoolWithTag((PVOID)hContext, AMDGPU_POOL_TAG);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * CreateProcess / DestroyProcess
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuCreateProcess(
    IN_CONST_HANDLE                 hAdapter,
    INOUT_PDXGKARG_CREATEPROCESS    pArgs
    )
{
    AMDGPU_PROCESS_CONTEXT *pProc;

    UNREFERENCED_PARAMETER(hAdapter);

    pProc = (AMDGPU_PROCESS_CONTEXT *)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(AMDGPU_PROCESS_CONTEXT), AMDGPU_POOL_TAG);
    if (pProc == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(pProc, sizeof(*pProc));
    pArgs->hKmdProcess = pProc;
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuDestroyProcess(
    IN_CONST_HANDLE     hAdapter
    )
{
    if (hAdapter != NULL)
        ExFreePoolWithTag((PVOID)hAdapter, AMDGPU_POOL_TAG);
    return STATUS_SUCCESS;
}
