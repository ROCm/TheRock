/*
 * ddi_device.c - Device lifecycle DDIs for MCDM miniport
 *
 * Implements AddDevice, StartDevice, StopDevice, RemoveDevice,
 * CreateDevice, DestroyDevice, CreateContext, DestroyContext,
 * CreateProcess, DestroyProcess.
 */

#include "amdgpu_mcdm.h"

/* ======================================================================
 * Per-device-handle and per-context tracking
 *
 * dxgkrnl creates "device handles" for each D3D device opened by
 * userspace (via CreateDevice), and "contexts" within those devices.
 * For our fake MCDM, these are trivial allocations.
 * ====================================================================== */

typedef struct _AMDGPU_DEVICE_CONTEXT {
    PVOID                   AdapterContext;  /* Back-pointer to AMDGPU_ADAPTER */
    HANDLE                  hDevice;
} AMDGPU_DEVICE_CONTEXT;

typedef struct _AMDGPU_CONTEXT {
    PVOID                   DeviceContext;   /* Back-pointer to AMDGPU_DEVICE_CONTEXT */
    UINT                    NodeOrdinal;
    UINT                    EngineAffinity;
} AMDGPU_CONTEXT;

typedef struct _AMDGPU_PROCESS_CONTEXT {
    HANDLE                  hKmdProcess;
} AMDGPU_PROCESS_CONTEXT;

/* ======================================================================
 * AddDevice — allocate per-adapter context
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

    pAdapter = (AMDGPU_ADAPTER *)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(AMDGPU_ADAPTER), AMDGPU_POOL_TAG);
    if (pAdapter == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(pAdapter, sizeof(*pAdapter));
    KeInitializeSpinLock(&pAdapter->DmaAllocsLock);
    KeInitializeSpinLock(&pAdapter->EventsLock);

    *MiniportDeviceContext = pAdapter;
    return STATUS_SUCCESS;
}

/* ======================================================================
 * StartDevice — store DXGK handles, enumerate PCI BARs, map BAR0
 *
 * v0.1: PCI config read for device identification.
 * v0.2: Full PCI BAR enumeration via DxgkCbGetDeviceInformation,
 *        BAR0 mapping for MMIO register access, VRAM size detection.
 * ====================================================================== */

/* Register index for VRAM size (NBIO v7.x, byte offset = index * 4) */
#define mmRCC_CONFIG_MEMSIZE        0xDE3
#define mmRCC_CONFIG_MEMSIZE_BYTE   (mmRCC_CONFIG_MEMSIZE * 4)  /* 0x378C */

/*
 * EnumerateBars — walk translated resource list from PnP manager.
 *
 * DxgkCbGetDeviceInformation returns the PCI resources assigned to us.
 * We iterate through CM_PARTIAL_RESOURCE_DESCRIPTOR entries looking for
 * CmResourceTypeMemory (PCI memory BARs) and CmResourceTypeInterrupt.
 *
 * PCI BARs appear in order (BAR0, BAR2, BAR4 for 64-bit BARs) with
 * odd BARs consumed as the upper 32 bits of their 64-bit predecessor.
 */
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

    /* Walk the first full resource descriptor (only one for PCI) */
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

                /*
                 * For CmResourceTypeMemoryLarge, length is encoded
                 * differently based on flags. For standard memory
                 * resources, Length is in the descriptor directly.
                 */
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
            /* Track interrupt info for v0.4 MSI-X support */
            pAdapter->InterruptEnabled = FALSE;
            break;

        default:
            /* Skip I/O ports and other resource types */
            break;
        }
    }

    pAdapter->NumBars = BarIndex;
    return STATUS_SUCCESS;
}

/*
 * MapBar0 — map the first BAR (MMIO registers) into kernel virtual space.
 *
 * BAR0 is typically the MMIO register aperture (256KB-512KB).
 * We use MmMapIoSpaceEx with PAGE_READWRITE | PAGE_NOCACHE.
 */
static NTSTATUS
MapBar0(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    PVOID MappedAddr;

    if (pAdapter->NumBars == 0)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    if (pAdapter->Bars[0].Length == 0)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    MappedAddr = MmMapIoSpaceEx(
        pAdapter->Bars[0].PhysicalAddress,
        (SIZE_T)pAdapter->Bars[0].Length,
        PAGE_READWRITE | PAGE_NOCACHE);

    if (MappedAddr == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    pAdapter->Bars[0].KernelAddress = MappedAddr;
    pAdapter->Bars[0].Mapped = TRUE;

    return STATUS_SUCCESS;
}

/*
 * UnmapBars — unmap all kernel-mapped BARs. Called during StopDevice.
 */
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

/*
 * DetectVramSize — read VRAM configuration from MMIO registers.
 *
 * Uses mmRCC_CONFIG_MEMSIZE (NBIO register at byte offset 0x378C)
 * which returns VRAM size in megabytes.
 */
static void
DetectVramSize(
    _Inout_ AMDGPU_ADAPTER *pAdapter
    )
{
    ULONG MemSizeMB;

    if (!pAdapter->Bars[0].Mapped || pAdapter->Bars[0].KernelAddress == NULL) {
        pAdapter->VramSize = 0;
        pAdapter->VisibleVramSize = 0;
        return;
    }

    /* Ensure the register offset is within BAR0 bounds */
    if (mmRCC_CONFIG_MEMSIZE_BYTE + sizeof(ULONG) > pAdapter->Bars[0].Length) {
        pAdapter->VramSize = 0;
        pAdapter->VisibleVramSize = 0;
        return;
    }

    MemSizeMB = READ_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress + mmRCC_CONFIG_MEMSIZE_BYTE));

    pAdapter->VramSize = (ULONGLONG)MemSizeMB * 1024ULL * 1024ULL;

    /*
     * Visible VRAM = min(VRAM size, BAR2 size).
     * BAR2 is typically the VRAM aperture. With ReBAR enabled,
     * it can be the full VRAM size. Without ReBAR, usually 256MB.
     */
    if (pAdapter->NumBars >= 3 && pAdapter->Bars[2].Length > 0) {
        pAdapter->VisibleVramSize =
            (pAdapter->VramSize < pAdapter->Bars[2].Length)
            ? pAdapter->VramSize
            : pAdapter->Bars[2].Length;
    } else {
        /* Conservative: assume 256MB visible without BAR2 info */
        pAdapter->VisibleVramSize =
            (pAdapter->VramSize < 256ULL * 1024 * 1024)
            ? pAdapter->VramSize
            : 256ULL * 1024 * 1024;
    }
}

/* PCI config offsets */
#define PCI_CFG_VENDOR_ID       0x00
#define PCI_CFG_DEVICE_ID       0x02
#define PCI_CFG_REVISION_ID     0x08
#define PCI_CFG_SUBSYS_VENDOR   0x2C
#define PCI_CFG_SUBSYS_ID       0x2E

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

    /* Save DXGK handles for later use (DxgkCbReadDeviceSpace, etc.) */
    pAdapter->DxgkHandle = DxgkInterface->DeviceHandle;
    pAdapter->DxgkStartInfo = *DxgkStartInfo;
    RtlCopyMemory(&pAdapter->DxgkInterface, DxgkInterface,
                   sizeof(DXGKRNL_INTERFACE));

    /* ---- Phase 1: Read PCI config space for device identification ---- */
    {
        ULONG BytesRead = 0;
        USHORT VendorId = 0, DeviceId = 0;
        USHORT SubsysVendor = 0, SubsysId = 0;
        UCHAR  RevisionId = 0;

        Status = pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &VendorId, PCI_CFG_VENDOR_ID,
            sizeof(VendorId), &BytesRead);
        if (NT_SUCCESS(Status))
            pAdapter->VendorId = VendorId;

        Status = pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &DeviceId, PCI_CFG_DEVICE_ID,
            sizeof(DeviceId), &BytesRead);
        if (NT_SUCCESS(Status))
            pAdapter->DeviceId = DeviceId;

        Status = pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &RevisionId, PCI_CFG_REVISION_ID,
            sizeof(RevisionId), &BytesRead);
        if (NT_SUCCESS(Status))
            pAdapter->RevisionId = RevisionId;

        Status = pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &SubsysVendor, PCI_CFG_SUBSYS_VENDOR,
            sizeof(SubsysVendor), &BytesRead);
        if (NT_SUCCESS(Status))
            pAdapter->SubsystemVendorId = SubsysVendor;

        Status = pAdapter->DxgkInterface.DxgkCbReadDeviceSpace(
            pAdapter->DxgkInterface.DeviceHandle,
            DXGK_WHICHSPACE_CONFIG, &SubsysId, PCI_CFG_SUBSYS_ID,
            sizeof(SubsysId), &BytesRead);
        if (NT_SUCCESS(Status))
            pAdapter->SubsystemId = SubsysId;
    }

    /* ---- Phase 2: Enumerate PCI BARs from translated resources ---- */
    Status = EnumerateBars(pAdapter);
    if (!NT_SUCCESS(Status)) {
        /* BAR enumeration failed — driver cannot operate without BARs.
         * Log but continue; register access will fail gracefully. */
        pAdapter->NumBars = 0;
    }

    /* ---- Phase 3: Map BAR0 for MMIO register access ---- */
    if (pAdapter->NumBars > 0) {
        Status = MapBar0(pAdapter);
        if (!NT_SUCCESS(Status)) {
            /* BAR0 mapping failed — register escape commands will return
             * STATUS_INVALID_PARAMETER (BAR not mapped). */
            pAdapter->Bars[0].Mapped = FALSE;
        }
    }

    /* ---- Phase 4: Detect VRAM size from MMIO registers ---- */
    DetectVramSize(pAdapter);

    /*
     * Compute-only: no video present sources and no child devices.
     * dxgkrnl checks these for display enumeration.
     */
    *NumberOfVideoPresentSources = 0;
    *NumberOfChildren = 0;

    pAdapter->Started = TRUE;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * StopDevice — release resources
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuStopDevice(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

    pAdapter->Started = FALSE;

    /* Disable IH ring */
    pAdapter->IhRing.Configured = FALSE;

    /* Release all registered event references */
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

    /* Unmap all kernel-mapped BARs */
    UnmapBars(pAdapter);

    return STATUS_SUCCESS;
}

/* ======================================================================
 * RemoveDevice — free adapter context
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
 * CreateDevice / DestroyDevice — per-D3D-device tracking
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
 * CreateContext / DestroyContext — per-context tracking
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
    pCreateContext->ContextInfo.DmaBufferSize = 4096;        /* Minimal */
    pCreateContext->ContextInfo.DmaBufferSegmentSet = 0;     /* System memory */
    pCreateContext->ContextInfo.DmaBufferPrivateDataSize = 0;
    pCreateContext->ContextInfo.AllocationListSize = 0;
    pCreateContext->ContextInfo.PatchLocationListSize = 0;

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
 * CreateProcess / DestroyProcess — WDDM 2.0+ per-process tracking
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
    /* hAdapter is actually the hKmdProcess handle from CreateProcess */
    if (hAdapter != NULL)
        ExFreePoolWithTag((PVOID)hAdapter, AMDGPU_POOL_TAG);
    return STATUS_SUCCESS;
}
