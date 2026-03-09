/*
 * ddi_escape.c - DxgkDdiEscape handler for ROCm Display Driver
 *
 * Identical to MCDM escape handler -- routes commands between Python
 * userspace driver and GPU hardware.
 */

#include "amdgpu_wddm.h"

#define IH_ENTRY_SIZE   32

static NTSTATUS
EscapeGetInfo(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_GET_INFO_DATA *pData
    )
{
    ULONG i;

    AmdGpuMapMmioIfNeeded(pAdapter);

    pData->VendorId = pAdapter->VendorId;
    pData->DeviceId = pAdapter->DeviceId;
    pData->SubsystemVendorId = pAdapter->SubsystemVendorId;
    pData->SubsystemId = pAdapter->SubsystemId;
    pData->RevisionId = pAdapter->RevisionId;
    pData->NumBars = pAdapter->NumBars;

    for (i = 0; i < AMDGPU_MAX_BARS && i < pAdapter->NumBars; i++) {
        pData->Bars[i].PhysicalAddress = pAdapter->Bars[i].PhysicalAddress;
        pData->Bars[i].Length = pAdapter->Bars[i].Length;
        pData->Bars[i].IsMemory = pAdapter->Bars[i].IsMemory;
        pData->Bars[i].Is64Bit = pAdapter->Bars[i].Is64Bit;
        pData->Bars[i].IsPrefetchable = pAdapter->Bars[i].IsPrefetchable;
    }

    pData->VramSizeBytes = pAdapter->VramSize;
    pData->VisibleVramSizeBytes = pAdapter->VisibleVramSize;
    pData->MmioBarIndex = pAdapter->MmioBarIndex;
    pData->VramBarIndex = pAdapter->VramBarIndex;
    pData->Headless = pAdapter->Headless;

    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeReadReg32(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_REG32_DATA *pData
    )
{
    ULONG BarIndex = pData->BarIndex;
    ULONG Offset = pData->Offset;

    /* BarIndex 0 from userspace means "MMIO register BAR" */
    if (BarIndex == 0)
        BarIndex = pAdapter->MmioBarIndex;

    if (!pAdapter->Bars[BarIndex].Mapped)
        AmdGpuMapMmioIfNeeded(pAdapter);

    if (BarIndex >= AMDGPU_MAX_BARS || !pAdapter->Bars[BarIndex].Mapped)
        return STATUS_INVALID_PARAMETER;

    if ((ULONGLONG)Offset + sizeof(ULONG) > pAdapter->Bars[BarIndex].Length)
        return STATUS_INVALID_PARAMETER;

    pData->Value = READ_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[BarIndex].KernelAddress + Offset));

    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeWriteReg32(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_REG32_DATA *pData
    )
{
    ULONG BarIndex = pData->BarIndex;
    ULONG Offset = pData->Offset;

    /* BarIndex 0 from userspace means "MMIO register BAR" */
    if (BarIndex == 0)
        BarIndex = pAdapter->MmioBarIndex;

    if (!pAdapter->Bars[BarIndex].Mapped)
        AmdGpuMapMmioIfNeeded(pAdapter);

    if (BarIndex >= AMDGPU_MAX_BARS || !pAdapter->Bars[BarIndex].Mapped)
        return STATUS_INVALID_PARAMETER;

    if ((ULONGLONG)Offset + sizeof(ULONG) > pAdapter->Bars[BarIndex].Length)
        return STATUS_INVALID_PARAMETER;

    WRITE_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[BarIndex].KernelAddress + Offset),
        pData->Value);

    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeMapBar(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_MAP_BAR_DATA *pData
    )
{
    ULONG BarIndex = pData->BarIndex;
    ULONGLONG Offset = pData->Offset;
    ULONGLONG Length = pData->Length;
    PHYSICAL_ADDRESS PhysAddr;
    PVOID KernelVa;
    PMDL Mdl;
    PVOID UserVa;

    if (BarIndex >= AMDGPU_MAX_BARS || BarIndex >= pAdapter->NumBars)
        return STATUS_INVALID_PARAMETER;

    if (!pAdapter->Bars[BarIndex].IsMemory)
        return STATUS_INVALID_PARAMETER;

    if (Length == 0)
        Length = pAdapter->Bars[BarIndex].Length - Offset;

    if (Offset + Length > pAdapter->Bars[BarIndex].Length)
        return STATUS_INVALID_PARAMETER;

    /* Limit single mapping to 16MB to avoid excessive resource usage */
    if (Length > 16 * 1024 * 1024)
        return STATUS_INVALID_PARAMETER;

    PhysAddr.QuadPart =
        pAdapter->Bars[BarIndex].PhysicalAddress.QuadPart + Offset;

    /*
     * Map the BAR region into kernel space first using MmMapIoSpace.
     * This is the correct way to map device memory (BAR space).
     * Then create an MDL from the kernel mapping and map to userspace.
     */
    KernelVa = MmMapIoSpace(PhysAddr, (SIZE_T)Length, MmNonCached);
    if (KernelVa == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    Mdl = IoAllocateMdl(KernelVa, (ULONG)Length, FALSE, FALSE, NULL);
    if (Mdl == NULL) {
        MmUnmapIoSpace(KernelVa, (SIZE_T)Length);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    MmBuildMdlForNonPagedPool(Mdl);

    __try {
        UserVa = MmMapLockedPagesSpecifyCache(
            Mdl, UserMode, MmNonCached,
            NULL, FALSE, NormalPagePriority);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        IoFreeMdl(Mdl);
        MmUnmapIoSpace(KernelVa, (SIZE_T)Length);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    if (UserVa == NULL) {
        IoFreeMdl(Mdl);
        MmUnmapIoSpace(KernelVa, (SIZE_T)Length);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    pData->MappedAddress = UserVa;
    /* Pack both handles: MDL in MappingHandle, KernelVa stored after unmap */
    pData->MappingHandle = (PVOID)Mdl;
    /* Store KernelVa and Length in adapter tracking for cleanup.
     * For now, use a simple approach: store KernelVa in the MDL's
     * MappedSystemVa field (it's not used for IoSpace MDLs). */
    Mdl->MappedSystemVa = KernelVa;
    Mdl->ByteCount = (ULONG)Length;
    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeUnmapBar(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_MAP_BAR_DATA *pData
    )
{
    PMDL Mdl;
    PVOID KernelVa;
    ULONG Length;

    UNREFERENCED_PARAMETER(pAdapter);

    if (pData->MappingHandle == NULL || pData->MappedAddress == NULL)
        return STATUS_INVALID_PARAMETER;

    Mdl = (PMDL)pData->MappingHandle;
    KernelVa = Mdl->MappedSystemVa;
    Length = Mdl->ByteCount;

    MmUnmapLockedPages(pData->MappedAddress, Mdl);
    IoFreeMdl(Mdl);

    /* Unmap the kernel IoSpace mapping */
    if (KernelVa != NULL)
        MmUnmapIoSpace(KernelVa, (SIZE_T)Length);

    pData->MappedAddress = NULL;
    pData->MappingHandle = NULL;
    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeAllocDma(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_ALLOC_DMA_DATA *pData
    )
{
    ULONGLONG Size = pData->Size;
    PHYSICAL_ADDRESS LowestAddr, HighestAddr, BoundaryAddr;
    PVOID KernelVa;
    PHYSICAL_ADDRESS PhysAddr;
    PMDL Mdl;
    PVOID UserVa;
    ULONG i;
    KIRQL OldIrql;

    if (Size == 0 || Size > 64 * 1024 * 1024)
        return STATUS_INVALID_PARAMETER;

    Size = (Size + PAGE_SIZE - 1) & ~((ULONGLONG)PAGE_SIZE - 1);

    LowestAddr.QuadPart = 0;
    HighestAddr.QuadPart = (LONGLONG)-1;
    BoundaryAddr.QuadPart = 0;

    KernelVa = MmAllocateContiguousMemorySpecifyCache(
        (SIZE_T)Size, LowestAddr, HighestAddr, BoundaryAddr,
        MmNonCached);
    if (KernelVa == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(KernelVa, (SIZE_T)Size);

    PhysAddr = MmGetPhysicalAddress(KernelVa);

    Mdl = IoAllocateMdl(KernelVa, (ULONG)Size, FALSE, FALSE, NULL);
    if (Mdl == NULL) {
        MmFreeContiguousMemory(KernelVa);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    MmBuildMdlForNonPagedPool(Mdl);

    __try {
        UserVa = MmMapLockedPagesSpecifyCache(
            Mdl, UserMode, MmNonCached,
            NULL, FALSE, NormalPagePriority);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        IoFreeMdl(Mdl);
        MmFreeContiguousMemory(KernelVa);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    if (UserVa == NULL) {
        IoFreeMdl(Mdl);
        MmFreeContiguousMemory(KernelVa);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    KeAcquireSpinLock(&pAdapter->DmaAllocsLock, &OldIrql);
    for (i = 0; i < AMDGPU_MAX_DMA_ALLOCS; i++) {
        if (!pAdapter->DmaAllocs[i].InUse) {
            pAdapter->DmaAllocs[i].KernelVa = KernelVa;
            pAdapter->DmaAllocs[i].BusAddress = PhysAddr;
            pAdapter->DmaAllocs[i].Size = Size;
            pAdapter->DmaAllocs[i].Mdl = Mdl;
            pAdapter->DmaAllocs[i].UserVa = UserVa;
            pAdapter->DmaAllocs[i].InUse = TRUE;
            break;
        }
    }
    KeReleaseSpinLock(&pAdapter->DmaAllocsLock, OldIrql);

    if (i == AMDGPU_MAX_DMA_ALLOCS) {
        MmUnmapLockedPages(UserVa, Mdl);
        IoFreeMdl(Mdl);
        MmFreeContiguousMemory(KernelVa);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    pData->CpuAddress = UserVa;
    pData->BusAddress = (ULONGLONG)PhysAddr.QuadPart;
    pData->AllocationHandle = (PVOID)(ULONG_PTR)i;
    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeFreeDma(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_ALLOC_DMA_DATA *pData
    )
{
    ULONG Index = (ULONG)(ULONG_PTR)pData->AllocationHandle;
    KIRQL OldIrql;

    if (Index >= AMDGPU_MAX_DMA_ALLOCS)
        return STATUS_INVALID_PARAMETER;

    KeAcquireSpinLock(&pAdapter->DmaAllocsLock, &OldIrql);

    if (!pAdapter->DmaAllocs[Index].InUse) {
        KeReleaseSpinLock(&pAdapter->DmaAllocsLock, OldIrql);
        return STATUS_INVALID_PARAMETER;
    }

    MmUnmapLockedPages(
        pAdapter->DmaAllocs[Index].UserVa,
        pAdapter->DmaAllocs[Index].Mdl);
    IoFreeMdl(pAdapter->DmaAllocs[Index].Mdl);
    MmFreeContiguousMemory(pAdapter->DmaAllocs[Index].KernelVa);

    RtlZeroMemory(&pAdapter->DmaAllocs[Index], sizeof(AMDGPU_DMA_ALLOC));

    KeReleaseSpinLock(&pAdapter->DmaAllocsLock, OldIrql);
    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeMapVram(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_MAP_VRAM_DATA *pData
    )
{
    AMDGPU_ESCAPE_MAP_BAR_DATA BarData;
    NTSTATUS Status;
    ULONG VramBarIndex = 0;

    {
        ULONG i;
        ULONGLONG MaxLen = 0;
        for (i = 0; i < pAdapter->NumBars; i++) {
            if (pAdapter->Bars[i].IsMemory &&
                pAdapter->Bars[i].Length > MaxLen) {
                MaxLen = pAdapter->Bars[i].Length;
                VramBarIndex = i;
            }
        }
    }

    if (VramBarIndex == 0 && pAdapter->NumBars < 2)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    RtlZeroMemory(&BarData, sizeof(BarData));
    BarData.BarIndex = VramBarIndex;
    BarData.Offset = pData->Offset;
    BarData.Length = pData->Length;

    Status = EscapeMapBar(pAdapter, &BarData);
    if (NT_SUCCESS(Status)) {
        pData->MappedAddress = BarData.MappedAddress;
        pData->MappingHandle = BarData.MappingHandle;
    }
    return Status;
}

static NTSTATUS
EscapeRegisterEvent(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_REGISTER_EVENT_DATA *pData
    )
{
    PKEVENT Event;
    NTSTATUS Status;
    KIRQL OldIrql;
    ULONG i;

    if (pData->EventHandle == NULL)
        return STATUS_INVALID_PARAMETER;

    Status = ObReferenceObjectByHandle(
        pData->EventHandle,
        EVENT_MODIFY_STATE,
        *ExEventObjectType,
        UserMode,
        (PVOID *)&Event,
        NULL);
    if (!NT_SUCCESS(Status))
        return Status;

    KeAcquireSpinLock(&pAdapter->EventsLock, &OldIrql);

    for (i = 0; i < AMDGPU_MAX_EVENTS; i++) {
        if (!pAdapter->Events[i].InUse) {
            pAdapter->Events[i].Event = Event;
            pAdapter->Events[i].SourceId = pData->InterruptSource;
            pAdapter->Events[i].InUse = TRUE;
            pData->RegistrationId = i;
            KeReleaseSpinLock(&pAdapter->EventsLock, OldIrql);
            return STATUS_SUCCESS;
        }
    }

    KeReleaseSpinLock(&pAdapter->EventsLock, OldIrql);
    ObDereferenceObject(Event);
    return STATUS_INSUFFICIENT_RESOURCES;
}

static NTSTATUS
EscapeEnableMsi(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_ENABLE_MSI_DATA *pData
    )
{
    ULONG Index;

    Index = (ULONG)(ULONG_PTR)pData->IhRingDmaHandle;
    if (Index >= AMDGPU_MAX_DMA_ALLOCS)
        return STATUS_INVALID_PARAMETER;

    if (!pAdapter->DmaAllocs[Index].InUse)
        return STATUS_INVALID_PARAMETER;

    if (pData->IhRingSize < IH_ENTRY_SIZE ||
        (pData->IhRingSize & (pData->IhRingSize - 1)) != 0)
        return STATUS_INVALID_PARAMETER;

    if (pData->IhRingSize > pAdapter->DmaAllocs[Index].Size)
        return STATUS_INVALID_PARAMETER;

    if (pAdapter->Bars[pAdapter->MmioBarIndex].Length == 0 ||
        !pAdapter->Bars[pAdapter->MmioBarIndex].Mapped)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    if (pData->IhRptrRegOffset + sizeof(ULONG) >
        pAdapter->Bars[pAdapter->MmioBarIndex].Length ||
        pData->IhWptrRegOffset + sizeof(ULONG) >
        pAdapter->Bars[pAdapter->MmioBarIndex].Length)
        return STATUS_INVALID_PARAMETER;

    pAdapter->IhRing.RingBuffer = pAdapter->DmaAllocs[Index].KernelVa;
    pAdapter->IhRing.RingSize = pData->IhRingSize;
    pAdapter->IhRing.RingMask = pData->IhRingSize - 1;
    pAdapter->IhRing.RptrRegOffset = pData->IhRptrRegOffset;
    pAdapter->IhRing.WptrRegOffset = pData->IhWptrRegOffset;
    pAdapter->IhRing.Rptr = 0;

    KeMemoryBarrier();
    pAdapter->IhRing.Configured = TRUE;

    pData->Enabled = TRUE;
    pData->NumVectors = 1;

    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeGetIommuInfo(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA *pData
    )
{
    UNREFERENCED_PARAMETER(pAdapter);

    pData->IommuPresent = TRUE;
    pData->IommuEnabled = TRUE;
    pData->DmaRemappingActive = TRUE;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * DxgkDdiEscape -- main dispatch
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuEscape(
    IN_CONST_HANDLE             hAdapter,
    IN_CONST_PDXGKARG_ESCAPE    pEscape
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    AMDGPU_ESCAPE_HEADER *pHeader;
    NTSTATUS Status;

    if (pEscape == NULL || pEscape->pPrivateDriverData == NULL)
        return STATUS_INVALID_PARAMETER;

    if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_HEADER))
        return STATUS_BUFFER_TOO_SMALL;

    pHeader = (AMDGPU_ESCAPE_HEADER *)pEscape->pPrivateDriverData;

    if (pHeader->Size > pEscape->PrivateDriverDataSize)
        return STATUS_BUFFER_TOO_SMALL;

    switch (pHeader->Command) {

    case AMDGPU_ESCAPE_GET_INFO:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_INFO_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeGetInfo(pAdapter,
                (AMDGPU_ESCAPE_GET_INFO_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_READ_REG32:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_REG32_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeReadReg32(pAdapter,
                (AMDGPU_ESCAPE_REG32_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_WRITE_REG32:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_REG32_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeWriteReg32(pAdapter,
                (AMDGPU_ESCAPE_REG32_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_MAP_BAR:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_BAR_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeMapBar(pAdapter,
                (AMDGPU_ESCAPE_MAP_BAR_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_UNMAP_BAR:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_BAR_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeUnmapBar(pAdapter,
                (AMDGPU_ESCAPE_MAP_BAR_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_ALLOC_DMA:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ALLOC_DMA_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeAllocDma(pAdapter,
                (AMDGPU_ESCAPE_ALLOC_DMA_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_FREE_DMA:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ALLOC_DMA_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeFreeDma(pAdapter,
                (AMDGPU_ESCAPE_ALLOC_DMA_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_MAP_VRAM:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_VRAM_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeMapVram(pAdapter,
                (AMDGPU_ESCAPE_MAP_VRAM_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_REGISTER_EVENT:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_REGISTER_EVENT_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeRegisterEvent(pAdapter,
                (AMDGPU_ESCAPE_REGISTER_EVENT_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_ENABLE_MSI:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ENABLE_MSI_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeEnableMsi(pAdapter,
                (AMDGPU_ESCAPE_ENABLE_MSI_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_GET_IOMMU_INFO:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeGetIommuInfo(pAdapter,
                (AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA *)pEscape->pPrivateDriverData);
        break;

    /* KFD-equivalent compute operations (Phase 2) */
    case AMDGPU_ESCAPE_ALLOC_MEMORY:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ALLOC_MEMORY_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeAllocMemory(pAdapter,
                (AMDGPU_ESCAPE_ALLOC_MEMORY_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_FREE_MEMORY:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_FREE_MEMORY_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeFreeMemory(pAdapter,
                (AMDGPU_ESCAPE_FREE_MEMORY_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_MAP_MEMORY:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_MEMORY_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeMapMemory(pAdapter,
                (AMDGPU_ESCAPE_MAP_MEMORY_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_UNMAP_MEMORY:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_UNMAP_MEMORY_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeUnmapMemory(pAdapter,
                (AMDGPU_ESCAPE_UNMAP_MEMORY_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_CREATE_QUEUE:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_CREATE_QUEUE_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeCreateQueue(pAdapter,
                (AMDGPU_ESCAPE_CREATE_QUEUE_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_DESTROY_QUEUE:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_DESTROY_QUEUE_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeDestroyQueue(pAdapter,
                (AMDGPU_ESCAPE_DESTROY_QUEUE_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_UPDATE_QUEUE:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_UPDATE_QUEUE_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeUpdateQueue(pAdapter,
                (AMDGPU_ESCAPE_UPDATE_QUEUE_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_CREATE_EVENT:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_CREATE_EVENT_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeCreateEvent(pAdapter,
                (AMDGPU_ESCAPE_CREATE_EVENT_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_DESTROY_EVENT:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_DESTROY_EVENT_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeDestroyEvent(pAdapter,
                (AMDGPU_ESCAPE_DESTROY_EVENT_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_SET_EVENT:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_SET_EVENT_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeSetEvent(pAdapter,
                (AMDGPU_ESCAPE_SET_EVENT_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_RESET_EVENT:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_RESET_EVENT_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeResetEvent(pAdapter,
                (AMDGPU_ESCAPE_RESET_EVENT_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_WAIT_EVENTS:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_WAIT_EVENTS_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeWaitEvents(pAdapter,
                (AMDGPU_ESCAPE_WAIT_EVENTS_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_GET_PROCESS_APERTURES:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeGetProcessApertures(pAdapter,
                (AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_SET_MEMORY_POLICY:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeSetMemoryPolicy(pAdapter,
                (AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_SET_SCRATCH_BACKING:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeSetScratchBacking(pAdapter,
                (AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_SET_TRAP_HANDLER:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeSetTrapHandler(pAdapter,
                (AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_GET_CLOCK_COUNTERS:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeGetClockCounters(pAdapter,
                (AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA *)pEscape->pPrivateDriverData);
        break;

    case AMDGPU_ESCAPE_GET_VERSION:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_VERSION_DATA))
            Status = STATUS_BUFFER_TOO_SMALL;
        else
            Status = EscapeGetVersion(pAdapter,
                (AMDGPU_ESCAPE_GET_VERSION_DATA *)pEscape->pPrivateDriverData);
        break;

    default:
        Status = STATUS_INVALID_PARAMETER;
        break;
    }

    pHeader->Status = Status;
    return Status;
}
