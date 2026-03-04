/*
 * ddi_escape.c - DxgkDdiEscape handler for MCDM miniport
 *
 * This is the heart of the driver: routes commands between the Python
 * userspace driver and GPU hardware. Python calls D3DKMTEscape, which
 * goes through dxgkrnl.sys to this handler at PASSIVE_LEVEL.
 *
 * v0.1: Dispatch framework + GET_INFO.
 * v0.2+: READ_REG32, WRITE_REG32, MAP_BAR, ALLOC_DMA, etc.
 */

#include "amdgpu_mcdm.h"

/* IH entry size in bytes (4 x DWORD) — must match ddi_interrupt.c */
#define IH_ENTRY_SIZE   32

/* ======================================================================
 * Escape sub-handlers
 * ====================================================================== */

static NTSTATUS
EscapeGetInfo(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_GET_INFO_DATA *pData
    )
{
    ULONG i;

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

    if (BarIndex >= AMDGPU_MAX_BARS || !pAdapter->Bars[BarIndex].Mapped)
        return STATUS_INVALID_PARAMETER;

    if ((ULONGLONG)Offset + sizeof(ULONG) > pAdapter->Bars[BarIndex].Length)
        return STATUS_INVALID_PARAMETER;

    WRITE_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[BarIndex].KernelAddress + Offset),
        pData->Value);

    return STATUS_SUCCESS;
}

/* Placeholder stubs for v0.2+ escape commands */

/*
 * EscapeMapBar — map a PCI BAR region into the calling process's address space.
 *
 * Uses MmMapIoSpaceEx for kernel mapping, then creates an MDL and maps
 * it to userspace via MmMapLockedPagesSpecifyCache.
 *
 * This is how Python gets direct MMIO access for register reads/writes
 * and BAR2 VRAM access, bypassing the per-register escape overhead.
 */
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
    PMDL Mdl;
    PVOID UserVa;

    if (BarIndex >= AMDGPU_MAX_BARS || BarIndex >= pAdapter->NumBars)
        return STATUS_INVALID_PARAMETER;

    if (!pAdapter->Bars[BarIndex].IsMemory)
        return STATUS_INVALID_PARAMETER;

    /* Default: map entire BAR */
    if (Length == 0)
        Length = pAdapter->Bars[BarIndex].Length - Offset;

    if (Offset + Length > pAdapter->Bars[BarIndex].Length)
        return STATUS_INVALID_PARAMETER;

    /* Physical address within the BAR */
    PhysAddr.QuadPart =
        pAdapter->Bars[BarIndex].PhysicalAddress.QuadPart + Offset;

    /*
     * Create an MDL describing the physical I/O range, then map it
     * into the calling process's user-mode address space.
     * MmAllocatePagesForMdlEx + MmMapLockedPagesSpecifyCache is the
     * standard pattern for mapping physical MMIO to userspace.
     */
    Mdl = IoAllocateMdl(NULL, (ULONG)Length, FALSE, FALSE, NULL);
    if (Mdl == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    /* Build MDL for the physical I/O range */
    MmBuildMdlForNonPagedPool(Mdl);

    /*
     * Overwrite the MDL page array with our physical addresses.
     * For I/O space, we manually set the PFNs.
     */
    {
        PPFN_NUMBER Pages = MmGetMdlPfnArray(Mdl);
        ULONG PageCount = (ULONG)((Length + PAGE_SIZE - 1) >> PAGE_SHIFT);
        ULONGLONG BasePfn = PhysAddr.QuadPart >> PAGE_SHIFT;
        ULONG i;

        /* Reallocate MDL with correct page count */
        IoFreeMdl(Mdl);
        Mdl = IoAllocateMdl(
            NULL, (ULONG)Length, FALSE, FALSE, NULL);
        if (Mdl == NULL)
            return STATUS_INSUFFICIENT_RESOURCES;

        /* Set MDL to describe our physical pages */
        Mdl->MdlFlags |= MDL_PAGES_LOCKED;
        Pages = MmGetMdlPfnArray(Mdl);
        for (i = 0; i < PageCount; i++) {
            Pages[i] = (PFN_NUMBER)(BasePfn + i);
        }
    }

    __try {
        UserVa = MmMapLockedPagesSpecifyCache(
            Mdl,
            UserMode,
            MmNonCached,
            NULL,           /* Preferred base address (NULL = any) */
            FALSE,          /* BugCheckOnFailure */
            NormalPagePriority);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        IoFreeMdl(Mdl);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    if (UserVa == NULL) {
        IoFreeMdl(Mdl);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    pData->MappedAddress = UserVa;
    pData->MappingHandle = (PVOID)Mdl;  /* Store MDL as opaque handle */
    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeUnmapBar(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_MAP_BAR_DATA *pData
    )
{
    PMDL Mdl;

    UNREFERENCED_PARAMETER(pAdapter);

    if (pData->MappingHandle == NULL || pData->MappedAddress == NULL)
        return STATUS_INVALID_PARAMETER;

    Mdl = (PMDL)pData->MappingHandle;
    MmUnmapLockedPages(pData->MappedAddress, Mdl);
    IoFreeMdl(Mdl);

    pData->MappedAddress = NULL;
    pData->MappingHandle = NULL;
    return STATUS_SUCCESS;
}

/*
 * EscapeAllocDma — allocate contiguous DMA-capable system memory.
 *
 * Uses MmAllocateContiguousMemorySpecifyCache for physically contiguous
 * pages, then creates an MDL to map into userspace. Returns both the
 * user VA (for CPU access) and the physical/bus address (for GPU DMA).
 *
 * This is used for ring buffers, MQDs, fence memory, and other
 * structures that the GPU reads/writes via DMA.
 */
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

    if (Size == 0 || Size > 64 * 1024 * 1024)  /* Cap at 64MB */
        return STATUS_INVALID_PARAMETER;

    /* Round up to page boundary */
    Size = (Size + PAGE_SIZE - 1) & ~((ULONGLONG)PAGE_SIZE - 1);

    LowestAddr.QuadPart = 0;
    HighestAddr.QuadPart = (LONGLONG)-1;  /* Any address */
    BoundaryAddr.QuadPart = 0;

    KernelVa = MmAllocateContiguousMemorySpecifyCache(
        (SIZE_T)Size, LowestAddr, HighestAddr, BoundaryAddr,
        MmNonCached);
    if (KernelVa == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;

    RtlZeroMemory(KernelVa, (SIZE_T)Size);

    /* Get physical address for DMA */
    PhysAddr = MmGetPhysicalAddress(KernelVa);

    /* Create MDL and map to userspace */
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

    /* Track the allocation */
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
    pData->AllocationHandle = (PVOID)(ULONG_PTR)i;  /* Index as handle */
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

/*
 * EscapeMapVram — map a VRAM region to userspace via BAR2.
 *
 * Uses the same MDL-based mapping as EscapeMapBar but specifically
 * targets BAR2 (the VRAM aperture). With ReBAR enabled, the entire
 * VRAM may be accessible through BAR2.
 */
static NTSTATUS
EscapeMapVram(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_MAP_VRAM_DATA *pData
    )
{
    AMDGPU_ESCAPE_MAP_BAR_DATA BarData;
    NTSTATUS Status;
    ULONG VramBarIndex;

    /*
     * Find the VRAM BAR (BAR2 for AMD GPUs, but identify by size:
     * it's the largest memory BAR).
     */
    VramBarIndex = 0;
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

    /* Delegate to MAP_BAR with the VRAM BAR index */
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

/*
 * EscapeRegisterEvent — register a Windows Event to be signaled on GPU interrupt.
 *
 * Python creates a Win32 Event (CreateEvent), passes the HANDLE here.
 * We convert it to a kernel PKEVENT via ObReferenceObjectByHandle.
 * When the ISR/DPC sees a matching source_id in the IH ring, it
 * calls KeSetEvent to signal it, unblocking the Python WaitForSingleObject.
 */
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

    /* Convert user-mode HANDLE to kernel event object reference */
    Status = ObReferenceObjectByHandle(
        pData->EventHandle,
        EVENT_MODIFY_STATE,
        *ExEventObjectType,
        UserMode,
        (PVOID *)&Event,
        NULL);
    if (!NT_SUCCESS(Status))
        return Status;

    /* Find a free event slot */
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

    /* No free slots — release the reference */
    ObDereferenceObject(Event);
    return STATUS_INSUFFICIENT_RESOURCES;
}

/*
 * EscapeEnableMsi — configure the IH ring for interrupt processing.
 *
 * Python allocates the IH ring buffer via ALLOC_DMA, programs the
 * IH hardware registers (IH_RB_BASE, IH_RB_CNTL, etc.) via WRITE_REG32,
 * then calls this escape to tell the kernel where the ring is and
 * which BAR0 offsets to use for RPTR/WPTR.
 *
 * After this call, the ISR will start reading IH ring entries and
 * the DPC will signal registered events.
 */
static NTSTATUS
EscapeEnableMsi(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_ENABLE_MSI_DATA *pData
    )
{
    ULONG Index;

    /* Look up the DMA allocation for the IH ring buffer */
    Index = (ULONG)(ULONG_PTR)pData->IhRingDmaHandle;
    if (Index >= AMDGPU_MAX_DMA_ALLOCS)
        return STATUS_INVALID_PARAMETER;

    if (!pAdapter->DmaAllocs[Index].InUse)
        return STATUS_INVALID_PARAMETER;

    /* Ring size must be a power of 2 and at least one entry */
    if (pData->IhRingSize < IH_ENTRY_SIZE ||
        (pData->IhRingSize & (pData->IhRingSize - 1)) != 0)
        return STATUS_INVALID_PARAMETER;

    /* Ring size must not exceed the DMA allocation */
    if (pData->IhRingSize > pAdapter->DmaAllocs[Index].Size)
        return STATUS_INVALID_PARAMETER;

    /* Validate register offsets are within BAR0 */
    if (pAdapter->Bars[0].Length == 0 || !pAdapter->Bars[0].Mapped)
        return STATUS_DEVICE_CONFIGURATION_ERROR;

    if (pData->IhRptrRegOffset + sizeof(ULONG) > pAdapter->Bars[0].Length ||
        pData->IhWptrRegOffset + sizeof(ULONG) > pAdapter->Bars[0].Length)
        return STATUS_INVALID_PARAMETER;

    /* Configure IH ring state */
    pAdapter->IhRing.RingBuffer = pAdapter->DmaAllocs[Index].KernelVa;
    pAdapter->IhRing.RingSize = pData->IhRingSize;
    pAdapter->IhRing.RingMask = pData->IhRingSize - 1;
    pAdapter->IhRing.RptrRegOffset = pData->IhRptrRegOffset;
    pAdapter->IhRing.WptrRegOffset = pData->IhWptrRegOffset;
    pAdapter->IhRing.Rptr = 0;

    /* Memory barrier before enabling */
    KeMemoryBarrier();
    pAdapter->IhRing.Configured = TRUE;

    pData->Enabled = TRUE;
    pData->NumVectors = 1;  /* Report at least one MSI vector */

    return STATUS_SUCCESS;
}

static NTSTATUS
EscapeGetIommuInfo(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA *pData
    )
{
    UNREFERENCED_PARAMETER(pAdapter);

    /* v0.1: report basic IOMMU status (hardcoded for now) */
    pData->IommuPresent = TRUE;     /* Windows 11 always has IOMMU */
    pData->IommuEnabled = TRUE;
    pData->DmaRemappingActive = TRUE;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * DxgkDdiEscape — main dispatch
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

    /* Validate buffer */
    if (pEscape == NULL || pEscape->pPrivateDriverData == NULL)
        return STATUS_INVALID_PARAMETER;

    if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_HEADER))
        return STATUS_BUFFER_TOO_SMALL;

    pHeader = (AMDGPU_ESCAPE_HEADER *)pEscape->pPrivateDriverData;

    /* Validate size field matches buffer */
    if (pHeader->Size > pEscape->PrivateDriverDataSize)
        return STATUS_BUFFER_TOO_SMALL;

    /* Dispatch by command */
    switch (pHeader->Command) {

    case AMDGPU_ESCAPE_GET_INFO:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_INFO_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeGetInfo(pAdapter,
                (AMDGPU_ESCAPE_GET_INFO_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_READ_REG32:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_REG32_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeReadReg32(pAdapter,
                (AMDGPU_ESCAPE_REG32_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_WRITE_REG32:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_REG32_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeWriteReg32(pAdapter,
                (AMDGPU_ESCAPE_REG32_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_MAP_BAR:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_BAR_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeMapBar(pAdapter,
                (AMDGPU_ESCAPE_MAP_BAR_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_UNMAP_BAR:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_BAR_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeUnmapBar(pAdapter,
                (AMDGPU_ESCAPE_MAP_BAR_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_ALLOC_DMA:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ALLOC_DMA_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeAllocDma(pAdapter,
                (AMDGPU_ESCAPE_ALLOC_DMA_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_FREE_DMA:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ALLOC_DMA_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeFreeDma(pAdapter,
                (AMDGPU_ESCAPE_ALLOC_DMA_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_MAP_VRAM:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_MAP_VRAM_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeMapVram(pAdapter,
                (AMDGPU_ESCAPE_MAP_VRAM_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_REGISTER_EVENT:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_REGISTER_EVENT_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeRegisterEvent(pAdapter,
                (AMDGPU_ESCAPE_REGISTER_EVENT_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_ENABLE_MSI:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_ENABLE_MSI_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeEnableMsi(pAdapter,
                (AMDGPU_ESCAPE_ENABLE_MSI_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    case AMDGPU_ESCAPE_GET_IOMMU_INFO:
        if (pEscape->PrivateDriverDataSize < sizeof(AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA)) {
            Status = STATUS_BUFFER_TOO_SMALL;
        } else {
            Status = EscapeGetIommuInfo(pAdapter,
                (AMDGPU_ESCAPE_GET_IOMMU_INFO_DATA *)pEscape->pPrivateDriverData);
        }
        break;

    default:
        Status = STATUS_INVALID_PARAMETER;
        break;
    }

    /* Write status back into the escape buffer so Python can read it */
    pHeader->Status = Status;
    return Status;
}
