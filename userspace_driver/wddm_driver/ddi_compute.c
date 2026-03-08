/*
 * ddi_compute.c - KFD-equivalent compute operations for WDDM driver
 *
 * Implements GPU memory allocation, compute queue management, and
 * event signaling through DxgkDdiEscape handlers. These provide the
 * kernel-side support for the wddm_lite libhsakmt backend.
 */

#include "amdgpu_wddm.h"

/* ======================================================================
 * Compute state initialization
 * ====================================================================== */

NTSTATUS
AmdGpuComputeInit(_Inout_ AMDGPU_ADAPTER *pAdapter)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;

    if (cs->Initialized)
        return STATUS_SUCCESS;

    RtlZeroMemory(cs, sizeof(*cs));
    KeInitializeSpinLock(&cs->AllocsLock);
    KeInitializeSpinLock(&cs->QueuesLock);
    KeInitializeSpinLock(&cs->EventsLock);

    cs->NextAllocHandle = 1;
    cs->NextQueueId = 1;
    cs->NextEventId = 1;

    /* Default GPUVM apertures for GFX12.
     * These match what KFD provides on Linux. */
    cs->GpuVmBase   = 0x0000000100000000ULL;  /* 4GB start */
    cs->GpuVmLimit  = 0x0000FFFFFFFFFFFFULL;  /* 256TB limit */
    cs->LdsBase     = 0x0000000000000000ULL;
    cs->LdsLimit    = 0x0000000000000000ULL;   /* LDS is per-workgroup, no global aperture */
    cs->ScratchBase = 0x0001000000000000ULL;
    cs->ScratchLimit = 0x0001FFFFFFFFFFFFULL;

    cs->Initialized = TRUE;

    DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "amdgpu_wddm: compute state initialized, GPUVM 0x%llx-0x%llx\n",
               cs->GpuVmBase, cs->GpuVmLimit);

    return STATUS_SUCCESS;
}

void
AmdGpuComputeCleanup(_Inout_ AMDGPU_ADAPTER *pAdapter)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG i;

    if (!cs->Initialized)
        return;

    /* Free all GPU allocations */
    for (i = 0; i < AMDGPU_MAX_GPU_ALLOCS; i++) {
        if (cs->Allocs[i].InUse) {
            if (cs->Allocs[i].Mdl) {
                if (cs->Allocs[i].CpuVa) {
                    MmUnmapLockedPages(cs->Allocs[i].CpuVa, cs->Allocs[i].Mdl);
                }
                IoFreeMdl(cs->Allocs[i].Mdl);
            }
            if (cs->Allocs[i].KernelVa) {
                MmFreeContiguousMemory(cs->Allocs[i].KernelVa);
            }
            cs->Allocs[i].InUse = FALSE;
        }
    }

    /* Destroy all queues */
    for (i = 0; i < AMDGPU_MAX_GPU_QUEUES; i++) {
        if (cs->Queues[i].InUse) {
            if (cs->Queues[i].MqdKernelVa) {
                MmFreeContiguousMemory(cs->Queues[i].MqdKernelVa);
            }
            cs->Queues[i].InUse = FALSE;
        }
    }

    /* Destroy all events */
    for (i = 0; i < AMDGPU_MAX_GPU_EVENTS; i++) {
        if (cs->Events[i].InUse) {
            if (cs->Events[i].KernelEvent) {
                ObDereferenceObject(cs->Events[i].KernelEvent);
            }
            cs->Events[i].InUse = FALSE;
        }
    }

    /* Free event page */
    if (cs->EventPage.Allocated) {
        if (cs->EventPage.UserVa && cs->EventPage.Mdl) {
            MmUnmapLockedPages(cs->EventPage.UserVa, cs->EventPage.Mdl);
        }
        if (cs->EventPage.Mdl) {
            IoFreeMdl(cs->EventPage.Mdl);
        }
        if (cs->EventPage.KernelVa) {
            MmFreeContiguousMemory(cs->EventPage.KernelVa);
        }
        cs->EventPage.Allocated = FALSE;
    }

    cs->Initialized = FALSE;
}

/* ======================================================================
 * Memory management escapes
 * ====================================================================== */

NTSTATUS
EscapeAllocMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                  _Inout_ AMDGPU_ESCAPE_ALLOC_MEMORY_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    PHYSICAL_ADDRESS lowAddr, highAddr, boundary;
    KIRQL oldIrql;
    ULONG i;
    ULONG slotIdx = (ULONG)-1;
    PVOID kernelVa = NULL;
    PHYSICAL_ADDRESS physAddr;
    SIZE_T allocSize;

    if (!cs->Initialized) {
        NTSTATUS st = AmdGpuComputeInit(pAdapter);
        if (!NT_SUCCESS(st))
            return st;
    }

    allocSize = (SIZE_T)pData->SizeInBytes;
    if (allocSize == 0)
        return STATUS_INVALID_PARAMETER;

    /* Round up to page boundary */
    allocSize = (allocSize + 0xFFF) & ~0xFFF;

    /* Find free allocation slot */
    KeAcquireSpinLock(&cs->AllocsLock, &oldIrql);
    for (i = 0; i < AMDGPU_MAX_GPU_ALLOCS; i++) {
        if (!cs->Allocs[i].InUse) {
            slotIdx = i;
            cs->Allocs[i].InUse = TRUE;
            break;
        }
    }
    KeReleaseSpinLock(&cs->AllocsLock, oldIrql);

    if (slotIdx == (ULONG)-1)
        return STATUS_INSUFFICIENT_RESOURCES;

    /* Allocate contiguous physical memory */
    lowAddr.QuadPart = 0;
    highAddr.QuadPart = 0xFFFFFFFFFFFFFFFFULL;
    boundary.QuadPart = 0;

    if (pData->Alignment > 0) {
        boundary.QuadPart = pData->Alignment;
    }

    kernelVa = MmAllocateContiguousMemorySpecifyCache(
        allocSize, lowAddr, highAddr, boundary,
        (pData->Flags & AMDGPU_MEM_FLAG_UNCACHED) ?
            MmNonCached : MmWriteCombined);

    if (!kernelVa) {
        cs->Allocs[slotIdx].InUse = FALSE;
        return STATUS_NO_MEMORY;
    }

    physAddr = MmGetPhysicalAddress(kernelVa);
    RtlZeroMemory(kernelVa, allocSize);

    /* Fill allocation record */
    cs->Allocs[slotIdx].Flags = pData->Flags;
    cs->Allocs[slotIdx].SizeInBytes = allocSize;
    cs->Allocs[slotIdx].KernelVa = kernelVa;
    cs->Allocs[slotIdx].PhysAddr = physAddr;
    cs->Allocs[slotIdx].GpuVa = physAddr.QuadPart; /* Identity map for now */

    /* Map to userspace if host-accessible */
    if (pData->Flags & AMDGPU_MEM_FLAG_HOST_ACCESS) {
        PMDL mdl = IoAllocateMdl(kernelVa, (ULONG)allocSize,
                                  FALSE, FALSE, NULL);
        if (mdl) {
            MmBuildMdlForNonPagedPool(mdl);
            __try {
                PVOID userVa = MmMapLockedPagesSpecifyCache(
                    mdl, UserMode, MmWriteCombined,
                    NULL, FALSE, NormalPagePriority);
                cs->Allocs[slotIdx].Mdl = mdl;
                cs->Allocs[slotIdx].CpuVa = userVa;
            }
            __except (EXCEPTION_EXECUTE_HANDLER) {
                IoFreeMdl(mdl);
                cs->Allocs[slotIdx].CpuVa = NULL;
            }
        }
    }

    /* Return results */
    pData->CpuAddress = cs->Allocs[slotIdx].CpuVa;
    pData->GpuAddress = cs->Allocs[slotIdx].GpuVa;
    pData->Handle = (ULONGLONG)slotIdx;

    DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_TRACE_LEVEL,
               "amdgpu_wddm: alloc memory slot %u, size 0x%llx, phys 0x%llx, "
               "cpu %p, gpu 0x%llx\n",
               slotIdx, (ULONGLONG)allocSize, physAddr.QuadPart,
               cs->Allocs[slotIdx].CpuVa, cs->Allocs[slotIdx].GpuVa);

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeFreeMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                 _Inout_ AMDGPU_ESCAPE_FREE_MEMORY_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG slotIdx = (ULONG)pData->Handle;
    KIRQL oldIrql;

    if (slotIdx >= AMDGPU_MAX_GPU_ALLOCS)
        return STATUS_INVALID_PARAMETER;

    KeAcquireSpinLock(&cs->AllocsLock, &oldIrql);

    if (!cs->Allocs[slotIdx].InUse) {
        KeReleaseSpinLock(&cs->AllocsLock, oldIrql);
        return STATUS_INVALID_PARAMETER;
    }

    if (cs->Allocs[slotIdx].Mdl) {
        if (cs->Allocs[slotIdx].CpuVa) {
            MmUnmapLockedPages(cs->Allocs[slotIdx].CpuVa,
                               cs->Allocs[slotIdx].Mdl);
        }
        IoFreeMdl(cs->Allocs[slotIdx].Mdl);
    }

    if (cs->Allocs[slotIdx].KernelVa) {
        MmFreeContiguousMemory(cs->Allocs[slotIdx].KernelVa);
    }

    RtlZeroMemory(&cs->Allocs[slotIdx], sizeof(AMDGPU_GPU_ALLOC));
    KeReleaseSpinLock(&cs->AllocsLock, oldIrql);

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeMapMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                _Inout_ AMDGPU_ESCAPE_MAP_MEMORY_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG slotIdx = (ULONG)pData->Handle;

    if (slotIdx >= AMDGPU_MAX_GPU_ALLOCS || !cs->Allocs[slotIdx].InUse)
        return STATUS_INVALID_PARAMETER;

    /* For now, GPU VA = physical address (identity mapping).
     * Full GPU page table programming will be added when GMC init is done. */
    pData->GpuAddress = cs->Allocs[slotIdx].GpuVa;

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeUnmapMemory(_In_ AMDGPU_ADAPTER *pAdapter,
                  _Inout_ AMDGPU_ESCAPE_UNMAP_MEMORY_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG slotIdx = (ULONG)pData->Handle;

    if (slotIdx >= AMDGPU_MAX_GPU_ALLOCS || !cs->Allocs[slotIdx].InUse)
        return STATUS_INVALID_PARAMETER;

    /* Unmap from GPU page tables when GMC is implemented */
    return STATUS_SUCCESS;
}

/* ======================================================================
 * Queue management escapes
 * ====================================================================== */

NTSTATUS
EscapeCreateQueue(_In_ AMDGPU_ADAPTER *pAdapter,
                  _Inout_ AMDGPU_ESCAPE_CREATE_QUEUE_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    KIRQL oldIrql;
    ULONG i;
    ULONG slotIdx = (ULONG)-1;

    if (!cs->Initialized)
        return STATUS_DEVICE_NOT_READY;

    /* Find free queue slot */
    KeAcquireSpinLock(&cs->QueuesLock, &oldIrql);
    for (i = 0; i < AMDGPU_MAX_GPU_QUEUES; i++) {
        if (!cs->Queues[i].InUse) {
            slotIdx = i;
            cs->Queues[i].InUse = TRUE;
            cs->Queues[i].QueueId = cs->NextQueueId++;
            break;
        }
    }
    KeReleaseSpinLock(&cs->QueuesLock, oldIrql);

    if (slotIdx == (ULONG)-1)
        return STATUS_INSUFFICIENT_RESOURCES;

    cs->Queues[slotIdx].QueueType = pData->QueueType;
    cs->Queues[slotIdx].RingBufferGpuVa = pData->QueueAddress;
    cs->Queues[slotIdx].RingSizeBytes = pData->QueueSizeInBytes;
    cs->Queues[slotIdx].Priority = pData->Priority;

    /* Calculate doorbell offset.
     * Each queue gets an 8-byte doorbell slot in the doorbell BAR. */
    cs->Queues[slotIdx].DoorbellOffset = (ULONGLONG)slotIdx * 8;

    /* TODO: Allocate MQD and program MES for queue activation.
     * This requires MES firmware to be loaded and initialized.
     * For now, we just record the queue metadata. */

    pData->QueueId = (ULONGLONG)cs->Queues[slotIdx].QueueId;
    pData->DoorbellOffset = cs->Queues[slotIdx].DoorbellOffset;

    DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "amdgpu_wddm: create queue %u, type %u, ring 0x%llx size 0x%llx, "
               "doorbell offset 0x%llx\n",
               cs->Queues[slotIdx].QueueId, pData->QueueType,
               pData->QueueAddress, pData->QueueSizeInBytes,
               cs->Queues[slotIdx].DoorbellOffset);

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeDestroyQueue(_In_ AMDGPU_ADAPTER *pAdapter,
                   _Inout_ AMDGPU_ESCAPE_DESTROY_QUEUE_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    KIRQL oldIrql;
    ULONG i;

    KeAcquireSpinLock(&cs->QueuesLock, &oldIrql);
    for (i = 0; i < AMDGPU_MAX_GPU_QUEUES; i++) {
        if (cs->Queues[i].InUse &&
            cs->Queues[i].QueueId == (ULONG)pData->QueueId) {

            /* TODO: Deactivate queue via MES before freeing */
            if (cs->Queues[i].MqdKernelVa) {
                MmFreeContiguousMemory(cs->Queues[i].MqdKernelVa);
            }
            RtlZeroMemory(&cs->Queues[i], sizeof(AMDGPU_GPU_QUEUE));
            KeReleaseSpinLock(&cs->QueuesLock, oldIrql);
            return STATUS_SUCCESS;
        }
    }
    KeReleaseSpinLock(&cs->QueuesLock, oldIrql);

    return STATUS_NOT_FOUND;
}

NTSTATUS
EscapeUpdateQueue(_In_ AMDGPU_ADAPTER *pAdapter,
                  _Inout_ AMDGPU_ESCAPE_UPDATE_QUEUE_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG i;

    for (i = 0; i < AMDGPU_MAX_GPU_QUEUES; i++) {
        if (cs->Queues[i].InUse &&
            cs->Queues[i].QueueId == (ULONG)pData->QueueId) {

            cs->Queues[i].Priority = pData->Priority;
            cs->Queues[i].RingBufferGpuVa = pData->QueueAddress;
            cs->Queues[i].RingSizeBytes = pData->QueueSizeInBytes;

            /* TODO: Update MQD and notify MES */
            return STATUS_SUCCESS;
        }
    }

    return STATUS_NOT_FOUND;
}

/* ======================================================================
 * Event management escapes
 * ====================================================================== */

static NTSTATUS
EnsureEventPage(_In_ AMDGPU_ADAPTER *pAdapter)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    PHYSICAL_ADDRESS lowAddr, highAddr, boundary;
    SIZE_T pageSize = 4096;

    if (cs->EventPage.Allocated)
        return STATUS_SUCCESS;

    lowAddr.QuadPart = 0;
    highAddr.QuadPart = 0xFFFFFFFFFFFFFFFFULL;
    boundary.QuadPart = 0;

    cs->EventPage.KernelVa = MmAllocateContiguousMemorySpecifyCache(
        pageSize, lowAddr, highAddr, boundary, MmNonCached);

    if (!cs->EventPage.KernelVa)
        return STATUS_NO_MEMORY;

    RtlZeroMemory(cs->EventPage.KernelVa, pageSize);
    cs->EventPage.PhysAddr = MmGetPhysicalAddress(cs->EventPage.KernelVa);

    /* Create MDL for user mapping */
    cs->EventPage.Mdl = IoAllocateMdl(cs->EventPage.KernelVa,
                                       (ULONG)pageSize,
                                       FALSE, FALSE, NULL);
    if (!cs->EventPage.Mdl) {
        MmFreeContiguousMemory(cs->EventPage.KernelVa);
        cs->EventPage.KernelVa = NULL;
        return STATUS_NO_MEMORY;
    }

    MmBuildMdlForNonPagedPool(cs->EventPage.Mdl);

    __try {
        cs->EventPage.UserVa = MmMapLockedPagesSpecifyCache(
            cs->EventPage.Mdl, UserMode, MmNonCached,
            NULL, FALSE, NormalPagePriority);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        IoFreeMdl(cs->EventPage.Mdl);
        MmFreeContiguousMemory(cs->EventPage.KernelVa);
        cs->EventPage.Mdl = NULL;
        cs->EventPage.KernelVa = NULL;
        return STATUS_UNSUCCESSFUL;
    }

    cs->EventPage.Allocated = TRUE;

    DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL,
               "amdgpu_wddm: event page allocated at kernel %p, user %p\n",
               cs->EventPage.KernelVa, cs->EventPage.UserVa);

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeCreateEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                  _Inout_ AMDGPU_ESCAPE_CREATE_EVENT_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    KIRQL oldIrql;
    ULONG i;
    ULONG slotIdx = (ULONG)-1;
    NTSTATUS status;

    if (!cs->Initialized) {
        status = AmdGpuComputeInit(pAdapter);
        if (!NT_SUCCESS(status))
            return status;
    }

    /* Ensure event page exists */
    status = EnsureEventPage(pAdapter);
    if (!NT_SUCCESS(status))
        return status;

    /* Find free event slot */
    KeAcquireSpinLock(&cs->EventsLock, &oldIrql);
    for (i = 0; i < AMDGPU_MAX_GPU_EVENTS; i++) {
        if (!cs->Events[i].InUse) {
            slotIdx = i;
            cs->Events[i].InUse = TRUE;
            cs->Events[i].EventId = cs->NextEventId++;
            break;
        }
    }
    KeReleaseSpinLock(&cs->EventsLock, oldIrql);

    if (slotIdx == (ULONG)-1)
        return STATUS_INSUFFICIENT_RESOURCES;

    cs->Events[slotIdx].EventType = pData->EventType;
    cs->Events[slotIdx].AutoReset = pData->AutoReset;
    cs->Events[slotIdx].Signaled = FALSE;
    cs->Events[slotIdx].EventPageSlot = (ULONGLONG)slotIdx * sizeof(ULONGLONG);

    /* Create a kernel event object for wait/signal */
    cs->Events[slotIdx].KernelEvent = (PKEVENT)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(KEVENT), AMDGPU_POOL_TAG);

    if (!cs->Events[slotIdx].KernelEvent) {
        cs->Events[slotIdx].InUse = FALSE;
        return STATUS_NO_MEMORY;
    }

    KeInitializeEvent(cs->Events[slotIdx].KernelEvent,
                      pData->AutoReset ? SynchronizationEvent : NotificationEvent,
                      FALSE);

    /* Return results */
    pData->EventId = cs->Events[slotIdx].EventId;
    pData->EventPageAddress = (ULONGLONG)cs->EventPage.UserVa;
    pData->EventSlotIndex = slotIdx;

    DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_TRACE_LEVEL,
               "amdgpu_wddm: create event %u, type %u, slot %u\n",
               pData->EventId, pData->EventType, slotIdx);

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeDestroyEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                   _Inout_ AMDGPU_ESCAPE_DESTROY_EVENT_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    KIRQL oldIrql;
    ULONG i;

    KeAcquireSpinLock(&cs->EventsLock, &oldIrql);
    for (i = 0; i < AMDGPU_MAX_GPU_EVENTS; i++) {
        if (cs->Events[i].InUse &&
            cs->Events[i].EventId == pData->EventId) {

            if (cs->Events[i].KernelEvent) {
                /* Signal any waiters before destroying */
                KeSetEvent(cs->Events[i].KernelEvent, 0, FALSE);
                ExFreePoolWithTag(cs->Events[i].KernelEvent, AMDGPU_POOL_TAG);
            }
            RtlZeroMemory(&cs->Events[i], sizeof(AMDGPU_GPU_EVENT));
            KeReleaseSpinLock(&cs->EventsLock, oldIrql);
            return STATUS_SUCCESS;
        }
    }
    KeReleaseSpinLock(&cs->EventsLock, oldIrql);

    return STATUS_NOT_FOUND;
}

NTSTATUS
EscapeSetEvent(_In_ AMDGPU_ADAPTER *pAdapter,
               _Inout_ AMDGPU_ESCAPE_SET_EVENT_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG i;

    for (i = 0; i < AMDGPU_MAX_GPU_EVENTS; i++) {
        if (cs->Events[i].InUse &&
            cs->Events[i].EventId == pData->EventId) {

            cs->Events[i].Signaled = TRUE;

            /* Update event page slot */
            if (cs->EventPage.KernelVa) {
                volatile ULONGLONG *slot = (volatile ULONGLONG *)
                    ((PUCHAR)cs->EventPage.KernelVa + cs->Events[i].EventPageSlot);
                InterlockedIncrement64((volatile LONG64 *)slot);
            }

            if (cs->Events[i].KernelEvent) {
                KeSetEvent(cs->Events[i].KernelEvent, 0, FALSE);
            }
            return STATUS_SUCCESS;
        }
    }

    return STATUS_NOT_FOUND;
}

NTSTATUS
EscapeResetEvent(_In_ AMDGPU_ADAPTER *pAdapter,
                 _Inout_ AMDGPU_ESCAPE_RESET_EVENT_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    ULONG i;

    for (i = 0; i < AMDGPU_MAX_GPU_EVENTS; i++) {
        if (cs->Events[i].InUse &&
            cs->Events[i].EventId == pData->EventId) {

            cs->Events[i].Signaled = FALSE;
            if (cs->Events[i].KernelEvent) {
                KeClearEvent(cs->Events[i].KernelEvent);
            }
            return STATUS_SUCCESS;
        }
    }

    return STATUS_NOT_FOUND;
}

NTSTATUS
EscapeWaitEvents(_In_ AMDGPU_ADAPTER *pAdapter,
                 _Inout_ AMDGPU_ESCAPE_WAIT_EVENTS_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;
    PVOID waitObjects[AMDGPU_MAX_WAIT_EVENTS];
    ULONG numWait = 0;
    ULONG i, j;
    LARGE_INTEGER timeout;
    NTSTATUS status;

    if (pData->NumEvents == 0 || pData->NumEvents > AMDGPU_MAX_WAIT_EVENTS)
        return STATUS_INVALID_PARAMETER;

    /* Collect kernel event objects */
    for (i = 0; i < pData->NumEvents; i++) {
        BOOLEAN found = FALSE;
        for (j = 0; j < AMDGPU_MAX_GPU_EVENTS; j++) {
            if (cs->Events[j].InUse &&
                cs->Events[j].EventId == pData->EventIds[i] &&
                cs->Events[j].KernelEvent) {
                waitObjects[numWait++] = cs->Events[j].KernelEvent;
                found = TRUE;
                break;
            }
        }
        if (!found)
            return STATUS_NOT_FOUND;
    }

    /* Convert timeout */
    if (pData->TimeoutMs == 0xFFFFFFFF) {
        /* Wait indefinitely */
        status = KeWaitForMultipleObjects(
            numWait, waitObjects,
            pData->WaitAll ? WaitAll : WaitAny,
            Executive, UserMode, TRUE,
            NULL, NULL);
    } else {
        timeout.QuadPart = -(LONGLONG)pData->TimeoutMs * 10000LL;
        status = KeWaitForMultipleObjects(
            numWait, waitObjects,
            pData->WaitAll ? WaitAll : WaitAny,
            Executive, UserMode, TRUE,
            &timeout, NULL);
    }

    if (status == STATUS_TIMEOUT) {
        pData->Header.Status = STATUS_TIMEOUT;
        return STATUS_TIMEOUT;
    }

    if (status >= STATUS_WAIT_0 && status < STATUS_WAIT_0 + numWait) {
        pData->SignaledIndex = (ULONG)(status - STATUS_WAIT_0);
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * Configuration escapes
 * ====================================================================== */

NTSTATUS
EscapeGetProcessApertures(_In_ AMDGPU_ADAPTER *pAdapter,
                          _Inout_ AMDGPU_ESCAPE_GET_PROCESS_APERTURES_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;

    if (!cs->Initialized)
        return STATUS_DEVICE_NOT_READY;

    pData->LdsBase = cs->LdsBase;
    pData->LdsLimit = cs->LdsLimit;
    pData->ScratchBase = cs->ScratchBase;
    pData->ScratchLimit = cs->ScratchLimit;
    pData->GpuVmBase = cs->GpuVmBase;
    pData->GpuVmLimit = cs->GpuVmLimit;

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeSetMemoryPolicy(_In_ AMDGPU_ADAPTER *pAdapter,
                      _Inout_ AMDGPU_ESCAPE_SET_MEMORY_POLICY_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;

    cs->DefaultCachePolicy = pData->DefaultPolicy;
    cs->AlternateCachePolicy = pData->AlternatePolicy;

    /* TODO: Program GPU MC registers for cache policy when GMC is up */
    return STATUS_SUCCESS;
}

NTSTATUS
EscapeSetScratchBacking(_In_ AMDGPU_ADAPTER *pAdapter,
                        _Inout_ AMDGPU_ESCAPE_SET_SCRATCH_BACKING_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;

    cs->ScratchBackingVa = pData->ScratchBackingVa;
    cs->ScratchBackingSize = pData->ScratchBackingSize;

    /* TODO: Program scratch backing address into GPU when GFX is up */
    return STATUS_SUCCESS;
}

NTSTATUS
EscapeSetTrapHandler(_In_ AMDGPU_ADAPTER *pAdapter,
                     _Inout_ AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA *pData)
{
    AMDGPU_COMPUTE_STATE *cs = &pAdapter->Compute;

    cs->TbaAddress = pData->TbaAddress;
    cs->TbaSize = pData->TbaSize;
    cs->TmaAddress = pData->TmaAddress;
    cs->TmaSize = pData->TmaSize;

    /* TODO: Program TBA/TMA registers when GFX is up */
    return STATUS_SUCCESS;
}

NTSTATUS
EscapeGetClockCounters(_In_ AMDGPU_ADAPTER *pAdapter,
                       _Inout_ AMDGPU_ESCAPE_GET_CLOCK_COUNTERS_DATA *pData)
{
    LARGE_INTEGER perfCounter, perfFreq;
    LARGE_INTEGER systemTime;

    KeQueryPerformanceCounter(&perfFreq);
    perfCounter = KeQueryPerformanceCounter(NULL);
    KeQuerySystemTime(&systemTime);

    /* Use performance counter as GPU clock proxy until we read
     * the actual GPU clock register (RLC_GPU_CLOCK_COUNT_LSB/MSB). */
    pData->GpuClockCounter = perfCounter.QuadPart;
    pData->CpuClockCounter = perfCounter.QuadPart;
    pData->SystemClockCounter = systemTime.QuadPart;
    pData->SystemClockFrequencyHz = perfFreq.QuadPart;
    pData->GpuClockFrequencyHz = perfFreq.QuadPart; /* Approximate */

    return STATUS_SUCCESS;
}

NTSTATUS
EscapeGetVersion(_In_ AMDGPU_ADAPTER *pAdapter,
                 _Inout_ AMDGPU_ESCAPE_GET_VERSION_DATA *pData)
{
    /* Report KFD version 1.14 to match what wddm_lite expects */
    pData->KfdMajorVersion = 1;
    pData->KfdMinorVersion = 14;

    return STATUS_SUCCESS;
}
