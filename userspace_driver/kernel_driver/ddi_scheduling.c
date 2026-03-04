/*
 * ddi_scheduling.c - Scheduling DDIs for MCDM miniport (stubs)
 *
 * Real GPU command submission goes through DxgkDdiEscape → Python.
 * These DDIs satisfy the WDDM contract with no-op implementations.
 *
 * SubmitCommand/SubmitCommandVirtual, PreemptCommand, Patch, and
 * QueryCurrentFence are all minimal stubs.
 */

#include "amdgpu_mcdm.h"

/* ======================================================================
 * SubmitCommand — legacy WDDM 1.x command submission
 *
 * dxgkrnl may call this for paging operations. We signal completion
 * immediately since BuildPagingBuffer is a no-op.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSubmitCommand(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_SUBMITCOMMAND     pSubmitCommand
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    DXGKARGCB_NOTIFY_INTERRUPT_DATA NotifyData;

    if (pSubmitCommand == NULL)
        return STATUS_SUCCESS;

    /*
     * Signal immediate completion to dxgkrnl. We do this by notifying
     * a "DMA completed" interrupt and then requesting a DPC.
     */
    RtlZeroMemory(&NotifyData, sizeof(NotifyData));
    NotifyData.InterruptType = DXGK_INTERRUPT_DMA_COMPLETED;
    NotifyData.DmaCompleted.SubmissionFenceId = pSubmitCommand->SubmissionFenceId;
    NotifyData.DmaCompleted.NodeOrdinal = pSubmitCommand->NodeOrdinal;
    NotifyData.DmaCompleted.EngineOrdinal = pSubmitCommand->EngineOrdinal;

    pAdapter->DxgkInterface.DxgkCbNotifyInterrupt(
        pAdapter->DxgkInterface.DeviceHandle, &NotifyData);

    pAdapter->DxgkInterface.DxgkCbQueueDpc(
        pAdapter->DxgkInterface.DeviceHandle);

    return STATUS_SUCCESS;
}

/* ======================================================================
 * SubmitCommandVirtual — WDDM 2.0+ virtual addressing mode
 *
 * Same as SubmitCommand: signal immediate completion.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSubmitCommandVirtual(
    IN_CONST_HANDLE                             hAdapter,
    IN_CONST_PDXGKARG_SUBMITCOMMANDVIRTUAL      pSubmitCommandVirtual
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    DXGKARGCB_NOTIFY_INTERRUPT_DATA NotifyData;

    if (pSubmitCommandVirtual == NULL)
        return STATUS_SUCCESS;

    RtlZeroMemory(&NotifyData, sizeof(NotifyData));
    NotifyData.InterruptType = DXGK_INTERRUPT_DMA_COMPLETED;
    NotifyData.DmaCompleted.SubmissionFenceId = pSubmitCommandVirtual->SubmissionFenceId;
    NotifyData.DmaCompleted.NodeOrdinal = pSubmitCommandVirtual->NodeOrdinal;
    NotifyData.DmaCompleted.EngineOrdinal = pSubmitCommandVirtual->EngineOrdinal;

    pAdapter->DxgkInterface.DxgkCbNotifyInterrupt(
        pAdapter->DxgkInterface.DeviceHandle, &NotifyData);

    pAdapter->DxgkInterface.DxgkCbQueueDpc(
        pAdapter->DxgkInterface.DeviceHandle);

    return STATUS_SUCCESS;
}

/* ======================================================================
 * PreemptCommand — no-op, since SubmitCommand signals immediately
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuPreemptCommand(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_PREEMPTCOMMAND        pPreemptCommand
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pPreemptCommand);
    /* Nothing to preempt — commands complete immediately */
    return STATUS_SUCCESS;
}

/* ======================================================================
 * Patch — fixup DMA buffer addresses (not needed for virtual mode)
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuPatch(
    IN_CONST_HANDLE                 hAdapter,
    IN_CONST_PDXGKARG_PATCH         pPatch
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pPatch);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * QueryCurrentFence — return the last completed fence
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryCurrentFence(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_QUERYCURRENTFENCE        pCurrentFence
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pCurrentFence != NULL) {
        /*
         * Since all submissions complete immediately, report the
         * highest possible fence value.
         */
        pCurrentFence->CurrentFence = 0xFFFFFFFF;
    }

    return STATUS_SUCCESS;
}
