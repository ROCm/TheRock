/*
 * ddi_tdr.c - TDR (Timeout Detection and Recovery) handlers
 *
 * These DDIs handle GPU timeout recovery. For development, TDR should
 * be disabled via registry (TdrLevel=0). These handlers provide the
 * framework for real GPU reset in later versions.
 *
 * v0.1: Minimal stubs that report success.
 * v0.5+: Real GPU soft reset via GRBM registers.
 */

#include "amdgpu_mcdm.h"

/* ======================================================================
 * ResetFromTimeout — full adapter reset after TDR
 *
 * Called when dxgkrnl detects a GPU hang (2-second default timeout).
 * We must reset the GPU and return it to a working state.
 *
 * v0.1: Return success (GPU isn't doing anything yet).
 * v0.5+: Write GRBM soft reset registers, wait for completion,
 *         re-initialize minimal state.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuResetFromTimeout(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

    UNREFERENCED_PARAMETER(pAdapter);

    /*
     * TODO (v0.5): Implement real GPU reset:
     * 1. Write GRBM_SOFT_RESET register to reset compute engines
     * 2. Wait for reset completion (poll GRBM_STATUS)
     * 3. Re-initialize doorbell and ring state
     *
     * For now, return success since no real GPU work is happening.
     */

    return STATUS_SUCCESS;
}

/* ======================================================================
 * RestartFromTimeout — re-initialize after reset
 *
 * Called after ResetFromTimeout succeeds. The driver should restore
 * any state needed for operation.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuRestartFromTimeout(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

    UNREFERENCED_PARAMETER(pAdapter);

    /* Nothing to restore in v0.1 */

    return STATUS_SUCCESS;
}

/* ======================================================================
 * ResetEngine — per-engine reset (WDDM 1.2+)
 *
 * More targeted than ResetFromTimeout: reset a specific GPU engine
 * without affecting others.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuResetEngine(
    IN_CONST_HANDLE                 hAdapter,
    INOUT_PDXGKARG_RESETENGINE      pResetEngine
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pResetEngine == NULL)
        return STATUS_INVALID_PARAMETER;

    /*
     * TODO (v0.5): Per-engine reset:
     * - NodeOrdinal 0 = compute engine → reset GFX pipe
     * - Additional nodes for SDMA engines
     *
     * Return the last completed fence so dxgkrnl knows which
     * submissions to retry.
     */
    pResetEngine->LastAbortedFenceId = 0;

    return STATUS_SUCCESS;
}
