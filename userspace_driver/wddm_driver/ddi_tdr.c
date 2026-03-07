/*
 * ddi_tdr.c - TDR handlers for ROCm Display Driver
 * Identical to MCDM.
 */

#include "amdgpu_wddm.h"

NTSTATUS
APIENTRY
AmdGpuResetFromTimeout(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuRestartFromTimeout(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    return STATUS_SUCCESS;
}

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

    pResetEngine->LastAbortedFenceId = 0;
    return STATUS_SUCCESS;
}
