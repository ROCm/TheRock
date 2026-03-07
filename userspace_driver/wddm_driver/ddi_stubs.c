/*
 * ddi_stubs.c - Remaining DDI stubs for ROCm Display Driver
 *
 * Reduced from MCDM: child DDIs moved to ddi_display.c,
 * VidPn DDIs moved to ddi_vidpn.c, Present moved to ddi_present.c.
 */

#include "amdgpu_wddm.h"

/* Registry diagnostic helper */
static void
StubDbgRegWrite(const WCHAR *Name, ULONG Value)
{
    UNICODE_STRING KeyPath;
    OBJECT_ATTRIBUTES ObjAttrs;
    HANDLE hKey;
    NTSTATUS s;
    UNICODE_STRING ValName;

    RtlInitUnicodeString(&KeyPath,
        L"\\Registry\\Machine\\SOFTWARE\\AmdGpuWddm");
    InitializeObjectAttributes(&ObjAttrs, &KeyPath,
        OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, NULL, NULL);
    s = ZwOpenKey(&hKey, KEY_SET_VALUE, &ObjAttrs);
    if (NT_SUCCESS(s)) {
        RtlInitUnicodeString(&ValName, Name);
        ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Value, sizeof(Value));
        ZwClose(hKey);
    }
}

/* ======================================================================
 * Display stubs -- return STATUS_NOT_SUPPORTED where appropriate
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSetPalette(
    IN_CONST_HANDLE                 hAdapter,
    IN_CONST_PDXGKARG_SETPALETTE   pSetPalette
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetPalette);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuSetPointerPosition(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_SETPOINTERPOSITION   pSetPointerPosition
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetPointerPosition);
    /* No hardware cursor -- DWM uses software cursor */
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuSetPointerShape(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_SETPOINTERSHAPE  pSetPointerShape
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetPointerShape);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuStopCapture(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_STOPCAPTURE       pStopCapture
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pStopCapture);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuControlInterrupt(
    IN_CONST_HANDLE                 hAdapter,
    IN_CONST_DXGK_INTERRUPT_TYPE    InterruptType,
    IN_BOOLEAN                      EnableInterrupt
    )
{
    StubDbgRegWrite(L"ControlInterrupt", (ULONG)InterruptType);
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(InterruptType);
    UNREFERENCED_PARAMETER(EnableInterrupt);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuCreateOverlay(
    IN_CONST_HANDLE                     hAdapter,
    INOUT_PDXGKARG_CREATEOVERLAY        pCreateOverlay
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pCreateOverlay);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuUpdateOverlay(
    IN_CONST_HANDLE                     hOverlay,
    IN_CONST_PDXGKARG_UPDATEOVERLAY     pUpdateOverlay
    )
{
    UNREFERENCED_PARAMETER(hOverlay);
    UNREFERENCED_PARAMETER(pUpdateOverlay);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuFlipOverlay(
    IN_CONST_HANDLE                     hOverlay,
    IN_CONST_PDXGKARG_FLIPOVERLAY      pFlipOverlay
    )
{
    UNREFERENCED_PARAMETER(hOverlay);
    UNREFERENCED_PARAMETER(pFlipOverlay);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuDestroyOverlay(
    IN_CONST_HANDLE     hOverlay
    )
{
    UNREFERENCED_PARAMETER(hOverlay);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuSetDisplayPrivateDriverFormat(
    IN_CONST_HANDLE                                 hAdapter,
    IN_CONST_PDXGKARG_SETDISPLAYPRIVATEDRIVERFORMAT pSetDisplayPrivateDriverFormat
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetDisplayPrivateDriverFormat);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuQueryVidPnHWCapability(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_QUERYVIDPNHWCAPABILITY   pQueryVidPnHWCapability
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pQueryVidPnHWCapability != NULL) {
        /* No hardware display capabilities -- all software */
        pQueryVidPnHWCapability->VidPnHWCaps.DriverRotation = FALSE;
        pQueryVidPnHWCapability->VidPnHWCaps.DriverScaling = FALSE;
        pQueryVidPnHWCapability->VidPnHWCaps.DriverCloning = FALSE;
        pQueryVidPnHWCapability->VidPnHWCaps.DriverColorConvert = FALSE;
        pQueryVidPnHWCapability->VidPnHWCaps.DriverLinkedAdapaterOutput = FALSE;
        pQueryVidPnHWCapability->VidPnHWCaps.DriverRemoteDisplay = FALSE;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * Power / ACPI stubs
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSetPowerState(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN_ULONG                DeviceUid,
    IN_DEVICE_POWER_STATE   DevicePowerState,
    IN_POWER_ACTION         ActionType
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ActionType);

    KdPrint(("AmdGpuWddm: SetPowerState uid=%u state=%d\n", DeviceUid, DevicePowerState));

    /* Diagnostic marker */
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
            RtlInitUnicodeString(&ValName, L"SetPowerState_Uid");
            Val = DeviceUid;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));
            RtlInitUnicodeString(&ValName, L"SetPowerState_State");
            Val = (ULONG)DevicePowerState;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));
            ZwClose(hKey);
        }
    }

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuNotifyAcpiEvent(
    IN_CONST_PVOID      MiniportDeviceContext,
    IN_DXGK_EVENT_TYPE  EventType,
    IN_ULONG            Event,
    IN_PVOID            Argument,
    OUT_PULONG          AcpiFlags
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(EventType);
    UNREFERENCED_PARAMETER(Event);
    UNREFERENCED_PARAMETER(Argument);
    UNREFERENCED_PARAMETER(AcpiFlags);
    return STATUS_SUCCESS;
}

VOID
APIENTRY
AmdGpuSetPowerComponentFState(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN UINT                 ComponentIndex,
    IN UINT                 FState
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ComponentIndex);
    UNREFERENCED_PARAMETER(FState);
}

NTSTATUS
APIENTRY
AmdGpuPowerRuntimeControlRequest(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN LPCGUID              PowerControlCode,
    IN PVOID OPTIONAL       InBuffer,
    IN SIZE_T               InBufferSize,
    OUT PVOID OPTIONAL      OutBuffer,
    IN SIZE_T               OutBufferSize,
    OUT PSIZE_T OPTIONAL    BytesReturned
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(PowerControlCode);
    UNREFERENCED_PARAMETER(InBuffer);
    UNREFERENCED_PARAMETER(InBufferSize);
    UNREFERENCED_PARAMETER(OutBuffer);
    UNREFERENCED_PARAMETER(OutBufferSize);
    UNREFERENCED_PARAMETER(BytesReturned);
    return STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Misc stubs
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuDispatchIoRequest(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN_ULONG                VidPnSourceId,
    IN_PVIDEO_REQUEST_PACKET VideoRequestPacket
    )
{
    StubDbgRegWrite(L"DispatchIoRequest", VidPnSourceId);
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(VidPnSourceId);
    UNREFERENCED_PARAMETER(VideoRequestPacket);
    return STATUS_NOT_SUPPORTED;
}

VOID
APIENTRY
AmdGpuResetDevice(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
}

VOID
APIENTRY
AmdGpuUnload(
    VOID
    )
{
}

NTSTATUS
APIENTRY
AmdGpuQueryInterface(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN_PQUERY_INTERFACE     QueryInterface
    )
{
    StubDbgRegWrite(L"QueryInterface", 1);
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(QueryInterface);
    return STATUS_NOT_SUPPORTED;
}

VOID
APIENTRY
AmdGpuControlEtwLogging(
    IN_BOOLEAN  Enable,
    IN_ULONG    Flags,
    IN_UCHAR    Level
    )
{
    StubDbgRegWrite(L"ControlEtwLogging", (ULONG)Enable);
    UNREFERENCED_PARAMETER(Enable);
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Level);
}

NTSTATUS
APIENTRY
AmdGpuCollectDbgInfo(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_COLLECTDBGINFO        pCollectDbgInfo
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pCollectDbgInfo);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuNotifySurpriseRemoval(
    IN_CONST_PVOID                  MiniportDeviceContext,
    _In_ DXGK_SURPRISE_REMOVAL_TYPE RemovalType
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(RemovalType);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuCancelCommand(
    IN_CONST_HANDLE                 hAdapter,
    IN_CONST_PDXGKARG_CANCELCOMMAND pCancelCommand
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pCancelCommand);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuControlInterrupt2(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_DXGKARG_CONTROLINTERRUPT2  InterruptControl
    )
{
    StubDbgRegWrite(L"ControlInterrupt2", 1);
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(InterruptControl);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuCalibrateGpuClock(
    IN_CONST_HANDLE                         hAdapter,
    IN UINT32                               NodeOrdinal,
    IN UINT32                               EngineOrdinal,
    OUT_PDXGKARG_CALIBRATEGPUCLOCK          pClockCalibration
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(NodeOrdinal);
    UNREFERENCED_PARAMETER(EngineOrdinal);
    if (pClockCalibration != NULL) {
        LARGE_INTEGER PerfFreq;
        KeQueryPerformanceCounter(&PerfFreq);
        pClockCalibration->GpuFrequency = (ULONGLONG)PerfFreq.QuadPart;
        pClockCalibration->GpuClockCounter = 0;
        pClockCalibration->CpuClockCounter = 0;
    }
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuFormatHistoryBuffer(
    IN_CONST_HANDLE                             hContext,
    IN DXGKARG_FORMATHISTORYBUFFER*             pFormatData
    )
{
    UNREFERENCED_PARAMETER(hContext);
    UNREFERENCED_PARAMETER(pFormatData);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * Render stubs
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuRender(
    IN_CONST_HANDLE         hContext,
    INOUT_PDXGKARG_RENDER   pRender
    )
{
    UNREFERENCED_PARAMETER(hContext);
    UNREFERENCED_PARAMETER(pRender);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuRenderKm(
    IN_CONST_HANDLE         hContext,
    INOUT_PDXGKARG_RENDER   pRender
    )
{
    UNREFERENCED_PARAMETER(hContext);
    UNREFERENCED_PARAMETER(pRender);
    return STATUS_NOT_SUPPORTED;
}
