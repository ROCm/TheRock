/*
 * ddi_stubs.c - Pure DDI stubs for compute-only MCDM miniport
 *
 * These DDIs are required by the DRIVER_INITIALIZATION_DATA structure
 * but have no real implementation for a compute-only device. They
 * return STATUS_SUCCESS or STATUS_NOT_SUPPORTED as appropriate.
 */

#include "amdgpu_mcdm.h"

/* ======================================================================
 * Display / VidPn stubs — all return STATUS_NOT_SUPPORTED
 *
 * A compute-only MCDM device has no display output. These are required
 * function pointers but should never be called by dxgkrnl when
 * ComputeOnly=TRUE in DXGK_DRIVERCAPS.MiscCaps.
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
    return STATUS_NOT_SUPPORTED;
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
AmdGpuIsSupportedVidPn(
    IN_CONST_HANDLE                     hAdapter,
    INOUT_PDXGKARG_ISSUPPORTEDVIDPN     pIsSupportedVidPn
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    /* No VidPn configurations are supported — compute-only device */
    if (pIsSupportedVidPn != NULL)
        pIsSupportedVidPn->IsVidPnSupported = FALSE;
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuRecommendFunctionalVidPn(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_RECOMMENDFUNCTIONALVIDPN*    pRecommendFunctionalVidPn
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pRecommendFunctionalVidPn);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuEnumVidPnCofuncModality(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_ENUMVIDPNCOFUNCMODALITY*     pEnumCofuncModality
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pEnumCofuncModality);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuSetVidPnSourceAddress(
    IN_CONST_HANDLE                             hAdapter,
    IN_CONST_PDXGKARG_SETVIDPNSOURCEADDRESS     pSetVidPnSourceAddress
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetVidPnSourceAddress);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuSetVidPnSourceVisibility(
    IN_CONST_HANDLE                                 hAdapter,
    IN_CONST_PDXGKARG_SETVIDPNSOURCEVISIBILITY      pSetVidPnSourceVisibility
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetVidPnSourceVisibility);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuCommitVidPn(
    IN_CONST_HANDLE                         hAdapter,
    _In_ const DXGKARG_COMMITVIDPN*         pCommitVidPn
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pCommitVidPn);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuUpdateActiveVidPnPresentPath(
    IN_CONST_HANDLE                                     hAdapter,
    _In_ const DXGKARG_UPDATEACTIVEVIDPNPRESENTPATH*    pUpdateActiveVidPnPresentPath
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pUpdateActiveVidPnPresentPath);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuRecommendMonitorModes(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_RECOMMENDMONITORMODES*       pRecommendMonitorModes
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pRecommendMonitorModes);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuRecommendVidPnTopology(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_RECOMMENDVIDPNTOPOLOGY*      pRecommendVidPnTopology
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pRecommendVidPnTopology);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuGetScanLine(
    IN_CONST_HANDLE                 hAdapter,
    INOUT_PDXGKARG_GETSCANLINE      pGetScanLine
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pGetScanLine);
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
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(InterruptType);
    UNREFERENCED_PARAMETER(EnableInterrupt);
    return STATUS_NOT_SUPPORTED;
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
    UNREFERENCED_PARAMETER(pQueryVidPnHWCapability);
    return STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * Child / connector stubs — compute-only has no children
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryChildRelations(
    IN_CONST_PVOID                          MiniportDeviceContext,
    INOUT_PDXGK_CHILD_DESCRIPTOR            ChildRelations,
    _In_ ULONG                              ChildRelationsSize
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ChildRelations);
    UNREFERENCED_PARAMETER(ChildRelationsSize);
    /* No child devices (no monitor outputs) */
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuQueryChildStatus(
    IN_CONST_PVOID                  MiniportDeviceContext,
    INOUT_PDXGK_CHILD_STATUS        ChildStatus,
    IN_BOOLEAN                      NonDestructiveOnly
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ChildStatus);
    UNREFERENCED_PARAMETER(NonDestructiveOnly);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuQueryDeviceDescriptor(
    IN_CONST_PVOID                          MiniportDeviceContext,
    IN_ULONG                                ChildUid,
    INOUT_PDXGK_DEVICE_DESCRIPTOR           pDeviceDescriptor
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ChildUid);
    UNREFERENCED_PARAMETER(pDeviceDescriptor);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
APIENTRY
AmdGpuGetChildContainerId(
    IN_CONST_PVOID                          MiniportDeviceContext,
    IN_ULONG                                ChildUid,
    _Inout_ PDXGK_CHILD_CONTAINER_ID       pContainerId
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ChildUid);
    UNREFERENCED_PARAMETER(pContainerId);
    return STATUS_NOT_SUPPORTED;
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
    UNREFERENCED_PARAMETER(DeviceUid);
    UNREFERENCED_PARAMETER(DevicePowerState);
    UNREFERENCED_PARAMETER(ActionType);
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
    /* Minimal reset — full implementation in ddi_tdr.c:ResetFromTimeout */
}

VOID
APIENTRY
AmdGpuUnload(
    VOID
    )
{
    /* Nothing to clean up at the global driver level */
}

NTSTATUS
APIENTRY
AmdGpuQueryInterface(
    IN_CONST_PVOID          MiniportDeviceContext,
    IN_PQUERY_INTERFACE     QueryInterface
    )
{
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
AmdGpuStopDeviceAndReleasePostDisplayOwnership(
    _In_ PVOID                          MiniportDeviceContext,
    _In_ D3DDDI_VIDEO_PRESENT_TARGET_ID TargetId,
    _Out_ PDXGK_DISPLAY_INFORMATION     DisplayInfo
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(TargetId);
    UNREFERENCED_PARAMETER(DisplayInfo);
    return STATUS_NOT_SUPPORTED;
}

NTSTATUS
AmdGpuSystemDisplayEnable(
    _In_ PVOID                              MiniportDeviceContext,
    _In_ D3DDDI_VIDEO_PRESENT_TARGET_ID     TargetId,
    _In_ PDXGKARG_SYSTEM_DISPLAY_ENABLE_FLAGS Flags,
    _Out_ UINT*                             Width,
    _Out_ UINT*                             Height,
    _Out_ D3DDDIFORMAT*                     ColorFormat
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(TargetId);
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Width);
    UNREFERENCED_PARAMETER(Height);
    UNREFERENCED_PARAMETER(ColorFormat);
    return STATUS_NOT_SUPPORTED;
}

VOID
AmdGpuSystemDisplayWrite(
    _In_ PVOID                      MiniportDeviceContext,
    _In_ PVOID                      Source,
    _In_ UINT                       SourceWidth,
    _In_ UINT                       SourceHeight,
    _In_ UINT                       SourceStride,
    _In_ UINT                       PositionX,
    _In_ UINT                       PositionY
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(Source);
    UNREFERENCED_PARAMETER(SourceWidth);
    UNREFERENCED_PARAMETER(SourceHeight);
    UNREFERENCED_PARAMETER(SourceStride);
    UNREFERENCED_PARAMETER(PositionX);
    UNREFERENCED_PARAMETER(PositionY);
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
 * Render stubs — compute-only does not use the command buffer path
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

NTSTATUS
APIENTRY
AmdGpuPresent(
    IN_CONST_HANDLE         hContext,
    INOUT_PDXGKARG_PRESENT  pPresent
    )
{
    UNREFERENCED_PARAMETER(hContext);
    UNREFERENCED_PARAMETER(pPresent);
    return STATUS_NOT_SUPPORTED;
}
