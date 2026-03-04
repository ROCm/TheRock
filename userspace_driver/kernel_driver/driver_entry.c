/*
 * driver_entry.c - DriverEntry for AMD GPU MCDM miniport driver
 *
 * Entry point: DriverEntry → DxgkInitialize
 * Populates DRIVER_INITIALIZATION_DATA with all required DDI callbacks.
 */

#include "amdgpu_mcdm.h"

/* Forward-declare DriverEntry before alloc_text pragma */
DRIVER_INITIALIZE DriverEntry;

#ifdef ALLOC_PRAGMA
#pragma alloc_text(INIT, DriverEntry)
#endif

/* ======================================================================
 * Forward declarations for all DDI callbacks
 *
 * These are implemented across ddi_device.c, ddi_escape.c, ddi_memory.c,
 * ddi_scheduling.c, ddi_interrupt.c, ddi_query.c, ddi_tdr.c, ddi_stubs.c.
 * ====================================================================== */

/* ddi_device.c */
DXGKDDI_ADD_DEVICE          AmdGpuAddDevice;
DXGKDDI_START_DEVICE        AmdGpuStartDevice;
DXGKDDI_STOP_DEVICE         AmdGpuStopDevice;
DXGKDDI_REMOVE_DEVICE       AmdGpuRemoveDevice;
DXGKDDI_CREATEDEVICE        AmdGpuCreateDevice;
DXGKDDI_DESTROYDEVICE       AmdGpuDestroyDevice;

/* ddi_escape.c */
DXGKDDI_ESCAPE              AmdGpuEscape;

/* ddi_memory.c */
DXGKDDI_CREATEALLOCATION    AmdGpuCreateAllocation;
DXGKDDI_DESTROYALLOCATION   AmdGpuDestroyAllocation;
DXGKDDI_DESCRIBEALLOCATION  AmdGpuDescribeAllocation;
DXGKDDI_GETSTANDARDALLOCATIONDRIVERDATA AmdGpuGetStandardAllocationDriverData;
DXGKDDI_OPENALLOCATIONINFO  AmdGpuOpenAllocation;
DXGKDDI_CLOSEALLOCATION     AmdGpuCloseAllocation;
DXGKDDI_BUILDPAGINGBUFFER   AmdGpuBuildPagingBuffer;

/* ddi_scheduling.c */
DXGKDDI_SUBMITCOMMAND       AmdGpuSubmitCommand;
DXGKDDI_PREEMPTCOMMAND      AmdGpuPreemptCommand;
DXGKDDI_PATCH               AmdGpuPatch;
DXGKDDI_PRESENT             AmdGpuPresent;
DXGKDDI_RENDER              AmdGpuRender;

/* ddi_interrupt.c */
DXGKDDI_INTERRUPT_ROUTINE   AmdGpuInterruptRoutine;
DXGKDDI_DPC_ROUTINE         AmdGpuDpcRoutine;

/* ddi_query.c */
DXGKDDI_QUERYADAPTERINFO    AmdGpuQueryAdapterInfo;
DXGKDDI_QUERYCURRENTFENCE   AmdGpuQueryCurrentFence;

/* ddi_tdr.c */
DXGKDDI_RESETFROMTIMEOUT    AmdGpuResetFromTimeout;
DXGKDDI_RESTARTFROMTIMEOUT  AmdGpuRestartFromTimeout;
DXGKDDI_RESETENGINE         AmdGpuResetEngine;

/* ddi_stubs.c */
DXGKDDI_DISPATCH_IO_REQUEST AmdGpuDispatchIoRequest;
DXGKDDI_QUERY_CHILD_RELATIONS AmdGpuQueryChildRelations;
DXGKDDI_QUERY_CHILD_STATUS AmdGpuQueryChildStatus;
DXGKDDI_QUERY_DEVICE_DESCRIPTOR AmdGpuQueryDeviceDescriptor;
DXGKDDI_SET_POWER_STATE     AmdGpuSetPowerState;
DXGKDDI_NOTIFY_ACPI_EVENT   AmdGpuNotifyAcpiEvent;
DXGKDDI_RESET_DEVICE        AmdGpuResetDevice;
DXGKDDI_UNLOAD              AmdGpuUnload;
DXGKDDI_QUERY_INTERFACE     AmdGpuQueryInterface;
DXGKDDI_CONTROL_ETW_LOGGING AmdGpuControlEtwLogging;
DXGKDDI_COLLECTDBGINFO      AmdGpuCollectDbgInfo;
DXGKDDI_SETPALETTE          AmdGpuSetPalette;
DXGKDDI_SETPOINTERPOSITION  AmdGpuSetPointerPosition;
DXGKDDI_SETPOINTERSHAPE     AmdGpuSetPointerShape;
DXGKDDI_ISSUPPORTEDVIDPN    AmdGpuIsSupportedVidPn;
DXGKDDI_RECOMMENDFUNCTIONALVIDPN AmdGpuRecommendFunctionalVidPn;
DXGKDDI_ENUMVIDPNCOFUNCMODALITY AmdGpuEnumVidPnCofuncModality;
DXGKDDI_SETVIDPNSOURCEADDRESS AmdGpuSetVidPnSourceAddress;
DXGKDDI_SETVIDPNSOURCEVISIBILITY AmdGpuSetVidPnSourceVisibility;
DXGKDDI_COMMITVIDPN         AmdGpuCommitVidPn;
DXGKDDI_UPDATEACTIVEVIDPNPRESENTPATH AmdGpuUpdateActiveVidPnPresentPath;
DXGKDDI_RECOMMENDMONITORMODES AmdGpuRecommendMonitorModes;
DXGKDDI_GETSCANLINE         AmdGpuGetScanLine;
DXGKDDI_STOPCAPTURE         AmdGpuStopCapture;
DXGKDDI_CONTROLINTERRUPT    AmdGpuControlInterrupt;
DXGKDDI_CREATEOVERLAY       AmdGpuCreateOverlay;
DXGKDDI_UPDATEOVERLAY       AmdGpuUpdateOverlay;
DXGKDDI_FLIPOVERLAY         AmdGpuFlipOverlay;
DXGKDDI_DESTROYOVERLAY      AmdGpuDestroyOverlay;
DXGKDDI_CREATECONTEXT       AmdGpuCreateContext;
DXGKDDI_DESTROYCONTEXT      AmdGpuDestroyContext;
DXGKDDI_SETDISPLAYPRIVATEDRIVERFORMAT AmdGpuSetDisplayPrivateDriverFormat;
DXGKDDI_RECOMMENDVIDPNTOPOLOGY AmdGpuRecommendVidPnTopology;

/* WDDM 1.1+ */
DXGKDDI_RENDER              AmdGpuRenderKm;
DXGKDDI_QUERYVIDPNHWCAPABILITY AmdGpuQueryVidPnHWCapability;

/* WDDM 1.2+ (Win8) */
DXGKDDISETPOWERCOMPONENTFSTATE AmdGpuSetPowerComponentFState;
DXGKDDI_QUERYDEPENDENTENGINEGROUP AmdGpuQueryDependentEngineGroup;
DXGKDDI_QUERYENGINESTATUS   AmdGpuQueryEngineStatus;
DXGKDDI_STOP_DEVICE_AND_RELEASE_POST_DISPLAY_OWNERSHIP AmdGpuStopDeviceAndReleasePostDisplayOwnership;
DXGKDDI_SYSTEM_DISPLAY_ENABLE AmdGpuSystemDisplayEnable;
DXGKDDI_SYSTEM_DISPLAY_WRITE AmdGpuSystemDisplayWrite;
DXGKDDI_CANCELCOMMAND       AmdGpuCancelCommand;
DXGKDDI_GET_CHILD_CONTAINER_ID AmdGpuGetChildContainerId;
DXGKDDIPOWERRUNTIMECONTROLREQUEST AmdGpuPowerRuntimeControlRequest;
DXGKDDI_NOTIFY_SURPRISE_REMOVAL AmdGpuNotifySurpriseRemoval;

/* WDDM 1.3+ */
DXGKDDI_GETNODEMETADATA     AmdGpuGetNodeMetadata;
DXGKDDI_CONTROLINTERRUPT2   AmdGpuControlInterrupt2;
DXGKDDI_CALIBRATEGPUCLOCK   AmdGpuCalibrateGpuClock;
DXGKDDI_FORMATHISTORYBUFFER AmdGpuFormatHistoryBuffer;

/* WDDM 2.0+ */
DXGKDDI_SUBMITCOMMANDVIRTUAL AmdGpuSubmitCommandVirtual;
DXGKDDI_SETROOTPAGETABLE    AmdGpuSetRootPageTable;
DXGKDDI_GETROOTPAGETABLESIZE AmdGpuGetRootPageTableSize;
DXGKDDI_MAPCPUHOSTAPERTURE  AmdGpuMapCpuHostAperture;
DXGKDDI_UNMAPCPUHOSTAPERTURE AmdGpuUnmapCpuHostAperture;
DXGKDDI_CREATEPROCESS       AmdGpuCreateProcess;
DXGKDDI_DESTROYPROCESS      AmdGpuDestroyProcess;

/* ======================================================================
 * DriverEntry
 * ====================================================================== */

NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT     DriverObject,
    _In_ PUNICODE_STRING    RegistryPath
    )
{
    DRIVER_INITIALIZATION_DATA  DriverInitData;

    RtlZeroMemory(&DriverInitData, sizeof(DriverInitData));

    /*
     * Set version to WDDM 2.6 — the minimum that supports ComputeOnly bit
     * in DXGK_DRIVERCAPS.MiscCaps.
     */
    DriverInitData.Version = DXGKDDI_INTERFACE_VERSION_WDDM2_6;

    /* --- Core PnP / Lifecycle --- */
    DriverInitData.DxgkDdiAddDevice                     = AmdGpuAddDevice;
    DriverInitData.DxgkDdiStartDevice                   = AmdGpuStartDevice;
    DriverInitData.DxgkDdiStopDevice                    = AmdGpuStopDevice;
    DriverInitData.DxgkDdiRemoveDevice                  = AmdGpuRemoveDevice;
    DriverInitData.DxgkDdiDispatchIoRequest              = AmdGpuDispatchIoRequest;
    DriverInitData.DxgkDdiInterruptRoutine               = AmdGpuInterruptRoutine;
    DriverInitData.DxgkDdiDpcRoutine                     = AmdGpuDpcRoutine;
    DriverInitData.DxgkDdiQueryChildRelations            = AmdGpuQueryChildRelations;
    DriverInitData.DxgkDdiQueryChildStatus               = AmdGpuQueryChildStatus;
    DriverInitData.DxgkDdiQueryDeviceDescriptor          = AmdGpuQueryDeviceDescriptor;
    DriverInitData.DxgkDdiSetPowerState                  = AmdGpuSetPowerState;
    DriverInitData.DxgkDdiNotifyAcpiEvent                = AmdGpuNotifyAcpiEvent;
    DriverInitData.DxgkDdiResetDevice                    = AmdGpuResetDevice;
    DriverInitData.DxgkDdiUnload                         = AmdGpuUnload;
    DriverInitData.DxgkDdiQueryInterface                 = AmdGpuQueryInterface;
    DriverInitData.DxgkDdiControlEtwLogging              = AmdGpuControlEtwLogging;

    /* --- Adapter / Render --- */
    DriverInitData.DxgkDdiQueryAdapterInfo               = AmdGpuQueryAdapterInfo;
    DriverInitData.DxgkDdiCreateDevice                   = AmdGpuCreateDevice;
    DriverInitData.DxgkDdiCreateAllocation               = AmdGpuCreateAllocation;
    DriverInitData.DxgkDdiDestroyAllocation              = AmdGpuDestroyAllocation;
    DriverInitData.DxgkDdiDescribeAllocation             = AmdGpuDescribeAllocation;
    DriverInitData.DxgkDdiGetStandardAllocationDriverData = AmdGpuGetStandardAllocationDriverData;
    DriverInitData.DxgkDdiAcquireSwizzlingRange          = NULL;
    DriverInitData.DxgkDdiReleaseSwizzlingRange          = NULL;
    DriverInitData.DxgkDdiPatch                          = AmdGpuPatch;
    DriverInitData.DxgkDdiSubmitCommand                  = AmdGpuSubmitCommand;
    DriverInitData.DxgkDdiPreemptCommand                 = AmdGpuPreemptCommand;
    DriverInitData.DxgkDdiBuildPagingBuffer              = AmdGpuBuildPagingBuffer;

    /* --- Display (stubs for compute-only) --- */
    DriverInitData.DxgkDdiSetPalette                     = AmdGpuSetPalette;
    DriverInitData.DxgkDdiSetPointerPosition             = AmdGpuSetPointerPosition;
    DriverInitData.DxgkDdiSetPointerShape                = AmdGpuSetPointerShape;
    DriverInitData.DxgkDdiResetFromTimeout                = AmdGpuResetFromTimeout;
    DriverInitData.DxgkDdiRestartFromTimeout              = AmdGpuRestartFromTimeout;
    DriverInitData.DxgkDdiEscape                         = AmdGpuEscape;
    DriverInitData.DxgkDdiCollectDbgInfo                 = AmdGpuCollectDbgInfo;
    DriverInitData.DxgkDdiQueryCurrentFence              = AmdGpuQueryCurrentFence;

    /* --- VidPn (stubs) --- */
    DriverInitData.DxgkDdiIsSupportedVidPn               = AmdGpuIsSupportedVidPn;
    DriverInitData.DxgkDdiRecommendFunctionalVidPn       = AmdGpuRecommendFunctionalVidPn;
    DriverInitData.DxgkDdiEnumVidPnCofuncModality        = AmdGpuEnumVidPnCofuncModality;
    DriverInitData.DxgkDdiSetVidPnSourceAddress          = AmdGpuSetVidPnSourceAddress;
    DriverInitData.DxgkDdiSetVidPnSourceVisibility       = AmdGpuSetVidPnSourceVisibility;
    DriverInitData.DxgkDdiCommitVidPn                    = AmdGpuCommitVidPn;
    DriverInitData.DxgkDdiUpdateActiveVidPnPresentPath   = AmdGpuUpdateActiveVidPnPresentPath;
    DriverInitData.DxgkDdiRecommendMonitorModes          = AmdGpuRecommendMonitorModes;
    DriverInitData.DxgkDdiRecommendVidPnTopology         = AmdGpuRecommendVidPnTopology;
    DriverInitData.DxgkDdiGetScanLine                    = AmdGpuGetScanLine;
    DriverInitData.DxgkDdiStopCapture                    = AmdGpuStopCapture;
    DriverInitData.DxgkDdiControlInterrupt               = AmdGpuControlInterrupt;
    DriverInitData.DxgkDdiCreateOverlay                  = AmdGpuCreateOverlay;

    /* --- Device / Render --- */
    DriverInitData.DxgkDdiDestroyDevice                  = AmdGpuDestroyDevice;
    DriverInitData.DxgkDdiOpenAllocation                 = AmdGpuOpenAllocation;
    DriverInitData.DxgkDdiCloseAllocation                = AmdGpuCloseAllocation;
    DriverInitData.DxgkDdiRender                         = AmdGpuRender;
    DriverInitData.DxgkDdiPresent                        = AmdGpuPresent;

    /* --- Overlay --- */
    DriverInitData.DxgkDdiUpdateOverlay                  = AmdGpuUpdateOverlay;
    DriverInitData.DxgkDdiFlipOverlay                    = AmdGpuFlipOverlay;
    DriverInitData.DxgkDdiDestroyOverlay                 = AmdGpuDestroyOverlay;

    /* --- Context --- */
    DriverInitData.DxgkDdiCreateContext                  = AmdGpuCreateContext;
    DriverInitData.DxgkDdiDestroyContext                 = AmdGpuDestroyContext;

    /* --- Link / Display format --- */
    DriverInitData.DxgkDdiLinkDevice                     = NULL;
    DriverInitData.DxgkDdiSetDisplayPrivateDriverFormat   = AmdGpuSetDisplayPrivateDriverFormat;

    /* --- WDDM 1.1 (Win7+) --- */
    DriverInitData.DxgkDdiRenderKm                       = AmdGpuRenderKm;
    DriverInitData.DxgkDdiQueryVidPnHWCapability         = AmdGpuQueryVidPnHWCapability;

    /* --- WDDM 1.2 (Win8+) --- */
    DriverInitData.DxgkDdiSetPowerComponentFState        = AmdGpuSetPowerComponentFState;
    DriverInitData.DxgkDdiQueryDependentEngineGroup      = AmdGpuQueryDependentEngineGroup;
    DriverInitData.DxgkDdiQueryEngineStatus              = AmdGpuQueryEngineStatus;
    DriverInitData.DxgkDdiResetEngine                    = AmdGpuResetEngine;
    DriverInitData.DxgkDdiStopDeviceAndReleasePostDisplayOwnership = AmdGpuStopDeviceAndReleasePostDisplayOwnership;
    DriverInitData.DxgkDdiSystemDisplayEnable            = AmdGpuSystemDisplayEnable;
    DriverInitData.DxgkDdiSystemDisplayWrite             = AmdGpuSystemDisplayWrite;
    DriverInitData.DxgkDdiCancelCommand                  = AmdGpuCancelCommand;
    DriverInitData.DxgkDdiGetChildContainerId            = AmdGpuGetChildContainerId;
    DriverInitData.DxgkDdiPowerRuntimeControlRequest     = AmdGpuPowerRuntimeControlRequest;
    DriverInitData.DxgkDdiSetVidPnSourceAddressWithMultiPlaneOverlay = NULL;
    DriverInitData.DxgkDdiNotifySurpriseRemoval          = AmdGpuNotifySurpriseRemoval;

    /* --- WDDM 1.3+ --- */
    DriverInitData.DxgkDdiGetNodeMetadata                = AmdGpuGetNodeMetadata;
    DriverInitData.DxgkDdiSetPowerPState                 = NULL;
    DriverInitData.DxgkDdiControlInterrupt2              = AmdGpuControlInterrupt2;
    DriverInitData.DxgkDdiCheckMultiPlaneOverlaySupport  = NULL;
    DriverInitData.DxgkDdiCalibrateGpuClock              = AmdGpuCalibrateGpuClock;
    DriverInitData.DxgkDdiFormatHistoryBuffer            = AmdGpuFormatHistoryBuffer;

    /* --- WDDM 2.0+ --- */
    DriverInitData.DxgkDdiRenderGdi                      = NULL;
    DriverInitData.DxgkDdiSubmitCommandVirtual           = AmdGpuSubmitCommandVirtual;
    DriverInitData.DxgkDdiSetRootPageTable               = AmdGpuSetRootPageTable;
    DriverInitData.DxgkDdiGetRootPageTableSize           = AmdGpuGetRootPageTableSize;
    DriverInitData.DxgkDdiMapCpuHostAperture             = AmdGpuMapCpuHostAperture;
    DriverInitData.DxgkDdiUnmapCpuHostAperture           = AmdGpuUnmapCpuHostAperture;
    DriverInitData.DxgkDdiCheckMultiPlaneOverlaySupport2 = NULL;
    DriverInitData.DxgkDdiCreateProcess                  = AmdGpuCreateProcess;
    DriverInitData.DxgkDdiDestroyProcess                 = AmdGpuDestroyProcess;
    DriverInitData.DxgkDdiSetVidPnSourceAddressWithMultiPlaneOverlay2 = NULL;
    DriverInitData.DxgkDdiPowerRuntimeSetDeviceHandle    = NULL;
    DriverInitData.DxgkDdiSetStablePowerState            = NULL;
    DriverInitData.DxgkDdiSetVideoProtectedRegion        = NULL;

    /* --- WDDM 2.6+ (needed for ComputeOnly bit) --- */
    DriverInitData.DxgkDdiSaveMemoryForHotUpdate         = NULL;
    DriverInitData.DxgkDdiRestoreMemoryForHotUpdate      = NULL;
    DriverInitData.DxgkDdiCollectDiagnosticInfo          = NULL;

    return DxgkInitialize(DriverObject, RegistryPath, &DriverInitData);
}
