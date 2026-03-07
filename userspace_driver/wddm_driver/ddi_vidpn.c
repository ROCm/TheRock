/*
 * ddi_vidpn.c - VidPn DDIs for WDDM display miniport
 *
 * All VidPn operations support exactly one mode: the POST framebuffer
 * dimensions and format. Single path: source 0 -> target 0.
 *
 * Key VidPn rules:
 * - Every pfnAcquire* needs matching pfnRelease*
 * - After pfnAddMode, mode is owned by mode set (don't release it)
 * - After pfnAssignSourceModeSet, mode set is owned by VidPn (don't release it)
 * - EnumVidPnCofuncModality must skip the pivot mode set
 * - Must create NEW mode sets, not modify acquired ones
 */

#include "amdgpu_wddm.h"

/* Registry diagnostic helper */
static void
VidPnDbgRegWrite(const WCHAR *Name, ULONG Value)
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
 * Helper: fill in our single supported source mode from POST display
 * ====================================================================== */

static void
FillSourceMode(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ D3DKMDT_VIDPN_SOURCE_MODE *pMode
    )
{
    /* Preserve Id assigned by pfnCreateNewModeInfo -- dxgkrnl owns it */
    D3DKMDT_VIDEO_PRESENT_SOURCE_MODE_ID SavedId = pMode->Id;
    RtlZeroMemory(pMode, sizeof(*pMode));
    pMode->Id = SavedId;
    pMode->Type = D3DKMDT_RMT_GRAPHICS;
    pMode->Format.Graphics.PrimSurfSize.cx = pAdapter->PostDisplay.Width;
    pMode->Format.Graphics.PrimSurfSize.cy = pAdapter->PostDisplay.Height;
    pMode->Format.Graphics.VisibleRegionSize.cx = pAdapter->PostDisplay.Width;
    pMode->Format.Graphics.VisibleRegionSize.cy = pAdapter->PostDisplay.Height;
    pMode->Format.Graphics.Stride = pAdapter->PostDisplay.Pitch;
    pMode->Format.Graphics.PixelFormat = D3DDDIFMT_X8R8G8B8;
    pMode->Format.Graphics.ColorBasis = D3DKMDT_CB_SRGB;
    pMode->Format.Graphics.PixelValueAccessMode = D3DKMDT_PVAM_DIRECT;
}

static void
FillTargetMode(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _Inout_ D3DKMDT_VIDPN_TARGET_MODE *pMode
    )
{
    /* Preserve Id assigned by pfnCreateNewModeInfo -- dxgkrnl owns it */
    D3DKMDT_VIDEO_PRESENT_TARGET_MODE_ID SavedId = pMode->Id;
    RtlZeroMemory(pMode, sizeof(*pMode));
    pMode->Id = SavedId;
    pMode->VideoSignalInfo.VideoStandard = D3DKMDT_VSS_OTHER;
    pMode->VideoSignalInfo.TotalSize.cx = pAdapter->PostDisplay.Width;
    pMode->VideoSignalInfo.TotalSize.cy = pAdapter->PostDisplay.Height;
    pMode->VideoSignalInfo.ActiveSize.cx = pAdapter->PostDisplay.Width;
    pMode->VideoSignalInfo.ActiveSize.cy = pAdapter->PostDisplay.Height;
    pMode->VideoSignalInfo.VSyncFreq.Numerator = 60000;
    pMode->VideoSignalInfo.VSyncFreq.Denominator = 1000;
    pMode->VideoSignalInfo.HSyncFreq.Numerator = 48000;
    pMode->VideoSignalInfo.HSyncFreq.Denominator = 1;
    pMode->VideoSignalInfo.PixelRate = (ULONGLONG)pAdapter->PostDisplay.Width *
        pAdapter->PostDisplay.Height * 60;
    pMode->VideoSignalInfo.ScanLineOrdering = D3DDDI_VSSLO_PROGRESSIVE;
}

/* ======================================================================
 * IsSupportedVidPn
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuIsSupportedVidPn(
    IN_CONST_HANDLE                     hAdapter,
    INOUT_PDXGKARG_ISSUPPORTEDVIDPN     pIsSupportedVidPn
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    NTSTATUS Status;
    CONST DXGK_VIDPN_INTERFACE *pVidPnInterface;
    D3DKMDT_HVIDPN hVidPn;
    CONST DXGK_VIDPNTOPOLOGY_INTERFACE *pTopology;
    D3DKMDT_HVIDPNTOPOLOGY hTopology;
    SIZE_T NumPaths;

    VidPnDbgRegWrite(L"IsSupportedVidPn", 1);

    if (pIsSupportedVidPn == NULL)
        return STATUS_INVALID_PARAMETER;

    hVidPn = pIsSupportedVidPn->hDesiredVidPn;

    /* NULL/empty VidPn is always supported */
    if (hVidPn == NULL) {
        pIsSupportedVidPn->IsVidPnSupported = TRUE;
        return STATUS_SUCCESS;
    }

    Status = pAdapter->DxgkInterface.DxgkCbQueryVidPnInterface(
        hVidPn, DXGK_VIDPN_INTERFACE_VERSION_V1, &pVidPnInterface);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pVidPnInterface->pfnGetTopology(hVidPn, &hTopology, &pTopology);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pTopology->pfnGetNumPaths(hTopology, &NumPaths);
    if (!NT_SUCCESS(Status))
        return Status;

    /* Accept empty topology or single path source 0 -> target 0 */
    if (NumPaths == 0) {
        pIsSupportedVidPn->IsVidPnSupported = TRUE;
    } else if (NumPaths == 1) {
        /* We support any single-path VidPn since we only have 1 source/target */
        pIsSupportedVidPn->IsVidPnSupported = TRUE;
    } else {
        pIsSupportedVidPn->IsVidPnSupported = FALSE;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * RecommendFunctionalVidPn
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuRecommendFunctionalVidPn(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_RECOMMENDFUNCTIONALVIDPN*    pRecommendFunctionalVidPn
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    NTSTATUS Status;
    CONST DXGK_VIDPN_INTERFACE *pVidPnInterface;
    D3DKMDT_HVIDPN hVidPn;
    CONST DXGK_VIDPNTOPOLOGY_INTERFACE *pTopology;
    D3DKMDT_HVIDPNTOPOLOGY hTopology;
    D3DKMDT_VIDPN_PRESENT_PATH *pPath;
    D3DKMDT_HVIDPNSOURCEMODESET hSourceModeSet;
    CONST DXGK_VIDPNSOURCEMODESET_INTERFACE *pSourceModeSetInterface;
    D3DKMDT_VIDPN_SOURCE_MODE *pSourceMode;
    D3DKMDT_HVIDPNTARGETMODESET hTargetModeSet;
    CONST DXGK_VIDPNTARGETMODESET_INTERFACE *pTargetModeSetInterface;
    D3DKMDT_VIDPN_TARGET_MODE *pTargetMode;

    VidPnDbgRegWrite(L"RecommendFuncVidPn", 1);

    hVidPn = pRecommendFunctionalVidPn->hRecommendedFunctionalVidPn;

    Status = pAdapter->DxgkInterface.DxgkCbQueryVidPnInterface(
        hVidPn, DXGK_VIDPN_INTERFACE_VERSION_V1, &pVidPnInterface);
    if (!NT_SUCCESS(Status))
        return Status;

    /* Get topology and add path: source 0 -> target 0 */
    Status = pVidPnInterface->pfnGetTopology(hVidPn, &hTopology, &pTopology);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pTopology->pfnCreateNewPathInfo(hTopology, &pPath);
    if (!NT_SUCCESS(Status))
        return Status;

    pPath->VidPnSourceId = 0;
    pPath->VidPnTargetId = 0;
    pPath->ImportanceOrdinal = D3DKMDT_VPPI_PRIMARY;
    pPath->ContentTransformation.Scaling = D3DKMDT_VPPS_IDENTITY;
    pPath->ContentTransformation.ScalingSupport.Identity = TRUE;
    pPath->ContentTransformation.Rotation = D3DKMDT_VPPR_IDENTITY;
    pPath->ContentTransformation.RotationSupport.Identity = TRUE;
    pPath->CopyProtection.CopyProtectionType = D3DKMDT_VPPMT_NOPROTECTION;
    pPath->GammaRamp.Type = D3DDDI_GAMMARAMP_DEFAULT;

    Status = pTopology->pfnAddPath(hTopology, pPath);
    if (!NT_SUCCESS(Status)) {
        pTopology->pfnReleasePathInfo(hTopology, pPath);
        return Status;
    }
    /* pPath now owned by topology */

    /* Create and pin source mode set */
    Status = pVidPnInterface->pfnCreateNewSourceModeSet(
        hVidPn, 0, &hSourceModeSet, &pSourceModeSetInterface);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pSourceModeSetInterface->pfnCreateNewModeInfo(
        hSourceModeSet, &pSourceMode);
    if (!NT_SUCCESS(Status)) {
        pVidPnInterface->pfnReleaseSourceModeSet(hVidPn, hSourceModeSet);
        return Status;
    }

    FillSourceMode(pAdapter, pSourceMode);

    Status = pSourceModeSetInterface->pfnAddMode(hSourceModeSet, pSourceMode);
    if (!NT_SUCCESS(Status)) {
        pSourceModeSetInterface->pfnReleaseModeInfo(hSourceModeSet, pSourceMode);
        pVidPnInterface->pfnReleaseSourceModeSet(hVidPn, hSourceModeSet);
        return Status;
    }
    /* pSourceMode now owned by hSourceModeSet */

    Status = pSourceModeSetInterface->pfnPinMode(hSourceModeSet, pSourceMode->Id);
    if (!NT_SUCCESS(Status)) {
        pVidPnInterface->pfnReleaseSourceModeSet(hVidPn, hSourceModeSet);
        return Status;
    }

    Status = pVidPnInterface->pfnAssignSourceModeSet(hVidPn, 0, hSourceModeSet);
    if (!NT_SUCCESS(Status)) {
        pVidPnInterface->pfnReleaseSourceModeSet(hVidPn, hSourceModeSet);
        return Status;
    }
    /* hSourceModeSet now owned by hVidPn */

    /* Create and pin target mode set */
    Status = pVidPnInterface->pfnCreateNewTargetModeSet(
        hVidPn, 0, &hTargetModeSet, &pTargetModeSetInterface);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pTargetModeSetInterface->pfnCreateNewModeInfo(
        hTargetModeSet, &pTargetMode);
    if (!NT_SUCCESS(Status)) {
        pVidPnInterface->pfnReleaseTargetModeSet(hVidPn, hTargetModeSet);
        return Status;
    }

    FillTargetMode(pAdapter, pTargetMode);

    Status = pTargetModeSetInterface->pfnAddMode(hTargetModeSet, pTargetMode);
    if (!NT_SUCCESS(Status)) {
        pTargetModeSetInterface->pfnReleaseModeInfo(hTargetModeSet, pTargetMode);
        pVidPnInterface->pfnReleaseTargetModeSet(hVidPn, hTargetModeSet);
        return Status;
    }

    Status = pTargetModeSetInterface->pfnPinMode(hTargetModeSet, pTargetMode->Id);
    if (!NT_SUCCESS(Status)) {
        pVidPnInterface->pfnReleaseTargetModeSet(hVidPn, hTargetModeSet);
        return Status;
    }

    Status = pVidPnInterface->pfnAssignTargetModeSet(hVidPn, 0, hTargetModeSet);
    if (!NT_SUCCESS(Status)) {
        pVidPnInterface->pfnReleaseTargetModeSet(hVidPn, hTargetModeSet);
        return Status;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * EnumVidPnCofuncModality
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuEnumVidPnCofuncModality(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_ENUMVIDPNCOFUNCMODALITY*     pEnumCofuncModality
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    NTSTATUS Status;
    CONST DXGK_VIDPN_INTERFACE *pVidPnInterface;
    D3DKMDT_HVIDPN hVidPn;
    CONST DXGK_VIDPNTOPOLOGY_INTERFACE *pTopology;
    D3DKMDT_HVIDPNTOPOLOGY hTopology;
    CONST D3DKMDT_VIDPN_PRESENT_PATH *pPath;
    SIZE_T NumPaths;

    VidPnDbgRegWrite(L"EnumCofuncModality", 1);

    hVidPn = pEnumCofuncModality->hConstrainingVidPn;

    Status = pAdapter->DxgkInterface.DxgkCbQueryVidPnInterface(
        hVidPn, DXGK_VIDPN_INTERFACE_VERSION_V1, &pVidPnInterface);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pVidPnInterface->pfnGetTopology(hVidPn, &hTopology, &pTopology);
    if (!NT_SUCCESS(Status))
        return Status;

    Status = pTopology->pfnGetNumPaths(hTopology, &NumPaths);
    if (!NT_SUCCESS(Status))
        return Status;

    /* Empty topology -- nothing to enumerate */
    if (NumPaths == 0)
        return STATUS_SUCCESS;

    /* Iterate paths (we only expect 1) */
    Status = pTopology->pfnAcquireFirstPathInfo(hTopology, &pPath);
    while (NT_SUCCESS(Status)) {
        D3DDDI_VIDEO_PRESENT_SOURCE_ID SourceId = pPath->VidPnSourceId;
        D3DDDI_VIDEO_PRESENT_TARGET_ID TargetId = pPath->VidPnTargetId;

        /* Update path scaling/rotation support */
        {
            CONST D3DKMDT_VIDPN_PRESENT_PATH *pAcquiredPath;
            D3DKMDT_VIDPN_PRESENT_PATH LocalPath;

            Status = pTopology->pfnAcquirePathInfo(hTopology, SourceId, TargetId, &pAcquiredPath);
            if (NT_SUCCESS(Status)) {
                /* Copy to local, release the const original, then modify */
                LocalPath = *pAcquiredPath;
                pTopology->pfnReleasePathInfo(hTopology, pAcquiredPath);

                LocalPath.ContentTransformation.ScalingSupport.Identity = TRUE;
                LocalPath.ContentTransformation.RotationSupport.Identity = TRUE;
                Status = pTopology->pfnUpdatePathSupportInfo(hTopology, &LocalPath);
            }
        }

        /* Create source mode set if not the pivot source */
        if (pEnumCofuncModality->EnumPivotType != D3DKMDT_EPT_VIDPNSOURCE ||
            pEnumCofuncModality->EnumPivot.VidPnSourceId != SourceId) {

            D3DKMDT_HVIDPNSOURCEMODESET hSourceModeSet;
            CONST DXGK_VIDPNSOURCEMODESET_INTERFACE *pSrcInterface;
            D3DKMDT_VIDPN_SOURCE_MODE *pSrcMode;

            Status = pVidPnInterface->pfnCreateNewSourceModeSet(
                hVidPn, SourceId, &hSourceModeSet, &pSrcInterface);
            if (NT_SUCCESS(Status)) {
                Status = pSrcInterface->pfnCreateNewModeInfo(hSourceModeSet, &pSrcMode);
                if (NT_SUCCESS(Status)) {
                    FillSourceMode(pAdapter, pSrcMode);
                    Status = pSrcInterface->pfnAddMode(hSourceModeSet, pSrcMode);
                    if (!NT_SUCCESS(Status))
                        pSrcInterface->pfnReleaseModeInfo(hSourceModeSet, pSrcMode);
                }

                if (NT_SUCCESS(Status)) {
                    Status = pVidPnInterface->pfnAssignSourceModeSet(
                        hVidPn, SourceId, hSourceModeSet);
                }
                if (!NT_SUCCESS(Status))
                    pVidPnInterface->pfnReleaseSourceModeSet(hVidPn, hSourceModeSet);
            }
        }

        /* Create target mode set if not the pivot target */
        if (pEnumCofuncModality->EnumPivotType != D3DKMDT_EPT_VIDPNTARGET ||
            pEnumCofuncModality->EnumPivot.VidPnTargetId != TargetId) {

            D3DKMDT_HVIDPNTARGETMODESET hTargetModeSet;
            CONST DXGK_VIDPNTARGETMODESET_INTERFACE *pTgtInterface;
            D3DKMDT_VIDPN_TARGET_MODE *pTgtMode;

            Status = pVidPnInterface->pfnCreateNewTargetModeSet(
                hVidPn, TargetId, &hTargetModeSet, &pTgtInterface);
            if (NT_SUCCESS(Status)) {
                Status = pTgtInterface->pfnCreateNewModeInfo(hTargetModeSet, &pTgtMode);
                if (NT_SUCCESS(Status)) {
                    FillTargetMode(pAdapter, pTgtMode);
                    Status = pTgtInterface->pfnAddMode(hTargetModeSet, pTgtMode);
                    if (!NT_SUCCESS(Status))
                        pTgtInterface->pfnReleaseModeInfo(hTargetModeSet, pTgtMode);
                }

                if (NT_SUCCESS(Status)) {
                    Status = pVidPnInterface->pfnAssignTargetModeSet(
                        hVidPn, TargetId, hTargetModeSet);
                }
                if (!NT_SUCCESS(Status))
                    pVidPnInterface->pfnReleaseTargetModeSet(hVidPn, hTargetModeSet);
            }
        }

        /* Move to next path */
        {
            CONST D3DKMDT_VIDPN_PRESENT_PATH *pNextPath;
            Status = pTopology->pfnAcquireNextPathInfo(hTopology, pPath, &pNextPath);
            pTopology->pfnReleasePathInfo(hTopology, pPath);
            if (Status == STATUS_GRAPHICS_NO_MORE_ELEMENTS_IN_DATASET) {
                Status = STATUS_SUCCESS;
                break;
            }
            pPath = pNextPath;
        }
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * CommitVidPn
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuCommitVidPn(
    IN_CONST_HANDLE                         hAdapter,
    _In_ const DXGKARG_COMMITVIDPN*         pCommitVidPn
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    UINT SourceId = pCommitVidPn->AffectedVidPnSourceId;

    VidPnDbgRegWrite(L"CommitVidPn", SourceId);
    KdPrint(("AmdGpuWddm: CommitVidPn source=%u\n", SourceId));
    UNREFERENCED_PARAMETER(SourceId);

    /* Accept the VidPn -- mark path as active */
    pAdapter->VidPnState.PathActive = TRUE;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * SetVidPnSourceAddress / SetVidPnSourceVisibility
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSetVidPnSourceAddress(
    IN_CONST_HANDLE                             hAdapter,
    IN_CONST_PDXGKARG_SETVIDPNSOURCEADDRESS     pSetVidPnSourceAddress
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;

    if (pSetVidPnSourceAddress->VidPnSourceId == 0) {
        pAdapter->VidPnState.PrimaryAddress = pSetVidPnSourceAddress->PrimaryAddress;
    }

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuSetVidPnSourceVisibility(
    IN_CONST_HANDLE                                 hAdapter,
    IN_CONST_PDXGKARG_SETVIDPNSOURCEVISIBILITY      pSetVidPnSourceVisibility
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;

    if (pSetVidPnSourceVisibility->VidPnSourceId == 0) {
        pAdapter->VidPnState.SourceVisible = pSetVidPnSourceVisibility->Visible;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * UpdateActiveVidPnPresentPath
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuUpdateActiveVidPnPresentPath(
    IN_CONST_HANDLE                                     hAdapter,
    _In_ const DXGKARG_UPDATEACTIVEVIDPNPRESENTPATH*    pUpdateActiveVidPnPresentPath
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pUpdateActiveVidPnPresentPath);

    /* Accept identity scaling/rotation only (that's all we support) */
    return STATUS_SUCCESS;
}

/* ======================================================================
 * RecommendMonitorModes
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuRecommendMonitorModes(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_RECOMMENDMONITORMODES*       pRecommendMonitorModes
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    D3DKMDT_HMONITORSOURCEMODESET hModeSet;
    CONST DXGK_MONITORSOURCEMODESET_INTERFACE *pModeSetInterface;
    D3DKMDT_MONITOR_SOURCE_MODE *pMonMode;
    NTSTATUS Status;

    VidPnDbgRegWrite(L"RecommendMonModes", 1);

    hModeSet = pRecommendMonitorModes->hMonitorSourceModeSet;
    pModeSetInterface = pRecommendMonitorModes->pMonitorSourceModeSetInterface;

    Status = pModeSetInterface->pfnCreateNewModeInfo(hModeSet, &pMonMode);
    if (!NT_SUCCESS(Status))
        return Status;

    /* Fill in the preferred mode (POST dimensions) */
    /* Preserve Id assigned by pfnCreateNewModeInfo -- dxgkrnl owns it */
    {
        D3DKMDT_MONITOR_SOURCE_MODE_ID SavedId = pMonMode->Id;
        RtlZeroMemory(pMonMode, sizeof(*pMonMode));
        pMonMode->Id = SavedId;
    }
    pMonMode->VideoSignalInfo.VideoStandard = D3DKMDT_VSS_OTHER;
    pMonMode->VideoSignalInfo.TotalSize.cx = pAdapter->PostDisplay.Width;
    pMonMode->VideoSignalInfo.TotalSize.cy = pAdapter->PostDisplay.Height;
    pMonMode->VideoSignalInfo.ActiveSize.cx = pAdapter->PostDisplay.Width;
    pMonMode->VideoSignalInfo.ActiveSize.cy = pAdapter->PostDisplay.Height;
    pMonMode->VideoSignalInfo.VSyncFreq.Numerator = 60000;
    pMonMode->VideoSignalInfo.VSyncFreq.Denominator = 1000;
    pMonMode->VideoSignalInfo.HSyncFreq.Numerator = 48000;
    pMonMode->VideoSignalInfo.HSyncFreq.Denominator = 1;
    pMonMode->VideoSignalInfo.PixelRate = (ULONGLONG)pAdapter->PostDisplay.Width *
        pAdapter->PostDisplay.Height * 60;
    pMonMode->VideoSignalInfo.ScanLineOrdering = D3DDDI_VSSLO_PROGRESSIVE;
    pMonMode->ColorBasis = D3DKMDT_CB_SRGB;
    pMonMode->ColorCoeffDynamicRanges.FirstChannel = 8;
    pMonMode->ColorCoeffDynamicRanges.SecondChannel = 8;
    pMonMode->ColorCoeffDynamicRanges.ThirdChannel = 8;
    pMonMode->ColorCoeffDynamicRanges.FourthChannel = 8;
    pMonMode->Origin = D3DKMDT_MCO_DRIVER;
    pMonMode->Preference = D3DKMDT_MP_PREFERRED;

    Status = pModeSetInterface->pfnAddMode(hModeSet, pMonMode);
    if (!NT_SUCCESS(Status)) {
        pModeSetInterface->pfnReleaseModeInfo(hModeSet, pMonMode);
        return Status;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * RecommendVidPnTopology
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuRecommendVidPnTopology(
    IN_CONST_HANDLE                                 hAdapter,
    _In_ const DXGKARG_RECOMMENDVIDPNTOPOLOGY*      pRecommendVidPnTopology
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pRecommendVidPnTopology);
    /* Let dxgkrnl use its default topology selection */
    return STATUS_GRAPHICS_NO_RECOMMENDED_VIDPN_TOPOLOGY;
}

/* ======================================================================
 * GetScanLine - fake vertical blank
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuGetScanLine(
    IN_CONST_HANDLE                 hAdapter,
    INOUT_PDXGKARG_GETSCANLINE      pGetScanLine
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pGetScanLine != NULL) {
        pGetScanLine->InVerticalBlank = TRUE;
        pGetScanLine->ScanLine = 0;
    }

    return STATUS_SUCCESS;
}
