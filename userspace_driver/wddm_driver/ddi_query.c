/*
 * ddi_query.c - QueryAdapterInfo and engine query DDIs for ROCm Display Driver
 *
 * Key difference from MCDM: no ComputeOnly bit, display capabilities.
 */

#include "amdgpu_wddm.h"
#include <ntstrsafe.h>

static void
QaiDbgRegWrite(const WCHAR *Name, ULONG Value)
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
    s = ZwOpenKey(&hKey, KEY_ALL_ACCESS, &ObjAttrs);
    if (NT_SUCCESS(s)) {
        RtlInitUnicodeString(&ValName, Name);
        ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Value, sizeof(Value));
        ZwClose(hKey);
    }
}

NTSTATUS
APIENTRY
AmdGpuQueryAdapterInfo(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_QUERYADAPTERINFO      pQueryAdapterInfo
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pQueryAdapterInfo == NULL || pQueryAdapterInfo->pOutputData == NULL)
        return STATUS_INVALID_PARAMETER;

    KdPrint(("AmdGpuWddm: QueryAdapterInfo type=%u outputSize=%u\n",
        pQueryAdapterInfo->Type, pQueryAdapterInfo->OutputDataSize));

    /* Log query type */
    {
        WCHAR ValNameBuf[32];
        RtlStringCbPrintfW(ValNameBuf, sizeof(ValNameBuf),
            L"QAI_%u", pQueryAdapterInfo->Type);
        QaiDbgRegWrite(ValNameBuf, pQueryAdapterInfo->OutputDataSize);
    }

    switch (pQueryAdapterInfo->Type) {

    case DXGKQAITYPE_DRIVERCAPS:
    {
        DXGK_DRIVERCAPS *pCaps;

        /*
         * dxgkrnl sizes the buffer for the WDDM version we registered.
         * We compile against WDDM 3.2 headers but register as WDDM 2.0,
         * so sizeof(DXGK_DRIVERCAPS) > OutputDataSize. Only zero/fill
         * the buffer dxgkrnl gave us.
         */
        pCaps = (DXGK_DRIVERCAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pCaps, pQueryAdapterInfo->OutputDataSize);

        pCaps->HighestAcceptableAddress.QuadPart = (LONGLONG)-1;
        pCaps->GpuEngineTopology.NbAsymetricProcessingNodes = 1;
        pCaps->SchedulingCaps.MultiEngineAware = 1;
        pCaps->WDDMVersion = DXGKDDI_WDDMv1_3;

        /* WDDM 1.2+ requirements */
        pCaps->PreemptionCaps.GraphicsPreemptionGranularity =
            D3DKMDT_GRAPHICS_PREEMPTION_NONE;
        pCaps->PreemptionCaps.ComputePreemptionGranularity =
            D3DKMDT_COMPUTE_PREEMPTION_NONE;
        pCaps->SupportNonVGA = TRUE;
        pCaps->SupportPerEngineTDR = TRUE;

        /* No hardware cursor */
        pCaps->MaxPointerWidth = 0;
        pCaps->MaxPointerHeight = 0;

        /* Display driver Present capabilities */
        pCaps->PresentationCaps.NoScreenToScreenBlt = 1;
        pCaps->PresentationCaps.NoOverlapScreenBlt = 1;

        QaiDbgRegWrite(L"QAI_1_Result", 0);
        QaiDbgRegWrite(L"QAI_1_WDDMVer", (ULONG)pCaps->WDDMVersion);
        QaiDbgRegWrite(L"QAI_1_Engines", pCaps->GpuEngineTopology.NbAsymetricProcessingNodes);
        QaiDbgRegWrite(L"QAI_1_OutSize", pQueryAdapterInfo->OutputDataSize);
        /* Log field offsets to verify they're within the buffer */
        {
            ULONG offWddm = (ULONG)((PUCHAR)&pCaps->WDDMVersion - (PUCHAR)pCaps);
            ULONG offMmCaps = (ULONG)((PUCHAR)&pCaps->MemoryManagementCaps - (PUCHAR)pCaps);
            ULONG offNonVGA = (ULONG)((PUCHAR)&pCaps->SupportNonVGA - (PUCHAR)pCaps);
            QaiDbgRegWrite(L"QAI_1_Off_WDDMVer", offWddm);
            QaiDbgRegWrite(L"QAI_1_Off_MmCaps", offMmCaps);
            QaiDbgRegWrite(L"QAI_1_Off_NonVGA", offNonVGA);
            QaiDbgRegWrite(L"QAI_1_SizeofCaps", (ULONG)sizeof(DXGK_DRIVERCAPS));
        }
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_QUERYSEGMENT3:
    {
        DXGK_QUERYSEGMENTOUT3 *pSegmentOut;

        QaiDbgRegWrite(L"QAI_5_Enter", pQueryAdapterInfo->OutputDataSize);

        pSegmentOut = (DXGK_QUERYSEGMENTOUT3 *)pQueryAdapterInfo->pOutputData;

        if (pSegmentOut->pSegmentDescriptor == NULL) {
            /* First pass: return count */
            pSegmentOut->NbSegment = 1;
            QaiDbgRegWrite(L"QAI_5_Pass1", 1);
            return STATUS_SUCCESS;
        }

        if (pSegmentOut->NbSegment >= 1) {
            DXGK_SEGMENTDESCRIPTOR3 *pSysMem =
                (DXGK_SEGMENTDESCRIPTOR3 *)(pSegmentOut->pSegmentDescriptor);
            RtlZeroMemory(pSysMem, sizeof(*pSysMem));

            pSysMem->Flags.Aperture = 1;
            pSysMem->Flags.CpuVisible = 1;
            pSysMem->Size = 256ULL * 1024 * 1024;
            pSysMem->CommitLimit = pSysMem->Size;
        }

        pSegmentOut->PagingBufferSegmentId = 0;
        pSegmentOut->PagingBufferSize = 4096;

        QaiDbgRegWrite(L"QAI_5_Pass2", 1);
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_QUERYSEGMENT4:
    {
        DXGK_QUERYSEGMENTOUT4 *pSegmentOut;

        QaiDbgRegWrite(L"QAI_11_Enter", pQueryAdapterInfo->OutputDataSize);

        /* Don't check against sizeof -- it may be compiled larger than what dxgkrnl provides */
        pSegmentOut = (DXGK_QUERYSEGMENTOUT4 *)pQueryAdapterInfo->pOutputData;

        if (pSegmentOut->pSegmentDescriptor == NULL) {
            /* First pass: return count */
            pSegmentOut->NbSegment = 1;
            pSegmentOut->SegmentDescriptorStride = sizeof(DXGK_SEGMENTDESCRIPTOR4);
            QaiDbgRegWrite(L"QAI_11_Pass1", 1);
            return STATUS_SUCCESS;
        }

        if (pSegmentOut->NbSegment >= 1) {
            DXGK_SEGMENTDESCRIPTOR4 *pSysMem =
                (DXGK_SEGMENTDESCRIPTOR4 *)(pSegmentOut->pSegmentDescriptor);
            RtlZeroMemory(pSysMem, sizeof(*pSysMem));

            pSysMem->Flags.Aperture = 1;
            pSysMem->Flags.CpuVisible = 1;
            pSysMem->Size = 256ULL * 1024 * 1024;
            pSysMem->CommitLimit = pSysMem->Size;
        }

        pSegmentOut->PagingBufferSegmentId = 0;
        pSegmentOut->PagingBufferSize = 4096;

        QaiDbgRegWrite(L"QAI_11_Pass2", 1);
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_WDDMDEVICECAPS:
    {
        DXGK_WDDMDEVICECAPS *pDevCaps;

        pDevCaps = (DXGK_WDDMDEVICECAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pDevCaps, pQueryAdapterInfo->OutputDataSize);
        pDevCaps->WDDMVersion = DXGKDDI_WDDMv1_3;
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_UMDRIVERPRIVATE:
        return STATUS_NOT_SUPPORTED;

    case DXGKQAITYPE_NUMPOWERCOMPONENTS:
    {
        /* Return 0 power components */
        if (pQueryAdapterInfo->OutputDataSize >= sizeof(UINT)) {
            *(UINT *)pQueryAdapterInfo->pOutputData = 0;
            return STATUS_SUCCESS;
        }
        return STATUS_BUFFER_TOO_SMALL;
    }

    case DXGKQAITYPE_PHYSICALADAPTERCAPS:
    {
        DXGK_PHYSICALADAPTERCAPS *pPhysCaps =
            (DXGK_PHYSICALADAPTERCAPS *)pQueryAdapterInfo->pOutputData;

        /*
         * DxgkPhysicalAdapterHandle is pre-filled by dxgkrnl as INPUT.
         * Do NOT zero it. Only set the fields we own.
         */
        QaiDbgRegWrite(L"QAI_15_InHandle",
            (ULONG)(ULONG_PTR)pPhysCaps->DxgkPhysicalAdapterHandle);

        pPhysCaps->NumExecutionNodes = 1;
        pPhysCaps->PagingNodeIndex = 0;
        pPhysCaps->Flags.Value = 0;

        QaiDbgRegWrite(L"QAI_15_Done", 1);
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_DISPLAY_DRIVERCAPS_EXTENSION:
    {
        /* Display caps extension -- all zeroed is fine */
        RtlZeroMemory(pQueryAdapterInfo->pOutputData,
            pQueryAdapterInfo->OutputDataSize);
        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_DEVICE_TYPE_CAPS:
    {
        DXGK_DEVICE_TYPE_CAPS *pTypeCaps =
            (DXGK_DEVICE_TYPE_CAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pTypeCaps, pQueryAdapterInfo->OutputDataSize);
        pTypeCaps->Discrete = 1;
        return STATUS_SUCCESS;
    }

    default:
        KdPrint(("AmdGpuWddm: QueryAdapterInfo UNSUPPORTED type=%u size=%u\n",
            pQueryAdapterInfo->Type, pQueryAdapterInfo->OutputDataSize));
        /*
         * Zero the output buffer and return success for unknown QAI types.
         * STATUS_NOT_SUPPORTED can be fatal for some mandatory types that
         * dxgkrnl queries based on WDDM version. Zeroed output is safe
         * and indicates "no capabilities" for most QAI types.
         */
        RtlZeroMemory(pQueryAdapterInfo->pOutputData,
            pQueryAdapterInfo->OutputDataSize);
        {
            WCHAR ValNameBuf[32];
            RtlStringCbPrintfW(ValNameBuf, sizeof(ValNameBuf),
                L"QAI_%u_Done", pQueryAdapterInfo->Type);
            QaiDbgRegWrite(ValNameBuf, 1);
        }
        return STATUS_SUCCESS;
    }
}

/* ======================================================================
 * GetNodeMetadata
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuGetNodeMetadata(
    IN_CONST_HANDLE                     hAdapter,
    UINT                                NodeOrdinalAndAdapterIndex,
    OUT_PDXGKARG_GETNODEMETADATA        pGetNodeMetadata
    )
{
    UINT NodeOrdinal = NodeOrdinalAndAdapterIndex & 0xFFFF;

    UNREFERENCED_PARAMETER(hAdapter);

    if (pGetNodeMetadata == NULL)
        return STATUS_INVALID_PARAMETER;

    RtlZeroMemory(pGetNodeMetadata, sizeof(*pGetNodeMetadata));

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
            RtlInitUnicodeString(&ValName, L"GetNodeMetadata");
            Val = NodeOrdinal;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));
            ZwClose(hKey);
        }
    }

    if (NodeOrdinal == 0) {
        pGetNodeMetadata->EngineType = DXGK_ENGINE_TYPE_3D;
        pGetNodeMetadata->FriendlyName[0] = L'3';
        pGetNodeMetadata->FriendlyName[1] = L'D';
        pGetNodeMetadata->FriendlyName[2] = L'\0';
        pGetNodeMetadata->GpuMmuSupported = FALSE;
        pGetNodeMetadata->IoMmuSupported = FALSE;
    } else {
        return STATUS_INVALID_PARAMETER;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * Engine status queries
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryDependentEngineGroup(
    IN_CONST_HANDLE                                 hAdapter,
    INOUT_DXGKARG_QUERYDEPENDENTENGINEGROUP         pQueryDependentEngineGroup
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pQueryDependentEngineGroup == NULL)
        return STATUS_INVALID_PARAMETER;

    pQueryDependentEngineGroup->DependentNodeOrdinalMask = 1;
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuQueryEngineStatus(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_QUERYENGINESTATUS        pQueryEngineStatus
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pQueryEngineStatus == NULL)
        return STATUS_INVALID_PARAMETER;

    pQueryEngineStatus->EngineStatus.Responsive = 1;
    return STATUS_SUCCESS;
}
