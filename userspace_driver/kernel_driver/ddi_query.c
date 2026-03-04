/*
 * ddi_query.c - QueryAdapterInfo and engine query DDIs
 *
 * The critical DDI here is QueryAdapterInfo with DXGKQAITYPE_DRIVERCAPS,
 * which tells dxgkrnl we are a compute-only (MCDM) device.
 */

#include "amdgpu_mcdm.h"

/* ======================================================================
 * QueryAdapterInfo — report driver capabilities
 *
 * dxgkrnl queries various adapter properties. The most important are:
 * - DXGKQAITYPE_DRIVERCAPS: declares ComputeOnly=TRUE
 * - DXGKQAITYPE_QUERYSEGMENT4: memory segment layout
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryAdapterInfo(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_QUERYADAPTERINFO      pQueryAdapterInfo
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;

    if (pQueryAdapterInfo == NULL || pQueryAdapterInfo->pOutputData == NULL)
        return STATUS_INVALID_PARAMETER;

    switch (pQueryAdapterInfo->Type) {

    case DXGKQAITYPE_DRIVERCAPS:
    {
        DXGK_DRIVERCAPS *pCaps;

        if (pQueryAdapterInfo->OutputDataSize < sizeof(DXGK_DRIVERCAPS))
            return STATUS_BUFFER_TOO_SMALL;

        pCaps = (DXGK_DRIVERCAPS *)pQueryAdapterInfo->pOutputData;
        RtlZeroMemory(pCaps, sizeof(*pCaps));

        /* Accept DMA from any physical address */
        pCaps->HighestAcceptableAddress.QuadPart = (LONGLONG)-1;

        /* GPU engine topology: one compute node */
        pCaps->GpuEngineTopology.NbAsymetricProcessingNodes = 1;

        /* Scheduling caps: no hardware-based scheduling */
        pCaps->SchedulingCaps.MultiEngineAware = 0;

        /* Memory management caps */
        pCaps->MemoryManagementCaps.IoMmuSupported = 1;  /* We rely on IOMMU */

        /* WDDM version: 2.6 for ComputeOnly support */
        pCaps->WDDMVersion = DXGKDDI_WDDMv2_6;

        /* Per-engine TDR support */
        pCaps->SupportPerEngineTDR = TRUE;

        /* The key bit: this is a compute-only (MCDM) adapter */
        pCaps->MiscCaps.ComputeOnly = 1;

        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_QUERYSEGMENT4:
    {
        DXGK_QUERYSEGMENTOUT4 *pSegmentOut;

        if (pQueryAdapterInfo->OutputDataSize < sizeof(DXGK_QUERYSEGMENTOUT4))
            return STATUS_BUFFER_TOO_SMALL;

        pSegmentOut = (DXGK_QUERYSEGMENTOUT4 *)pQueryAdapterInfo->pOutputData;

        if (pSegmentOut->pSegmentDescriptor == NULL) {
            /* First call: return count of segments */
            pSegmentOut->NbSegment = 2;  /* VRAM + System */
            pSegmentOut->SegmentDescriptorStride = sizeof(DXGK_SEGMENTDESCRIPTOR4);
            return STATUS_SUCCESS;
        }

        /* Second call: fill in segment descriptors using stride-based indexing */
        if (pSegmentOut->NbSegment >= 1) {
            DXGK_SEGMENTDESCRIPTOR4 *pVram =
                (DXGK_SEGMENTDESCRIPTOR4 *)(pSegmentOut->pSegmentDescriptor);
            RtlZeroMemory(pVram, sizeof(*pVram));

            /* Segment 1: VRAM */
            pVram->Flags.Aperture = 0;
            pVram->Flags.CpuVisible = 1;           /* BAR-mapped VRAM */
            pVram->BaseAddress.QuadPart = 0;
            pVram->Size = pAdapter->VramSize;
            if (pVram->Size == 0)
                pVram->Size = 256 * 1024 * 1024;   /* Default 256MB until real detection */
            pVram->CpuTranslatedAddress.QuadPart = 0; /* Filled in v0.2+ */
            pVram->CommitLimit = pVram->Size;
        }

        if (pSegmentOut->NbSegment >= 2) {
            DXGK_SEGMENTDESCRIPTOR4 *pSys =
                (DXGK_SEGMENTDESCRIPTOR4 *)(pSegmentOut->pSegmentDescriptor +
                    pSegmentOut->SegmentDescriptorStride);
            RtlZeroMemory(pSys, sizeof(*pSys));

            /* Segment 2: System memory (aperture) */
            pSys->Flags.Aperture = 1;
            pSys->Flags.CpuVisible = 1;
            pSys->Size = 1ULL * 1024 * 1024 * 1024;  /* 1GB aperture */
            pSys->CommitLimit = pSys->Size;
        }

        pSegmentOut->PagingBufferSegmentId = 0;     /* System memory */
        pSegmentOut->PagingBufferSize = 4096;        /* Minimal */

        return STATUS_SUCCESS;
    }

    case DXGKQAITYPE_UMDRIVERPRIVATE:
        /* No UMD private data */
        return STATUS_NOT_SUPPORTED;

    default:
        return STATUS_NOT_SUPPORTED;
    }
}

/* ======================================================================
 * GetNodeMetadata — describe GPU engines
 *
 * We report one compute-only engine node.
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

    if (NodeOrdinal == 0) {
        pGetNodeMetadata->EngineType = DXGK_ENGINE_TYPE_3D;
        /* For compute-only, 3D engine type is used by convention */
        pGetNodeMetadata->FriendlyName[0] = L'C';
        pGetNodeMetadata->FriendlyName[1] = L'o';
        pGetNodeMetadata->FriendlyName[2] = L'm';
        pGetNodeMetadata->FriendlyName[3] = L'p';
        pGetNodeMetadata->FriendlyName[4] = L'u';
        pGetNodeMetadata->FriendlyName[5] = L't';
        pGetNodeMetadata->FriendlyName[6] = L'e';
        pGetNodeMetadata->FriendlyName[7] = L'\0';
        pGetNodeMetadata->GpuMmuSupported = TRUE;
        pGetNodeMetadata->IoMmuSupported = TRUE;
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

    /* Single engine — depends only on itself */
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

    /* Report engine as active and not hung */
    pQueryEngineStatus->EngineStatus.Responsive = 1;

    return STATUS_SUCCESS;
}
