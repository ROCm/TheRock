/*
 * ddi_memory.c - Memory management DDIs for MCDM miniport
 *
 * These satisfy the WDDM memory management contract with minimal
 * implementations. Real GPU memory management is done in Python
 * through the escape interface.
 *
 * BuildPagingBuffer returns success without doing anything — Python
 * manages GPU page tables via MMIO writes through the escape handler.
 */

#include "amdgpu_mcdm.h"

/* ======================================================================
 * Per-allocation tracking
 * ====================================================================== */

typedef struct _AMDGPU_ALLOCATION {
    ULONGLONG       Size;
    ULONG           SegmentId;      /* 1 = VRAM, 2 = System */
    BOOLEAN         InUse;
} AMDGPU_ALLOCATION;

/* ======================================================================
 * CreateAllocation / DestroyAllocation
 *
 * dxgkrnl calls these when userspace creates GPU resources. For our
 * fake MCDM, we just track metadata — no real GPU allocation happens.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuCreateAllocation(
    IN_CONST_HANDLE                     hAdapter,
    INOUT_PDXGKARG_CREATEALLOCATION     pCreateAllocation
    )
{
    UINT i;

    UNREFERENCED_PARAMETER(hAdapter);

    for (i = 0; i < pCreateAllocation->NumAllocations; i++) {
        DXGK_ALLOCATIONINFO *pAllocInfo = &pCreateAllocation->pAllocationInfo[i];
        AMDGPU_ALLOCATION *pAlloc;

        pAlloc = (AMDGPU_ALLOCATION *)ExAllocatePool2(
            POOL_FLAG_NON_PAGED, sizeof(AMDGPU_ALLOCATION), AMDGPU_POOL_TAG);
        if (pAlloc == NULL)
            return STATUS_INSUFFICIENT_RESOURCES;

        RtlZeroMemory(pAlloc, sizeof(*pAlloc));
        pAlloc->Size = pAllocInfo->Size;
        pAlloc->InUse = TRUE;

        /* Report to dxgkrnl */
        pAllocInfo->hAllocation = pAlloc;
        pAllocInfo->Size = (pAllocInfo->Size == 0) ? 4096 : pAllocInfo->Size;
        pAllocInfo->PreferredSegment.Value = 0;
        pAllocInfo->SupportedReadSegmentSet = 0x3;   /* Segments 1 and 2 */
        pAllocInfo->SupportedWriteSegmentSet = 0x3;

        /* Mark allocation as CPU-visible and cacheable */
        pAllocInfo->Flags.CpuVisible = 1;
    }

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuDestroyAllocation(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_DESTROYALLOCATION     pDestroyAllocation
    )
{
    UINT i;

    UNREFERENCED_PARAMETER(hAdapter);

    if (pDestroyAllocation == NULL)
        return STATUS_SUCCESS;

    for (i = 0; i < pDestroyAllocation->NumAllocations; i++) {
        PVOID hAlloc = pDestroyAllocation->pAllocationList[i];
        if (hAlloc != NULL)
            ExFreePoolWithTag(hAlloc, AMDGPU_POOL_TAG);
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * DescribeAllocation — tell dxgkrnl about an allocation's properties
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuDescribeAllocation(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_DESCRIBEALLOCATION       pDescribeAllocation
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pDescribeAllocation == NULL)
        return STATUS_INVALID_PARAMETER;

    /* Minimal description — no display surfaces */
    pDescribeAllocation->Width = 0;
    pDescribeAllocation->Height = 0;
    pDescribeAllocation->Format = D3DDDIFMT_UNKNOWN;
    pDescribeAllocation->MultisampleMethod.NumSamples = 0;
    pDescribeAllocation->MultisampleMethod.NumQualityLevels = 0;
    pDescribeAllocation->RefreshRate.Numerator = 0;
    pDescribeAllocation->RefreshRate.Denominator = 0;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * GetStandardAllocationDriverData
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuGetStandardAllocationDriverData(
    IN_CONST_HANDLE                                         hAdapter,
    INOUT_PDXGKARG_GETSTANDARDALLOCATIONDRIVERDATA          pGetStandardAllocationDriverData
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pGetStandardAllocationDriverData);
    return STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * OpenAllocation / CloseAllocation — ref counting on allocation handles
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuOpenAllocation(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_OPENALLOCATION    pOpenAllocation
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pOpenAllocation);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuCloseAllocation(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_CLOSEALLOCATION   pCloseAllocation
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pCloseAllocation);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * BuildPagingBuffer — stub that returns success
 *
 * dxgkrnl calls this to ask the driver to build DMA commands for
 * paging operations (fill, transfer, map aperture). Since Python
 * manages GPU page tables directly, we just return success.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuBuildPagingBuffer(
    IN_CONST_HANDLE                     hAdapter,
    IN_PDXGKARG_BUILDPAGINGBUFFER       pBuildPagingBuffer
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pBuildPagingBuffer != NULL) {
        /*
         * Tell dxgkrnl we consumed zero bytes. We don't actually build
         * any DMA commands since Python handles memory management.
         */
        pBuildPagingBuffer->MultipassOffset = 0;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * WDDM 2.0+ GPU virtual address support
 *
 * SetRootPageTable / GetRootPageTableSize / MapCpuHostAperture —
 * minimal implementations since Python manages the real GPU page tables.
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSetRootPageTable(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_SETROOTPAGETABLE  pSetPageTable
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pSetPageTable);
    /* Store the page table base if needed; Python manages content */
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuGetRootPageTableSize(
    IN_CONST_HANDLE                             hAdapter,
    INOUT_PDXGKARG_GETROOTPAGETABLESIZE         pGetPageTableSize
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pGetPageTableSize != NULL)
        pGetPageTableSize->NumberOfPte = 1;  /* Minimal */

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuMapCpuHostAperture(
    IN_CONST_HANDLE                             hAdapter,
    IN_CONST_PDXGKARG_MAPCPUHOSTAPERTURE        pMapCpuHostAperture
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pMapCpuHostAperture);
    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuUnmapCpuHostAperture(
    IN_CONST_HANDLE                             hAdapter,
    IN_CONST_PDXGKARG_UNMAPCPUHOSTAPERTURE      pUnmapCpuHostAperture
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pUnmapCpuHostAperture);
    return STATUS_SUCCESS;
}
