/*
 * ddi_memory.c - Memory management DDIs for ROCm Display Driver
 *
 * Extended from MCDM: GetStandardAllocationDriverData handles standard
 * surface types (SharedPrimary, Shadow, Staging) needed by DWM/WARP.
 * CreateAllocation reads AMDGPU_ALLOC_PRIVATE from private driver data.
 * DescribeAllocation returns surface dimensions.
 */

#include "amdgpu_wddm.h"
#include <ntstrsafe.h>

static void
MemDbgRegWrite(const WCHAR *Name, ULONG Value)
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

/* Helper: compute pitch aligned to 256 bytes */
static UINT
AlignedPitch(UINT Width, UINT BytesPerPixel)
{
    UINT Pitch = Width * BytesPerPixel;
    return (Pitch + 255) & ~255U;
}

static UINT
FormatBpp(D3DDDIFORMAT Format)
{
    switch (Format) {
    case D3DDDIFMT_A8R8G8B8:
    case D3DDDIFMT_X8R8G8B8:
        return 4;
    case D3DDDIFMT_R5G6B5:
    case D3DDDIFMT_A1R5G5B5:
    case D3DDDIFMT_X1R5G5B5:
        return 2;
    case D3DDDIFMT_A8:
    case D3DDDIFMT_P8:
        return 1;
    default:
        return 4;
    }
}

/* ======================================================================
 * GetStandardAllocationDriverData
 *
 * Handle standard surface types that dxgkrnl creates for DWM:
 * - SharedPrimarySurface: the desktop primary
 * - ShadowSurface: DWM shadow for composition
 * - StagingSurface: CPU-accessible staging
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuGetStandardAllocationDriverData(
    IN_CONST_HANDLE                                         hAdapter,
    INOUT_PDXGKARG_GETSTANDARDALLOCATIONDRIVERDATA          pStdAllocData
    )
{
    AMDGPU_ALLOC_PRIVATE *pPrivate;
    UINT Width, Height, Bpp, Pitch;
    D3DDDIFORMAT Format;
    BOOLEAN IsPrimary = FALSE;

    UNREFERENCED_PARAMETER(hAdapter);

    if (pStdAllocData == NULL)
        return STATUS_INVALID_PARAMETER;

    switch (pStdAllocData->StandardAllocationType) {

    case D3DKMDT_STANDARDALLOCATION_SHAREDPRIMARYSURFACE:
    {
        D3DKMDT_SHAREDPRIMARYSURFACEDATA *pPrimary =
            pStdAllocData->pCreateSharedPrimarySurfaceData;
        Width = pPrimary->Width;
        Height = pPrimary->Height;
        Format = pPrimary->Format;
        IsPrimary = TRUE;
        break;
    }

    case D3DKMDT_STANDARDALLOCATION_SHADOWSURFACE:
    {
        D3DKMDT_SHADOWSURFACEDATA *pShadow =
            pStdAllocData->pCreateShadowSurfaceData;
        Width = pShadow->Width;
        Height = pShadow->Height;
        Format = pShadow->Format;
        break;
    }

    case D3DKMDT_STANDARDALLOCATION_STAGINGSURFACE:
    {
        D3DKMDT_STAGINGSURFACEDATA *pStaging =
            pStdAllocData->pCreateStagingSurfaceData;
        Width = pStaging->Width;
        Height = pStaging->Height;
        Format = D3DDDIFMT_X8R8G8B8;
        break;
    }

    default:
        return STATUS_NOT_SUPPORTED;
    }

    Bpp = FormatBpp(Format);
    Pitch = AlignedPitch(Width, Bpp);

    /* First call: return required sizes */
    pStdAllocData->AllocationPrivateDriverDataSize = sizeof(AMDGPU_ALLOC_PRIVATE);
    pStdAllocData->ResourcePrivateDriverDataSize = 0;

    /* Second call: fill in private data if buffer provided */
    if (pStdAllocData->pAllocationPrivateDriverData != NULL) {
        pPrivate = (AMDGPU_ALLOC_PRIVATE *)pStdAllocData->pAllocationPrivateDriverData;
        pPrivate->Width = Width;
        pPrivate->Height = Height;
        pPrivate->Pitch = Pitch;
        pPrivate->Format = Format;
        pPrivate->IsPrimary = IsPrimary;
    }

    /* Write back pitch for shadow/staging surfaces */
    switch (pStdAllocData->StandardAllocationType) {
    case D3DKMDT_STANDARDALLOCATION_SHADOWSURFACE:
        pStdAllocData->pCreateShadowSurfaceData->Pitch = Pitch;
        break;
    case D3DKMDT_STANDARDALLOCATION_STAGINGSURFACE:
        pStdAllocData->pCreateStagingSurfaceData->Pitch = Pitch;
        break;
    default:
        break;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * CreateAllocation / DestroyAllocation
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
    MemDbgRegWrite(L"CreateAllocation", pCreateAllocation->NumAllocations);

    for (i = 0; i < pCreateAllocation->NumAllocations; i++) {
        DXGK_ALLOCATIONINFO *pAllocInfo = &pCreateAllocation->pAllocationInfo[i];
        AMDGPU_ALLOCATION *pAlloc;

        pAlloc = (AMDGPU_ALLOCATION *)ExAllocatePool2(
            POOL_FLAG_NON_PAGED, sizeof(AMDGPU_ALLOCATION), AMDGPU_POOL_TAG);
        if (pAlloc == NULL)
            return STATUS_INSUFFICIENT_RESOURCES;

        RtlZeroMemory(pAlloc, sizeof(*pAlloc));
        pAlloc->InUse = TRUE;

        /* Read private driver data if provided (from GetStandardAllocationDriverData) */
        if (pAllocInfo->pPrivateDriverData != NULL &&
            pAllocInfo->PrivateDriverDataSize >= sizeof(AMDGPU_ALLOC_PRIVATE)) {

            AMDGPU_ALLOC_PRIVATE *pPriv =
                (AMDGPU_ALLOC_PRIVATE *)pAllocInfo->pPrivateDriverData;
            pAlloc->Width = pPriv->Width;
            pAlloc->Height = pPriv->Height;
            pAlloc->Pitch = pPriv->Pitch;
            pAlloc->Format = pPriv->Format;
            pAlloc->IsPrimary = pPriv->IsPrimary;

            /* Compute size from dimensions */
            pAlloc->Size = (ULONGLONG)pPriv->Pitch * pPriv->Height;
        } else {
            pAlloc->Size = pAllocInfo->Size;
        }

        /* Report to dxgkrnl */
        pAllocInfo->hAllocation = pAlloc;
        if (pAlloc->Size > 0) {
            pAllocInfo->Size = pAlloc->Size;
        } else {
            pAllocInfo->Size = (pAllocInfo->Size == 0) ? 4096 : pAllocInfo->Size;
            pAlloc->Size = pAllocInfo->Size;
        }

        pAllocInfo->PreferredSegment.Value = 0;
        pAllocInfo->SupportedReadSegmentSet = 0x1;   /* Segment 1 */
        pAllocInfo->SupportedWriteSegmentSet = 0x1;

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
 * DescribeAllocation - return surface dimensions
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuDescribeAllocation(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_DESCRIBEALLOCATION       pDescribeAllocation
    )
{
    AMDGPU_ALLOCATION *pAlloc;

    UNREFERENCED_PARAMETER(hAdapter);

    if (pDescribeAllocation == NULL)
        return STATUS_INVALID_PARAMETER;

    pAlloc = (AMDGPU_ALLOCATION *)pDescribeAllocation->hAllocation;

    if (pAlloc != NULL && pAlloc->InUse) {
        pDescribeAllocation->Width = pAlloc->Width;
        pDescribeAllocation->Height = pAlloc->Height;
        pDescribeAllocation->Format = pAlloc->Format;
    } else {
        pDescribeAllocation->Width = 0;
        pDescribeAllocation->Height = 0;
        pDescribeAllocation->Format = D3DDDIFMT_UNKNOWN;
    }

    pDescribeAllocation->MultisampleMethod.NumSamples = 0;
    pDescribeAllocation->MultisampleMethod.NumQualityLevels = 0;
    pDescribeAllocation->RefreshRate.Numerator = 0;
    pDescribeAllocation->RefreshRate.Denominator = 0;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * OpenAllocation / CloseAllocation
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
 * BuildPagingBuffer - no-op
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuBuildPagingBuffer(
    IN_CONST_HANDLE                     hAdapter,
    IN_PDXGKARG_BUILDPAGINGBUFFER       pBuildPagingBuffer
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pBuildPagingBuffer != NULL)
        pBuildPagingBuffer->MultipassOffset = 0;

    return STATUS_SUCCESS;
}

/* ======================================================================
 * WDDM 2.0+ GPU virtual address support
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
        pGetPageTableSize->NumberOfPte = 1;

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
