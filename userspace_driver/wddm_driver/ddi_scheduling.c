/*
 * ddi_scheduling.c - Scheduling DDIs for ROCm Display Driver
 *
 * Extended from MCDM: SubmitCommand now executes CPU blits for
 * software Present, and Patch resolves physical addresses.
 *
 * SubmitCommand runs at DISPATCH_LEVEL. MmMapIoSpaceEx/MmUnmapIoSpace
 * are callable at DISPATCH_LEVEL.
 */

#include "amdgpu_wddm.h"

/* ======================================================================
 * Helper: execute a CPU blit to POST framebuffer
 * ====================================================================== */

static void
ExecuteBlt(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _In_ AMDGPU_DMA_CMD *pCmd
    )
{
    /*
     * For software present, we copy the dirty rect to the POST framebuffer.
     * The src/dst physical addresses were resolved during Patch.
     *
     * We map the source allocation, then copy rows to the POST fb kernel VA.
     * This avoids mapping both src and dst (dst is always the POST fb).
     */
    PVOID SrcVa = NULL;
    PUCHAR FbBase;
    ULONG SrcWidth, SrcHeight;
    ULONG DstX, DstY;
    ULONG Bpp;
    ULONG Row;
    ULONG CopyWidthBytes;
    SIZE_T SrcMapSize;

    if (!pAdapter->PostDisplay.FramebufferKernelVa)
        return;

    if (!pCmd->DstIsPrimary)
        return;

    FbBase = (PUCHAR)pAdapter->PostDisplay.FramebufferKernelVa;
    Bpp = pCmd->BytesPerPixel;
    if (Bpp == 0)
        Bpp = 4;

    SrcWidth = pCmd->SrcRect.right - pCmd->SrcRect.left;
    SrcHeight = pCmd->SrcRect.bottom - pCmd->SrcRect.top;
    DstX = pCmd->DstRect.left;
    DstY = pCmd->DstRect.top;

    if (SrcWidth == 0 || SrcHeight == 0)
        return;

    /* Clip to POST framebuffer bounds */
    if (DstX >= pAdapter->PostDisplay.Width ||
        DstY >= pAdapter->PostDisplay.Height)
        return;

    if (DstX + SrcWidth > pAdapter->PostDisplay.Width)
        SrcWidth = pAdapter->PostDisplay.Width - DstX;
    if (DstY + SrcHeight > pAdapter->PostDisplay.Height)
        SrcHeight = pAdapter->PostDisplay.Height - DstY;

    CopyWidthBytes = SrcWidth * Bpp;

    /* Map source physical address */
    if (pCmd->SrcPhysAddr.QuadPart == 0)
        return;

    SrcMapSize = (SIZE_T)pCmd->SrcPitch * (pCmd->SrcRect.bottom);
    if (SrcMapSize == 0)
        SrcMapSize = CopyWidthBytes * SrcHeight;

    SrcVa = MmMapIoSpaceEx(
        pCmd->SrcPhysAddr,
        SrcMapSize,
        PAGE_READWRITE | PAGE_NOCACHE);
    if (SrcVa == NULL)
        return;

    /* Scanline copy */
    for (Row = 0; Row < SrcHeight; Row++) {
        PUCHAR SrcRow = (PUCHAR)SrcVa +
            (pCmd->SrcRect.top + Row) * pCmd->SrcPitch +
            pCmd->SrcRect.left * Bpp;
        PUCHAR DstRow = FbBase +
            (DstY + Row) * pAdapter->PostDisplay.Pitch +
            DstX * Bpp;

        RtlCopyMemory(DstRow, SrcRow, CopyWidthBytes);
    }

    MmUnmapIoSpace(SrcVa, SrcMapSize);
}

static void
ExecuteColorFill(
    _In_ AMDGPU_ADAPTER *pAdapter,
    _In_ AMDGPU_DMA_CMD *pCmd
    )
{
    PUCHAR FbBase;
    ULONG FillWidth, FillHeight;
    ULONG DstX, DstY;
    ULONG Bpp;
    ULONG Row, Col;

    if (!pAdapter->PostDisplay.FramebufferKernelVa)
        return;

    if (!pCmd->DstIsPrimary)
        return;

    FbBase = (PUCHAR)pAdapter->PostDisplay.FramebufferKernelVa;
    Bpp = pCmd->BytesPerPixel;
    if (Bpp == 0)
        Bpp = 4;

    FillWidth = pCmd->DstRect.right - pCmd->DstRect.left;
    FillHeight = pCmd->DstRect.bottom - pCmd->DstRect.top;
    DstX = pCmd->DstRect.left;
    DstY = pCmd->DstRect.top;

    if (FillWidth == 0 || FillHeight == 0)
        return;

    /* Clip */
    if (DstX >= pAdapter->PostDisplay.Width ||
        DstY >= pAdapter->PostDisplay.Height)
        return;

    if (DstX + FillWidth > pAdapter->PostDisplay.Width)
        FillWidth = pAdapter->PostDisplay.Width - DstX;
    if (DstY + FillHeight > pAdapter->PostDisplay.Height)
        FillHeight = pAdapter->PostDisplay.Height - DstY;

    for (Row = 0; Row < FillHeight; Row++) {
        PUCHAR DstRow = FbBase +
            (DstY + Row) * pAdapter->PostDisplay.Pitch +
            DstX * Bpp;

        if (Bpp == 4) {
            PULONG Pixels = (PULONG)DstRow;
            for (Col = 0; Col < FillWidth; Col++)
                Pixels[Col] = pCmd->FillColor;
        } else {
            RtlFillMemory(DstRow, FillWidth * Bpp, (UCHAR)pCmd->FillColor);
        }
    }
}

/* ======================================================================
 * SubmitCommand
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSubmitCommand(
    IN_CONST_HANDLE                     hAdapter,
    IN_CONST_PDXGKARG_SUBMITCOMMAND     pSubmitCommand
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    DXGKARGCB_NOTIFY_INTERRUPT_DATA NotifyData;

    if (pSubmitCommand == NULL)
        return STATUS_SUCCESS;

    /* Execute DMA command if present */
    if (pSubmitCommand->DmaBufferPrivateDataSubmissionStartOffset == 0 &&
        pSubmitCommand->DmaBufferPrivateDataSubmissionEndOffset >= sizeof(AMDGPU_DMA_CMD)) {

        AMDGPU_DMA_CMD *pCmd = (AMDGPU_DMA_CMD *)
            ((PUCHAR)pSubmitCommand->pDmaBufferPrivateData);

        switch (pCmd->Command) {
        case AMDGPU_CMD_BLT:
            ExecuteBlt(pAdapter, pCmd);
            break;
        case AMDGPU_CMD_COLORFILL:
            ExecuteColorFill(pAdapter, pCmd);
            break;
        case AMDGPU_CMD_NOP:
        default:
            break;
        }
    }

    /* Signal immediate completion */
    RtlZeroMemory(&NotifyData, sizeof(NotifyData));
    NotifyData.InterruptType = DXGK_INTERRUPT_DMA_COMPLETED;
    NotifyData.DmaCompleted.SubmissionFenceId = pSubmitCommand->SubmissionFenceId;
    NotifyData.DmaCompleted.NodeOrdinal = pSubmitCommand->NodeOrdinal;
    NotifyData.DmaCompleted.EngineOrdinal = pSubmitCommand->EngineOrdinal;

    pAdapter->DxgkInterface.DxgkCbNotifyInterrupt(
        pAdapter->DxgkInterface.DeviceHandle, &NotifyData);

    pAdapter->DxgkInterface.DxgkCbQueueDpc(
        pAdapter->DxgkInterface.DeviceHandle);

    return STATUS_SUCCESS;
}

/* ======================================================================
 * SubmitCommandVirtual - WDDM 2.0+
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuSubmitCommandVirtual(
    IN_CONST_HANDLE                             hAdapter,
    IN_CONST_PDXGKARG_SUBMITCOMMANDVIRTUAL      pSubmitCommandVirtual
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)hAdapter;
    DXGKARGCB_NOTIFY_INTERRUPT_DATA NotifyData;

    if (pSubmitCommandVirtual == NULL)
        return STATUS_SUCCESS;

    RtlZeroMemory(&NotifyData, sizeof(NotifyData));
    NotifyData.InterruptType = DXGK_INTERRUPT_DMA_COMPLETED;
    NotifyData.DmaCompleted.SubmissionFenceId = pSubmitCommandVirtual->SubmissionFenceId;
    NotifyData.DmaCompleted.NodeOrdinal = pSubmitCommandVirtual->NodeOrdinal;
    NotifyData.DmaCompleted.EngineOrdinal = pSubmitCommandVirtual->EngineOrdinal;

    pAdapter->DxgkInterface.DxgkCbNotifyInterrupt(
        pAdapter->DxgkInterface.DeviceHandle, &NotifyData);

    pAdapter->DxgkInterface.DxgkCbQueueDpc(
        pAdapter->DxgkInterface.DeviceHandle);

    return STATUS_SUCCESS;
}

/* ======================================================================
 * PreemptCommand
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuPreemptCommand(
    IN_CONST_HANDLE                         hAdapter,
    IN_CONST_PDXGKARG_PREEMPTCOMMAND        pPreemptCommand
    )
{
    UNREFERENCED_PARAMETER(hAdapter);
    UNREFERENCED_PARAMETER(pPreemptCommand);
    return STATUS_SUCCESS;
}

/* ======================================================================
 * Patch - resolve physical addresses in DMA private data
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuPatch(
    IN_CONST_HANDLE                 hAdapter,
    IN_CONST_PDXGKARG_PATCH         pPatch
    )
{
    AMDGPU_DMA_CMD *pCmd;

    UNREFERENCED_PARAMETER(hAdapter);

    if (pPatch == NULL)
        return STATUS_SUCCESS;

    /* Check if we have private data to patch */
    if (pPatch->DmaBufferPrivateDataSize < sizeof(AMDGPU_DMA_CMD))
        return STATUS_SUCCESS;

    pCmd = (AMDGPU_DMA_CMD *)
        ((PUCHAR)pPatch->pDmaBufferPrivateData +
         pPatch->DmaBufferPrivateDataSubmissionStartOffset);

    switch (pCmd->Command) {
    case AMDGPU_CMD_BLT:
        /* Resolve source and destination physical addresses */
        if (pCmd->SrcAllocIndex < pPatch->AllocationListSize) {
            const DXGK_ALLOCATIONLIST *pSrcAlloc =
                &pPatch->pAllocationList[pCmd->SrcAllocIndex];
            pCmd->SrcPhysAddr = pSrcAlloc->PhysicalAddress;
            if (pCmd->SrcPitch == 0) {
                ULONG Width = pCmd->SrcRect.right;
                pCmd->SrcPitch = Width * pCmd->BytesPerPixel;
                pCmd->SrcPitch = (pCmd->SrcPitch + 255) & ~255U;
            }
        }
        if (pCmd->DstAllocIndex < pPatch->AllocationListSize) {
            const DXGK_ALLOCATIONLIST *pDstAlloc =
                &pPatch->pAllocationList[pCmd->DstAllocIndex];
            pCmd->DstPhysAddr = pDstAlloc->PhysicalAddress;
            if (pCmd->DstPitch == 0) {
                ULONG Width = pCmd->DstRect.right;
                pCmd->DstPitch = Width * pCmd->BytesPerPixel;
                pCmd->DstPitch = (pCmd->DstPitch + 255) & ~255U;
            }
        }
        break;

    case AMDGPU_CMD_COLORFILL:
        if (pCmd->DstAllocIndex < pPatch->AllocationListSize) {
            const DXGK_ALLOCATIONLIST *pDstAlloc =
                &pPatch->pAllocationList[pCmd->DstAllocIndex];
            pCmd->DstPhysAddr = pDstAlloc->PhysicalAddress;
        }
        break;

    default:
        break;
    }

    return STATUS_SUCCESS;
}

/* ======================================================================
 * QueryCurrentFence
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryCurrentFence(
    IN_CONST_HANDLE                         hAdapter,
    INOUT_PDXGKARG_QUERYCURRENTFENCE        pCurrentFence
    )
{
    UNREFERENCED_PARAMETER(hAdapter);

    if (pCurrentFence != NULL)
        pCurrentFence->CurrentFence = 0xFFFFFFFF;

    return STATUS_SUCCESS;
}
