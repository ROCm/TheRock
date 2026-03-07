/*
 * ddi_present.c - Software Present DDI for ROCm Display Driver
 *
 * DWM uses WARP (software D3D) as UMD, renders to shadow surfaces,
 * then calls Present. We encode a DMA command describing the blit,
 * which is later executed by SubmitCommand as a CPU blit to the
 * POST framebuffer.
 *
 * DMA buffer flow:
 *   DxgkDdiPresent -> DxgkDdiPatch -> DxgkDdiSubmitCommand
 *   (build command)   (resolve addrs)   (execute CPU blit)
 */

#include "amdgpu_wddm.h"

/* DXGK_PRESENT_SOURCE_INDEX and DXGK_PRESENT_DESTINATION_INDEX */
#define PRESENT_SRC_INDEX   0
#define PRESENT_DST_INDEX   1

/* ======================================================================
 * DxgkDdiPresent - encode software blit command
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuPresent(
    IN_CONST_HANDLE         hContext,
    INOUT_PDXGKARG_PRESENT  pPresent
    )
{
    AMDGPU_DMA_CMD *pCmd;
    PULONG pDmaBuffer;

    UNREFERENCED_PARAMETER(hContext);

    if (pPresent == NULL)
        return STATUS_INVALID_PARAMETER;

    /* Check that we have space for our private command */
    if (pPresent->pDmaBufferPrivateData == NULL ||
        pPresent->DmaBufferPrivateDataSize < sizeof(AMDGPU_DMA_CMD)) {
        return STATUS_GRAPHICS_INSUFFICIENT_DMA_BUFFER;
    }

    /* Check DMA buffer space for minimum 1 DWORD */
    if (pPresent->DmaSize < sizeof(ULONG)) {
        return STATUS_GRAPHICS_INSUFFICIENT_DMA_BUFFER;
    }

    pCmd = (AMDGPU_DMA_CMD *)pPresent->pDmaBufferPrivateData;
    RtlZeroMemory(pCmd, sizeof(*pCmd));

    if (pPresent->Flags.Blt) {
        pCmd->Command = AMDGPU_CMD_BLT;
        pCmd->SrcAllocIndex = PRESENT_SRC_INDEX;
        pCmd->DstAllocIndex = PRESENT_DST_INDEX;
        pCmd->SrcRect = pPresent->SrcRect;
        pCmd->DstRect = pPresent->DstRect;

        /*
         * Determine if destination is the primary surface.
         * dxgkrnl sets pPresent->pAllocationList[DST_INDEX].hDeviceSpecificAllocation
         * to the allocation handle. We check if it's marked as primary
         * in Patch when we have access to physical addresses.
         * For now, assume Dst is always the primary for display Present.
         */
        pCmd->DstIsPrimary = TRUE;

        /* Default to 32bpp -- will be refined if we have allocation info */
        pCmd->BytesPerPixel = 4;

    } else if (pPresent->Flags.ColorFill) {
        pCmd->Command = AMDGPU_CMD_COLORFILL;
        pCmd->DstAllocIndex = PRESENT_DST_INDEX;
        pCmd->DstRect = pPresent->DstRect;
        pCmd->FillColor = pPresent->Color;
        pCmd->DstIsPrimary = TRUE;
        pCmd->BytesPerPixel = 4;

    } else if (pPresent->Flags.Flip) {
        /* No hardware flip support -- return not supported */
        return STATUS_NOT_SUPPORTED;

    } else {
        /* Unknown present type -- encode as NOP */
        pCmd->Command = AMDGPU_CMD_NOP;
    }

    /* Write minimum DMA buffer content (1 DWORD NOP marker) */
    pDmaBuffer = (PULONG)pPresent->pDmaBuffer;
    *pDmaBuffer = 0x00000000;

    /* Advance pointers */
    pPresent->pDmaBuffer = (PUCHAR)pPresent->pDmaBuffer + sizeof(ULONG);
    pPresent->pDmaBufferPrivateData =
        (PUCHAR)pPresent->pDmaBufferPrivateData + sizeof(AMDGPU_DMA_CMD);

    return STATUS_SUCCESS;
}
