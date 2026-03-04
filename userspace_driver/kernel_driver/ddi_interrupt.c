/*
 * ddi_interrupt.c - Interrupt DDIs for MCDM miniport
 *
 * Implements the WDDM interrupt model:
 *   ISR (InterruptRoutine) at DIRQL — check IH ring for new entries
 *   DPC (DpcRoutine) at DISPATCH_LEVEL — process entries, signal events
 *
 * The IH (Interrupt Handler) ring is a circular buffer in DMA memory
 * that the GPU writes interrupt entries to. Each entry is 4 DWORDs:
 *   DWORD 0: source_id[7:0], source_data[31:8]
 *   DWORD 1: ring_id[7:0], vm_id[15:8], pasid[31:16]
 *   DWORD 2: timestamp_lo
 *   DWORD 3: timestamp_hi
 *
 * Python allocates the ring (ALLOC_DMA), programs IH registers
 * (WRITE_REG32), then calls ENABLE_MSI to tell us the ring location
 * and register offsets. Events registered via REGISTER_EVENT are
 * signaled when matching IH entries arrive.
 */

#include "amdgpu_mcdm.h"

/* IH entry size in bytes (4 x DWORD) */
#define IH_ENTRY_SIZE   32

/* ======================================================================
 * InterruptRoutine — ISR at DIRQL
 *
 * Called by dxgkrnl when the device signals an interrupt.
 * We read the IH ring write pointer from BAR0 MMIO. If it differs
 * from our cached read pointer, the GPU has written new entries.
 *
 * Must be fast: just detect new work and queue the DPC.
 * ====================================================================== */

BOOLEAN
APIENTRY
AmdGpuInterruptRoutine(
    IN_CONST_PVOID  MiniportDeviceContext,
    IN_ULONG        MessageNumber
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;
    ULONG Wptr;

    UNREFERENCED_PARAMETER(MessageNumber);

    /* If IH ring is not configured, this interrupt isn't ours */
    if (!pAdapter->IhRing.Configured)
        return FALSE;

    if (!pAdapter->Bars[0].Mapped || pAdapter->Bars[0].KernelAddress == NULL)
        return FALSE;

    /* Bounds-check the WPTR register offset before reading */
    if (pAdapter->IhRing.WptrRegOffset + sizeof(ULONG) >
        pAdapter->Bars[0].Length)
        return FALSE;

    /* Read IH_RB_WPTR from BAR0 */
    Wptr = READ_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress +
                 pAdapter->IhRing.WptrRegOffset));

    /* Mask to ring size (WPTR is in bytes) */
    Wptr &= pAdapter->IhRing.RingMask;

    /* If write pointer matches read pointer, no new entries — not our IRQ */
    if (Wptr == pAdapter->IhRing.Rptr)
        return FALSE;

    /* Flag new work and queue DPC for processing */
    InterlockedExchange(&pAdapter->IhPending, 1);
    pAdapter->DxgkInterface.DxgkCbQueueDpc(
        pAdapter->DxgkInterface.DeviceHandle);

    return TRUE;
}

/* ======================================================================
 * DpcRoutine — deferred procedure call after ISR
 *
 * Runs at DISPATCH_LEVEL. Processes IH ring entries and signals
 * registered Windows Events for matching interrupt sources.
 * ====================================================================== */

VOID
APIENTRY
AmdGpuDpcRoutine(
    IN_CONST_PVOID  MiniportDeviceContext
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;
    ULONG Wptr;
    ULONG MaxEntries;
    ULONG Count;

    if (InterlockedExchange(&pAdapter->IhPending, 0)) {
        /* Re-read WPTR (may have advanced since ISR) */
        Wptr = READ_REGISTER_ULONG(
            (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress +
                     pAdapter->IhRing.WptrRegOffset));
        Wptr &= pAdapter->IhRing.RingMask;

        /* Process entries. Cap at ring size / entry size to avoid
         * infinite loop if hardware reports garbage. */
        MaxEntries = pAdapter->IhRing.RingSize / IH_ENTRY_SIZE;
        Count = 0;

        while (pAdapter->IhRing.Rptr != Wptr && Count < MaxEntries) {
            PULONG Entry = (PULONG)(
                (PUCHAR)pAdapter->IhRing.RingBuffer +
                pAdapter->IhRing.Rptr);
            /*
             * IH entry format (8 DWORDs):
             *   DW[0]: [7:0]=client_id, [15:8]=source_id, [23:16]=ring_id,
             *          [27:24]=vmid, [31]=vmid_src
             *   DW[1]: timestamp_lo
             *   DW[2]: [15:0]=timestamp_hi, [31]=timestamp_src
             *   DW[3]: [15:0]=pasid, [23:16]=node_id
             *   DW[4-7]: src_data[0-3]
             */
            ULONG SourceId = (Entry[0] >> 8) & 0xFF;
            ULONG i;

            /* Signal all events registered for this source ID */
            for (i = 0; i < AMDGPU_MAX_EVENTS; i++) {
                if (pAdapter->Events[i].InUse &&
                    pAdapter->Events[i].SourceId == SourceId) {
                    KeSetEvent(pAdapter->Events[i].Event,
                               IO_NO_INCREMENT, FALSE);
                }
            }

            /* Advance read pointer (wrapping) */
            pAdapter->IhRing.Rptr =
                (pAdapter->IhRing.Rptr + IH_ENTRY_SIZE) &
                pAdapter->IhRing.RingMask;
            Count++;
        }

        /* Update RPTR register so GPU knows we've consumed entries */
        if (pAdapter->IhRing.RptrRegOffset + sizeof(ULONG) <=
            pAdapter->Bars[0].Length) {
            WRITE_REGISTER_ULONG(
                (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress +
                         pAdapter->IhRing.RptrRegOffset),
                pAdapter->IhRing.Rptr);
        }
    }

    /* Notify dxgkrnl that DPC processing is complete */
    pAdapter->DxgkInterface.DxgkCbNotifyDpc(
        pAdapter->DxgkInterface.DeviceHandle);
}
