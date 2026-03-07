/*
 * ddi_interrupt.c - Interrupt DDIs for ROCm Display Driver
 *
 * Identical to MCDM: ISR reads IH ring, DPC signals registered events.
 */

#include "amdgpu_wddm.h"

#define IH_ENTRY_SIZE   32

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

    if (!pAdapter->IhRing.Configured)
        return FALSE;

    if (!pAdapter->Bars[0].Mapped || pAdapter->Bars[0].KernelAddress == NULL)
        return FALSE;

    if (pAdapter->IhRing.WptrRegOffset + sizeof(ULONG) >
        pAdapter->Bars[0].Length)
        return FALSE;

    Wptr = READ_REGISTER_ULONG(
        (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress +
                 pAdapter->IhRing.WptrRegOffset));

    Wptr &= pAdapter->IhRing.RingMask;

    if (Wptr == pAdapter->IhRing.Rptr)
        return FALSE;

    InterlockedExchange(&pAdapter->IhPending, 1);
    pAdapter->DxgkInterface.DxgkCbQueueDpc(
        pAdapter->DxgkInterface.DeviceHandle);

    return TRUE;
}

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
        Wptr = READ_REGISTER_ULONG(
            (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress +
                     pAdapter->IhRing.WptrRegOffset));
        Wptr &= pAdapter->IhRing.RingMask;

        MaxEntries = pAdapter->IhRing.RingSize / IH_ENTRY_SIZE;
        Count = 0;

        while (pAdapter->IhRing.Rptr != Wptr && Count < MaxEntries) {
            PULONG Entry = (PULONG)(
                (PUCHAR)pAdapter->IhRing.RingBuffer +
                pAdapter->IhRing.Rptr);
            ULONG SourceId = (Entry[0] >> 8) & 0xFF;
            ULONG i;

            for (i = 0; i < AMDGPU_MAX_EVENTS; i++) {
                if (pAdapter->Events[i].InUse &&
                    pAdapter->Events[i].SourceId == SourceId) {
                    KeSetEvent(pAdapter->Events[i].Event,
                               IO_NO_INCREMENT, FALSE);
                }
            }

            pAdapter->IhRing.Rptr =
                (pAdapter->IhRing.Rptr + IH_ENTRY_SIZE) &
                pAdapter->IhRing.RingMask;
            Count++;
        }

        if (pAdapter->IhRing.RptrRegOffset + sizeof(ULONG) <=
            pAdapter->Bars[0].Length) {
            WRITE_REGISTER_ULONG(
                (PULONG)((PUCHAR)pAdapter->Bars[0].KernelAddress +
                         pAdapter->IhRing.RptrRegOffset),
                pAdapter->IhRing.Rptr);
        }
    }

    pAdapter->DxgkInterface.DxgkCbNotifyDpc(
        pAdapter->DxgkInterface.DeviceHandle);
}
