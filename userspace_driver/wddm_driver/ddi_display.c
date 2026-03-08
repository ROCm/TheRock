/*
 * ddi_display.c - Child device and POST display DDIs for WDDM miniport
 *
 * Implements:
 *   QueryChildRelations - 1 always-connected video output child
 *   QueryChildStatus    - always connected
 *   QueryDeviceDescriptor - no EDID (STATUS_MONITOR_NO_DESCRIPTOR)
 *   GetChildContainerId - default container
 *   StopDeviceAndReleasePostDisplayOwnership - return POST fb info
 *   SystemDisplayEnable  - BSOD display setup
 *   SystemDisplayWrite   - BSOD pixel output (works at HIGH_IRQL)
 */

#include "amdgpu_wddm.h"

/* ======================================================================
 * Child device DDIs - report 1 always-connected monitor output
 * ====================================================================== */

NTSTATUS
APIENTRY
AmdGpuQueryChildRelations(
    IN_CONST_PVOID                          MiniportDeviceContext,
    INOUT_PDXGK_CHILD_DESCRIPTOR            ChildRelations,
    _In_ ULONG                              ChildRelationsSize
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);

    KdPrint(("AmdGpuWddm: QueryChildRelations size=%u\n", ChildRelationsSize));

    /* Write diagnostic marker */
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
            RtlInitUnicodeString(&ValName, L"QueryChildRelations");
            Val = ChildRelationsSize;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));
            ZwClose(hKey);
        }
    }

    /*
     * ChildRelations array has (NumberOfChildren + 1) entries.
     * We report 1 child. The last entry must be zeroed (sentinel).
     * dxgkrnl pre-zeros the buffer, so we only need to fill entry 0.
     */
    if (ChildRelationsSize < 2 * sizeof(DXGK_CHILD_DESCRIPTOR))
        return STATUS_BUFFER_TOO_SMALL;

    ChildRelations[0].ChildDeviceType = TypeVideoOutput;
    ChildRelations[0].ChildCapabilities.HpdAwareness = HpdAwarenessAlwaysConnected;
    ChildRelations[0].ChildCapabilities.Type.VideoOutput.InterfaceTechnology = D3DKMDT_VOT_OTHER;
    ChildRelations[0].ChildCapabilities.Type.VideoOutput.MonitorOrientationAwareness = D3DKMDT_MOA_NONE;
    ChildRelations[0].ChildCapabilities.Type.VideoOutput.SupportsSdtvModes = FALSE;
    ChildRelations[0].AcpiUid = 0;
    ChildRelations[0].ChildUid = 1;

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuQueryChildStatus(
    IN_CONST_PVOID                  MiniportDeviceContext,
    INOUT_PDXGK_CHILD_STATUS        ChildStatus,
    IN_BOOLEAN                      NonDestructiveOnly
    )
{
    UNREFERENCED_PARAMETER(NonDestructiveOnly);

    KdPrint(("AmdGpuWddm: QueryChildStatus type=%u\n", ChildStatus->Type));

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
            RtlInitUnicodeString(&ValName, L"QueryChildStatus");
            Val = (ULONG)ChildStatus->Type;
            ZwSetValueKey(hKey, &ValName, 0, REG_DWORD, &Val, sizeof(Val));
            ZwClose(hKey);
        }
    }

    if (ChildStatus->Type == StatusConnection) {
        AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

        /*
         * Report connected only if we have a display.
         *
         * For headless compute GPUs (MI100, MI200, MI300, etc.), report
         * disconnected so dxgkrnl doesn't try to establish a display path.
         * The escape channel for compute works regardless of display status.
         *
         * For GPUs with display output (consumer/pro cards), report
         * connected so the POST framebuffer display path works.
         */
        ChildStatus->HotPlug.Connected = !pAdapter->Headless;
        KdPrint(("AmdGpuWddm: QueryChildStatus Connected=%u (Headless=%u)\n",
            ChildStatus->HotPlug.Connected, pAdapter->Headless));
        return STATUS_SUCCESS;
    }

    return STATUS_SUCCESS;
}

NTSTATUS
APIENTRY
AmdGpuQueryDeviceDescriptor(
    IN_CONST_PVOID                          MiniportDeviceContext,
    IN_ULONG                                ChildUid,
    INOUT_PDXGK_DEVICE_DESCRIPTOR           pDeviceDescriptor
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ChildUid);
    UNREFERENCED_PARAMETER(pDeviceDescriptor);

    /*
     * No EDID -- return STATUS_MONITOR_NO_DESCRIPTOR.
     * dxgkrnl will use the monitor modes we provide via
     * RecommendMonitorModes instead.
     */
    return STATUS_MONITOR_NO_DESCRIPTOR;
}

NTSTATUS
APIENTRY
AmdGpuGetChildContainerId(
    IN_CONST_PVOID                          MiniportDeviceContext,
    IN_ULONG                                ChildUid,
    _Inout_ PDXGK_CHILD_CONTAINER_ID       pContainerId
    )
{
    UNREFERENCED_PARAMETER(MiniportDeviceContext);
    UNREFERENCED_PARAMETER(ChildUid);
    UNREFERENCED_PARAMETER(pContainerId);

    /* Return not supported to let dxgkrnl use default container ID */
    return STATUS_NOT_SUPPORTED;
}

/* ======================================================================
 * POST display DDIs
 * ====================================================================== */

/*
 * StopDeviceAndReleasePostDisplayOwnership
 *
 * Called when another driver replaces us, or during device stop.
 * Return the POST framebuffer information so the next driver can
 * continue displaying.
 */
NTSTATUS
AmdGpuStopDeviceAndReleasePostDisplayOwnership(
    _In_ PVOID                          MiniportDeviceContext,
    _In_ D3DDDI_VIDEO_PRESENT_TARGET_ID TargetId,
    _Out_ PDXGK_DISPLAY_INFORMATION     DisplayInfo
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

    UNREFERENCED_PARAMETER(TargetId);

    KdPrint(("AmdGpuWddm: StopDeviceAndReleasePostDisplayOwnership\n"));

    if (!pAdapter->PostDisplay.Acquired) {
        return STATUS_NOT_SUPPORTED;
    }

    DisplayInfo->Width = pAdapter->PostDisplay.Width;
    DisplayInfo->Height = pAdapter->PostDisplay.Height;
    DisplayInfo->Pitch = pAdapter->PostDisplay.Pitch;
    DisplayInfo->ColorFormat = pAdapter->PostDisplay.ColorFormat;
    DisplayInfo->PhysicAddress = pAdapter->PostDisplay.FramebufferPhysAddr;
    DisplayInfo->TargetId = 0;
    DisplayInfo->AcpiId = 0;

    return STATUS_SUCCESS;
}

/*
 * SystemDisplayEnable
 *
 * Called during BSOD / system display path. Must work at HIGH_IRQL.
 * Return the current display dimensions.
 */
NTSTATUS
AmdGpuSystemDisplayEnable(
    _In_ PVOID                              MiniportDeviceContext,
    _In_ D3DDDI_VIDEO_PRESENT_TARGET_ID     TargetId,
    _In_ PDXGKARG_SYSTEM_DISPLAY_ENABLE_FLAGS Flags,
    _Out_ UINT*                             Width,
    _Out_ UINT*                             Height,
    _Out_ D3DDDIFORMAT*                     ColorFormat
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;

    UNREFERENCED_PARAMETER(TargetId);
    UNREFERENCED_PARAMETER(Flags);

    *Width = pAdapter->PostDisplay.Width;
    *Height = pAdapter->PostDisplay.Height;
    *ColorFormat = pAdapter->PostDisplay.ColorFormat;

    return STATUS_SUCCESS;
}

/*
 * SystemDisplayWrite
 *
 * Called during BSOD to render pixels. Must work at ANY IRQL.
 * Copy source pixels to the POST framebuffer kernel VA.
 */
VOID
AmdGpuSystemDisplayWrite(
    _In_ PVOID                      MiniportDeviceContext,
    _In_ PVOID                      Source,
    _In_ UINT                       SourceWidth,
    _In_ UINT                       SourceHeight,
    _In_ UINT                       SourceStride,
    _In_ UINT                       PositionX,
    _In_ UINT                       PositionY
    )
{
    AMDGPU_ADAPTER *pAdapter = (AMDGPU_ADAPTER *)MiniportDeviceContext;
    PUCHAR FbBase;
    PUCHAR SrcBase;
    UINT Row;
    UINT CopyWidth;
    UINT Bpp;

    if (pAdapter->PostDisplay.FramebufferKernelVa == NULL)
        return;

    FbBase = (PUCHAR)pAdapter->PostDisplay.FramebufferKernelVa;
    SrcBase = (PUCHAR)Source;

    /* Determine bytes per pixel from format */
    switch (pAdapter->PostDisplay.ColorFormat) {
    case D3DDDIFMT_A8R8G8B8:
    case D3DDDIFMT_X8R8G8B8:
        Bpp = 4;
        break;
    case D3DDDIFMT_R5G6B5:
        Bpp = 2;
        break;
    default:
        Bpp = 4;
        break;
    }

    /* Clip to framebuffer bounds */
    CopyWidth = SourceWidth;
    if (PositionX + CopyWidth > pAdapter->PostDisplay.Width)
        CopyWidth = pAdapter->PostDisplay.Width - PositionX;

    for (Row = 0; Row < SourceHeight; Row++) {
        UINT DstY = PositionY + Row;
        if (DstY >= pAdapter->PostDisplay.Height)
            break;

        RtlCopyMemory(
            FbBase + DstY * pAdapter->PostDisplay.Pitch + PositionX * Bpp,
            SrcBase + Row * SourceStride,
            CopyWidth * Bpp);
    }
}
