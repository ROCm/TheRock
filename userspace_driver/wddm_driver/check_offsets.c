/* Quick compile check for DXGK_DRIVERCAPS field offsets */
#include <ntddk.h>
#include <dispmprt.h>
#include <d3dkmddi.h>
#include <d3dkmdt.h>
#include <stdio.h>
#include <stddef.h>

/* Dummy function to force compile and see offsets in output */
void check(void) {
    /* Use pragma to print at compile time */
    #pragma message("sizeof(DXGK_DRIVERCAPS) = " __pragma(warning(suppress:4474)) )
}

/* We'll use a different approach - build a tiny .c that prints at runtime */
int main(void) {
    printf("sizeof(DXGK_DRIVERCAPS) = %zu\n", sizeof(DXGK_DRIVERCAPS));
    printf("offsetof WDDMVersion = %zu\n", offsetof(DXGK_DRIVERCAPS, WDDMVersion));
    printf("offsetof MemoryManagementCaps = %zu\n", offsetof(DXGK_DRIVERCAPS, MemoryManagementCaps));
    printf("offsetof SupportNonVGA = %zu\n", offsetof(DXGK_DRIVERCAPS, SupportNonVGA));
    printf("offsetof SupportPerEngineTDR = %zu\n", offsetof(DXGK_DRIVERCAPS, SupportPerEngineTDR));
    printf("offsetof PreemptionCaps = %zu\n", offsetof(DXGK_DRIVERCAPS, PreemptionCaps));
    printf("offsetof PresentationCaps = %zu\n", offsetof(DXGK_DRIVERCAPS, PresentationCaps));
    return 0;
}
