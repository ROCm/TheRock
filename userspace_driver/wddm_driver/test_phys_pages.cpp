/*
 * test_phys_pages.cpp - Minimal test for AMDGPU_ESCAPE_GET_PHYS_PAGES
 *
 * Allocates memory via ALLOC_MEMORY escape, then calls GET_PHYS_PAGES
 * to verify the driver returns physical addresses correctly.
 *
 * Build: cl /nologo /EHsc /O2 test_phys_pages.cpp gdi32.lib
 */
#include <windows.h>
#include <d3dkmthk.h>
#include <stdio.h>
#include <stdlib.h>

/* Inline the escape definitions we need — must match amdgpu_wddm.h */
typedef struct _AMDGPU_ESCAPE_HEADER {
    ULONG Command;      /* AMDGPU_ESCAPE_CODE enum */
    LONG  Status;        /* NTSTATUS */
    ULONG Size;
} AMDGPU_ESCAPE_HEADER;

#define AMDGPU_ESCAPE_GET_INFO       0x0001
#define AMDGPU_ESCAPE_ALLOC_MEMORY   0x0100
#define AMDGPU_ESCAPE_FREE_MEMORY    0x0101
#define AMDGPU_ESCAPE_GET_PHYS_PAGES 0x0160

#define AMDGPU_MEM_TYPE_SYSTEM       0x0004
#define AMDGPU_MEM_FLAG_HOST_ACCESS  0x0020

typedef struct _AMDGPU_ESCAPE_GET_INFO_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    USHORT VendorId;
    USHORT DeviceId;
    USHORT SubsystemVendorId;
    USHORT SubsystemId;
    UCHAR  RevisionId;
    UCHAR  Reserved[3];
    ULONG  NumBars;
    struct {
        LARGE_INTEGER PhysicalAddress;
        ULONGLONG Length;
        BOOLEAN IsMemory;
        BOOLEAN Is64Bit;
        BOOLEAN IsPrefetchable;
        UCHAR   Reserved;
    } Bars[6];
    ULONGLONG VramSizeBytes;
    ULONGLONG VisibleVramSizeBytes;
    ULONG MmioBarIndex;
    ULONG VramBarIndex;
    BOOLEAN Headless;
} AMDGPU_ESCAPE_GET_INFO_DATA;

typedef struct _AMDGPU_ESCAPE_ALLOC_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONG     GpuId;
    ULONGLONG SizeInBytes;
    ULONGLONG Alignment;
    ULONG     Flags;
    ULONGLONG VaAddress;
    /* Output */
    PVOID     CpuAddress;
    ULONGLONG GpuAddress;
    ULONGLONG Handle;
} AMDGPU_ESCAPE_ALLOC_MEMORY_DATA;

typedef struct _AMDGPU_ESCAPE_FREE_MEMORY_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG Handle;
} AMDGPU_ESCAPE_FREE_MEMORY_DATA;

#define AMDGPU_MAX_PHYS_PAGES 256
typedef struct _AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA {
    AMDGPU_ESCAPE_HEADER Header;
    ULONGLONG Handle;
    ULONG     PageOffset;
    ULONG     NumPages;
    ULONG     TotalPages;
    ULONGLONG PhysAddrs[AMDGPU_MAX_PHYS_PAGES];
} AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA;

/* D3DKMT function pointers */
typedef NTSTATUS (APIENTRY *PFN_D3DKMTOpenAdapterFromLuid)(D3DKMT_OPENADAPTERFROMLUID*);
typedef NTSTATUS (APIENTRY *PFN_D3DKMTCloseAdapter)(const D3DKMT_CLOSEADAPTER*);
typedef NTSTATUS (APIENTRY *PFN_D3DKMTEscape)(const D3DKMT_ESCAPE*);
typedef NTSTATUS (APIENTRY *PFN_D3DKMTEnumAdapters2)(D3DKMT_ENUMADAPTERS2*);

static PFN_D3DKMTEscape pfnEscape;

static int do_escape(D3DKMT_HANDLE hAdapter, void *data, ULONG size)
{
    D3DKMT_ESCAPE esc = {};
    esc.hAdapter = hAdapter;
    esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
    esc.pPrivateDriverData = data;
    esc.PrivateDriverDataSize = size;
    NTSTATUS st = pfnEscape(&esc);
    if (st != 0) {
        printf("  D3DKMTEscape NTSTATUS=0x%08x\n", (unsigned)st);
        return -1;
    }
    return 0;
}

int main()
{
    HMODULE hGdi = LoadLibraryA("gdi32.dll");
    if (!hGdi) { printf("FAIL: LoadLibrary gdi32\n"); return 1; }

    auto pfnEnum = (PFN_D3DKMTEnumAdapters2)GetProcAddress(hGdi, "D3DKMTEnumAdapters2");
    auto pfnOpen = (PFN_D3DKMTOpenAdapterFromLuid)GetProcAddress(hGdi, "D3DKMTOpenAdapterFromLuid");
    auto pfnClose = (PFN_D3DKMTCloseAdapter)GetProcAddress(hGdi, "D3DKMTCloseAdapter");
    pfnEscape = (PFN_D3DKMTEscape)GetProcAddress(hGdi, "D3DKMTEscape");

    if (!pfnEnum || !pfnOpen || !pfnClose || !pfnEscape) {
        printf("FAIL: missing D3DKMT functions\n");
        return 1;
    }

    /* Enumerate adapters */
    D3DKMT_ENUMADAPTERS2 enumAdapters = {};
    enumAdapters.NumAdapters = 0;
    enumAdapters.pAdapters = NULL;
    pfnEnum(&enumAdapters);

    if (enumAdapters.NumAdapters == 0) {
        printf("FAIL: no adapters\n");
        return 1;
    }

    D3DKMT_ADAPTERINFO *infos = (D3DKMT_ADAPTERINFO*)malloc(
        enumAdapters.NumAdapters * sizeof(D3DKMT_ADAPTERINFO));
    enumAdapters.pAdapters = infos;
    pfnEnum(&enumAdapters);

    /* Find AMD GPU (VEN_1002) */
    D3DKMT_HANDLE hAdapter = 0;
    for (ULONG i = 0; i < enumAdapters.NumAdapters; i++) {
        D3DKMT_OPENADAPTERFROMLUID open = {};
        open.AdapterLuid = infos[i].AdapterLuid;
        if (pfnOpen(&open) == 0) {
            /* Try GET_INFO to see if it's our driver */
            AMDGPU_ESCAPE_GET_INFO_DATA info = {};
            info.Header.Command = AMDGPU_ESCAPE_GET_INFO;
            info.Header.Size = sizeof(info);
            if (do_escape(open.hAdapter, &info, sizeof(info)) == 0 &&
                info.VendorId == 0x1002) {
                hAdapter = open.hAdapter;
                printf("Found AMD GPU: DeviceId=0x%04x\n", info.DeviceId);
                break;
            }
            D3DKMT_CLOSEADAPTER ca = {};
            ca.hAdapter = open.hAdapter;
            pfnClose(&ca);
        }
    }
    free(infos);

    if (!hAdapter) {
        printf("FAIL: no AMD GPU found\n");
        return 1;
    }

    /* Step 1: Allocate 8 pages (32KB) of system memory */
    printf("\n=== Test 1: Allocate 32KB system memory ===\n");
    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc = {};
    alloc.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    alloc.Header.Size = sizeof(alloc);
    alloc.SizeInBytes = 32768;
    alloc.Flags = AMDGPU_MEM_TYPE_SYSTEM | AMDGPU_MEM_FLAG_HOST_ACCESS;

    if (do_escape(hAdapter, &alloc, sizeof(alloc)) != 0 || alloc.Header.Status != 0) {
        printf("FAIL: ALLOC_MEMORY failed, status=0x%08x\n", (unsigned)alloc.Header.Status);
        return 1;
    }
    printf("OK: handle=%llu, cpu=%p, gpu=0x%llx\n",
           alloc.Handle, alloc.CpuAddress, alloc.GpuAddress);

    /* Step 2: Get physical pages */
    printf("\n=== Test 2: GET_PHYS_PAGES ===\n");
    AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA phys = {};
    phys.Header.Command = AMDGPU_ESCAPE_GET_PHYS_PAGES;
    phys.Header.Size = sizeof(phys);
    phys.Handle = alloc.Handle;
    phys.PageOffset = 0;

    if (do_escape(hAdapter, &phys, sizeof(phys)) != 0 || phys.Header.Status != 0) {
        printf("FAIL: GET_PHYS_PAGES failed, status=0x%08x\n", (unsigned)phys.Header.Status);
        return 1;
    }

    printf("OK: NumPages=%u, TotalPages=%u\n", phys.NumPages, phys.TotalPages);
    for (ULONG i = 0; i < phys.NumPages; i++) {
        printf("  page[%u]: phys=0x%012llx\n", i, phys.PhysAddrs[i]);
    }

    /* Verify: since MmAllocateContiguousMemorySpecifyCache returns
     * contiguous memory, pages should be sequential */
    int contiguous = 1;
    for (ULONG i = 1; i < phys.NumPages; i++) {
        if (phys.PhysAddrs[i] != phys.PhysAddrs[i-1] + 4096) {
            contiguous = 0;
            break;
        }
    }
    printf("Contiguous: %s\n", contiguous ? "YES (expected for MmAllocateContiguous)" : "NO");

    /* Step 3: Test with a larger allocation (256KB = 64 pages) */
    printf("\n=== Test 3: Allocate 256KB ===\n");
    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc2 = {};
    alloc2.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    alloc2.Header.Size = sizeof(alloc2);
    alloc2.SizeInBytes = 256 * 1024;
    alloc2.Flags = AMDGPU_MEM_TYPE_SYSTEM | AMDGPU_MEM_FLAG_HOST_ACCESS;

    if (do_escape(hAdapter, &alloc2, sizeof(alloc2)) != 0 || alloc2.Header.Status != 0) {
        printf("FAIL: ALLOC_MEMORY 256KB failed, status=0x%08x\n", (unsigned)alloc2.Header.Status);
    } else {
        printf("OK: handle=%llu, cpu=%p, gpu=0x%llx\n",
               alloc2.Handle, alloc2.CpuAddress, alloc2.GpuAddress);

        AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA phys2 = {};
        phys2.Header.Command = AMDGPU_ESCAPE_GET_PHYS_PAGES;
        phys2.Header.Size = sizeof(phys2);
        phys2.Handle = alloc2.Handle;
        phys2.PageOffset = 0;

        if (do_escape(hAdapter, &phys2, sizeof(phys2)) != 0 || phys2.Header.Status != 0) {
            printf("FAIL: GET_PHYS_PAGES 256KB failed, status=0x%08x\n", (unsigned)phys2.Header.Status);
        } else {
            printf("OK: NumPages=%u, TotalPages=%u, first=0x%012llx last=0x%012llx\n",
                   phys2.NumPages, phys2.TotalPages,
                   phys2.PhysAddrs[0],
                   phys2.PhysAddrs[phys2.NumPages - 1]);
        }

        /* Free the 256KB allocation */
        AMDGPU_ESCAPE_FREE_MEMORY_DATA free2 = {};
        free2.Header.Command = AMDGPU_ESCAPE_FREE_MEMORY;
        free2.Header.Size = sizeof(free2);
        free2.Handle = alloc2.Handle;
        do_escape(hAdapter, &free2, sizeof(free2));
    }

    /* Free the 32KB allocation */
    AMDGPU_ESCAPE_FREE_MEMORY_DATA free1 = {};
    free1.Header.Command = AMDGPU_ESCAPE_FREE_MEMORY;
    free1.Header.Size = sizeof(free1);
    free1.Handle = alloc.Handle;
    do_escape(hAdapter, &free1, sizeof(free1));

    printf("\n=== All tests passed ===\n");

    D3DKMT_CLOSEADAPTER ca = {};
    ca.hAdapter = hAdapter;
    pfnClose(&ca);
    return 0;
}
