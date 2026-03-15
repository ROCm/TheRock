/**
 * hello_hip.c
 *
 * Self-contained "Hello World" HIP program for hip-remote.
 * The GPU kernel (vector_add) is embedded as a byte array at build time.
 *
 * Build (from a Developer Command Prompt):
 *   1. Compile the kernel:
 *      amdclang++ --offload-arch=gfx942 --cuda-device-only -o vector_add.co -x hip vector_add_kernel.hip
 *   2. Generate the embedded header:
 *      python -c "d=open('vector_add.co','rb').read(); print('static const unsigned char vector_add_co[] = {'+','.join(f'0x{b:02x}' for b in d)+'};'); print(f'static const size_t vector_add_co_size = {len(d)};')" > vector_add_co.h
 *   3. Compile the host program:
 *      cl /I ../../core/hip-remote-client/include hello_hip.c /link ../../build-hip-remote/amdhip64.lib
 *
 * Run:
 *   set TF_WORKER_HOST=<linux-ip>
 *   hello_hip.exe
 *
 * Or just run: build.bat
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hip_remote/hip_remote_client.h"

/* Embedded GPU code object — generated at build time from vector_add_kernel.hip */
#include "vector_add_co.h"

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, hipGetErrorString(err), err); \
        exit(1); \
    } \
} while (0)

int main(void) {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    /* ---- Query remote GPU ---- */
    printf("=== Hello HIP (Remote) ===\n\n");

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    printf("Remote GPU count: %d\n", device_count);

    char props[4096];
    memset(props, 0, sizeof(props));
    HIP_CHECK(hipGetDeviceProperties(props, 0));
    printf("Device 0: %s\n\n", props);

    HIP_CHECK(hipSetDevice(0));

    /* ---- Load embedded kernel ---- */
    printf("Loading embedded kernel (%zu bytes of gfx942 ISA)\n", vector_add_co_size);

    hipModule_t module = NULL;
    HIP_CHECK(hipModuleLoadData(&module, vector_add_co));

    hipFunction_t kernel = NULL;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "vector_add"));
    printf("Kernel ready.\n\n");

    /* ---- Prepare data ---- */
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    void *d_a = NULL, *d_b = NULL, *d_c = NULL;
    HIP_CHECK(hipMalloc(&d_a, size));
    HIP_CHECK(hipMalloc(&d_b, size));
    HIP_CHECK(hipMalloc(&d_c, size));

    HIP_CHECK(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

    /* ---- Launch kernel ---- */
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int n = N;
    void* args[] = { &d_a, &d_b, &d_c, &n, NULL };

    printf("vector_add<<<%d, %d>>>(%d elements)...\n", blocks, threads, N);

    HIP_CHECK(hipModuleLaunchKernel(
        kernel,
        (unsigned int)blocks, 1, 1,
        (unsigned int)threads, 1, 1,
        0, NULL, args, NULL
    ));
    HIP_CHECK(hipDeviceSynchronize());

    /* ---- Verify ---- */
    HIP_CHECK(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (fabsf(h_c[i] - expected) > 1e-5f) {
            if (errors < 5)
                printf("  MISMATCH [%d]: %.1f != %.1f\n", i, h_c[i], expected);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASSED: all %d elements correct\n", N);
        printf("  c[0]=%g  c[511]=%g  c[1023]=%g\n", h_c[0], h_c[511], h_c[1023]);
    } else {
        printf("FAILED: %d errors\n", errors);
    }

    /* ---- Cleanup ---- */
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipModuleUnload(module));
    free(h_a); free(h_b); free(h_c);

    return errors > 0 ? 1 : 0;
}
