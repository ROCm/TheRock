/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_basic.c
 * @brief Basic integration tests for remote HIP client
 *
 * These tests require a running HIP worker service.
 * Set TF_WORKER_HOST and TF_WORKER_PORT environment variables.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip_remote/hip_remote_client.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n  %s:%d\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

#define TEST_HIP_OK(call) do { \
    hipError_t _err = (call); \
    if (_err != hipSuccess) { \
        fprintf(stderr, "FAIL: %s returned %d (%s)\n  %s:%d\n", \
                #call, _err, hipGetErrorString(_err), __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static int test_device_count(void) {
    printf("Test: hipGetDeviceCount\n");

    int count = -1;
    TEST_HIP_OK(hipGetDeviceCount(&count));
    TEST_ASSERT(count >= 0, "Device count should be non-negative");

    printf("  Found %d device(s)\n", count);
    return 0;
}

static int test_set_get_device(void) {
    printf("Test: hipSetDevice / hipGetDevice\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Set device 0 */
    TEST_HIP_OK(hipSetDevice(0));

    /* Get current device */
    int device = -1;
    TEST_HIP_OK(hipGetDevice(&device));
    TEST_ASSERT(device == 0, "Device should be 0");

    printf("  Set and get device 0: OK\n");
    return 0;
}

static int test_device_properties(void) {
    printf("Test: hipGetDeviceProperties\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Allocate buffer for properties */
    char prop_buffer[4096];
    memset(prop_buffer, 0, sizeof(prop_buffer));

    TEST_HIP_OK(hipGetDeviceProperties(prop_buffer, 0));

    /* Check that name is not empty */
    TEST_ASSERT(prop_buffer[0] != '\0', "Device name should not be empty");

    printf("  Device 0: %s\n", prop_buffer);
    return 0;
}

static int test_malloc_free(void) {
    printf("Test: hipMalloc / hipFree\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    void* ptr = NULL;
    size_t size = 1024 * 1024;  /* 1MB */

    TEST_HIP_OK(hipMalloc(&ptr, size));
    TEST_ASSERT(ptr != NULL, "Pointer should not be NULL");

    TEST_HIP_OK(hipFree(ptr));

    printf("  Allocated and freed 1MB: OK\n");
    return 0;
}

static int test_memcpy(void) {
    printf("Test: hipMemcpy\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    /* Allocate host and device memory */
    size_t size = 1024;
    unsigned char* host_src = (unsigned char*)malloc(size);
    unsigned char* host_dst = (unsigned char*)malloc(size);
    void* device_ptr = NULL;

    TEST_ASSERT(host_src != NULL, "Host source allocation failed");
    TEST_ASSERT(host_dst != NULL, "Host destination allocation failed");

    /* Initialize source */
    for (size_t i = 0; i < size; i++) {
        host_src[i] = (unsigned char)(i & 0xFF);
    }
    memset(host_dst, 0, size);

    /* Allocate device memory */
    TEST_HIP_OK(hipMalloc(&device_ptr, size));

    /* Copy to device */
    TEST_HIP_OK(hipMemcpy(device_ptr, host_src, size, hipMemcpyHostToDevice));

    /* Copy back */
    TEST_HIP_OK(hipMemcpy(host_dst, device_ptr, size, hipMemcpyDeviceToHost));

    /* Verify */
    int match = 1;
    for (size_t i = 0; i < size; i++) {
        if (host_src[i] != host_dst[i]) {
            match = 0;
            break;
        }
    }
    TEST_ASSERT(match, "Data mismatch after round-trip");

    /* Cleanup */
    TEST_HIP_OK(hipFree(device_ptr));
    free(host_src);
    free(host_dst);

    printf("  Round-trip memcpy 1KB: OK\n");
    return 0;
}

static int test_stream_create_destroy(void) {
    printf("Test: hipStreamCreate / hipStreamDestroy\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    void* stream = NULL;

    /* Note: hipStreamCreate takes hipStream_t* which is void** */
    TEST_HIP_OK(hipStreamCreate(&stream));
    TEST_ASSERT(stream != NULL, "Stream should not be NULL");

    TEST_HIP_OK(hipStreamSynchronize(stream));
    TEST_HIP_OK(hipStreamDestroy(stream));

    printf("  Created and destroyed stream: OK\n");
    return 0;
}

static int test_event_create_destroy(void) {
    printf("Test: hipEventCreate / hipEventDestroy\n");

    int count = 0;
    TEST_HIP_OK(hipGetDeviceCount(&count));

    if (count == 0) {
        printf("  SKIP: No devices available\n");
        return 0;
    }

    void* event = NULL;

    TEST_HIP_OK(hipEventCreate(&event));
    TEST_ASSERT(event != NULL, "Event should not be NULL");

    TEST_HIP_OK(hipEventDestroy(event));

    printf("  Created and destroyed event: OK\n");
    return 0;
}

static int test_runtime_version(void) {
    printf("Test: hipRuntimeGetVersion\n");

    int version = 0;
    TEST_HIP_OK(hipRuntimeGetVersion(&version));
    TEST_ASSERT(version > 0, "Version should be positive");

    printf("  Runtime version: %d\n", version);
    return 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Remote HIP Basic Tests ===\n\n");

    int failures = 0;

    /* Check if worker is configured */
    const char* host = getenv("TF_WORKER_HOST");
    if (!host || host[0] == '\0') {
        printf("NOTE: TF_WORKER_HOST not set, using localhost\n");
    }

    /* Run tests */
    failures += test_device_count();
    failures += test_set_get_device();
    failures += test_device_properties();
    failures += test_malloc_free();
    failures += test_memcpy();
    failures += test_stream_create_destroy();
    failures += test_event_create_destroy();
    failures += test_runtime_version();

    printf("\n=== Results ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
}
