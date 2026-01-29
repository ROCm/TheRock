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
 * @file hip_remote_client.h
 * @brief Internal client API for remote HIP execution
 *
 * This header provides the internal API for managing connections to
 * the remote HIP worker service and sending/receiving protocol messages.
 */

#ifndef HIP_REMOTE_CLIENT_H
#define HIP_REMOTE_CLIENT_H

#include "hip_remote_protocol.h"
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error Codes (matching hipError_t values from hip_runtime_api.h)
 *
 * These definitions match the HIP runtime exactly. The error string functions
 * (hipGetErrorName, hipGetErrorString) are ported from hip_error.cpp.
 * ============================================================================ */

typedef int hipError_t;

#define hipSuccess                      0
#define hipErrorInvalidValue            1
#define hipErrorOutOfMemory             2
#define hipErrorNotInitialized          3
#define hipErrorDeinitialized           4
#define hipErrorProfilerDisabled        5
#define hipErrorProfilerNotInitialized  6
#define hipErrorProfilerAlreadyStarted  7
#define hipErrorProfilerAlreadyStopped  8
#define hipErrorInvalidConfiguration    9
#define hipErrorInvalidPitchValue       12
#define hipErrorInvalidSymbol           13
#define hipErrorInvalidDevicePointer    17
#define hipErrorInvalidMemcpyDirection  21
#define hipErrorInsufficientDriver      35
#define hipErrorMissingConfiguration    52
#define hipErrorPriorLaunchFailure      53
#define hipErrorInvalidDeviceFunction   98
#define hipErrorNoDevice                100
#define hipErrorInvalidDevice           101
#define hipErrorInvalidImage            200
#define hipErrorInvalidContext          201
#define hipErrorContextAlreadyCurrent   202
#define hipErrorMapFailed               205
#define hipErrorUnmapFailed             206
#define hipErrorArrayIsMapped           207
#define hipErrorAlreadyMapped           208
#define hipErrorNoBinaryForGpu          209
#define hipErrorAlreadyAcquired         210
#define hipErrorNotMapped               211
#define hipErrorNotMappedAsArray        212
#define hipErrorNotMappedAsPointer      213
#define hipErrorECCNotCorrectable       214
#define hipErrorUnsupportedLimit        215
#define hipErrorContextAlreadyInUse     216
#define hipErrorPeerAccessUnsupported   217
#define hipErrorInvalidKernelFile       218
#define hipErrorInvalidGraphicsContext  219
#define hipErrorInvalidSource           300
#define hipErrorFileNotFound            301
#define hipErrorSharedObjectSymbolNotFound 302
#define hipErrorSharedObjectInitFailed  303
#define hipErrorOperatingSystem         304
#define hipErrorInvalidHandle           400
#define hipErrorInvalidResourceHandle   400
#define hipErrorIllegalState            401
#define hipErrorNotFound                500
#define hipErrorNotReady                600
#define hipErrorIllegalAddress          700
#define hipErrorLaunchOutOfResources    701
#define hipErrorLaunchTimeOut           702
#define hipErrorPeerAccessAlreadyEnabled 704
#define hipErrorPeerAccessNotEnabled    705
#define hipErrorSetOnActiveProcess      708
#define hipErrorContextIsDestroyed      709
#define hipErrorAssert                  710
#define hipErrorHostMemoryAlreadyRegistered 712
#define hipErrorHostMemoryNotRegistered 713
#define hipErrorLaunchFailure           719
#define hipErrorCooperativeLaunchTooLarge 720
#define hipErrorNotSupported            801
#define hipErrorStreamCaptureUnsupported 900
#define hipErrorStreamCaptureInvalidated 901
#define hipErrorStreamCaptureMerge      902
#define hipErrorStreamCaptureUnmatched  903
#define hipErrorStreamCaptureUnjoined   904
#define hipErrorStreamCaptureIsolation  905
#define hipErrorStreamCaptureImplicit   906
#define hipErrorCapturedEvent           907
#define hipErrorStreamCaptureWrongThread 908
#define hipErrorGraphExecUpdateFailure  910
#define hipErrorUnknown                 999
#define hipErrorRuntimeMemory           1052
#define hipErrorRuntimeOther            1053
#define hipErrorTbd                     1054

/* ============================================================================
 * Memory Copy Kind
 * ============================================================================ */

typedef enum {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

/* ============================================================================
 * Type Definitions
 * ============================================================================ */

typedef void* hipStream_t;

/* ============================================================================
 * Client State
 * ============================================================================ */

/**
 * Client connection state
 */
typedef struct {
    int socket_fd;                  /**< Socket file descriptor */
    pthread_mutex_t lock;           /**< Mutex for thread safety */
    uint32_t next_request_id;       /**< Next request ID */
    bool connected;                 /**< Connection status */
    bool debug_enabled;             /**< Debug logging enabled */
    char worker_host[256];          /**< Worker hostname */
    int worker_port;                /**< Worker port */
    int connect_timeout_sec;        /**< Connection timeout (seconds) */
    int io_timeout_sec;             /**< I/O timeout (seconds) */
    hipError_t last_error;          /**< Last error code */
} HipRemoteClientState;

/**
 * Get the global client state.
 * Thread-safe after first call.
 */
HipRemoteClientState* hip_remote_get_client_state(void);

/* ============================================================================
 * Connection Management
 * ============================================================================ */

/**
 * Ensure client is connected to worker.
 * Automatically connects on first call or after disconnect.
 *
 * @return 0 on success, -1 on failure
 */
int hip_remote_ensure_connected(void);

/**
 * Disconnect from worker.
 */
void hip_remote_disconnect(void);

/**
 * Check if client is connected.
 */
bool hip_remote_is_connected(void);

/* ============================================================================
 * Message I/O
 * ============================================================================ */

/**
 * Send a request and receive a response (synchronous).
 *
 * @param op_code Operation code
 * @param request Request payload (may be NULL)
 * @param request_size Request payload size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
);

/**
 * Send a request with inline data and receive response.
 *
 * @param op_code Operation code
 * @param request Request payload
 * @param request_size Request payload size
 * @param data Inline data to send
 * @param data_size Inline data size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request_with_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size,
    void* response,
    size_t response_size
);

/**
 * Send a request and receive response with inline data.
 *
 * @param op_code Operation code
 * @param request Request payload
 * @param request_size Request payload size
 * @param response Response buffer
 * @param response_size Response buffer size
 * @param data_out Buffer for received inline data
 * @param data_size Size of data to receive
 * @return hipError_t from response, or error if communication failed
 */
hipError_t hip_remote_request_receive_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size,
    void* data_out,
    size_t data_size
);

/* ============================================================================
 * Logging
 * ============================================================================ */

/**
 * Log a debug message (only if debug enabled).
 */
void hip_remote_log_debug(const char* fmt, ...);

/**
 * Log an error message.
 */
void hip_remote_log_error(const char* fmt, ...);

/* ============================================================================
 * HIP API Functions
 * These functions implement the HIP runtime API by forwarding to the worker.
 * ============================================================================ */

/* Device Management */
hipError_t hipGetDeviceCount(int* count);
hipError_t hipSetDevice(int deviceId);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceProperties(void* prop, int deviceId);
hipError_t hipDeviceGetAttribute(int* value, int attr, int deviceId);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceReset(void);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipRuntimeGetVersion(int* runtimeVersion);

/* Error Handling */
const char* hipGetErrorString(hipError_t error);
const char* hipGetErrorName(hipError_t error);
hipError_t hipGetLastError(void);
hipError_t hipPeekAtLastError(void);

/* Memory Management */
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMallocHost(void** ptr, size_t size);
hipError_t hipFreeHost(void* ptr);
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, void* stream);
hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t sizeBytes);
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, void* stream);
hipError_t hipMemGetInfo(size_t* free, size_t* total);

/* Stream Management */
hipError_t hipStreamCreate(void** stream);
hipError_t hipStreamCreateWithFlags(void** stream, unsigned int flags);
hipError_t hipStreamCreateWithPriority(void** stream, unsigned int flags, int priority);
hipError_t hipStreamDestroy(void* stream);
hipError_t hipStreamSynchronize(void* stream);
hipError_t hipStreamWaitEvent(void* stream, void* event, unsigned int flags);
hipError_t hipStreamQuery(void* stream);

/* Event Management */
hipError_t hipEventCreate(void** event);
hipError_t hipEventCreateWithFlags(void** event, unsigned int flags);
hipError_t hipEventDestroy(void* event);
hipError_t hipEventRecord(void* event, void* stream);
hipError_t hipEventSynchronize(void* event);
hipError_t hipEventQuery(void* event);
hipError_t hipEventElapsedTime(float* ms, void* start, void* stop);

/* Module Management */
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef enum {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock,
    hipJitOptionInfoLogBuffer,
    hipJitOptionInfoLogBufferSizeBytes,
    hipJitOptionErrorLogBuffer,
    hipJitOptionErrorLogBufferSizeBytes,
    hipJitOptionOptimizationLevel,
    hipJitOptionTargetFromContext,
    hipJitOptionTarget,
    hipJitOptionFallbackStrategy,
    hipJitOptionGenerateDebugInfo,
    hipJitOptionLogVerbose,
    hipJitOptionGenerateLineInfo,
    hipJitOptionCacheMode,
    hipJitOptionNumOptions
} hipJitOption;

hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                unsigned int numOptions, hipJitOption* options,
                                void** optionValues);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                 const char* kname);

/* Kernel Launch */
typedef struct {
    unsigned int x, y, z;
} dim3;

/**
 * Launch a kernel function.
 *
 * The remote HIP client queries kernel metadata from the worker to determine
 * the number of arguments (requires ROCm 7.2+ on the worker). For older ROCm
 * versions, the kernelParams array must be NULL-terminated:
 *
 *     void* args[] = { &d_a, &d_b, &d_c, &N, NULL };
 *     hipModuleLaunchKernel(function, gridX, 1, 1, blockX, 1, 1, 0, stream, args, NULL);
 *
 * Note: All arguments are currently assumed to be pointer-sized (8 bytes).
 * The 'extra' parameter is not supported in remote mode.
 */
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gridDimX,
                                  unsigned int gridDimY,
                                  unsigned int gridDimZ,
                                  unsigned int blockDimX,
                                  unsigned int blockDimY,
                                  unsigned int blockDimZ,
                                  unsigned int sharedMemBytes,
                                  hipStream_t stream,
                                  void** kernelParams,
                                  void** extra);

hipError_t hipLaunchKernel(const void* function_address,
                            dim3 numBlocks,
                            dim3 dimBlocks,
                            void** args,
                            size_t sharedMemBytes,
                            hipStream_t stream);

hipError_t hipLaunchCooperativeKernel(const void* f,
                                       dim3 gridDim,
                                       dim3 blockDim,
                                       void** kernelParams,
                                       unsigned int sharedMemBytes,
                                       hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_CLIENT_H */
