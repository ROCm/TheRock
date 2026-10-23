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
 * @file hip_remote_protocol.h
 * @brief Binary protocol for remote HIP API execution
 *
 * This protocol enables HIP API calls to be forwarded from a client
 * (e.g., macOS development machine) to a worker service running on
 * a Linux system with AMD GPUs.
 *
 * Protocol design principles:
 * - Fixed-size headers for efficient parsing
 * - Request-response model with correlation IDs
 * - Support for bulk data transfer (memory copies)
 * - Extensible via reserved fields and flags
 */

#ifndef HIP_REMOTE_PROTOCOL_H
#define HIP_REMOTE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Cross-platform struct packing */
#ifdef _MSC_VER
#define HIP_PACK_PUSH __pragma(pack(push, 1))
#define HIP_PACK_POP  __pragma(pack(pop))
#define HIP_PACKED_ATTR
#else
#define HIP_PACK_PUSH
#define HIP_PACK_POP
#define HIP_PACKED_ATTR __attribute__((packed))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Protocol Constants
 * ============================================================================ */

/** Protocol magic number: 'HIPR' in ASCII */
#define HIP_REMOTE_MAGIC 0x48495052

/** Protocol version (major.minor as 0xMMmm) */
#define HIP_REMOTE_VERSION 0x0100

/** Default port for worker service */
#define HIP_REMOTE_DEFAULT_PORT 18515

/** Maximum payload size (64MB) */
#define HIP_REMOTE_MAX_PAYLOAD_SIZE (64u * 1024u * 1024u)

/** Maximum number of kernel arguments */
#define HIP_REMOTE_MAX_KERNEL_ARGS 64

/** Maximum size of a single kernel argument */
#define HIP_REMOTE_MAX_ARG_SIZE 256

/* ============================================================================
 * Operation Codes
 * ============================================================================ */

typedef enum {
    /* Connection management (0x00xx) */
    HIP_OP_INIT                     = 0x0001,
    HIP_OP_SHUTDOWN                 = 0x0002,
    HIP_OP_PING                     = 0x0003,

    /* Device management (0x01xx) */
    HIP_OP_GET_DEVICE_COUNT         = 0x0100,
    HIP_OP_SET_DEVICE               = 0x0101,
    HIP_OP_GET_DEVICE               = 0x0102,
    HIP_OP_GET_DEVICE_PROPERTIES    = 0x0103,
    HIP_OP_DEVICE_SYNCHRONIZE       = 0x0104,
    HIP_OP_DEVICE_RESET             = 0x0105,
    HIP_OP_DEVICE_GET_ATTRIBUTE     = 0x0106,

    /* Memory allocation (0x02xx) */
    HIP_OP_MALLOC                   = 0x0200,
    HIP_OP_FREE                     = 0x0201,
    HIP_OP_MALLOC_HOST              = 0x0202,
    HIP_OP_FREE_HOST                = 0x0203,
    HIP_OP_MALLOC_MANAGED           = 0x0204,
    HIP_OP_MALLOC_ASYNC             = 0x0205,
    HIP_OP_FREE_ASYNC               = 0x0206,

    /* Memory transfer (0x021x) */
    HIP_OP_MEMCPY                   = 0x0210,
    HIP_OP_MEMCPY_ASYNC             = 0x0211,
    HIP_OP_MEMCPY_2D                = 0x0212,
    HIP_OP_MEMCPY_2D_ASYNC          = 0x0213,
    HIP_OP_MEMCPY_3D                = 0x0214,
    HIP_OP_MEMCPY_3D_ASYNC          = 0x0215,
    HIP_OP_MEMCPY_DTOD              = 0x0216,
    HIP_OP_MEMCPY_DTOD_ASYNC        = 0x0217,
    HIP_OP_MEMCPY_HTOD              = 0x0218,
    HIP_OP_MEMCPY_HTOD_ASYNC        = 0x0219,
    HIP_OP_MEMCPY_DTOH              = 0x021A,
    HIP_OP_MEMCPY_DTOH_ASYNC        = 0x021B,

    /* Memory set (0x022x) */
    HIP_OP_MEMSET                   = 0x0220,
    HIP_OP_MEMSET_ASYNC             = 0x0221,
    HIP_OP_MEMSET_D8                = 0x0222,
    HIP_OP_MEMSET_D16               = 0x0223,
    HIP_OP_MEMSET_D32               = 0x0224,

    /* Memory info (0x023x) */
    HIP_OP_MEM_GET_INFO             = 0x0230,
    HIP_OP_POINTER_GET_ATTRIBUTES   = 0x0231,

    /* Stream operations (0x03xx) */
    HIP_OP_STREAM_CREATE            = 0x0300,
    HIP_OP_STREAM_CREATE_WITH_FLAGS = 0x0301,
    HIP_OP_STREAM_CREATE_WITH_PRIORITY = 0x0302,
    HIP_OP_STREAM_DESTROY           = 0x0303,
    HIP_OP_STREAM_SYNCHRONIZE       = 0x0304,
    HIP_OP_STREAM_QUERY             = 0x0305,
    HIP_OP_STREAM_WAIT_EVENT        = 0x0306,
    HIP_OP_STREAM_GET_FLAGS         = 0x0307,
    HIP_OP_STREAM_GET_PRIORITY      = 0x0308,

    /* Event operations (0x04xx) */
    HIP_OP_EVENT_CREATE             = 0x0400,
    HIP_OP_EVENT_CREATE_WITH_FLAGS  = 0x0401,
    HIP_OP_EVENT_DESTROY            = 0x0402,
    HIP_OP_EVENT_RECORD             = 0x0403,
    HIP_OP_EVENT_SYNCHRONIZE        = 0x0404,
    HIP_OP_EVENT_QUERY              = 0x0405,
    HIP_OP_EVENT_ELAPSED_TIME       = 0x0406,

    /* Module operations (0x05xx) */
    HIP_OP_MODULE_LOAD_DATA         = 0x0500,
    HIP_OP_MODULE_LOAD_DATA_EX      = 0x0501,
    HIP_OP_MODULE_UNLOAD            = 0x0502,
    HIP_OP_MODULE_GET_FUNCTION      = 0x0503,
    HIP_OP_MODULE_GET_GLOBAL        = 0x0504,

    /* Kernel launch (0x051x) */
    HIP_OP_LAUNCH_KERNEL            = 0x0510,
    HIP_OP_LAUNCH_COOPERATIVE_KERNEL = 0x0511,
    HIP_OP_MODULE_LAUNCH_KERNEL     = 0x0512,

    /* Error handling (0x06xx) */
    HIP_OP_GET_LAST_ERROR           = 0x0600,
    HIP_OP_PEEK_AT_LAST_ERROR       = 0x0601,
    HIP_OP_GET_ERROR_STRING         = 0x0602,
    HIP_OP_GET_ERROR_NAME           = 0x0603,

    /* Runtime info (0x07xx) */
    HIP_OP_RUNTIME_GET_VERSION      = 0x0700,
    HIP_OP_DRIVER_GET_VERSION       = 0x0701,

    /* AMD SMI operations (0x08xx) */
    SMI_OP_INIT                     = 0x0800,
    SMI_OP_SHUTDOWN                 = 0x0801,
    SMI_OP_GET_PROCESSOR_COUNT      = 0x0802,
    SMI_OP_GET_GPU_METRICS          = 0x0820,
    SMI_OP_GET_POWER_INFO           = 0x0821,
    SMI_OP_GET_CLOCK_INFO           = 0x0822,
    SMI_OP_GET_TEMP_METRIC          = 0x0823,
    SMI_OP_GET_GPU_ACTIVITY         = 0x0824,
    SMI_OP_GET_VRAM_USAGE           = 0x0825,
    SMI_OP_GET_ASIC_INFO            = 0x0830,

} HipRemoteOpCode;

/* ============================================================================
 * Message Flags
 * ============================================================================ */

/** Response flag - set in responses */
#define HIP_REMOTE_FLAG_RESPONSE        (1u << 0)

/** Error flag - set when operation failed */
#define HIP_REMOTE_FLAG_ERROR           (1u << 1)

/** Has inline data - payload contains bulk data after structured payload */
#define HIP_REMOTE_FLAG_HAS_INLINE_DATA (1u << 2)

HIP_PACK_PUSH
/* ============================================================================
 * Protocol Header
 * ============================================================================ */

/**
 * Common header for all protocol messages.
 * Total size: 20 bytes
 */
typedef struct {
    uint32_t magic;           /**< Must be HIP_REMOTE_MAGIC */
    uint16_t version;         /**< Protocol version */
    uint16_t op_code;         /**< Operation code (HipRemoteOpCode) */
    uint32_t request_id;      /**< Correlation ID for async matching */
    uint32_t payload_length;  /**< Bytes following this header */
    uint32_t flags;           /**< Message flags */
} HIP_PACKED_ATTR HipRemoteHeader;

/* ============================================================================
 * Common Response Header
 * ============================================================================ */

/**
 * Common response header included in all responses.
 */
typedef struct {
    int32_t error_code;       /**< hipError_t value */
} HIP_PACKED_ATTR HipRemoteResponseHeader;

/* ============================================================================
 * Device Operations
 * ============================================================================ */

/* HIP_OP_SET_DEVICE / HIP_OP_GET_DEVICE / HIP_OP_DEVICE_GET_ATTRIBUTE */
typedef struct {
    int32_t device_id;
} HIP_PACKED_ATTR HipRemoteDeviceRequest;

typedef struct {
    int32_t device_id;
    int32_t attribute;        /**< For HIP_OP_DEVICE_GET_ATTRIBUTE */
} HIP_PACKED_ATTR HipRemoteDeviceAttributeRequest;

/* HIP_OP_GET_DEVICE_COUNT response */
typedef struct {
    HipRemoteResponseHeader header;
    int32_t count;
} HIP_PACKED_ATTR HipRemoteDeviceCountResponse;

/* HIP_OP_GET_DEVICE response */
typedef struct {
    HipRemoteResponseHeader header;
    int32_t device_id;
} HIP_PACKED_ATTR HipRemoteGetDeviceResponse;

/* HIP_OP_DEVICE_GET_ATTRIBUTE response */
typedef struct {
    HipRemoteResponseHeader header;
    int32_t value;
} HIP_PACKED_ATTR HipRemoteDeviceAttributeResponse;

/* HIP_OP_GET_DEVICE_PROPERTIES response */
typedef struct {
    HipRemoteResponseHeader header;
    /* Device properties - matches hipDeviceProp_t layout for key fields */
    char name[256];
    uint64_t total_global_mem;
    uint64_t shared_mem_per_block;
    int32_t regs_per_block;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t max_threads_dim[3];
    int32_t max_grid_size[3];
    int32_t clock_rate;
    int32_t memory_clock_rate;
    int32_t memory_bus_width;
    int32_t major;
    int32_t minor;
    int32_t multi_processor_count;
    int32_t l2_cache_size;
    int32_t max_threads_per_multi_processor;
    int32_t compute_mode;
    int32_t pci_bus_id;
    int32_t pci_device_id;
    int32_t pci_domain_id;
    int32_t integrated;
    int32_t can_map_host_memory;
    int32_t concurrent_kernels;
    char gcn_arch_name[256];
} HIP_PACKED_ATTR HipRemoteDevicePropertiesResponse;

/* ============================================================================
 * Memory Operations
 * ============================================================================ */

/* HIP_OP_MALLOC / HIP_OP_MALLOC_HOST / HIP_OP_MALLOC_MANAGED */
typedef struct {
    uint64_t size;
    uint32_t flags;           /**< For managed memory flags */
} HIP_PACKED_ATTR HipRemoteMallocRequest;

typedef struct {
    HipRemoteResponseHeader header;
    uint64_t device_ptr;      /**< Remote device pointer (opaque handle) */
} HIP_PACKED_ATTR HipRemoteMallocResponse;

/* HIP_OP_FREE / HIP_OP_FREE_HOST */
typedef struct {
    uint64_t device_ptr;
} HIP_PACKED_ATTR HipRemoteFreeRequest;

/* HIP_OP_MEMCPY / HIP_OP_MEMCPY_ASYNC */
typedef struct {
    uint64_t dst;             /**< Destination pointer */
    uint64_t src;             /**< Source pointer */
    uint64_t size;            /**< Size in bytes */
    int32_t kind;             /**< hipMemcpyKind */
    uint64_t stream;          /**< Stream handle (0 for default) */
} HIP_PACKED_ATTR HipRemoteMemcpyRequest;

/* For H2D copies, inline data follows this header */
/* For D2H copies, response includes inline data */

typedef struct {
    HipRemoteResponseHeader header;
    /* For D2H: inline data follows of size 'size' from request */
} HIP_PACKED_ATTR HipRemoteMemcpyResponse;

/* HIP_OP_MEMSET / HIP_OP_MEMSET_ASYNC */
typedef struct {
    uint64_t dst;
    int32_t value;
    uint64_t size;
    uint64_t stream;
} HIP_PACKED_ATTR HipRemoteMemsetRequest;

/* HIP_OP_MEM_GET_INFO response */
typedef struct {
    HipRemoteResponseHeader header;
    uint64_t free_bytes;
    uint64_t total_bytes;
} HIP_PACKED_ATTR HipRemoteMemGetInfoResponse;

/* ============================================================================
 * Stream Operations
 * ============================================================================ */

/* HIP_OP_STREAM_CREATE / HIP_OP_STREAM_CREATE_WITH_FLAGS */
typedef struct {
    uint32_t flags;
    int32_t priority;         /**< For HIP_OP_STREAM_CREATE_WITH_PRIORITY */
} HIP_PACKED_ATTR HipRemoteStreamCreateRequest;

typedef struct {
    HipRemoteResponseHeader header;
    uint64_t stream;          /**< Remote stream handle */
} HIP_PACKED_ATTR HipRemoteStreamCreateResponse;

/* HIP_OP_STREAM_DESTROY / HIP_OP_STREAM_SYNCHRONIZE / HIP_OP_STREAM_QUERY */
typedef struct {
    uint64_t stream;
} HIP_PACKED_ATTR HipRemoteStreamRequest;

/* HIP_OP_STREAM_WAIT_EVENT */
typedef struct {
    uint64_t stream;
    uint64_t event;
    uint32_t flags;
} HIP_PACKED_ATTR HipRemoteStreamWaitEventRequest;

/* ============================================================================
 * Event Operations
 * ============================================================================ */

/* HIP_OP_EVENT_CREATE / HIP_OP_EVENT_CREATE_WITH_FLAGS */
typedef struct {
    uint32_t flags;
} HIP_PACKED_ATTR HipRemoteEventCreateRequest;

typedef struct {
    HipRemoteResponseHeader header;
    uint64_t event;           /**< Remote event handle */
} HIP_PACKED_ATTR HipRemoteEventCreateResponse;

/* HIP_OP_EVENT_DESTROY / HIP_OP_EVENT_SYNCHRONIZE / HIP_OP_EVENT_QUERY */
typedef struct {
    uint64_t event;
} HIP_PACKED_ATTR HipRemoteEventRequest;

/* HIP_OP_EVENT_RECORD */
typedef struct {
    uint64_t event;
    uint64_t stream;
} HIP_PACKED_ATTR HipRemoteEventRecordRequest;

/* HIP_OP_EVENT_ELAPSED_TIME */
typedef struct {
    uint64_t start_event;
    uint64_t end_event;
} HIP_PACKED_ATTR HipRemoteEventElapsedTimeRequest;

typedef struct {
    HipRemoteResponseHeader header;
    float milliseconds;
} HIP_PACKED_ATTR HipRemoteEventElapsedTimeResponse;

/* ============================================================================
 * Module Operations
 * ============================================================================ */

/* HIP_OP_MODULE_LOAD_DATA
 * Payload: module data (code object) follows immediately after this struct
 */
typedef struct {
    uint64_t data_size;       /**< Size of code object data */
    /* Code object data follows */
} HIP_PACKED_ATTR HipRemoteModuleLoadRequest;

typedef struct {
    HipRemoteResponseHeader header;
    uint64_t module;          /**< Remote module handle */
} HIP_PACKED_ATTR HipRemoteModuleLoadResponse;

/* HIP_OP_MODULE_UNLOAD */
typedef struct {
    uint64_t module;
} HIP_PACKED_ATTR HipRemoteModuleUnloadRequest;

/* HIP_OP_MODULE_GET_FUNCTION */
typedef struct {
    uint64_t module;
    char function_name[256];
} HIP_PACKED_ATTR HipRemoteModuleGetFunctionRequest;

typedef struct {
    HipRemoteResponseHeader header;
    uint64_t function;        /**< Remote function handle */
    uint32_t num_args;        /**< Number of kernel arguments (from kernel metadata) */
    uint32_t reserved;        /**< Reserved for future use (alignment) */
} HIP_PACKED_ATTR HipRemoteModuleGetFunctionResponse;

/* ============================================================================
 * Kernel Launch
 * ============================================================================ */

/**
 * Kernel argument descriptor
 */
typedef struct {
    uint32_t size;            /**< Argument size in bytes */
    uint32_t offset;          /**< Offset into arg_data array */
} HIP_PACKED_ATTR HipRemoteKernelArg;

/**
 * HIP_OP_LAUNCH_KERNEL
 * Variable-size message: arg_data follows the fixed portion
 */
typedef struct {
    uint64_t function;        /**< Function handle from MODULE_GET_FUNCTION */
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint64_t stream;
    uint32_t num_args;
    /* HipRemoteKernelArg args[num_args] follows */
    /* uint8_t arg_data[] follows (concatenated argument values) */
} HIP_PACKED_ATTR HipRemoteLaunchKernelRequest;

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/* HIP_OP_GET_ERROR_STRING / HIP_OP_GET_ERROR_NAME */
typedef struct {
    int32_t error_code;
} HIP_PACKED_ATTR HipRemoteErrorStringRequest;

typedef struct {
    HipRemoteResponseHeader header;
    char error_string[256];
} HIP_PACKED_ATTR HipRemoteErrorStringResponse;

/* ============================================================================
 * Runtime Info
 * ============================================================================ */

/* HIP_OP_RUNTIME_GET_VERSION / HIP_OP_DRIVER_GET_VERSION response */
typedef struct {
    HipRemoteResponseHeader header;
    int32_t version;
} HIP_PACKED_ATTR HipRemoteVersionResponse;

/* ============================================================================
 * AMD SMI Operations
 * ============================================================================ */

/* SMI_OP_INIT request */
typedef struct {
    uint64_t init_flags;          /**< amdsmi_init_flags_t */
} HIP_PACKED_ATTR SmiRemoteInitRequest;

/* SMI_OP_GET_PROCESSOR_COUNT response */
typedef struct {
    HipRemoteResponseHeader header;
    uint32_t processor_count;
} HIP_PACKED_ATTR SmiRemoteProcessorCountResponse;

/* Request with processor index (used by most SMI queries) */
typedef struct {
    uint32_t processor_index;     /**< Maps to remote amdsmi_processor_handle */
} HIP_PACKED_ATTR SmiRemoteProcessorRequest;

/* SMI_OP_GET_GPU_METRICS response - summary of key metrics */
typedef struct {
    HipRemoteResponseHeader header;
    int32_t temperature_edge;     /**< Edge temperature (C) */
    int32_t temperature_hotspot;  /**< Hotspot/junction temperature (C) */
    int32_t temperature_mem;      /**< Memory temperature (C) */
    uint32_t average_socket_power;/**< Average socket power (W) */
    uint32_t gfx_activity;        /**< GFX engine activity (%) */
    uint32_t umc_activity;        /**< Memory controller activity (%) */
    uint32_t mm_activity;         /**< Multimedia engine activity (%) */
    uint32_t current_gfxclk;      /**< Current GFX clock (MHz) */
    uint32_t current_uclk;        /**< Current memory clock (MHz) */
    uint32_t current_socclk;      /**< Current SOC clock (MHz) */
    uint64_t vram_total;          /**< Total VRAM (bytes) */
    uint64_t vram_used;           /**< Used VRAM (bytes) */
    uint32_t fan_speed_rpm;       /**< Fan speed (RPM) */
    uint32_t pcie_bandwidth;      /**< PCIe bandwidth (MB/s) */
    uint32_t throttle_status;     /**< Throttle status flags */
    uint32_t reserved;            /**< Padding for alignment */
} HIP_PACKED_ATTR SmiRemoteGpuMetricsResponse;

/* SMI_OP_GET_POWER_INFO response */
typedef struct {
    HipRemoteResponseHeader header;
    uint32_t current_socket_power;/**< Current socket power (W) */
    uint32_t average_socket_power;/**< Average socket power (W) */
    uint32_t gfx_voltage;         /**< GFX voltage (mV) */
    uint32_t soc_voltage;         /**< SOC voltage (mV) */
    uint32_t mem_voltage;         /**< Memory voltage (mV) */
    uint32_t power_limit;         /**< Power limit/cap (W) */
} HIP_PACKED_ATTR SmiRemotePowerInfoResponse;

/* SMI_OP_GET_CLOCK_INFO request */
typedef struct {
    uint32_t processor_index;
    uint32_t clock_type;          /**< amdsmi_clk_type_t */
} HIP_PACKED_ATTR SmiRemoteClockInfoRequest;

/* SMI_OP_GET_CLOCK_INFO response */
typedef struct {
    HipRemoteResponseHeader header;
    uint32_t current_clk;         /**< Current clock (MHz) */
    uint32_t min_clk;             /**< Minimum clock (MHz) */
    uint32_t max_clk;             /**< Maximum clock (MHz) */
    uint8_t clk_locked;           /**< Clock locked flag */
    uint8_t clk_deep_sleep;       /**< Deep sleep flag */
    uint16_t reserved;            /**< Padding */
} HIP_PACKED_ATTR SmiRemoteClockInfoResponse;

/* SMI_OP_GET_TEMP_METRIC request */
typedef struct {
    uint32_t processor_index;
    uint32_t sensor_type;         /**< amdsmi_temperature_type_t */
} HIP_PACKED_ATTR SmiRemoteTempMetricRequest;

/* SMI_OP_GET_TEMP_METRIC response */
typedef struct {
    HipRemoteResponseHeader header;
    int32_t temperature;          /**< Temperature (milli-Celsius) */
} HIP_PACKED_ATTR SmiRemoteTempMetricResponse;

/* SMI_OP_GET_GPU_ACTIVITY response */
typedef struct {
    HipRemoteResponseHeader header;
    uint32_t gfx_activity;        /**< GFX activity (%) */
    uint32_t umc_activity;        /**< Memory controller activity (%) */
    uint32_t mm_activity;         /**< Multimedia activity (%) */
    uint32_t reserved;            /**< Padding */
} HIP_PACKED_ATTR SmiRemoteGpuActivityResponse;

/* SMI_OP_GET_VRAM_USAGE response */
typedef struct {
    HipRemoteResponseHeader header;
    uint64_t vram_total;          /**< Total VRAM (bytes) */
    uint64_t vram_used;           /**< Used VRAM (bytes) */
} HIP_PACKED_ATTR SmiRemoteVramUsageResponse;

/* SMI_OP_GET_ASIC_INFO response */
typedef struct {
    HipRemoteResponseHeader header;
    char market_name[256];        /**< Marketing name (e.g., "AMD Instinct MI300X") */
    uint32_t vendor_id;           /**< PCI vendor ID */
    uint32_t device_id;           /**< PCI device ID */
    uint32_t rev_id;              /**< Revision ID */
    uint32_t num_compute_units;   /**< Number of compute units */
    char asic_serial[64];         /**< ASIC serial number */
} HIP_PACKED_ATTR SmiRemoteAsicInfoResponse;

HIP_PACK_POP

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Initialize a protocol header with the given parameters.
 */
static inline void hip_remote_init_header(
    HipRemoteHeader* header,
    HipRemoteOpCode op_code,
    uint32_t request_id,
    uint32_t payload_length
) {
    header->magic = HIP_REMOTE_MAGIC;
    header->version = HIP_REMOTE_VERSION;
    header->op_code = (uint16_t)op_code;
    header->request_id = request_id;
    header->payload_length = payload_length;
    header->flags = 0;
}

/**
 * Validate a received protocol header.
 * @return 0 on success, negative error code on failure
 */
static inline int hip_remote_validate_header(const HipRemoteHeader* header) {
    if (header->magic != HIP_REMOTE_MAGIC) {
        return -1;  /* Invalid magic */
    }
    if ((header->version >> 8) != (HIP_REMOTE_VERSION >> 8)) {
        return -2;  /* Major version mismatch */
    }
    if (header->payload_length > HIP_REMOTE_MAX_PAYLOAD_SIZE) {
        return -3;  /* Payload too large */
    }
    return 0;
}

/**
 * Get human-readable name for an operation code.
 */
static inline const char* hip_remote_op_name(HipRemoteOpCode op_code) {
    switch (op_code) {
        case HIP_OP_INIT: return "hipInit(remote)";
        case HIP_OP_SHUTDOWN: return "hipShutdown(remote)";
        case HIP_OP_PING: return "ping";

        case HIP_OP_GET_DEVICE_COUNT: return "hipGetDeviceCount";
        case HIP_OP_SET_DEVICE: return "hipSetDevice";
        case HIP_OP_GET_DEVICE: return "hipGetDevice";
        case HIP_OP_GET_DEVICE_PROPERTIES: return "hipGetDeviceProperties";
        case HIP_OP_DEVICE_SYNCHRONIZE: return "hipDeviceSynchronize";
        case HIP_OP_DEVICE_RESET: return "hipDeviceReset";
        case HIP_OP_DEVICE_GET_ATTRIBUTE: return "hipDeviceGetAttribute";

        case HIP_OP_MALLOC: return "hipMalloc";
        case HIP_OP_FREE: return "hipFree";
        case HIP_OP_MALLOC_HOST: return "hipMallocHost";
        case HIP_OP_FREE_HOST: return "hipFreeHost";
        case HIP_OP_MALLOC_MANAGED: return "hipMallocManaged";
        case HIP_OP_MALLOC_ASYNC: return "hipMallocAsync";
        case HIP_OP_FREE_ASYNC: return "hipFreeAsync";

        case HIP_OP_MEMCPY: return "hipMemcpy";
        case HIP_OP_MEMCPY_ASYNC: return "hipMemcpyAsync";
        case HIP_OP_MEMCPY_DTOD: return "hipMemcpyDtoD";
        case HIP_OP_MEMCPY_DTOD_ASYNC: return "hipMemcpyDtoDAsync";
        case HIP_OP_MEMCPY_HTOD: return "hipMemcpyHtoD";
        case HIP_OP_MEMCPY_HTOD_ASYNC: return "hipMemcpyHtoDAsync";
        case HIP_OP_MEMCPY_DTOH: return "hipMemcpyDtoH";
        case HIP_OP_MEMCPY_DTOH_ASYNC: return "hipMemcpyDtoHAsync";

        case HIP_OP_MEMSET: return "hipMemset";
        case HIP_OP_MEMSET_ASYNC: return "hipMemsetAsync";
        case HIP_OP_MEMSET_D8: return "hipMemsetD8";
        case HIP_OP_MEMSET_D16: return "hipMemsetD16";
        case HIP_OP_MEMSET_D32: return "hipMemsetD32";

        case HIP_OP_MEM_GET_INFO: return "hipMemGetInfo";
        case HIP_OP_POINTER_GET_ATTRIBUTES: return "hipPointerGetAttributes";

        case HIP_OP_STREAM_CREATE: return "hipStreamCreate";
        case HIP_OP_STREAM_CREATE_WITH_FLAGS: return "hipStreamCreateWithFlags";
        case HIP_OP_STREAM_CREATE_WITH_PRIORITY: return "hipStreamCreateWithPriority";
        case HIP_OP_STREAM_DESTROY: return "hipStreamDestroy";
        case HIP_OP_STREAM_SYNCHRONIZE: return "hipStreamSynchronize";
        case HIP_OP_STREAM_QUERY: return "hipStreamQuery";
        case HIP_OP_STREAM_WAIT_EVENT: return "hipStreamWaitEvent";
        case HIP_OP_STREAM_GET_FLAGS: return "hipStreamGetFlags";
        case HIP_OP_STREAM_GET_PRIORITY: return "hipStreamGetPriority";

        case HIP_OP_EVENT_CREATE: return "hipEventCreate";
        case HIP_OP_EVENT_CREATE_WITH_FLAGS: return "hipEventCreateWithFlags";
        case HIP_OP_EVENT_DESTROY: return "hipEventDestroy";
        case HIP_OP_EVENT_RECORD: return "hipEventRecord";
        case HIP_OP_EVENT_SYNCHRONIZE: return "hipEventSynchronize";
        case HIP_OP_EVENT_QUERY: return "hipEventQuery";
        case HIP_OP_EVENT_ELAPSED_TIME: return "hipEventElapsedTime";

        case HIP_OP_MODULE_LOAD_DATA: return "hipModuleLoadData";
        case HIP_OP_MODULE_LOAD_DATA_EX: return "hipModuleLoadDataEx";
        case HIP_OP_MODULE_UNLOAD: return "hipModuleUnload";
        case HIP_OP_MODULE_GET_FUNCTION: return "hipModuleGetFunction";
        case HIP_OP_MODULE_GET_GLOBAL: return "hipModuleGetGlobal";

        case HIP_OP_LAUNCH_KERNEL: return "hipLaunchKernel";
        case HIP_OP_LAUNCH_COOPERATIVE_KERNEL: return "hipLaunchCooperativeKernel";
        case HIP_OP_MODULE_LAUNCH_KERNEL: return "hipModuleLaunchKernel";

        case HIP_OP_GET_LAST_ERROR: return "hipGetLastError";
        case HIP_OP_PEEK_AT_LAST_ERROR: return "hipPeekAtLastError";
        case HIP_OP_GET_ERROR_STRING: return "hipGetErrorString";
        case HIP_OP_GET_ERROR_NAME: return "hipGetErrorName";

        case HIP_OP_RUNTIME_GET_VERSION: return "hipRuntimeGetVersion";
        case HIP_OP_DRIVER_GET_VERSION: return "hipDriverGetVersion";

        case SMI_OP_INIT: return "amdsmi_init";
        case SMI_OP_SHUTDOWN: return "amdsmi_shut_down";
        case SMI_OP_GET_PROCESSOR_COUNT: return "amdsmi_get_processor_count";
        case SMI_OP_GET_GPU_METRICS: return "amdsmi_get_gpu_metrics";
        case SMI_OP_GET_POWER_INFO: return "amdsmi_get_power_info";
        case SMI_OP_GET_CLOCK_INFO: return "amdsmi_get_clock_info";
        case SMI_OP_GET_TEMP_METRIC: return "amdsmi_get_temp_metric";
        case SMI_OP_GET_GPU_ACTIVITY: return "amdsmi_get_gpu_activity";
        case SMI_OP_GET_VRAM_USAGE: return "amdsmi_get_vram_usage";
        case SMI_OP_GET_ASIC_INFO: return "amdsmi_get_asic_info";

        default: return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* HIP_REMOTE_PROTOCOL_H */
