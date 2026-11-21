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
 * @file smi_remote_client.h
 * @brief Remote AMD SMI client API
 *
 * This library provides remote access to AMD SMI functionality by
 * tunneling requests through the HIP remote execution infrastructure.
 *
 * On macOS, this provides the only way to query AMD GPU metrics.
 * On Linux, this can be used to query remote GPUs in addition to local ones.
 *
 * Environment variables:
 *   TF_WORKER_HOST  - Hostname of the remote HIP worker (required)
 *   TF_WORKER_PORT  - Port number (default: 18515)
 *   TF_DEBUG        - Enable debug logging (1/0)
 */

#ifndef SMI_REMOTE_CLIENT_H
#define SMI_REMOTE_CLIENT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Status Codes
 * ============================================================================ */

typedef enum {
    SMI_STATUS_SUCCESS = 0,
    SMI_STATUS_INVALID_ARGS = 1,
    SMI_STATUS_NOT_SUPPORTED = 2,
    SMI_STATUS_NOT_FOUND = 3,
    SMI_STATUS_NOT_INITIALIZED = 4,
    SMI_STATUS_IO_ERROR = 5,
    SMI_STATUS_API_FAILED = 6,
} smi_remote_status_t;

/* ============================================================================
 * Data Structures
 * ============================================================================ */

/**
 * GPU metrics summary
 */
typedef struct {
    int32_t temperature_edge;      /**< Edge temperature (C) */
    int32_t temperature_hotspot;   /**< Hotspot/junction temperature (C) */
    int32_t temperature_mem;       /**< Memory temperature (C) */
    uint32_t power_watts;          /**< Average socket power (W) */
    uint32_t gfx_activity;         /**< GFX engine activity (%) */
    uint32_t mem_activity;         /**< Memory controller activity (%) */
    uint32_t mm_activity;          /**< Multimedia engine activity (%) */
    uint32_t gfx_clock_mhz;        /**< Current GFX clock (MHz) */
    uint32_t mem_clock_mhz;        /**< Current memory clock (MHz) */
    uint32_t soc_clock_mhz;        /**< Current SOC clock (MHz) */
    uint64_t vram_total_bytes;     /**< Total VRAM (bytes) */
    uint64_t vram_used_bytes;      /**< Used VRAM (bytes) */
    uint32_t fan_speed_rpm;        /**< Fan speed (RPM) */
    uint32_t throttle_status;      /**< Throttle status flags */
} smi_remote_gpu_metrics_t;

/**
 * Power information
 */
typedef struct {
    uint32_t current_power_watts;  /**< Current socket power (W) */
    uint32_t average_power_watts;  /**< Average socket power (W) */
    uint32_t gfx_voltage_mv;       /**< GFX voltage (mV) */
    uint32_t soc_voltage_mv;       /**< SOC voltage (mV) */
    uint32_t mem_voltage_mv;       /**< Memory voltage (mV) */
    uint32_t power_limit_watts;    /**< Power limit/cap (W) */
} smi_remote_power_info_t;

/**
 * ASIC information
 */
typedef struct {
    char market_name[256];         /**< Marketing name */
    uint32_t vendor_id;            /**< PCI vendor ID */
    uint32_t device_id;            /**< PCI device ID */
    uint32_t rev_id;               /**< Revision ID */
    uint32_t num_compute_units;    /**< Number of compute units */
    char serial[64];               /**< ASIC serial number */
} smi_remote_asic_info_t;

/* ============================================================================
 * Initialization and Connection
 * ============================================================================ */

/**
 * Initialize the SMI remote client.
 * Must be called before any other smi_remote_* functions.
 *
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_init(void);

/**
 * Shutdown the SMI remote client.
 * Closes the connection to the remote worker.
 */
void smi_remote_shutdown(void);

/**
 * Check if connected to a remote worker.
 *
 * @return true if connected
 */
bool smi_remote_is_connected(void);

/* ============================================================================
 * Device Enumeration
 * ============================================================================ */

/**
 * Get the number of GPUs available on the remote worker.
 *
 * @param count Pointer to receive the GPU count
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_get_processor_count(uint32_t* count);

/* ============================================================================
 * Metrics Queries
 * ============================================================================ */

/**
 * Get comprehensive GPU metrics for a processor.
 *
 * @param processor_index Index of the GPU (0-based)
 * @param metrics Pointer to receive metrics
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_get_gpu_metrics(
    uint32_t processor_index,
    smi_remote_gpu_metrics_t* metrics
);

/**
 * Get power information for a processor.
 *
 * @param processor_index Index of the GPU (0-based)
 * @param power_info Pointer to receive power info
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_get_power_info(
    uint32_t processor_index,
    smi_remote_power_info_t* power_info
);

/**
 * Get ASIC information for a processor.
 *
 * @param processor_index Index of the GPU (0-based)
 * @param asic_info Pointer to receive ASIC info
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_get_asic_info(
    uint32_t processor_index,
    smi_remote_asic_info_t* asic_info
);

/**
 * Get VRAM usage for a processor.
 *
 * @param processor_index Index of the GPU (0-based)
 * @param total_bytes Pointer to receive total VRAM
 * @param used_bytes Pointer to receive used VRAM
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_get_vram_usage(
    uint32_t processor_index,
    uint64_t* total_bytes,
    uint64_t* used_bytes
);

/**
 * Get GPU activity (utilization) for a processor.
 *
 * @param processor_index Index of the GPU (0-based)
 * @param gfx_activity Pointer to receive GFX activity (%)
 * @param mem_activity Pointer to receive memory activity (%)
 * @param mm_activity Pointer to receive multimedia activity (%)
 * @return SMI_STATUS_SUCCESS on success
 */
smi_remote_status_t smi_remote_get_gpu_activity(
    uint32_t processor_index,
    uint32_t* gfx_activity,
    uint32_t* mem_activity,
    uint32_t* mm_activity
);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get a human-readable string for a status code.
 *
 * @param status Status code
 * @return Static string describing the status
 */
const char* smi_remote_status_string(smi_remote_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* SMI_REMOTE_CLIENT_H */
