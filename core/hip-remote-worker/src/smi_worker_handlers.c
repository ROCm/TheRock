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
 * @file smi_worker_handlers.c
 * @brief AMD SMI request handlers for the HIP remote worker
 */

#include "smi_worker_handlers.h"
#include "hip_remote/hip_remote_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <amd_smi/amdsmi.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

#define SMI_MAX_DEVICES 64

static bool g_smi_initialized = false;
static amdsmi_processor_handle g_processor_handles[SMI_MAX_DEVICES];
static uint32_t g_processor_count = 0;
static bool g_debug = false;

/* ============================================================================
 * Logging
 * ============================================================================ */

#define SMI_LOG_DEBUG(fmt, ...) \
    do { if (g_debug) fprintf(stderr, "[SMI-Worker] " fmt "\n", ##__VA_ARGS__); } while(0)

#define SMI_LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[SMI-Worker ERROR] " fmt "\n", ##__VA_ARGS__)

/* ============================================================================
 * Response Helpers
 * ============================================================================ */

static int send_response(int fd, uint16_t op_code, uint32_t request_id,
                        const void* payload, size_t payload_size) {
    HipRemoteHeader header;
    header.magic = HIP_REMOTE_MAGIC;
    header.version = HIP_REMOTE_VERSION;
    header.op_code = op_code;
    header.request_id = request_id;
    header.payload_length = (uint32_t)payload_size;
    header.flags = HIP_REMOTE_FLAG_RESPONSE;

    /* Send header */
    ssize_t sent = send(fd, &header, sizeof(header), 0);
    if (sent != sizeof(header)) {
        SMI_LOG_ERROR("Failed to send response header");
        return -1;
    }

    /* Send payload */
    if (payload_size > 0 && payload) {
        sent = send(fd, payload, payload_size, 0);
        if (sent != (ssize_t)payload_size) {
            SMI_LOG_ERROR("Failed to send response payload");
            return -1;
        }
    }

    return 0;
}

static int send_simple_response(int fd, uint16_t op_code, uint32_t request_id,
                               int32_t error_code) {
    HipRemoteResponseHeader resp = { .error_code = error_code };
    return send_response(fd, op_code, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * SMI Initialization
 * ============================================================================ */

int smi_worker_init(void) {
    if (g_smi_initialized) {
        return 0;
    }

    /* Check for debug environment variable */
    const char* debug_env = getenv("TF_DEBUG");
    g_debug = (debug_env != NULL && strcmp(debug_env, "1") == 0);

    SMI_LOG_DEBUG("Initializing AMD SMI...");

    amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
        SMI_LOG_ERROR("amdsmi_init failed: %d", status);
        return -1;
    }

    /* Get socket handles */
    uint32_t socket_count = 0;
    amdsmi_socket_handle sockets[SMI_MAX_DEVICES];
    status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        SMI_LOG_ERROR("amdsmi_get_socket_handles (count) failed: %d", status);
        amdsmi_shut_down();
        return -1;
    }

    if (socket_count > SMI_MAX_DEVICES) {
        socket_count = SMI_MAX_DEVICES;
    }

    status = amdsmi_get_socket_handles(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        SMI_LOG_ERROR("amdsmi_get_socket_handles failed: %d", status);
        amdsmi_shut_down();
        return -1;
    }

    /* Get processor handles from all sockets */
    g_processor_count = 0;
    for (uint32_t s = 0; s < socket_count && g_processor_count < SMI_MAX_DEVICES; s++) {
        uint32_t proc_count = SMI_MAX_DEVICES - g_processor_count;
        status = amdsmi_get_processor_handles(sockets[s], &proc_count,
                                              &g_processor_handles[g_processor_count]);
        if (status == AMDSMI_STATUS_SUCCESS) {
            g_processor_count += proc_count;
        }
    }

    SMI_LOG_DEBUG("AMD SMI initialized: %u processors", g_processor_count);
    g_smi_initialized = true;
    return 0;
}

void smi_worker_shutdown(void) {
    if (g_smi_initialized) {
        SMI_LOG_DEBUG("Shutting down AMD SMI...");
        amdsmi_shut_down();
        g_smi_initialized = false;
        g_processor_count = 0;
    }
}

bool smi_worker_is_available(void) {
    return g_smi_initialized;
}

uint32_t smi_worker_get_processor_count(void) {
    return g_processor_count;
}

/* ============================================================================
 * Handler Implementations
 * ============================================================================ */

static void handle_smi_init(int fd, uint32_t request_id,
                           const void* payload, size_t payload_size) {
    (void)payload;
    (void)payload_size;

    int result = smi_worker_init();
    send_simple_response(fd, SMI_OP_INIT, request_id,
                        result == 0 ? 0 : -1);
}

static void handle_smi_shutdown(int fd, uint32_t request_id,
                               const void* payload, size_t payload_size) {
    (void)payload;
    (void)payload_size;

    /* Send response first, then shutdown (amdsmi_shut_down can block) */
    send_simple_response(fd, SMI_OP_SHUTDOWN, request_id, 0);
    /* Note: Don't actually shutdown SMI here - let the worker manage its lifecycle */
}

static void handle_smi_get_processor_count(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    (void)payload;
    (void)payload_size;

    SmiRemoteProcessorCountResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized) {
        resp.header.error_code = -1;
        resp.processor_count = 0;
    } else {
        resp.header.error_code = 0;
        resp.processor_count = g_processor_count;
    }

    send_response(fd, SMI_OP_GET_PROCESSOR_COUNT, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_gpu_metrics(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    SmiRemoteGpuMetricsResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_GPU_METRICS, request_id, &resp, sizeof(resp));
        return;
    }

    if (!payload || payload_size < sizeof(SmiRemoteProcessorRequest)) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_GPU_METRICS, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteProcessorRequest* req = (const SmiRemoteProcessorRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -3;
        send_response(fd, SMI_OP_GET_GPU_METRICS, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];
    amdsmi_status_t status;

    /* Get GPU metrics */
    amdsmi_gpu_metrics_t metrics;
    memset(&metrics, 0, sizeof(metrics));
    status = amdsmi_get_gpu_metrics_info(handle, &metrics);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.temperature_edge = metrics.temperature_edge;
        resp.temperature_hotspot = metrics.temperature_hotspot;
        resp.temperature_mem = metrics.temperature_mem;
        resp.average_socket_power = metrics.average_socket_power;
        resp.gfx_activity = metrics.average_gfx_activity;
        resp.umc_activity = metrics.average_umc_activity;
        resp.mm_activity = metrics.average_mm_activity;
        resp.current_gfxclk = metrics.current_gfxclk;
        resp.current_uclk = metrics.current_uclk;
        resp.current_socclk = metrics.current_socclk;
        resp.fan_speed_rpm = metrics.current_fan_speed;
        resp.throttle_status = metrics.throttle_status;
    }

    /* Get VRAM usage */
    amdsmi_vram_usage_t vram;
    memset(&vram, 0, sizeof(vram));
    status = amdsmi_get_gpu_vram_usage(handle, &vram);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.vram_total = vram.vram_total;
        resp.vram_used = vram.vram_used;
    }

    resp.header.error_code = 0;
    send_response(fd, SMI_OP_GET_GPU_METRICS, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_power_info(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    SmiRemotePowerInfoResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized || !payload || payload_size < sizeof(SmiRemoteProcessorRequest)) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_POWER_INFO, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteProcessorRequest* req = (const SmiRemoteProcessorRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_POWER_INFO, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];

    amdsmi_power_info_t power_info;
    memset(&power_info, 0, sizeof(power_info));
    amdsmi_status_t status = amdsmi_get_power_info(handle, &power_info);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.current_socket_power = power_info.current_socket_power;
        resp.average_socket_power = power_info.average_socket_power;
        resp.gfx_voltage = power_info.gfx_voltage;
        resp.soc_voltage = power_info.soc_voltage;
        resp.mem_voltage = power_info.mem_voltage;
        resp.power_limit = power_info.power_limit;
        resp.header.error_code = 0;
    } else {
        resp.header.error_code = (int32_t)status;
    }

    send_response(fd, SMI_OP_GET_POWER_INFO, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_clock_info(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    SmiRemoteClockInfoResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized || !payload || payload_size < sizeof(SmiRemoteClockInfoRequest)) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_CLOCK_INFO, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteClockInfoRequest* req = (const SmiRemoteClockInfoRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_CLOCK_INFO, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];

    amdsmi_clk_info_t clk_info;
    memset(&clk_info, 0, sizeof(clk_info));
    amdsmi_status_t status = amdsmi_get_clock_info(handle,
                                                   (amdsmi_clk_type_t)req->clock_type,
                                                   &clk_info);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.current_clk = clk_info.clk;
        resp.min_clk = clk_info.min_clk;
        resp.max_clk = clk_info.max_clk;
        resp.clk_locked = clk_info.clk_locked;
        resp.clk_deep_sleep = clk_info.clk_deep_sleep;
        resp.header.error_code = 0;
    } else {
        resp.header.error_code = (int32_t)status;
    }

    send_response(fd, SMI_OP_GET_CLOCK_INFO, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_temp_metric(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    SmiRemoteTempMetricResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized || !payload || payload_size < sizeof(SmiRemoteTempMetricRequest)) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_TEMP_METRIC, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteTempMetricRequest* req = (const SmiRemoteTempMetricRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_TEMP_METRIC, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];

    int64_t temperature = 0;
    amdsmi_status_t status = amdsmi_get_temp_metric(handle,
                                                    (amdsmi_temperature_type_t)req->sensor_type,
                                                    AMDSMI_TEMP_CURRENT,
                                                    &temperature);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.temperature = (int32_t)temperature;
        resp.header.error_code = 0;
    } else {
        resp.header.error_code = (int32_t)status;
    }

    send_response(fd, SMI_OP_GET_TEMP_METRIC, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_gpu_activity(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    SmiRemoteGpuActivityResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized || !payload || payload_size < sizeof(SmiRemoteProcessorRequest)) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_GPU_ACTIVITY, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteProcessorRequest* req = (const SmiRemoteProcessorRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_GPU_ACTIVITY, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];

    amdsmi_engine_usage_t usage;
    memset(&usage, 0, sizeof(usage));
    amdsmi_status_t status = amdsmi_get_gpu_activity(handle, &usage);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.gfx_activity = usage.gfx_activity;
        resp.umc_activity = usage.umc_activity;
        resp.mm_activity = usage.mm_activity;
        resp.header.error_code = 0;
    } else {
        resp.header.error_code = (int32_t)status;
    }

    send_response(fd, SMI_OP_GET_GPU_ACTIVITY, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_vram_usage(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    SmiRemoteVramUsageResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized || !payload || payload_size < sizeof(SmiRemoteProcessorRequest)) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_VRAM_USAGE, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteProcessorRequest* req = (const SmiRemoteProcessorRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_VRAM_USAGE, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];

    amdsmi_vram_usage_t vram;
    memset(&vram, 0, sizeof(vram));
    amdsmi_status_t status = amdsmi_get_gpu_vram_usage(handle, &vram);
    if (status == AMDSMI_STATUS_SUCCESS) {
        resp.vram_total = vram.vram_total;
        resp.vram_used = vram.vram_used;
        resp.header.error_code = 0;
    } else {
        resp.header.error_code = (int32_t)status;
    }

    send_response(fd, SMI_OP_GET_VRAM_USAGE, request_id, &resp, sizeof(resp));
}

static void handle_smi_get_asic_info(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    SmiRemoteAsicInfoResponse resp;
    memset(&resp, 0, sizeof(resp));

    if (!g_smi_initialized || !payload || payload_size < sizeof(SmiRemoteProcessorRequest)) {
        resp.header.error_code = -1;
        send_response(fd, SMI_OP_GET_ASIC_INFO, request_id, &resp, sizeof(resp));
        return;
    }

    const SmiRemoteProcessorRequest* req = (const SmiRemoteProcessorRequest*)payload;
    if (req->processor_index >= g_processor_count) {
        resp.header.error_code = -2;
        send_response(fd, SMI_OP_GET_ASIC_INFO, request_id, &resp, sizeof(resp));
        return;
    }

    amdsmi_processor_handle handle = g_processor_handles[req->processor_index];

    amdsmi_asic_info_t asic_info;
    memset(&asic_info, 0, sizeof(asic_info));
    amdsmi_status_t status = amdsmi_get_gpu_asic_info(handle, &asic_info);
    if (status == AMDSMI_STATUS_SUCCESS) {
        strncpy(resp.market_name, asic_info.market_name, sizeof(resp.market_name) - 1);
        resp.vendor_id = asic_info.vendor_id;
        resp.device_id = (uint32_t)asic_info.device_id;
        resp.rev_id = asic_info.rev_id;
        resp.num_compute_units = asic_info.num_of_compute_units;
        strncpy(resp.asic_serial, asic_info.asic_serial, sizeof(resp.asic_serial) - 1);
        resp.header.error_code = 0;
    } else {
        resp.header.error_code = (int32_t)status;
    }

    send_response(fd, SMI_OP_GET_ASIC_INFO, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Dispatch
 * ============================================================================ */

int smi_worker_dispatch(int client_fd, uint16_t op_code, uint32_t request_id,
                       const void* payload, size_t payload_size) {
    SMI_LOG_DEBUG("Dispatching SMI op 0x%04x", op_code);

    switch (op_code) {
        case SMI_OP_INIT:
            handle_smi_init(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_SHUTDOWN:
            handle_smi_shutdown(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_PROCESSOR_COUNT:
            handle_smi_get_processor_count(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_GPU_METRICS:
            handle_smi_get_gpu_metrics(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_POWER_INFO:
            handle_smi_get_power_info(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_CLOCK_INFO:
            handle_smi_get_clock_info(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_TEMP_METRIC:
            handle_smi_get_temp_metric(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_GPU_ACTIVITY:
            handle_smi_get_gpu_activity(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_VRAM_USAGE:
            handle_smi_get_vram_usage(client_fd, request_id, payload, payload_size);
            break;
        case SMI_OP_GET_ASIC_INFO:
            handle_smi_get_asic_info(client_fd, request_id, payload, payload_size);
            break;
        default:
            SMI_LOG_ERROR("Unknown SMI op code: 0x%04x", op_code);
            send_simple_response(client_fd, op_code, request_id, -1);
            return -1;
    }

    return 0;
}
