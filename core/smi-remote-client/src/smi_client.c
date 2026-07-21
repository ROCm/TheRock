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
 * @file smi_client.c
 * @brief Remote AMD SMI client implementation
 *
 * Cross-platform: Works on Windows, macOS, and Linux.
 */

#include "smi_remote/smi_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"
#include "hip_remote/hip_remote_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Client State
 * ============================================================================ */

typedef struct {
    hip_socket_t socket_fd;
    hip_mutex_t lock;
    uint32_t next_request_id;
    bool connected;
    bool debug_enabled;
    char worker_host[256];
    int worker_port;
    bool initialized;
} SmiClientState;

static SmiClientState g_client = {
    .socket_fd = HIP_INVALID_SOCKET,
    .lock = HIP_MUTEX_INIT,
    .next_request_id = 1,
    .connected = false,
    .debug_enabled = false,
    .worker_port = HIP_REMOTE_DEFAULT_PORT,
    .initialized = false,
};

static hip_once_t g_init_once = HIP_ONCE_INIT;

/* ============================================================================
 * Logging
 * ============================================================================ */

#define LOG_DEBUG(fmt, ...) do { \
    if (g_client.debug_enabled) { \
        fprintf(stderr, "[SMI-Remote] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

#define LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[SMI-Remote ERROR] " fmt "\n", ##__VA_ARGS__)

/* ============================================================================
 * Network Helpers
 * ============================================================================ */

static int send_all(hip_socket_t fd, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    while (len > 0) {
        int n = hip_send(fd, p, len, 0);
        if (n < 0) {
            int err = hip_socket_errno();
            if (err == HIP_EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(hip_socket_t fd, void* data, size_t len) {
    uint8_t* p = (uint8_t*)data;
    while (len > 0) {
        int n = hip_recv(fd, p, len, 0);
        if (n < 0) {
            int err = hip_socket_errno();
            if (err == HIP_EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

/* ============================================================================
 * Connection Management
 * ============================================================================ */

static void init_client_state(void) {
    /* Initialize socket subsystem */
    hip_socket_init();

    /* Read configuration from environment */
    const char* host = getenv("TF_WORKER_HOST");
    if (host && *host) {
        strncpy(g_client.worker_host, host, sizeof(g_client.worker_host) - 1);
    }

    const char* port_str = getenv("TF_WORKER_PORT");
    if (port_str && *port_str) {
        g_client.worker_port = atoi(port_str);
        if (g_client.worker_port <= 0) {
            g_client.worker_port = HIP_REMOTE_DEFAULT_PORT;
        }
    }

    const char* debug = getenv("TF_DEBUG");
    g_client.debug_enabled = (debug != NULL && strcmp(debug, "1") == 0);

    LOG_DEBUG("Client configured: host=%s port=%d",
              g_client.worker_host, g_client.worker_port);
}

static int ensure_connected(void) {
    hip_call_once(&g_init_once, init_client_state);

    if (g_client.connected && g_client.socket_fd != HIP_INVALID_SOCKET) {
        return 0;
    }

    if (g_client.worker_host[0] == '\0') {
        LOG_ERROR("TF_WORKER_HOST not set");
        return -1;
    }

    LOG_DEBUG("Connecting to %s:%d...", g_client.worker_host, g_client.worker_port);

    /* Resolve hostname */
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", g_client.worker_port);

    int err = getaddrinfo(g_client.worker_host, port_str, &hints, &result);
    if (err != 0) {
        LOG_ERROR("Failed to resolve %s", g_client.worker_host);
        return -1;
    }

    /* Create socket */
    g_client.socket_fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (g_client.socket_fd == HIP_INVALID_SOCKET) {
        LOG_ERROR("socket() failed: %s", hip_socket_strerror(hip_socket_errno()));
        freeaddrinfo(result);
        return -1;
    }

    /* Set socket options */
    hip_set_nodelay(g_client.socket_fd);
    hip_set_socket_timeout(g_client.socket_fd, 30);

    /* Connect */
    if (connect(g_client.socket_fd, result->ai_addr, (int)result->ai_addrlen) != 0) {
        LOG_ERROR("connect() failed: %s", hip_socket_strerror(hip_socket_errno()));
        hip_close_socket(g_client.socket_fd);
        g_client.socket_fd = HIP_INVALID_SOCKET;
        freeaddrinfo(result);
        return -1;
    }

    freeaddrinfo(result);
    g_client.connected = true;
    LOG_DEBUG("Connected successfully");

    return 0;
}

static void disconnect(void) {
    if (g_client.socket_fd != HIP_INVALID_SOCKET) {
        hip_close_socket(g_client.socket_fd);
        g_client.socket_fd = HIP_INVALID_SOCKET;
    }
    g_client.connected = false;
}

/* ============================================================================
 * Request/Response
 * ============================================================================ */

static smi_remote_status_t send_request(
    uint16_t op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
) {
    hip_mutex_lock(&g_client.lock);

    if (ensure_connected() != 0) {
        hip_mutex_unlock(&g_client.lock);
        return SMI_STATUS_IO_ERROR;
    }

    /* Prepare header */
    HipRemoteHeader header;
    header.magic = HIP_REMOTE_MAGIC;
    header.version = HIP_REMOTE_VERSION;
    header.op_code = op_code;
    header.request_id = g_client.next_request_id++;
    header.payload_length = (uint32_t)request_size;
    header.flags = 0;

    LOG_DEBUG("Sending %s (id=%u, payload=%zu)",
              hip_remote_op_name((HipRemoteOpCode)op_code), header.request_id, request_size);

    /* Send header */
    if (send_all(g_client.socket_fd, &header, sizeof(header)) < 0) {
        LOG_ERROR("Failed to send header: %s", hip_socket_strerror(hip_socket_errno()));
        disconnect();
        hip_mutex_unlock(&g_client.lock);
        return SMI_STATUS_IO_ERROR;
    }

    /* Send payload */
    if (request_size > 0 && request) {
        if (send_all(g_client.socket_fd, request, request_size) < 0) {
            LOG_ERROR("Failed to send payload: %s", hip_socket_strerror(hip_socket_errno()));
            disconnect();
            hip_mutex_unlock(&g_client.lock);
            return SMI_STATUS_IO_ERROR;
        }
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client.socket_fd, &resp_header, sizeof(resp_header)) < 0) {
        LOG_ERROR("Failed to receive response header: %s", hip_socket_strerror(hip_socket_errno()));
        disconnect();
        hip_mutex_unlock(&g_client.lock);
        return SMI_STATUS_IO_ERROR;
    }

    /* Validate response */
    if (resp_header.magic != HIP_REMOTE_MAGIC) {
        LOG_ERROR("Invalid response magic");
        disconnect();
        hip_mutex_unlock(&g_client.lock);
        return SMI_STATUS_IO_ERROR;
    }

    /* Receive response payload */
    size_t recv_size = resp_header.payload_length;
    if (recv_size > response_size) {
        LOG_ERROR("Response too large: %zu > %zu", recv_size, response_size);
        disconnect();
        hip_mutex_unlock(&g_client.lock);
        return SMI_STATUS_IO_ERROR;
    }

    if (recv_size > 0 && response) {
        if (recv_all(g_client.socket_fd, response, recv_size) < 0) {
            LOG_ERROR("Failed to receive response payload: %s", hip_socket_strerror(hip_socket_errno()));
            disconnect();
            hip_mutex_unlock(&g_client.lock);
            return SMI_STATUS_IO_ERROR;
        }
    }

    LOG_DEBUG("Received response: %zu bytes", recv_size);

    hip_mutex_unlock(&g_client.lock);
    return SMI_STATUS_SUCCESS;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

smi_remote_status_t smi_remote_init(void) {
    hip_call_once(&g_init_once, init_client_state);

    hip_mutex_lock(&g_client.lock);

    if (g_client.initialized) {
        hip_mutex_unlock(&g_client.lock);
        return SMI_STATUS_SUCCESS;
    }

    hip_mutex_unlock(&g_client.lock);

    /* Send SMI_OP_INIT to worker */
    SmiRemoteInitRequest req;
    memset(&req, 0, sizeof(req));
    HipRemoteResponseHeader resp;

    smi_remote_status_t status = send_request(SMI_OP_INIT, &req, sizeof(req),
                                              &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.error_code != 0) {
        LOG_ERROR("SMI init failed on worker: %d", resp.error_code);
        return SMI_STATUS_API_FAILED;
    }

    hip_mutex_lock(&g_client.lock);
    g_client.initialized = true;
    hip_mutex_unlock(&g_client.lock);

    LOG_DEBUG("SMI initialized on remote worker");
    return SMI_STATUS_SUCCESS;
}

void smi_remote_shutdown(void) {
    hip_mutex_lock(&g_client.lock);

    disconnect();
    g_client.initialized = false;

    hip_mutex_unlock(&g_client.lock);
}

bool smi_remote_is_connected(void) {
    return g_client.connected;
}

smi_remote_status_t smi_remote_get_processor_count(uint32_t* count) {
    if (!count) {
        return SMI_STATUS_INVALID_ARGS;
    }

    SmiRemoteProcessorCountResponse resp;
    smi_remote_status_t status = send_request(SMI_OP_GET_PROCESSOR_COUNT,
                                              NULL, 0, &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.header.error_code != 0) {
        return SMI_STATUS_API_FAILED;
    }

    *count = resp.processor_count;
    return SMI_STATUS_SUCCESS;
}

smi_remote_status_t smi_remote_get_gpu_metrics(
    uint32_t processor_index,
    smi_remote_gpu_metrics_t* metrics
) {
    if (!metrics) {
        return SMI_STATUS_INVALID_ARGS;
    }

    SmiRemoteProcessorRequest req;
    memset(&req, 0, sizeof(req));
    req.processor_index = processor_index;
    SmiRemoteGpuMetricsResponse resp;

    smi_remote_status_t status = send_request(SMI_OP_GET_GPU_METRICS,
                                              &req, sizeof(req), &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.header.error_code != 0) {
        return SMI_STATUS_API_FAILED;
    }

    metrics->temperature_edge = resp.temperature_edge;
    metrics->temperature_hotspot = resp.temperature_hotspot;
    metrics->temperature_mem = resp.temperature_mem;
    metrics->power_watts = resp.average_socket_power;
    metrics->gfx_activity = resp.gfx_activity;
    metrics->mem_activity = resp.umc_activity;
    metrics->mm_activity = resp.mm_activity;
    metrics->gfx_clock_mhz = resp.current_gfxclk;
    metrics->mem_clock_mhz = resp.current_uclk;
    metrics->soc_clock_mhz = resp.current_socclk;
    metrics->vram_total_bytes = resp.vram_total;
    metrics->vram_used_bytes = resp.vram_used;
    metrics->fan_speed_rpm = resp.fan_speed_rpm;
    metrics->throttle_status = resp.throttle_status;

    return SMI_STATUS_SUCCESS;
}

smi_remote_status_t smi_remote_get_power_info(
    uint32_t processor_index,
    smi_remote_power_info_t* power_info
) {
    if (!power_info) {
        return SMI_STATUS_INVALID_ARGS;
    }

    SmiRemoteProcessorRequest req;
    memset(&req, 0, sizeof(req));
    req.processor_index = processor_index;
    SmiRemotePowerInfoResponse resp;

    smi_remote_status_t status = send_request(SMI_OP_GET_POWER_INFO,
                                              &req, sizeof(req), &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.header.error_code != 0) {
        return SMI_STATUS_API_FAILED;
    }

    power_info->current_power_watts = resp.current_socket_power;
    power_info->average_power_watts = resp.average_socket_power;
    power_info->gfx_voltage_mv = resp.gfx_voltage;
    power_info->soc_voltage_mv = resp.soc_voltage;
    power_info->mem_voltage_mv = resp.mem_voltage;
    power_info->power_limit_watts = resp.power_limit;

    return SMI_STATUS_SUCCESS;
}

smi_remote_status_t smi_remote_get_asic_info(
    uint32_t processor_index,
    smi_remote_asic_info_t* asic_info
) {
    if (!asic_info) {
        return SMI_STATUS_INVALID_ARGS;
    }

    SmiRemoteProcessorRequest req;
    memset(&req, 0, sizeof(req));
    req.processor_index = processor_index;
    SmiRemoteAsicInfoResponse resp;

    smi_remote_status_t status = send_request(SMI_OP_GET_ASIC_INFO,
                                              &req, sizeof(req), &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.header.error_code != 0) {
        return SMI_STATUS_API_FAILED;
    }

    strncpy(asic_info->market_name, resp.market_name, sizeof(asic_info->market_name) - 1);
    asic_info->vendor_id = resp.vendor_id;
    asic_info->device_id = resp.device_id;
    asic_info->rev_id = resp.rev_id;
    asic_info->num_compute_units = resp.num_compute_units;
    strncpy(asic_info->serial, resp.asic_serial, sizeof(asic_info->serial) - 1);

    return SMI_STATUS_SUCCESS;
}

smi_remote_status_t smi_remote_get_vram_usage(
    uint32_t processor_index,
    uint64_t* total_bytes,
    uint64_t* used_bytes
) {
    if (!total_bytes || !used_bytes) {
        return SMI_STATUS_INVALID_ARGS;
    }

    SmiRemoteProcessorRequest req;
    memset(&req, 0, sizeof(req));
    req.processor_index = processor_index;
    SmiRemoteVramUsageResponse resp;

    smi_remote_status_t status = send_request(SMI_OP_GET_VRAM_USAGE,
                                              &req, sizeof(req), &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.header.error_code != 0) {
        return SMI_STATUS_API_FAILED;
    }

    *total_bytes = resp.vram_total;
    *used_bytes = resp.vram_used;

    return SMI_STATUS_SUCCESS;
}

smi_remote_status_t smi_remote_get_gpu_activity(
    uint32_t processor_index,
    uint32_t* gfx_activity,
    uint32_t* mem_activity,
    uint32_t* mm_activity
) {
    if (!gfx_activity || !mem_activity || !mm_activity) {
        return SMI_STATUS_INVALID_ARGS;
    }

    SmiRemoteProcessorRequest req;
    memset(&req, 0, sizeof(req));
    req.processor_index = processor_index;
    SmiRemoteGpuActivityResponse resp;

    smi_remote_status_t status = send_request(SMI_OP_GET_GPU_ACTIVITY,
                                              &req, sizeof(req), &resp, sizeof(resp));
    if (status != SMI_STATUS_SUCCESS) {
        return status;
    }

    if (resp.header.error_code != 0) {
        return SMI_STATUS_API_FAILED;
    }

    *gfx_activity = resp.gfx_activity;
    *mem_activity = resp.umc_activity;
    *mm_activity = resp.mm_activity;

    return SMI_STATUS_SUCCESS;
}

const char* smi_remote_status_string(smi_remote_status_t status) {
    switch (status) {
        case SMI_STATUS_SUCCESS: return "success";
        case SMI_STATUS_INVALID_ARGS: return "invalid arguments";
        case SMI_STATUS_NOT_SUPPORTED: return "not supported";
        case SMI_STATUS_NOT_FOUND: return "not found";
        case SMI_STATUS_NOT_INITIALIZED: return "not initialized";
        case SMI_STATUS_IO_ERROR: return "I/O error";
        case SMI_STATUS_API_FAILED: return "API failed";
        default: return "unknown error";
    }
}
