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
 * @file hip_client.c
 * @brief Core client implementation for remote HIP execution
 *
 * Cross-platform: Works on Windows, macOS, and Linux.
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"
#include "hip_remote/hip_remote_platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

static HipRemoteClientState g_client_state = {
    .socket_fd = HIP_INVALID_SOCKET,
    .lock = HIP_MUTEX_INIT,
    .next_request_id = 1,
    .connected = false,
    .debug_enabled = false,
    .worker_host = "localhost",
    .worker_port = HIP_REMOTE_DEFAULT_PORT,
    .connect_timeout_sec = 30,
    .io_timeout_sec = 60,
    .last_error = hipSuccess
};

static hip_once_t g_init_once = HIP_ONCE_INIT;
static int g_wsa_initialized = 0;

/* ============================================================================
 * Logging
 * ============================================================================ */

void hip_remote_log_debug(const char* fmt, ...) {
    if (!g_client_state.debug_enabled) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[HIP-Remote] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void hip_remote_log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[HIP-Remote ERROR] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * Send all bytes, handling partial sends.
 */
static int send_all(hip_socket_t fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
        int n = hip_send(fd, p, len, HIP_MSG_NOSIGNAL);
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

/**
 * Receive all bytes, handling partial receives.
 */
static int recv_all(hip_socket_t fd, void* buf, size_t len) {
    uint8_t* p = (uint8_t*)buf;
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

/**
 * Mark connection as disconnected.
 */
static void mark_disconnected_locked(const char* reason) {
    if (reason) {
        int err = hip_socket_errno();
        hip_remote_log_debug("Disconnected: %s (errno=%d: %s)",
                             reason, err, hip_socket_strerror(err));
    }
    if (g_client_state.socket_fd != HIP_INVALID_SOCKET) {
        hip_close_socket(g_client_state.socket_fd);
    }
    g_client_state.socket_fd = HIP_INVALID_SOCKET;
    g_client_state.connected = false;
}

/**
 * Initialize client state from environment.
 */
static void init_from_environment(void) {
    /* Initialize socket subsystem (Winsock on Windows, no-op on POSIX) */
    if (hip_socket_init() != 0) {
        hip_remote_log_error("Failed to initialize socket subsystem");
    } else {
        g_wsa_initialized = 1;
    }

    const char* debug = getenv("TF_DEBUG");
    if (debug && strcmp(debug, "1") == 0) {
        g_client_state.debug_enabled = true;
    }

    const char* host = getenv("TF_WORKER_HOST");
    if (host && host[0] != '\0') {
        strncpy(g_client_state.worker_host, host,
                sizeof(g_client_state.worker_host) - 1);
        g_client_state.worker_host[sizeof(g_client_state.worker_host) - 1] = '\0';
    }

    const char* port_str = getenv("TF_WORKER_PORT");
    if (port_str && port_str[0] != '\0') {
        int port = atoi(port_str);
        if (port > 0 && port < 65536) {
            g_client_state.worker_port = port;
        }
    }

    const char* connect_timeout = getenv("TF_CONNECT_TIMEOUT");
    if (connect_timeout && connect_timeout[0] != '\0') {
        int timeout = atoi(connect_timeout);
        if (timeout > 0) {
            g_client_state.connect_timeout_sec = timeout;
        }
    }

    const char* io_timeout = getenv("TF_IO_TIMEOUT");
    if (io_timeout && io_timeout[0] != '\0') {
        int timeout = atoi(io_timeout);
        if (timeout > 0) {
            g_client_state.io_timeout_sec = timeout;
        }
    }

    hip_remote_log_debug("Client initialized: host=%s port=%d",
                         g_client_state.worker_host, g_client_state.worker_port);
}

/**
 * Connect to worker service.
 */
static int connect_to_worker_locked(void) {
    if (g_client_state.connected) {
        return 0;
    }

    hip_remote_log_debug("Connecting to %s:%d...",
                         g_client_state.worker_host, g_client_state.worker_port);

    /* Resolve hostname using getaddrinfo (cross-platform) */
    struct addrinfo hints, *result = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", g_client_state.worker_port);

    int gai_err = getaddrinfo(g_client_state.worker_host, port_str, &hints, &result);
    if (gai_err != 0) {
        hip_remote_log_error("Failed to resolve host: %s", g_client_state.worker_host);
        return -1;
    }

    /* Create socket */
    g_client_state.socket_fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (g_client_state.socket_fd == HIP_INVALID_SOCKET) {
        hip_remote_log_error("Failed to create socket: %s",
                             hip_socket_strerror(hip_socket_errno()));
        freeaddrinfo(result);
        return -1;
    }

    /* Set socket options */
    hip_set_nodelay(g_client_state.socket_fd);
    hip_set_socket_timeout(g_client_state.socket_fd, g_client_state.io_timeout_sec);

    /* Connect */
    if (connect(g_client_state.socket_fd,
                result->ai_addr, (int)result->ai_addrlen) != 0) {
        hip_remote_log_error("Failed to connect: %s",
                             hip_socket_strerror(hip_socket_errno()));
        mark_disconnected_locked("connect");
        freeaddrinfo(result);
        return -1;
    }

    freeaddrinfo(result);
    g_client_state.connected = true;
    hip_remote_log_debug("Connected successfully");

    /* Send init message */
    HipRemoteHeader header;
    hip_remote_init_header(&header, HIP_OP_INIT,
                           g_client_state.next_request_id++, 0);

    if (send_all(g_client_state.socket_fd, &header, sizeof(header)) != 0) {
        mark_disconnected_locked("send init");
        return -1;
    }

    /* Receive init response */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv init header");
        return -1;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        hip_remote_log_error("Invalid response header from worker");
        mark_disconnected_locked("bad header");
        return -1;
    }

    if (resp_header.payload_length > 0) {
        HipRemoteResponseHeader resp;
        size_t to_read = sizeof(resp) < resp_header.payload_length ?
                         sizeof(resp) : resp_header.payload_length;
        if (recv_all(g_client_state.socket_fd, &resp, to_read) != 0) {
            mark_disconnected_locked("recv init body");
            return -1;
        }
        if (resp.error_code != hipSuccess) {
            hip_remote_log_error("Worker init failed: %d", resp.error_code);
            mark_disconnected_locked("init error");
            return -1;
        }
    }

    hip_remote_log_debug("Init handshake complete");
    return 0;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

HipRemoteClientState* hip_remote_get_client_state(void) {
    hip_call_once(&g_init_once, init_from_environment);
    return &g_client_state;
}

int hip_remote_ensure_connected(void) {
    hip_call_once(&g_init_once, init_from_environment);

    hip_mutex_lock(&g_client_state.lock);
    int result = connect_to_worker_locked();
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

void hip_remote_disconnect(void) {
    hip_mutex_lock(&g_client_state.lock);

    if (g_client_state.connected) {
        /* Send shutdown message (best effort) */
        HipRemoteHeader header;
        hip_remote_init_header(&header, HIP_OP_SHUTDOWN,
                               g_client_state.next_request_id++, 0);
        (void)send_all(g_client_state.socket_fd, &header, sizeof(header));
        mark_disconnected_locked("shutdown");
    }

    hip_mutex_unlock(&g_client_state.lock);
}

bool hip_remote_is_connected(void) {
    hip_mutex_lock(&g_client_state.lock);
    bool connected = g_client_state.connected;
    hip_mutex_unlock(&g_client_state.lock);
    return connected;
}

hipError_t hip_remote_request(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
) {
    hip_call_once(&g_init_once, init_from_environment);

    hip_mutex_lock(&g_client_state.lock);

    /* Ensure connected */
    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Send request */
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint32_t)request_size);

    hip_remote_log_debug("Sending %s (id=%u, payload=%zu)",
                         hip_remote_op_name(op_code),
                         header.request_id, request_size);

    if (send_all(g_client_state.socket_fd, &header, sizeof(header)) != 0) {
        mark_disconnected_locked("send header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (request && request_size > 0) {
        if (send_all(g_client_state.socket_fd, request, request_size) != 0) {
            mark_disconnected_locked("send payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    /* Receive response payload */
    hipError_t result = hipSuccess;
    if (response && response_size > 0 && resp_header.payload_length > 0) {
        size_t to_read = resp_header.payload_length < response_size ?
                         resp_header.payload_length : response_size;
        if (recv_all(g_client_state.socket_fd, response, to_read) != 0) {
            mark_disconnected_locked("recv payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        /* Drain extra bytes if response larger than buffer */
        if (resp_header.payload_length > response_size) {
            size_t extra = resp_header.payload_length - response_size;
            uint8_t drain[256];
            while (extra > 0) {
                size_t chunk = extra < sizeof(drain) ? extra : sizeof(drain);
                if (recv_all(g_client_state.socket_fd, drain, chunk) != 0) {
                    mark_disconnected_locked("drain");
                    hip_mutex_unlock(&g_client_state.lock);
                    g_client_state.last_error = hipErrorNotInitialized;
                    return hipErrorNotInitialized;
                }
                extra -= chunk;
            }
        }

        /* Extract error code from response */
        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    hip_remote_log_debug("Received response for %s: error=%d",
                         hip_remote_op_name(op_code), result);

    g_client_state.last_error = result;
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

hipError_t hip_remote_request_with_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    const void* data,
    size_t data_size,
    void* response,
    size_t response_size
) {
    hip_call_once(&g_init_once, init_from_environment);

    hip_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Send request with inline data */
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint32_t)(request_size + data_size));
    header.flags |= HIP_REMOTE_FLAG_HAS_INLINE_DATA;

    hip_remote_log_debug("Sending %s with data (id=%u, payload=%zu, data=%zu)",
                         hip_remote_op_name(op_code),
                         header.request_id, request_size, data_size);

    if (send_all(g_client_state.socket_fd, &header, sizeof(header)) != 0) {
        mark_disconnected_locked("send header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (request && request_size > 0) {
        if (send_all(g_client_state.socket_fd, request, request_size) != 0) {
            mark_disconnected_locked("send payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    if (data && data_size > 0) {
        if (send_all(g_client_state.socket_fd, data, data_size) != 0) {
            mark_disconnected_locked("send data");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    /* Receive response */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    hipError_t result = hipSuccess;
    if (response && response_size > 0 && resp_header.payload_length > 0) {
        size_t to_read = resp_header.payload_length < response_size ?
                         resp_header.payload_length : response_size;
        if (recv_all(g_client_state.socket_fd, response, to_read) != 0) {
            mark_disconnected_locked("recv payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    g_client_state.last_error = result;
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

hipError_t hip_remote_request_receive_data(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size,
    void* data_out,
    size_t data_size
) {
    hip_call_once(&g_init_once, init_from_environment);

    hip_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    /* Send request */
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code,
                           g_client_state.next_request_id++,
                           (uint32_t)request_size);

    hip_remote_log_debug("Sending %s expecting data (id=%u, payload=%zu, expect_data=%zu)",
                         hip_remote_op_name(op_code),
                         header.request_id, request_size, data_size);

    if (send_all(g_client_state.socket_fd, &header, sizeof(header)) != 0) {
        mark_disconnected_locked("send header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (request && request_size > 0) {
        if (send_all(g_client_state.socket_fd, request, request_size) != 0) {
            mark_disconnected_locked("send payload");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        hip_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    /* Receive response struct */
    hipError_t result = hipSuccess;
    if (response && response_size > 0) {
        if (recv_all(g_client_state.socket_fd, response, response_size) != 0) {
            mark_disconnected_locked("recv response");
            hip_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    /* Receive inline data */
    if (result == hipSuccess && data_out && data_size > 0) {
        size_t remaining = resp_header.payload_length - response_size;
        size_t to_read = remaining < data_size ? remaining : data_size;
        if (to_read > 0) {
            if (recv_all(g_client_state.socket_fd, data_out, to_read) != 0) {
                mark_disconnected_locked("recv data");
                hip_mutex_unlock(&g_client_state.lock);
                g_client_state.last_error = hipErrorNotInitialized;
                return hipErrorNotInitialized;
            }
        }
    }

    g_client_state.last_error = result;
    hip_mutex_unlock(&g_client_state.lock);
    return result;
}

/* ============================================================================
 * Library Init/Cleanup
 *
 * On POSIX: __attribute__((constructor/destructor))
 * On Windows: DllMain or explicit init via first API call
 * ============================================================================ */

#ifdef _WIN32

/* On Windows, DllMain handles init/cleanup for DLL builds.
 * For static/exe builds, we rely on hip_call_once in each API call. */
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    (void)hinstDLL; (void)lpReserved;
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            init_from_environment();
            hip_remote_log_debug("Remote HIP client library loaded");
            break;
        case DLL_PROCESS_DETACH:
            hip_remote_disconnect();
            if (g_wsa_initialized) {
                hip_socket_cleanup();
            }
            hip_remote_log_debug("Remote HIP client library unloaded");
            break;
    }
    return TRUE;
}

#else

__attribute__((constructor))
static void hip_remote_client_init(void) {
    init_from_environment();
    hip_remote_log_debug("Remote HIP client library loaded");
}

__attribute__((destructor))
static void hip_remote_client_cleanup(void) {
    hip_remote_disconnect();
    hip_remote_log_debug("Remote HIP client library unloaded");
}

#endif
