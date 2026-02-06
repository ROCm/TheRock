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
 */

#define _POSIX_C_SOURCE 200809L

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

static HipRemoteClientState g_client_state = {
    .socket_fd = -1,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .next_request_id = 1,
    .connected = false,
    .debug_enabled = false,
    .worker_host = "localhost",
    .worker_port = HIP_REMOTE_DEFAULT_PORT,
    .connect_timeout_sec = 30,
    .io_timeout_sec = 60,
    .last_error = hipSuccess
};

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

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
static int send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, p, len, 0);
#endif
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = EPIPE;
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
static int recv_all(int fd, void* buf, size_t len) {
    uint8_t* p = (uint8_t*)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = ECONNRESET;
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
        hip_remote_log_debug("Disconnected: %s (errno=%d: %s)",
                             reason, errno, strerror(errno));
    }
    if (g_client_state.socket_fd >= 0) {
        close(g_client_state.socket_fd);
    }
    g_client_state.socket_fd = -1;
    g_client_state.connected = false;
}

/**
 * Initialize client state from environment.
 */
static void init_from_environment(void) {
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

    /* Create socket */
    g_client_state.socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_client_state.socket_fd < 0) {
        hip_remote_log_error("Failed to create socket: %s", strerror(errno));
        return -1;
    }

    /* Set socket options */
    int nodelay = 1;
    setsockopt(g_client_state.socket_fd, IPPROTO_TCP, TCP_NODELAY,
               &nodelay, sizeof(nodelay));

    struct timeval timeout;
    timeout.tv_sec = g_client_state.io_timeout_sec;
    timeout.tv_usec = 0;
    setsockopt(g_client_state.socket_fd, SOL_SOCKET, SO_RCVTIMEO,
               &timeout, sizeof(timeout));
    setsockopt(g_client_state.socket_fd, SOL_SOCKET, SO_SNDTIMEO,
               &timeout, sizeof(timeout));

    /* Resolve hostname */
    struct hostent* server = gethostbyname(g_client_state.worker_host);
    if (!server) {
        hip_remote_log_error("Failed to resolve host: %s", g_client_state.worker_host);
        mark_disconnected_locked("resolve");
        return -1;
    }

    /* Connect */
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr_list[0], server->h_length);
    server_addr.sin_port = htons((uint16_t)g_client_state.worker_port);

    if (connect(g_client_state.socket_fd,
                (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        hip_remote_log_error("Failed to connect: %s", strerror(errno));
        mark_disconnected_locked("connect");
        return -1;
    }

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
        if (recv_all(g_client_state.socket_fd, &resp,
                     sizeof(resp) < resp_header.payload_length ?
                     sizeof(resp) : resp_header.payload_length) != 0) {
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
    pthread_once(&g_init_once, init_from_environment);
    return &g_client_state;
}

int hip_remote_ensure_connected(void) {
    pthread_once(&g_init_once, init_from_environment);

    pthread_mutex_lock(&g_client_state.lock);
    int result = connect_to_worker_locked();
    pthread_mutex_unlock(&g_client_state.lock);
    return result;
}

void hip_remote_disconnect(void) {
    pthread_mutex_lock(&g_client_state.lock);

    if (g_client_state.connected) {
        /* Send shutdown message (best effort) */
        HipRemoteHeader header;
        hip_remote_init_header(&header, HIP_OP_SHUTDOWN,
                               g_client_state.next_request_id++, 0);
        (void)send_all(g_client_state.socket_fd, &header, sizeof(header));
        mark_disconnected_locked("shutdown");
    }

    pthread_mutex_unlock(&g_client_state.lock);
}

bool hip_remote_is_connected(void) {
    pthread_mutex_lock(&g_client_state.lock);
    bool connected = g_client_state.connected;
    pthread_mutex_unlock(&g_client_state.lock);
    return connected;
}

hipError_t hip_remote_request(
    HipRemoteOpCode op_code,
    const void* request,
    size_t request_size,
    void* response,
    size_t response_size
) {
    pthread_once(&g_init_once, init_from_environment);

    pthread_mutex_lock(&g_client_state.lock);

    /* Ensure connected */
    if (connect_to_worker_locked() != 0) {
        pthread_mutex_unlock(&g_client_state.lock);
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
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (request && request_size > 0) {
        if (send_all(g_client_state.socket_fd, request, request_size) != 0) {
            mark_disconnected_locked("send payload");
            pthread_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        pthread_mutex_unlock(&g_client_state.lock);
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
            pthread_mutex_unlock(&g_client_state.lock);
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
                    pthread_mutex_unlock(&g_client_state.lock);
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
    pthread_mutex_unlock(&g_client_state.lock);
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
    pthread_once(&g_init_once, init_from_environment);

    pthread_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        pthread_mutex_unlock(&g_client_state.lock);
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
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (request && request_size > 0) {
        if (send_all(g_client_state.socket_fd, request, request_size) != 0) {
            mark_disconnected_locked("send payload");
            pthread_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    if (data && data_size > 0) {
        if (send_all(g_client_state.socket_fd, data, data_size) != 0) {
            mark_disconnected_locked("send data");
            pthread_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    /* Receive response */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    hipError_t result = hipSuccess;
    if (response && response_size > 0 && resp_header.payload_length > 0) {
        size_t to_read = resp_header.payload_length < response_size ?
                         resp_header.payload_length : response_size;
        if (recv_all(g_client_state.socket_fd, response, to_read) != 0) {
            mark_disconnected_locked("recv payload");
            pthread_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }

        HipRemoteResponseHeader* resp = (HipRemoteResponseHeader*)response;
        result = (hipError_t)resp->error_code;
    }

    g_client_state.last_error = result;
    pthread_mutex_unlock(&g_client_state.lock);
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
    pthread_once(&g_init_once, init_from_environment);

    pthread_mutex_lock(&g_client_state.lock);

    if (connect_to_worker_locked() != 0) {
        pthread_mutex_unlock(&g_client_state.lock);
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
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (request && request_size > 0) {
        if (send_all(g_client_state.socket_fd, request, request_size) != 0) {
            mark_disconnected_locked("send payload");
            pthread_mutex_unlock(&g_client_state.lock);
            g_client_state.last_error = hipErrorNotInitialized;
            return hipErrorNotInitialized;
        }
    }

    /* Receive response header */
    HipRemoteHeader resp_header;
    if (recv_all(g_client_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) {
        mark_disconnected_locked("recv header");
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorNotInitialized;
        return hipErrorNotInitialized;
    }

    if (hip_remote_validate_header(&resp_header) != 0) {
        mark_disconnected_locked("invalid header");
        pthread_mutex_unlock(&g_client_state.lock);
        g_client_state.last_error = hipErrorInvalidValue;
        return hipErrorInvalidValue;
    }

    /* Receive response struct */
    hipError_t result = hipSuccess;
    if (response && response_size > 0) {
        if (recv_all(g_client_state.socket_fd, response, response_size) != 0) {
            mark_disconnected_locked("recv response");
            pthread_mutex_unlock(&g_client_state.lock);
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
                pthread_mutex_unlock(&g_client_state.lock);
                g_client_state.last_error = hipErrorNotInitialized;
                return hipErrorNotInitialized;
            }
        }
    }

    g_client_state.last_error = result;
    pthread_mutex_unlock(&g_client_state.lock);
    return result;
}

/* ============================================================================
 * Library Constructor/Destructor
 * ============================================================================ */

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
