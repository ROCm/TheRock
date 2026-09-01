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
 * @file hip_remote_platform.h
 * @brief Cross-platform abstractions for sockets, threads, and packing
 *
 * Provides a unified API for:
 * - TCP sockets (POSIX vs Winsock2)
 * - Mutexes and once-init (pthread vs Windows SRWLOCK/INIT_ONCE)
 * - Struct packing (__attribute__((packed)) vs #pragma pack)
 * - Library init/cleanup (constructor/destructor vs DllMain)
 */

#ifndef HIP_REMOTE_PLATFORM_H
#define HIP_REMOTE_PLATFORM_H

#ifdef _WIN32
/* ============================================================================
 * Windows Platform
 * ============================================================================ */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#pragma comment(lib, "ws2_32.lib")

/* Socket type abstraction */
typedef SOCKET hip_socket_t;
#define HIP_INVALID_SOCKET INVALID_SOCKET
#define HIP_SOCKET_ERROR   SOCKET_ERROR

/* Socket operations */
static inline int hip_socket_init(void) {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa);
}
static inline void hip_socket_cleanup(void) {
    WSACleanup();
}
static inline int hip_close_socket(hip_socket_t s) {
    return closesocket(s);
}
static inline int hip_socket_errno(void) {
    return WSAGetLastError();
}
static inline const char* hip_socket_strerror(int err) {
    /* Thread-local static buffer for FormatMessage */
    static __declspec(thread) char buf[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, (DWORD)err, 0, buf, sizeof(buf), NULL);
    /* Strip trailing newline */
    size_t len = strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) {
        buf[--len] = '\0';
    }
    return buf;
}

/* send/recv return int on Windows */
static inline int hip_send(hip_socket_t s, const void* buf, size_t len, int flags) {
    return send(s, (const char*)buf, (int)len, flags);
}
static inline int hip_recv(hip_socket_t s, void* buf, size_t len, int flags) {
    return recv(s, (char*)buf, (int)len, flags);
}

/* Timeout uses DWORD (milliseconds) on Windows */
static inline void hip_set_socket_timeout(hip_socket_t s, int timeout_sec) {
    DWORD tv = (DWORD)(timeout_sec * 1000);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
}

/* TCP_NODELAY */
static inline void hip_set_nodelay(hip_socket_t s) {
    int nodelay = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
}

/* Mutex abstraction using SRWLOCK (slim reader/writer lock) */
typedef SRWLOCK hip_mutex_t;
#define HIP_MUTEX_INIT SRWLOCK_INIT

static inline void hip_mutex_lock(hip_mutex_t* m) { AcquireSRWLockExclusive(m); }
static inline void hip_mutex_unlock(hip_mutex_t* m) { ReleaseSRWLockExclusive(m); }

/* Once-init abstraction using INIT_ONCE */
typedef INIT_ONCE hip_once_t;
#define HIP_ONCE_INIT INIT_ONCE_STATIC_INIT

typedef void (*hip_once_fn)(void);

/* Windows INIT_ONCE callback wrapper */
static inline BOOL CALLBACK _hip_once_callback(PINIT_ONCE once, PVOID param, PVOID* ctx) {
    (void)once; (void)ctx;
    ((hip_once_fn)param)();
    return TRUE;
}
static inline void hip_call_once(hip_once_t* once, hip_once_fn fn) {
    InitOnceExecuteOnce(once, _hip_once_callback, (PVOID)fn, NULL);
}

/* EINTR / EPIPE / ECONNRESET equivalents */
#define HIP_EINTR       WSAEINTR
#define HIP_EPIPE       WSAECONNRESET
#define HIP_ECONNRESET  WSAECONNRESET

/* setenv equivalent - just use a macro */
#define hip_setenv(name, value) _putenv_s(name, value)

/* MSG_NOSIGNAL doesn't exist on Windows (SIGPIPE doesn't exist) */
#define HIP_MSG_NOSIGNAL 0

#else
/* ============================================================================
 * POSIX Platform (macOS, Linux)
 * ============================================================================ */

/* Ensure POSIX.1-2008 functions (getaddrinfo, etc.) are available */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

/* Socket type abstraction */
typedef int hip_socket_t;
#define HIP_INVALID_SOCKET (-1)
#define HIP_SOCKET_ERROR   (-1)

/* Socket operations */
static inline int hip_socket_init(void) { return 0; /* no-op on POSIX */ }
static inline void hip_socket_cleanup(void) { /* no-op on POSIX */ }
static inline int hip_close_socket(hip_socket_t s) { return close(s); }
static inline int hip_socket_errno(void) { return errno; }
static inline const char* hip_socket_strerror(int err) { return strerror(err); }

/* send/recv wrappers (return int to match Windows API; max payload is 64MB) */
static inline int hip_send(hip_socket_t s, const void* buf, size_t len, int flags) {
    return (int)send(s, buf, len, flags);
}
static inline int hip_recv(hip_socket_t s, void* buf, size_t len, int flags) {
    return (int)recv(s, buf, len, flags);
}

/* Timeout uses struct timeval on POSIX */
static inline void hip_set_socket_timeout(hip_socket_t s, int timeout_sec) {
    struct timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

/* TCP_NODELAY */
static inline void hip_set_nodelay(hip_socket_t s) {
    int nodelay = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
}

/* Mutex abstraction using pthread_mutex */
typedef pthread_mutex_t hip_mutex_t;
#define HIP_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER

static inline void hip_mutex_lock(hip_mutex_t* m) { pthread_mutex_lock(m); }
static inline void hip_mutex_unlock(hip_mutex_t* m) { pthread_mutex_unlock(m); }

/* Once-init abstraction using pthread_once */
typedef pthread_once_t hip_once_t;
#define HIP_ONCE_INIT PTHREAD_ONCE_INIT

typedef void (*hip_once_fn)(void);
static inline void hip_call_once(hip_once_t* once, hip_once_fn fn) {
    pthread_once(once, fn);
}

/* Error codes */
#define HIP_EINTR       EINTR
#define HIP_EPIPE       EPIPE
#define HIP_ECONNRESET  ECONNRESET

/* setenv */
static inline int hip_setenv(const char* name, const char* value) {
    return setenv(name, value, 1);
}

/* MSG_NOSIGNAL */
#ifdef MSG_NOSIGNAL
#define HIP_MSG_NOSIGNAL MSG_NOSIGNAL
#else
#define HIP_MSG_NOSIGNAL 0
#endif

#endif /* _WIN32 */

/*
 * Note: Cross-platform struct packing macros (HIP_PACK_PUSH, HIP_PACK_POP,
 * HIP_PACKED_ATTR) are defined in hip_remote_protocol.h, which is the
 * canonical location for all wire-format struct definitions.
 */

#endif /* HIP_REMOTE_PLATFORM_H */
