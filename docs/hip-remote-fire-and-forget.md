# Fire-and-Forget Kernel Launches in hip-remote

## Overview

hip-remote enables a Windows client to execute GPU workloads on a remote Linux
machine equipped with AMD MI300X GPUs. HIP API calls are intercepted on the
client, serialized over TCP, and dispatched to a worker process on the GPU
server.

This document describes the **fire-and-forget** optimization that eliminates
synchronous network round-trips for asynchronous GPU operations, dramatically
reducing the impact of network latency on throughput.

---

## 1. The Problem: Synchronous Round-Trips

In the original protocol every HIP API call — even those that are inherently
asynchronous on the GPU — required a full TCP round-trip before the client
could issue the next call:

```
 Client (Windows)                         Worker (Linux + MI300X)
 ────────────────                         ──────────────────────

  hipLaunchKernel ──── request ──────────►  dispatch to GPU
                  ◄─── response (400ms) ──  hipSuccess

  hipMemcpyAsync  ──── request ──────────►  enqueue copy
                  ◄─── response (400ms) ──  hipSuccess

  hipLaunchKernel ──── request ──────────►  dispatch to GPU
                  ◄─── response (400ms) ──  hipSuccess
                       ...
```

With ~200 ms one-way latency (~400 ms round-trip), a GPT-2 forward pass
requiring hundreds of operations serializes into:

```
  total_time ≈ 400ms × N_operations + GPU_compute_time

  Example: 300 operations × 400ms = 120 seconds of pure network wait
           before any GPU time is even counted.
```

The GPU sits idle between operations while bytes travel across the internet.

---

## 2. The Solution: Fire-and-Forget Pipelining

The key insight: most GPU operations are **asynchronous** — the HIP runtime
returns immediately and the GPU executes later. The only reason we waited for
a response was to check the error code, which is almost always `hipSuccess`.

Fire-and-forget sends the request and **immediately moves on** to the next
operation without waiting for a response:

```
 Client (Windows)                         Worker (Linux + MI300X)
 ────────────────                         ──────────────────────

  hipLaunchKernel ──── request (no reply) ─►┐
  hipMemcpy H2D  ──── request (no reply) ──►│  worker processes
  hipLaunchKernel ──── request (no reply) ─►│  each request in
  hipMemset       ──── request (no reply) ─►│  order as it
  hipLaunchKernel ──── request (no reply) ─►│  arrives from the
  hipFree         ──── request (no reply) ─►┘  TCP stream
                       ...
  hipDeviceSynchronize ─── request ────────►  flush GPU queue
                       ◄─── response ──────  hipSuccess (or error)
```

All fire-and-forget requests are **pipelined** into the TCP stream. The
worker drains them in order. The client only blocks when it genuinely needs
a result — a synchronization point, a D2H memory copy, or an allocation
that returns a pointer.

```
  total_time ≈ network_transfer_time(all_payloads) + GPU_compute_time

  The 400ms-per-operation penalty is gone.
```

---

## 3. Protocol Design

### 3.1 The NO_REPLY Flag

A single bit in the request header flags field:

```c
#define HIP_REMOTE_FLAG_NO_REPLY  (1u << 3)
```

When set, the worker processes the request normally but **skips sending a
response**. The client returns `hipSuccess` immediately after the `send()`.

### 3.2 Client-Side Functions

Two new request functions mirror the existing synchronous ones:

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│ Synchronous (waits for response)    │ Fire-and-Forget (no response)       │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ hip_remote_request()                │ hip_remote_request_fire_and_forget()│
│ hip_remote_request_with_data()      │ hip_remote_request_with_data_       │
│                                     │   fire_and_forget()                 │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

The `_with_data` variant is used for H2D memcpy where the host buffer is
sent inline with the request (flag `HIP_REMOTE_FLAG_HAS_INLINE_DATA` is
combined with `HIP_REMOTE_FLAG_NO_REPLY`).

### 3.3 Worker-Side Suppression

The worker uses a global flag checked inside `send_response()`:

```c
static int g_suppress_response = 0;

static int send_response(...) {
    if (g_suppress_response) return 0;  // no-op
    // ... normal send logic ...
}
```

Before dispatching each request, the dispatch loop sets the flag based on
the incoming header:

```c
g_suppress_response = (header.flags & HIP_REMOTE_FLAG_NO_REPLY) != 0;
```

This approach requires **zero changes** to existing handler functions — they
call `send_response()` / `send_simple_response()` as before, and the calls
silently become no-ops for fire-and-forget requests.

---

## 4. Which Operations Are Fire-and-Forget?

```
┌──────────────────────────┬─────────────┬──────────────────────────────────┐
│ Operation                │ Fire & Forget│ Reason                          │
├──────────────────────────┼─────────────┼──────────────────────────────────┤
│ hipLaunchKernel          │     ✓       │ Async GPU dispatch, no return    │
│ hipModuleLaunchKernel    │     ✓       │ Same — flat kernarg path         │
│ hipMemcpy H2D            │     ✓       │ Data flows client → worker only  │
│ hipMemcpy D2D            │     ✓       │ No data crosses the network      │
│ hipMemcpyAsync (H2D/D2D) │     ✓       │ Same as above                    │
│ hipMemset / Async        │     ✓       │ No return value needed           │
│ hipFree / FreeAsync      │     ✓       │ No return value needed           │
│ hipEventRecord           │     ✓       │ No return value needed           │
├──────────────────────────┼─────────────┼──────────────────────────────────┤
│ hipMemcpy D2H            │     ✗       │ Data must come back to client    │
│ hipMalloc / MallocAsync  │     ✗       │ Returns device pointer           │
│ hipDeviceSynchronize     │     ✗       │ Synchronization barrier          │
│ hipStreamSynchronize     │     ✗       │ Synchronization barrier          │
│ hipEventSynchronize      │     ✗       │ Synchronization barrier          │
│ hipModuleLoadData        │     ✗       │ Returns module handle            │
│ hipModuleGetFunction     │     ✗       │ Returns function handle          │
│ hipGetDeviceProperties   │     ✗       │ Returns data to client           │
└──────────────────────────┴─────────────┴──────────────────────────────────┘
```

**Rule of thumb:** if the client needs a return value or data from the
worker, it must wait. Everything else can be fire-and-forget.

---

## 5. Why It's Safe

Three properties guarantee correctness:

### 5.1 TCP Preserves Ordering

TCP is a byte stream — requests arrive at the worker in the exact order the
client sent them. There is no reordering.

### 5.2 Worker Is Single-Threaded

The worker reads one request at a time from the socket, fully processes it
(including the HIP API call), then reads the next. So a fire-and-forget
`hipMemcpy H2D` completes on the worker before the subsequent
`hipLaunchKernel` is even read from the socket.

```
  Worker event loop:
  ┌─────────────────────────────────────────────────┐
  │ while (connected) {                             │
  │   read header + payload from TCP                │
  │   set g_suppress_response from NO_REPLY flag    │
  │   dispatch to handler (blocks until HIP returns)│
  │   g_suppress_response = 0                       │
  │   free(payload)                                 │
  │ }                                               │
  └─────────────────────────────────────────────────┘
```

### 5.3 GPU Stream Ordering

All operations submitted to the same HIP stream execute in submission
order on the GPU. A kernel launched after a memcpy will always see the
copied data. This is a fundamental GPU programming guarantee.

### 5.4 Error Deferral

If a fire-and-forget operation fails on the GPU, the error surfaces at the
next synchronization point (`hipDeviceSynchronize`, `hipMemcpy D2H`, etc.)
where the client is waiting for a response. This matches the native HIP
behavior where asynchronous errors are reported at subsequent sync calls.

---

## 6. Performance Impact

### Before (synchronous)

```
  ┌─────┐  400ms  ┌─────┐  400ms  ┌─────┐  400ms  ┌─────┐
  │ op1 │────────►│ op2 │────────►│ op3 │────────►│ op4 │ ...
  └─────┘         └─────┘         └─────┘         └─────┘

  Time for 300 ops: ~120 seconds of network wait alone
```

### After (fire-and-forget)

```
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─── ───┬──────┐
  │ op1 │ op2 │ op3 │ op4 │ op5 │ ... │ op300 │ sync │
  └─────┴─────┴─────┴─────┴─────┴─────┴─── ───┴──────┘
  |◄──────── pipelined into TCP stream ────────►|      |
                                                 400ms
                                                 (one wait)

  Time for 300 ops: ~1 network round-trip + GPU compute time
```

### GPT-2 Result

| Metric | Value |
|--------|-------|
| Model | GPT-2 (124M params, FP16) |
| Client | Windows PC |
| Worker | Linux with 8× MI300X |
| Network | ~200ms one-way latency |
| Tokens generated | 5 |
| Total generation time | 19.6 seconds |
| Time per token | ~3.9 seconds |

The generation completed successfully, producing coherent text:

> *"The future of AI is in the hands of the"*

---

## 7. Implementation Files

| File | Changes |
|------|---------|
| `hip_remote_protocol.h` | Added `HIP_REMOTE_FLAG_NO_REPLY` flag definition |
| `hip_remote_client.h` | Declared `hip_remote_request_fire_and_forget()` and `_with_data` variant |
| `hip_client.c` | Implemented the two fire-and-forget request functions |
| `hip_api_module.c` | `hipLaunchKernel` / `hipModuleLaunchKernel` use fire-and-forget |
| `hip_api_memory.c` | `hipMemcpy H2D/D2D`, `hipMemset`, `hipFree` use fire-and-forget |
| `hip_api_stream.c` | `hipEventRecord` uses fire-and-forget |
| `hip_worker_main.c` | `g_suppress_response` mechanism; per-function COMGR cache; `kernelParams` reconstruction |

---

## 8. Related Fix: Worker Kernel Launch Path

During testing, a pre-existing bug was discovered where
`hipExtModuleLaunchKernel` with a flat `extra` buffer did not correctly
handle kernel arguments for regular (non-Tensile) kernels — only the first
element of vectorized kernels was being processed.

The fix caches COMGR metadata (parameter offset and size) per function
handle on the worker. During launch, if metadata is available, the worker
reconstructs a `kernelParams` pointer array from the flat buffer and calls
`hipModuleLaunchKernel` instead. `hipExtModuleLaunchKernel` with `extra` is
now only used as a fallback for kernels without COMGR metadata (e.g. Tensile
library kernels).
