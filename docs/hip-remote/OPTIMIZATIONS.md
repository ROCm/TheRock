# hip-remote Performance Optimizations

## Overview

hip-remote enables a Windows or macOS client to execute GPU workloads on a
remote Linux machine equipped with AMD MI300X GPUs. HIP API calls are
intercepted on the client, serialized over TCP, and dispatched to a worker
process on the GPU server.

This document describes the two key optimization strategies that minimize the
impact of network latency:

1. **Fire-and-forget pipelining** -- eliminates synchronous round-trips for
   asynchronous GPU operations.
2. **Virtual address allocation** -- eliminates round-trips for memory
   allocation by assigning opaque handles on the client side.

---

## 1. The Problem: Synchronous Round-Trips

In the original protocol every HIP API call required a full TCP round-trip:

```
 Client (Windows)                         Worker (Linux + MI300X)
 ────────────────                         ──────────────────────

  hipMalloc      ──── request ──────────►  allocate GPU memory
                 ◄─── response (RTT) ────  device_ptr
  hipMemcpy H2D  ──── request ──────────►  copy data to GPU
                 ◄─── response (RTT) ────  hipSuccess
  hipLaunchKernel ─── request ──────────►  dispatch to GPU
                 ◄─── response (RTT) ────  hipSuccess
```

With ~130 ms round-trip time, a model with 700 parameters requires:

```
  hipMalloc round-trips:  700 × 130ms = 91 seconds
  Kernel launch trips:    300 × 130ms = 39 seconds
  Total network wait:     130+ seconds (before any GPU time)
```

---

## 2. Fire-and-Forget Pipelining

### Concept

Most GPU operations are asynchronous -- the HIP runtime returns immediately
and the GPU executes later. The only reason we waited for a response was to
check the error code, which is almost always `hipSuccess`.

Fire-and-forget sends the request and immediately moves on:

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

### Protocol

A single bit in the request header:

```c
#define HIP_REMOTE_FLAG_NO_REPLY  (1u << 3)
```

When set, the worker processes the request but skips sending a response.
The `send_response()` function checks `g_suppress_response` and becomes a
no-op, requiring zero changes to existing handler functions.

### Write Coalescing

Fire-and-forget requests are accumulated in a client-side write buffer
(64 KB) and flushed in bulk at the next synchronous call. This batches
many small TCP writes into a single large `send()`, improving throughput.

### Which Operations Are Fire-and-Forget?

| Operation                | FnF | Reason                           |
|--------------------------|-----|----------------------------------|
| hipLaunchKernel          | Yes | Async GPU dispatch, no return    |
| hipModuleLaunchKernel    | Yes | Same -- flat kernarg path        |
| hipMemcpy H2D            | Yes | Data flows client to worker only |
| hipMemcpy D2D            | Yes | No data crosses the network      |
| hipMemset / Async        | Yes | No return value needed           |
| hipFree / FreeAsync      | Yes | No return value needed           |
| hipEventRecord           | Yes | No return value needed           |
| hipStreamDestroy         | Yes | No return value needed           |
| hipStreamWaitEvent       | Yes | No return value needed           |
| hipMemcpy D2H            | No  | Data must come back to client    |
| hipDeviceSynchronize     | No  | Synchronization barrier          |
| hipStreamSynchronize     | No  | Synchronization barrier          |
| hipMalloc                | No* | Returns device pointer           |
| hipModuleLoadData        | No  | Returns module handle            |

\* hipMalloc is no longer synchronous -- see section 3.

---

## 3. Virtual Address Allocation

### Concept

`hipMalloc` was the last major source of synchronous round-trips. It must
return a device pointer to the caller, which previously required waiting
for the worker's response.

The key insight: **the client never dereferences GPU pointers**. They are
opaque 64-bit handles passed between HIP API calls. The client can assign
a virtual handle locally and send the allocation as fire-and-forget. The
worker performs the real allocation and stores a mapping from virtual
handle to real GPU pointer.

This is the same technique X11 uses for resource creation: the server
assigns an ID range, the client picks IDs locally, no round-trip needed.

```
 BEFORE: Each hipMalloc is a sync round-trip

  Client                              Worker
  ──────                              ──────
  hipMalloc(1024) ─── request ──────► real hipMalloc → ptr=0x7e01000
                  ◄── response ────── ptr=0x7e01000
  hipMemcpy H2D   ─── FnF ──────────► copy data

 AFTER: hipMalloc becomes fire-and-forget

  Client                              Worker
  ──────                              ──────
  hipMalloc(1024) → vaddr=V1 (local)
    ─── FnF: alloc V1, 1024 ────────► real hipMalloc → map V1→0x7e01000
  hipMemcpy H2D(V1) ─── FnF ───────► translate V1→0x7e01000, copy
  hipMalloc(2048) → vaddr=V2 (local)
    ─── FnF: alloc V2, 2048 ────────► real hipMalloc → map V2→0x7e02000
  hipMemcpy H2D(V2) ─── FnF ───────► translate V2→0x7e02000, copy
```

All hipMalloc calls become fire-and-forget. The client assigns virtual
addresses from a monotonically increasing counter starting at
`0x7F0000000000` (chosen to avoid collisions with real host pointers).

### Worker-Side Translation

The worker maintains two data structures:

1. **Hash map** (1M slots, open addressing) -- O(1) exact vaddr lookup.
   Used for base allocations and stream handles.

2. **Sorted alloc list** (up to 64K entries) -- O(log N) binary search
   for range lookups. Used when PyTorch's caching allocator sub-allocates
   from a large block (e.g., a tensor at `vaddr_base + offset`).

3. **Last-hit cache** -- a single-entry cache that remembers the most
   recent allocation block. Since consecutive kernel launches typically
   operate on the same tensors, the cache hits on the vast majority of
   lookups, making translation effectively O(1).

Every handler that receives a pointer or stream handle calls
`vaddr_translate()` before passing it to the real HIP API. Kernel
argument translation uses a tiered approach based on available metadata:

1. **COMGR-guided** (most kernels): The worker extracts `value_kind`
   from COMGR metadata for each parameter. `global_buffer` params are
   translated via range lookup. `by_value` params with size == 8 are
   also translated if their value >= VADDR_BASE (Tensile marks pointer
   params as `by_value`; real scalars never reach this range).
   `by_value` struct params (size > 8) have all 8-byte sub-fields
   scanned. Params smaller than 8 bytes are skipped.

2. **Blind scan** (Tensile/rocBLAS assembly kernels): When no COMGR
   metadata is available (assembly kernels lack standard metadata), all
   8-byte-aligned positions in the flat kernarg buffer are scanned.
   Tensile kernel scalars (matrix dimensions, strides, alpha/beta) are
   small values that never collide with the vaddr range.

### Virtual Stream Handles

The same approach applies to `hipStreamCreate`. The client assigns a
virtual stream handle from a separate counter (`0x5F0000000000`), sends
the creation request as fire-and-forget, and uses the virtual handle in
all subsequent stream operations. The worker creates the real stream and
stores the mapping.

### D2H Correctness

When the client needs GPU data (D2H memcpy, `.item()`, `.cpu()`), the
operation remains synchronous. This works correctly because:

1. Every sync call **flushes the write buffer** first, sending all
   accumulated FnF requests.
2. TCP preserves ordering, so the worker processes the FnF
   malloc/memcpy/launch operations **before** it receives the D2H request.
3. By the time the worker handles the D2H, the vaddr is mapped and the
   kernel results are ready.

### Deferred Error Handling

If `hipMalloc` fails on the worker (e.g., out of memory), the error is
stored in the vaddr map. The next synchronization point
(`hipDeviceSynchronize`, D2H memcpy) returns the deferred error to the
client. This matches native CUDA/HIP async error semantics.

---

## 4. Safety Guarantees

Three properties guarantee correctness for both optimizations:

### TCP Preserves Ordering

TCP is a byte stream -- requests arrive at the worker in the exact order
the client sent them. There is no reordering.

### Worker Is Single-Threaded

The worker reads one request at a time, fully processes it (including the
HIP API call), then reads the next. A fire-and-forget `hipMemcpy H2D`
completes on the worker before the subsequent `hipLaunchKernel` is read.

### GPU Stream Ordering

All operations submitted to the same HIP stream execute in submission
order on the GPU. A kernel launched after a memcpy will always see the
copied data.

---

## 5. Performance Results

### Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Client    | Windows 11, AMD Ryzen |
| Worker    | Linux, 8x AMD Instinct MI300X |
| Network   | ~130ms round-trip time |
| PyTorch   | 2.12.0a0+devrocm7.12.0.dev0 |

### Results

| Test | Without Optimizations | With FnF Only | With FnF + Vaddr | Speedup |
|------|-----------------------|---------------|-------------------|---------|
| **GPT-2 total** | ~300s+ | 167s | **19s** | **16x** |
| **GPT-2 per token** | N/A | 6.64s | **3.72s** | **1.8x** |
| **SDXL total** | N/A | 594s | **281s** | **2.1x** |
| **SDXL generation (30 steps)** | N/A | 60s | **42.5s** | **1.4x** |
| **Vroom per-op (SDPA/Conv/GEMM)** | ~400ms | 135ms | **131ms** | **3.0x** |

The virtual address optimization primarily accelerates model loading
(`model.to("cuda")`), where hundreds of `hipMalloc` calls were previously
serialized round-trips. With vaddr allocation, the entire model transfer
becomes a stream of fire-and-forget operations bounded only by network
bandwidth.

---

## 6. Implementation Files

| File | Changes |
|------|---------|
| `hip_remote_protocol.h` | `HIP_REMOTE_FLAG_NO_REPLY`, `HIP_OP_MALLOC_VADDR` / `HIP_OP_MALLOC_ASYNC_VADDR` opcodes, `HipRemoteMallocVaddrRequest` struct, `vhandle` field in `HipRemoteStreamCreateRequest` |
| `hip_api_memory.c` | Virtual address allocator (`vaddr_alloc`), hipMalloc/MallocManaged/MallocAsync converted to FnF with vaddr, hipFreeHost made local-only |
| `hip_api_stream.c` | Virtual stream handle allocator, hipStreamCreate FnF, capture state tracking (`g_capture_depth`), `hipStreamGetCaptureInfo` forwarding |
| `hip_api_module.c` | hipLaunchKernel/hipModuleLaunchKernel fire-and-forget |
| `hip_api_device.c` | CPU pointer validation, raw `hipDeviceProp_t` transfer (all fields populated) |
| `hip_client.c` | Write coalescing buffer, `hip_remote_request_fire_and_forget()` and `_with_data` variant |
| `hip_worker_main.c` | vaddr hash map + sorted alloc list + last-hit cache, `vaddr_translate()`, COMGR `is_pointer` extraction, cache invalidation on module unload, blind scan for assembly kernels, GPU error guard, deferred error reporting, dynamically growable caches, raw `hipDeviceProp_t` response, TCP keepalive |

---

## 7. Kernel Argument Translation Details

### COMGR-Guided Translation

For most HIP C++ kernels (PyTorch, Triton JIT), the worker extracts
COMGR metadata from the code object ELF. Each kernel parameter has a
`value_kind` field:

- **`global_buffer`** (`is_pointer=1`): An 8-byte GPU pointer. Translated
  via range lookup to handle sub-array pointers (e.g., `tensor[100:]`).
- **`by_value`** with size == 8: Translated if value >= VADDR_BASE.
  Tensile/rocBLAS marks pointer params as `by_value` in COMGR metadata;
  real scalars (strides, counts, alpha/beta) never reach the vaddr range.
- **`by_value`** with size > 8: A struct passed by value. All 8-byte
  sub-fields are scanned for vaddrs, since PyTorch packs multiple tensor
  pointers into structs (e.g., `TrivialOffsetCalculator`).
- **size < 8**: Skipped (definitely a scalar -- too small for a pointer).

The `is_pointer` flag is stored in `HipRemoteParamDesc` and flows from
the worker's COMGR extraction through the protocol to the client's
function info cache and back to the worker during kernel launch.

### Blind Scan Fallback

Tensile/rocBLAS assembly kernels lack standard COMGR metadata (their code
objects don't have `.amdhsa_kernel` entries with `.args` sections). For
these kernels, the worker falls back to scanning all 8-byte-aligned
positions in the flat kernarg buffer. This is safe because Tensile kernel
scalars (matrix dimensions, strides, alpha/beta) are small values that
never reach the vaddr range (≥ 0x7F0000000000).

### Cache Invalidation

The HIP runtime can reuse module handle addresses after `hipModuleUnload`.
Without cache invalidation, a new module loaded at the same address would
get stale COMGR metadata from the previous module -- potentially with
wrong `is_pointer` flags and parameter sizes. The worker now invalidates
`CachedKernelArgs` and `LoadedModuleEntry` on every module unload.

### GPU Error Guard

A `hipDeviceSynchronize` call before every `hipModuleLoadData` drains
pending async GPU errors. Errors are logged and cleared (via
`hipGetLastError`) so module loads can proceed -- MIOpen's solver search
triggers many module loads and recovers from transient errors.

### CUDA Graph Capture

During `hipStreamBeginCapture` / `hipStreamEndCapture`, the client
tracks capture depth. While capturing, `hipMalloc` and `hipMallocAsync`
fall back to synchronous allocation instead of FnF MALLOC_VADDR, since
device-level allocations outside the capture context would invalidate
the graph.

### Device Properties

The worker sends the raw `hipDeviceProp_t` struct (all 80+ fields) via
`memcpy`. Both sides include the same `hip_runtime_api.h`, so the struct
layout is identical. This ensures every field is populated, including
`regsPerMultiprocessor` (required by `torch.compile` / Inductor).

## 8. torch.compile Support

`torch.compile` works over hip-remote. Inductor fuses operations into
Triton-JIT kernels that are compiled on the client and loaded on the
worker. Significant per-token speedup after a one-time warmup:

| Test | Eager | Compiled | Speedup |
|------|-------|----------|---------|
| **GPT-2 per token** | 3.74s | **1.35s** | **2.8x** |
| **SDXL generation** | 45.7s | **89.9s*** | -- |

\*SDXL compiled time includes Triton JIT warmup during the first few
inference steps. Subsequent runs with Triton cache are faster.

## 9. Known Limitations

### Device-Side Assertions

Triton's `debug=True` mode enables device-side assertions that require
the GPU to write assertion failure info to a host-pinned buffer. This
mechanism requires shared host-device memory that cannot work over a TCP
connection. Tests using `@triton.jit(debug=True)` will fail with
`hipErrorLaunchFailure`.

### CPU Pointer Detection

The client validates pointer arguments by checking if the address falls
in the vaddr range (≥ `VADDR_BASE`). CPU tensor pointers are outside this
range and correctly return `hipErrorInvalidValue` from
`hipPointerGetAttribute`, allowing callers like Triton to detect and
reject them.
