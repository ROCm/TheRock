# HIP Remote Protocol Specification

This document describes the binary wire protocol used between the hip-remote
client and worker.

## Overview

The protocol is a request-response model over a single TCP connection. The
client sends requests, the worker processes them and sends responses. Requests
can optionally be marked fire-and-forget, in which case the worker does not
send a response.

All multi-byte integers are little-endian. All structs are packed (no padding)
using `__attribute__((packed))` on GCC/Clang and `#pragma pack(push, 1)` on
MSVC.

## Connection Lifecycle

```
Client                              Worker
  |                                    |
  |--- TCP connect (port 18515) ------>|
  |                                    |
  |--- HIP_OP_INIT ------------------->|
  |<-- HIP_OP_INIT response ----------|
  |                                    |
  |--- HIP API requests ------------->|
  |<-- responses (if not FnF) --------|
  |    ...                             |
  |                                    |
  |--- HIP_OP_SHUTDOWN -------------->|
  |<-- HIP_OP_SHUTDOWN response ------|
  |                                    |
  |--- TCP close --------------------->|
```

## Message Format

Every message (request or response) starts with a 24-byte header:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         magic (4B)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        version (2B)           |        op_code (2B)           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       request_id (4B)                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                    payload_length (8B)                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         flags (4B)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Field | Size | Description |
|-------|------|-------------|
| `magic` | 4 | `0x48495052` (`"HIPR"` in ASCII). Used to validate message framing. |
| `version` | 2 | Protocol version. Currently `0x0100` (v1.0). |
| `op_code` | 2 | Operation code identifying the HIP API call. |
| `request_id` | 4 | Client-assigned correlation ID. Responses echo this value. |
| `payload_length` | 8 | Number of bytes following the header. Can be 0. |
| `flags` | 4 | Bitmask of message flags (see below). |

The payload immediately follows the header. Its structure depends on the
`op_code`.

## Flags

| Flag | Value | Description |
|------|-------|-------------|
| `HIP_REMOTE_FLAG_RESPONSE` | `0x01` | Set in response messages. |
| `HIP_REMOTE_FLAG_ERROR` | `0x02` | Set when the operation failed. |
| `HIP_REMOTE_FLAG_HAS_INLINE_DATA` | `0x04` | Payload contains structured data followed by bulk inline data (e.g., H2D memcpy sends the request struct + the raw bytes). |
| `HIP_REMOTE_FLAG_NO_REPLY` | `0x08` | Fire-and-forget. Worker processes the request but does not send a response. Used for async GPU operations. |

## Response Format

All responses include a 4-byte `HipRemoteResponseHeader` at the start of the
payload:

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       error_code (4B)                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

`error_code` is the `hipError_t` value. 0 = `hipSuccess`.

Operation-specific response fields follow after the error code.

## Operation Codes

### Connection Management (0x00xx)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0001` | `INIT` | (none) | `HipRemoteResponseHeader` |
| `0x0002` | `SHUTDOWN` | (none) | `HipRemoteResponseHeader` |
| `0x0003` | `PING` | (none) | `HipRemoteResponseHeader` |

### Device Management (0x01xx)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0100` | `GET_DEVICE_COUNT` | (none) | `{error_code, count}` |
| `0x0101` | `SET_DEVICE` | `{device_id}` | `{error_code}` |
| `0x0102` | `GET_DEVICE` | (none) | `{error_code, device_id}` |
| `0x0103` | `GET_DEVICE_PROPERTIES` | `{device_id}` | `{error_code, name[256], total_global_mem, ...}` |
| `0x0104` | `DEVICE_SYNCHRONIZE` | (none) | `{error_code}` |
| `0x0106` | `DEVICE_GET_ATTRIBUTE` | `{device_id, attribute}` | `{error_code, value}` |
| `0x0107` | `DEVICE_GET_LIMIT` | `{limit, value}` | `{error_code, value}` |
| `0x0108` | `DEVICE_SET_LIMIT` | `{limit, value}` | (fire-and-forget) |

### Memory Allocation (0x02xx)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0200` | `MALLOC` | `{size, flags}` | `{error_code, device_ptr}` |
| `0x0201` | `FREE` | `{device_ptr}` | (fire-and-forget) |
| `0x0205` | `MALLOC_ASYNC` | `{size, stream}` | `{error_code, device_ptr}` |
| `0x0206` | `FREE_ASYNC` | `{device_ptr, stream}` | (fire-and-forget) |

### Memory Transfer (0x021x)

| Code | Name | Direction | Behaviour |
|------|------|-----------|-----------|
| `0x0210` | `MEMCPY` | H2D | Fire-and-forget. Inline data follows request struct. |
| `0x0210` | `MEMCPY` | D2H | Synchronous. Response includes inline data. |
| `0x0210` | `MEMCPY` | D2D | Fire-and-forget. |
| `0x0211` | `MEMCPY_ASYNC` | (same as MEMCPY per direction) | |
| `0x0212` | `MEMCPY_2D` | H2D/D2D | Fire-and-forget. |
| `0x0212` | `MEMCPY_2D` | D2H | Synchronous with inline data. |
| `0x021C` | `MEMCPY_PEER` | D2D | Fire-and-forget. |

For H2D copies, the request has `HIP_REMOTE_FLAG_HAS_INLINE_DATA` set. The
payload is `HipRemoteMemcpyRequest` followed by `size` bytes of source data.

For D2H copies, the response payload is `HipRemoteMemcpyResponse` followed by
`size` bytes of destination data.

### Memory Set (0x022x)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0220` | `MEMSET` | `{dst, value, size, stream}` | (fire-and-forget) |
| `0x0221` | `MEMSET_ASYNC` | `{dst, value, size, stream}` | (fire-and-forget) |

### Stream Operations (0x03xx)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0300` | `STREAM_CREATE` | `{flags, priority}` | `{error_code, stream}` |
| `0x0303` | `STREAM_DESTROY` | `{stream}` | (fire-and-forget) |
| `0x0304` | `STREAM_SYNCHRONIZE` | `{stream}` | `{error_code}` |
| `0x0306` | `STREAM_WAIT_EVENT` | `{stream, event, flags}` | (fire-and-forget) |

### Event Operations (0x04xx)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0400` | `EVENT_CREATE` | `{flags}` | `{error_code, event}` |
| `0x0402` | `EVENT_DESTROY` | `{event}` | (fire-and-forget) |
| `0x0403` | `EVENT_RECORD` | `{event, stream}` | (fire-and-forget) |
| `0x0404` | `EVENT_SYNCHRONIZE` | `{event}` | `{error_code}` |
| `0x0405` | `EVENT_QUERY` | `{event}` | `{error_code}` |
| `0x0406` | `EVENT_ELAPSED_TIME` | `{start_event, stop_event}` | `{error_code, milliseconds}` |

### Module Operations (0x05xx)

| Code | Name | Request Payload | Response Payload |
|------|------|-----------------|-----------------|
| `0x0500` | `MODULE_LOAD_DATA` | `{data_size}` + inline code object bytes | `{error_code, module}` |
| `0x0502` | `MODULE_UNLOAD` | `{module}` | (fire-and-forget) |
| `0x0503` | `MODULE_GET_FUNCTION` | `{module, function_name[256]}` | `{error_code, function, num_args, num_params, params[]}` |

The `MODULE_GET_FUNCTION` response includes kernel argument metadata extracted
via COMGR from the code object. Each `params[i]` contains `{offset, size}` for
the kernel argument at index `i`. The client uses this metadata to build the
flat kernarg buffer with correct argument placement.

### Kernel Launch (0x0510)

The kernel launch request is the most complex message in the protocol:

```
HipRemoteLaunchKernelRequest:
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       function (8B)                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  grid_dim_x (4B)  |  grid_dim_y (4B)  |  grid_dim_z (4B)    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| block_dim_x (4B)  | block_dim_y (4B)  | block_dim_z (4B)    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| shared_mem (4B)   |       stream (8B)                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  num_args (4B)    | launch_flags (4B) |   start_event (8B)   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   stop_event (8B)                     |  ext_flags (4B)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Followed by:
  HipRemoteKernelArg[num_args]:  {offset (4B), size (4B)} per arg
  uint8_t arg_data[]:  concatenated argument values
```

| Field | Description |
|-------|-------------|
| `function` | Remote function handle from `MODULE_GET_FUNCTION`. |
| `grid_dim_*` | Grid dimensions (number of blocks). |
| `block_dim_*` | Block dimensions (threads per block). |
| `shared_mem` | Dynamic shared memory in bytes. |
| `stream` | Stream handle (0 for default stream). |
| `num_args` | Number of argument descriptors following the header. |
| `launch_flags` | `0` = individual `kernelParams`, `1` = flat buffer via `extra`. |
| `start_event` | Event handle for profiling start (0 = none). |
| `stop_event` | Event handle for profiling stop (0 = none). |
| `ext_flags` | Flags for `hipExtModuleLaunchKernel` (0 = default). |

When `launch_flags = 1` (flat buffer mode), `num_args` is typically 1 with a
single descriptor covering the entire buffer. The worker passes this buffer to
`hipExtModuleLaunchKernel` via the `extra` parameter.

When `launch_flags = 0` (kernelParams mode), each `arg_data` segment is a
separate argument value. The worker reconstructs a `kernelParams[]` array and
calls `hipModuleLaunchKernel`.

Kernel launches are fire-and-forget unless `start_event` or `stop_event` is
non-zero (profiling mode requires a synchronous response).

### Combined Operations (0x074x)

| Code | Name | Description |
|------|------|-------------|
| `0x0744` | `MODULE_LOAD_AND_GET_FUNCTION` | Combines module load + get function in one round-trip. Payload: `{data_size}` + `{name_length, _pad}` + kernel name bytes + code object bytes. Response: `{error_code, module, function, num_args, num_params, params[]}`. |
| `0x0745` | `MALLOC_BATCH` | Allocate up to 64 buffers in one round-trip. Request: `{count, sizes[64]}`. Response: `{error_code, count, ptrs[64]}`. |
| `0x0746` | `STREAM_CREATE_BATCH` | Pre-allocate up to 32 stream handles. Request: `{count, flags}`. Response: `{error_code, count, handles[32]}`. |
| `0x0747` | `EVENT_CREATE_BATCH` | Pre-allocate up to 32 event handles. Request: `{count, flags}`. Response: `{error_code, count, handles[32]}`. |

### Version Info (0x07xx)

| Code | Name | Response Payload |
|------|------|-----------------|
| `0x0700` | `RUNTIME_GET_VERSION` | `{error_code, version}` |
| `0x0701` | `DRIVER_GET_VERSION` | `{error_code, version}` |

## Fire-and-Forget

Operations that are asynchronous on the GPU do not need a response from the
worker. The client sets `HIP_REMOTE_FLAG_NO_REPLY` in the header flags and
does not wait for a response.

Fire-and-forget operations:

- Kernel launches (without profiling events)
- `hipMemset` / `hipMemsetAsync`
- `hipMemcpy` H2D and D2D
- `hipFree` / `hipFreeAsync`
- `hipEventRecord`
- `hipEventDestroy` / `hipStreamDestroy`
- `hipStreamWaitEvent`
- `hipModuleUnload`
- `hipGraphLaunch` / `hipGraphDestroy`
- `hipDeviceSetLimit`, `hipDeviceEnablePeerAccess`

Operations that always require a response:

- `hipMalloc` (returns device pointer)
- `hipMemcpy` D2H (returns data)
- `hipDeviceSynchronize` (implicit error check)
- `hipEventSynchronize` / `hipEventElapsedTime` (timing)
- `hipModuleLoadData` / `hipModuleGetFunction` (returns handles)

## Write Coalescing

The client accumulates fire-and-forget requests in a 64 KB buffer instead of
sending each one immediately. The buffer is flushed when:

1. A synchronous (round-trip) request is needed.
2. The buffer is full.
3. An explicit flush is requested (`hip_remote_flush()`).

This turns hundreds of small `send()` calls into 1-2 bulk sends, reducing
syscall and TCP overhead.

## Scatter-Gather I/O

All send paths use `writev()` (POSIX) or `WSASend()` (Windows) to combine the
header, payload, and inline data into a single syscall, avoiding multiple TCP
segments from `TCP_NODELAY`.

## Error Handling

Errors from fire-and-forget operations are deferred. The worker stores the
error state, and the next synchronous operation (e.g.,
`hipDeviceSynchronize`) returns the accumulated error. This matches how async
GPU errors work natively in HIP.

## Handle Semantics

Remote handles (device pointers, stream handles, event handles, module
handles, function handles) are opaque 64-bit values. The client stores them
as-is and passes them back to the worker in subsequent requests. The worker
interprets them as native HIP pointers/handles.

Device pointers returned by `hipMalloc` on the worker are valid GPU addresses
on the remote device. The client never dereferences them -- they are passed
through to subsequent `hipMemcpy`, `hipLaunchKernel`, etc. calls.
