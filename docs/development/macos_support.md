# macOS Support via Remote HIP

This document describes TheRock's approach to supporting macOS as a development and build platform using a remote HIP execution model.

> [!NOTE]
> macOS support is experimental and focused on enabling development workflows where
> code is written and compiled on macOS but executed on remote Linux systems with
> AMD GPUs.

## Overview

Unlike Linux (which has native AMD GPU drivers) or Windows (which uses PAL), macOS
lacks kernel-level support for AMD discrete GPUs. Instead of attempting to port the
kernel driver stack, TheRock implements a **Remote HIP Runtime** that forwards HIP
API calls over the network to a Linux GPU server.

This approach:
- Enables macOS as a first-class development platform
- Provides real AMD GPU execution (not emulation)
- Integrates naturally with cloud GPU infrastructure
- Requires no closed-source dependencies

## Architecture

![Remote HIP Architecture](images/remote-hip-architecture.png)

<details>
<summary>Text diagram (for terminal viewing)</summary>

```
┌─────────────────────────────────────────┐     ┌──────────────────────────────────────┐
│            macOS Development Host       │     │         Linux GPU Server             │
│              (Apple Silicon)            │     │                                      │
│                                         │     │   ┌──────────────────────────────┐  │
│  ┌─────────────────────────────────┐   │     │   │      HIP Worker Service      │  │
│  │       HIP Application           │   │     │   │                              │  │
│  │    (PyTorch, JAX, custom)       │   │     │   │   ┌──────────────────────┐   │  │
│  └─────────────┬───────────────────┘   │     │   │   │   Real HIP Runtime   │   │  │
│                │                        │     │   │   │   (ROCR-Runtime)     │   │  │
│                │ HIP API calls          │     │   │   └──────────┬───────────┘   │  │
│                ▼                        │     │   │              │               │  │
│  ┌─────────────────────────────────┐   │     │   │   ┌──────────▼───────────┐   │  │
│  │      libamdhip64.dylib          │   │     │   │   │     AMD GPU(s)       │   │  │
│  │      (Remote HIP Client)        │◄──┼─TCP─┼──►│   │  (MI300X, etc.)      │   │  │
│  │                                 │   │     │   │   └──────────────────────┘   │  │
│  │  • Intercepts all HIP calls     │   │     │   │                              │  │
│  │  • Serializes to binary protocol│   │     │   └──────────────────────────────┘  │
│  │  • Manages remote memory handles│   │     │                                      │
│  │  • Transfers data over network  │   │     └──────────────────────────────────────┘
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │      amd-llvm Toolchain         │   │
│  │  • clang (host + device)        │   │
│  │  • hipcc compiler driver        │   │
│  │  • Device code → AMDGPU ISA     │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

</details>

### Key Components

1. **Remote HIP Client** (`libamdhip64.dylib`)
   - Drop-in replacement for the standard HIP runtime
   - Implements the full HIP C API
   - Forwards calls to remote worker over TCP
   - Manages handle translation (local ↔ remote)

2. **HIP Worker Service** (`hip-worker`)
   - Runs on Linux system with AMD GPU
   - Receives HIP API requests
   - Executes using real HIP runtime
   - Returns results to client

3. **Remote HIP Protocol**
   - Binary protocol for efficient serialization
   - Supports all HIP operations
   - Handles bulk data transfer (memory copies)
   - Connection multiplexing for async operations

4. **Compiler Toolchain** (native macOS)
   - Full amd-llvm built natively on macOS
   - hipcc generates AMDGPU device code
   - Device code objects (`.hsaco`) are portable

## Remote HIP Protocol

### Protocol Format

All messages use a common header followed by operation-specific payload:

```c
typedef struct __attribute__((packed)) {
    uint32_t magic;           // 0x48495052 ('HIPR')
    uint16_t version;         // Protocol version
    uint16_t op_code;         // Operation code
    uint32_t request_id;      // Correlation ID for async
    uint32_t payload_length;  // Bytes following header
    uint32_t flags;           // Reserved for future use
} HipRemoteHeader;
```

### Operation Categories

#### Device Management
| Op Code | Operation | Description |
|---------|-----------|-------------|
| 0x0001 | `HIP_OP_INIT` | Initialize connection |
| 0x0002 | `HIP_OP_SHUTDOWN` | Close connection |
| 0x0100 | `HIP_OP_GET_DEVICE_COUNT` | Get number of GPUs |
| 0x0101 | `HIP_OP_SET_DEVICE` | Set active device |
| 0x0102 | `HIP_OP_GET_DEVICE` | Get active device |
| 0x0103 | `HIP_OP_GET_DEVICE_PROPERTIES` | Get device properties |
| 0x0104 | `HIP_OP_DEVICE_SYNCHRONIZE` | Synchronize device |

#### Memory Operations
| Op Code | Operation | Description |
|---------|-----------|-------------|
| 0x0200 | `HIP_OP_MALLOC` | Allocate device memory |
| 0x0201 | `HIP_OP_FREE` | Free device memory |
| 0x0202 | `HIP_OP_MALLOC_HOST` | Allocate pinned host memory |
| 0x0203 | `HIP_OP_FREE_HOST` | Free pinned host memory |
| 0x0210 | `HIP_OP_MEMCPY` | Synchronous memory copy |
| 0x0211 | `HIP_OP_MEMCPY_ASYNC` | Asynchronous memory copy |
| 0x0220 | `HIP_OP_MEMSET` | Synchronous memset |
| 0x0221 | `HIP_OP_MEMSET_ASYNC` | Asynchronous memset |

#### Stream Operations
| Op Code | Operation | Description |
|---------|-----------|-------------|
| 0x0300 | `HIP_OP_STREAM_CREATE` | Create stream |
| 0x0301 | `HIP_OP_STREAM_DESTROY` | Destroy stream |
| 0x0302 | `HIP_OP_STREAM_SYNCHRONIZE` | Synchronize stream |
| 0x0303 | `HIP_OP_STREAM_QUERY` | Query stream status |

#### Event Operations
| Op Code | Operation | Description |
|---------|-----------|-------------|
| 0x0400 | `HIP_OP_EVENT_CREATE` | Create event |
| 0x0401 | `HIP_OP_EVENT_DESTROY` | Destroy event |
| 0x0402 | `HIP_OP_EVENT_RECORD` | Record event |
| 0x0403 | `HIP_OP_EVENT_SYNCHRONIZE` | Wait for event |
| 0x0404 | `HIP_OP_EVENT_ELAPSED_TIME` | Get elapsed time |

#### Module/Kernel Operations
| Op Code | Operation | Description |
|---------|-----------|-------------|
| 0x0500 | `HIP_OP_MODULE_LOAD_DATA` | Load module from memory |
| 0x0501 | `HIP_OP_MODULE_UNLOAD` | Unload module |
| 0x0502 | `HIP_OP_MODULE_GET_FUNCTION` | Get kernel function |
| 0x0510 | `HIP_OP_LAUNCH_KERNEL` | Launch kernel |

#### Error Handling
| Op Code | Operation | Description |
|---------|-----------|-------------|
| 0x0600 | `HIP_OP_GET_LAST_ERROR` | Get and clear last error |
| 0x0601 | `HIP_OP_PEEK_AT_LAST_ERROR` | Get last error without clearing |

### Handle Translation

The client maintains mappings between local handles and remote handles:

```c
// Client-side handle maps
typedef struct {
    void* local_ptr;      // Address returned to application
    uint64_t remote_ptr;  // Actual device pointer on server
    size_t size;          // Allocation size
} MemoryHandle;

typedef struct {
    hipStream_t local;    // Local stream handle
    uint64_t remote;      // Remote stream ID
} StreamHandle;
```

### Data Transfer

For memory copies involving host data:

```
Client                                    Server
  │                                         │
  │  HIP_OP_MEMCPY (H2D)                   │
  │  ┌─────────────────────────────────┐   │
  │  │ Header                          │   │
  │  │ MemcpyRequest (dst, size, kind) │   │
  │  │ [Inline data payload]           │──►│
  │  └─────────────────────────────────┘   │
  │                                         │
  │  Response                               │
  │  ┌─────────────────────────────────┐   │
  │◄─│ Header                          │   │
  │  │ MemcpyResponse (error_code)     │   │
  │  └─────────────────────────────────┘   │
  │                                         │
  │  HIP_OP_MEMCPY (D2H)                   │
  │  ┌─────────────────────────────────┐   │
  │  │ Header                          │   │
  │  │ MemcpyRequest (src, size, kind) │──►│
  │  └─────────────────────────────────┘   │
  │                                         │
  │  Response                               │
  │  ┌─────────────────────────────────┐   │
  │◄─│ Header                          │   │
  │  │ MemcpyResponse (error_code)     │   │
  │  │ [Inline data payload]           │   │
  │  └─────────────────────────────────┘   │
```

## Build System Integration

### Platform Detection

```cmake
# cmake/therock_globals.cmake
set(THEROCK_CONDITION_IS_MACOS OFF)
if(APPLE)
  set(THEROCK_CONDITION_IS_MACOS ON)
endif()
```

### Compiler Requirements

| Platform | Compiler | Notes |
|----------|----------|-------|
| Linux | GCC or Clang | Default |
| Windows | MSVC | Required |
| macOS | AppleClang | Xcode 15+ recommended |

### Target Architecture

- **macOS**: Apple Silicon (ARM64) only
- **Minimum version**: macOS 13.0 (Ventura)
- **GPU targets**: Standard AMDGPU targets (gfx90a, gfx942, gfx1100, etc.)

### Component Availability

| Component | Linux | Windows | macOS |
|-----------|-------|---------|-------|
| amd-llvm | ✅ | ✅ | ✅ |
| hipcc | ✅ | ✅ | ✅ |
| hipify | ✅ | ✅ | ✅ |
| amd-comgr | ✅ | ✅ | ✅ |
| hip-clr (local) | ✅ | 🟡 PAL | ❌ |
| hip-remote-client | N/A | N/A | ✅ |
| hip-worker | ✅ | ❌ | ❌ |
| ROCR-Runtime | ✅ | ❌ | ❌ |
| rocminfo | ✅ | ❌ | ❌ |
| Math libraries | ✅ | ✅ | ✅* |
| ML libraries | ✅ | 🟡 | ✅* |
| Profilers | ✅ | ❌ | ❌ |
| Debug tools | ✅ | ❌ | ❌ |

\* Execution via remote HIP

## Usage

### Building on macOS

```bash
# Prerequisites
brew install cmake ninja ccache python@3.11
pip3 install -r requirements.txt

# Configure with remote HIP
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release

# Build compiler toolchain
cmake --build build --target amd-llvm hipcc

# Build remote HIP client
cmake --build build --target hip-remote-client

# Build math libraries (optional)
cmake --build build --target rocBLAS rocFFT
```

### Running with Remote GPU

1. Start the worker on a Linux GPU server:
   ```bash
   # On Linux GPU server
   export HIP_VISIBLE_DEVICES=0
   ./hip-worker --port 8000
   ```

2. Configure the client on macOS:
   ```bash
   # On macOS
   export TF_WORKER_HOST=gpu-server.example.com
   export TF_WORKER_PORT=8000
   export TF_DEBUG=1  # Optional: enable debug logging
   ```

3. Run your application:
   ```bash
   ./my_hip_application
   ```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_WORKER_HOST` | `localhost` | Remote worker hostname |
| `TF_WORKER_PORT` | `8000` | Remote worker port |
| `TF_DEBUG` | `0` | Enable debug logging (0/1) |
| `TF_CONNECT_TIMEOUT` | `30` | Connection timeout (seconds) |
| `TF_IO_TIMEOUT` | `60` | I/O timeout (seconds) |

## Implementation Details

### Source Organization

```
core/
├── hip-remote-client/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── hip_remote/
│   │       ├── hip_remote_protocol.h    # Protocol definitions
│   │       └── hip_remote_client.h      # Internal client API
│   └── src/
│       ├── hip_client.c                 # Connection management
│       ├── hip_api_device.c             # Device APIs
│       ├── hip_api_memory.c             # Memory APIs
│       ├── hip_api_stream.c             # Stream/Event APIs
│       ├── hip_api_module.c             # Module/Kernel APIs
│       └── hip_api_error.c              # Error handling
│
├── hip-remote-worker/
│   ├── CMakeLists.txt
│   └── src/
│       ├── hip_worker_main.c            # Server main loop
│       └── hip_worker_handlers.c        # Operation handlers
```

### Thread Safety

- Client library is thread-safe
- Each thread can use its own stream
- Connection is protected by mutex
- Request IDs enable async correlation

### Error Handling

- Network errors map to `hipErrorNotInitialized`
- Protocol errors map to `hipErrorInvalidValue`
- Remote HIP errors pass through directly
- Connection auto-reconnect on failure

## Performance Considerations

### Latency

Remote execution adds network latency:
- LAN: ~0.1-1ms per operation
- WAN: 10-100ms+ per operation

**Mitigation strategies:**
- Batch operations where possible
- Use async APIs with streams
- Minimize host-device transfers
- Larger kernel granularity

### Throughput

Memory transfer bandwidth limited by network:
- 10GbE: ~1GB/s
- 100GbE: ~10GB/s
- InfiniBand: 25-100GB/s

### Best Practices

1. **Minimize transfers**: Keep data on device
2. **Use async APIs**: Overlap compute and transfer
3. **Batch operations**: Reduce round-trips
4. **Profile locally first**: Debug on Linux, then use macOS for development

## Testing

### Unit Tests

```bash
# Build and run client tests (no GPU needed)
cmake --build build --target hip-remote-client-tests
ctest --test-dir build/core/hip-remote-client
```

### Integration Tests

Requires access to a Linux GPU server:

```bash
# Set up worker connection
export TF_WORKER_HOST=gpu-server
export TF_WORKER_PORT=8000

# Run integration tests
cmake --build build --target hip-remote-integration-tests
ctest --test-dir build/core/hip-remote-client --label-regex integration
```

## Limitations

1. **No local GPU execution**: All kernels run on remote server
2. **Network dependency**: Requires stable connection to GPU server
3. **Latency sensitive**: Not suitable for latency-critical real-time applications
4. **No profiling**: rocprof and other profilers require local GPU
5. **No debugging**: GPU debugging requires local device access

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PAL (Windows model) | AMD-maintained | Closed-source, no macOS libs | Rejected |
| Full HSA port | Native performance | Requires kernel driver | Rejected |
| Metal backend | Apple-native GPU | Incompatible architecture | Rejected |
| CPU emulation | Simple | Slow, not practical | Rejected |
| Remote HIP | Open, real GPUs | Network latency | **Chosen** |

## Future Work

1. **Connection pooling**: Multiple connections for parallel streams
2. **Compression**: Compress large memory transfers
3. **Caching**: Cache code objects on worker
4. **Multi-GPU**: Support for multi-GPU configurations
5. **Kubernetes integration**: Native tensor-fusion integration

## References

- [Tensor Fusion GPU-over-IP](https://github.com/NexusGPU/tensor-fusion)
- [HIP API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [TheRock Build System](./build_system.md)
- [Windows Support](./windows_support.md)
