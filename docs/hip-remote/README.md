# HIP Remote Execution

HIP Remote enables running GPU workloads on a remote Linux machine with AMD
GPUs from a Windows (or macOS) client. A lightweight proxy DLL intercepts HIP
API calls on the client and forwards them over TCP to a worker process on the
GPU server.

## Architecture

```
Windows/macOS Client                  Linux GPU Server
+------------------+                  +-------------------+
|  PyTorch / App   |                  |   hip-worker      |
|        |         |     TCP/IP       |        |          |
| amdhip64_7.dll   | ===============>|  libamdhip64.so   |
| (proxy DLL)      |   port 18515    |  (real HIP)       |
+------------------+                  +-------------------+
```

The proxy DLL (`amdhip64_7.dll`) exports the same symbols as the real HIP
runtime. When `TF_WORKER_HOST` is set, all HIP calls are serialised into a
binary protocol and sent to the worker. Fire-and-forget and write-coalescing
optimisations keep the overhead low for asynchronous operations.

## Prerequisites

| Component  | Requirement                                            |
| ---------- | ------------------------------------------------------ |
| **Client** | Windows with Python 3.10-3.13, VS 2022 Build Tools    |
| **Server** | Linux with ROCm 7.2+, AMD GPU (MI300X, MI250, etc.)   |
| **Network**| TCP connectivity on port 18515 between client & server |

## Building

### Worker (Linux)

```bash
cd rocm-systems/projects/hip-remote-worker
mkdir -p build && cd build
cmake -GNinja ..
ninja
```

The worker binary is `build/hip-worker`.

### Client proxy DLL (Windows)

**Important:** All commands must run from a **VS x64 Developer Command Prompt**
(or after sourcing `vcvars64.bat`) so MSVC is on the PATH.

```cmd
cd rocm-systems\projects\hip-remote-client

:: Configure with Ninja, proxy mode, and ROCm SDK headers
:: ROCM_PATH enables the real hipDeviceProp_t struct from the SDK.
:: Get it via: rocm-sdk path --root
cmake -B build -G Ninja -DHIP_REMOTE_PROXY_MODE=ON ^
    -DROCM_PATH=path\to\venv\Lib\site-packages\_rocm_sdk_devel .

:: Build the proxy DLL
ninja -C build amdhip64

:: Build the hipRTC proxy DLL (separate step)
cd build
cl /LD /Fe:hiprtc0702.dll /I..\include ..\src\hiprtc_proxy.c
```

If `ROCM_PATH` is omitted, the build uses a fallback struct that covers the
most common fields. Setting it is recommended for full Triton compatibility.

After this you will have two DLLs in the `build/` directory:

- `amdhip64_7.dll` -- the HIP proxy
- `hiprtc0702.dll` -- the hipRTC proxy

### Client library (macOS)

```bash
cd rocm-systems/projects/hip-remote-client
cmake -B build
cmake --build build
```

The shared library is `build/libamdhip64.1.0.0.dylib`.

### TheRock SDK + PyTorch (Windows)

All commands below run in a **VS 2022 x64 Native Tools Command Prompt**
(PowerShell).

#### 1. Build TheRock

```powershell
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942
cmake --build build
```

#### 2. Package and install TheRock wheels

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

python ./build_tools/build_python_packages.py `
    --artifact-dir=build/artifacts `
    --dest-dir=build/packages

pip install $(ls build/packages/dist/* | % {$_.FullName})
```

#### 3. Clone PyTorch

```powershell
python external-builds/pytorch/pytorch_torch_repo.py checkout `
    --checkout-dir D:/b_temp `
    --repo-hashtag jam/fix-int4mm-memcpy-win32-aotriton `
    --gitrepo-origin https://github.com/jammm/pytorch
```

#### 4. Build PyTorch wheels

```powershell
python external-builds/pytorch/build_prod_wheels.py build `
    --output-dir build/packages/dist `
    --pytorch-dir D:\b_temp `
    --build-pytorch-vision `
    --pytorch-vision-dir .\external-builds/pytorch/pytorch_vision `
    --enable-pytorch-flash-attention-windows `
    --pytorch-rocm-arch "gfx942"
```

The built wheels are installed into the active venv automatically.

## Running

### 1. Start the worker on the GPU server

```bash
cd rocm-systems/projects/hip-remote-worker/build

# Optional: restrict to a specific GPU
export HIP_VISIBLE_DEVICES=1

# Start with debug logging
TF_DEBUG=1 nohup ./hip-worker > /tmp/hip-worker.log 2>&1 &

# Verify
tail -3 /tmp/hip-worker.log
# Should show: [HIP-Worker] Listening on port 18515
```

The worker supports multiple concurrent clients via fork-per-client. Each
connection gets its own process with an independent HIP context.

### 2. Run PyTorch with hip-remote

Set the environment variables and run your script:

```powershell
$env:TF_WORKER_HOST = "<gpu-server-ip>"
$env:TF_DEBUG = "1"
$env:HIP_REMOTE_LIB_DIR = "path\to\hip-remote-client\build"
$env:MIOPEN_SYSTEM_DB_PATH = (rocm-sdk path --bin)

python your_script.py
```

| Variable               | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `TF_WORKER_HOST`       | IP address or hostname of the GPU server        |
| `TF_WORKER_PORT`       | Worker port (default: 18515)                    |
| `TF_DEBUG`             | Set to `1` for debug logging                    |
| `TRITON_LIBHIP_PATH`   | For Triton: set to the same `amdhip64_7.dll` path as `HIP_REMOTE_LIB_DIR` |
| `HIP_REMOTE_LIB_DIR`  | Directory containing `amdhip64_7.dll` proxy     |
| `MIOPEN_SYSTEM_DB_PATH`| Path to MIOpen TunaNet models. Use `rocm-sdk path --bin`. |

### 3. Verify

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # AMD Instinct MI300X
```

## Performance

The proxy uses several optimisations to minimise network overhead:

- **Fire-and-forget**: Async GPU operations (kernel launches, memset, D2D
  copies, event record, free) are sent without waiting for a response.
- **Write coalescing**: A 64 KB buffer accumulates fire-and-forget requests
  and flushes them in bulk at sync points.
- **Scatter-gather I/O**: `writev`/`WSASend` combines header, payload, and
  data into a single syscall.
- **Combined opcodes**: Module load + get function in one round-trip.
- **Client-side caching**: Device properties, driver version, occupancy
  queries, and event handles are cached locally.

## Troubleshooting

**Connection refused**: Ensure the worker is running and the firewall allows
port 18515.

**MIOpen solver search is slow on first run**: MIOpen benchmarks multiple
solver variants for each unique convolution shape. This is a one-time cost;
results are cached in `~/.config/miopen/` on the server. Set
`MIOPEN_SYSTEM_DB_PATH` to the `_rocm_sdk_devel/bin` directory so MIOpen can
use TunaNet AI heuristics to skip exhaustive benchmarking. Use
`rocm-sdk path --bin` to get the correct path.

**Worker log is empty**: Ensure the worker is started with `TF_DEBUG=1` and
stderr is captured (`> /tmp/hip-worker.log 2>&1`).

**`hipErrorNotInitialized`**: The client failed to connect to the worker.
Check `TF_WORKER_HOST` and network connectivity.

**`WinError 127 - procedure not found`**: The proxy DLL is missing a symbol
that an SDK library needs. Ensure `hip_api_stubs_gen.c` is compiled into the
DLL. This file provides stub implementations for ~490 HIP APIs that the proxy
does not implement but that SDK libraries (rocblas, hipblas, etc.) import.

## Protocol

The client and worker communicate using a binary protocol over TCP. Each
message has a fixed-size header followed by a variable-length payload:

```
+--------+--------+--------+--------+--------+
| magic  | version| opcode | req_id | flags  |
| (4B)   | (2B)   | (2B)   | (4B)   | (4B)   |
+--------+--------+--------+--------+--------+
| payload_length (8B)      | payload (var)    |
+--------+--------+--------+------------------+
```

Fire-and-forget requests set `HIP_REMOTE_FLAG_NO_REPLY` in the flags field.
The worker processes the request but does not send a response.

See [PROTOCOL.md](PROTOCOL.md) for the full protocol specification.
