# HIP Remote Execution

HIP Remote enables running GPU workloads on a remote Linux machine with AMD
GPUs from a Windows (or other) client. A lightweight proxy DLL intercepts HIP
API calls on the client and forwards them over TCP to a worker process on the
GPU server.

## Architecture

```
Windows Client                         Linux GPU Server
+------------------+                   +-------------------+
|  PyTorch / App   |                   |   hip-worker      |
|        |         |     TCP/IP        |        |          |
| amdhip64_7.dll   | ===============> |  libamdhip64.so   |
| (proxy DLL)      |   port 18515     |  (real HIP)       |
+------------------+                   +-------------------+
```

The proxy DLL (`amdhip64_7.dll`) exports the same symbols as the real HIP
runtime. When `TF_WORKER_HOST` is set, all HIP calls are serialised into a
binary protocol and sent to the worker. Fire-and-forget and write-coalescing
optimisations keep the overhead low for asynchronous operations.

## Prerequisites

| Component | Requirement |
|-----------|------------|
| **Client** | Windows with Python 3.10-3.13, VS 2022 Build Tools |
| **Server** | Linux with ROCm 7.2+, AMD GPU (MI300X, MI250, etc.) |
| **Network** | TCP connectivity on port 18515 between client and server |

## Building

### Worker (Linux)

```bash
cd rocm-systems/projects/hip-remote-worker
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The worker binary is `build/hip-worker`.

### Client proxy DLL (Windows)

Open a **VS x64 Developer Command Prompt** (or run `vcvars64.bat`), then:

```cmd
cd rocm-systems\projects\hip-remote-client
cmake -B build
cmake --build build --target amdhip64
```

The proxy DLL is `build\amdhip64_7.dll`.

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
# Use a specific GPU (optional)
export HIP_VISIBLE_DEVICES=1

# Enable debug logging
export TF_DEBUG=1

# Start the worker
./build/hip-worker > /tmp/hip-worker.log 2>&1 &

# Verify it's listening
ss -tlnp | grep 18515
```

The worker listens on port 18515 by default.

### 2. Run PyTorch with hip-remote

### 3. Run PyTorch with hip-remote

Set the environment variables and run your script:

```powershell
$env:TF_WORKER_HOST = "<gpu-server-ip>"
$env:TF_DEBUG = "1"
$env:HIP_REMOTE_LIB_DIR = "path\to\hip-remote-client\build"
$env:MIOPEN_SYSTEM_DB_PATH = (rocm-sdk path --bin)

python your_script.py
```

| Variable | Description |
|----------|------------|
| `TF_WORKER_HOST` | IP address or hostname of the GPU server |
| `TF_WORKER_PORT` | Worker port (default: 18515) |
| `TF_DEBUG` | Set to `1` for debug logging |
| `HIP_REMOTE_LIB_DIR` | Directory containing `amdhip64_7.dll` proxy |
| `MIOPEN_SYSTEM_DB_PATH` | Path to MIOpen TunaNet models (`.tn.model` files). Set to `_rocm_sdk_devel/bin` from the venv. |

### 4. Verify

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

Benchmark results (Windows client to remote MI300X):

| Metric | Per-operation time |
|--------|-------------------|
| SDPA (Flash Attention) | 0.53 ms |
| Conv2d | 1.86 ms |
| GEMM | 0.75 ms |

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
