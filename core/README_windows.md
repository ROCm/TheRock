# Building and Testing hip-remote on Windows

This guide covers building the HIP remote client libraries on Windows and
connecting them to a hip-worker service running on a Linux machine with AMD
GPUs.

## Architecture

```
 Windows (client)                    Linux (worker)
┌───────────────────┐    TCP/IP    ┌─────────────────────┐
│  Your application │◄──────────►│  hip-worker service  │
│  links amdhip64   │   :18515    │  (real HIP + GPUs)   │
│  (remote client)  │             │                      │
└───────────────────┘             └─────────────────────┘
```

The client libraries (`amdhip64.dll`, `amdsmi-remote.dll`) implement the HIP
and AMD SMI APIs by forwarding calls over TCP to a worker process on a remote
Linux system. Applications link against these libraries as if they were the
real HIP runtime.

## Prerequisites

**Windows (client side):**

- CMake 3.25 or later
- Visual Studio 2019+ or the MSVC Build Tools (C11 support required)
- Ninja (recommended) or MSBuild

**Linux (worker side):**

- ROCm 6.x or 7.x installed (provides HIP runtime and optionally AMD SMI)
- CMake 3.25+, Ninja or Make, and a C/C++ compiler (GCC or Clang)
- One or more AMD GPUs

## Building on Windows

### 1. Build the HIP remote client

Open a Developer Command Prompt (or any terminal with MSVC on PATH) and run:

```bat
cd core\hip-remote-client
cmake -B build -G Ninja -DBUILD_TESTING=ON
ninja -C build
```

This produces:

| File | Description |
|------|-------------|
| `build\amdhip64.dll` | HIP remote client shared library |
| `build\amdhip64.lib` | Import library for linking |
| `build\hip_remote_test_basic.exe` | Integration test binary |

### 2. Build the SMI remote client (optional)

```bat
cd core\smi-remote-client
cmake -B build -G Ninja -DBUILD_TESTING=ON
ninja -C build
```

This produces:

| File | Description |
|------|-------------|
| `build\amdsmi-remote.dll` | AMD SMI remote client shared library |
| `build\amdsmi-remote.lib` | Import library for linking |
| `build\smi-remote.exe` | CLI tool for querying GPU metrics |
| `build\test_smi_basic.exe` | Integration test binary |

## Setting Up the Worker (Linux)

The worker runs on your Linux machine with AMD GPUs. Copy the worker source
from `core/hip-remote-worker/` and the protocol headers from
`core/hip-remote-client/include/` to the Linux machine, preserving the
relative directory layout:

```bash
# On the Linux machine
mkdir -p ~/hip-remote/core/hip-remote-worker/src
mkdir -p ~/hip-remote/core/hip-remote-client/include/hip_remote

# Copy files (from your Windows machine via scp, rsync, etc.)
scp core/hip-remote-worker/CMakeLists.txt          user@linux:~/hip-remote/core/hip-remote-worker/
scp core/hip-remote-worker/hip-worker.service.in   user@linux:~/hip-remote/core/hip-remote-worker/
scp core/hip-remote-worker/src/*                   user@linux:~/hip-remote/core/hip-remote-worker/src/
scp core/hip-remote-client/include/hip_remote/*.h  user@linux:~/hip-remote/core/hip-remote-client/include/hip_remote/
```

Then build and start the worker on the Linux machine:

```bash
cd ~/hip-remote/core/hip-remote-worker
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C build

# Start the worker (listens on port 18515 by default)
./build/hip-worker -v
```

You should see output like:

```
[HIP-Worker] Initializing HIP on device 0...
[HIP-Worker] Device: AMD Instinct MI300X
[HIP-Worker]   Memory: 192.0 GB
[HIP-Worker]   Compute: 9.4
[HIP-Worker] AMD SMI: 8 GPU(s) available
[HIP-Worker] Listening on port 18515
```

### Worker options

| Flag | Environment Variable | Description |
|------|---------------------|-------------|
| `-p PORT` | `TF_WORKER_PORT` | Listen port (default: 18515) |
| `-d DEVICE` | `TF_DEVICE_ID` | Default GPU device (default: 0) |
| `-v` | `TF_DEBUG=1` | Enable verbose logging |

To run the worker in the background:

```bash
nohup ./build/hip-worker -v > /tmp/hip-worker.log 2>&1 &
```

## Testing

### Running the HIP client tests

Set the worker host and run the test binary from your Windows machine:

```bat
set TF_WORKER_HOST=<linux-ip-address>
build\hip_remote_test_basic.exe
```

Or in PowerShell:

```powershell
$env:TF_WORKER_HOST="<linux-ip-address>"
.\build\hip_remote_test_basic.exe
```

Optionally set `TF_DEBUG=1` for verbose protocol logging.

Expected output:

```
=== Remote HIP Basic Tests ===

Test: hipGetDeviceCount
  Found 8 device(s)
Test: hipSetDevice / hipGetDevice
  Set and get device 0: OK
Test: hipGetDeviceProperties
  Device 0: AMD Instinct MI300X
Test: hipMalloc / hipFree
  Allocated and freed 1MB: OK
Test: hipMemcpy
  Round-trip memcpy 1KB: OK
Test: hipStreamCreate / hipStreamDestroy
  Created and destroyed stream: OK
Test: hipEventCreate / hipEventDestroy
  Created and destroyed event: OK
Test: hipRuntimeGetVersion
  Runtime version: 70226015

=== Results ===
All tests PASSED
```

### Running the SMI remote CLI

```bat
build\smi-remote.exe --host <linux-ip-address> list
build\smi-remote.exe --host <linux-ip-address> metrics
build\smi-remote.exe --host <linux-ip-address> info 0
```

### Running SMI client tests

```bat
set TF_WORKER_HOST=<linux-ip-address>
build\test_smi_basic.exe
```

### Running the Hello HIP example (GPU kernel launch)

The `examples/hello_hip/` directory contains a self-contained example that
compiles a GPU kernel and launches it on the remote machine. The GPU code
object (AMDGPU ISA) can be compiled on any platform -- it contains no
host-specific code.

```bat
cd examples\hello_hip
build.bat
set TF_WORKER_HOST=<linux-ip-address>
hello_hip.exe
```

The build script performs four steps:

1. Compiles `vector_add_kernel.hip` to a GPU code object using `amdclang++`
2. Embeds the code object as a C byte array (`vector_add_co.h`)
3. Compiles the host program with MSVC, linking against `amdhip64.lib`
4. Copies `amdhip64.dll` alongside the executable

If `amdclang++` is not on your PATH, install it via `pip install rocm` or
compile the `.co` file on the Linux worker machine instead:

```bash
hipcc --genco --offload-arch=gfx942 -o vector_add.co vector_add_kernel.hip
```

Copy `vector_add.co` back to the `examples/hello_hip/` directory and re-run
`build.bat` -- it will use the existing `.co` file.

Expected output:

```
=== Hello HIP (Remote) ===

Remote GPU count: 8
Device 0: AMD Instinct MI300X

Loading embedded kernel (9784 bytes of gfx942 ISA)
Kernel ready.

vector_add<<<4, 256>>>(1024 elements)...
PASSED: all 1024 elements correct
  c[0]=0  c[511]=1533  c[1023]=3069
```

## Environment Variables

Both the HIP and SMI clients read configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_WORKER_HOST` | `localhost` | Hostname or IP of the hip-worker |
| `TF_WORKER_PORT` | `18515` | TCP port of the hip-worker |
| `TF_DEBUG` | `0` | Set to `1` for verbose debug logging |

## Using the Client Library in Your Application

Link your application against `amdhip64.lib` and include the HIP headers.
The remote client exposes the same API surface as the standard HIP runtime:

```c
#include "hip_remote/hip_remote_client.h"

int main() {
    int count = 0;
    hipGetDeviceCount(&count);
    printf("Found %d GPU(s)\n", count);

    void* ptr = NULL;
    hipMalloc(&ptr, 1024);
    hipFree(ptr);
    return 0;
}
```

Compile and link:

```bat
cl /I core\hip-remote-client\include myapp.c /link build\amdhip64.lib
```

At runtime, ensure `amdhip64.dll` is in your PATH or alongside the
executable, and `TF_WORKER_HOST` points to your worker machine.

## Troubleshooting

**Connection refused:** Verify the worker is running on the Linux machine
(`ss -tlnp | grep 18515`) and that port 18515 is not blocked by a firewall.

**Timeout errors:** Check network connectivity (`ping <linux-ip>`). The
default I/O timeout is 60 seconds.

**Worker crashes on client disconnect:** This is expected behavior; the worker
logs `Client disconnected` and waits for the next connection. It does not
exit.

**Build error about `ws2_32`:** Ensure you are building with MSVC (not
MinGW). The Winsock2 library is linked automatically via CMakeLists.txt.
