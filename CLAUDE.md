# TheRock

TheRock is a CMake super-project for building HIP and ROCm from source.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 ./build_tools/fetch_sources.py

# Configure (adjust AMDGPU_FAMILIES for your GPU)
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Build
ninja -C build
```

See [README.md](README.md) for full setup and [docs/development/development_guide.md](docs/development/development_guide.md) for details.

## Development Workflows

### Build Directory Layout

Each component produces:

```
build/component/
├── build/    # CMake build tree
├── stage/    # Install tree (this component only)
├── dist/     # stage/ + runtime dependencies merged
└── stamp/    # Incremental build tracking
```

Final unified output: `build/dist/rocm/` - combined ROCm installation.

### Component Build Targets

Every component exposes these targets (replace `component` with actual name like `hipify`, `clr`, `rocblas`):

| Target                    | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `ninja component`         | Full build (configure + build + stage + dist) |
| `ninja component+build`   | Rebuild after source changes                  |
| `ninja component+dist`    | Update artifacts without full rebuild         |
| `ninja component+expunge` | Clean slate - remove all intermediate files   |

### Common Development Patterns

**Iterate on a single component:**

```bash
# After making changes to component source
ninja -C build clr+build

# Force complete rebuild of one component
ninja -C build clr+expunge && ninja -C build clr
```

**Build subset of ROCm:**

```bash
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPIFY=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

ninja -C build
```

**Test a built component:**

```bash
# Run tests from unified distribution
LD_LIBRARY_PATH=build/dist/rocm/lib build/dist/rocm/bin/test_rocrand_basic

# Or use ctest
ctest --test-dir build
```

**Faster rebuilds with ccache:**

```bash
cmake -B build -GNinja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

**Debug build for specific component:**

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -Drocblas_BUILD_TYPE=RelWithDebInfo \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

### Top-Level Targets

| Target                 | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `ninja` / `ninja dist` | Build everything, populate `build/dist/rocm/` |
| `ninja artifacts`      | Generate artifact directories and manifests   |
| `ninja archives`       | Create `.tar.xz` distribution archives        |
| `ninja expunge`        | Remove all build artifacts                    |

### Submodule Management

- Components are git submodules - use normal git within each
- `./build_tools/fetch_sources.py` resets all submodules and reapplies patches (**destructive** - commit first!)
- Recover lost work: check `git reflog` in affected submodule

### IDE Support

Generate combined compile_commands.json for IDE support:

```bash
cmake --build build --target therock_merged_compile_commands
```

## Code Quality

```bash
pip install pre-commit
pre-commit run              # staged files
pre-commit run --all-files  # all files
pre-commit install          # auto-run on commit
```

Hooks: Black (Python), clang-format (C++), mdformat (Markdown), actionlint (GitHub Actions).

## Style Guidelines

See the [docs/development/style_guides/](docs/development/style_guides/)
directory for each style guide:

- [README.md - General principles](docs/development/style_guides/README.md#general-principles)
- [bash_style_guide.md](docs/development/style_guides/bash_style_guide.md)
- [cmake_style_guide.md](docs/development/style_guides/cmake_style_guide.md)
- [github_actions_style_guide.md](docs/development/style_guides/github_actions_style_guide.md)
- [python_style_guide.md](docs/development/style_guides/python_style_guide.md)

**Python:**

- Use `pathlib.Path` for filesystem operations
- Add type hints to function signatures
- Use `argparse` for CLI with help text
- Don't assume cwd - use script-relative paths

**CMake:**

- Dependencies at super-project level ([docs/development/dependencies.md](docs/development/dependencies.md))
- Build phases: configure → build → stage → dist

## Git Workflow

**Branches:** `users/<username>/<description>` or `shared/<description>`

**PRs:** Target `main`, ensure workflows pass.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Project Structure

```
base/           # rocm-systems (driver, runtime foundations)
compiler/       # LLVM/Clang/LLD, device libraries
core/           # HIP, CLR, ROCr
math-libs/      # rocBLAS, rocFFT, etc.
media-libs/     # rocDecode, rocJPEG
ml-libs/        # MIOpen, composable_kernel
comm-libs/      # RCCL, rocSHMEM
profiler/       # rocprofiler, roctracer
build_tools/    # Python build scripts
cmake/          # CMake infrastructure
docs/           # Documentation
```

## HIP Remote Development

HIP Remote lets a Windows/macOS client run GPU workloads on a remote Linux
machine over TCP. Source lives in `rocm-systems/projects/hip-remote-*`.

### Current Status and Known Issues

**READ FIRST**: See [docs/hip-remote-fire-and-forget.md](docs/hip-remote-fire-and-forget.md)
for the full architecture documentation covering fire-and-forget pipelining
and virtual address allocation.

**Virtual Address (vaddr) Optimization**: hipMalloc/hipStreamCreate are
fire-and-forget using client-assigned virtual addresses. The worker maintains
a hash map + sorted alloc list + last-hit cache for translation. This gives
4x speedup for GPT-2 model loading and 1.75x for SDXL total time.

**OPEN BUG - Triton test_scaled_dot failures**: When running
`pytest test_core.py -k test_scaled_dot` (576 tests), the first ~83 tests
pass (fp16/bf16 types) but all FP8/e2m1 variants fail with
`hipErrorNotInitialized` (code 3) during `hipModuleLoadData`. Root cause:
a kernel launched via FnF with incorrectly translated arguments causes an
async GPU illegal memory access. The error surfaces at the next sync point
(`hipModuleLoadData` for the FP8 kernel). The vaddr translation scans
8-byte values in kernel args for values >= VADDR_BASE. When a non-pointer
8-byte value (like a PhiloxState counter in a 32-byte struct parameter)
coincidentally falls in the vaddr range, it gets incorrectly translated,
corrupting the kernel arguments.

**Key finding**: Moving VADDR_BASE to 0xABCD00000000 caused
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` -- the GPU can't address
that high. Virtual addresses MUST be within the GPU's addressable range
(~0x7xxx_xxxx_xxxx). The 0x7F0000000000 base is correct but the
translation false-positive theory is WRONG. The actual cause of the
FP8 test failure needs further investigation. It may be unrelated to
vaddr translation -- possibly a kernel launch ordering issue, a
missing hipDeviceSynchronize before hipModuleLoadData, or a Triton/ROCm
FP8 support issue specific to remote execution.

**Attempted fixes that didn't resolve the FP8 test issue**:
- COMGR `is_pointer` / `value_kind` field (reverted)
- Only translating first 8 bytes of struct params
- Moving VADDR_BASE higher (GPU can't address it)
- Blind scan removal (breaks kernels without COMGR metadata)
- Dynamic cache growth (fixed other overflow issues)
- TCP keepalive + no-timeout (fixed stuck/disconnect issues)
- Permanent disconnect flag (fixed stale reconnection issues)

**Next steps to investigate**:
1. Run `test_scaled_dot[...-e2m1-...]` natively on the worker to confirm
   it passes without hip-remote
2. Add hipDeviceSynchronize before hipModuleLoadData in the worker to
   drain any pending async errors
3. Check if the FP8 test failure is a Triton-on-ROCm issue unrelated
   to hip-remote (test with a direct hipModuleLoadData call)

**Other recent fixes**:
- Dynamic growable caches (g_loaded_modules, g_func_cache, g_kernarg_sizes,
  g_kernel_arg_cache) to prevent overflow with many JIT-compiled kernels
- TCP keepalive instead of hard I/O timeout on worker
- Client io_timeout_sec = 0 (no timeout)
- Permanent disconnect flag to prevent stale reconnections
- hipFreeHost made local-only (no spurious network call)

### Build & Test

```powershell
# Build client proxy DLL (Windows):
cmake -B build-hip-remote -S rocm-systems/projects/hip-remote-client -G Ninja `
  -DHIP_REMOTE_PROXY_MODE=ON -DROCM_PATH=$(rocm-sdk path --root)
cmake --build build-hip-remote

# Build worker on Windows (from VS x64 Developer prompt, with
# _rocm_sdk_devel/lib/llvm/bin in PATH for clang-cl):
#   1. Generate import lib: gendef + llvm-dlltool from the real amdhip64_7.dll
#   2. cmake -B build-hip-worker ... -DCMAKE_C_COMPILER=clang-cl
#   3. Copy zstd.dll next to hip-worker.exe
# See docs/hip-remote/README.md for full steps.

# Environment (remote worker):
$env:TF_WORKER_HOST = "149.28.118.43"
$env:HIP_REMOTE_LIB_DIR = "D:\jam\temp\TheRock\build-hip-remote"
$env:TRITON_LIBHIP_PATH = "D:\jam\temp\TheRock\build-hip-remote\amdhip64_7.dll"
$env:HIP_REMOTE_CACHE = "1"   # content cache for fast model reload

# Environment (local worker on Windows):
$env:TF_WORKER_HOST = "localhost"
# Start worker with _rocm_sdk_core/bin in PATH:
# d:\...\build-hip-worker\hip-worker.exe

# Worker (Linux): ssh jam@149.28.118.43
HIP_VISIBLE_DEVICES=1 nohup ~/hip-remote/core/hip-remote-worker/build/hip-worker > /tmp/hip-worker.log 2>&1 &

# Tests:
python test_gpt2.py               # GPT-2 generation
python test_sdxl_cat.py           # SDXL image generation
python test_sd15.py               # SD 1.5 (smaller/faster)
```

### Worker Lifecycle and GPU Cleanup

**IMPORTANT:** Always stop the worker with `pkill hip-worker` (SIGTERM), never
`pkill -9` (SIGKILL). SIGTERM triggers the cleanup handler which frees all GPU
allocations and unloads modules. SIGKILL cannot be caught and leaves orphaned
VRAM on the GPU, corrupting GPU 0 for subsequent runs.

```bash
# Correct: graceful shutdown (frees GPU resources)
pkill hip-worker

# WRONG: kills without cleanup, corrupts GPU state
pkill -9 hip-worker  # AVOID THIS
```

On client disconnect, the worker automatically cleans up GPU resources (frees
allocations, unloads modules, calls `hipDeviceSynchronize`). The cleanup is
logged:
```
[HIP-Worker] Cleanup: freed 19 allocations, unloaded 24 modules
[HIP-Worker] Client disconnected (GPU resources cleaned up)
```

If GPU 0 becomes corrupted (from a prior `pkill -9`), use `HIP_VISIBLE_DEVICES=1`
to run on a different GPU, or reboot the server to clear GPU 0.

### Shared Memory IPC Transport (localhost)

When `TF_WORKER_HOST` is `localhost` or `127.0.0.1`, the client automatically
uses shared memory (SHM) instead of TCP for communication with the worker.
This eliminates all TCP syscall overhead, replacing `send()`/`recv()` with
`memcpy` + atomic signaling via a lock-free SPSC ring buffer.

The SHM region (~20MB) contains:
- **FnF ring buffer** (4MB): kernel launches, mallocs, memcpy -- lock-free
- **Sync request slot** (16MB): module loads, device queries -- request/response
- **Sync response slot** (256KB): worker responses

Connection bootstrap: TCP is used only for the init handshake (to exchange
the SHM region name). All subsequent traffic uses shared memory.

Override the auto-detection:
```powershell
$env:HIP_REMOTE_TRANSPORT = "shm"   # force shared memory
$env:HIP_REMOTE_TRANSPORT = "tcp"   # force TCP (e.g., for debugging)
```

### Eager Flush (TCP optimization)

When using TCP transport, the client flushes the FnF write buffer every 64
requests on localhost to reduce GPU starvation. For remote workers over
high-latency links, eager flush is auto-disabled because frequent small TCP
sends hurt throughput. Override with:
```powershell
$env:HIP_REMOTE_EAGER_FLUSH = "64"   # force enable (N requests per flush)
$env:HIP_REMOTE_EAGER_FLUSH = "0"    # force disable
```

## Key Documentation

Reference the below for specialty tasks and deeper analysis, asking questions with subagents in order to avoid polluting context:

- [README.md](README.md) - Build setup, feature flags
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/development/build_system.md](docs/development/build_system.md) - Build architecture
- [docs/development/development_guide.md](docs/development/development_guide.md) - Component development
- [docs/development/dependencies.md](docs/development/dependencies.md) - Dependency management

If development patterns become useful for certain development styles, prefer to document the salient details locally in this CLAUDE.md in addition to exhaustive documentation elsewhere.
