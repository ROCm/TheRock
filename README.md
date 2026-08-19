# TheRock

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Multi-arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml/badge.svg?branch=main&event=push)](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml?query=branch%3Amain) [![Ubuntu 26.04](https://img.shields.io/badge/Ubuntu-26.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com) [![GCC 15](https://img.shields.io/badge/GCC-15.2-blue?logo=gnu)](https://gcc.gnu.org)

TheRock (The HIP Environment and ROCm Kit) is a lightweight open source build platform for HIP and ROCm. It is designed for ROCm contributors as well as developers, researchers, and advanced users who need access to the latest ROCm capabilities without the complexity of traditional package-based installations.

This fork provides out-of-the-box support for **Ubuntu 26.04 LTS (Resolute Raccoon)**, **GCC 15.x**, **CMake 4.x**, and next-generation AMD hardware including **AMD Strix Halo APUs (`gfx1151` / Radeon 8060S / 8050S)**.

---

## Features

- **Full ROCm Stack & HIP Super-Project**: Build ROCm from source in a single unified build tree.
- **Ubuntu 26.04 LTS & GCC 15 Ready**: Complete compatibility with GCC 15 (C23 default, stricter C++20 template semantics, ISO C++ `<version>` header conformance).
- **CMake 4.x Lifecycle Compatibility**: Deferred top-level dependency provider initialization.
- **AMD Strix Halo (`gfx1151`) First-Class Support**: Targeted builds for AMD Ryzen AI MAX+ 395 and Radeon 8060S/8050S graphics.
- **Unified Distribution Output**: All components installed into a relocatable, single folder at `build/dist/rocm/`.
- **Framework Support**: Build PyTorch and JAX with full ROCm GPU acceleration from source.

For detailed technical notes on the GCC 15 and Ubuntu 26.04 port, see [docs/GCC15_UBUNTU2604_PORTING_GUIDE.md](docs/GCC15_UBUNTU2604_PORTING_GUIDE.md).

---

## Quick Start (Ubuntu 26.04 LTS / GCC 15)

### 1. Prerequisites & System Dependencies

```bash
# Update package lists and install build dependencies
sudo apt update
sudo apt install -y \
  build-essential \
  gcc \
  g++ \
  gfortran \
  git \
  ninja-build \
  cmake \
  pkg-config \
  xxd \
  automake \
  libtool \
  python3-venv \
  python3-dev \
  libegl1-mesa-dev \
  libsqlite3-dev \
  texinfo \
  bison \
  flex \
  curl \
  make \
  ccache
```

### 2. Rust Toolchain Setup (for Mirage emulator & tools)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0
source "$HOME/.cargo/env"
```

### 3. Clone Repository & Setup Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/analogbox/TheRock.git
cd TheRock

# Initialize Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download submodules and apply patches
python3 ./build_tools/fetch_sources.py
```

---

## Building ROCm from Source

### Hardware Architecture Targets

Select the GPU architecture matching your system using `-DTHEROCK_AMDGPU_FAMILIES` or `-DTHEROCK_AMDGPU_TARGETS`:

| GPU / Architecture | Target Flag |
| :--- | :--- |
| **AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S / 8050S)** | `-DTHEROCK_AMDGPU_FAMILIES=gfx1151` |
| **AMD Radeon RX 7900 XTX / XT / GRE (RDNA3)** | `-DTHEROCK_AMDGPU_FAMILIES=gfx1100` |
| **AMD Radeon RX 7800 XT / 7700 XT (RDNA3)** | `-DTHEROCK_AMDGPU_FAMILIES=gfx1101` |
| **AMD Radeon RX 7600 (RDNA3)** | `-DTHEROCK_AMDGPU_FAMILIES=gfx1102` |
| **AMD Instinct MI300X / MI300A (CDNA3)** | `-DTHEROCK_AMDGPU_FAMILIES=gfx942` |
| **All Supported RDNA3 / 3.5 Architectures** | `-DTHEROCK_AMDGPU_FAMILIES=gfx110X-all` |

### Recommended Build with `ccache` Acceleration

Building ROCm from scratch compiles LLVM, HIP runtimes, and tens of thousands of GPU kernels. Utilizing `ccache` speeds up subsequent and iterative builds dramatically:

```bash
# 1. Configure CCache environment
eval "$(./build_tools/setup_ccache.py)"

# 2. Configure CMake super-project (adjust target for your GPU)
cmake -B build -GNinja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1151

# 3. Compile all ROCm components
ninja -C build
```

---

## Using the Built ROCm Installation

Upon build completion, the complete unified ROCm environment is staged in `build/dist/rocm/`.

### Setting Environment Variables

```bash
export ROCM_PATH="$(pwd)/build/dist/rocm"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH"
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
```

### Verifying GPU Detection

Verify that the runtime and GPU agent are active:

```bash
rocminfo
```

Example output on **AMD Strix Halo (Radeon 8060S)**:
```
=====================    
HSA Agents               
=====================    
Agent 1: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (CPU)
Agent 2: gfx1151 / AMD Radeon 8060S Graphics (GPU)
  Supported ISA:
    - amdgcn-amd-amdhsa--gfx1151
    - amdgcn-amd-amdhsa--gfx11-generic
```

---

## Modular Component Development

TheRock allows compiling and iterating on individual components without rebuilding the entire project:

| Action | Command |
| :--- | :--- |
| **Rebuild HIP / CLR Runtime** | `ninja -C build clr+build` |
| **Rebuild MIOpen** | `ninja -C build MIOpen+build` |
| **Rebuild rocBLAS** | `ninja -C build rocblas+build` |
| **Rebuild rocprofiler-sdk** | `ninja -C build rocprofiler-sdk+build` |
| **Clean component build** | `ninja -C build <component>+expunge` |

### Building a Lightweight Subset of ROCm

To build only specific components (e.g., HIP runtime + BLAS + MIOpen):

```bash
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_BLAS=ON \
  -DTHEROCK_ENABLE_MIOPEN=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1151

ninja -C build
```

---

## Project Structure

```
TheRock/
├── base/           # Core foundations (driver, rocm-cmake, half)
├── compiler/       # AMD LLVM / Clang 23, LLD, Device Libs, hipify
├── core/           # CLR (HIP & OpenCL runtime), ROCR-Runtime, amdsmi
├── math-libs/      # rocBLAS, hipBLASLt, rocFFT, rocRAND, rocPRIM, rocThrust, rocSOLVER
├── ml-libs/        # MIOpen, composable_kernel, hipDNN
├── comm-libs/      # RCCL, rocSHMEM
├── profiler/       # rocprofiler-sdk, rocprofiler-systems, roctracer
├── media-libs/     # rocDecode, rocJPEG
├── third-party/    # Bundled sysdeps (libdrm, amd-mesa, elfutils, libnl, sqlite3)
├── build_tools/    # Python build utilities and patch automation
└── docs/           # Documentation and porting guides
```

---

## Documentation & References

- [GCC 15 & Ubuntu 26.04 Porting Guide](docs/GCC15_UBUNTU2604_PORTING_GUIDE.md): Technical details of GCC 15, C23, and CMake 4.x changes.
- [Development Guide](docs/development/development_guide.md): Guide for component developers.
- [Supported GPUs](SUPPORTED_GPUS.md): GPU architecture roadmap and details.
- [CONTRIBUTING.md](CONTRIBUTING.md): Guidelines for contributing to TheRock.
