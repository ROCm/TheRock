# TheRock

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Multi-arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml/badge.svg?branch=main&event=push)](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml?query=branch%3Amain) [![Ubuntu 26.04](https://img.shields.io/badge/Ubuntu-26.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com) [![GCC 15](https://img.shields.io/badge/GCC-15.2-blue?logo=gnu)](https://gcc.gnu.org)

TheRock (The HIP Environment and ROCm Kit) is a lightweight open source build platform for HIP and ROCm. It is designed for ROCm contributors as well as developers, researchers, and advanced users who need access to the latest ROCm capabilities without the complexity of traditional package-based installations.

This fork provides out-of-the-box support for **Ubuntu 26.04 LTS (Resolute Raccoon)**, **GCC 15.x**, **CMake 4.x**, and next-generation AMD hardware including **AMD Strix Halo APUs (`gfx1151` / Radeon 8060S / 8050S)**.

---

## 🖥️ Verified Hardware & Testbed Environment

This fork has been completely compiled, tested, and validated on the following hardware platform:

| Component | Specification |
| :--- | :--- |
| **System / Model** | **GMKtec NucBox EVO-X2** (SKU: EVO-X2-001 / BIOS: v1.12) |
| **APU / Processor** | **AMD Ryzen™ AI MAX+ 395** (16 Cores, 32 Threads, Strix Halo) |
| **Integrated Graphics** | **AMD Radeon™ 8060S Graphics** (40 Compute Units / 2560 SPs, RDNA 3.5, ISA: `gfx1151`) |
| **System Memory** | **128 GB LPDDR5X** Unified High-Speed Memory |
| **Operating System** | **Ubuntu 26.04 LTS (Resolute Raccoon)** |
| **Linux Kernel** | `Linux 7.0.0-29-generic` (x86_64) |
| **Host Toolchain** | GCC 15.2.0 (`gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0`) / G++ 15.2.0 |
| **Build Tools** | CMake 4.2.3, Ninja 1.12.1, Python 3.14, `uv` 0.x |

---

## 💡 Key Concepts & Modular Architecture

Building the entire ROCm superproject from scratch compiles over 50 libraries and takes **~4.5 to 5.5 hours**. However, with TheRock's modular workflow, you can build **only what you need in ~20 to 35 minutes**:

```
🍕 Python Virtual Environments (via uv):
   Keeps Python versions (3.14, 3.13, etc.) isolated in clean, dedicated directories (~/virtualenv/venv314/).

🍔 Modular Presets:
   Instead of waiting hours for 50+ unused components, select targeted packages:
   - "llm" : Tailored for llama.cpp, vLLM, and Ollama (~30 min build).
   - "hip" : Minimal HIP runtime & Clang 23 compiler (~20 min build).
   - "vulkan": Mesa graphics, video decoding (rocDecode), and JPEG acceleration.

🧀 Custom Component Toggles (--with-*):
   Easily add extra components (e.g. MIOpen, Profiler, rocFFT) to any preset using simple flags!

🔒 100% Hermetic Virtual Environment Isolation:
   ROCm executables (rocminfo, hipcc, clang) and environment paths are installed DIRECTLY
   into the active virtual environment (.venv/bin/), completely isolating them from any
   existing system-wide /opt/rocm installation!
```

---

## 🚀 Quick Start & Automated Bootstrapper

### ⚡ One-Line Automated Bootstrapper (`bootstrap.sh`)

Instead of manually creating directories and configuring paths, run `bootstrap.sh` to automatically install `uv`, configure the hierarchical workspace, provision the isolated virtual environment, and compile your preset:

```bash
# Automated Zero-to-Hero Build (Sets up ~/virtualenv/therock-7.14/py314-llm/ in one step)
./bootstrap.sh --rocm 7.14 --python 3.14 --preset llm
```

---

### 📁 Hierarchical Multi-Version Workspace Architecture

To prevent commands (`rocminfo`, `amd-smi`) and Python packages from colliding between builds and ROCm releases, the workspace is structured across 3 dimensions (**ROCm Version × Python Version × Preset**):

```
~/virtualenv/
└── therock-7.14/                      # [ROCm / TheRock Release]
    ├── TheRock/                       # Shared Source Repository
    │
    ├── py314-llm/                     # [Python 3.14 + LLM Inference Environment]
    │   ├── .venv/                     # Dedicated venv (with hermetic rocminfo/hipcc wrappers)
    │   └── build/                     # Dedicated Build Tree (build/dist/rocm)
    │
    ├── py314-vulkan/                  # [Python 3.14 + Vulkan / Multimedia Environment]
    │   ├── .venv/
    │   └── build/
    │
    └── py313-llm/                     # [Python 3.13 + LLM Environment]
        ├── .venv/
        └── build/
```

---

### Manual 4-Step Walkthrough

If you prefer to configure steps individually:

#### Step 1: Install System Prerequisites (One-time setup)

```bash
# Install essential compilers and build dependencies
sudo apt update
sudo apt install -y \
  build-essential gcc g++ gfortran git ninja-build cmake \
  pkg-config xxd automake libtool python3-dev libegl1-mesa-dev \
  libsqlite3-dev texinfo bison flex curl make ccache

# Install uv (fast Python package and environment manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Install Rust 1.95 for Mirage emulator & tools
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0
source "$HOME/.cargo/env"
```

#### Step 2: Provision Virtual Environment

```bash
# Provision Python 3.14 virtual environment with uv (takes <1 second)
./therock-env setup-venv 3.14
```

---

### Step 3: Build Your Desired Preset (in ~20 to 35 minutes)

Choose **one** of the following options based on your target workload:

#### ⚡ Option A: AI & LLM Inference (llama.cpp, vLLM, Ollama) - Recommended!
> **Build Time: ~30 - 35 minutes**  
> Compiles Clang 23, HIP runtime, rocBLAS, hipBLASLt, rocPRIM, rocRAND, and hipTensor.
```bash
./therock-env build --preset llm --python 3.14
```

#### ⚡ Option B: Minimal HIP Runtime (Fastest Build)
> **Build Time: ~20 - 25 minutes**  
> Compiles Clang 23 compiler, ROCR, HIP runtime (`clr`), AMDSMI, and `rocminfo`.
```bash
./therock-env build --preset hip --python 3.14
```

#### ⚡ Option C: Multimedia & Vulkan Acceleration (rocDecode + rocJPEG)
> **Build Time: ~25 - 35 minutes**  
> Compiles AMD Mesa (RADV Vulkan), hardware video decoder (`rocDecode`), and JPEG decoder (`rocJPEG`).
```bash
./therock-env build --preset vulkan --python 3.14
```

#### ⚡ Option D: Math & Scientific Computing (FFT, Matrix Solvers)
> **Build Time: ~1.5 - 2 hours**
```bash
./therock-env build --preset math --python 3.14
```

#### ⚡ Option E: Full PyTorch AI Training Stack (with MIOpen & RCCL)
> **Build Time: ~3.5 - 4.5 hours**
```bash
./therock-env build --preset ai --python 3.14
```

---

### Step 4: Activate Environment & Verify GPU Detection

When the build finishes, **220+ ROCm executables are automatically installed directly into your virtual environment (`.venv/bin/`)**.

Simply activate your virtual environment, and the built ROCm binaries will take precedence over any system `/opt/rocm`:

```bash
# 1. Activate the Python virtual environment (ROCm paths are auto-loaded!)
source ~/virtualenv/venv314/.venv314/bin/activate

# 2. Verify binary location (points to your venv, NOT system /opt/rocm)
which rocminfo
# Output: /home/analogbox/virtualenv/venv314/.venv314/bin/rocminfo

# 3. Check GPU detection
rocminfo
```

When you see your GPU device (**AMD Radeon 8060S / gfx1151**) in the output, your environment is 100% operational:

```
=====================    
HSA Agents               
=====================    
Agent 1: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (CPU)
Agent 2: gfx1151 / AMD Radeon 8060S Graphics (GPU, 40 Compute Units)
  Supported ISA:
    - amdgcn-amd-amdhsa--gfx1151
    - amdgcn-amd-amdhsa--gfx11-generic
```

---

## 🍕 Available Presets

| Preset Name | Aliases | Included Libraries & Components | Recommended Use Case | Build Time |
| :--- | :--- | :--- | :--- | :--- |
| **`llm-inference`** | **`llm`**, `inference` | HIP + `rocBLAS` + `hipBLASLt` + `rocPRIM` + `hipTensor` | **vLLM, llama.cpp, Ollama, ExLlamaV2** | **~30 min** |
| **`hip`** | `hip` | AMD Clang 23 + HIP Runtime + AMDSMI + `rocminfo` | Lightweight C++/HIP development | **~20 min** |
| **`cv-vision`** | `vision`, `cv` | HIP + `RPP` + `rocDecode` + `rocJPEG` + AMD Mesa | OpenCV, realtime video AI preprocessing | **~30 min** |
| **`vulkan`** | `vulkan` | HIP + AMD Mesa (RADV Vulkan) + Video codecs | Vulkan graphics & hardware decoding | **~30 min** |
| **`math`** | `math` | HIP + `rocBLAS` + `rocFFT` + `rocRAND` + `rocSOLVER` | FFT signal processing & numerical math | ~1.5 hours |
| **`hpc`** | `hpc` | HIP + Math Stack + `rocALUTION` + `rocSPARSE` | Physics simulations & engineering HPC | ~1.5 hours |
| **`ai`** | `ai` | HIP + Math + `MIOpen` (CK) + `RCCL` + `hipDNN` | Full PyTorch / JAX training framework | ~4 hours |
| **`profiler`** | `profiler` | `rocprofiler-sdk`, `rocprofiler-systems`, `rocgdb` | Performance tracing & GPU debugging | ~40 min |
| **`full`** | `full` | Complete ROCm stack (all 50+ libraries) | Full distribution release build | ~5 hours |

---

## 🧀 Component Customization Flags (`--with-*`)

You can customize any preset on-the-fly by adding or removing individual components:

```bash
# Example 1: Build LLM preset with GPU Profiler and Debugger
./therock-env build --preset llm --python 3.14 --with-profiler

# Example 2: Build LLM preset with MIOpen (Deep Learning Convolutions)
./therock-env build --preset llm --python 3.14 --with-miopen

# Example 3: Build HIP preset with Fast Fourier Transforms (rocFFT)
./therock-env build --preset hip --python 3.14 --with-fft

# Example 4: Batch matrix build across multiple presets sequentially
./therock-env build-matrix --presets hip,llm,vulkan --python 3.14
```

### Supported Component Toggles

* **`--with-miopen`**: Adds MIOpen and Composable Kernel.
* **`--with-rccl`**: Adds multi-GPU collective communications (RCCL).
* **`--with-profiler`**: Adds `rocprofv3`, `rocprofiler-sdk`, and `rocgdb`.
* **`--with-fft`**: Adds `rocFFT` math library.
* **`--with-media`** / **`--with-vulkan`**: Adds AMD Mesa, `rocDecode`, and `rocJPEG`.
* **`--without-blas`**: Excludes BLAS matrix math libraries.

---

## 📁 Managing Environments and Builds

```bash
# 1. List all detected Python virtual environments
./therock-env list-envs

# 2. List all completed ROCm build trees and disk usage
./therock-env list-builds

# 3. Hermetically install an existing build into a virtual environment
./therock-env install-to-venv --build-dir build_1151 --python 3.14
```

---

## 📚 Documentation & References

* [GCC 15 & Ubuntu 26.04 Technical Porting Guide](docs/GCC15_UBUNTU2604_PORTING_GUIDE.md): Technical details on GCC 15 libstdc++ `<version>` migration, CMake 4.x deferred dependency provider, and Meson symbol version script fixes.
* [Development Guide](docs/development/development_guide.md): Guide for component developers.
* [Supported GPUs](SUPPORTED_GPUS.md): AMD GPU architecture roadmap.
* [CONTRIBUTING.md](CONTRIBUTING.md): Guidelines for contributing.
