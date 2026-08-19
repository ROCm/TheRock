# TheRock for AMD Strix Halo

### ⚡ Out-of-the-Box ROCm & HIP Platform for AMD Ryzen™ AI MAX+ (`gfx1151` / RDNA 3.5) on Ubuntu 26.04 & GCC 15

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Multi-arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml/badge.svg?branch=main&event=push)](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml?query=branch%3Amain) [![Ubuntu 26.04](https://img.shields.io/badge/Ubuntu-26.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com) [![GCC 15](https://img.shields.io/badge/GCC-15.2-blue?logo=gnu)](https://gcc.gnu.org) [![AMD Strix Halo](https://img.shields.io/badge/AMD%20Strix%20Halo-gfx1151-ED1C24?logo=amd&logoColor=white)](https://www.amd.com)

TheRock (The HIP Environment and ROCm Kit) is a lightweight open source build platform for HIP and ROCm.

This fork provides verified out-of-the-box support for **AMD Strix Halo APUs (`gfx1151` / Radeon 8060S / 8050S / Ryzen AI MAX+ 395)** on **Ubuntu 26.04 LTS (Resolute Raccoon)**, **GCC 15.2**, and **CMake 4.x**, featuring modular 30-minute builds and hermetic Python virtual environment isolation.

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
🍕 3-Dimensional Hierarchical Workspaces (via uv):
   Keeps ROCm releases, Python versions (3.14, 3.13), and build presets cleanly separated
   (e.g. ~/virtualenv/therock-7.14/py314-llm/ vs ~/virtualenv/therock-7.14/py314-vulkan/).

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

### ⚡ One-Line Zero-Install Bootstrapper (No Prior Clone Required!)

On a fresh machine without having cloned the repository yet, run this single command. It will **automatically install `uv`, create the hierarchical workspace directory, clone your fork repository, provision the Python 3.14 virtual environment, and compile ROCm**:

```bash
# Direct Single-Command Installation (Fresh Machine / Zero-Install)
curl -fsSL https://raw.githubusercontent.com/analogbox/TheRock/main/bootstrap.sh | bash -s -- --rocm 7.14 --python 3.14 --preset llm
```

Or if you prefer downloading `bootstrap.sh` first:

```bash
# Download bootstrap script and run
curl -O https://raw.githubusercontent.com/analogbox/TheRock/main/bootstrap.sh
chmod +x bootstrap.sh

# Run automated build (e.g. for LLM Inference)
./bootstrap.sh --rocm 7.14 --python 3.14 --preset llm
```

---

---

### 🔍 Under the Hood: Exactly What Happens During Installation

Most installation scripts leave you guessing where files go and what gets modified. Here is the exact, step-by-step breakdown of how TheRock automates and isolates your environments:

#### 1. The 6-Step Automated Lifecycle
When you execute `./bootstrap.sh --rocm 7.14 --python 3.14 --preset llm`:
1. **Directory Provisioning (`mkdir -p`)**: If `~/virtualenv` does not exist, it creates `~/virtualenv/therock-7.14/py314-llm/` from scratch.
2. **Dependency & `uv` Check**: Checks if `uv`, `cmake`, `ninja`, and GCC 15 exist. If `uv` is missing, it installs `uv` in <1 second without touching system files.
3. **Smart Source Sharing (Disk Optimization)**: Clones `analogbox/TheRock` into `~/virtualenv/therock-7.14/TheRock/`. If it was already cloned, it reuses the existing clone without re-downloading (~15 GB saved).
4. **Dedicated Virtual Environment**: Provisions a hermetic Python 3.14 virtual environment at `~/virtualenv/therock-7.14/py314-llm/.venv/`.
5. **Targeted Compilation**: Compiles only the requested preset (e.g. LLM inference) in `~/virtualenv/therock-7.14/py314-llm/build/` with `ccache` acceleration (~30 min).
6. **Hermetic Command Injection**: Generates **226 executable wrappers** (`rocminfo`, `hipcc`, `amdclang`, `rocm-smi`) directly inside `py314-llm/.venv/bin/` and connects environment hooks in `py314-llm/.venv/bin/activate`.

---

#### 2. Running Multiple Python Versions Side-by-Side (e.g. Python 3.14 vs 3.13)
When you build different Python versions or different presets, each one gets its own dedicated folder. They **never overwrite each other's binaries, pip packages, or libraries**:

```
~/virtualenv/
└── therock-7.14/                      # ROCm 7.14 Workspace Root
    │
    ├── TheRock/                       # [Shared Source Code] Cloned ONCE for this version
    │
    ├── py314-llm/                     # [Python 3.14 + LLM Inference]
    │   ├── .venv/                     # Python 3.14 runtime + packages (torch, numpy)
    │   │   └── bin/                   # Contains 3.14-linked rocminfo, hipcc, amdclang
    │   └── build/                     # 3.14 LLM build tree (dist/rocm)
    │
    ├── py313-llm/                     # [Python 3.13 + LLM Inference]
    │   ├── .venv/                     # Python 3.13 runtime + packages (completely separate!)
    │   │   └── bin/                   # Contains 3.13-linked rocminfo, hipcc, amdclang
    │   └── build/                     # 3.13 LLM build tree (dist/rocm)
    │
    └── py314-vulkan/                  # [Python 3.14 + Vulkan / Multimedia]
        ├── .venv/                     # Python 3.14 with Mesa/Vulkan tools
        └── build/                     # Vulkan build tree
```

#### 3. How Switching Environments Works in 1 Second
Because each environment is completely self-contained, switching between Python versions or presets is as simple as activating the target `.venv`:

```bash
# Work on Python 3.14 LLM models:
source ~/virtualenv/therock-7.14/py314-llm/.venv/bin/activate
rocminfo   # Executes py314-llm's ROCm!

# Switch to Python 3.13 LLM models:
source ~/virtualenv/therock-7.14/py313-llm/.venv/bin/activate
rocminfo   # Executes py313-llm's ROCm!

# Switch to Vulkan / Media experiments:
source ~/virtualenv/therock-7.14/py314-vulkan/.venv/bin/activate
```

---

#### 4. The "Source Factory" vs "Virtual Environment Injection" Model

To understand how commands like `rocminfo` and `hipcc` run inside your virtual environment without conflicting with system `/opt/rocm`:

```
[1. Source Code Factory]
  ~/virtualenv/therock-7.14/TheRock/ (Shared Git Source & Build Scripts)
             │
             ▼ (Compiles targeted preset in ~30 min)
[2. Build Output Artifacts]
  ~/virtualenv/therock-7.14/py314-llm/build/dist/rocm/ (Compiled ROCm Libraries)
             │
             ▼ (Automatic Wrapper Injection & Path Linking)
[3. Hermetically Installed inside your Virtualenv!]
  ~/virtualenv/therock-7.14/py314-llm/.venv/bin/
             ├── python3
             ├── pip
             ├── rocminfo     ← (Auto-injected wrapper for this specific build!)
             ├── hipcc        ← (Auto-injected wrapper!)
             ├── amdclang     ← (Auto-injected wrapper!)
             └── rocm-smi     ← (Auto-injected wrapper!)
```

* **When you activate (`source .../.venv/bin/activate`)**:
  `which rocminfo` returns `/home/analogbox/virtualenv/therock-7.14/py314-llm/.venv/bin/rocminfo`. Your shell automatically uses this environment's ROCm and ignores `/opt/rocm`.
* **When you deactivate (`deactivate`)**:
  All ROCm environment variables are unset, and your shell instantly reverts to the clean default Ubuntu environment with zero leftover residue.

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

### Step 3: Choose Your Workload Preset

Select the preset matching your project goals:

#### ⚡ Tier 1: AI & LLM Workloads (Recommended for Most Users)

##### Option A: LLM Inference & LoRA / QLoRA Fine-Tuning (`llm`) - Most Popular!
> **Build Time: ~30 - 35 minutes**  
> **Included**: Clang 23, HIP runtime, `rocBLAS`, `hipBLASLt`, `rocPRIM`, `rocRAND`, `hipTensor`.  
> **Use Case**: Running **vLLM, Ollama, llama.cpp (HIP)**, and **LoRA / QLoRA / SFT fine-tuning with Unsloth / Hugging Face**.  
> *(Note: LLMs use matrix multiplication, so they do NOT need heavy convolution libraries like MIOpen!)*
```bash
./therock-env build --preset llm --python 3.14
```

##### Option B: Full PyTorch AI Training & Vision/CNN Stack (`ai-full`)
> **Build Time: ~3.5 - 4.5 hours**  
> **Included**: Everything in `llm` + `MIOpen` (Deep Learning Convolutions) + `RCCL` (Multi-GPU Distributed) + `hipDNN`.  
> **Use Case**: Full pre-training from scratch, CNN/Vision models (ResNet, YOLO), and Stable Diffusion.
```bash
./therock-env build --preset ai-full --python 3.14
```

---

#### ⚡ Tier 2: Graphics & Multimedia Acceleration

##### Option C: Vulkan & Hardware Video Acceleration (`vulkan-media`)
> **Build Time: ~25 - 35 minutes**  
> **Included**: AMD Mesa (RADV Vulkan), hardware video decoder (`rocDecode` 4K/8K AV1/HEVC), and `rocJPEG`.  
> **Use Case**: Vulkan graphics rendering, hardware video decoding pipelines, and `llama.cpp` Vulkan backend.
```bash
./therock-env build --preset vulkan-media --python 3.14
```

---

#### ⚡ Tier 3: Core Foundations & Scientific HPC

##### Option D: Minimal HIP Foundation Engine (`core-hip`)
> **Build Time: ~20 - 25 minutes**  
> **Included**: AMD Clang 23 compiler, ROCR, HIP runtime (`clr`), AMDSMI, and `rocminfo`.  
> **Use Case**: Lightweight C++/HIP kernel development and basic GPU testing.
```bash
./therock-env build --preset core-hip --python 3.14
```

##### Option E: Scientific Math & Simulation (`math-hpc`)
> **Build Time: ~1.5 - 2 hours**  
> **Included**: HIP + `rocBLAS` + `rocFFT` + `rocSOLVER` + `rocSPARSE` + `rocALUTION`.  
> **Use Case**: Fast Fourier Transforms (FFT), sparse matrix solvers, and physics simulations.
```bash
./therock-env build --preset math-hpc --python 3.14
```

---

### Step 4: Activate Environment & Verify GPU Detection

When the build finishes, **220+ ROCm executables are automatically installed directly into your virtual environment (`.venv/bin/`)**.

Simply activate your virtual environment, and the built ROCm binaries will take precedence over any system `/opt/rocm`:

```bash
# 1. Activate the Python virtual environment (ROCm paths are auto-loaded!)
source ~/virtualenv/therock-7.14/py314-llm/.venv/bin/activate

# 2. Verify binary location (points to your venv, NOT system /opt/rocm)
which rocminfo
# Output: /home/analogbox/virtualenv/therock-7.14/py314-llm/.venv/bin/rocminfo

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

## 🍕 Available Presets Overview

| Workload Tier | Preset Name | Convenient Aliases | Included Components | Recommended Use Case | Build Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (AI / LLM)** | **`llm`** | `llm-inference`, `lora`, `finetuning` | HIP + `rocBLAS` + `hipBLASLt` + `rocPRIM` + `hipTensor` | **vLLM, llama.cpp (HIP), Ollama, LoRA/QLoRA Fine-Tuning** | **~30 min** |
| **Tier 1 (AI / LLM)** | **`ai-full`** | `ai`, `training`, `pytorch` | `llm` stack + `MIOpen` (CK) + `RCCL` + `hipDNN` | Full PyTorch training, CNN/Vision, Stable Diffusion | ~4 hours |
| **Tier 2 (Media)** | **`vulkan-media`** | `vulkan`, `media`, `vision`, `cv` | AMD Mesa (RADV Vulkan) + `rocDecode` + `rocJPEG` | Vulkan graphics, 4K/8K video decode, llama.cpp (Vulkan) | **~25 min** |
| **Tier 3 (Engine)** | **`core-hip`** | `hip`, `core`, `minimal` | AMD Clang 23 + HIP Runtime + AMDSMI + `rocminfo` | Minimal C++/HIP kernel development | **~20 min** |
| **Tier 3 (Math)** | **`math-hpc`** | `math`, `hpc`, `scientific` | `rocBLAS` + `rocFFT` + `rocSOLVER` + `rocSPARSE` + `rocALUTION` | FFT signal processing, matrix solvers, simulations | ~1.5 hours |
| **Tools** | **`profiler`** | `profiler` | `rocprofiler-sdk`, `rocprofiler-systems`, `rocgdb` | GPU performance tracing & interactive debugging | ~40 min |
| **Monolithic** | **`full`** | `full` | Complete ROCm stack (all 50+ libraries) | Full monolithic distribution release build | ~5 hours |

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

* [Upstream Sync & Rebase Guide](docs/UPSTREAM_SYNC_AND_REBASE_GUIDE.md): Step-by-step workflow for upgrading your fork when AMD releases new ROCm versions.
* [GCC 15 & Ubuntu 26.04 Technical Porting Guide](docs/GCC15_UBUNTU2604_PORTING_GUIDE.md): Technical details on GCC 15 libstdc++ `<version>` migration, CMake 4.x deferred dependency provider, and Meson symbol version script fixes.
* [Development Guide](docs/development/development_guide.md): Guide for component developers.
* [Supported GPUs](SUPPORTED_GPUS.md): AMD GPU architecture roadmap.
* [CONTRIBUTING.md](CONTRIBUTING.md): Guidelines for contributing.
