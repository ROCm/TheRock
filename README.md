# TheRock for AMD Strix Halo

### ⚡ Out-of-the-Box ROCm & HIP Platform for AMD Ryzen™ AI MAX+ (`gfx1151` / RDNA 3.5) on Ubuntu 26.04 & GCC 15

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Multi-arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml/badge.svg?branch=main&event=push)](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml?query=branch%3Amain) [![Ubuntu 26.04](https://img.shields.io/badge/Ubuntu-26.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com) [![GCC 15](https://img.shields.io/badge/GCC-15.2-blue?logo=gnu)](https://gcc.gnu.org) [![AMD Strix Halo](https://img.shields.io/badge/AMD%20Strix%20Halo-gfx1151-ED1C24?logo=amd&logoColor=white)](https://www.amd.com)

TheRock is a modular open-source build platform for HIP and ROCm. This fork provides verified, zero-friction support for **AMD Strix Halo APUs (`gfx1151` / Radeon 8060S / 8050S / Ryzen AI MAX+ 395)** on **Ubuntu 26.04 LTS (Resolute Raccoon)**, **GCC 15.2**, and **CMake 4.x**, featuring modular 30-minute builds and hermetic Python virtual environment isolation.

> [!IMPORTANT]
> **🚀 Purpose-Built & Validated for Ubuntu 26.04 LTS (Resolute Raccoon)**  
> Standard ROCm packages and upstream builds fail on Ubuntu 26.04 due to strict **GCC 15.2 ISO C++20 standard header migration**, **CMake 4.x deferred dependency providers**, and **Linux 7.0+ kernel driver ABIs**. This repository is specifically engineered, patched, and benchmarked to provide a **100% stable, out-of-the-box ROCm platform on Ubuntu 26.04 LTS**.

---

## 🖥️ Verified Testbed Platform

| Component | Specification |
| :--- | :--- |
| **System / Model** | **GMKtec NucBox EVO-X2** (SKU: EVO-X2-001 / BIOS: v1.12) |
| **APU / Processor** | **AMD Ryzen™ AI MAX+ 395** (16 Cores, 32 Threads, Strix Halo) |
| **Integrated Graphics** | **AMD Radeon™ 8060S Graphics** (40 Compute Units / 2560 SPs, RDNA 3.5, ISA: `gfx1151`) |
| **System Memory** | **128 GB LPDDR5X** Unified High-Speed Memory |
| **Operating System** | **Ubuntu 26.04 LTS (Resolute Raccoon)** / `Linux 7.0.0-29-generic` (x86_64) |
| **Host Toolchain** | GCC 15.2.0 / G++ 15.2.0, CMake 4.2.3, Ninja 1.12.1, Python 3.14, `uv` 0.x |

---

## 🚀 Quick Start

### 1. Automated Zero-Install (Recommended)
On a fresh machine without cloning the repository, run this single command to automatically install dependencies, clone the repo, provision Python 3.14 virtualenv, and build:

```bash
# Complete LLM Inference & Fine-Tuning Stack (~30 min build):
curl -fsSL https://raw.githubusercontent.com/analogbox/TheRock/main/bootstrap.sh | bash -s -- --preset llm --python 3.14
```

### 2. Manual / Local Workflow
If you already cloned the repository locally:

```bash
# 1. Build desired preset (e.g. LLM Stack)
./therock-env build --preset llm --python 3.14

# 2. Activate environment & verify GPU
source ~/virtualenv/therock-7.14/py314-llm/.venv/bin/activate
rocminfo   # Shows AMD Radeon 8060S / gfx1151
```

---

## 🍕 Workload Presets

Instead of waiting 5+ hours for 50+ unused components, select targeted packages:

| Workload Tier | Preset | Aliases | Included Components | Recommended For | Build Time |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Tier 1 (AI / LLM)** | **`llm`** | `lora`, `finetuning` | HIP + `rocBLAS` + `hipBLASLt` + `rocPRIM` + `hipTensor` + `AMD Mesa (RADV Vulkan)` | **vLLM, llama.cpp (HIP & Vulkan), Ollama, MLC-LLM, LoRA/QLoRA Fine-Tuning** | **~30 min** |
| **Tier 1 (AI / LLM)** | **`ai-full`** | `ai`, `training`, `pytorch` | `llm` stack + `MIOpen` (CK) + `RCCL` + `hipDNN` | Full PyTorch training from scratch, CNN/Vision, Stable Diffusion | ~4 hours |
| **Tier 2 (Media)** | **`vulkan-media`** | `vulkan`, `media` | AMD Mesa (RADV Vulkan) + `rocDecode` + `rocJPEG` | Vulkan graphics, 4K/8K video decode, llama.cpp (Vulkan) | **~25 min** |
| **Tier 3 (Engine)** | **`core-hip`** | `hip`, `minimal` | AMD Clang 23 + HIP Runtime + AMDSMI + `rocminfo` | Minimal C++/HIP kernel development & testing | **~20 min** |
| **Tier 3 (Math)** | **`math-hpc`** | `math`, `scientific` | `rocBLAS` + `rocFFT` + `rocSOLVER` + `rocSPARSE` + `rocALUTION` | FFT signal processing, matrix solvers, simulations | ~1.5 hours |
| **Tools** | **`profiler`** | `profiler` | `rocprofiler-sdk`, `rocprofiler-systems`, `rocgdb` | GPU performance tracing & interactive debugging | ~40 min |
| **Monolithic** | **`full`** | `full` | Complete ROCm stack (all 50+ libraries) | Monolithic distribution release build | ~5 hours |

---

## 🎨 Custom Component Builds (`--components` & `--with-*`)

### 1. Build a Custom Environment (`--components`)
Select precisely the components you want:

```bash
# Example: Build custom environment with BLAS, Vulkan, MIOpen, and Fast Fourier Transforms (rocFFT)
./bootstrap.sh --components blas,vulkan,miopen,fft --python 3.14

# Or via therock-env directly
./therock-env build --preset custom --components blas,vulkan,profiler --python 3.14
```

**Supported Component Tokens**:
* `blas` (`rocblas`, `hipblas`, `hipblaslt`): Matrix multiplication & GEMM kernels.
* `vulkan` (`mesa`, `radv`): AMD Mesa Vulkan runtime & shader compiler.
* `miopen` (`ck`): Deep learning convolutions & attention operators.
* `rccl`: Multi-GPU collective communications.
* `fft` (`rocfft`): Fast Fourier Transforms (1D, 2D, 3D).
* `solver` (`rocsolver`): Dense linear system & eigenvalue solvers.
* `sparse` (`rocsparse`): Sparse matrix operations.
* `media` (`rocdecode`, `rocjpeg`): 4K/8K hardware video decoding & JPEG codec.
* `profiler` (`rocgdb`): Profiler, tracer, and GDB debugger.

### 2. On-The-Fly Mix-and-Match Flags (`--with-*`)

You can customize any preset on-the-fly by adding or removing individual components:

```bash
# Example 1: Build LLM preset with GPU Profiler & GDB Debugger
./therock-env build --preset llm --python 3.14 --with-profiler

# Example 2: Build LLM preset with MIOpen for CNN/Vision hybrid tasks
./therock-env build --preset llm --python 3.14 --with-miopen

# Example 3: Build HIP foundation engine with Fast Fourier Transforms (rocFFT)
./therock-env build --preset core-hip --python 3.14 --with-fft

# Example 4: Build LLM preset with ccache compiler acceleration
./bootstrap.sh --preset llm --python 3.14 --with-ccache
```

**Supported Flags**:
* `--with-miopen`: Adds MIOpen & Composable Kernel.
* `--with-rccl`: Adds multi-GPU collective communications (RCCL).
* `--with-profiler`: Adds `rocprofv3`, `rocprofiler-sdk`, and `rocgdb`.
* `--with-fft`: Adds `rocFFT` math library.
* `--with-media` / `--with-vulkan`: Adds AMD Mesa, `rocDecode`, and `rocJPEG`.
* `--with-ccache`: Enables compiler caching (auto-installs `ccache` via `apt` if missing).
* `--without-blas`: Excludes BLAS math libraries.

---

## 🔍 Under the Hood: Virtual Environment Isolation

```
[1. Source Code Factory]
  ~/virtualenv/therock-7.14/TheRock/ (Shared Git Source & Build Scripts)
             │
             ▼ (Compiles targeted preset in ~25-35 min)
[2. Build Output Artifacts]
  ~/virtualenv/therock-7.14/py314-llm/build/dist/rocm/ (Compiled ROCm Libraries)
             │
             ▼ (Automatic Wrapper Injection & Path Linking)
[3. Hermetically Installed inside Virtualenv!]
  ~/virtualenv/therock-7.14/py314-llm/.venv/bin/
             ├── python3 & pip
             ├── rocminfo     ← (Auto-injected wrapper for this specific build!)
             ├── hipcc        ← (Auto-injected wrapper!)
             ├── amdclang     ← (Auto-injected wrapper!)
             └── rocm-smi     ← (Auto-injected wrapper!)
```

* **No System Pollution**: All 220+ ROCm executable wrappers and environment variables (`ROCM_PATH`, `HIP_DEVICE_LIB_PATH`) are installed directly inside the virtual environment (`.venv/bin/`). System `/opt/rocm` is 100% bypassed.
* **Side-by-Side Isolation**: Python 3.14 (`py314-llm`) and Python 3.13 (`py313-llm`) live in isolated sibling folders sharing the source repo without duplicating git history.
* **1-Second Switching**: Activating any `.venv` switches the entire ROCm toolchain instantly. Running `deactivate` reverts to clean Ubuntu.

---

## 📁 Managing Environments & Builds

```bash
# 1. List all detected Python virtual environments
./therock-env list-envs

# 2. List all completed ROCm build trees and disk usage
./therock-env list-builds

# 3. Hermetically install an existing build into a virtual environment
./therock-env install-to-venv --build-dir build_py314_llm --python 3.14
```

---

## 📚 Technical Guides & References

* [Upstream Sync & Rebase Guide](docs/UPSTREAM_SYNC_AND_REBASE_GUIDE.md): Step-by-step workflow for upgrading your fork when AMD releases new upstream ROCm tags.
* [GCC 15 & Ubuntu 26.04 Technical Porting Guide](docs/GCC15_UBUNTU2604_PORTING_GUIDE.md): Deep-dive into GCC 15 `<version>` migration, CMake 4.x deferred dependency providers, and Meson symbol version scripts.
* [Development Guide](docs/development/development_guide.md): Developer architecture guide.
* [Supported GPUs](SUPPORTED_GPUS.md): AMD GPU architecture roadmap.
* [CONTRIBUTING.md](CONTRIBUTING.md): Contributing guidelines.
