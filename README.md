# TheRock - Custom gfx103X Build

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

TheRock (The HIP Environment and ROCm Kit) is a lightweight open source build platform for HIP and ROCm. This is a **custom branch** with ROCm 7.11 optimized for AMD RDNA2 gfx103X GPUs. For the official upstream project, see [ROCm/TheRock](https://github.com/ROCm/TheRock).

______________________________________________________________________

## 🚀 Custom Build: ROCm 7.11 for gfx103X GPUs (hashcat branch)

This branch (`hashcat/rocm-7.11-gfx103X`) contains a **custom ROCm 7.11 build** specifically optimized for **AMD RDNA2 gfx103X GPUs**, with extensive testing on the **AMD Radeon RX 6700 XT (gfx1031)**.

### What's Different in This Build

- **ROCm 7.11** custom build from TheRock main
- **Native gfx103X support** (gfx1030, gfx1031, gfx1032, gfx1035, gfx1036)
- **AI/LLM workload optimization** including:
  - llama.cpp server integration with ROCm backend
  - Ollama with ROCm support
  - Open Interpreter configuration and best practices
  - Automated Python package update tooling
- **Comprehensive documentation** reorganized into topic-based guides
- **Real-world testing** on Fedora 43 with AMD RX 6700 XT

### Quick Start for This Build

```bash
# Clone this branch
git clone -b hashcat/rocm-7.11-gfx103X https://github.com/tlee933/TheRock.git
cd TheRock

# Setup virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Fetch sources
python3 ./build_tools/fetch_sources.py

# Build for gfx103X family (includes all RDNA2 consumer GPUs)
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all
cmake --build build

# Or build for specific target (e.g., RX 6700 XT)
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1031
cmake --build build
```

### Supported gfx103X GPUs in This Build

| Target  | GPU Model              | Type | Status       |
| ------- | ---------------------- | ---- | ------------ |
| gfx1030 | AMD RX 6800 / XT       | dGPU | ✅ Supported |
| gfx1031 | AMD RX 6700 XT         | dGPU | ✅ Tested    |
| gfx1032 | AMD RX 6600            | dGPU | ✅ Supported |
| gfx1035 | AMD Radeon 680M Laptop | iGPU | ✅ Supported |
| gfx1036 | AMD Raphael Integrated | iGPU | ✅ Supported |

### Documentation for This Build

This branch includes **comprehensive guides** in the reorganized `docs/` directory:

- **[Documentation Index](docs/README.md)** - Complete guide to all documentation
- **[Custom Build Guide](docs/guides/custom-build.md)** - Building with custom configs
- **[Package Updates Guide](docs/guides/package-updates.md)** - Safe Python package updates
- **[Open Interpreter Best Practices](docs/guides/open-interpreter-best-practices.md)** - OI usage tips
- **[SELinux + ROCm Setup](docs/guides/selinux-rocm-setup.md)** - SELinux configuration
- **[Browser Search Fix](docs/troubleshooting/browser-search-fix.md)** - Fixing OI browser.search()

### Build & Test Environment

This build has been developed and tested on:

**System:**

- **OS:** Fedora 43 Linux (kernel 6.17.10)
- **GPU:** AMD Radeon RX 6700 XT (12GB VRAM, gfx1031/RDNA2)
- **ROCm:** 7.11 (custom build from TheRock)

**Build Tools:**

- **Python:** 3.14.0 (in venv)
- **CMake:** 3.31.6
- **Ninja:** 1.13.1
- **GCC:** 15.2.1 (Red Hat 15.2.1-4)
- **Git:** Latest from Fedora 43 repos

**Tested Workloads:**

- llama.cpp server with ROCm backend
- Ollama with ROCm support
- PyTorch with ROCm
- Open Interpreter 0.4.3

### Modifications & Enhancements

**Build System:**

- gfx103X target family support (already in upstream, validated here)
- Build scripts and optimization flags tested for RDNA2

**AI/LLM Infrastructure:**

- llama-server systemd service configuration
- Ollama integration with ROCm backend
- Open Interpreter profiles and custom instructions
- Python package update automation with rollback support

**Documentation:**

- Reorganized into `docs/guides/`, `docs/troubleshooting/`, `docs/custom/`
- Added comprehensive user guides for AI/ML workflows
- Real-world testing notes and workarounds

**Development Tools:**

- Automated package update script with testing and rollback
- Custom shell aliases and environment setup
- System monitoring and performance tuning utilities

______________________________________________________________________

## Features

TheRock includes:

- A CMake super-project for HIP and ROCm source builds
- Support for building PyTorch with ROCm from source
  - [JAX support](https://github.com/ROCm/TheRock/issues/247) and other external project builds are in the works!
- Linux distribution support (tested on Fedora 43, Ubuntu also supported)
- Tools for developing individual ROCm components
- Comprehensive build and testing infrastructure

## Building from source

We keep the following instructions for recent, commonly used operating system
versions. Most build failures are due to minor operating system differences in
dependencies and project setup. Refer to the
[Environment Setup Guide](docs/guides/environment-setup.md) for contributed
instructions and configurations for alternatives.

> [!TIP]
> While building from source offers the greatest flexibility,
> [installing from releases](#installing-from-releases) in supported
> configurations is often faster and easier.

> [!IMPORTANT]
> Frequent setup and building problems and their solutions can be found in section [Common Issues](docs/guides/environment-setup.md#common-issues).

### Setup - Fedora 43 (Tested Configuration)

This is the configuration used to build and test this custom branch:

```bash
# Install Fedora dependencies
sudo dnf install gfortran git ninja-build cmake gcc gcc-c++ pkg-config xxd patchelf automake libtool python3-devel mesa-libEGL-devel texinfo bison flex

# Clone this branch
git clone -b hashcat/rocm-7.11-gfx103X https://github.com/tlee933/TheRock.git
cd TheRock

# Init python virtual environment and install python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download submodules and apply patches
python3 ./build_tools/fetch_sources.py
```

### Setup - Ubuntu (24.04)

> [!TIP]
> `dvc` is used for version control of pre-compiled MIOpen kernels.
> `dvc` is not a hard requirement, but it does reduce compile time.
> `snap install --classic dvc` can be used to install on Ubuntu.
> Visit the [DVC website](https://dvc.org/doc/install/linux) for other installation methods.

```bash
# Install Ubuntu dependencies
sudo apt update
sudo apt install gfortran git ninja-build cmake g++ pkg-config xxd patchelf automake libtool python3-venv python3-dev libegl1-mesa-dev texinfo bison flex

# Clone this branch (or upstream)
git clone -b hashcat/rocm-7.11-gfx103X https://github.com/tlee933/TheRock.git
cd TheRock

# Init python virtual environment and install python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download submodules and apply patches
python3 ./build_tools/fetch_sources.py
```

### Build configuration

The build can be customized through cmake feature flags.

#### Required configuration flags

- `-DTHEROCK_AMDGPU_FAMILIES=`

  or

- `-DTHEROCK_AMDGPU_TARGETS=`

> [!NOTE]
> Not all family and targets are currently supported.
> See [therock_amdgpu_targets.cmake](cmake/therock_amdgpu_targets.cmake) file
> for available options.

#### Discovering available targets on your system

In case you don't have an existing ROCm/HIP installation from which you can run any of these tools:

| Tool                    | Platform |
| ----------------------- | -------- |
| `amd-smi`               | Linux    |
| `rocm-smi`              | Linux    |
| `rocm_agent_enumerator` | Linux    |
| `hipinfo`               | Windows  |
| `offload-arch`          | Both     |

You can install the `rocm` Python package for any architecture inside a venv and run `offload-arch` from there:

1. `python build_tools/setup_venv.py --index-name nightly --index-subdir gfx110X-all --packages rocm .tmpvenv`
1. `.tmpvenv/bin/offload-arch` on Linux, `.tmpvenv\Scripts\offload-arch` on Windows
1. `rm -rf .tmpvenv`

#### Optional configuration flags

By default, the project builds everything available. The following group flags
enable/disable selected subsets:

| Group flag                       | Description                          |
| -------------------------------- | ------------------------------------ |
| `-DTHEROCK_ENABLE_ALL=OFF`       | Disables all optional components     |
| `-DTHEROCK_ENABLE_CORE=OFF`      | Disables all core components         |
| `-DTHEROCK_ENABLE_COMM_LIBS=OFF` | Disables all communication libraries |
| `-DTHEROCK_ENABLE_MATH_LIBS=OFF` | Disables all math libraries          |
| `-DTHEROCK_ENABLE_ML_LIBS=OFF`   | Disables all ML libraries            |
| `-DTHEROCK_ENABLE_PROFILER=OFF`  | Disables profilers                   |
| `-DTHEROCK_ENABLE_DC_TOOLS=OFF`  | Disables data center tools           |

Individual features can be controlled separately (typically in combination with
`-DTHEROCK_ENABLE_ALL=OFF` or `-DTHEROCK_RESET_FEATURES=ON` to force a
minimal build):

| Component flag                      | Description                                   |
| ----------------------------------- | --------------------------------------------- |
| `-DTHEROCK_ENABLE_COMPILER=ON`      | Enables the GPU+host compiler toolchain       |
| `-DTHEROCK_ENABLE_HIPIFY=ON`        | Enables the hipify tool                       |
| `-DTHEROCK_ENABLE_CORE_RUNTIME=ON`  | Enables the core runtime components and tools |
| `-DTHEROCK_ENABLE_HIP_RUNTIME=ON`   | Enables the HIP runtime components            |
| `-DTHEROCK_ENABLE_OCL_RUNTIME=ON`   | Enables the OpenCL runtime components         |
| `-DTHEROCK_ENABLE_ROCPROFV3=ON`     | Enables rocprofv3                             |
| `-DTHEROCK_ENABLE_ROCPROFSYS=ON`    | Enables rocprofiler-systems                   |
| `-DTHEROCK_ENABLE_RCCL=ON`          | Enables RCCL                                  |
| `-DTHEROCK_ENABLE_PRIM=ON`          | Enables the PRIM library                      |
| `-DTHEROCK_ENABLE_BLAS=ON`          | Enables the BLAS libraries                    |
| `-DTHEROCK_ENABLE_RAND=ON`          | Enables the RAND libraries                    |
| `-DTHEROCK_ENABLE_SOLVER=ON`        | Enables the SOLVER libraries                  |
| `-DTHEROCK_ENABLE_SPARSE=ON`        | Enables the SPARSE libraries                  |
| `-DTHEROCK_ENABLE_MIOPEN=ON`        | Enables MIOpen                                |
| `-DTHEROCK_ENABLE_MIOPEN_PLUGIN=ON` | Enables MIOpen_plugin                         |
| `-DTHEROCK_ENABLE_HIPDNN=ON`        | Enables hipDNN                                |
| `-DTHEROCK_ENABLE_ROCWMMA=ON`       | Enables rocWMMA                               |
| `-DTHEROCK_ENABLE_RDC=ON`           | Enables ROCm Data Center Tool (Linux only)    |

> [!TIP]
> Enabling any features will implicitly enable their *minimum* dependencies. Some
> libraries (like MIOpen) have a number of *optional* dependencies, which must
> be enabled manually if enabling/disabling individual features.

> [!TIP]
> A report of enabled/disabled features and flags will be printed on every
> CMake configure.

By default, components are built from the sources fetched via the submodules.
For some components, external sources can be used instead.

| External source settings                        | Description                                    |
| ----------------------------------------------- | ---------------------------------------------- |
| `-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=OFF`  | Use external composable-kernel source location |
| `-DTHEROCK_USE_EXTERNAL_RCCL=OFF`               | Use external rccl source location              |
| `-DTHEROCK_USE_EXTERNAL_RCCL_TESTS=OFF`         | Use external rccl-tests source location        |
| `-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=<PATH>` | Path to composable-kernel sources              |
| `-DTHEROCK_RCCL_SOURCE_DIR=<PATH>`              | Path to rccl sources                           |
| `-DTHEROCK_RCCL_TESTS_SOURCE_DIR=<PATH>`        | Path to rccl-tests sources                     |

Further flags allow to build components with specific features enabled.

| Other flags                | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| `-DTHEROCK_ENABLE_MPI=OFF` | Enables building components with Message Passing Interface (MPI) support |

> [!NOTE]
> Building components with MPI support, currently requires MPI to be
> pre-installed until [issue #1284](https://github.com/ROCm/TheRock/issues/1284)
> is resolved.

### CMake build usage

For workflows that demand frequent rebuilds, it is _recommended to build it with ccache_ enabled to speed up the build.
See instructions in the next section for [Linux](#ccache-usage-on-linux) and [Windows](#ccache-usage-on-windows).

Otherwise, ROCm/HIP can be configured and build with just the following commands:

```bash
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all
cmake --build build
```

#### CCache usage on Linux

To build with the [ccache](https://ccache.dev/) compiler cache:

- You must have a recent ccache (>= 4.11 at the time of writing) that supports
  proper caching with the `--offload-compress` option used for compressing
  AMDGPU device code.
- `export CCACHE_SLOPPINESS=include_file_ctime` to support hard-linking
- Proper setup of the `compiler_check` directive to do safe caching in the
  presence of compiler bootstrapping
- Set the C/CXX compiler launcher options to cmake appropriately.

Since these options are very fiddly and prone to change over time, we recommend
using the `./build_tools/setup_ccache.py` script to create a `.ccache` directory
in the repository root with hard coded configuration suitable for the project.

Example:

```bash
# Any shell used to build must eval setup_ccache.py to set environment
# variables.
eval "$(./build_tools/setup_ccache.py)"
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  .

cmake --build build
```

#### CCache usage on Windows

We are still investigating the exact proper options for ccache on Windows and
do not currently recommend that end users enable it.

### Running tests

Project-wide testing can be controlled with the standard CMake `-DBUILD_TESTING=ON|OFF` flag. This gates both setup of build tests and compilation of installed testing artifacts.

Tests of the integrity of the build are enabled by default and can be run
with ctest:

```
ctest --test-dir build
```

Testing functionality on an actual GPU is in progress and will be documented
separately.

## Development manuals

- **[Documentation Index](docs/README.md)**: Complete catalog of all documentation organized by topic
- **[Contribution Guidelines](CONTRIBUTING.md)**: Documentation for the process of contributing to this project including a quick pointer to its governance
- **[Development Guide](docs/development/development_guide.md)**: Documentation on how to use TheRock as a daily driver for developing any of its contained ROCm components
- **[Build System](docs/development/build_system.md)**: More detailed information about TheRock's build system relevant to people looking to extend TheRock, add components, etc
- **[Environment Setup Guide](docs/guides/environment-setup.md)**: Comprehensive guide for setting up a build environment, known workarounds, and other operating specific information
- **[Git Chores](docs/development/git_chores.md)**: Procedures for managing the codebase, specifically focused on version control, upstream/downstream, etc
- **[Dependencies](docs/development/dependencies.md)**: Further specifications on ROCm-wide standards for depending on various components
- **[Build Containers](docs/development/build_containers.md)**: Further information about containers used for building TheRock on CI
- **[Build Artifacts](docs/development/artifacts.md)**: Documentation about the outputs of the build system
- **[Releases Page](RELEASES.md)**: Documentation for how to leverage our build artifacts
- **[Roadmap for Support](ROADMAP.md)**: Documentation for our prioritized roadmap to support AMD GPUs
- **[Supported GPUs](docs/supported-gpus.md)**: List of all AMD GPUs supported by TheRock
