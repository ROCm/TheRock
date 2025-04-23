# TheRock

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Description

TheRock (The HIP Environment and ROCm Kit) is a lightweight open source build platform for HIP and ROCm. The project is currently in an **early preview state** but is under active development and welcomes contributors. Come try us out! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more info.

Currently, the platform offers developers the option to build HIP and ROCm from source. Additionally, a GitHub actions pipeline will offer a nightly build with compiled ROCm/HIP software available in S3 and in the GitHub releases section.

## Table of Contents

- [Installation From Source](#installation-from-source)
- [Configuration](#configuration)
- [Usage](#usage)
- [Tests](#tests)
- [Development Manuals](#development-manuals)

## Installation From Source

### Ubuntu

```bash
# Clone the repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock

# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install gfortran git-lfs ninja-build cmake g++ pkg-config xxd patchelf automake
python ./build_tools/fetch_sources.py # Downloads submodules and applies patches
```

### Windows

```bash
# Clone the repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock

# Install dependencies
python3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> [!WARNING]
> Windows support is still early in development. Not all subprojects or packages build for Windows yet.

See [windows_support.md](./docs/development/windows_support.md).

```bash
python ./build_tools/fetch_sources.py  # Downloads submodules and applies patches
```

## Configuration

The build can be customized through cmake feature flags.

**Required Flags:**

- `-DTHEROCK_AMDGPU_FAMILIES=`

  or

- `-DTHEROCK_AMDGPU_TARGETS=`

Note: *Not all family and targets are currently supported. See [therock_amdgpu_targets.cmake](cmake/therock_amdgpu_targets.cmake) file for available options*

**Optional Flags**

By default, the project builds everything available. The following group flags
allow enable/disable of selected subsets:

- `-DTHEROCK_ENABLE_ALL=OFF`: Disables all optional components.
- `-DTHEROCK_ENABLE_CORE=OFF`: Disables all core components.
- `-DTHEROCK_ENABLE_COMM_LIBS=OFF`: Disables all communication libraries.
- `-DTHEROCK_ENABLE_MATH_LIBS=OFF`: Disables all math libraries.
- `-DTHEROCK_ENABLE_ML_LIBS=OFF`: Disables all ML libraries.

Individual features can be controlled separately (typically in combination with
`-DTHEROCK_ENABLE_ALL=OFF` or `-DTHEROCK_RESET_FEATURES=ON` to force a
minimal build):

- `-DTHEROCK_ENABLE_COMPILER=ON`: Enables the GPU+host compiler toolchain.
- `-DTHEROCK_ENABLE_HIPIFY=ON`: Enables the hipify tool.
- `-DTHEROCK_ENABLE_CORE_RUNTIME=ON`: Enables the core runtime components and tools.
- `-DTHEROCK_ENABLE_HIP_RUNTIME=ON`: Enables the HIP runtime components.
- `-DTHEROCK_ENABLE_RCCL=ON`: Enables RCCL.
- `-DTHEROCK_ENABLE_PRIM=ON`: Enables the PRIM library.
- `-DTHEROCK_ENABLE_BLAS=ON`: Enables the BLAS libraries.
- `-DTHEROCK_ENABLE_RAND=ON`: Enables the RAND libraries.
- `-DTHEROCK_ENABLE_SOLVER=ON`: Enables the SOLVER libraries.
- `-DTHEROCK_ENABLE_SPARSE=ON`: Enables the SPARSE libraries.
- `-DTHEROCK_ENABLE_MIOPEN=ON`: Enables MIOpen.

Enabling any features will implicitly enable its *minimum* dependencies. Some
libraries (like MIOpen) have a number of *optional* dependencies, which must
be enabled manually if enabling/disabling individual features.

A report of enabled/disabled features and flags will be printed on every
CMake configure.

## Usage

To build ROCm/HIP:

```bash
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu
cmake --build build
```

To build with cacheing:

```bash
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .
cmake --build build
```

## Tests

Project-wide testing can be controlled with the standard CMake `-DBUILD_TESTING=ON|OFF` flag. This gates both setup of build tests and compilation of installed testing artifacts.

Tests of the integrity of the build are enabled by default and can be run
with ctest:

```
ctest --test-dir build
```

Testing functionality on an actual GPU is in progress and will be documented
separately.

## Development Manuals

- [Contribution Guidelines](CONTRIBUTING.md): Documentation for the process of contributing to this project including a quick pointer to its governance.
- [Development Guide](docs/development/development_guide.md): Documentation on how to use TheRock as a daily driver for developing any of its contained ROCm components (i.e. vs interacting with each component build individually).
- [Build System](docs/development/build_system.md): More detailed information about TheRock's build system relevant to people looking to extend TheRock, add components, etc.
- [Git Chores](docs/development/git_chores.md): Procedures for managing the codebase, specifically focused on version control, upstream/downstream, etc.
- [Dependencies](docs/development/dependencies.md): Further specifications on ROCm-wide standards for depending on various components.
- [Build Containers](docs/development/build_containers.md): Further information about containers used for building TheRock on CI.
- [Build Artifacts](docs/development/artifacts.md): Documentation about the outputs of the build system.
- [Releases Page](RELEASES.md): Documentation for how to leverage our build artifacts.
- [Roadmap for Support](ROADMAP.md): Documentation for our prioritized roadmap to support AMD GPUs.

## Provisioning TheRock 🪨

In order to provision TheRock using either a developer/automated nightly release or a specific CI runner build, use the `build_tool/provision.py` script.

Provisioning script setup:

- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `python build_tools/provision.py --help`

Examples:

- `python build_tools/provision.py --run-id 14474448215 --amdgpu-family gfx94X-dcgpu`: Downloads the gfx94X S3 artifacts from GitHub CI workflow run 14474448215 to the default output directory `therock-build`

<br/>

- `python build_tools/provision.py --release latest --amdgpu-family gfx110X-dgpu --output-dir build`: Downloads the latest gfx110X artifacts from GitHub release tag `nightly-release` to the specified output directory `build`

<br/>

- `python build_tools/provision.py --release 6.4.0rc20250416 --amdgpu-family gfx110X-dgpu --output-dir build`: Downloads the version `6.4.0rc20250416` gfx110X artifacts from GitHub release tag `nightly-release` to the specified output directory `build`

<br/>

- `python build_tools/provision.py --release 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9 --amdgpu-family gfx120X-all`: Downloads the version `6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9` gfx120X artifacts from GitHub release tag `dev-release` to the default output directory `therock-build`

Select your AMD GPU family from this file [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81)
