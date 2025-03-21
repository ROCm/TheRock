# TheRock

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

The HIP Environment and ROCm Kit - A lightweight open source build system for HIP and ROCm.

We are currently in an **early preview state** but welcome contributors. Come try us out!
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more info.

If you're looking to quickly see how far along we are, check the [Releases Page](RELEASES.md).

# Install Deps

By default on Linux, the project builds with -DTHEROCK_BUNDLE_SYSDEPS=ON, which
builds low-level system libraries from source and private links to them. This
requires some additional development tools, which are included below.

## Common

```
pip install CppHeaderParser==2.7.4 meson==1.7.0 PyYAML==6.0.2
```

Python 3.10 also requires `tomli`:

```
pip install tomli==2.2.1
```

## On Ubuntu

Dev tools:

```
sudo apt install gfortran git-lfs ninja-build cmake g++ pkg-config xxd libgtest-dev patchelf automake
```

## On Windows

> [!WARNING]
> Windows support is still early in development. Not all subprojects or packages build for Windows yet.

See [windows_support.md](./docs/development/windows_support.md).

# Checkout Sources

```
python ./build_tools/fetch_sources.py
```

This uses a custom procedure to download submodules and apply patches while
we are transitioning from the [repo](https://source.android.com/docs/setup/reference/repo).
It will eventually be replaced by a normal `git submodule update` command.

This will also apply the patches to the downloaded source files.

# Build

Note that you must specify GPU targets or families to build for with either
`-DTHEROCK_AMDGPU_FAMILIES=` or `-DTHEROCK_AMDGPU_TARGETS=` and will get an
error if there is an issue. Supported families and targets are in the
[therock_amdgpu_targets.cmake](cmake/therock_amdgpu_targets.cmake) file. Not
all combinations are presently supported.

```
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu
# Or if iterating and wishing to cache:
#   cmake -Bbuild -GNinja -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .
cmake --build build
```

## Feature Flags

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

## Testing

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
