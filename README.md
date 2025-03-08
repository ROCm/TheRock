# TheRock

The HIP Environment and ROCm Kit - A lightweight open source build system for HIP and ROCm

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Install Deps

By default on Linux, the project builds with -DTHEROCK_BUNDLE_SYSDEPS=ON, which
builds low-level system libraries from source and private links to them. This
requires some additional development tools, which are included below.

## Common

```
pip install -r requirements.txt
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

## Sanity Checks

Tests of the integrity of the build are enabled by default and can be run
with ctest:

```
ctest --test-dir build
```

Testing functionality on an actual GPU is in progress and will be documented
separately.
