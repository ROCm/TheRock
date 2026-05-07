# hipBLAS-common

## Introduction

hipBLAS-common is a header-only library that provides common files for hipBLAS and hipBLASLt.

## Configure and build

hipBLAS-common provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. Below are usage examples to get started.

### Using CMake presets

> [!NOTE]
> When using presets, assumptions are made about search paths, built-in CMake variables, and output directories. Consult [CMakePresets.json](./CMakePresets.json) to understand which variables are set, or refer to [Using CMake variables directly](#using-cmake-variables-directly) for a fully custom configuration.

```bash
# show available presets
cmake --list-presets
# configure
cmake --preset default:release
# build
cmake --build build
# install
cmake --install build
```

### Using CMake variables directly

```bash
# configure
cmake -B build -S . -D CMAKE_INSTALL_PREFIX=<preferred installation path>
# build
cmake --build build
```

### Using rmake script

```bash
python3 ./rmake.py --install
```

## CMake targets

hipBLAS-common generates the following targets for consumption in internal and external projects.

* `roc::hipblas-common`
