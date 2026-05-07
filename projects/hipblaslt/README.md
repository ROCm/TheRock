# hipBLASLt

hipBLASLt is a library that provides general matrix-matrix operations. It has a flexible API that extends
functionalities beyond a traditional BLAS library, such as adding flexibility to matrix data layouts, input
types, compute types, and algorithmic implementations and heuristics.

> [!NOTE]
> The published hipBLASLt documentation is available at [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipBLASLt/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

hipBLASLt uses the HIP programming language with an underlying optimized generator as its backend
kernel provider.

After you specify a set of options for a matrix-matrix operation, you can reuse these for different
inputs. The general matrix-multiply (GEMM) operation is performed by the `hipblasLtMatmul` API.

The equation is:

```math
D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)
```

Where *op( )* refers to in-place operations, such as transpose and non-transpose, and *alpha* and
*beta* are scalars.

The activation function supports GELU, ReLU, and Swish (SiLU). the bias vector matches matrix D rows and
broadcasts to all D columns.

For the supported data types, see
[Supported data types](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/data-type-support.html).

## Getting started

> [!NOTE]
> The steps in this section are intended to help users get started building hipblaslt. However, hipblaslt is a complex
> project and it is recommended to consult the [hipBLASLt installation documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/installation.html)
> for complete setup and installation instructions.

### Setup

The simplest option is to clone all of [rocm-libraries](https://github.com/ROCm/rocm-libraries) and navigate to the hipblaslt project:

```bash
# Clone rocm-libraries
git clone https://github.com/ROCm/rocm-libraries.git
# Go to hipBLASLt directory
cd rocm-libraries/projects/hipBLASLt
```

For a shorter download process, use sparse checkout to only clone the hipblaslt project:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt
git checkout develop # or the branch you are starting from
```

### Configure and build

hipBLASLt provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. For details on all configuration options, see the [Options](#options) section.

This section provides usage examples on how to configure, build and install hipBLASLt using various supported methods.
It assumes the user has a ROCm installation (conventionally installed to `/opt/rocm`), Python 3.8 or newer, 
and a CMake version greater than or equal to the `cmake_minimum_required` defined at [CMakeLists.txt](CMakeLists.txt#L4).

#### Using CMake presets

> [!NOTE]
> When using presets, assumptions are made about search paths, built-in CMake variables, and output directories. 
> Consult [CMakePresets.json](./CMakePresets.json) to understand which variables are set, 
> or refer to [Using CMake variables directly](#using-cmake-variables-directly) for a fully custom configuration.

**Full build for all architectures**

```bash
# show available presets
cmake --list-presets
# configure
cmake --preset default:release
# build
cmake --build build --parallel
# install
cmake --install build
```

**Building GEMM libraries**

```bash
# configure
cmake --preset gemm-libs
# build
cmake --build build --parallel
```

#### Using CMake variables directly

**Full build for gfx950**

```bash
# configure
cmake -B build -S .                                  \
      -D CMAKE_BUILD_TYPE=Release                    \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
      -D CMAKE_PREFIX_PATH=/opt/rocm                 \         
      -D GPU_TARGETS=gfx950
# build
cmake --build build --parallel
```

#### Using the installation script

Refer to the available build options using `./install.sh --help`:

```bash
# Command line options:
#   -h|--help         - prints help message
#   -i|--install      - install after build
#   -d|--dependencies - install build dependencies
#   -c|--clients      - build library clients too (combines with -i & -d)
#   -g|--debug        - build with debug flag
./install.sh -idc
# build for gfx950 only without installation
./install.sh -c -a gfx950
```

> [!NOTE]
> To build hipBLASLt for ROCm <= 6.2, pass the `--legacy_hipblas_direct` flag to `install.sh`.

### Options

> [!NOTE]
> When using the install script these variables are either hardcoded or set via its command line options.

*CMake options*:
* `CMAKE_BUILD_TYPE`: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* `CMAKE_INSTALL_PREFIX`: Base installation directory
* `CMAKE_PREFIX_PATH`: Find package search path (consider setting to ``$ROCM_PATH``)

*Project wide options*:

* `HIPBLASLT_ENABLE_BLIS`: Enable BLIS support (default `ON`)
* `HIPBLASLT_ENABLE_HIP`: Use the HIP runtime (default `ON`)
* `HIPBLASLT_ENABLE_LLVM`: Use msgpack for parsing configuration files (default `OFF`)
* `HIPBLASLT_ENABLE_MSGPACK` Use msgpack for parsing configuration files (default `ON`)
* `HIPBLASLT_ENABLE_OPENMP`: "Use OpenMP to improve performance (default `ON`)
* `HIPBLASLT_ENABLE_ROCROLLER:` Use RocRoller library (default `OFF`)
* `GPU_TARGETS:` Semicolon separated list of gfx targets to build

*hipblaslt options*

* `HIPBLASLT_ENABLE_HOST`: Enables generation of host library (default: `ON`)
* `HIPBLASLT_ENABLE_DEVICE`: Enables generation of device libraries (default: `ON`)
* `HIPBLASLT_ENABLE_CLIENT`: Enables generation of client applications (default: `ON`)
* `HIPBLASLT_BUILD_TESTING:` Build hipblaslt client tests (default `ON`)
* `HIPBLASLT_ENABLE_SAMPLES:` Build client samples (default `ON`)
* `HIPBLASLT_ENABLE_LAZY_LOAD` Enable lazy loading of runtime code oject files to reduce init costs (default: `ON`)

*tensilelite options*

* `TENSILELITE_ENABLE_HOST`: Enables generation of tensilelite host (default: `ON`)
* `TENSILELITE_ENABLE_CLIENT`: Enables generation of tensilelite client application (default: `ON`)
* `TENSILELITE_ENABLE_AUTOBUILD`: Generate wrapper scripts that set PYTHONPATH and trigger rebuilds of rocisa (default: `OFF`)
* `TENSILELITE_BUILD_TESTING`: Build tensilelite host library tests (default: `OFF`)

*Device libraries options:*

* `TENSILELITE_BUILD_PARALLEL_LEVEL` Number of CPU cores to use for building device libraries (will use nproc if unset)
* `TENSILELITE_KEEP_BUILD_TMP` OFF CACHE STRING Keep temporary build directory for device libraries (default: see below)
* `TENSILELITE_LIBRARY_FORMAT` Format of master solution library files (msgpack or yaml) (default: see below)
* `TENSILELITE_ASM_DEBUG` Keep debug information for built code objects (default: see below)
* `TENSILELITE_LOGIC_FILTER` Cutomsized logic filter, default is *, i.e. all logics (default: see below)
* `TENSILELITE_NO_COMPRESS` Do not compress device code object files (default: see below)
* `TENSILELITE_EXPERIMENTAL` Process experimental logic files (default: see below)
* `HIPBLASLT_LIBLOGIC_PATH` Path to library logic files (will use 'library' if unset) (default: `Off`)
* `HIPBLASLT_TENSILE_LIBPATH` Path to output the device gemm libraries (default: `build/Tensile`)

> [!NOTE]
> To determine defaults for the `TensileCreateLibrary` command generated when building the device
> libraries, run `Tensile/bin/TensileCreateLibrary --help` from the tensilelite directory.

> [!NOTE]
> Refer to the tensilelite [README](./tensilelite/README.md) for instructions on building for the tensile workflow.

## Unit tests

All unit tests are located in `build/release/clients/`. To build these tests, you must build
hipBLASLt with `--clients`.

You can find more information at the following links:

* [hipblaslt-test](clients/gtest/README.md)
* [hipblaslt-bench](clients/benchmarks/README.md)

## Documentation

Full documentation for hipBLASLt is available at
[rocm.docs.amd.com/projects/hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

Run the steps below to build documentation locally.

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Alternatively, build with CMake:

```bash
cmake -DBUILD_DOCS=ON ...
```

## Contribute

If you want to submit an issue, you can do so on
[GitHub](https://github.com/ROCm/rocm-libraries/issues).

To contribute to our repository, you can create a GitHub pull request.
