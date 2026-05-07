# Composable Kernel

> [!NOTE]
> The published documentation is available at [Composable Kernel](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

The Composable Kernel (CK) library provides a programming model for writing performance-critical
kernels for machine learning workloads across multiple architectures (GPUs, CPUs, etc.). The CK library
uses general purpose kernel languages, such as HIP C++.

CK uses two concepts to achieve performance portability and code maintainability:

* A tile-based programming model
* Algorithm complexity reduction for complex machine learning (ML) operators. This uses an innovative
   technique called *Tensor Coordinate Transformation*.

![ALT](/docs/data/ck_component.png "CK Components")

The current CK library is structured into four layers:

* Templated Tile Operators
* Templated Kernel and Invoker
* Instantiated Kernel and Invoker
* Client API

![ALT](/docs/data/ck_layer.png "CK Layers")

## General information

* [CK supported operations](include/ck/README.md)
* [CK Tile supported operations](include/ck_tile/README.md)
* [CK wrapper](client_example/25_wrapper/README.md)
* [CK codegen](codegen/README.md)
* [CK profiler](profiler/README.md)
* [Examples (Custom use of CK supported operations)](example/README.md)
* [Client examples (Use of CK supported operations with instance factory)](client_example/README.md)
* [Terminology](/TERMINOLOGY.md)
* [Contributors](/CONTRIBUTORS.md)

CK is released under the **[MIT license](/LICENSE)**.

## Building CK

We recommend building CK inside Docker containers, which include all necessary packages. Pre-built
Docker images are available on [DockerHub](https://hub.docker.com/r/rocm/composable_kernel/tags).

1. To build a new Docker image, use the Dockerfile provided with the source code:

    ```bash
    DOCKER_BUILDKIT=1 docker build -t ck:latest -f Dockerfile .
    ```

2. Launch the Docker container:

    ```bash
    docker run                                     \
    -it                                            \
    --privileged                                   \
    --group-add sudo                               \
    -w /root/workspace                             \
    -v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
    ck:latest                                      \
    /bin/bash
    ```

3. Clone CK source code from the GitHub repository and start the build:

    ```bash
    git clone https://github.com/ROCm/composable_kernel.git && \
    cd composable_kernel && \
    mkdir build && \
    cd build
    ```

    You must set the `GPU_TARGETS` macro to specify the GPU target architecture(s) you want
    to run CK on. You can specify single or multiple architectures. If you specify multiple architectures,
    use a semicolon between each; for example, `gfx908;gfx90a;gfx942`.

    ```bash
    cmake                                                                                             \
    -D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
    -D CMAKE_BUILD_TYPE=Release                                                                       \
    -D GPU_TARGETS="gfx908;gfx90a"                                                                    \
    ..
    ```

    If you don't set `GPU_TARGETS` on the cmake command line, CK is built for all GPU targets
    supported by the current compiler (this may take a long time). 
    Tests and examples will only get built if the GPU_TARGETS is set by the user on the cmake command line.

    NOTE: If you try setting `GPU_TARGETS` to a list of architectures, the build will only work if the 
    architectures are similar, e.g., `gfx908;gfx90a`, or `gfx1100;gfx1101;gfx11012`. Otherwise, if you 
    want to build the library for a list of different architectures,
    you should use the `GPU_ARCHS` build argument, for example `GPU_ARCHS=gfx908;gfx1030;gfx1100;gfx942`.

    **Convenience script for development builds:**

    Alternatively, you can use the provided convenience script `script/cmake-ck-dev.sh` which automatically 
    configures CK for development with sensible defaults. In the build directory:

    ```bash
    ../script/cmake-ck-dev.sh
    ```

    This script:
    * Cleans CMake cache files before configuring
    * Sets `BUILD_DEV=ON` for development mode
    * Defaults to GPU targets: `gfx908;gfx90a;gfx942`
    * Enables verbose makefile output
    * Sets additional compiler flags for better error messages

    By default, it considers the parent directory to be the project source directory.

    You can specify the source directory as the first argument.
    You can specify custom GPU targets (semicolon-separated) as the second argument:

    ```bash
    ../script/cmake-ck-dev.sh .. gfx1100
    ```

    Or pass additional cmake arguments:

    ```bash
    ../script/cmake-ck-dev.sh .. gfx90a -DCMAKE_BUILD_TYPE=Release
    ```

5. Build the entire CK library:

    ```bash
    make -j"$(nproc)"
    ```

6. Install CK:

    ```bash
    make -j install
    ```
    **[See Note on -j](#notes)**

## Optional post-install steps

* Build examples and tests:

    ```bash
    make -j examples tests
    ```

* Build and run all examples and tests:

    ```bash
    make -j check
    ```

    You can find instructions for running each individual example in [example](/example).

* Build and run smoke/regression examples and tests:

    ```bash
    make -j smoke # tests and examples that run for < 30 seconds each
    ```
     ```bash
    make -j regression # tests and examples that run for >= 30 seconds each
    ```

* Build ckProfiler:

    ```bash
    make -j ckProfiler
    ```

    You can find instructions for running ckProfiler in [profiler](/profiler).

* Build our documentation locally:

    ``` bash
    cd docs
    pip3 install -r sphinx/requirements.txt
    python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
    ```

### Notes
The `-j` option for building with multiple threads in parallel, which speeds up the build significantly.
However, `-j` launches unlimited number of threads, which can cause the build to run out of memory and
crash. On average, you should expect each thread to use ~2Gb of RAM.
Depending on the number of CPU cores and the amount of RAM on your system, you may want to
limit the number of threads. For example, if you have a 128-core CPU and 128 Gb of RAM it's advisable to use `-j32`.

Additional cmake flags can be used to significantly speed-up the build:

* `DTYPES` (default is not set) can be set to any subset of "fp64;fp32;tf32;fp16;fp8;bf16;int8" to build
  instances of select data types only. The main default data types are fp32 and fp16; you can safely skip
  other data types.

* `DISABLE_DL_KERNELS` (default is OFF) must be set to ON in order not to build instances, such as `gemm_dl` or
  `batched_gemm_multi_d_dl`. These instances are useful on architectures like the NAVI2x, as most
  other platforms have faster instances, such as `xdl` or `wmma`, available.

* `DISABLE_DPP_KERNELS` (default is OFF) must be set to ON in order not to build instances, such as `gemm_dpp`. 
  These instances offer a slightly better performance of fp16 gemms on NAVI2x. But on other architectures faster alternatives are available.

* `CK_USE_FP8_ON_UNSUPPORTED_ARCH` (default is OFF) must be set to ON in order to build instances,
  such as `gemm_universal`, `gemm_universal_streamk` and `gemm_multiply_multiply` for fp8 data type for GPU targets which do not  have native support for fp8 data type, such as gfx908 or gfx90a. These instances are useful on
  architectures like the MI100/MI200 for the functional support only.

## Using sccache for building

The default CK Docker images come with a pre-installed version of sccache, which supports clang
being used as hip-compiler (" -x hip"). Using sccache can help reduce the time to re-build code from
hours to 1-2 minutes. In order to invoke sccache, you need to run:

```bash
 sccache --start-server
```

then add the following flags to the cmake command line:

```bash
 -DCMAKE_HIP_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache
```

You may need to clean up the build folder and repeat the cmake and make steps in order to take
advantage of the sccache during subsequent builds.

## Using CK as pre-built kernel library

You can find instructions for using CK as a pre-built kernel library in [client_example](/client_example).

## Contributing to CK

When you contribute to CK, make sure you run `clang-format` on all changed files. We highly
recommend using git hooks that are managed by the `pre-commit` framework. To install hooks, run:

```bash
sudo script/install_precommit.sh
```

With this approach, `pre-commit` adds the appropriate hooks to your local repository and
automatically runs `clang-format` (and possibly additional checks) before any commit is created.

If you need to uninstall hooks from the repository, you can do so by running the following command:

```bash
script/uninstall_precommit.sh
```

If you need to temporarily disable pre-commit hooks, you can add the `--no-verify` option to the
`git commit` command.
