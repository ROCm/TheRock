# Windows Support

TheRock aims to support as many subprojects as possible on "native" Windows
(as opposed to WSL 1 or WSL 2) using standard build tools like MSVC.

> [!WARNING]
> This is still under development. Not all subprojects build for Windows yet.

## Supported subprojects

ROCm is composed of many subprojects, some of which are supported on Windows:

- https://rocm.docs.amd.com/en/latest/what-is-rocm.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/component-support.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/release-versioning.html#windows-builds-from-source

This table tracks current support status for each subproject in TheRock on
Windows. Some subprojects may need extra patches to build within TheRock (on
mainline, in open source, using MSVC, etc.).

| Component subset | Subproject                                                                   | Supported | Notes                                         |
| ---------------- | ---------------------------------------------------------------------------- | --------- | --------------------------------------------- |
| base             | aux-overlay                                                                  | ✅        |                                               |
| base             | [rocm-cmake](https://github.com/ROCm/rocm-cmake)                             | ✅        |                                               |
| base             | [rocm-core](https://github.com/ROCm/rocm-core)                               | ✅        | No shared libraries                           |
| base             | [rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib)                         | ❌        | Unsupported                                   |
| base             | [rocprofiler-register](https://github.com/ROCm/rocprofiler-register)         | ❌        | Unsupported                                   |
| base             | [rocm-half](https://github.com/ROCm/half)                                    | ✅        |                                               |
|                  |                                                                              |           |                                               |
| compiler         | [amd-llvm](https://github.com/ROCm/llvm-project)                             | ✅        | No shared libraries, limited runtimes         |
| compiler         | [amd-comgr](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr) | ✅        | No shared libraries                           |
| compiler         | [hipcc](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc)     | ✅        |                                               |
| compiler         | [hipify](https://github.com/ROCm/HIPIFY)                                     | ✅        | Patched for Ninja                             |
|                  |                                                                              |           |                                               |
| core             | [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime)                         | ❌        | Unsupported                                   |
| core             | [rocminfo](https://github.com/ROCm/rocminfo)                                 | ❌        | Unsupported                                   |
| core             | [clr](https://github.com/ROCm/clr)                                           | ⭕        | Needs a folder with prebuilt static libraries |
|                  |                                                                              |           |                                               |
| profiler         | [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk)                   | ❔        |                                               |
|                  |                                                                              |           |                                               |
| comm-libs        | [rccl](https://github.com/ROCm/rccl)                                         | ❔        |                                               |
|                  |                                                                              |           |                                               |
| math-libs        | [rocRAND](https://github.com/ROCm/rocRAND)                                   | ✅        |                                               |
| math-libs        | [hipRAND](https://github.com/ROCm/hipRAND)                                   | ✅        |                                               |
| math-libs        | [rocPRIM](https://github.com/ROCm/rocPRIM)                                   | ✅        |                                               |
| math-libs        | [hipCUB](https://github.com/ROCm/hipCUB)                                     | ✅        |                                               |
| math-libs        | [rocThrust](https://github.com/ROCm/rocThrust)                               | ✅        |                                               |
| math-libs        | [rocFFT](https://github.com/ROCm/rocFFT)                                     | ✅        | No shared libraries                           |
| math-libs        | [hipFFT](https://github.com/ROCm/hipFFT)                                     | ✅        | No shared libraries                           |
| math-libs (blas) | [hipBLAS-common](https://github.com/ROCm/hipBLAS-common)                     | ✅        |                                               |
| math-libs (blas) | [hipBLASLt](https://github.com/ROCm/hipBLASLt)                               | ✅        |                                               |
| math-libs (blas) | [rocBLAS](https://github.com/ROCm/rocBLAS)                                   | ✅        | Running tests needs PyYAML and a dll copied   |
| math-libs (blas) | [rocSPARSE](https://github.com/ROCm/rocSPARSE)                               | ✅        | Tests need rocblas.dll and can't find files   |
| math-libs (blas) | [hipSPARSE](https://github.com/ROCm/hipSPARSE)                               | ❔        |                                               |
| math-libs (blas) | [rocSOLVER](https://github.com/ROCm/rocSOLVER)                               | ✅        |                                               |
| math-libs (blas) | [hipSOLVER](https://github.com/ROCm/hipSOLVER)                               | ✅        | Tests need dlls                               |
| math-libs (blas) | [hipBLAS](https://github.com/ROCm/hipBLAS)                                   | ❔        |                                               |
|                  |                                                                              |           |                                               |
| ml-libs          | [MIOpen](https://github.com/ROCm/MIOpen)                                     | ❔        |                                               |

## Building from source

These instructions mostly mirror the instructions in the root
[README.md](../../README.md), with some extra Windows-specific callouts.

### Prerequisites

#### Set up your system

- Choose your shell between cmd, powershell, and git bash as well as your
  terminal application. Some developers report good experiences with
  [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/)
  and [Cmder](https://cmder.app/).

- A Dev Drive is recommended, due to how many source and build files are used.
  See the
  [Set up a Dev Drive on Windows 11](https://learn.microsoft.com/en-us/windows/dev-drive/)
  article for setup instructions.

- Symlink support is recommended.

  Test if symlinks work from cmd:

  ```cmd
  echo "Test 1 2 3" > test.txt
  mklink link_from_cmd.txt test.txt
  ```

  Test if symlinks work from Powershell:

  ```powershell
  echo "Test 1 2 3" > test.txt
  New-Item -Path link_from_powershell.txt -ItemType SymbolicLink -Value test.txt
  ```

  If symlink support is not enabled, enable developer mode and/or grant your
  account the "Create symbolic links" permission. These resources may help:

  - https://portal.perforce.com/s/article/3472
  - https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/create-symbolic-links
  - https://stackoverflow.com/a/59761201

#### Install tools

You will need:

- Git: https://git-scm.com/downloads

  - Suggested: enable symlinks with `git config --global core.symlinks true`

- CMake: https://cmake.org/download/

- Ninja: https://ninja-build.org/

- (Optional) ccache: https://ccache.dev/, or sccache:
  https://github.com/mozilla/sccache

- Python: https://www.python.org/downloads/ (3.11+ recommended)

- The MSVC compiler from https://visualstudio.microsoft.com/downloads/
  (typically from either Visual Studio or the Build Tools for Visual Studio),
  including these components:

  - MSVC
  - C++ CMake tools for Windows
  - C++ ATL
  - C++ AddressSanitizer (optional)

  After installing MSVC, use it in your build environment. If you build from an
  editor like VSCode, CMake can discover the compiler among other "kits". If you
  use the command line, see
  https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170.
  (typically run the appropriate `vcvarsall.bat`)

> [!TIP]
> Some of these tools are available via package managers like
> https://github.com/chocolatey/choco
>
> ```
> choco install git
> choco install cmake
> choco install ninja
> choco install ccache
> choco install sccache
> choco install python
> ```

### Clone and fetch sources

```bash
git clone https://github.com/ROCm/TheRock.git

# Clone interop library from https://github.com/nod-ai/amdgpu-windows-interop
# for CLR (the "HIP runtime") on Windows. The path used can also be configured
# using the `THEROCK_AMDGPU_WINDOWS_INTEROP_DIR` CMake variable.
git clone https://github.com/nod-ai/amdgpu-windows-interop.git

cd TheRock
python ./build_tools/fetch_sources.py
```

### Configure

Some components do not build for Windows yet, so disable them:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -DTHEROCK_ENABLE_HIPIFY=ON \
  -DTHEROCK_ENABLE_CORE=OFF \
  -DTHEROCK_ENABLE_CORE_RUNTIME=OFF \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_PROFILER_SDK=OFF \
  -DTHEROCK_ENABLE_COMM_LIBS=OFF \
  -DTHEROCK_ENABLE_MATH_LIBS=OFF \
  -DTHEROCK_ENABLE_RAND=ON \
  -DTHEROCK_ENABLE_PRIM=ON \
  -DTHEROCK_ENABLE_FFT=ON \
  -DTHEROCK_ENABLE_BLAS=OFF \
  -DTHEROCK_ENABLE_SPARSE=OFF \
  -DTHEROCK_ENABLE_SOLVER=OFF \
  -DTHEROCK_ENABLE_ML_LIBS=OFF

# If iterating and wishing to cache, add these:
#  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
#  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
#  -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
```

> [!TIP]
> ccache [does not support](https://github.com/ccache/ccache/issues/1040)
> MSVC's `/Zi` flag which may be set by default when a project (e.g. LLVM) opts
> in to
> [policy CMP0141](https://cmake.org/cmake/help/latest/policy/CMP0141.html).
> Setting
> [`-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded`](https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_DEBUG_INFORMATION_FORMAT.html)
> instructs CMake to compile with `/Z7` or equivalent, which is supported by
> ccache.

Ensure that MSVC is used by looking for lines like these in the logs:

```text
-- The C compiler identification is MSVC 19.42.34436.0
-- The CXX compiler identification is MSVC 19.42.34436.0
```

### Build

```bash
cmake --build build
```

At the moment this should build some projects in [`base/`](../../base/) as well
as [`compiler/`](../../compiler/).

### Building CLR from partial sources

We are actively working on enabling source builds of
https://github.com/ROCm/clr (notably for `amdhip64_6.dll`) on Windows.
Historically this has been a closed source component due to the dependency on
[Platform Abstraction Library (PAL)](https://github.com/GPUOpen-Drivers/pal)
and providing a fully open source build will take more time. As an incremental
step towards a fully open source build, we will use an interop folder containing
header files and static library `.lib` files for PAL and related components.

An incremental rollout is planned:

1. The interop folder must be manually copied into place in the source tree.
   This will allow AMD developers to iterate on integration into TheRock while
   we work on making this folder or more source files available.
1. *(We are here today)* The interop folder will be available publicly
   (currently at https://github.com/nod-ai/amdgpu-windows-interop).
1. The interop folder will be included automatically from either a git
   repository or cloud storage (like the existing third party dep mirrors in
   [`third-party/`](../../third-party/)).
1. A more permanent open source strategy for building the CLR (the HIP runtime)
   from source on Windows will eventually be available.

If configured correctly, outputs like
`build/core/clr/dist/bin/amdhip64_6.dll` should be generated by the build.

If the interop folder is _not_ available, sub-project support is limited and
features should be turned off:

```bash
-DTHEROCK_ENABLE_CORE=OFF \
-DTHEROCK_ENABLE_HIP_RUNTIME=OFF \
-DTHEROCK_ENABLE_RAND=OFF \
-DTHEROCK_ENABLE_PRIM=OFF \
-DTHEROCK_ENABLE_FFT=OFF \
```

### Testing

Test builds can be enabled with `-DBUILD_TESTING=ON`.

Some subproject tests have been validated on Windows, like rocPRIM:

```bash
ctest --test-dir build/math-libs/rocPRIM/dist/bin/rocprim --output-on-failure
```
