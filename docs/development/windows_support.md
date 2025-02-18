# Windows Support

TheRock aims to support as many subprojects as possible on "native" Windows
(as opposed to WSL 1 or WSL 2) using standard build tools like MSVC.

> [!WARNING]
> This is still under development. Not all subprojects build for Windows yet.

## Supported subprojects

This table tracks current support status for each subproject in TheRock on
Windows. Some subprojects do not support Windows at all, while others may need
extra patches to build within TheRock (on mainline, in open source, using MSVC,
etc.).

Component subset | Subproject | Supported | Notes
---------------- | ---------- | --------- | -----
base | aux-overlay | ✅ |
base | [rocm-cmake](https://github.com/ROCm/rocm-cmake) | ✅ |
base | [rocm-core](https://github.com/ROCm/rocm-core) | ✅ | No shared libraries
base | [rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib) | ❌ | Unsupported
base | [rocprofiler-register](https://github.com/ROCm/rocprofiler-register) | ⭕ | Patched to no-op
base | [rocm-half](https://github.com/ROCm/half) | ✅ |
| | | | |
compiler | [amd-llvm](https://github.com/ROCm/llvm-project) | ✅ | No shared libraries, limited runtimes
compiler | [amd-comgr](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr) | ✅ | No shared libraries
compiler | [hipcc](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc) | ✅ |
compiler | [hipify](https://github.com/ROCm/HIPIFY) | ❔ | Possible to support? Uses clang++ instead of MSVC?
| | | | |
core | [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) | ❌ | Unsupported
core | [rocminfo](https://github.com/ROCm/rocminfo) | ❌ | Unsupported
core | [clr](https://github.com/ROCm/clr) | ❔ |
| | | | |
profiler | [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) | ❔
| | | | |
comm-libs | [rccl](https://github.com/ROCm/rccl) | ❔ |
| | | | |
math-libs | [rocRAND](https://github.com/ROCm/rocRAND) | ❔ |
math-libs | [hipRAND](https://github.com/ROCm/hipRAND) | ❔ |
math-libs | [rocPRIM](https://github.com/ROCm/rocPRIM) | ❔ |
math-libs | [hipCUB](https://github.com/ROCm/hipCUB) | ❔ |
math-libs | [rocThrust](https://github.com/ROCm/rocThrust) | ❔ |
math-libs | [rocFFT](https://github.com/ROCm/rocFFT) | ❔ |
math-libs | [hipFFT](https://github.com/ROCm/hipFFT) | ❔ |
math-libs (blas) | [hipBLAS-common](https://github.com/ROCm/hipBLAS-common) | ❔ |
math-libs (blas) | [hipBLASlt](https://github.com/ROCm/hipBLASlt) | ❔ |
math-libs (blas) | [rocBLAS](https://github.com/ROCm/rocBLAS) | ❔ |
math-libs (blas) | [rocSPARSE](https://github.com/ROCm/rocSPARSE) | ❔ |
math-libs (blas) | [hipSPARSE](https://github.com/ROCm/hipSPARSE) | ❔ |
math-libs (blas) | [rocSOLVER](https://github.com/ROCm/rocSOLVER) | ❔ |
math-libs (blas) | [hipSOLVER](https://github.com/ROCm/hipSOLVER) | ❔ |
math-libs (blas) | [hipBLAS](https://github.com/ROCm/hipBLAS) | ❔ |
| | | | |
ml-libs | [MIOpen](https://github.com/ROCm/MIOpen) | ❔ |

## Building from source

These instructions mostly mirror the instructions in the root
[README.md](../../README.md), with some extra Windows-specific callouts.

### Prerequisites

#### Set up your system

* Choose your shell between cmd, powershell, and git bash as well as your
  terminal application. Some developers report good experiences with
  [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/)
  and [Cmder](https://cmder.app/).
* Symlink support must be enabled.

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

  * https://portal.perforce.com/s/article/3472
  * https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/create-symbolic-links
  * https://stackoverflow.com/a/59761201
* A Dev Drive is recommended, due to how many source and build files are used.
  See the
  [Set up a Dev Drive on Windows 11](https://learn.microsoft.com/en-us/windows/dev-drive/)
  article for setup instructions.

#### Install tools

You will need:

* Git: https://git-scm.com/downloads
  * Also enable symlinks with `git config --global core.symlinks true`
* CMake: https://cmake.org/download/
* Ninja: https://ninja-build.org/
* (Optional) ccache: https://ccache.dev/, or sccache:
  https://github.com/mozilla/sccache
* Python: https://www.python.org/downloads/ (3.11+ recommended)
* A compiler like MSVC from https://visualstudio.microsoft.com/downloads/
  (typically from either Visual Studio or the Build Tools for Visual Studio),
  including these components:
  * MSVC
  * C++ CMake tools for Windows
  * C++ ATL
  * C++ AddressSanitizer (optional)

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
> choco install cmake
> choco install ninja
> choco install ccache
> choco install sccache
> ```

### Clone and fetch sources

```bash
git clone https://github.com/nod-ai/TheRock.git
```

Check that symlinks were created:

```bash
stat base/rocm-core
  File: base/rocm-core -> ../sources/rocm-core
  Size: 20              Blocks: 0          IO Block: 65536  symbolic link
```

If symlinks were not created, follow the instructions at
https://stackoverflow.com/a/59761201.

Next, fetch sources:

```bash
python ./build_tools/fetch_sources.py
```

### Configure

Some components do not build for Windows yet, so disable them:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -DTHEROCK_ENABLE_HIPIFY=OFF \
  -DTHEROCK_ENABLE_CORE=OFF \
  -DTHEROCK_ENABLE_CORE_RUNTIME=OFF \
  -DTHEROCK_ENABLE_HIP_RUNTIME=OFF \
  -DTHEROCK_ENABLE_COMM_LIBS=OFF \
  -DTHEROCK_ENABLE_MATH_LIBS=OFF \
  -DTHEROCK_ENABLE_ML_LIBS=OFF \
  -DTHEROCK_ENABLE_PROFILER_SDK=OFF

# If iterating and wishing to cache, add these:
#  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
#  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
```

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
