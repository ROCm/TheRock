# Windows Support

TheRock aims to support as many subprojects as possible on "native" Windows
(as opposed to WSL 1 or WSL 2) using standard build tools like MSVC.

> [!WARNING]
> While Windows source builds of TheRock (including PyTorch!) are working for
> some expert developers, this support is relatively new and is not yet mature.
> There are several [known issues](#build-troubleshooting-and-known-issues) and
> active development areas on the
> [Windows platform support bringup](https://github.com/ROCm/TheRock/issues/36).
> If you encounter any issues not yet represented there, please file an issue.

## Supported subprojects

ROCm is composed of many subprojects, some of which are supported on Windows:

- https://rocm.docs.amd.com/en/latest/what-is-rocm.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/component-support.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/release-versioning.html#windows-builds-from-source

This table tracks current support status for each subproject in TheRock on
Windows. Some subprojects may need extra patches to build within TheRock (on
mainline, in open source, using MSVC, etc.).

| Component subset    | Subproject                                                                   | Supported | Notes                                         |
| ------------------- | ---------------------------------------------------------------------------- | --------- | --------------------------------------------- |
| base                | aux-overlay                                                                  | ✅        |                                               |
| base                | [rocm-cmake](https://github.com/ROCm/rocm-cmake)                             | ✅        |                                               |
| base                | [rocm-core](https://github.com/ROCm/rocm-core)                               | ✅        |                                               |
| base                | [rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib)                         | ❌        | Unsupported                                   |
| base                | [rocprofiler-register](https://github.com/ROCm/rocprofiler-register)         | ❌        | Unsupported                                   |
| base                | [rocm-half](https://github.com/ROCm/half)                                    | ✅        |                                               |
|                     |                                                                              |           |                                               |
| compiler            | [amd-llvm](https://github.com/ROCm/llvm-project)                             | ✅        | Limited runtimes                              |
| compiler            | [amd-comgr](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr) | ✅        |                                               |
| compiler            | [hipcc](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc)     | ✅        |                                               |
| compiler            | [hipify](https://github.com/ROCm/HIPIFY)                                     | ✅        |                                               |
|                     |                                                                              |           |                                               |
| core                | [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime)                         | ❌        | Unsupported                                   |
| core                | [rocminfo](https://github.com/ROCm/rocminfo)                                 | ❌        | Unsupported                                   |
| core                | [clr](https://github.com/ROCm/clr)                                           | 🟡        | Needs a folder with prebuilt static libraries |
|                     |                                                                              |           |                                               |
| profiler            | [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk)                   | ❌        | Unsupported                                   |
|                     |                                                                              |           |                                               |
| comm-libs           | [rccl](https://github.com/ROCm/rccl)                                         | ❌        | Unsupported                                   |
|                     |                                                                              |           |                                               |
| math-libs           | [rocRAND](https://github.com/ROCm/rocRAND)                                   | ✅        |                                               |
| math-libs           | [hipRAND](https://github.com/ROCm/hipRAND)                                   | ✅        |                                               |
| math-libs           | [rocPRIM](https://github.com/ROCm/rocPRIM)                                   | ✅        |                                               |
| math-libs           | [hipCUB](https://github.com/ROCm/hipCUB)                                     | ✅        |                                               |
| math-libs           | [rocThrust](https://github.com/ROCm/rocThrust)                               | ✅        |                                               |
| math-libs           | [rocFFT](https://github.com/ROCm/rocFFT)                                     | ✅        |                                               |
| math-libs           | [hipFFT](https://github.com/ROCm/hipFFT)                                     | ✅        |                                               |
| math-libs (support) | [mxDataGenerator](https://github.com/ROCm/mxDataGenerator)                   | ❌        | Unsupported                                   |
| math-libs (BLAS)    | [hipBLAS-common](https://github.com/ROCm/hipBLAS-common)                     | ✅        |                                               |
| math-libs (BLAS)    | [rocRoller](https://github.com/ROCm/rocRoller)                               | ❌        | Unsupported                                   |
| math-libs (BLAS)    | [hipBLASLt](https://github.com/ROCm/hipBLASLt)                               | ✅        |                                               |
| math-libs (BLAS)    | [rocBLAS](https://github.com/ROCm/rocBLAS)                                   | ✅        |                                               |
| math-libs (BLAS)    | [rocSPARSE](https://github.com/ROCm/rocSPARSE)                               | ✅        |                                               |
| math-libs (BLAS)    | [hipSPARSE](https://github.com/ROCm/hipSPARSE)                               | ✅        |                                               |
| math-libs (BLAS)    | [rocSOLVER](https://github.com/ROCm/rocSOLVER)                               | ✅        |                                               |
| math-libs (BLAS)    | [hipSOLVER](https://github.com/ROCm/hipSOLVER)                               | ✅        |                                               |
| math-libs (BLAS)    | [hipBLAS](https://github.com/ROCm/hipBLAS)                                   | ✅        |                                               |
|                     |                                                                              |           |                                               |
| ml-libs             | [Composable Kernel](https://github.com/ROCm/composable_kernel)               | ❌        | Unsupported                                   |
| ml-libs             | [MIOpen](https://github.com/ROCm/MIOpen)                                     | ✅        |                                               |

## Building TheRock from source

These instructions mostly mirror the instructions in the root
[README.md](../../README.md), with some extra Windows-specific callouts.

### Prerequisites

#### Set up your system

- You will need a powerful system with sufficient RAM (16GB+),
  CPU (8+ cores), and storage (200GB+) for a full source build.
  Partial source builds bootstrapped from prebuilt binaries are on our roadmap
  to enable.

  - To set expectations, on powerful build servers, the full source build can
    still take over an hour.

  - The Windows build uses
    [hard links](https://learn.microsoft.com/en-us/windows/win32/fileio/hard-links-and-junctions)
    to save space, but storage sizes do not often account for this savings.
    Reported size used and actual size used may differ substantially.

- Long path support is required. As needed, enable long paths for your system:

  - https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#registry-setting-to-enable-long-paths

- There are some [known issues](https://github.com/ROCm/TheRock/issues/651)
  with preexisting HIP SDK / ROCm installs causing errors during the build
  process. Until these are resolved, we recommend uninstalling the HIP SDK
  before trying to build TheRock.

- A Dev Drive is recommended, due to how many source and build files are used.
  See the
  [Set up a Dev Drive on Windows 11](https://learn.microsoft.com/en-us/windows/dev-drive/)
  article for setup instructions.

- Symlink support is recommended. If symlink support is not enabled, enable
  developer mode and/or grant your
  account the "Create symbolic links" permission. These resources may help:

  - https://portal.perforce.com/s/article/3472
  - https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/create-symbolic-links
  - https://stackoverflow.com/a/59761201

- We recommend cmd or git bash over PowerShell. Some developers also report
  good experiences with
  [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/)
  and [Cmder](https://cmder.app/).

#### Install tools

> [!TIP]
> These tools are available via package managers like
> [chocolatey](https://chocolatey.org/):
>
> ```bash
> choco install visualstudio2022buildtools -y --params "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
> choco install git.install -y --params "'/GitAndUnixToolsOnPath'"
> choco install cmake --version=3.31.0 -y
> choco install ninja -y
> choco install ccache -y
> choco install python -y
> choco install strawberryperl -y
> ```

If you prefer to install tools manually, you will need:

- The MSVC compiler from https://visualstudio.microsoft.com/downloads/
  (Using either "Visual Studio" or "Build Tools for Visual Studio"),
  including these components:

  - MSVC
  - C++ CMake tools for Windows
  - C++ ATL
  - C++ AddressSanitizer (optional)

- Git: https://git-scm.com/downloads

  - With "Use Git and optional Unix tools from the Windows Command Prompt" as certain build scripts use Bash.

- CMake: https://cmake.org/download/, version < 4.0.0
  (see [Issue#318](https://github.com/ROCm/TheRock/issues/318))

- Ninja: https://ninja-build.org/

- (Optional) ccache: https://ccache.dev/, or sccache:
  https://github.com/mozilla/sccache

- Python: https://www.python.org/downloads/ (3.11+ recommended)

- Strawberry Perl, which comes with gfortran: https://strawberryperl.com/

#### Important tool settings

> [!IMPORTANT]
> Git should be configured with support for symlinks and long paths:
>
> ```bash
> git config --global core.symlinks true
> git config --global core.longpaths true
> ```

> [!IMPORTANT]
> After installing MSVC, use its 64 bit tools in your build environment.
>
> - If you use the command line, see
>   [Use the Microsoft C++ toolset from the command line](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170).
>   Typically this means either `Start > x64 Native Tools Command Prompt for VS 2022` or
>   running `vcvars64.bat` in an existing shell.
> - If you build from an editor like VSCode, CMake can discover the compiler
>   among other "kits".
> - You can also tell CMake to use MSVC's tools explicitly with
>   `-DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_LINKER=link.exe`

### Set the locale

If the build system is a non-English system. Make sure to switch to `utf-8`.

```cmd
chcp 65001
```

### Clone and fetch sources

```bash
# Clone the repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock

# Init python virtual environment and install python dependencies
python -m venv .venv
.venv\Scripts\Activate.bat
pip install -r requirements.txt

# Download submodules and apply patches
python ./build_tools/fetch_sources.py
```

### Build configuration

Unsupported subprojects like RCCL are automatically disabled on Windows. See
the [instructions in the root README](../../README.md#configuration) for other
options you may want to set.

```bash
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu

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

> [!TIP]
> Ensure that MSVC is used by looking for lines like these in the logs:
>
> ```text
> -- The C compiler identification is MSVC 19.42.34436.0
> -- The CXX compiler identification is MSVC 19.42.34436.0
> ```
>
> If you see some other compiler there, refer to the MSVC setup instructions up
> in [Important tool settings](#important-tool-settings).

### CMake build usage

```bash
cmake --build build --target therock-dist
cmake --build build --target therock-archives
```

This will start building using MSVC. Once the amd-llvm subproject is built,
subprojects like the ROCm math libraries will be compiled using `clang.exe` and
other tools from the amd-llvm toolchain.

When the builds complete, you should have a build of ROCm / the HIP SDK
in `build/dist/rocm/` and artifacts in `build/artifacts`. See the
[Build Artifacts guide](./artifacts.md) for more information about the build
outputs.

#### Building ROCm Python wheels

To build Python wheels, you will need an "artifacts" directory, either from a
source build of `therock-archives` (see above) or by running the
[`fetch_artifacts.py`](../../build_tools/fetch_artifacts.py) script to download
artifacts from a CI run.

Once you have an artifacts directory, you can run the
[`build_python_packages.py`](../../build_tools/build_python_packages.py) script.

#### Building PyTorch

PyTorch builds require Python wheels, either by building from source (see above)
or by downloading from one of TheRock's release indices. See the instructions at
[external-builds/pytorch](../../external-builds/pytorch/README.md).

### Run tests

Test builds can be enabled with `-DBUILD_TESTING=ON` (the default).

Some subproject tests have been validated on Windows, like rocPRIM:

```bash
ctest --test-dir build/math-libs/rocPRIM/dist/bin/rocprim --output-on-failure
```

### Build troubleshooting and known issues

#### `No such file or directory: (long path to a source file)`

If the build looks for a source file with a long path that does not exist,
ensure that you have long path support enabled in git (see
[Important tool settings](#important-tool-settings)), then re-run

```bash
python ./build_tools/fetch_sources.py
```

Once the git operations have run with the new setting, confirm that the missing
files now exist.

#### `error: directive requires gfx90a+`

Errors like this indicate that the value of `-DTHEROCK_AMDGPU_FAMILIES=` or
`-DTHEROCK_AMDGPU_TARGETS=` is currently unsupported by one or more libraries.

#### `lld-link: error: duplicate symbol`

Several developers have reported link errors in rocBLAS and rocSPARSE like

```
[rocSPARSE] lld-link: error: duplicate symbol: __hip_cuid_2f7b343d50a9613
[rocSPARSE] >>> defined at library/CMakeFiles/rocsparse.dir/src/level3/csrmm/row_split/csrmm_device_row_split_256_16_8_3_4.cpp.obj
[rocSPARSE] >>> defined at library/CMakeFiles/rocsparse.dir/src/level3/csrmm/row_split/csrmm_device_row_split_256_16_8_7_4.cpp.obj
```

```
[rocBLAS] lld-link: error: duplicate symbol: rocblas_stbsv_batched
[rocBLAS] >>> defined at library/src/CMakeFiles/rocblas.dir/blas2/rocblas_tbsv_batched.cpp.obj
[rocBLAS] >>> defined at library/src/CMakeFiles/rocblas.dir/handle.cpp.obj
[rocBLAS]
[rocBLAS] lld-link: error: duplicate symbol: rocblas_dtbsv_batched
[rocBLAS] >>> defined at library/src/CMakeFiles/rocblas.dir/blas2/rocblas_tbsv_batched.cpp.obj
[rocBLAS] >>> defined at library/src/CMakeFiles/rocblas.dir/handle.cpp.obj
[rocBLAS]
[rocBLAS] lld-link: error: duplicate symbol: __hip_cuid_a201499d9a86b1da
```

These have been worked around by disabling ccache.

## Other notes

### Platform support lessons, tips, and gotchas

#### Filesystem paths

- Windows paths may have spaces in them, so properly quote paths
- Windows paths may use `\` instead of `/` as a component delimiter
  - The `\` character is often an escape character in raw strings, so properly escape paths if passing them between systems
  - In Python, [`pathlib`](https://docs.python.org/3/library/pathlib.html) has good support for performing transformations on paths like joining or splitting components in a cross-platform way, but some outgoing uses still need to use `.resolve()`, `.as_posix()`, and other functions to resolve symlinks, convert slashes, etc.
  - In C++, [`<filesystem>`](https://en.cppreference.com/w/cpp/filesystem.html) has good support for performing transformations on paths like joining or splitting components in a cross-platform way. See https://github.com/ROCm/rocm-libraries/pull/948 for example.
  - In CMake, take care when loading environment variables into strings and treating them as paths. Use https://cmake.org/cmake/help/latest/command/file.html#path-conversion and https://cmake.org/cmake/help/latest/command/cmake_path.html instead of working with strings directly, like so:
    ```diff
    -set(HIP_PATH $ENV{HIP_PATH})
    +file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_PATH)`
    ```
- Windows paths (and commands) can have max length limitations, so avoid long file names
- Windows paths do not allow some characters (https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file)

#### Other platform differences

- Symlink support on Windows is not always available
- Windows uses `[library].dll` files instead of `lib[library].so` files, which use different APIs for loading/linking (and have different versioning in the file names too). CMake can use https://cmake.org/cmake/help/latest/variable/CMAKE_SHARED_LIBRARY_PREFIX.html and https://cmake.org/cmake/help/latest/variable/CMAKE_SHARED_LIBRARY_SUFFIX.html
- Windows uses the .exe extension for executables while Linux uses files with no suffix. CMake can use https://cmake.org/cmake/help/latest/variable/CMAKE_EXECUTABLE_SUFFIX.html
- GitHub actions can use bash.exe, cmd.exe, and powershell on Windows. Switching workflows to use Python scripts helps a bit, but some actions / steps / tools / scripts still expect paths in a certain format or with a certain escaping style, or expect certain tools to be available (e.g. `sed`, `grep`)

### Building CLR from partial sources

We are working on enabling flexible open source builds of
https://github.com/ROCm/clr (notably for `amdhip64_6.dll`) on Windows.
Historically this has been a closed source component due to the dependency on
[Platform Abstraction Library (PAL)](https://github.com/GPUOpen-Drivers/pal)
and providing a fully open source build will take more time. As an incremental
step towards a fully open source build, we are using an interop folder
containing header files and static library `.lib` files for PAL and related
components.

An incremental rollout is planned:

1. The interop folder must be manually copied into place in the source tree.
   This will allow AMD developers to iterate on integration into TheRock while
   we work on making this folder or more source files available.
1. The interop folder will be available publicly
   (currently at https://github.com/ROCm/amdgpu-windows-interop).
1. *(We are here today)* The interop folder will be included automatically from
   a git repository using git LFS.
1. A more permanent open source strategy for building the CLR (the HIP runtime)
   from source on Windows will eventually be available.

If configured correctly, outputs like
`build/core/clr/dist/bin/amdhip64_6.dll` should be generated by the build.

If the interop folder is _not_ available, sub-project support is limited and
features should be turned off:

```bash
-DTHEROCK_ENABLE_CORE=OFF \
-DTHEROCK_ENABLE_MATH_LIBS=OFF \
-DTHEROCK_ENABLE_ML_LIBS=OFF \
```
