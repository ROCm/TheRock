# AOCL-FFTZ

This experimental integration builds [AMD AOCL-FFTZ](https://github.com/amd/aocl-fftz)
and its FFTW compatibility wrapper. It does not change rocFFT or hipFFT's CPU
reference provider.

For a minimal Windows configuration, run these commands from an MSVC developer
environment after fetching sources with `build_tools/fetch_sources.py`:

```powershell
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe `
  -DCMAKE_LINKER=link.exe -DTHEROCK_AMDGPU_FAMILIES=gfx1100 `
  -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_BASE=ON `
  -DTHEROCK_ENABLE_FFTZ=ON
cmake --build build --target therock-fftz+dist --parallel 8
cmake --build build --target artifact-fftz
```

`BASE` supplies `rocm-cmake`, a build dependency of the compiler. The FFTZ
feature enables the compiler and its system dependencies automatically.

The generated feature defaults to the `HOST_MATH` group, which is off by default.
FFTZ uses TheRock's `amd-llvm` toolchain, builds shared libraries, and disables
upstream tests and documentation. The top-level build builds the compiler first.

On Windows, configure TheRock from a Visual Studio developer environment with
MSVC as the host compiler (`cl.exe` for C and C++). FFTZ uses the GNU frontend
of TheRock's built Clang. A source patch selects flags by compiler frontend;
the fetch step applies it with CMake's discovered Git executable.
OpenMP is disabled for FFTZ on Windows because TheRock's Windows compiler build
does not currently include the OpenMP runtime. On Linux, OpenMP is enabled;
that configuration has not yet been tested here.

## Install layout

The stage and dist contain a separate prefix at `lib/host-math/fftz`, keeping
FFTZ's `fftw3.h` and FFTW-compatible library names separate from FFTW itself.
Point an independent consumer's `CMAKE_PREFIX_PATH` at this prefix to use
`find_package(aocl-fftz CONFIG)` or `find_package(FFTW3 CONFIG)`.
TheRock advertises only `aocl-fftz`; the existing FFTW provider retains the
`FFTW3` package name in the superbuild.

Windows DLLs are installed under the private prefix's `lib` directory. Add it
to `PATH` when running a consumer.

## Source and maintenance

The source is pinned to commit `7bdcb28b96b3f2c64529105886249b55fa879ee1`
with a SHA256-verified archive. AMD maintains the upstream library. Its license
is BSD-3-Clause, and a copy is installed under `share/licenses/fftz` inside
the private prefix. The post hook moves wrapper build outputs out of the
source directory.

The FFTW wrapper's float guru64 dimension types and opaque plan types still
require compatibility work before rocFFT can use it. This build integration
does not patch those interfaces or establish full numerical compatibility.

Generated with Codex.
