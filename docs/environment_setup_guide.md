# Environment Setup Guide

We only regularly build and test on certain OS combinations, but we aim to enable users wishing to build on a variety of systems, so long as they are relatively modern, have compatible dependencies, and do not create a support burden to accomodate. This page documents known workarounds and instructions for alternative environment setup. See the main project README for quick instructions on latest versions of certain popular distributions.

The advice on this page is not necessarily validated by the project maintainers. For any of these combinations that have known CI coverage, that will be noted. Otherwise, this is best effort information collected in the hope that it will help future users with niche issues.

If you have a configuration that you have found workarounds to support, please send a PR adding it to this page and we will consider including it for the benefit of future users.

## Primary Configurations

See the [project README](../README.md) for quick getting started instructions the following combinations:

- Fedora (TODO: looking for contribution; see [patchelf](#patchelf) for a
  Fedora-specific note)
- Ubuntu 24.04
- Windows (VS2022)

In general, we will keep the home page updated with quick start instructions for recent versions of the above. Additional advanced advice may be found below for specialty quirks and workarounds.

## Required Build Toolchains

### Rust 1.95

Rust 1.95 and Cargo are general build dependencies on Linux and Windows. Install
the required toolchain with [`rustup`](https://rustup.rs/).

On Linux:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- --default-toolchain 1.95.0
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

On Windows:

```powershell
winget install --id Rustlang.Rustup --exact --source winget
# Open a new terminal before running the remaining commands.
rustup install 1.95
rustc --version
cargo --version
```

If `rustup` is already installed, only `rustup install 1.95` is needed.

## Reference Build Environments

When interactively verifying that various Linux based operating systems build properly, we generally use the following procedure:

```
./build_tools/linux_portable_build.py --interactive --image <<some reference image>> [--docker=podman]
... Follow OS specific setup instructions to install packages, etc ...
cmake -S /therock/src -B /therock/output/build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx1100
cmake --build /therock/output/build
```

If having trouble building on a system, we will typically want to eliminate environmental issues by building under a clean/known docker image first using the above procedure. If this succeeds but the build fails on your system, it may still be an issue that we want to know more about, as there can always be bugs related to conflicting package versions, etc. However, it is a much more open ended problem to debug a user issue in the field based on system state that cannot be recreated.

## Alternative Configurations

### Manylinux x86-64

Our open-source binaries are typically built within a [manylinux container](https://github.com/pypa/manylinux) (see [the docker file](../dockerfiles/build_manylinux_x86_64.Dockerfile)). These images are versioned by the glibc version they target, and if dependencies are controlled carefully, binaries built on them should work on systems with the same or higher glibc version.

Present version: glibc 2.28
Based on upstream: AlmaLinux 8 with gcc toolset 13

While this generally implies that the project should build on similarly versioned alternative EL distributions, do note that we install several upgraded tools (see dockerfile above) in our standard CI pipelines.

Reference image: `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:a382085df3ba2419b58aa9051350883a0d0b732a4bc0a4ef60458f8161bb08c6`

### Ubuntu 22.04

Reference image: `ubuntu:22.04`

Workarounds:

- Shipping CMake is too old (3.22): see above advice for CMake

### Arch Linux / EndeavourOS

Arch-based distributions ship the latest toolchain versions, which occasionally
surface new failures. The following notes apply to rolling-release Arch,
EndeavourOS, and similar derivatives.

#### Required packages

```bash
sudo pacman -S cmake ninja patchelf ccache base-devel
```

#### GPU permissions

After installing, ensure your user has access to the GPU by adding yourself to
the `video` and `render` groups (required for ROCm to access the GPU at
runtime). This matches the [upstream ROCm prerequisite][rocm-prereqs]:

```bash
sudo usermod -a -G video,render $LOGNAME
# Log out and back in (or reboot) for the group change to take effect.
groups  # verify 'video' and 'render' appear in the output
```

On Arch, these groups are typically created by the `amdgpu` kernel module but
users are **not** added automatically. Without this step, ROCm will fail at
runtime with permission errors (e.g., `hsaKmtInit` returning
`HSA_STATUS_ERROR_NOT_INITIALIZED` or `HIP` returning `hipErrorNoDevice`).

Arch provides `patchelf` via `pacman`. **Verify that the installed version
includes the PHDR fix** (see [patchelf section](#patchelf) above) — without it,
builds that invoke `patchelf` on split ELF binaries will produce corrupt output:

```bash
pacman -Q patchelf

# After a build that uses patchelf, verify the fix is present:
readelf -l build/dist/rocm/lib/libhsa-runtime64.so 2>/dev/null | grep -A1 PHDR
# If VirtAddr is 0xfffffffffff79040 or similar, you have a broken patchelf.
```

If the fix is not present, build `patchelf` from source using the script above.
On Arch, you will need the build tools:

```bash
sudo pacman -S curl autoconf automake
sudo env INSTALL_PREFIX=/usr/local ./dockerfiles/install_pinned_patchelf.sh
```

#### GCC version considerations

Arch ships the latest stable GCC. As of GCC 15+, several TheRock subprojects
(especially `rocprofiler-systems` and its bundled `dyninst`) fail to compile
under the host GCC due to:

- **`-Werror`-by-default dialect rules** — `incompatible-pointer-types`,
  `discarded-qualifiers`, `unterminated-string-initialization`.
- **`<cstdint>` no longer transitively included** — many subprojects rely on
  the transitive include and fail without an explicit `#include <cstdint>`.

**Workaround:** Disable components known to fail on GCC 15+ until upstream
fixes land. See [TheRock issue #5540](https://github.com/ROCm/TheRock/issues/5540):

```bash
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1032 \
  -DTHEROCK_ENABLE_DEBUG_TOOLS=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Setting `THEROCK_ENABLE_DEBUG_TOOLS=OFF` skips `rocprofiler-systems` (the
primary GCC-15-sensitive component). Most other components compile cleanly
because they are built with TheRock's bundled `amd-llvm` toolchain rather than
the host GCC.

#### Memory and parallelism

Arch kernels ship with `systemd-oomd` enabled by default on many installations.
Combined with high core counts (e.g., 14600K with 20 threads), this can kill
the build during `amd-llvm` link steps. See [Resource Utilization](#resource-utilization)
below for guidance — `-j8` is a safe starting point on a 32 GB system.

## Common Issues

### CMake

Different project components enforce different CMake version ranges. The `cmake_minimum_version` in the top level CMake file (presently 3.25) should be considered the project wide minimum. As of September 2025, CMake 4 is supported on Linux - but not on Windows.

There are various, easy ways to acquire specific CMake versions. For Windows and users wanting to use CMake 3, it can be easily installed with:

1. Be in your venv for TheRock:
   - Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\Activate.bat`
1. `pip install 'cmake<4'`
1. For Linux: if afterwards cmake is not found anymore, run `hash -r` to forget the previously cached location of cmake

### patchelf

Building with `THEROCK_BUNDLE_SYSDEPS=ON` (the default for portable Linux
builds), `THEROCK_ENABLE_ROCGDB=ON`, or generating Python wheels via
`build_tools/build_python_packages.py` all invoke `patchelf` to rewrite
`RPATH`, `SONAME`, and `DT_NEEDED` entries on ELF binaries. Upstream
`patchelf` releases through 0.18.0 contain a bug that corrupts the PHDR
virtual address on any ELF whose PHDR sits in a trailing LOAD segment,
which is how `kpack` leaves libraries after splitting device code from
host code.

#### Issue with patchelf

When the wrong `patchelf` rewrites an affected library you will see one
or more of:

- `OSError: failed to map segment from shared object` at load time (e.g.
  during `rocm-sdk test testSharedLibrariesLoad`).
- `readelf -l <file>` reports `Error: the PHDR segment is not covered by a LOAD segment`.
- The PHDR `VirtAddr` in `readelf -l` is `0xfffffffffff79040` (a
  sign-extended negative).

If you see any of these after a local wheel build or `BUNDLE_SYSDEPS`
build, suspect your host `patchelf`.

#### Compatible patchelf verion

The fix is [NixOS/patchelf PR #544](https://github.com/NixOS/patchelf/pull/544)
("Allocate PHT/SHT at the end of the ELF file"), merged 2025-01-07 to
master but not yet in a tagged release. Any supported build path needs a
`patchelf` that includes this commit.

#### Supported install paths

Pick whichever applies to your host:

1. **Portable / manylinux container.** If you build inside
   `ghcr.io/rocm/therock_build_manylinux_x86_64`, the image already ships
   a patched `patchelf` built from source and installed at
   `/usr/local/bin/patchelf`. Nothing to do. See
   [`dockerfiles/build_manylinux_x86_64.Dockerfile`][dockerfile].

1. **Fedora.** Recent Fedora releases ship the fix as a downstream patch
   on the packaged `patchelf 0.18.0` (the Fedora `patchelf` SRPM carries
   upstream PR #544 as `0001-Allocate-PHT-SHT-at-the-end-of-the-.elf-file.patch`).
   Verify with:

   ```bash
   rpm -q --changelog patchelf | head
   ```

   The changelog entry referencing the "PHT/SHT at the end" patch
   indicates a good build. `dnf install patchelf` is sufficient on a
   release that carries it.

1. **Any other Linux (Ubuntu, Debian, Arch, openSUSE, ...).** Build
   `patchelf` from source using the script the manylinux image uses:

   ```bash
   sudo env INSTALL_PREFIX=/usr/local ./dockerfiles/install_pinned_patchelf.sh

   patchelf --version
   # -> patchelf 0.18.0+therock.<short-ref>
   ```

   The script needs `curl`, `autoconf`, `automake`, `make`, and a C++
   compiler. On Ubuntu: `sudo apt install curl autoconf automake make g++`.

### Resource Utilization

ROCm is a very resource hungry project to build. The `compiler/amd-llvm` component alone involves linking multi-gigabyte binaries that can consume 4-8 GB of RAM per link job, and LLVM's configure+bootstrap phase is especially memory-intensive. On systems with a high core:memory ratio (e.g., 16+ cores with 32 GB RAM), Ninja's default `nproc`-level parallelism will frequently exceed available memory and get killed by `systemd-oomd` or the kernel OOM killer.

#### Controlling Build Parallelism

The most effective way to bound memory usage is to cap the number of concurrent build jobs. Note that `-j` passed to the outer Ninja/CMake invocation controls parallelism at the super-project level; subproject builds (e.g., `amd-llvm`) spawn their own Ninja instances and are not directly bounded by this setting. See [TheRock issue #XXXX](https://github.com/ROCm/TheRock/issues) for tracking a Ninja job server that would propagate limits into subprojects.

1. **Per-invocation via `ninja -j`:**

   ```bash
   # Use only 8 concurrent jobs at the super-project level (safe for 32 GB RAM)
   ninja -C build -j8

   # Or even lower for very memory-constrained systems
   ninja -C build -j4
   ```

1. **Via the `CMAKE_BUILD_PARALLEL_LEVEL` environment variable** (applies to any `cmake --build` invocation):

   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build build

   # Or export it persistently for the session:
   export CMAKE_BUILD_PARALLEL_LEVEL=8
   cmake --build build
   ```

1. **Via `NINJA_STATUS` to see real-time job counts** (helpful for debugging OOM):

   ```bash
   NINJA_STATUS="[%f/%t (%j running)] " ninja -C build
   ```

#### Choosing the right `-j` for your system

| RAM    | Cores | Recommended `-j` | Notes                                |
| ------ | ----- | ---------------- | ------------------------------------ |
| 16 GB  | 8+    | `-j4`            | Link steps will saturate RAM         |
| 32 GB  | 16    | `-j8`            | Leaves headroom for system + linker  |
| 32 GB  | 20+   | `-j8` to `-j10`  | More cores than RAM can safely serve |
| 64 GB+ | any   | `-j16` or higher | Link jobs still peak at ~8 GB each   |

If you observe OOM kills during the `amd-llvm` build, drop `-j` further. The OOM typically manifests as `ninja: build stopped: subcommand failed` with no compiler error — check `dmesg | tail -50` for `Out of memory: Killed process` entries.

#### Using ccache to reduce rebuild times

`ccache` dramatically speeds up incremental rebuilds (common when iterating on a single component) by caching compilation results. TheRock ships a project-aware ccache configuration:

```bash
# Initialize project-local ccache config (stored in .ccache/ within the repo)
eval "$(./build_tools/setup_ccache.py)"

# Pass compiler launchers to CMake
cmake -B build -GNinja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1032

# Build with limited parallelism
ninja -C build -j8
```

Monitor ccache effectiveness with `ccache -s` — on subsequent rebuilds you should see cache hit rates of 60-90% for incremental work.

#### Reducing build scope

If memory is tight and you only need a specific component, build that target
directly rather than the full stack. For example, to work on rocBLAS:

```bash
ninja -C build rocBLAS+build
```

Or configure with only the components you need enabled:

```bash
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPIFY=ON \
  -DTHEROCK_ENABLE_CORE=ON \
  -DTHEROCK_ENABLE_MATH_LIBS=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1032
```

See the top-level `CMakeLists.txt` for the full list of `THEROCK_ENABLE_*` options.

[dockerfile]: ../dockerfiles/build_manylinux_x86_64.Dockerfile
[rocm-prereqs]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html
