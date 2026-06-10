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

Reference image: `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:702a5133851e6d1daf1207d2c9fbb01c2667914a5b6dc5a01faeb3ce66ea6421`

### Ubuntu 22.04

Reference image: `ubuntu:22.04`

Workarounds:

- Shipping CMake is too old (3.22): see above advice for CMake

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

ROCm is a very resource hungry project to build. If running with high parallelism (i.e. on systems with a high core:memory ratio), it will likely use more memory than you have without special consideration. Sometimes this will result in a transient "resource exhausted" problem which clears on a restart. Sufficient swap and controlling concurrency may be necessary. TODO: Link to guide on how to control concurrency and resource utilization.

[dockerfile]: ../dockerfiles/build_manylinux_x86_64.Dockerfile
