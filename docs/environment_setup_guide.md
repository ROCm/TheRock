# Environment Setup Guide

We only regularly build and test on certain OS combinations, but we aim to enable users wishing to build on a variety of systems, so long as they are relatively modern, have compatible dependencies, and do not create a support burden to accomodate. This page documents known workarounds and instructions for alternative environment setup. See the main project README for quick instructions on latest versions of certain popular distributions.

The advice on this page is not necessarily validated by the project maintainers. For any of these combinations that have known CI coverage, that will be noted. Otherwise, this is best effort information collected in the hope that it will help future users with niche issues.

If you have a configuration that you have found workarounds to support, please send a PR adding it to this page and we will consider including it for the benefit of future users.

## Primary Configurations

See the [project README](../README.md) for quick getting started instructions the following combinations:

- Fedora (TODO: looking for contribution)
- Ubuntu 24.04
- Windows (VS2022)

In general, we will keep the home page updated with quick start instructions for recent versions of the above. Additional advanced advice may be found below for specialty quirks and workarounds.

## Common Issues

### CMake

Different project components enforce different CMake version ranges. The `cmake_minimum_version` in the top level CMake file (presently 3.25) should be considered the project wide minimum. As of September 2025, CMake 4 is supported on Linux - but not on Windows.

There are various, easy ways to acquire specific CMake versions. For Windows and users wanting to use CMake 3, it can be easily installed with:

1. Be in your venv for TheRock:
   - Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\Activate.bat`
1. `pip install 'cmake<4'`
1. For Linux: if afterwards cmake is not found anymore, run `hash -r` to forget the previously cached location of cmake

### Resource Utilization

ROCm is a very resource hungry project to build. If running with high parallelism (i.e. on systems with a high core:memory ratio), it will likely use more memory than you have without special consideration. Sometimes this will result in a transient "resource exhausted" problem which clears on a restart. Sufficient swap and controlling concurrency may be necessary. TODO: Link to guide on how to control concurrency and resource utilization.

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

### Manylinux x84-64

Our open-source binaries are typically built within a [manylinux container](https://github.com/pypa/manylinux) (see [the docker file](../dockerfiles/build_manylinux_x86_64.Dockerfile)). These images are versioned by the glibc version they target, and if dependencies are controlled carefully, binaries built on them should work on systems with the same or higher glibc version.

Present version: glibc 2.28
Based on upstream: AlmaLinux 8 with gcc toolset 12

While this generally implies that the project should build on similarly versioned alternative EL distributions, do note that we install several upgraded tools (see dockerfile above) in our standard CI pipelines.

Reference image: `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:3ac188c17f88f08ce522297b616d8308361b8e9a9b31bcc3c3bbb9429b1efa6c`

### Ubuntu 22.04

Reference image: `ubuntu:22.04`

Workarounds:

- Shipping CMake is too old (3.22): see above advice for CMake
