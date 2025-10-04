# Build UCCL with ROCm support

This directory provides tooling for building UCCL with ROCm Python wheels.

Table of contents:

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Running/testing UCCL](#runningtesting-uccl)
- [Development instructions](#development-instructions)

## Support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| UCCL              | ✅ Supported  | ❌ Not Supported |

## Build instructions

UCCL builds in an Ubuntu 22.04 docker container and outputs a
manylinux wheel to a directory named `wheelbase-therock` in the repo
on the host.

### Prerequisites and setup

The build container installs a specific version Python from PyPA, if
explicitly specified, or the host's python version otherwise.

The build script currently requires that TheRock index URL be provided
explicitly and uses that to install UCCL's ROCm prerequisites inside
the build container. The specific versions of those ROCm packages are
then recorded in the UCCL wheel's dependences.

### Quickstart

Building is done with a single command that will clone the UCCL
sources, launch a docker container, and perform the build in one go.

Example:

```bash
python build_prod_wheels.py --output-dir outputs \
  --index-url http://rocm.nightlies.amd.com/v2/gfx94X-dcgpu
```

Optional arguments for the name of the directory with cloned UCCL
sources (default `uccl`) and specific python version are provided.
