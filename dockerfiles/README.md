# Dockerfiles for TheRock

This directory is home to the Dockerfiles we use as part of ROCm development, as
well as supporting scripts.

## Available Dockerfiles

### Manylinux Dockerfiles for building ROCm

- [`build_manylinux_x86_x64`](build_manylinux_x86_64.Dockerfile)
- [`build_manylinux_rccl_x86_64.Dockerfile`](build_manylinux_rccl_x86_64.Dockerfile)

<!-- TODO: document properties of these files

* https://github.com/pypa/manylinux
* Used to build packages (native or Python) that are compatible with any Linux
  distro using glibc $VERSION or later

 -->

<!-- TODO: document requirements for what goes in these files

* Generally no -dev packages (like... example?) as these can leak into source
  builds so they produces packages that are no longer portable to multiple
  operating systems without extra system dependencies installed
  (see third-party/sysdeps)
* Minimum versions of tools we use (CMake, ninja, etc.)
* Development _tools_ like PyYAML are okay, so long as they don't affect
  packaging

 -->

### "No ROCm" Dockerfiles for testing ROCm

- [`no_rocm_image_ubuntu24_04.Dockerfile`](no_rocm_image_ubuntu24_04.Dockerfile)

<!-- TODO: document properties of these files

* https://github.com/pypa/manylinux
* Used to build packages (native or Python) that are compatible with any Linux
  distro using glibc $VERSION or later

 -->

<!-- TODO: document requirements for what goes in these files

* no system install of ROCm
* minimal other system dependencies (especially -dev packages), to not hide
  packaging issues
* some extra tools like `lit` for LLVM testing are okay

 -->

## Using published images

<!-- TODO: document ghcr hosting, tags (e.g. `main`), -->

## Working on the Dockerfiles themselves

### Testing and debugging

<!-- TODO: document building locally -->

<!-- TODO: document running locally (mount options, entrypoints, etc.) -->

<!-- TODO: document building on CI with a custom tag then testing on CI -->

### Automated image publishing

<!-- TODO: document workflows:

.github\workflows\publish_build_manylinux_x86_64.yml
.github\workflows\publish_dockerfile.yml
etc.

-->

### Updating images used by GitHub Actions workflows

<!-- TODO: document commit sequence

1. test locally and/or on CI
2. commit Dockerfile updates
3. wait for workflows to build new images
4. commit Dockerfile pin updates in workflow files

 -->
