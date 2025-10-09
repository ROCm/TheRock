# Build Containers

## ManyLinux

The project aims to build on a wide variety of Linux operating systems and compilers, but when it comes to producing portable builds, we use EL containers based on the [manylinux](https://github.com/pypa/manylinux) project. This gives us the ability to produce binaries with wide compatibility by default to facilitate tarball distribution and embedding into other packages.

The CI uses a custom built [therock_build_manylinux_x86_64](https://github.com/ROCm/TheRock/pkgs/container/therock_build_manylinux_x86_64). See the [`dockerfiles/build_manylinux_x86_64.Dockerfile`](../../dockerfiles/build_manylinux_x86_64.Dockerfile). It is automatically rebuilt on pushes to `main`. For testing, changes can be pushed to a `stage/docker/BRANCH_NAME` branch, which will automatically result in a corresponding container rebuilt with `stage-BRANCH_NAME` tag in the package registry. See the dockerfile for instructions to build locally.
