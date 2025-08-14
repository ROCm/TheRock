# Build Containers

## ManyLinux

The project aims to build on a wide variety of Linux operating systems and compilers, but when it comes to producing portable builds, we use EL containers based on the [manylinux](https://github.com/pypa/manylinux) project. This gives us the ability to produce binaries with wide compatibility by default to facilitate tarball distribution and embedding into other packages.

The CI uses a custom built [therock_build_manylinux_x86_64](https://github.com/ROCm/TheRock/pkgs/container/therock_build_manylinux_x86_64). See the `/dockerfiles/build_manylinux_x86_64.dockerfile`. It is automatically rebuilt on pushes to `main`. For testing, changes can be pushed to a `test-docker-*` branch, which will result in a corresponding tag in the package registry. See the dockerfile for instructions to build locally.
