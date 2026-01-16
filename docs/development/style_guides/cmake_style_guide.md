# CMake style guide

## General recommendations

> [!TIP]
> The "Mastering CMake" book hosted at
> https://cmake.org/cmake/help/book/mastering-cmake/index.html is a good
> resource.

## Style guidelines

### CMake dependencies

See [dependencies.md](/docs/development/dependencies.md) for guidance on how to
add dependencies between subprojects and third party sources.

Note that within each superrepo
([rocm-systems](https://github.com/ROCm/rocm-systems),
[rocm-libraries](https://github.com/ROCm/rocm-libraries)), subprojects **must**
be compatible with one another at the same git commit, and TheRock enforces
this.
