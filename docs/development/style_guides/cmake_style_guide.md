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

### Put build and test behavior in the build system

Keep shared build and test behavior where both local development and CI use it:

- **CMake presets ([`CMakePresets.json`](../../../CMakePresets.json)):** Named configurations with reusable
  cache variables, generators, and build/test settings.
- **CMake source files ([`CMakeLists.txt`](../../../CMakeLists.txt)):** Build targets, dependencies, test
  registration, and settings computed from build machine properties.
- **CMake configure/build helpers:** Logic that translates higher-level inputs
  into build configuration or coordinates build commands. For example,
  [`build_tools/configure_stage.py`](../../../build_tools/configure_stage.py)
  generates CMake arguments for selected stages or artifacts.

GitHub Actions workflows should invoke these shared interfaces, with overrides
only where CI requires them. Developers should not need to copy calculations
from YAML to get the same build and test behavior locally. See
[Avoid inline logic; call script files instead](github_actions_style_guide.md#avoid-inline-logic-call-script-files-instead)
in the GitHub Actions style guide.
