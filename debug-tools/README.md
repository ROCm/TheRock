# Debug Tools

This is the home of the TheRock's build config for ROCm debug tools, which
include ROCgdb, ROCdbgapi, and the ROCr debug agent.

The source code for these components is not hosted in TheRock itself. ROCdbgapi
and the ROCr debug agent live in the rocm-systems super-repo, while ROCgdb
lives in its own repository. TheRock references all of them via git submodule
pointers; `debug-tools/rocgdb/source` is the submodule pointer to the ROCgdb
sources, and the rocm-systems submodule provides the other two.

## Structure

The submodule pointers to the debug tools sources are organized as follows:

```
debug-tools/rocgdb/source:                  Submodule pointer to the ROCgdb source code.
rocm-systems/projects/rocdbgapi:            ROCdbgapi source code (via the rocm-systems submodule).
rocm-systems/projects/rocr-debug-agent:     ROCr debug agent source code (via the rocm-systems submodule).
```

## Developer's Guide

This guide covers configuration options and development workflows for building
and debugging the debug-tools components.

**Note:** This guide assumes all commands are executed from within a manylinux
container. TheRock can also be built outside of a container, but this requires
distro-specific environment configuration. For container setup, see
[Container Environments](#build-container).

Outside of the container, TheRock sources can be cloned with the following command:

```bash
# Clone the repository
git clone https://github.com/ROCm/TheRock.git
cd TheRock
```

### Quick Start

For users that just need a prebuilt ROCm stack, please refer to the [releases page](../RELEASES.md).

For users and developers that need to build the ROCm stack and debug-tools from
sources, the default configuration works out of the box:

```bash
cmake -B /therock/output/build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES="gfx942" \
  /therock/src
cmake --build /therock/output/build -t rocgdb amd-dbgapi rocr-debug-agent
```

**Note:** `-DTHEROCK_TEST_AMDGPU_FAMILIES` can optionally be supplied to restrict
the number of targets for which rocr-debug-agent-tests is built, as these tests
are device-specific. If not specified, tests will be built for all available gfx
targets.

For development and debugging, you may want to build in Debug mode by passing
the following option:

```bash
  -DDEBUG_TOOLS_BUILD_TYPE=Debug
```

### Component Dependencies

The debug-tools components have dependencies on other ROCm components and system
libraries. The handling of system library dependencies is controlled by the
`THEROCK_BUNDLE_SYSDEPS` CMake option.

| Component        | ROCm Dependencies        | Bundled System Dependencies (THEROCK_BUNDLE_SYSDEPS=ON) | System Dependencies (THEROCK_BUNDLE_SYSDEPS=OFF) |
| ---------------- | ------------------------ | ------------------------------------------------------- | ------------------------------------------------ |
| amd-dbgapi       | amd-comgr                | libbacktrace                                            | System libbacktrace                              |
| rocr-debug-agent | amd-dbgapi, ROCR-Runtime | elfutils                                                | System elfutils                                  |
| rocgdb           | amd-dbgapi               | bzip2, gmp, mpfr, expat, ncurses, liblzma, zlib, zstd   | System versions of these libraries               |

**THEROCK_BUNDLE_SYSDEPS behavior:**

- **ON** (default): TheRock builds and bundles system dependencies, ensuring
  consistent versions across different platforms and avoiding conflicts with
  system libraries.
- **OFF**: Components link against system-provided libraries. This reduces build
  time but requires that the necessary development packages are installed on the
  build system.

**Note:** When `THEROCK_BUNDLE_SYSDEPS=OFF`, ensure the required development
libraries listed in the table above are installed on your system.

### Configuration Variables

The debug-tools build system provides several CMake configuration variables
for customizing build behavior.

#### DEBUG_TOOLS_BUILD_TYPE

Set the build type for all debug-tools components. This variable propagates to
`amd-dbgapi`, `rocr-debug-agent`, and `rocgdb`.

```bash
  -DDEBUG_TOOLS_BUILD_TYPE=Debug
```

This is equivalent to:

```bash
  -Damd-dbgapi_BUILD_TYPE=Debug
  -Drocr-debug-agent_BUILD_TYPE=Debug
  -Drocgdb_BUILD_TYPE=Debug
```

Individual component build type variables take precedence over
`DEBUG_TOOLS_BUILD_TYPE`. For example, if you set
`-DDEBUG_TOOLS_BUILD_TYPE=Debug` and `-Drocgdb_BUILD_TYPE=Release`, ROCgdb will
build in Release mode while other components build in Debug mode.

#### DEBUG_TOOLS_C_FLAGS_DEBUG and DEBUG_TOOLS_CXX_FLAGS_DEBUG

Customize debug flags for C and C++ code when building in Debug mode. These
provide fine-grained control over optimization levels and debug information.

```bash
  -DDEBUG_TOOLS_BUILD_TYPE=Debug
  -DDEBUG_TOOLS_C_FLAGS_DEBUG="-O0 -ggdb"
  -DDEBUG_TOOLS_CXX_FLAGS_DEBUG="-O0 -ggdb"
```

**Common use cases:**

Disable optimization for debugging:

```bash
-DDEBUG_TOOLS_C_FLAGS_DEBUG="-O0 -ggdb"
-DDEBUG_TOOLS_CXX_FLAGS_DEBUG="-O0 -ggdb"
```

Maximum debug information:

```bash
-DDEBUG_TOOLS_C_FLAGS_DEBUG="-O0 -g3"
-DDEBUG_TOOLS_CXX_FLAGS_DEBUG="-O0 -g3"
```

Debug with minimal optimization for better runtime performance:

```bash
-DDEBUG_TOOLS_C_FLAGS_DEBUG="-Og -ggdb"
-DDEBUG_TOOLS_CXX_FLAGS_DEBUG="-Og -ggdb"
```

These flags are appended to component-specific `CMAKE_C_FLAGS_DEBUG` and
`CMAKE_CXX_FLAGS_DEBUG` and only apply when the build type is `Debug`.

**Note:** When `CMAKE_BUILD_TYPE` is set globally, by default the debug-tools
will honors the Debug compilation flags provided by TheRock. Specifically for
rocgdb this includes -O0 and -g3. These additional options are meant to give
more control to the debug-tools developer.

#### THEROCK_SHARED_PYTHON_EXECUTABLES

ROCgdb supports Python scripting integration, which requires linking against a
shared `libpython` library. By default, TheRock automatically detects suitable
Python installations during configuration.

**Automatic detection:**

TheRock searches for Python installations with shared library support and
displays the detected versions:

```
-- Building rocgdb against the following libpython versions: /usr/bin/python3.10;/usr/bin/python3.11
```

If no suitable Python is found:

```
-- No Python with shared libpython found. rocgdb will be built without Python support.
```

In this mode it is recommended to use an activated Python virtual environment with
the desired Python version.

**Manual configuration:**

Override automatic detection by specifying Python executable(s) explicitly:

```bash
# Single Python version
  -DTHEROCK_SHARED_PYTHON_EXECUTABLES=/opt/python3.11/bin/python3
```

```bash
# Multiple Python versions
  -DTHEROCK_SHARED_PYTHON_EXECUTABLES="/usr/bin/python3.10;/usr/bin/python3.11"
```

**Requirements:**

The Python executable must support a shared `libpython` library (e.g.,
`libpython3.11.so`). Static-only Python builds are not supported.

To check if your Python has shared library support:

```bash
ldd $(which python3) | grep libpython
```

If this shows a `libpython*.so` file, the Python installation is suitable.

#### External Source Directories

For development workflows, you can use a modified source directory instead of
the default submodule locations. This is useful for testing local changes.

**Available variables:**

- `THEROCK_USE_EXTERNAL_AMD_DBGAPI` and `THEROCK_AMD_DBGAPI_SOURCE_DIR`
- `THEROCK_USE_EXTERNAL_ROCR_DEBUG_AGENT` and `THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR`
- `THEROCK_USE_EXTERNAL_ROCGDB` and `THEROCK_ROCGDB_SOURCE_DIR`

**Example options:**

```bash
# Use external ROCgdb and dbgapi sources
  -DTHEROCK_USE_EXTERNAL_ROCGDB=ON
  -DTHEROCK_ROCGDB_SOURCE_DIR=/path/to/my/rocgdb
  -DTHEROCK_USE_EXTERNAL_AMD_DBGAPI=ON
  -DTHEROCK_AMD_DBGAPI_SOURCE_DIR=/path/to/my/amd-dbgapi
```

A warning message will be displayed during configuration when using external
source directories.

### Container Environments

TheRock debug-tools development and testing use two Docker containers with
different purposes.

#### Build Container

For building debug-tools components, use the TheRock build container:

```bash
docker run --rm -i -t \
  --mount type=bind,src=$HOME/therock/output,dst=/therock/output \
  --mount type=bind,src=$HOME/therock,dst=/therock/src \
  --name $(whoami) ghcr.io/rocm/therock_build_manylinux_x86_64:latest \
  /bin/bash
```

This container includes all necessary build dependencies and toolchains for
building TheRock components.

**Note:** CI builds use a pinned image version rather than `:latest`. When using
`:latest`, you may encounter different behavior or updated dependencies compared
to what CI uses.

For additional information on the manylinux image, see [dockerfiles/README.md](../dockerfiles/README.md#build_manylinux_dockerfile).

**GPU access for testing:**

While the build itself does not depend on a GPU, you can enable GPU access within
the container for testing after a build by adding these options to the `docker run`
command:

```bash
  --ipc host \
  --group-add video \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE
```

**Note:** `--security-opt seccomp=unconfined` and `--cap-add=SYS_PTRACE` are
required for debug-tools testing, as debuggers need ptrace capabilities.

**Environment setup within the container:**

Before building, set up the Python virtual environment and ccache:

```bash
# Create and activate virtual environment
python3 -m venv ~/v3
source ~/v3/bin/activate

# Install Python dependencies
pip install -r /therock/src/requirements.txt
pip install -r /therock/src/requirements-test.txt

# Initialize ccache
eval "$(/therock/src/build_tools/setup_ccache.py --init)"
```

Please refer to the top level [README.md](../README.md) file for more general
setup steps and additional information.

**Building debug-tools within the container:**

When building only debug-tools:

```bash
# Configure with ccache and resource limits
cmake -B /therock/output/build -GNinja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DFLANG_PARALLEL_COMPILE_JOBS=20 \
  -DLLVM_PARALLEL_LINK_JOBS=20 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_DEBUG_TOOLS=ON \
  -DTHEROCK_AMDGPU_FAMILIES="<gfx family>" \
  /therock/src

# Build debug-tools components
cmake --build /therock/output/build
```

For building the entirely of the ROCm stack it is enough to pass
`THEROCK_ENABLE_ALL=ON`.

**Configuration notes:**

- **ccache**: Using `-DCMAKE_C_COMPILER_LAUNCHER=ccache` and
  `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` enables compilation caching for faster
  rebuilds.
- **Resource limits**: `-DFLANG_PARALLEL_COMPILE_JOBS=20` and
  `-DLLVM_PARALLEL_LINK_JOBS=20` limit parallel compilation and linking to
  prevent memory and CPU exhaustion.
- **Debug-tools only**: When building only debug-tools, use
  `-DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_DEBUG_TOOLS=ON` to avoid building
  unnecessary components.

#### Testing Container

For testing ROCgdb and rocr-debug-agent, use a dedicated Ubuntu testing
container. This container is purely for testing purposes and should not be used
for building.

```bash
docker run --rm -i -t \
  --ipc host \
  --group-add video \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --mount type=bind,src=$HOME/therock/output,dst=/therock/output \
  --mount type=bind,src=$HOME/therock,dst=/therock/src \
  --name $(whoami) ghcr.io/rocm/no_rocm_image_ubuntu24_04_rocgdb:latest \
  /bin/bash
```

This container is specifically configured for testing debug tools. It uses
`no_rocm_image_ubuntu24_04_rocgdb:latest` instead of the base
`no_rocm_image_ubuntu24_04:latest` because it includes additional testing tools
required by ROCgdb, such as dejagnu (for the GDB test suite), gfortran (for
Fortran debugging tests), and other utilities needed for comprehensive debugger
testing. No builds are performed in this container.

**Note:** CI tests use a pinned image version rather than `:latest`. To exactly
reproduce CI test results, use the same pinned version that CI uses. When using
`:latest`, you may encounter different behavior or updated testing tools compared
to the CI environment.

For additional information on the testing container image, see [dockerfiles/README.md](../dockerfiles/README.md#no_rocm_image_dockerfile).

For testing reproduction steps, see [docs/development/test_environment_reproduction.md](../docs/development/test_environment_reproduction.md).

### Building with Sanitizers

TheRock provides built-in support for compiler sanitizers to help detect memory errors, data races, and other runtime bugs during development. For comprehensive documentation on sanitizer modes (`ASAN`, `HOST_ASAN`, `TSAN`), configuration options, CMake presets, and implementation details, see [docs/development/sanitizers.md](../docs/development/sanitizers.md).

#### Debug-Tools Specific Configuration

Enable sanitizers for debug-tools components using either project-wide or per-component settings:

```bash
# Project-wide
-DTHEROCK_SANITIZER=HOST_ASAN

# Per-component
-Damd-dbgapi_SANITIZER=HOST_ASAN
-Drocr-debug-agent_SANITIZER=HOST_ASAN
-Drocgdb_SANITIZER=HOST_ASAN

# Combined with debug builds
-DDEBUG_TOOLS_BUILD_TYPE=Debug
-DTHEROCK_SANITIZER=HOST_ASAN
```

#### Runtime Environment Configuration

**Address Sanitizer (ASAN_OPTIONS):**

```bash
export ASAN_OPTIONS=detect_leaks=1:verbosity=1:log_path=/tmp/asan_log
```

Common ASAN_OPTIONS:

- `detect_leaks=1` - Enable memory leak detection (enabled by default)
- `verbosity=N` - Increase diagnostic verbosity (0-2)
- `check_initialization_order=1` - Detect initialization order bugs
- `detect_stack_use_after_return=1` - Detect use-after-return bugs (more expensive)
- `halt_on_error=1` - Stop on first error
- `log_path=<path>` - Write logs to file instead of stderr

**Thread Sanitizer (TSAN_OPTIONS):**

```bash
export TSAN_OPTIONS=verbosity=1:second_deadlock_stack=1:history_size=7
```

Common TSAN_OPTIONS:

- `verbosity=N` - Increase diagnostic verbosity (0-2)
- `halt_on_error=1` - Stop on first error
- `second_deadlock_stack=1` - Show second stack trace for deadlocks
- `history_size=N` - Size of per-thread history buffer (default 2, max 7)
- `report_bugs=0` - Disable bug reporting (useful for performance testing)
- `log_path=<path>` - Write logs to file instead of stderr

### Testing Debug-Tools

TheRock provides test scripts for ROCgdb and ROCr Debug Agent. Both scripts work with locally-built ROCm trees or downloaded nightly tarballs from https://rocm.nightlies.amd.com/.

**Scripts location:** `build_tools/github_actions/test_executable_scripts/`

#### Setup

Set environment variables pointing to your ROCm installation:

```bash
# For TheRock builds
export THEROCK_BIN_DIR=/therock/output/build/dist/rocm/bin
export OUTPUT_ARTIFACTS_DIR=/therock/output/build/dist/rocm

# For nightly tarballs
export THEROCK_BIN_DIR=/path/to/rocm-developer-nightly/bin
export OUTPUT_ARTIFACTS_DIR=/path/to/rocm-developer-nightly
```

For container testing, use the testing container described in [Container Environments](#testing-container).

#### Running Tests

**ROCgdb:**

```bash
python3 build_tools/github_actions/test_executable_scripts/test_rocgdb.py
```

The script runs the GDB test suite against ROCgdb using both GCC and LLVM compilers. By default it will run the gdb.rocm and gdb.dwarf2 test categories.

**ROCr Debug Agent:**

```bash
python3 build_tools/github_actions/test_executable_scripts/test_rocr-debug-agent.py
```

The script includes automatic retry logic for flaky GPU-dependent tests.

#### Results

Both scripts report pass/fail status and exit with code 0 on success. ROCgdb tracks expected failures in the script's `XFAILED_TESTS` dictionary and reports:

- Per-compiler results (GCC and LLVM)
- Flaky tests (passed on retry)
- Unexpected failures
- Comparison of compiler-specific failures

## Additional information

### ROCgdb dependency on terminfo for TUI mode

ROCgdb’s TUI (Text User Interface) mode uses ncurses. This library relies on
finding a valid terminfo database to function properly.

TheRock builds its own ncurses library, which includes the terminfo
database. However, because ncurses does not provide a way to specify a
relative path to the database at configure/build time, it is important to
understand how the terminfo lookup works to ensure TUI mode remains functional.

If ROCgdb is launched via its launcher shell script (bin/rocgdb), the script
automatically points to the database using the TERMINFO environment variable.

If the ROCgdb binary is invoked directly and TERMINFO is not set, the terminfo
lookup logic checks the following paths in order:

- `$HOME/.terminfo`
- `/usr/share/terminfo`
- `/usr/lib/terminfo`
- `/etc/terminfo`
- `/lib/terminfo`
- The build-time prefix path (which may no longer be accessible)

If the lookup fails to return any valid entries, ncurses provides the following
terminal types as compiled-in fallbacks:

- xterm / xterm-256color
- vt100
- linux
- screen / screen-256color
- tmux / tmux-256color
- ansi

If the user's system terminal type does not match any of these fallbacks, the
user must install a terminfo dependency in one of the lookup paths listed
above. Otherwise, the TUI mode in ROCgdb will remain unavailable.

## Configuration Variables

### THEROCK_ROCGDB_UPSTREAM_BUILD

- **Default**: `OFF` - Build ROCm's rocgdb branch with AMD-specific files (NOTICES.txt, roccoremerge)
- **Non-default**: `ON` - Build upstream GDB master, skip AMD-specific files not yet in upstream

Usage: `-DTHEROCK_ROCGDB_UPSTREAM_BUILD=ON`

### THEROCK_ROCGDB_DOWNLOAD_CI_SCRIPT

- **Default**: `OFF` - Require `.github/scripts/test_rocgdb.py` to exist in the rocgdb source tree; configure fails if it is missing.
- **Non-default**: `ON` - If `test_rocgdb.py` is missing from the source tree, download it from `ROCm/ROCgdb` `amd-staging` at configure time (with retries) and install it into `tests/rocgdb`.

Use this when building a rocgdb source tree (e.g. upstream master) that is not part of the branches exercised by TheRock CI.

Usage: `-DTHEROCK_ROCGDB_DOWNLOAD_CI_SCRIPT=ON`

**Note:** `curl` must be available on the system when this option is enabled, as
it is used to download `test_rocgdb.py` at configure time. Most environments
provide `curl` by default, but it may need to be installed otherwise.
