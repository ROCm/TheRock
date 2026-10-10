# Build llama.cpp with ROCm Support

This integration adds **`llama.cpp`** as an external subproject to **TheRock**, enabling AMD GPU acceleration via the **ROCm** platform.

[`llama.cpp`](https://github.com/ggerganov/llama.cpp) is an open-source project providing a lightweight, efficient C/C++ implementation of **Large Language Model** architectures. It enables running LLMs locally on CPU and GPU with minimal dependencies. Designed for **portability**, **performance**, and **ease of integration**, it serves as an ideal backend for research, development, and embedded AI applications.

This document describes how to configure, build, and test `llama.cpp` as part of TheRock build system with ROCm acceleration enabled.

______________________________________________________________________

## Table of Contents

- [Support Status](#support-status)
- [Build Instructions](#build-instructions)
- [Running / Testing llama.cpp](#running--testing-llamacpp)

## Support Status

| Project / Feature | Linux Support   | Windows Support |
| ----------------- | --------------- | --------------- |
| **llama.cpp**     | ✅ Passes Tests | ✅ Passes Tests |

> [!NOTE]
> Building **rocWMMA** is currently **not implemented** in TheRock.

> [!WARNING]
> Certain tests are known to fail on **gfx1030** and **gfx1031** GPU targets.

## Build Instructions

By default, scripts use the **upstream** `llama.cpp` repository.
You can modify the source path or reference your own fork if desired.

### Prerequisites and Setup

The provided build scripts assume:

- You have already completed a **full ROCm build** using TheRock.
- You have the **`rocm-sdk`** tool installed, which is used internally to:
  - Locate the ROCm build directory.
  - Identify available AMD GPU targets.
- You have environemnt variables set up correctly for ROCm, including PATH updates to find ROCm libraries at runtime.

Before building, ensure the following:

- **ROCm is installed** and accessible.
- You have **Python 3.8+** installed and accessible.

> [!NOTE]
> The default path is determined automatically using `rocm-sdk path --root`

### Quickstart

Building `llama.cpp` is a two-step process:

1. **Clone the Repository**

   Use the `llamacpp_repo.py` script to download and prepare the `llama.cpp` sources:

   ```bash
   python llamacpp_repo.py
   ```

1. **Build with ROCm Support**

   Then build the project using the `llamacpp_build.py` script:

   ```bash
   python llamacpp_build.py
   ```

> [!NOTE]
> Check scripts for detailed explanation

The build process automatically detects supported AMD GPU targets and configures CMake.

## Running / Testing llama.cpp

After the build completes, the compiled binaries and shared libraries can be found under:

```
llama.cpp/build/bin
```

You can manually run the generated executables, but the recommended approach is to use the automated test runner.

> [!TIP]
> If you built `llama.cpp` in a custom directory, you can define an environment variable to make it easier to locate:
>
> ```bash
> export LLAMACPP_BUILD_DIR=/path/to/custom/build
> ```
>
> The scripts will automatically reference this variable.

### Test Execution

To verify functionality, use the **`llamacpp_test.py`** script.
It automatically discovers all test binaries in `build/bin`, runs them, and provides a clean summary.

Example:

```bash
# From external-builds/llama.cpp directory
python ./llamacpp_test.py
```

You can optionally save logs for later analysis:

```bash
python ./llamacpp_test.py --save-logging test_results.log
```

The test runner:

- Finds all valid test executables under the `build/bin` directory.
- Skips known failing or unsupported tests.
- Prints a concise summary with success/failure counts.

> [!NOTE]
> Refer to the source code of [llamacpp_test.py](./llamacpp_test.py) for detailed information on skip rules, supported arguments, and logging options.
