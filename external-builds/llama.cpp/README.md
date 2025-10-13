# Build llama.cpp with ROCm support

This integration adds `llama.cpp` as an external subproject to TheRock, enabling AMD GPU acceleration via ROCm.

`llama.cpp` is an open-source project that provides a lightweight, efficient C/C++ implementation of LLaMA (Large Language Model) models. It enables running these models locally on CPU and GPU with minimal dependencies, making it accessible for experimentation, research, and integration into custom applications. The project emphasizes portability and performance, supporting a wide range of platforms and hardware configurations.

## Support status

| Project / feature | Linux support       | Windows support |
| ----------------- | ------------------- | --------------- |
| llama.cpp         | ❌ Not Fully Tested | ❌ Not Tested   |

- Note: Building rocWMMA is currently not implemented in TheRock.

## Build instructions

### Prerequisites and setup

Before building, ensure the following:

- ROCm is installed (default path assumed: /opt/rocm).
- Python 3 is available.
- CMake is installed.
- You have access to an AMD GPU supported by ROCm (default target: gfx1100).

### Quickstart

1. Prepare llama.cpp

Use the provided script to clone and prepare the repository:

```bash
python /llamacpp_repo.py

# Optional arguments:
#
# --repo: Local path to clone or use the repository (default: ./llama.cpp)
# --repo-hashtag: Branch, tag, or commit to fetch (default: #amd-integration)
# --gitrepo-origin: Remote URL (default: https://github.com/ROCm/llama.cpp.git)
# --depth: Shallow fetch depth
# --jobs: Number of parallel fetch jobs (default: 10)
```

Check the script for detailed explanation.

2. Build llama.cpp
   Run the build script:

```bash
./llamacpp_build.py

# Optional arguments:
#
# --rocm-dir: Path to ROCm installation (default: /opt/rocm)
# --llama-dir: Path to llama.cpp source (default: ./llama.cpp)
# --build-dir: Path to build directory (default: ./llama.cpp/build)
# --cmake-build-type: CMake build type (default: Release)
# --gpu-targets: AMD GPU targets (default: gfx1100)
# --jobs: Number of parallel build jobs
# --llama-curl: Enable libcurl support
# --llama-openssl: Enable OpenSSL support
# --llama-llguidance: Enable LLGuidance support
```

## Sanity Check

TODO: Not Implemented yet.

- Verify that the expected binaries are present in the build directory.
- Use ldd to confirm that shared library dependencies are correctly linked.
- Run a minimal test using a small model to validate basic functionality.

## Detailed Test

TODO: Not Implemented yet.

Execute all available test binaries located in llama.cpp/build/bin/:

```bash
./llama.cpp/build/bin/test-*
```
