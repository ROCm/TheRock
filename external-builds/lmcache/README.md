# Build LMCache with ROCm support

This directory provides tooling for building LMCache with ROCm Python wheels.

Table of contents:

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Installation](#installation)
- [Running/testing LMCache](#runningtesting-lmcache)

## Support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| LMCache           | ✅ Supported  | ❌ Not Supported |

## Build instructions

LMCache is a PyTorch extension that provides KV cache management for LLMs. It requires PyTorch with ROCm as a build dependency.

This build uses TheRock's official manylinux build container ([`therock_build_manylinux_x86_64`](https://github.com/ROCm/TheRock/pkgs/container/therock_build_manylinux_x86_64)). Inside the container, we:

1. Install ROCm as a Python package from the ROCm pre-releases index (`rocm[libraries,devel]`) and initialize it with `rocm-sdk init`
2. Install PyTorch with ROCm from the same index
3. Build LMCache against this ROCm environment
4. Use `auditwheel` to create a manylinux-compatible wheel

### Build Prerequisites

You need:

- Docker installed and running
- A ROCm pre-releases index URL for your GPU architecture

Find the latest index URLs at: https://rocm.prereleases.amd.com/whl/

Example index URL for gfx950: `https://rocm.prereleases.amd.com/whl/gfx950-dcgpu`

### Building the Wheel

Run the build script with your architecture's index URL:

```bash
python build_prod_wheels.py \
  --output-dir outputs \
  --rocm-index-url https://rocm.prereleases.amd.com/whl/gfx950-dcgpu \
  --rocm-arch gfx950
```

Build script options:

- `--output-dir`: Directory for output wheel (required)
- `--rocm-index-url`: ROCm pre-releases Python index URL for ROCm and PyTorch (required)
- `--python-version`: Target Python version (default: current Python version)
- `--rocm-arch`: GPU architectures to build for, semicolon-separated (default: `gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201`)
- `--lmcache-branch`: LMCache git branch/tag to build (default: `dev`)
- `--no-cache`: Build without using Docker cache

The build takes ~3-5 minutes and produces a wheel in the output directory (e.g., `lmcache-0.3.14.dev46-cp312-cp312-linux_x86_64.whl`).

## Installation

The LMCache wheel depends on PyTorch with ROCm. Install them together:

### Step 1: Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### Step 2: Install PyTorch with ROCm First

Install PyTorch from the ROCm pre-releases index (use `--index-url` to ensure you get the ROCm version):

```bash
pip install --index-url https://rocm.prereleases.amd.com/whl/gfx950-dcgpu torch
```

**Important**: Use the same index URL that you used for building. This ensures PyTorch and ROCm versions match what LMCache was compiled against.

### Step 3: Install LMCache Wheel

Now install LMCache with `--extra-index-url` to allow fetching other dependencies from PyPI:

```bash
pip install --extra-index-url https://rocm.prereleases.amd.com/whl/gfx950-dcgpu \
  outputs/lmcache-*.whl
```

## Running/testing LMCache

### Quick Test Script

Use the provided test script to verify the installation:

```bash
./test_lmcache.sh
```

This automatically sets up the required environment and runs basic tests.

### Manual Testing

When using LMCache in Python, activate your virtual environment and run:

```bash
source venv/bin/activate
python3
```

Then in Python:

```python
import lmcache
print("✓ LMCache imported successfully")

# Test C++ extensions
import lmcache.c_ops
print("✓ C++ extensions loaded successfully")

# Verify PyTorch has ROCm support
import torch
print(f"PyTorch: {torch.__version__}")
print(f"HIP version: {torch.version.hip}")
print(f"HIP available: {torch.cuda.is_available()}")
```


## Usage with LLM Frameworks

LMCache is typically used with vLLM or other LLM serving frameworks that support KV cache management. Ensure you set the `LD_LIBRARY_PATH` environment variable before starting your application.
