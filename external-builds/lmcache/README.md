# Build LMCache with ROCm support

This directory provides tooling for building LMCache with ROCm Python wheels.

Table of contents:

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Running/testing LMCache](#runningtesting-lmcache)

## Support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| LMCache           | ✅ Supported  | ❌ Not Supported |

## Build instructions

LMCache is a PyTorch extension that provides KV cache management for LLMs.
The build uses TheRock's manylinux build container, pulling ROCm and PyTorch
from the ROCm pre-releases index inside Docker.

### Prerequisites

You need:

- Docker installed and running
- A ROCm pre-releases index URL for your GPU architecture

Find the latest index URLs at: https://rocm.prereleases.amd.com/whl/

### Quickstart

Build the wheel by providing your GPU architecture's index URL:

```bash
python build_prod_wheels.py \
  --output-dir outputs \
  --rocm-index-url https://rocm.prereleases.amd.com/whl/gfx950-dcgpu \
  --rocm-arch gfx950
```

The build takes ~3-5 minutes and produces a manylinux-compatible wheel in the
output directory (e.g., `outputs/lmcache-0.3.14.dev46-cp312-cp312-manylinux_2_28_x86_64.whl`).

The resulting wheel can then be installed like so:

```bash
python3.12 -m venv venv
. venv/bin/activate
pip install --extra-index-url https://rocm.prereleases.amd.com/whl/gfx950-dcgpu \
  outputs/lmcache-*.whl
```

Note the use of `--extra-index-url` instead of `--index-url` to allow
resolution of non-ROCm dependencies (e.g., `safetensors`, `transformers`) from
the default PyPI index.

### Build script options

| Option | Description |
| --- | --- |
| `--output-dir` | Directory for the output wheel (required) |
| `--rocm-index-url` | ROCm pre-releases index URL for ROCm and PyTorch (required) |
| `--rocm-arch` | GPU architectures, semicolon-separated (default: `gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201`) |
| `--python-version` | Target Python version (default: current Python version) |
| `--lmcache-branch` | LMCache git branch/tag to build (default: `dev`) |
| `--no-cache` | Build without Docker layer cache |

## Running/testing LMCache

Use the provided test script to verify the installation:

```bash
./test_lmcache.sh
```

Or verify manually in Python:

```python
import lmcache
print("✓ LMCache imported successfully")

import lmcache.c_ops
print("✓ C++ extensions loaded successfully")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"HIP version: {torch.version.hip}")
print(f"HIP available: {torch.cuda.is_available()}")
```
