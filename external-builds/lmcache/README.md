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

LMCache is a PyTorch extension that provides KV cache management for LLMs. It requires PyTorch with ROCm as a build dependency. 

This build uses TheRock's official manylinux build container ([`therock_build_manylinux_x86_64`](https://github.com/ROCm/TheRock/pkgs/container/therock_build_manylinux_x86_64)), which includes Python and build tools. We then install PyTorch with Python-packaged ROCm via pip.

### Prerequisites and setup

The build script requires that an index URL for Python-packaged ROCm be provided. This index URL should match your GPU's gfx architecture. PyTorch and ROCm Python packages (which include `hipcc` and other build tools needed by LMCache) are installed from this index.

Example index URLs:
- TheRock prerelease for gfx94X: `https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu`
- TheRock prerelease for gfx90a: `https://rocm.prereleases.amd.com/whl/gfx90a-dcgpu`
- PyTorch official with ROCm 6.2: `https://download.pytorch.org/whl/rocm6.2`

### Quickstart

Execute the build script to prepare a build container and produce the LMCache wheel artifact. The Dockerfile will clone LMCache inside the container.

Example:

```bash
python build_prod_wheels.py --output-dir outputs \
  --python-version 3.12 \
  --index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu
```

The build script has optional arguments for:
- `--python-version`: Target a specific python version (default: current Python version)
- `--lmcache-version`: Git ref/tag to build (default: main)
- `--image`: Use a specific TheRock build image (default: latest)
- `--rocm-arch`: GPU architectures to build for (default: common gfx targets)

The resulting wheel can then be installed like so:

```bash
python3.12 -m venv venv
. venv/bin/activate
pip install --extra-index-url https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu \
  outputs/lmcache-*.whl
```

Note the use of `--extra-index-url` instead of `--index-url` to accommodate resolution of non-PyTorch dependencies from the default PyPI index.

## Running/testing LMCache

After installation, you can verify LMCache works:

```python
import lmcache
print(f"LMCache version: {lmcache.__version__}")

# Test C++ extensions loaded
import lmcache.c_ops
print("✓ C++ extensions loaded successfully")
```

LMCache is typically used with vLLM or other LLM serving frameworks that support KV cache management.
