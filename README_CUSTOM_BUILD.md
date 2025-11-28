# ROCm 7.11 Custom Build for gfx103X (RDNA 2)

**Custom build by hashcat** - A personalized fork of [TheRock](https://github.com/ROCm/TheRock) configured for building ROCm 7.11 with optimized support for AMD RDNA 2 GPUs (gfx103X family).

This build focuses on native gfx103X support, eliminating the need for `HSA_OVERRIDE_GFX_VERSION` workarounds and providing first-class support for RDNA 2 architecture GPUs.

## Build Configuration

### Target GPUs

- **gfx1030**: AMD RX 6800 / RX 6800 XT
- **gfx1031**: AMD RX 6700 XT (primary test GPU)
- **gfx1032**: AMD RX 6600 / RX 6600 XT
- **gfx1035**: AMD Radeon 680M (Laptop iGPU)
- **gfx1036**: AMD Ryzen 7000 series iGPU

### Build Settings

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DTHEROCK_AMDGPU_TARGETS= \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx103X-all
```

## System Requirements

- **OS**: Fedora 43 (tested) or compatible Linux
- **RAM**: 32GB minimum (64GB recommended for parallel builds)
- **Swap**: 32GB+ recommended for memory-intensive compilation
- **Disk**: ~50GB for build directory
- **CPU**: Multi-core recommended (tested on Ryzen 9 5900X)

## Installation

### 1. Build ROCm 7.11

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=gfx103X-all \
  -DTHEROCK_AMDGPU_TARGETS= \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx103X-all

cmake --build build -j$(nproc)
```

### 2. Install to /opt/rocm

```bash
sudo cmake --install build --prefix /opt/rocm
```

### 3. Configure SELinux (Fedora/RHEL)

See [SELINUX_ROCM_SETUP.md](SELINUX_ROCM_SETUP.md) for complete instructions.

```bash
# Set correct SELinux contexts
sudo semanage fcontext -a -t lib_t "/opt/rocm(/.*)?"
sudo semanage fcontext -a -t bin_t "/opt/rocm/bin(/.*)?"
sudo semanage fcontext -a -t bin_t "/opt/rocm/llvm/bin(/.*)?"
sudo restorecon -Rv /opt/rocm
```

### 4. Update library cache

```bash
sudo sh -c 'echo -e "/opt/rocm/lib\n/opt/rocm/lib64\n/opt/rocm/lib/llvm/lib" > /etc/ld.so.conf.d/rocm.conf'
sudo ldconfig
```

### 5. Verify installation

```bash
/opt/rocm/bin/rocminfo
/opt/rocm/bin/rocm-smi
/opt/rocm/bin/hipcc --version
```

## Environment Configuration

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH
export HIP_PLATFORM=amd
export HIP_COMPILER=clang
export HSA_XNACK=0
export HSA_ENABLE_SDMA=0
```

**Note**: With ROCm 7.11, `HSA_OVERRIDE_GFX_VERSION` is no longer required for gfx103X GPUs!

## Testing with AI Frameworks

### Ollama

Rebuild Ollama against this ROCm build:

```bash
cd /path/to/ollama
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export CMAKE_ARGS="-DGGML_HIPBLAS=on -DAMDGPU_TARGETS=gfx1031"
go generate ./...
go build -v .
sudo cp ollama /usr/local/bin/
```

Tested with:

- Llama 3.2 3B (Q4_K_M) - Working ✓
- Native gfx1031 detection ✓

### LMStudio / llama.cpp

Works with bundled llama.cpp builds or rebuild from source against this ROCm installation.

## Custom Documentation

- [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) - Future optimization strategies with `-march=native`, LTO, PGO
- [SELINUX_ROCM_SETUP.md](SELINUX_ROCM_SETUP.md) - Complete SELinux configuration guide
- [ollama.service.new](ollama.service.new) - Systemd service file for Ollama with ROCm

## Key Differences from Upstream

1. **Target Focus**: Optimized for gfx103X family instead of multi-architecture
1. **Documentation**: Added SELinux, systemd, and optimization guides
1. **Testing**: Verified with Ollama, LMStudio, llama-server on Fedora 43

## Author

**hashcat** - Custom ROCm build maintainer

## Upstream

Original repository: https://github.com/ROCm/TheRock

## License

Same as upstream ROCm/TheRock project.

## Contributing

This is a personal fork optimized for specific hardware. For general ROCm issues, please refer to the upstream repository.

## Build Notes

- Build time: ~2-4 hours on Ryzen 9 5900X (12 cores)
- Peak memory usage: ~30GB (with swap)
- Final installation size: ~15GB in /opt/rocm

______________________________________________________________________

**Build Date**: 2025-11-28
**ROCm Version**: 7.11
**Test GPU**: AMD Radeon RX 6700 XT (gfx1031)
**Test OS**: Fedora Linux 43 (KDE Plasma)
