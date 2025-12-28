#!/bin/bash
# Build environment configuration for PyTorch with ROCm 7.11 for gfx1031
# AMD Radeon RX 6700 XT

# ROCm paths
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export ROCM_VERSION=7.11.0
export HIP_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode

# Target architecture
export PYTORCH_ROCM_ARCH=gfx1031
export AMDGPU_TARGETS=gfx1031

# Build configuration
export USE_ROCM=ON
export USE_CUDA=OFF
export USE_MPI=OFF
export USE_NUMA=OFF

# Disable gfx1031-incompatible features
export USE_FLASH_ATTENTION=OFF
export USE_MEM_EFF_ATTENTION=OFF
# FBGEMM_GENAI: Attempted to enable but PyTorch 2.11 has dependency issues
# Error: 'ck/config.h' not found - Composable Kernel not built
# Works only with PyTorch 2.9, not 2.11+ (see: ROCm/TheRock#2056)
export USE_FBGEMM_GENAI=OFF

# Disable distributed training (RCCL not fully available)
export USE_NCCL=OFF
export USE_DISTRIBUTED=OFF

# Disable MIOpen (CMake config issues with custom install)
export USE_MIOPEN=OFF

# Disable mobile features (flatbuffers version mismatch)
export BUILD_LITE_INTERPRETER=OFF
export INTERN_DISABLE_MOBILE_INTERP=ON
export BUILD_MOBILE_AUTOGRAD=OFF
export BUILD_MOBILE_BENCHMARK=OFF
export BUILD_MOBILE_TEST=OFF

# Enable Linux features
export USE_GLOO=ON
export USE_KINETO=ON

# OpenBLAS configuration
export BLAS=OpenBLAS
export OpenBLAS_HOME=/opt/rocm/lib/host-math
export OpenBLAS_LIB_NAME=rocm-openblas

# CMake and PATH
export CMAKE_PREFIX_PATH="/opt/rocm/lib/cmake/miopen:/opt/rocm/lib/cmake/hip:/opt/rocm/lib/cmake/rocblas:/opt/rocm/lib/cmake/hipblas:/opt/rocm/lib/cmake:/opt/rocm"
export PATH=/opt/rocm/bin:$PATH

# Compiler flags for ROCm compatibility
export CXXFLAGS="-Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict -I/opt/rocm/lib/rocm_sysdeps/include -I/opt/rocm/include/roctracer"
export CPPFLAGS="-Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
export LDFLAGS="-L/opt/rocm/lib/rocm_sysdeps/lib"

# For aotriton libs
export PKG_CONFIG_PATH=/opt/rocm/lib/rocm_sysdeps/lib/pkgconfig
export LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib

# Performance optimization
export USE_CCACHE=1
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CCACHE_DIR=/home/hashcat/TheRock/external-builds/pytorch/ccache

# Build parallelism
export MAX_JOBS=10

# Version configuration
export PYTORCH_BUILD_VERSION=2.11.0a0+rocm7.11.gfx1031
export PYTORCH_BUILD_NUMBER=1

# UTF-8 encoding
export PYTHONUTF8=1

echo "✓ Build environment configured for gfx1031"
echo "  ROCM_HOME: $ROCM_HOME"
echo "  PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  CCACHE_DIR: $CCACHE_DIR"
