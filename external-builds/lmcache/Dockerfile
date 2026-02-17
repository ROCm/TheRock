# Dockerfile for building LMCache wheels with ROCm support
# Uses TheRock's ROCm tarball + PyTorch from TheRock's index
ARG BASE_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:latest

# ============================================================================
# Stage 1: Install ROCm from tarball and PyTorch from index
# ============================================================================
FROM ${BASE_IMAGE} AS builder

ARG THEROCK_INDEX_URL=https://rocm.prereleases.amd.com/whl/gfx950-dcgpu
ARG THEROCK_TARBALL_URL=https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx950-dcgpu-7.11.0.tar.gz
ARG PYTORCH_ROCM_ARCH=gfx950

# Install build dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install ninja wheel setuptools setuptools_scm packaging auditwheel pybind11

# Download and extract TheRock ROCm tarball
RUN echo "=== Downloading TheRock ROCm tarball ===" && \
    mkdir -p /opt/rocm && \
    cd /tmp && \
    curl -L -o rocm.tar.gz ${THEROCK_TARBALL_URL} && \
    echo "=== Extracting ROCm tarball ===" && \
    tar -xzf rocm.tar.gz -C /opt/rocm && \
    rm rocm.tar.gz && \
    echo "=== ROCm tarball extraction complete ===" && \
    ls -la /opt/rocm/

# Configure ROCm paths
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV ROCM_HOME=/opt/rocm
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm/share/rocm/cmake
ENV DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
ENV HIP_DEVICE_LIB_PATH=${DEVICE_LIB_PATH}

# Verify ROCm installation
RUN echo "=== ROCm verification ===" && \
    which hipcc && \
    hipcc --version && \
    echo "=== Verifying development files ===" && \
    ls -la /opt/rocm/lib/llvm/bin/clang++ && \
    echo "=== Checking for CMake files ===" && \
    find /opt/rocm -name "hipConfig.cmake" -o -name "hip-config.cmake" | head -5 && \
    echo "=== Checking for Thrust headers ===" && \
    ls -la /opt/rocm/include/thrust/ | head -10 || echo "Thrust headers not in tarball - will need to build"

# Install PyTorch with ROCm from TheRock's index
RUN python3 -m pip install --index-url ${THEROCK_INDEX_URL} torch torchaudio torchvision

# Verify PyTorch
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'HIP version: {getattr(torch.version, \"hip\", \"N/A\")}')"

# ============================================================================
# Stage 2: Build LMCache
# ============================================================================
FROM builder AS build

ARG LMCACHE_BRANCH=dev
ARG PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201"
ARG MAX_JOBS=8

# Clone LMCache
RUN git clone https://github.com/LMCache/LMCache.git /workspace/lmcache && \
    cd /workspace/lmcache && \
    git checkout ${LMCACHE_BRANCH}

WORKDIR /workspace/lmcache

# Fix pyproject.toml for modern setuptools compatibility
RUN sed -i 's/^license = "Apache-2.0"/license = {text = "Apache-2.0"}/' pyproject.toml && \
    sed -i '/^license-files/d' pyproject.toml

# Set build environment for HIP/ROCm
ENV BUILD_WITH_HIP=1
ENV CXX=hipcc
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV TORCH_DONT_CHECK_COMPILER_ABI=1
ENV MAX_JOBS=${MAX_JOBS}

# Build wheel
RUN python3 setup.py bdist_wheel --dist-dir=/output

# Basic import test
RUN python3 -m pip install /output/lmcache-*.whl && \
    python3 -c "import lmcache; print('✓ LMCache installed successfully')"

# ============================================================================
# Stage 3: Audit wheel for manylinux compliance
# ============================================================================
FROM build AS audit

# Repair wheel for portability
RUN auditwheel show /output/lmcache-*.whl && \
    auditwheel repair \
        --plat manylinux_2_28_x86_64 \
        --exclude libtorch.so \
        --exclude libtorch_python.so \
        --exclude libc10.so \
        --exclude libc10_hip.so \
        --exclude libamdhip64.so \
        -w /wheelhouse \
        /output/lmcache-*.whl || cp /output/lmcache-*.whl /wheelhouse/

# ============================================================================
# Stage 4: Export wheel
# ============================================================================
FROM scratch AS export
COPY --from=audit /wheelhouse/*.whl /
