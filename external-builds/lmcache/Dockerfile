# Dockerfile for building LMCache wheels with ROCm support
# Uses TheRock's official manylinux build container
ARG BASE_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:latest

# ============================================================================
# Stage 1: Setup Python and install PyTorch with ROCm
# ============================================================================
FROM ${BASE_IMAGE} AS builder

ARG INDEX_URL
ARG PYTORCH_ROCM_ARCH


# Install build dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install ninja wheel setuptools setuptools_scm packaging

# Install Python build tools
RUN python3 -m pip install --no-cache-dir build auditwheel pybind11

# Install Python-packaged ROCm from index
# This includes architecture-specific libraries for the target GPU
RUN python3 -m pip install --no-cache-dir rocm[libraries,devel] --index-url ${INDEX_URL}


# Build and install rocThrust headers from source
# Python-packaged ROCm doesn't include Thrust headers, but PyTorch needs them
# Following official instructions from https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocthrust
# Set ROCM_PATH/HIP_PATH since Python-packaged ROCm is not in /opt/rocm
RUN PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    export ROCM_PATH="${PYTHON_SITE_PACKAGES}/_rocm_sdk_core" && \
    export HIP_PATH="${PYTHON_SITE_PACKAGES}/_rocm_sdk_core" && \
    git clone --no-checkout --depth=1 --filter=tree:0  https://github.com/ROCm/rocm-libraries.git /tmp/rocm-libraries && \
    cd /tmp/rocm-libraries && \
    git sparse-checkout init --cone && \
    git sparse-checkout set projects/rocthrust && \
    git checkout develop && \
    cd projects/rocthrust && \
    mkdir build && cd build && \
    CXX="${HIP_PATH}/bin/hipcc" cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DROCPRIM_FETCH_METHOD=DOWNLOAD && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/rocm-libraries

# Install PyTorch with ROCm from the gfx-specific index
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url ${INDEX_URL}

# Verify PyTorch + ROCm
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA/HIP available: {torch.cuda.is_available()}'); \
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
# 1. Change: license = "Apache-2.0" -> license = {text = "Apache-2.0"}
# 2. Remove: license-files (deprecated field)
RUN sed -i 's/^license = "Apache-2.0"/license = {text = "Apache-2.0"}/' pyproject.toml && \
    sed -i '/^license-files/d' pyproject.toml

# Set build environment for HIP/ROCm
# With Python-packaged ROCm, hipcc and libraries are in Python site-packages
ENV BUILD_WITH_HIP=1
ENV CXX=hipcc
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV TORCH_DONT_CHECK_COMPILER_ABI=1
ENV MAX_JOBS=${MAX_JOBS}

# Build wheel
RUN python3 setup.py bdist_wheel --dist-dir=/output

# Basic import test
RUN python3 -m pip install /output/lmcache-*.whl && \
    python3 -c "import lmcache; print(f'✓ LMCache: {lmcache.__version__}')"

# ============================================================================
# Stage 3: Audit wheel for manylinux compliance
# ============================================================================
FROM build AS audit

# Repair wheel for portability (exclude PyTorch and ROCm libs)
# These libraries come from Python packages installed via --extra-index-url
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
