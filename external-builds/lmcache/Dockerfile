# Dockerfile for building LMCache wheels with ROCm support
# Uses TheRock's official manylinux build container
ARG BASE_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:latest
ARG PYTHON_VERSION=3.12

# ============================================================================
# Stage 1: Setup Python and install PyTorch with ROCm
# ============================================================================
FROM ${BASE_IMAGE} AS builder

ARG PYTHON_VERSION
ARG INDEX_URL

# Use specific Python version from manylinux
# TheRock container has cp312 in PATH by default, but we'll be explicit
ENV PYTHON=/opt/python/cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}/bin/python
ENV PATH=/opt/python/cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}/bin:${PATH}

# Install build dependencies (most are already in TheRock base)
RUN ${PYTHON} -m pip install --upgrade pip && \
    ${PYTHON} -m pip install ninja wheel setuptools setuptools_scm packaging

# Install PyTorch and ROCm Python packages from index URL
# The index URL provides Python-packaged ROCm for a specific gfx architecture
# This includes hipcc and other build tools needed for LMCache
# NEED ROCM FORK HERE
RUN ${PYTHON} -m pip install torch --index-url ${INDEX_URL}

# Verify PyTorch + ROCm
RUN ${PYTHON} -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA/HIP available: {torch.cuda.is_available()}'); \
    print(f'HIP version: {getattr(torch.version, \"hip\", \"N/A\")}')"

# ============================================================================
# Stage 2: Build LMCache
# ============================================================================
FROM builder AS build

ARG LMCACHE_VERSION=main
ARG PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201"
ARG MAX_JOBS=8

# Clone LMCache
RUN git clone https://github.com/LMCache/LMCache.git /workspace/lmcache && \
    cd /workspace/lmcache && \
    git checkout ${LMCACHE_VERSION}

WORKDIR /workspace/lmcache

# Set build environment for HIP/ROCm
# Note: With Python-packaged ROCm, we don't set ROCM_PATH because
# ROCm is installed in the Python environment, not /opt/rocm
ENV BUILD_WITH_HIP=1
ENV CXX=hipcc
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV TORCH_DONT_CHECK_COMPILER_ABI=1
ENV MAX_JOBS=${MAX_JOBS}

# Build wheel
RUN ${PYTHON} setup.py bdist_wheel --dist-dir=/output

# Basic import test
RUN ${PYTHON} -m pip install /output/lmcache-*.whl && \
    ${PYTHON} -c "import lmcache; print(f'✓ LMCache: {lmcache.__version__}')"

# ============================================================================
# Stage 3: Audit wheel for manylinux compliance
# ============================================================================
FROM build AS audit

RUN ${PYTHON} -m pip install auditwheel

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
