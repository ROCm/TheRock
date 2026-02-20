# Dockerfile for building LMCache wheels with ROCm support
# Uses ROCm + PyTorch from ROCm pre-releases index
ARG BASE_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:latest

# ============================================================================
# Stage 1: Install ROCm and PyTorch from index
# ============================================================================
FROM ${BASE_IMAGE} AS builder

ARG ROCM_INDEX_URL=https://rocm.prereleases.amd.com/whl/gfx950-dcgpu
ARG PYTORCH_ROCM_ARCH=gfx950

ENV PATH=/opt/rocm/bin:$PATH

# Install build dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install ninja wheel setuptools setuptools_scm packaging auditwheel pybind11

# Install ROCm from ROCm pre-releases index
RUN python3 -m pip install --no-cache-dir --index-url ${ROCM_INDEX_URL} "rocm[libraries,devel]" && \
    rocm-sdk init

ENV ROCM_PATH=/opt/python/cp312-cp312/lib/python3.12/site-packages/_rocm_sdk_devel
ENV PATH=/opt/python/cp312-cp312/bin:$PATH

# Install PyTorch with ROCm from ROCm pre-releases index
RUN python3 -m pip install --index-url ${ROCM_INDEX_URL} torch torchaudio torchvision

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
    python3 -c "import lmcache.c_ops; print('✓ LMCache installed successfully')"

# ============================================================================
# Stage 3: Audit wheel for manylinux compliance
# ============================================================================
FROM build AS audit

# Repair wheel for portability
RUN auditwheel show /output/lmcache-*.whl && \
    auditwheel repair \
        --plat manylinux_2_28_x86_64 \
        -w /wheelhouse \
        /output/lmcache-*.whl || cp /output/lmcache-*.whl /wheelhouse/

# ============================================================================
# Stage 4: Export wheel
# ============================================================================
FROM scratch AS export
COPY --from=audit /wheelhouse/*.whl /