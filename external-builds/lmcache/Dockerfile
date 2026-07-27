# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:a382085df3ba2419b58aa9051350883a0d0b732a4bc0a4ef60458f8161bb08c6

FROM ${BASE_IMAGE} AS builder

ARG PYTHON_ABI=cp312-cp312
ARG BUILD_DEVICE_ARCH=gfx942
ARG ROCM_INDEX_URL=https://rocm.prereleases.amd.com/whl-multi-arch/
ARG ROCM_VERSION=7.14.0rc3
ARG TORCH_VERSION=2.11.0+rocm7.14.0rc3

ENV PATH="/opt/python/${PYTHON_ABI}/bin:${PATH}"

RUN test -x "/opt/python/${PYTHON_ABI}/bin/python"

RUN python -m pip install --upgrade pip && \
    python -m pip install \
        auditwheel \
        build \
        ninja \
        packaging \
        pybind11 \
        "setuptools>=77,<81" \
        setuptools_scm \
        wheel

RUN python -m pip install \
        --no-cache-dir \
        --index-url "${ROCM_INDEX_URL}" \
        "rocm[libraries,devel,device-${BUILD_DEVICE_ARCH}]==${ROCM_VERSION}" \
        "torch[device-${BUILD_DEVICE_ARCH}]==${TORCH_VERSION}" && \
    rocm-sdk init

RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'HIP version: {torch.version.hip}')" && \
    rocm-sdk path --root

FROM builder AS build

ARG MAX_JOBS=8
ARG ROCM_VERSION
ARG ROCM_ARCHES=gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1250

WORKDIR /workspace/lmcache

# build_prod_wheels.py supplies the checked-out LMCache tree as the build context.
COPY . /workspace/lmcache

ENV BUILD_WITH_HIP=1
ENV CXX=hipcc
ENV MAX_JOBS=${MAX_JOBS}
ENV PYTORCH_ROCM_ARCH=${ROCM_ARCHES}
ENV TORCH_DONT_CHECK_COMPILER_ABI=1

RUN base_version="$(python -m setuptools_scm)" && \
    wheel_version="$( \
        BASE_VERSION="${base_version}" ROCM_VERSION="${ROCM_VERSION}" \
        python -c "import os; from packaging.version import Version; \
base = Version(os.environ['BASE_VERSION']).public; \
rocm = os.environ['ROCM_VERSION']; \
label = 'devrocm' if 'dev' in rocm else 'rocm'; \
print(f'{base}+{label}{str(Version(rocm)).replace(\"+\", \"-\")}')" \
    )" && \
    echo "Building LMCache wheel version ${wheel_version}" && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${wheel_version}" \
    ROCM_PATH="$(rocm-sdk path --root)" \
    python -m build \
        --wheel \
        --no-isolation \
        --outdir /output

RUN python -m pip install --no-deps /output/lmcache-*.whl && \
    cd /tmp && \
    python -c "import torch; import lmcache.c_ops; \
    import lmcache.native_storage_ops; import lmcache.lmcache_fs; \
    import lmcache.lmcache_redis; print('LMCache extensions imported successfully')"

FROM build AS audit

RUN auditwheel show /output/lmcache-*.whl && \
    auditwheel repair \
        --plat manylinux_2_28_x86_64 \
        --exclude libtorch.so \
        --exclude libtorch_hip.so \
        --exclude libtorch_python.so \
        --exclude libtorch_cpu.so \
        --exclude libc10.so \
        --exclude libc10_hip.so \
        --exclude libamdhip64.so.7 \
        --wheel-dir /wheelhouse \
        /output/lmcache-*.whl

FROM scratch AS export
COPY --from=audit /wheelhouse/*.whl /
