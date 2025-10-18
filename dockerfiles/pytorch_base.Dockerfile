# Base and versions
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.12
ARG ROCM_VERSION
# URL to TheRock tarball that contains ROCm payload (optional)
ARG THEROCK_URL

# Release parameters (set defaults to silence warnings; overridden at build time)
ARG RELEASE_NUMBER=7.9.0rc20251008
ARG AMDGPU_FAMILY=gfx94X-dcgpu

ENV DEBIAN_FRONTEND=noninteractive

# Preconfigure timezone to avoid interactive prompt
RUN echo "tzdata tzdata/Areas select Etc" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/Etc select UTC" | debconf-set-selections

# Core tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget curl rsync dialog git software-properties-common \
      ca-certificates gnupg tar xz-utils bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Python toolchain:
# - Ubuntu 24.04 ships Python 3.12/3.13 in main; for 3.11 use deadsnakes.
RUN set -eux; \
  apt-get update; \
  if [ "${PYTHON_VERSION}" = "3.12" ]; then \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip; \
  elif [ "${PYTHON_VERSION}" = "3.11" ] || [ "${PYTHON_VERSION}" = "3.13" ]; then \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip; \
    if [ "${PYTHON_VERSION}" = "3.11" ]; then \
      apt-get install -y --no-install-recommends python3.11-distutils || true; \
    fi; \
  else \
    echo "Unsupported PYTHON_VERSION=${PYTHON_VERSION}. Use 3.11, 3.12, or 3.13." >&2; \
    exit 1; \
  fi; \
  apt-get clean; \
  rm -rf /var/lib/apt/lists/*
# Create and activate python virtual environment; upgrade pip/setuptools
RUN python${PYTHON_VERSION} -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Install ROCm from TheRock repo
# -----------------------------
RUN python${PYTHON_VERSION} -m venv /opt/venv && \
    git clone https://github.com/ROCm/TheRock.git && \
    /opt/venv/bin/python -m pip install -r TheRock/requirements.txt && \
    cd TheRock && \
    /opt/venv/bin/python build_tools/install_rocm_from_artifacts.py \
      --release ${RELEASE_NUMBER} \
      --amdgpu-family ${AMDGPU_FAMILY} \
      --output-dir /opt/rocm

# Environment config for tools
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib

# torch Arg
ARG TORCH_VERSION
ARG PACKAGE_INDEX_URL
ARG AMDGPU_FAMILY
# Install torch
RUN /opt/venv/bin/python -m pip install --no-cache-dir --index-url ${PACKAGE_INDEX_URL}${AMDGPU_FAMILY}/ "torch==${TORCH_VERSION}"
