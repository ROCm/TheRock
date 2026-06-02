# Dockerfile for Spack integration on Ubuntu 24.04

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies in one layer
RUN apt-get update -y && apt-get install -y \
    sudo \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    ca-certificates \
    gnupg \
    lsb-release \
    file \
    unzip \
    patch \
    gfortran \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create spack user with passwordless sudo
RUN useradd -m -s /bin/bash -U -G sudo spack && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER spack
WORKDIR /home/spack

ENV SPACK_ROOT=/home/spack/spack \
    SPACK_ENV_NAME=rocm
ENV PATH="${SPACK_ROOT}/bin:${PATH}"

# Clone Spack and the ROCm package repository
RUN git clone --depth 1 https://github.com/spack/spack.git ${SPACK_ROOT} && \
    git clone -b rocm-spack-ci-changes --depth 1 https://github.com/ROCm/rocm-spack-packages.git /home/spack/spack-packages

# Bootstrap Spack: detect compilers and register the ROCm package repo
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack compiler find && \
    spack repo set --destination  /home/spack/spack-packages builtin

# Create the named environment and install all ROCm-tagged packages
COPY --chown=spack:spack rocm-spack-env/spack.yaml /home/spack/rocm-spack-env/spack.yaml
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack clean -m && \
    spack env create ${SPACK_ENV_NAME} /home/spack/rocm-spack-env/spack.yaml && \
    spack env activate ${SPACK_ENV_NAME} && \
    spack concretize -f && \
    spack clean -m && \
    spack install --fail-fast --reuse --use-cache

# Activate the environment for interactive sessions
RUN echo ". ${SPACK_ROOT}/share/spack/setup-env.sh" >> /home/spack/.bashrc && \
    echo "spack env activate ${SPACK_ENV_NAME}" >> /home/spack/.bashrc

# Entrypoint activates the spack environment for all commands
COPY --chown=spack:spack rocm-spack-env/entrypoint.sh /home/spack/entrypoint.sh
RUN chmod +x /home/spack/entrypoint.sh

ENTRYPOINT ["/home/spack/entrypoint.sh"]
CMD ["/bin/bash"]
