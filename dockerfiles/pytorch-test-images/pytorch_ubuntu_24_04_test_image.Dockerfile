# This Dockerfile is used for testing PyTorch, ensuring that no system ROCm is used

FROM ubuntu:24.04

RUN apt update -y

# Install base TheRock dependencies
RUN apt install -y gfortran \
    git \
    git-lfs \
    ninja-build \
    cmake \
    g++ \
    pkg-config \
    xxd \
    patchelf \
    automake \
    libtool \
    python3-venv \
    python3-dev \
    libegl1-mesa-dev \
    wget \
    gpg \
    curl \
    ca-certificates \
    jq \
    unzip \
    zip \
    clang \
    lld \
    psmisc

RUN groupadd -g 109 render

RUN usermod -a -G render,video root
