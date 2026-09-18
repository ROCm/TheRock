# This Docker image is used for TheRock builds and tests, providing a clean ROCm-less container

FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        jq \
        libgfortran5 \
        lld \
        ninja-build \
        psmisc \
        python3-setuptools \
        python3-wheel \
        software-properties-common \
        sudo \
        unzip \
        valgrind \
        wget \
        zip \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Create tester user with sudo privileges and render/video permissions
RUN useradd -m -s /bin/bash -U -G sudo tester
RUN groupadd -g 109 render && usermod -a -G render,video tester
# New added for disable sudo password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set as default user
USER tester

WORKDIR /home/tester/
