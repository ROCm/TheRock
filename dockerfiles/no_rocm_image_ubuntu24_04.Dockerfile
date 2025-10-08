# This Docker image is used for TheRock builds and tests, providing a clean ROCm-less container

FROM ubuntu:24.04

RUN apt update && apt install sudo -y

# Create tester user with sudo privileges and render/video permissions
RUN useradd -m -s /bin/bash -U -G sudo tester
RUN groupadd -g 109 render && usermod -a -G render,video tester
# New added for disable sudo password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set as default user
USER tester

RUN sudo apt-get update -y \
    && sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository -y ppa:git-core/ppa \
    && sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    jq \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    psmisc \
    libgfortran5

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo apt-get update -y && \
    sudo apt-get install -y python3-setuptools python3-wheel

WORKDIR /home/tester/
