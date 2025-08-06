# This Docker image is used for TheRock builds and tests, providing a clean ROCm-less container
# Mirrored from https://github.com/saienduri/docker-images/blob/d28cece7d73f57f0191b0e5c195c75703149be65/ghascale-rocm.Dockerfile

FROM ubuntu:24.04

RUN apt update && apt install sudo -y

# Create tester user with sudo privileges
RUN useradd -ms /bin/bash tester && \
    usermod -aG sudo tester
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
    psmisc

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo groupadd -g 109 render

RUN sudo apt-get update -y && \
    sudo apt-get install -y python3-setuptools python3-wheel
