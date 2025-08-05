# This Docker image is used to for TheRock builds and tests, providing a clean container with no ROCm pre-installed
# Mirrored from https://github.com/saienduri/docker-images/blob/d28cece7d73f57f0191b0e5c195c75703149be65/ghascale-rocm.Dockerfile

FROM ghcr.io/actions/actions-runner:latest

RUN sudo apt-get update -y \
    && sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository -y ppa:git-core/ppa \
    && sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    jq \
    sudo \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    psmisc \
    && sudo rm -rf /var/lib/apt/lists/*

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs

RUN sudo groupadd -g 109 render

RUN sudo apt update -y \
    && sudo apt install -y python3-setuptools python3-wheel \
    && sudo usermod -a -G render,video runner
