# This Docker image is used to for TheRock builds and tests, providing a clean container with no ROCm pre-installed
# Mirrored from https://github.com/saienduri/docker-images/blob/d28cece7d73f57f0191b0e5c195c75703149be65/ghascale-rocm.Dockerfile

FROM ubuntu:24.04

RUN apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
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
    &&  rm -rf /var/lib/apt/lists/*

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs

RUN groupadd -g 109 render

RUN apt-get update -y
RUN apt-get install -y python3-setuptools python3-wheel
RUN usermod -a -G render,video runner
