# This Docker image is used for TheRock builds and tests, providing a clean ROCm-less container

FROM ubuntu:24.04

RUN apt update && apt install sudo -y

# Create tester user with render/video permissions (no sudo group membership).
RUN useradd -m -s /bin/bash -U tester
RUN groupadd -g 109 render && usermod -a -G render,video tester

# Grant tester passwordless sudo only for the specific commands needed by CI:
#   apt / apt-get   - runtime package installation in test workflows
#   tee             - writing APT keyrings and source-list entries under /etc/apt/
#   mkdir / chmod   - creating the /etc/apt/keyrings/ directory and fixing its permissions
#   dmesg           - reading kernel ring buffer in GPU diagnostics
RUN echo 'tester ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/apt, /usr/bin/tee, /bin/mkdir, /bin/chmod, /usr/bin/mkdir, /usr/bin/chmod, /usr/bin/dmesg, /bin/dmesg' \
    > /etc/sudoers.d/tester \
    && chmod 0440 /etc/sudoers.d/tester

# Install build-time packages as root before switching to tester.
RUN apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    git-lfs \
    jq \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    psmisc \
    libgfortran5 \
    valgrind \
    python3-setuptools \
    python3-wheel

# Set as default user
USER tester

WORKDIR /home/tester/
