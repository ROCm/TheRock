# This Docker image is used for TheRock builds and tests, providing a clean ROCm-less container
#
# Base image: registry.access.redhat.com/ubi10/ubi:latest (Red Hat UBI 10). Catalog:
# https://catalog.redhat.com/en/software/containers/ubi10/ubi/66f2b46b122803e4937d11ae
FROM registry.access.redhat.com/ubi10/ubi:latest

RUN dnf install -y --nodocs sudo && dnf clean all

# Create tester user with sudo privileges and render/video permissions
RUN useradd -m -s /bin/bash -U -G wheel tester
# UBI 10 may already ship a `render` group; only create if missing
RUN (getent group render >/dev/null || groupadd -g 109 render) \
    && usermod -a -G render,video tester
# Disable sudo password for wheel group
RUN echo '%wheel ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set as default user
USER tester

RUN sudo dnf install -y --nodocs \
    git \
    jq \
    unzip \
    zip \
    cmake \
    ninja-build \
    clang \
    lld \
    wget \
    libgfortran \
    libquadmath \
    libatomic \
    valgrind \
    && sudo dnf clean all

RUN sudo dnf install -y --nodocs git-lfs \
    && sudo dnf clean all

RUN sudo dnf install -y --nodocs python3-pip python3-setuptools python3-wheel \
    python-unversioned-command \
    && sudo dnf clean all

WORKDIR /home/tester/
