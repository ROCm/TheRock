# rocm_runtime.Dockerfile
#
# ROCm Runtime Docker Image Builder
# - Single Dockerfile supporting multiple Linux distributions via `BASE_IMAGE`.
#
# Maintained by: AMD ROCm Engineering (contact: dl.MLSE.DevOps@amd.com)
#
# Support policy:
# - Intended for evaluation and development workflows.
# - Nightlies/prereleases/devreleases are subject to change and may be removed or temporarily unavailable.
#
# Trademark:
# - ROCm is a trademark of Advanced Micro Devices, Inc.
#
# Compliance note:
# - This Dockerfile downloads ROCm tarballs or deb/rpm packages during `docker build` and installs OS packages from the selected base image.
# - If you redistribute a resulting image, you are responsible for ensuring compliance with applicable licenses/terms
#   for (1) downloaded ROCm artifacts and (2) the base OS packages you install.
#
# Supply-chain / reproducibility note:
# - For production or CI reproducibility, pin `BASE_IMAGE` by digest (e.g., ubuntu@sha256:...).
#
# Runtime note:
# - This is a user-space ROCm runtime image. It does NOT include kernel drivers.
# - Host must provide compatible AMDGPU/ROCm kernel components and device access (e.g., --device=/dev/kfd --device=/dev/dri).
# - Typical run flags: --device=/dev/kfd --device=/dev/dri (and often appropriate group/permission settings).
#
# Supported base images (examples)
# - ubuntu:24.04
# - almalinux:8
# - mcr.microsoft.com/azurelinux/base/core:3.0
# - registry.access.redhat.com/ubi8/ubi:8.10 (RHEL 8.10)
# - registry.access.redhat.com/ubi9/ubi:9.7 (RHEL 9.7)
# - registry.access.redhat.com/ubi10/ubi:10.1 (RHEL 10.1)
# - registry.suse.com/bci/bci-base:15.7 (SLES 15.7)
# - registry.suse.com/bci/bci-base:16.0 (SLES 16.0)
#
# Build arguments
# - BASE_IMAGE       : Base Docker image (default: ubuntu:24.04)
# - VERSION          : Full version string (e.g., 7.11.0a20251211, 7.10.0)
# - AMDGPU_FAMILY    : AMD GPU family (e.g., gfx110X-all, gfx94X-dcgpu)
# - RELEASE_TYPE     : Release type (default: nightlies). Options: prereleases, devreleases, stable
#                      Note: devreleases is only supported with INSTALL_METHOD=tarball.
# - INSTALL_METHOD   : Installation method (default: tarball). Options: tarball, packages
#
# Build examples (tarball, default):
#
#   # Ubuntu 24.04 + gfx110X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=ubuntu:24.04 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx110X-all \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:ubuntu24.04-gfx110X-7.11.0a20251211 \
#     dockerfiles/
#
#   # AlmaLinux 8 + gfx94X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=almalinux:8 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx94X-dcgpu \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:almalinux8-gfx94X-7.11.0a20251211 \
#     dockerfiles/
#
#   # Azure Linux 3 + gfx120X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=mcr.microsoft.com/azurelinux/base/core:3.0 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx120X-all \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:azurelinux3-gfx120X-7.11.0a20251211 \
#     dockerfiles/
#
#   # Ubuntu 22.04 + gfx94X (stable release)
#   docker build \
#     --build-arg BASE_IMAGE=ubuntu:22.04 \
#     --build-arg VERSION=7.10.0 \
#     --build-arg AMDGPU_FAMILY=gfx94X-dcgpu \
#     --build-arg RELEASE_TYPE=stable \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm:ubuntu22.04-gfx94X-7.10.0 \
#     dockerfiles/
#
#   # RHEL 8.10 UBI + gfx94X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:8.10 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx94X-dcgpu \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:rhel8.10-gfx94X-7.11.0a20251211 \
#     dockerfiles/
#
#   # RHEL 9.7 UBI + gfx94X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=registry.access.redhat.com/ubi9/ubi:9.7 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx94X-dcgpu \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:rhel9.7-gfx94X-7.11.0a20251211 \
#     dockerfiles/
#
#   # RHEL 10.1 UBI + gfx110X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=registry.access.redhat.com/ubi10/ubi:10.1 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx110X-all \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:rhel10.1-gfx110X-7.11.0a20251211 \
#     dockerfiles/
#
#   # SLES 15.7 BCI + gfx110X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=registry.suse.com/bci/bci-base:15.7 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx110X-all \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:sles15.7-gfx110X-7.11.0a20251211 \
#     dockerfiles/
#
#   # SLES 16.0 BCI + gfx94X (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=registry.suse.com/bci/bci-base:16.0 \
#     --build-arg VERSION=7.11.0a20251211 \
#     --build-arg AMDGPU_FAMILY=gfx94X-dcgpu \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly:sles16.0-gfx94X-7.11.0a20251211 \
#     dockerfiles/
#
# Build examples (packages):
#
#   # Ubuntu 24.04 + gfx110x via packages (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=ubuntu:24.04 \
#     --build-arg VERSION=7.13.0a20260322 \
#     --build-arg AMDGPU_FAMILY=gfx110x \
#     --build-arg INSTALL_METHOD=packages \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly-pkg:ubuntu24.04-gfx110x-7.13.0a20260322 \
#     dockerfiles/
#
#   # RHEL 9.7 UBI + gfx94x via packages (nightly)
#   docker build \
#     --build-arg BASE_IMAGE=registry.access.redhat.com/ubi9/ubi:9.7 \
#     --build-arg VERSION=7.13.0a20260322 \
#     --build-arg AMDGPU_FAMILY=gfx94x \
#     --build-arg INSTALL_METHOD=packages \
#     -f dockerfiles/rocm_runtime.Dockerfile \
#     -t rocm-nightly-pkg:rhel9.7-gfx94x-7.13.0a20260322 \
#     dockerfiles/
#
# Run example:
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined \
#     rocm-nightly:ubuntu24.04-gfx110X-7.11.0a20251211 rocminfo

# Base image selection
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# ROCm configuration arguments
ARG VERSION
ARG AMDGPU_FAMILY
ARG RELEASE_TYPE=nightlies
ARG INSTALL_METHOD=tarball

LABEL org.opencontainers.image.title="ROCm runtime image (TheRock)" \
    org.opencontainers.image.description="ROCm user-space runtime image built from TheRock project; installs ROCm from prebuilt tarballs or packages during build." \
    org.opencontainers.image.version="${VERSION}"

# Copy installation scripts
COPY install_rocm_deps.sh /tmp/
COPY install_rocm_tarball.sh /tmp/
COPY install_rocm_packages.sh /tmp/

# Install system dependencies
# The script auto-detects the distribution and uses the appropriate package manager
RUN chmod +x /tmp/install_rocm_deps.sh && \
    /tmp/install_rocm_deps.sh

# Install ROCm via tarball or packages based on INSTALL_METHOD
RUN chmod +x /tmp/install_rocm_tarball.sh /tmp/install_rocm_packages.sh && \
    if [ "${INSTALL_METHOD}" = "packages" ]; then \
        /tmp/install_rocm_packages.sh \
            "${VERSION}" \
            "${AMDGPU_FAMILY}" \
            "${RELEASE_TYPE}"; \
    else \
        /tmp/install_rocm_tarball.sh \
            "${VERSION}" \
            "${AMDGPU_FAMILY}" \
            "${RELEASE_TYPE}"; \
    fi && \
    rm -f /tmp/install_rocm_deps.sh /tmp/install_rocm_tarball.sh /tmp/install_rocm_packages.sh

# Configure environment variables
ENV ROCM_PATH=/opt/rocm
ENV PATH="/opt/rocm/bin:${PATH}"
