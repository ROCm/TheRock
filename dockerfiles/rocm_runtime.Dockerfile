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
# - This Dockerfile downloads ROCm tarballs during `docker build` and installs OS packages from the selected base image.
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
#
# Build arguments
# - BASE_IMAGE       : Base Docker image (default: ubuntu:24.04)
# - VERSION          : Full version string (e.g., 7.11.0a20251211, 7.10.0)
# - AMDGPU_FAMILY    : AMD GPU family (e.g., gfx110X-all, gfx94X-dcgpu)
# - RELEASE_TYPE     : Release type (default: nightlies). Options: prereleases, devreleases, stable
#
# Build examples:
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
#   docker build --network=host \
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

LABEL org.opencontainers.image.title="ROCm runtime image (TheRock)" \
    org.opencontainers.image.description="ROCm user-space runtime image built from TheRock project; installs ROCm from prebuilt tarballs during build." \
    org.opencontainers.image.version="${VERSION}"

# Copy installation scripts
COPY install_rocm_deps.sh /tmp/
COPY install_rocm_tarball.sh /tmp/

# Install system dependencies
# The script auto-detects the distribution and uses the appropriate package manager
RUN chmod +x /tmp/install_rocm_deps.sh && \
    /tmp/install_rocm_deps.sh

# Install ROCm from tarball
# Tarball extracts to /opt/rocm-{VERSION}/, with symlink /opt/rocm -> /opt/rocm-{VERSION}
RUN chmod +x /tmp/install_rocm_tarball.sh && \
    /tmp/install_rocm_tarball.sh \
        "${VERSION}" \
        "${AMDGPU_FAMILY}" \
        "${RELEASE_TYPE}" && \
    rm -f /tmp/install_rocm_deps.sh /tmp/install_rocm_tarball.sh

# Configure environment variables
ENV ROCM_PATH=/opt/rocm
ENV PATH="/opt/rocm/bin:${PATH}"
