FROM ghcr.io/rocm/no_rocm_image_ubi10:latest

# Extend the base ubi10 image with packages needed to build opencl-cts
# against distro-provided OpenCL and SPIR-V dependencies.
RUN sudo dnf install -y --nodocs \
    ocl-icd-devel \
    opencl-headers \
    spirv-headers-devel \
    spirv-tools \
    spirv-tools-devel \
    && sudo dnf clean all
