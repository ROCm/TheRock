FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# no_rocm_image_ubuntu24_04_rocprofiler:
# Extend the base no_rocm_image_ubuntu24_04 image with an MPI implementation so
# that find_package(MPI) succeeds when building the rocprofiler-sdk tests. This
# enables the rocprofv3 mpi-ranks integration tests, which are otherwise
# disabled when no MPI is present. OpenMPI is not bundled in TheRock artifacts
# and must be available at test build/run time. The corresponding published
# image is:
#   ghcr.io/rocm/no_rocm_image_ubuntu24_04_rocprofiler:latest
RUN sudo apt-get update -y \
    && sudo apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin
