FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# no_rocm_image_ubuntu24_04_openmpi:
# Extend the base no_rocm_image_ubuntu24_04 image with a system OpenMPI install
# required to build and run the rocprofv3 mpi-ranks integration tests. OpenMPI is
# intentionally not bundled in TheRock artifacts (to avoid forcing a specific MPI
# onto users), so it is provided at the system level for the test environment. The
# corresponding published image is:
#   ghcr.io/rocm/no_rocm_image_ubuntu24_04_openmpi:latest
RUN sudo apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin
