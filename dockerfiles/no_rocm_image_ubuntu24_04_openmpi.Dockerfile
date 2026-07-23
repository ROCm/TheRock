FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# no_rocm_image_ubuntu24_04_openmpi:
# Test-only image (not a ROCm build image): extends no_rocm_image_ubuntu24_04 with
# a system OpenMPI install (libopenmpi-dev, openmpi-bin) for the rocprofv3 mpi-ranks
# integration tests. The rocprofiler-sdk tests are configured and compiled
# in-container at test time, so find_package(MPI) needs the OpenMPI headers and
# mpiexec. OpenMPI is intentionally not bundled in TheRock artifacts (to avoid
# forcing a specific MPI onto users), so it is provided at the system level here.
# The corresponding published image is:
#   ghcr.io/rocm/no_rocm_image_ubuntu24_04_openmpi:latest
RUN sudo apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin
