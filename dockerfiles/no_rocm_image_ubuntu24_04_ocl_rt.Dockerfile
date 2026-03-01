FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# packages for runtime hip/ocl validation
RUN sudo apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev
