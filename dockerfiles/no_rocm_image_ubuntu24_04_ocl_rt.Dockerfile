FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# packages for runtime hip/ocl validation
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        ocl-icd-opencl-dev \
        strace \
    && rm -rf /var/lib/apt/lists/*
USER tester
