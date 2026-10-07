FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# Extend the image by adding the dependencies we would like to have for
# a more complete rocgdb validation.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        dejagnu \
        g++ \
        gcc \
        gfortran \
        make \
    && rm -rf /var/lib/apt/lists/*
USER tester
