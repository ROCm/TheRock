FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# Extend the image by adding the dependencies we would like to have for
# a more complete rocgdb validation.
RUN sudo apt-get install -y --no-install-recommends \
    dejagnu \
    gcc \
    g++ \
    make \
    gfortran
