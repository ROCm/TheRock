ARG BASE_IMAGE=ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:f6741eb54c20d3219bdcd25742dbaa2c6b14394253671bf018be89b324b31a48
FROM ${BASE_IMAGE}

# Extend the image by adding the dependencies we would like to have for
# a more complete rocgdb validation.
RUN sudo apt-get install -y --no-install-recommends \
    dejagnu \
    gcc \
    g++ \
    make \
    gfortran
