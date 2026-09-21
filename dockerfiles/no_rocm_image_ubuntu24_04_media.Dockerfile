ARG BASE_IMAGE=ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:f6741eb54c20d3219bdcd25742dbaa2c6b14394253671bf018be89b324b31a48
FROM ${BASE_IMAGE}

# no_rocm_image_ubuntu24_04_media:
# Extend the base no_rocm_image_ubuntu24_04 image with media / video codec
# dependencies required for rocdecode test validation. These video codec
# libraries are not bundled in TheRock artifacts and must be present at test
# build time. The corresponding published image is:
#   ghcr.io/rocm/no_rocm_image_ubuntu24_04_media:latest
RUN sudo apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    pkg-config
