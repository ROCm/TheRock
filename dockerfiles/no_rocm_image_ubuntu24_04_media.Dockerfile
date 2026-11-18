FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# no_rocm_image_ubuntu24_04_media:
# Extend the base no_rocm_image_ubuntu24_04 image with media / video codec
# dependencies required for rocdecode test validation. These video codec
# libraries are not bundled in TheRock artifacts and must be present at test
# build time. The corresponding published image is:
#   ghcr.io/rocm/no_rocm_image_ubuntu24_04_media:latest
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
USER tester
