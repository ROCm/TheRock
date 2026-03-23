FROM ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest

# Extend the image by adding the dependencies required for rocdecode
# test validation. These video codec libraries are not bundled in
# TheRock artifacts and must be present at test build time.
RUN sudo apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    pkg-config
