# Multi-stage build which builds rocm, then pytorch, and finally produces
# a docker image with the built pytorch installed.
# Build with:
#   docker buildx --build-arg AMDGPU_TARGETS=gfx1100 \
#     --file dockerfiles/pytorch-dev/pytorch_dev_ubuntu_24.04.Dockerfile .
FROM ubuntu:24.04 AS build_rocm
ARG AMDGPU_TARGETS
WORKDIR /
RUN apt update && apt install -y \
  python3 python-is-python3 python3-pip python3-pip-whl \
  gfortran git-lfs ninja-build cmake g++ pkg-config \
  xxd libgtest-dev libgmock-dev patchelf automake
# TODO: Remove once https://github.com/ROCm/TheRock/issues/160 is resolved.
RUN apt install -y libgl-dev
# TODO: Remove once https://github.com/ROCm/TheRock/issues/161 is resolved.
RUN apt install -y python3-venv

RUN python3 -m pip install --break-system-packages \
  CppHeaderParser==2.7.4 meson==1.7.0 PyYAML==6.0.2
COPY dockerfiles/pytorch-dev/install_rocm_from_release.sh /
# TODO: The ROCM components still output some things to the source dir. Remove
# "rw" when fixed. See https://github.com/ROCm/TheRock/issues/159
RUN --mount=type=bind,target=/therock/src,rw bash /install_rocm_from_release.sh "$AMDGPU_TARGETS"


################################################################################
# PyTorch sources
################################################################################

FROM ubuntu:24.04 AS pytorch_sources
WORKDIR /

RUN apt update && apt install -y \
  python3 python-is-python3 python3-pip python3-pip-whl python3-venv git \
  ninja-build cmake g++ pkg-config sox ffmpeg libavformat-dev libavcodec-dev && \
  apt clean

RUN git config --global user.email "you@example.com" && \
    git config --global user.name "Your Name"

# prepare PyTorch sources
# 1. vision
# podman on fedora 42 has permission denied error when trying to access
# /therock/src
# workaround is to disable selinux (/etc/selinux/config)
RUN --mount=type=bind,target=/therock/src \
  python3 /therock/src/external-builds/pytorch/ptbuild.py checkout \
    --no-pytorch --no-audio \
    --vision-src-dir /therock/pytorch_vision --depth 1 --jobs 10
# 2. audio
RUN --mount=type=bind,target=/therock/src \
  python3 /therock/src/external-builds/pytorch/ptbuild.py checkout \
    --no-pytorch --no-vision \
    --audio-src-dir /therock/pytorch_audio --depth 1 --jobs 10
COPY external-builds/pytorch/env_init.sh /therock/
COPY external-builds/pytorch/build_pytorch_torch.sh /therock/
COPY external-builds/pytorch/build_pytorch_vision.sh /therock/
COPY external-builds/pytorch/build_pytorch_audio.sh /therock/
# 3. pytorch main repo
# We do this in two steps so that we get an image checkpoint
# with clean sources first (faster iteration).
RUN --mount=type=bind,target=/therock/src \
  python3 /therock/src/external-builds/pytorch/ptbuild.py checkout \
    --no-audio --no-vision \
    --no-patch --no-hipify \
    --pytorch-src-dir /therock/pytorch --depth 1 --jobs 10
RUN --mount=type=bind,target=/therock/src \
  python3 /therock/src/external-builds/pytorch/ptbuild.py checkout \
    --no-audio --no-vision \
    --pytorch-src-dir /therock/pytorch --depth 1 --jobs 10

# Copy ROCM
COPY --from=build_rocm /therock/build/dist/rocm /opt/rocm

# Setup environment.
# Note that the rocm_sysdeps lib dir should not be strictly required if all
# RPATH entries are set up correctly, but it is safer this way.
ENV PATH="/opt/rocm/bin:$PATH"
RUN (echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf) && \
    (echo "/opt/rocm/lib/rocm_sysdeps/lib" >> /etc/ld.so.conf.d/rocm.conf) && \
    ldconfig -v


################################################################################
# PyTorch Build
################################################################################

FROM pytorch_sources AS pytorch_build
ARG AMDGPU_TARGETS

RUN python3 -m pip install --break-system-packages -r /therock/pytorch/requirements.txt
# Downgrade CMake to avoid protobuf build failure
#RUN python3 -m pip install --break-system-packages cmake==3.26.4

ENV CMAKE_PREFIX_PATH=/opt/rocm
ENV USE_KINETO=OFF
ENV PYTORCH_ROCM_ARCH=$AMDGPU_TARGETS

WORKDIR /therock
# TODO: PYTORCH_ROCM_ARCH from environment variables seems broken. So we
# configure it manually for now.

ENV ROCM_HOME=/opt/rocm
ENV PATH="/opt/rocm/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/rocm/lib"

RUN ./build_pytorch_torch.sh /therock/pytorch
RUN ./build_pytorch_vision.sh /therock/pytorch_vision
RUN ./build_pytorch_audio.sh /therock/pytorch_audio

################################################################################
# PyTorch Install
################################################################################

FROM ubuntu:24.04 AS pytorch

RUN apt update && apt install -y \
    python3 python-is-python3 python3-pip python3-pip-whl python3-venv python3-numpy ffmpeg

# Copy ROCM
COPY --from=pytorch_build /opt/rocm /opt/rocm

# Setup environment.
ENV PATH="/opt/rocm/bin:$PATH"
RUN (echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf) && \
    (echo "/opt/rocm/lib/rocm_sysdeps/lib" >> /etc/ld.so.conf.d/rocm.conf) && \
    ldconfig -v

# install some commonly required python apps by pytorch and pytorch audio
RUN python3 -m pip install --break-system-packages pandas psutil IPython

# Bind mount the prior stage and install the wheel directly (saves the size of
# the wheel vs copying).
RUN --mount=type=bind,from=pytorch_build,source=/therock/pytorch/dist,target=/wheels \
    ls -lh /wheels && \
    python3 -m pip install --break-system-packages --no-cache-dir \
      $(find /wheels -name '*.whl')
# same for pytorch vision
RUN --mount=type=bind,from=pytorch_build,source=/therock/pytorch_vision/dist,target=/wheels \
    ls -lh /wheels && \
    python3 -m pip install --break-system-packages --no-cache-dir \
      $(find /wheels -name '*.whl')
# and pytorch audio
RUN --mount=type=bind,from=pytorch_build,source=/therock/pytorch_audio/dist,target=/wheels \
    ls -lh /wheels && \
    python3 -m pip install --break-system-packages --no-cache-dir \
      $(find /wheels -name '*.whl')
