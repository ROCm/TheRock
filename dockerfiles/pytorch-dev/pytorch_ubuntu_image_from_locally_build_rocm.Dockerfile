# copies locally build rocm to docker-image and then
# uses it to build the pytorch.
# This dockerfile does not require docker buildkit to be installed.
#
# Following files are expected to be located in dockerfile directory:
# - rocm.tar rocm-binary directory
# - pytorch.tar pytorch source directory
# - pytorch_vision.tar pytorch vision source directory
# - pytorch_audio.tar pytorch audio source directory
#
# Following environment variable is expected to be set:
# - AMDGPU_TARGETS
#
# ./build_pytorch_docker_image_from_locally_build_rocm.sh is helper script
# that will handle the creation of tar-files required and then finally the invoke of
# dockerimage build
#
# Build with:
#   AMDGPU_TARGETS=gfx1150 ././build_pytorch_docker_image_from_locally_build_rocm.sh

################################################################################
# PyTorch sources
################################################################################

FROM ubuntu:24.04 AS pytorch_sources
ARG AMDGPU_TARGETS
WORKDIR /t 

RUN if [ -z "$AMDGPU_TARGETS" ]; then \
    echo "Error1: AMDGPU_TARGETS environment variable is not set."; \
    exit 1; \
    fi

RUN apt update && apt install -y \
  python3 python-is-python3 python3-pip python3-pip-whl python3-venv git \
  ninja-build cmake g++ pkg-config sox ffmpeg libavformat-dev libavcodec-dev && \
  apt clean

RUN git config --global user.email "you@example.com" && \
    git config --global user.name "Your Name"

# Copy ROCM
RUN mkdir -p /opt
ADD dockerfiles/pytorch-dev/rocm.tar /opt/

# copy pytorch sources under /therock folder
RUN mkdir -p /therock
ADD dockerfiles/pytorch-dev/pytorch.tar /therock
ADD dockerfiles/pytorch-dev/pytorch_vision.tar /therock
ADD dockerfiles/pytorch-dev/pytorch_audio.tar /therock

COPY external-builds/pytorch/env_init.sh /therock/
COPY external-builds/pytorch/build_pytorch_torch.sh /therock/
COPY external-builds/pytorch/build_pytorch_vision.sh /therock/
COPY external-builds/pytorch/build_pytorch_audio.sh /therock/

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

WORKDIR /therock

ENV ROCM_HOME=/opt/rocm
ENV CMAKE_PREFIX_PATH=/opt/rocm
ENV PATH="/opt/rocm/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/rocm/lib"
ENV PYTORCH_ROCM_ARCH=$AMDGPU_TARGETS

RUN ./build_pytorch_torch.sh
RUN ./build_pytorch_vision.sh
RUN ./build_pytorch_audio.sh

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

# copy pytorch wheels
COPY --from=pytorch_build /therock/pytorch/dist/. /therock/pytorch/dist
COPY --from=pytorch_build /therock/pytorch_vision/dist/. /therock/pytorch_vision/dist
COPY --from=pytorch_build /therock/pytorch_audio/dist/. /therock/pytorch_audio/dist
RUN python3 -m pip install --break-system-packages --no-cache-dir $(find /therock/pytorch/dist -name '*.whl')
RUN python3 -m pip install --break-system-packages --no-cache-dir $(find /therock/pytorch_vision/dist -name '*.whl')
RUN python3 -m pip install --break-system-packages --no-cache-dir $(find /therock/pytorch_audio/dist -name '*.whl')

