FROM rocm/dev-ubuntu-22.04:latest

ARG KEEP_BUILD_FOLDER="False"
RUN echo "KEEP_BUILD_FOLDER $KEEP_BUILD_FOLDER"

RUN set -xe

# Install hipBLASLt dependent packages, then build and install hipBLASLt from source
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
  git \
  cmake \
  llvm \
  python3 \
  python3-pip \
  python3-venv \
  hipblas-common \
  && cd /root/ \
  && git clone https://github.com/ROCm/hipBLASLt \
  && cd hipBLASLt \
  && ./install.sh -idc \
  && if [ "$KEEP_BUILD_FOLDER" = "False" ] ; then \
  echo "Removing hipBLASLt build folder"; \
  rm -rf /root/hipBLASLt/build; \
  fi \
  && apt-get clean

RUN groupadd -f render
