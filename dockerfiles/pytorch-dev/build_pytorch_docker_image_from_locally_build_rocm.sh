#!/usr/bin/bash

# Causes bash process to die immediately after child process returns error
# to make sure that script does not continue logic if error has happened.
set -e
set -o pipefail

if [ ! -n "$AMDGPU_TARGETS" ]; then
	echo "You need to specify TARGET GPUS. For example:"
	echo "AMDGPU_TARGETS=gfx1150 ./build_pytorch_docker_image_from_locally_build_rocm.sh"
	exit 1
else
	echo "Building pytorch docker image for AMD GPUs: ${AMDGPU_TARGETS}"
	sleep 2
fi

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"

# we need to make the tar from the directories we want to copy and put them to same directory
# with the Dockerfile because docker does not allow copying things from the parent directories
tar -C ${SCRIPT_DIR}/../../build/dist -cvf rocm.tar rocm

# move to therock root directory
cd ${SCRIPT_DIR}/../..

# clone and copy the source code in the host machine instead inside the docker image
# as it will make it faster to do the next builds (cloning of pytorch and it's git submodules is slow operation)
./external-builds/pytorch/checkout_pytorch_all.sh

# exclude the build directory that may exist because same source code has been build locally 
tar -C external-builds/pytorch --exclude='pytorch/build' -cvf dockerfiles/pytorch-dev/pytorch.tar pytorch
tar -C external-builds/pytorch --exclude='pytorch_vision/build' -cvf dockerfiles/pytorch-dev/pytorch_vision.tar pytorch_vision
tar -C external-builds/pytorch --exclude='pytorch_audio/build' -cvf dockerfiles/pytorch-dev/pytorch_audio.tar pytorch_audio

docker build --build-arg AMDGPU_TARGETS=${AMDGPU_TARGETS} --file dockerfiles/pytorch-dev/pytorch_ubuntu_image_from_locally_build_rocm.Dockerfile .
