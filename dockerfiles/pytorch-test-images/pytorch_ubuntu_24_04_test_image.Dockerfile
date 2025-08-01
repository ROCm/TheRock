# This Dockerfile is used for testing PyTorch, ensuring that no system ROCm is used

FROM ubuntu:24.04

RUN apt update -y

# Install base TheRock dependencies
RUN apt install gfortran git git-lfs ninja-build cmake g++ pkg-config xxd patchelf automake libtool python3-venv python3-dev libegl1-mesa-dev wget gpg -y

# Install AMD GPU DKMS (sourced from https://rocm.docs.amd.com/en/docs-7.0-alpha/preview/install/instinct-driver.html)
RUN apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" -y

RUN mkdir --parents --mode=0755 /etc/apt/keyrings

RUN  wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/30.10_alpha/ubuntu noble main" | tee /etc/apt/sources.list.d/amdgpu.list

RUN apt update -y

RUN apt install amdgpu-dkms -y
