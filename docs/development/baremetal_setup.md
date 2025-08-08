# Bare metal machine setup with TheRock

## Linux

1. The machine will need AMD GPU drivers. To install on `Ubuntu 24.04`:

   ```bash
   # Installing the amdgpu-install package
   sudo apt update
   wget https://repo.radeon.com/amdgpu-install/6.4/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
   sudo apt install ./amdgpu-install_6.4.60400-1_all.deb
   sudo apt update

   # Installing the GPU drivers
   amdgpu-install --usecase=dkms
   ```

   For other Linux distributions, you can find the commands at [Installation via AMDGPU installer](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html#installation)

1. After drivers have been installed, run:

   ```bash
   sudo modprobe amdgpu
   sudo usermod -a -G render,video $LOGNAME
   ```

1. After the GPU drivers have been setup, it's now time to setup `TheRock`. There are a few ways to install it:

   - [Building from source for Ubuntu 24.04](https://github.com/ROCm/TheRock?tab=readme-ov-file#building-from-source)

     ```bash
     # Install Ubuntu dependencies
     sudo apt install gfortran git git-lfs ninja-build cmake g++ pkg-config xxd patchelf automake python3-venv python3-dev libegl1-mesa-dev

     # Clone the repository
     git clone https://github.com/ROCm/TheRock.git
     cd TheRock

     # Init python virtual environment and install python dependencies
     python3 -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt

     # Download submodules and apply patches
     python ./build_tools/fetch_sources.py

     # For building ROCm/HIP
     cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu
     cmake --build build
     ```

   - [Using `install_rocm_from_artifacts.py`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#from-install_rocm_from_artifactspy)

     ```bash
     # Clone the repository
     git clone https://github.com/ROCm/TheRock.git
     cd TheRock

     # Downloads the version 6.4.0rc20250516 gfx110X artifacts from GitHub release tag nightly-tarball to the specified output directory build
     python build_tools/install_rocm_from_artifacts.py --release 6.4.0rc20250516 --amdgpu-family gfx110X-dgpu --output-dir build

     # Downloads all gfx94X S3 artifacts from GitHub CI workflow run 15052158890 to the default output directory therock-build
     python build_tools/install_rocm_from_artifacts.py --run-id 15052158890 --amdgpu-family gfx94X-dcgpu --tests

     ```

1. After installing TheRock, please add the `bin/` directory from TheRock build directory to your environment variables. After that, you can run `rocminfo`, `rocm-smi` and other commands!
