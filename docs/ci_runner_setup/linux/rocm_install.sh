#!/bin/bash

# ROCm install
wget https://repo.radeon.com/amdgpu-install/6.4/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
sudo apt install ./amdgpu-install_6.4.60400-1_all.deb -y
sudo apt update
sudo apt install python3-setuptools python3-wheel -y
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm -y

# AMD drive install
wget https://repo.radeon.com/amdgpu-install/6.4/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
sudo apt install ./amdgpu-install_6.4.60400-1_all.deb -y
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" -y
sudo apt install amdgpu-dkms -y

# required
sudo systemctl reboot
