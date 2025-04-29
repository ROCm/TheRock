#!/bin/bash

# sudo enablement
sudo usermod -a -G sudo "$(id -un)"
echo "%sudo ALL = (ALL) NOPASSWD: ALL" | sudo tee -a /etc/sudoers

# additional packages 
sudo apt install gfortran git git-lfs ninja-build cmake g++ pkg-config xxd patchelf automake python3-venv python3-dev libegl1-mesa-dev

# svc install
sudo ./svc.sh install root
echo ROCR_VISIBLE_DEVICES=$1 >> .env

sudo ./svc.sh start
