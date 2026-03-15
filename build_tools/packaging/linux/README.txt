#Scope:
The current scope of this is for producing AMD vendor packaging for hosting in AMD repositories. We expect that a good deal of this work can be adapted for future upstream OS packaging activities, but those are currently out of scope of what is being built here

#Prerequisites:
Python version required : python 3.12 or above
 Almalinux:
dnf install rpm-build
dnf install llvm
pip install -r requirements.txt

 Ubuntu:
apt update
apt install -y python3
apt install -y python3-pip
apt install -y debhelper
apt install -y llvm
pip install -r requirements.txt

#Usage:
Almalinux:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR --target gfx94X-dcgpu --dest-dir ./OUTPUT_PKG --rocm-version 7.1.0 --pkg-type rpm --version-suffix build_type

Ubuntu:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR --target gfx94X-dcgpu --dest-dir ./OUTPUT_PKG --rocm-version 7.1.0 --pkg-type deb --version-suffix build_type

For more options ./build_package.py -h

Local install (from .deb/.rpm files):
./install_package.py --dest-dir ./PKG_DIR --package-json ./packages.json --rocm-version 6.2.0 --artifact-group gfx94X
-dcgpu --version true/false --composite true/false

Repo install (from remote repository using run-id):
./install_package.py --run-id 123456 --package-json ./packages.json --rocm-version 6.2.0 --artifact-group gfx94X-dcgp
u --version true/false --composite true/false

uninstaller.py
Uninstalls ROCm packages using the OS package manager. Supports composite and non-composite uninstallation.
Usage examples:

Composite uninstall:
./uninstall_package.py --run-id 123456 --package-json ./packages.json --rocm-version 6.2.0 --artifact-group gfx94X-dc
gpu --composite true

Non-composite uninstall:
./uninstall_package.py --run-id 123456 --package-json ./packages.json --rocm-version 6.2.0 --artifact-group gfx94X-dc
gpu --composite false

