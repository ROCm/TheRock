#Scope:
The current scope of this is for producing AMD vendor packaging for hosting in AMD repositories. We expect that a good deal of this work can be adapted for future upstream OS packaging activities, but those are currently out of scope of what is being built here

#Prerequisites:
Python version required : python 3.12 or above
 Almalinux:
dnf install rpm-build rpm-sign
dnf install llvm
pip install -r requirements.txt
export GPG_SIGNING_SERVER='http://your-signing-server.company.com'
echo '%_gpg_path /TheRock/build_tools/packaging/linux/gpgshim' >> ~/.rpmmacros


 Ubuntu:
apt update
apt install -y python3
apt install -y python3-pip
apt install -y debhelper
apt install -y llvm
pip install -r requirements.txt

#Usage:
Almalinux:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR --target gfx94X-dcgpu --dest-dir ./OUTPUT_PKG --rocm-version 7.1.0 --pkg-type rpm --version-suffix build_type [--sign signer2@example.com]

Ubuntu:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR --target gfx94X-dcgpu --dest-dir ./OUTPUT_PKG --rocm-version 7.1.0 --pkg-type deb --version-suffix build_type

For more options ./build_package.py -h
