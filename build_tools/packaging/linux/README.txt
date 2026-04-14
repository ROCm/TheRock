#Scope:
The current scope of this is for producing AMD vendor packaging for hosting in AMD repositories. We expect that a good deal of this work can be adapted for future upstream OS packaging activities, but those are currently out of scope of what is being built here

#Prerequisites:
Python version required : python 3.12 or above
 Ubuntu(24.04):
apt update
apt install -y python3 python3-venv python3-pip
apt install -y debhelper
apt install -y llvm-20
apt install -y rpm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt (build_tools/packaging/linux/requirements.txt)

#Usage:
Ubuntu:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR --target gfx94X-dcgpu --dest-dir ./OUTPUT_PKG --rocm-version 7.1.0 --pkg-type deb --version-suffix build_type

For more options ./build_package.py -h
