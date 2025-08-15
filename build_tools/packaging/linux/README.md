Prerequisites:

Python version required:  python 3.12.3
Install rpm package
Ex: apt install rpm (in Ubuntu)
pip install -r requirements.txt

usage: build_package.py [-h/\s\+$//e --artifact-url ARTIFACT_URL --dest-dir DEST_DIR [--gfx-arch GFX_ARCH] [--pkg-type PKG_TYPE] [--rocm-version ROCM_VERSION]
                        [--version-suffix VERSION_SUFFIX] [--install-prefix INSTALL_PREFIX] [--rpath-pkg] [--clean-build]
                        [--pkg-names PKG_NAMES [PKG_NAMES ...]]

options:
  -h, --help            show this help message and exit
  --artifact-url ARTIFACT_URL
                        Source artifacts/ dir from a build
  --dest-dir DEST_DIR   Destination directory in which to materialize packages
  --gfx-arch GFX_ARCH   Graphix architecture used for building
  --pkg-type PKG_TYPE   Choose the package format to be generated: DEB or RPM
  --rocm-version ROCM_VERSION
                        ROCm Release version
  --version-suffix VERSION_SUFFIX
                        Version suffix to append to package names
  --install-prefix INSTALL_PREFIX
                        Base directory where package will be installed
  --rpath-pkg           Enable rpath-pkg mode
  --clean-build         Clean the packaging environment
  --pkg-names PKG_NAMES [PKG_NAMES ...]
                        Specify the packages to be created: single composite or any specific package name
