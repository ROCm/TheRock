# ROCm Native Packaging via TheRock

Given ROCk artifact directories, perform surgery to re-layout them for
distribution as Debian and RPM packages.

This process involves both the reorganization of artifacts and the
execution of package management tools

## General Design

The required packages and their details are stored in a JSON file
named package.json. Packages are generated based on the entries in this file

We generate two types of packages:

- RPM packages: Used for installation on Linux distributions such as
  RHEL, SLES, and AlmaLinux. A spec file is created using the Jinja2 template.
  This spec file is then populated with package details obtained from
  the JSON file. Finally, the package is generated using rpmbuild

- Debian Packages: Used for installation on Ubuntu.
  A Debian changelog, control, install, and rules file are created
  using the Jinja2 template. These files are then populated with the
  package details. The required package contents are copied to the
  debian folder. Finally, the package is generated using dpkg-buildpackage.

The generated packages are moved to the destination directory.

## Workflow

![Building Package](assets/native_packaging.drawio.svg#gh-light-mode-only)
![Building Package (dark)](assets/native_packaging.drawio.svg#gh-dark-mode-only)

## Versioned and Non-Versioned Packages

For each entry in the JSON file, a non-versioned package,
a versioned package( and an associated debuginfo/dbgsym package) will be created.
Each non-versioned package is associated with the corresponding versioned
package. A non-versioned package acts as a meta-package and does not
contain any contents. It always has a dependency on the versioned package.
The versioned package contains the actual package contents and may
depend on other packages.

## Development Packages

The development packages are named with the suffix -devel in package.json.
For the RPM use case, this naming convention is valid. However, for the
Debian use case, the suffix should be -dev. Therefore, for Debian packages,
the names are updated to use -dev.

## RPATH Packages

The RUNPATH in binaries and libraries will be replaced with RPATH if
the --rpath-pkg option is enabled in the build arguments. This option creates
only versioned packages

## Fields in package.json

Mandatory fields for a package entry in package.json.

- Package: The name of the package to be created.
- Architecture: The architecture of the package.
- Maintainer: The maintainer of the package.
- Description_Short: A short summary of the package.
- Description_Long: A detailed description of the package.
- Artifact: The ROCk artifact name, excluding the component field and everything that follows it.
- Artifact_Subdir: The ROCk artifact subdirectory from which the package contents are copied.
- Components: Specifies the ROCk artifact components required for the package.
- Homepage: The homepage URL of the package.
- Vendor: The vendor of the package.
- License: The license under which the package is distributed.
- Group: The group or category of the package.
- Priority: The priority level of the package.
- Section: The section under which the package falls.
- Gfxarch: Indicates that the package is associated with a graphics architecture.
- DEBDepends: The list of dependency packages. Applies to Debian packages.
- RPMRequires: The list of dependency packages. Applies to RPM packages.

Optional Fields

- Composite: Indicates that the ROCk package is marked as composite.
- Includes: Used only for composite packages. If the Includes field is present,
- the package is created using the packages listed in Includes. If no Includes
  field is specified, the Components field will be used instead.
- DisablePackaging: Disables the creation of the package.
- Disable_Debug_Package: Disables the generation of the debug symbol package.
- Disable_DWZ: Skip DWZ processing. Applies to Debian packages.
- Disable_DH_STRIP: Disables dh_strip. Applies to Debian packages.
- Provides: Indicates that a package provides the functionality of another package.
- Replaces: Indicates that a package replaces another package.

## Building Packages

The `--target` argument specifies the GFX architecture(s) for which packages should be built.
If not provided, the script will automatically detect available architectures from the artifact directory.

### Auto-detection (Recommended)

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --pkg-type rpm
```

The script will scan the artifact directory and automatically detect GFX architectures
from directory names matching the pattern `{name}_{component}_{target_family}`
(e.g., `blas_lib_gfx1100`, `miopen_lib_gfx942`).

### Explicit Target Specification

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --pkg-type rpm
```

You can specify one or more targets explicitly if needed. Multiple targets can be provided
space-separated, comma-separated, or semicolon-separated.

To install locally built packages

- Directly install the RPM packages by file name:<br>
  rpm -i package_name

- Directly install the Debian packages by file name:<br>
  dpkg -i package_name

### Configuring your Python Project Dependencies

Python version required : python 3.12 or above<br>
Ubuntu(24.04):<br>
apt update<br>
apt install -y python3 python3-venv python3-pip git<br>
apt install -y debhelper<br>
apt install -y rpm<br>
apt install -y llvm-20<br>

# Clone the repository

git clone https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv && source .venv/bin/activate<br>
pip install -r build_tools/packaging/linux/requirements.txt<br>

### Usage

RPM package with auto-detected architectures:<br>

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --version-suffix build_type \
   --pkg-type rpm
```

RPM package with explicit target:<br>

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --version-suffix build_type \
   --pkg-type rpm
```

Debian package with auto-detected architectures:<br>

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --version-suffix build_type \
   --pkg-type deb
```

Debian package with explicit target:<br>

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --version-suffix build_type \
   --pkg-type deb
```

Debian RPATH package with auto-detected architectures:<br>

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --pkg-type deb \
   --version-suffix build_type \
   --rpath-pkg
```

Debian RPATH package with explicit target:<br>

```bash
./build_tools/packaging/linux/build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --pkg-type deb \
   --version-suffix build_type \
   --rpath-pkg
```
