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

## S3 Configuration and Package Repository URLs

The following table shows the S3 bucket configuration and public repository URLs for each release type. This configuration is used by `build_tools/packaging/linux/get_s3_config.py` and `build_tools/packaging/linux/upload_package_repo.py`.

| Release Type | S3 Bucket | S3 Prefix | DEB Install URL | RPM Install URL |
|-------------|-----------|-----------|-----------------|-----------------|
| **dev** | `therock-dev-packages` | `{pkg_type}/{yyyymmdd}-{artifact_id}`<br>Example: `deb/20260406-12345678` | `https://rocm.devreleases.amd.com/deb/{yyyymmdd}-{artifact_id}` | `https://rocm.devreleases.amd.com/rpm/{yyyymmdd}-{artifact_id}/x86_64/` |
| **nightly** | `therock-nightly-packages` | `{pkg_type}/{yyyymmdd}-{artifact_id}`<br>Example: `deb/20260406-12345678` | `https://rocm.nightlies.amd.com/deb/{yyyymmdd}-{artifact_id}` | `https://rocm.nightlies.amd.com/rpm/{yyyymmdd}-{artifact_id}/x86_64/` |
| **prerelease** | `therock-prerelease-packages` | `v3/packages/{pkg_type}`<br>Example: `v3/packages/deb` | `https://rocm.prereleases.amd.com/packages/ubuntu2404` | `https://rocm.prereleases.amd.com/packages/rhel10/x86_64/` |
| **release** | `therock-release-packages` | `v3/packages/{pkg_type}`<br>Example: `v3/packages/deb` | `https://repo.amd.com/rocm/packages/ubuntu2404` | `https://repo.amd.com/rocm/packages/rhel10/x86_64/` |
| **ci** (ROCm/TheRock)<br>*DEFAULT* | `therock-ci-artifacts` | `{artifact_id}-{platform}/packages/{pkg_type}`<br>Example: `12345678-linux/packages/deb` | `https://therock-ci-artifacts.s3.amazonaws.com/{artifact_id}-{platform}/packages/deb` | `https://therock-ci-artifacts.s3.amazonaws.com/{artifact_id}-{platform}/packages/rpm/x86_64/` |
| **ci** (Fork/External) | `therock-ci-artifacts-external` | `{repo_name}/{artifact_id}-{platform}/packages/{pkg_type}`<br>Example: `someone-fork/12345678-linux/packages/deb` | `https://therock-ci-artifacts-external.s3.amazonaws.com/{repo_name}/{artifact_id}-{platform}/packages/deb` | `https://therock-ci-artifacts-external.s3.amazonaws.com/{repo_name}/{artifact_id}-{platform}/packages/rpm/x86_64/` |

**Notes:**

- RPM repositories always append `/x86_64/` subdirectory for yum/dnf compatibility
- Both `get_s3_config.py` and `upload_package_repo.py` use matching S3 prefix formats
- Default job type is `dev` when not specified
- Variables:
  - `{pkg_type}`: Package type (`deb` or `rpm`)
  - `{yyyymmdd}`: Date in YYYYMMDD format
  - `{artifact_id}`: GitHub Actions run ID or custom artifact identifier
  - `{platform}`: Platform name (`linux` or `windows`)
  - `{repo_name}`: Repository name with `/` replaced by `-` (e.g., `ROCm-TheRock`, `someone-fork`)

## Building Packages

```bash
./build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --pkg-type rpm
```

To install locally built packages

- Directly install the RPM packages by file name:<br>
  rpm -i package_name

- Directly install the Debian packages by file name:<br>
  dpkg -i package_name

### Configuring your Python Project Dependencies

Python version required : python 3.12 or above<br>
Almalinux:<br>
dnf install rpm-build<br>
dnf install llvm<br>
pip install -r requirements.txt<br>

Ubuntu:<br>
apt update<br>
apt install -y python3<br>
apt install -y python3-pip<br>
apt install -y debhelper<br>
apt install -y llvm-20<br>
pip install -r requirements.txt<br>

### Usage

RPM package:<br>

```bash
./build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --version-suffix build_type \
   --pkg-type rpm
```

Debian package:<br>

```bash
./build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --version-suffix build_type \
   --pkg-type deb
```

Debian RPATH package:<br>

```bash
./build_package.py \
   --artifacts-dir ./ARTIFACTS_DIR \
   --target gfx94X-dcgpu \
   --dest-dir ./OUTPUT_PKG \
   --rocm-version 7.1.0 \
   --pkg-type deb \
   --version-suffix build_type \
   --rpath-pkg True
```
