# Full Test Workflow Documentation

## Overview

This document describes the **Full Test of Native Linux Packages** workflow and supporting tools for testing ROCm package installation from ROCm nightly build repositories.

The test workflow installs packages directly from ROCm nightly repositories using native package managers (apt/dnf)

## Files Created

### 1. Workflow File
**Location:** `.github/workflows/native_linux_packages_full_test.yml`

**Purpose:** GitHub Actions workflow that performs complete end-to-end testing of ROCm packages by:
- Setting up ROCm nightly repositories
- Installing packages using native package managers (apt/dnf)
- Verifying the installation in a clean container environment

### 2. Python Test Script
**Location:** `build_tools/packaging/linux/package_full_test.py`

**Purpose:** Python script that handles repository setup, package installation, and verification.

---

## Workflow: native_linux_packages_full_test.yml

### Trigger Methods

1. **Workflow Call** - Called by other workflows
2. **Manual Dispatch** - Manual trigger via GitHub Actions UI

### Input Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `artifact_id` | Yes | - | Artifact run ID (e.g., 21658678136) |
| `rocm_version` | Yes | - | ROCm version (e.g., 8.0.0, 8.0.1rc1) |
| `native_package_type` | Yes | `deb` | Package type: `deb` or `rpm` |
| `os_profile` | Yes | - | OS profile (e.g., ubuntu2404, rhel8, debian12, sles16) |
| `date` | Yes | - | Build date in YYYYMMDD format (e.g., 20260204) |
| `gfx_arch` | No | `gfx94x` | GPU architecture (e.g., gfx94x, gfx110x, gfx1151) |
| `repo_base_url` | No | `https://rocm.nightlies.amd.com` | Base URL for nightly repository |
| `install_prefix` | No | `/opt/rocm/core` | Installation prefix |

### Workflow Steps

#### Step 1: Install System Prerequisites
- Installs Python, git, curl
- Different packages based on DEB (Ubuntu) vs RPM (AlmaLinux/RHEL)

#### Step 2: Checkout Repository
- Clones TheRock repository into container
- Checks out the correct branch

#### Step 3: Install Python Dependencies
- Installs: `boto3`, `pyelftools`, `requests` (if needed)

#### Step 4: Full Package Installation Test
- **Main test step** - Runs `package_full_test.py`
- Sets up ROCm nightly repository
- Installs packages using apt/dnf from repository
- Verifies installation

#### Step 5: Verify ROCm Installation
- Checks for ROCm directory structure
- Validates key components exist
- Attempts to run rocminfo

#### Step 6: Test Report
- Generates summary report
- Shows success/failure status

### Container Environment

| Package Type | Container Image | Purpose |
|--------------|----------------|---------|
| DEB | `ubuntu:24.04` | Test Debian packages on Ubuntu |
| RPM | `almalinux:9` or `rhel:8` | Test RPM packages on AlmaLinux/RHEL |

**Note:** Containers run in `--privileged` mode for full system access

---

## Python Script: package_full_test.py

### Class: PackageFullTester

Main class that handles the full installation test process.

#### Constructor Parameters

```python
PackageFullTester(
    package_type: str,          # 'deb' or 'rpm'
    repo_base_url: str,         # Base URL for nightly repository
    artifact_id: str,           # Artifact run ID
    rocm_version: str,          # ROCm version
    os_profile: str,            # OS profile (e.g., ubuntu2404, rhel8)
    date: str,                  # Build date in YYYYMMDD format
    install_prefix: str,        # Installation prefix (default: /opt/rocm/core)
    gfx_arch: str              # GPU architecture (default: gfx94x)
)
```

### Key Methods

#### 1. `construct_repo_url_with_os() -> str`

**Purpose:** Constructs the full repository URL including OS profile for nightly builds

**Repository Structure:**
- Base URL: `{repo_base_url}/{package_type}/{YYYYMMDD-RUNID}/`
- DEB: `{repo_url}pool/main/`
- RPM: `{repo_url}x86_64/`

**Example:**
```
https://rocm.nightlies.amd.com/deb/20260204-21658678136/pool/main/
https://rocm.nightlies.amd.com/rpm/20260204-21658678136/x86_64/
```

#### 2. `setup_deb_repository() -> bool`

**Purpose:** Sets up DEB repository on the system

**Process:**
1. Constructs repository URL
2. Adds repository to `/etc/apt/sources.list.d/rocm-test.list`
3. Updates apt package lists
4. Returns success/failure

**Repository Entry Format:**
```
deb [arch=amd64] {repo_url}pool/main/ ./
```

#### 3. `setup_rpm_repository() -> bool`

**Purpose:** Sets up RPM repository on the system

**Process:**
1. Constructs repository URL
2. Creates repository file `/etc/yum.repos.d/rocm-test.repo`
3. Cleans dnf cache
4. Returns success/failure

**Repository File Format:**
```
[rocm-test]
name=ROCm Test Repository
baseurl={repo_url}x86_64/
enabled=1
gpgcheck=0
```

#### 4. `install_deb_packages() -> bool`

**Purpose:** Installs ROCm DEB packages from repository

**Process:**
1. Constructs package name: `amdrocm-{gfx_arch}`
2. Runs `apt install -y amdrocm-{gfx_arch}`
3. Streams installation output in real-time
4. Returns success/failure

**Timeout:** 30 minutes

#### 5. `install_rpm_packages() -> bool`

**Purpose:** Installs ROCm RPM packages from repository

**Process:**
1. Constructs package name: `amdrocm-{gfx_arch}`
2. Runs `dnf install -y amdrocm-{gfx_arch}`
3. Streams installation output in real-time
4. Returns success/failure

**Timeout:** 30 minutes

#### 6. `verify_rocm_installation() -> bool`

**Purpose:** Verifies ROCm installation

**Checks:**
- ✅ Installation directory exists (default: `/opt/rocm/core`)
- ✅ Key components present:
  - `bin/rocminfo`
  - `bin/hipcc`
  - `include/hip/hip_runtime.h`
  - `lib/libamdhip64.so`
- ✅ ROCm packages are installed (via dpkg/rpm)
- ✅ Attempts to run `rocminfo` (if available)

**Success Criteria:** At least 2 out of 4 key components found

#### 7. `run() -> bool`

**Purpose:** Main execution method

**Process:**
1. Setup repository (apt/dnf)
2. Install packages from repository
3. Verify installation
4. Return overall success/failure

### Command Line Usage

```bash
# DEB packages (Ubuntu 24.04)
python package_full_test.py \
    --package-type deb \
    --repo-base-url https://rocm.nightlies.amd.com \
    --artifact-id 21658678136 \
    --date 20260204 \
    --rocm-version 8.0.0 \
    --os-profile ubuntu2404 \
    --gfx-arch gfx94x

# RPM packages (RHEL 8)
python package_full_test.py \
    --package-type rpm \
    --repo-base-url https://rocm.nightlies.amd.com \
    --artifact-id 21658678136 \
    --date 20260204 \
    --rocm-version 8.0.0 \
    --os-profile rhel8 \
    --gfx-arch gfx94x

# Different GPU architecture (Strix Halo)
python package_full_test.py \
    --package-type deb \
    --repo-base-url https://rocm.nightlies.amd.com \
    --artifact-id 21658678136 \
    --date 20260204 \
    --rocm-version 8.0.0 \
    --os-profile ubuntu2404 \
    --gfx-arch gfx1151 \
    --install-prefix /opt/rocm/core
```

---
## Usage Examples

### Manual Workflow Trigger

1. Go to GitHub Actions
2. Select "Full Test of Native Linux Packages"
3. Click "Run workflow"
4. Fill in parameters:
   - `artifact_id`: `21658678136` (from build workflow)
   - `rocm_version`: `8.0.0`
   - `native_package_type`: `deb` or `rpm`
   - `os_profile`: `ubuntu2404` (for DEB) or `rhel8` (for RPM)
   - `date`: `20260204` (build date in YYYYMMDD format)
   - `gfx_arch`: `gfx94x` (optional, default: gfx94x)
   - `repo_base_url`: `https://rocm.nightlies.amd.com` (optional)
5. Click "Run workflow"

### Expected Output

```
==========================================================================
FULL INSTALLATION TEST - NATIVE LINUX PACKAGES
==========================================================================

Package Type: DEB
Repository Base URL: https://rocm.nightlies.amd.com
Artifact ID: 21658678136
Build Date: 20260204
ROCm Version: 8.0.0
OS Profile: ubuntu2404
GPU Architecture: gfx94x
Install Prefix: /opt/rocm/core

Repository URL: https://rocm.nightlies.amd.com/deb/20260204-21658678136/pool/main/

================================================================================
SETTING UP DEB REPOSITORY
================================================================================

Repository URL: https://rocm.nightlies.amd.com/deb/20260204-21658678136/pool/main/
OS Profile: ubuntu2404

Adding ROCm repository...
[PASS] Repository added to /etc/apt/sources.list.d/rocm-test.list
       deb [arch=amd64] https://rocm.nightlies.amd.com/deb/20260204-21658678136/pool/main/ ./

Updating package lists...
================================================================================
[PASS] Package lists updated

================================================================================
INSTALLING DEB PACKAGES FROM REPOSITORY
================================================================================

Package to install: amdrocm-gfx94x

Running: apt install -y amdrocm-gfx94x
================================================================================
Installation progress (streaming output):

Reading package lists...
Building dependency tree...
...
[PASS] DEB packages installed successfully from repository

================================================================================
VERIFYING ROCM INSTALLATION
================================================================================

[PASS] Installation directory exists: /opt/rocm/core

Checking for key ROCm components:
   [PASS] bin/rocminfo
   [PASS] bin/hipcc
   [PASS] include/hip/hip_runtime.h
   [PASS] lib/libamdhip64.so

Components found: 4/4

Checking installed packages:
   Found 25 ROCm packages installed

   Sample packages:
      amdrocm-core
      amdrocm-hip
      amdrocm-gfx94x
      ...

Trying to run rocminfo...
   [PASS] rocminfo executed successfully

   First few lines of rocminfo output:
      ROCm version: 8.0.0
      ...

[PASS] ROCm installation verification PASSED

================================================================================
[PASS] FULL INSTALLATION TEST PASSED

ROCm has been successfully installed from repository and verified!
================================================================================
```

---

## Error Handling

### Common Errors and Solutions

#### 1. Repository Setup Fails
```
[FAIL] Failed to add repository
```
**Solutions:**
- Check repository URL is accessible
- Verify date and artifact_id are correct
- Check network connectivity
- Verify OS profile matches container image

#### 2. Package Installation Fails
```
[FAIL] Failed to install DEB/RPM packages
```
**Solutions:**
- Check for dependency issues
- Verify package compatibility with OS
- Check disk space
- Review package metadata
- Verify gfx_arch matches available packages
- Check repository URL structure

#### 3. Verification Fails
```
[FAIL] ROCm installation verification FAILED
```
**Solutions:**
- Check installation logs
- Verify install prefix is correct (default: `/opt/rocm/core`)
- Check file permissions
- Review package contents
- Verify at least 2 key components are found

#### 4. Date Format Error
```
Invalid date format: {date}. Must be YYYYMMDD (e.g., 20260204)
```
**Solutions:**
- Ensure date is exactly 8 digits
- Format: YYYYMMDD (e.g., 20260204 for February 4, 2026)
- No dashes or slashes

---

## Key Features

### ✅ Complete End-to-End Testing
- Sets up real ROCm nightly repositories
- Installs packages using native package managers
- Verifies actual installation in clean environment

### ✅ Container Isolation
- Tests in Ubuntu 24.04 (DEB) or AlmaLinux 9/RHEL 8 (RPM)
- No impact on host system
- Reproducible environment

### ✅ Comprehensive Verification
- Checks directory structure
- Validates key components
- Attempts to run ROCm tools
- Reports detailed status

### ✅ Flexible Configuration
- Supports multiple OS profiles
- Configurable install prefix (default: `/opt/rocm/core`)
- Works with different GPU architectures
- Supports custom repository base URLs

### ✅ Real-Time Output
- Streams installation progress
- Shows detailed repository setup
- Provides immediate feedback

---

## Supported OS Profiles

### DEB (Debian-based)
- `ubuntu2404` - Ubuntu 24.04
- `ubuntu2204` - Ubuntu 22.04
- `debian12` - Debian 12

### RPM (Red Hat-based)
- `rhel8` - Red Hat Enterprise Linux 8
- `rhel9` - Red Hat Enterprise Linux 9
- `sles16` - SUSE Linux Enterprise Server 16

---

## Supported GPU Architectures

- `gfx94x` - Default (e.g., MI300 series)
- `gfx110x` - RDNA 3 (e.g., RX 7900 series)
- `gfx1151` - Strix Halo
- Other architectures as available

---

## Next Steps

1. **Run Full Test after building packages**
   - Ensures packages work before production
   - Validates repository structure

2. **Integrate into CI/CD pipeline**
   - Call native_linux_packages_full_test.yml after build_native_linux_packages.yml  passes
   - Automate testing for all supported OS profiles

3. **Add GPU-specific tests**
   - If GPU hardware available, run rocminfo
   - Test HIP compilation
   - Run sample programs

4. **Add performance benchmarks**
   - Measure installation time
   - Check package sizes
   - Verify startup time

5. **Expand OS profile support**
   - Add more OS profiles as needed
   - Test compatibility across distributions

---

## Author

Created for ROCm/TheRock project  
Branch: users/acheruva/build_test_J27

