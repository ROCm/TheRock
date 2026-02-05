# Full Test Workflow Documentation

## Overview

This document describes the **Full Test of Native Linux Packages** workflow and supporting tools for testing ROCm package installation from S3.

## Files Created

### 1. Workflow File
**Location:** `.github/workflows/Full_Test.yml`

**Purpose:** GitHub Actions workflow that performs complete end-to-end testing of ROCm packages by:
- Downloading packages from S3
- Installing them in a clean container environment
- Verifying the installation

### 2. Python Test Script
**Location:** `build_tools/packaging/linux/package_full_test.py`

**Purpose:** Python script that handles the actual package download, installation, and verification.

---

## Workflow: Full_Test.yml

### Trigger Methods

1. **Workflow Call** - Called by other workflows
2. **Manual Dispatch** - Manual trigger via GitHub Actions UI

### Input Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `artifact_group` | No | `gfx94X-dcgpu` | GPU architecture group |
| `artifact_run_id` | Yes | - | Workflow run ID to download artifacts from |
| `rocm_version` | Yes | `0.0.1` | ROCm version (e.g., 8.0.0, 8.0.1rc1) |
| `native_package_type` | Yes | `deb` | Package type: `deb` or `rpm` |
| `release_type` | No | `dev` | Release type: dev/nightly/prerelease |
| `s3_download_path` | No | - | Custom S3 path (optional) |

### Workflow Steps

#### Step 1: Install System Prerequisites
- Installs Python, git, curl, AWS CLI dependencies
- Different packages based on DEB (Ubuntu) vs RPM (AlmaLinux)

#### Step 2: Checkout Repository
- Clones TheRock repository into container
- Checks out the correct branch

#### Step 3: Install Python Dependencies
- Installs: `boto3`, `pyelftools`, `requests`

#### Step 4: Install AWS CLI
- Uses TheRock's install script for AWS CLI

#### Step 5: Configure AWS Credentials
- Authenticates with AWS using OIDC
- Assumes role: `therock-{release_type}`

#### Step 6: Full Package Installation Test
- **Main test step** - Runs `package_full_test.py`
- Downloads packages from S3
- Installs packages using apt/dnf
- Verifies installation

#### Step 7: Verify ROCm Installation
- Checks for ROCm directory structure
- Validates key components exist
- Attempts to run rocminfo

#### Step 8: Test Report
- Generates summary report
- Shows success/failure status

### Container Environment

| Package Type | Container Image | Purpose |
|--------------|----------------|---------|
| DEB | `ubuntu:24.04` | Test Debian packages on Ubuntu |
| RPM | `almalinux:9` | Test RPM packages on AlmaLinux |

**Note:** Containers run in `--privileged` mode for full system access

---

## Python Script: package_full_test.py

### Class: PackageFullTester

Main class that handles the full installation test process.

#### Constructor Parameters

```python
PackageFullTester(
    package_type: str,          # 'deb' or 'rpm'
    s3_bucket: str,             # S3 bucket name
    artifact_group: str,        # GPU architecture group
    artifact_id: str,           # Artifact run ID
    rocm_version: str,          # ROCm version
    download_dir: str,          # Download directory path
    install_prefix: str,        # Installation prefix (default: /opt/rocm)
    s3_path: Optional[str]      # Custom S3 path (optional)
)
```

### Key Methods

#### 1. `download_packages_from_s3() -> List[Path]`

**Purpose:** Downloads packages from S3 bucket

**Process:**
1. Constructs S3 path (or uses custom path)
2. Uses AWS CLI `s3 sync` to download packages
3. Filters by file extension (.deb or .rpm)
4. Returns list of downloaded package paths

**S3 Path Structure:**
```
s3://{bucket}/{artifact_group}/{artifact_id}/
```

Example:
```
s3://therock-dev-packages/gfx94X-dcgpu/12345/
```

#### 2. `install_deb_packages(packages: List[Path]) -> bool`

**Purpose:** Installs DEB packages using apt

**Process:**
1. Updates apt cache
2. Runs `apt install -y {packages}`
3. Returns success/failure

#### 3. `install_rpm_packages(packages: List[Path]) -> bool`

**Purpose:** Installs RPM packages using dnf

**Process:**
1. Runs `dnf install -y {packages}`
2. Returns success/failure

#### 4. `verify_rocm_installation() -> bool`

**Purpose:** Verifies ROCm installation

**Checks:**
- ✅ Installation directory exists (`/opt/rocm`)
- ✅ Key components present:
  - `bin/rocminfo`
  - `bin/hipcc`
  - `include/hip/hip_runtime.h`
  - `lib/libamdhip64.so`
- ✅ ROCm packages are installed (via dpkg/rpm)
- ✅ Attempts to run `rocminfo` (if available)

**Success Criteria:** At least 2 out of 4 key components found

#### 5. `run() -> bool`

**Purpose:** Main execution method

**Process:**
1. Download packages from S3
2. Install packages
3. Verify installation
4. Return overall success/failure

### Command Line Usage

```bash
# DEB packages
python package_full_test.py \
    --package-type deb \
    --s3-bucket therock-dev-packages \
    --artifact-group gfx94X-dcgpu \
    --artifact-id 12345 \
    --rocm-version 8.0.0 \
    --download-dir /tmp/rocm_packages

# RPM packages
python package_full_test.py \
    --package-type rpm \
    --s3-bucket therock-nightly-packages \
    --artifact-group gfx110X-all \
    --artifact-id 67890 \
    --rocm-version 8.0.1 \
    --download-dir /tmp/rocm_packages \
    --install-prefix /opt/rocm
```

---

## Workflow Comparison

### Test.yml (Simulation Test)
- ✅ Fast (~30 seconds)
- ✅ Uses `--simulate` (no actual install)
- ✅ Validates package metadata
- ✅ Checks dependencies
- ❌ Doesn't test actual installation
- ❌ Doesn't verify file placement

### Full_Test.yml (Full Installation Test)
- ⏱️ Slower (~5-15 minutes)
- ✅ Downloads from S3
- ✅ Actually installs packages
- ✅ Verifies file placement
- ✅ Tests real installation
- ✅ Runs in clean container

---

## Complete Testing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  build_native_linux_packages.yml                        │
│  - Builds DEB/RPM packages                              │
│  - Uploads to S3                                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├──────────────┬─────────────────────────┐
                 ▼              ▼                         ▼
    ┌────────────────┐  ┌──────────────┐    ┌────────────────────┐
    │   Test.yml     │  │ Full_Test    │    │  Production       │
    │   (Simulate)   │  │  .yml        │    │  Deployment       │
    │                │  │  (Full Test) │    │                   │
    │ Quick check    │  │ Complete     │    │ After all tests   │
    │ Dependencies   │  │ Installation │    │ pass              │
    │ Conflicts      │  │ Verification │    │                   │
    └────────────────┘  └──────────────┘    └────────────────────┘
```

---

## Usage Examples

### Manual Workflow Trigger

1. Go to GitHub Actions
2. Select "Full Test of Native Linux Packages"
3. Click "Run workflow"
4. Fill in parameters:
   - `artifact_group`: `gfx94X-dcgpu`
   - `artifact_run_id`: `12345` (from build workflow)
   - `rocm_version`: `8.0.0`
   - `native_package_type`: `deb` or `rpm`
   - `release_type`: `dev`
5. Click "Run workflow"

### Expected Output

```
==========================================================================
FULL INSTALLATION TEST - NATIVE LINUX PACKAGES
==========================================================================
Package Type: DEB
Artifact Group: gfx94X-dcgpu
Artifact Run ID: 12345
ROCm Version: 8.0.0
Release Type: dev
S3 Bucket: therock-dev-packages
==========================================================================

==========================================================================
DOWNLOADING PACKAGES FROM S3
==========================================================================
S3 Path: s3://therock-dev-packages/gfx94X-dcgpu/12345/
Download Directory: /tmp/rocm_packages

✅ Downloaded 25 packages:
   - amdrocm-core_8.0.0-12345_amd64.deb (15.23 MB)
   - amdrocm-hip_8.0.0-12345_amd64.deb (125.45 MB)
   ...

==========================================================================
INSTALLING DEB PACKAGES
==========================================================================
Packages to install (25):
   - amdrocm-core_8.0.0-12345_amd64.deb
   - amdrocm-hip_8.0.0-12345_amd64.deb
   ...

✅ DEB packages installed successfully

==========================================================================
VERIFYING ROCM INSTALLATION
==========================================================================
✅ Installation directory exists: /opt/rocm

Checking for key ROCm components:
   ✅ bin/rocminfo
   ✅ bin/hipcc
   ✅ include/hip/hip_runtime.h
   ✅ lib/libamdhip64.so

Components found: 4/4

✅ ROCm installation verification PASSED

==========================================================================
✅ FULL INSTALLATION TEST PASSED
==========================================================================
```

---

## Error Handling

### Common Errors and Solutions

#### 1. S3 Download Fails
```
❌ Failed to download packages from S3
```
**Solutions:**
- Check AWS credentials are configured
- Verify S3 bucket exists
- Confirm artifact_run_id is correct
- Check S3 path structure

#### 2. Package Installation Fails
```
❌ Failed to install DEB/RPM packages
```
**Solutions:**
- Check for dependency issues
- Verify package compatibility with OS
- Check disk space
- Review package metadata

#### 3. Verification Fails
```
❌ ROCm installation verification FAILED
```
**Solutions:**
- Check installation logs
- Verify install prefix is correct
- Check file permissions
- Review package contents

---

## Key Features

### ✅ Complete End-to-End Testing
- Downloads real packages from S3
- Installs in clean environment
- Verifies actual installation

### ✅ Container Isolation
- Tests in Ubuntu 24.04 (DEB) or AlmaLinux 9 (RPM)
- No impact on host system
- Reproducible environment

### ✅ Comprehensive Verification
- Checks directory structure
- Validates key components
- Attempts to run ROCm tools
- Reports detailed status

### ✅ Flexible Configuration
- Supports custom S3 paths
- Configurable install prefix
- Works with dev/nightly/prerelease

---

## Next Steps

1. **Run Full Test after building packages**
   - Ensures packages work before production
   
2. **Integrate into CI/CD pipeline**
   - Call Full_Test.yml after Test.yml passes
   
3. **Add GPU-specific tests**
   - If GPU hardware available, run rocminfo
   - Test HIP compilation
   - Run sample programs

4. **Add performance benchmarks**
   - Measure installation time
   - Check package sizes
   - Verify startup time

---

## Author

Created for ROCm/TheRock project  
Branch: users/acheruva/build_test_J27


