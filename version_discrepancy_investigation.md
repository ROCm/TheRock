# Version Discrepancy Investigation: Nightly (7.11) vs Dev (7.10) Tarballs

## Summary
The version discrepancy between nightly build tarballs (7.11) and dev tarballs (7.10) is caused by **different branches** being used for these two types of builds.

## Root Cause Analysis

### 1. Version Configuration
The version number is stored in `version.json` and differs across branches:
- **Main branch**: `"rocm-version": "7.11.0"`
- **Release branch (release/therock-7.10)**: `"rocm-version": "7.10.0"`

### 2. Build Workflows

#### Nightly Builds (`release_portable_linux_packages.yml` and `release_windows_packages.yml`)
- **Trigger**: Scheduled cron job (`schedule: cron: '0 04 * * *'`)
- **Branch used**: `main` (default branch for scheduled workflows)
- **Version computation**: 
  - Line 92-94: `python ./build_tools/compute_rocm_package_version.py --release-type=nightly`
  - When triggered by schedule, `release_type` defaults to `'nightly'` (line 69/73)
  - Reads from `version.json` on `main` branch → **7.11.0**
  - Adds suffix `a{YYYYMMDD}`
  - **Result**: `7.11.0aYYYYMMDD`

#### Dev Builds (via `setup.yml` used by `ci.yml` and `ci_nightly.yml`)
- **Trigger**: Manual workflow_dispatch or pull requests
- **Branch used**: Typically the current working branch or PR branch (often `release/therock-7.10` or feature branches)
- **Version computation**:
  - Line 79: `python ./build_tools/compute_rocm_package_version.py --release-type=dev`
  - Reads from `version.json` on the checked-out branch → **7.10.0**
  - Adds suffix `.dev0+{git_sha}`
  - **Result**: `7.10.0.dev0+...`

### 3. Key Code Locations

#### Version Computation Logic (`build_tools/compute_rocm_package_version.py`)
```python
def compute_version(release_type, ...):
    base_version = load_rocm_version()  # Reads from version.json
    
    if release_type == "dev":
        version_suffix = f".dev0+{git_sha}"  # Dev builds
    elif release_type == "nightly":
        version_suffix = f"a{current_date}"  # Nightly builds
    
    return base_version + version_suffix
```

#### Git History
- Main branch update: `285b31bd` - "Updating version.json to 7.11.0 before next branch cut (#2257)"
- Release branch stays at: `9f586e87` - "Raise version to 7.10.0 (#1722)"

## Why This Happens

1. **Scheduled workflows** (like nightly builds) always run on the **default branch** (`main`) unless explicitly configured otherwise
2. The `main` branch has already been updated to version `7.11.0` in preparation for the next release cycle
3. **Dev builds** are typically triggered from:
   - Feature branches based on `release/therock-7.10`
   - Pull requests targeting `release/therock-7.10`
   - Manual runs from the release branch

## Solutions

### Option 1: Configure Nightly Builds to Use Release Branch
Modify the nightly workflow to explicitly checkout the release branch:

In `.github/workflows/release_portable_linux_packages.yml` and `release_windows_packages.yml`, add a `ref` input with default value:

```yaml
on:
  schedule:
    - cron: '0 04 * * *'
  workflow_dispatch:
    inputs:
      ref:
        description: "Branch, tag or SHA to checkout"
        type: string
        default: "release/therock-7.10"  # Add this
```

### Option 2: Use Branch-Specific Schedules
Create separate scheduled workflows for different release branches, or use repository variables to control which branch scheduled builds use.

### Option 3: Accept the Difference
If the intention is:
- Nightly builds = bleeding edge from `main` (7.11.0)
- Dev builds = stable development from release branch (7.10.0)

Then this behavior is actually correct and intentional.

## Recommendation

Clarify the **intended purpose** of each build type:
- If nightly builds should reflect the current release (7.10), use **Option 1**
- If nightly builds should be from the latest development (main branch), this is **working as intended**

The most likely scenario is that nightly builds should be using the release branch, so **Option 1** is recommended.

