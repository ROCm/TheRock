# Using GitHub Actions to Trigger ROCm+PyTorch Dev Builds

> **⚠️ Important Context**
>
> This guide documents the current process for triggering pytorch dev builds to assist with debugging and development of changes in TheRock and ROCm software stack.
> Please review issues [#2587](https://github.com/ROCm/TheRock/issues/2587) and [#2608](https://github.com/ROCm/TheRock/issues/2608) for known limitations and ongoing improvements to processes and documentation.

## Overview

This guide explains how to use TheRock's GitHub Actions workflows to trigger ROCm and PyTorch development builds for specific GPU architectures with custom submodule branches or commits for debugging and development. Your GitHub account must have access to manually trigger workflows in TheRock repo to follow this guide.

TheRock provides the `release_portable_linux_packages.yml` workflow for building ROCm SDK packages. Once completed, this workflow automatically triggers the PyTorch wheel build via `release_portable_linux_pytorch_wheels.yml`.

Note that the full stack build can take several hours, depending on system availability.

## Setting Up Your Development Environment

### Initial Environment Setup

Review the documentation on [TheRock's README](https://github.com/ROCm/TheRock/blob/main/README.md) to setup your development environment and fetch the ROCm sources.

### Create Your Development Branch

```bash
# Create a development branch on TheRock for the custom build
git checkout -b users/github-account/dev-branch
```

## Preparing Your Development Branch

### Setting Up Custom Submodule References

To test changes in ROCm subprojects, you need to update the submodule references in the branch created above.

#### Updating a Submodule

```bash
cd /path/to/TheRock
pushd rocm-libraries  # Navigate to the submodule

# Checkout the desired branch, tag, or commit
git checkout fix/my-gemm-fix      # branch
# or
git checkout a1b2c3d4e5f6         # specific commit

# Return to TheRock root
popd

# Update the submodule reference
git add rocm-libraries
git commit -m "Set rocm-libraries to fix/my-gemm-fix"
git push origin users/github-account/dev-branch
```

#### Updating Multiple Submodules

For changes across multiple subprojects:

```bash
cd /path/to/TheRock

# Update first submodule
pushd rocm-libraries
git checkout feature/my-optimization
popd
git add rocm-libraries

# Update second submodule
pushd compiler/amd-llvm
git checkout llvm30bringup
popd
git add compiler/amd-llvm

# Commit all changes
git commit -m "Update rocm-libraries and amd-llvm for testing"
git push origin users/github-account/dev-branch
```

**Note:** See [`.gitmodules`](https://github.com/ROCm/TheRock/blob/main/.gitmodules) for the complete list of submodule paths.

#### Accounting for Patches

Depending on the state of the subprojects in your development branch for this process, the patches from the `patches` directory may need to be modified or dropped in order for the compilation to proceed. Refer to the [documentation on patches in TheRock](https://github.com/ROCm/TheRock/blob/main/patches/README.md).

## Triggering the Workflow

### Accessing the Workflow

1. Navigate to the [Release portable Linux packages workflow](https://github.com/ROCm/TheRock/actions/workflows/release_portable_linux_packages.yml)
1. Click **"Run workflow"** button (top right)
1. Select your development branch from the previous step in the **"Use workflow from"** dropdown.
1. It is important to leave the first input as `dev`.
1. The main input that will be different than the default options will be the `amdgpu_families` input. Please only select the GPU architectures relevant to the development or debugging effort.

### Example: Single GPU Family

To build for gfx110X (RX 7000 series):

```
Branch: users/github-account/dev-branch
amdgpu_families: gfx110X
```

### Example: Multiple GPU Families

To build for multiple architectures:

```
Branch: users/github-account/dev-branch
amdgpu_families: gfx94X,gfx103X
```

**Note:** Multiple amdgpu_families inputs will create parallel build jobs and increase compute resource usage.

## Monitoring the Build

### ROCm Build

After triggering the workflow, a run entry mapped to the branch name will appear on the Actions tab.
It will have a URL in the following format: `https://github.com/ROCm/TheRock/actions/runs/[RUN_ID]`

Note the run ID from the URL if you wish to download its artifacts locally.

[Additional documentation on debugging GitHub Actions](https://github.com/ROCm/TheRock/blob/main/docs/development/github_actions_debugging.md).

### Automatic PyTorch Build

The PyTorch wheel build triggers automatically upon successful ROCm build completion. You will see a new workflow run appear on the Actions tab shortly after the ROCm build finishes.
