# Test Environment Reproduction

This guide explains how to reproduce test failures from CI locally.

## Quick Start (Automated)

The easiest way to reproduce a test failure is using the `reproduce_test_failure.py`
script, which automates the entire setup process.

```bash
git clone https://github.com/ROCm/TheRock.git
cd TheRock
python build_tools/github_actions/reproduce_test_failure.py --run-id {CI_RUN_ID} --repository {GITHUB_REPO} --amdgpu-family {GPU_FAMILY} --test-script "{TEST_SCRIPT}"
```

Options:

- `--setup-only`: Set up the environment and drop into a shell without running
  the test

When a test fails in CI, the reproduction command is printed in the job output.

## Linux

Linux reproduction uses Docker to ensure a clean environment with ROCm sourced
from TheRock.

### Docker Image

The base image is available at:

```
ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest
```

### Manual Steps

If you prefer to set up manually:

```bash
docker run -it \
    --ipc host \
    --group-add video \
    --device /dev/kfd \
    --device /dev/dri \
    ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest /bin/bash

# Inside the container:
curl -LsSf https://astral.sh/uv/install.sh | bash && source $HOME/.local/bin/env
git clone https://github.com/ROCm/TheRock.git && cd TheRock
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements-test.txt
GITHUB_REPOSITORY={GITHUB_REPO} python build_tools/install_rocm_from_artifacts.py \
    --run-id {CI_RUN_ID} \
    --amdgpu-family {GPU_FAMILY} \
    {ADDITIONAL_FLAGS}
export THEROCK_BIN_DIR=./therock-build/bin
export OUTPUT_ARTIFACTS_DIR=./therock-build

# Run your test
python build_tools/github_actions/test_executable_scripts/test_rocblas.py
```

## Windows

Windows reproduction runs on bare metal and installs dependencies via Chocolatey.

### Prerequisites

- Windows machine with AMD GPU
- Administrator PowerShell access

### Installed Packages

The script will install:

- Chocolatey (package manager)
- git, python, cmake, ninja, ccache
- uv (Python package manager)

### Manual Steps

If you prefer to set up manually:

```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
Import-Module $env:ChocolateyInstall\\helpers\\chocolateyProfile.psm1; refreshenv

# Install dependencies
choco install -y git python cmake ninja ccache

# Install uv
irm https://astral.sh/uv/install.ps1 | iex

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')

# Clone and setup
git clone https://github.com/ROCm/TheRock.git
Set-Location TheRock
uv venv .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements-test.txt

# Download artifacts
$env:GITHUB_REPOSITORY='{GITHUB_REPO}'
python build_tools/install_rocm_from_artifacts.py `
    --run-id {CI_RUN_ID} `
    --amdgpu-family {GPU_FAMILY} `
    {ADDITIONAL_FLAGS}

# Set environment
$env:THEROCK_BIN_DIR='./therock-build/bin'
$env:OUTPUT_ARTIFACTS_DIR='./therock-build'

# Run your test
python build_tools/github_actions/test_executable_scripts/test_rocblas.py
```

## Parameters

| Parameter          | Description                                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CI_RUN_ID`        | GitHub Actions run ID (e.g., from `https://github.com/ROCm/TheRock/actions/runs/16948046392` → `16948046392`)                                          |
| `GPU_FAMILY`       | LLVM target name (e.g., `gfx94X-dcgpu`, `gfx1151`, `gfx110X-all`)                                                                                      |
| `GITHUB_REPO`      | Repository where the CI run was executed (e.g., `ROCm/TheRock`, `ROCm/rocm-libraries`)                                                                 |
| `ADDITIONAL_FLAGS` | Optional flags for `install_rocm_from_artifacts.py`. See [installing_artifacts.md](installing_artifacts.md#component-selection) for available options. |
| `TEST_SCRIPT`      | The test command to run (e.g., `python build_tools/github_actions/test_executable_scripts/test_rocblas.py`)                                            |

## Test Scripts

Available test wrappers are in
[`build_tools/github_actions/test_executable_scripts/`](https://github.com/ROCm/TheRock/tree/main/build_tools/github_actions/test_executable_scripts).
