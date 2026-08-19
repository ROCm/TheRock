#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ==============================================================================
# TheRock Automated Zero-to-Hero Bootstrapper
#
# Automates:
# 1. Installing uv, compilers, and system build prerequisites.
# 2. Creating structured workspace: ~/virtualenv/therock-<ver>/py<pyver>-<preset>/
# 3. Cloning user's TheRock fork repository if not present.
# 4. Creating dedicated Python virtual environment per (ROCm + Python + Preset).
# 5. Configuring, building, and hermetically installing ROCm into that virtual environment.
# ==============================================================================

set -euo pipefail

# Default Configuration
ROCM_VER="7.14"
PYTHON_VER="3.14"
PRESET="llm"
BRANCH="main"
REPO_URL="https://github.com/analogbox/TheRock.git"
BASE_DIR="$HOME/virtualenv"
DRY_RUN=false
INSTALL_SYSDEPS=false

# Parse Command Line Arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --rocm) ROCM_VER="$2"; shift 2 ;;
        --python) PYTHON_VER="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --install-sysdeps) INSTALL_SYSDEPS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: ./bootstrap.sh [OPTIONS]"
            echo "Options:"
            echo "  --rocm <version>     TheRock / ROCm version tag (default: 7.14)"
            echo "  --python <version>   Python version to use (default: 3.14)"
            echo "  --preset <name>      Build preset: llm, hip, vulkan, math, ai, hpc, full (default: llm)"
            echo "  --branch <name>      Git branch to checkout (default: feature/ubuntu-26.04-gcc15-gfx1151)"
            echo "  --repo-url <url>     Git repository URL to clone"
            echo "  --base-dir <path>    Base directory for virtualenvs (default: ~/virtualenv)"
            echo "  --install-sysdeps    Force installing apt system packages with sudo"
            echo "  --dry-run            Print what would be executed without running build"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "\033[1;34m====================================================================\033[0m"
echo -e "\033[1;32m  TheRock Automated Workspace & Environment Bootstrapper\033[0m"
echo -e "  ROCm / TheRock Version : \033[1;36m$ROCM_VER\033[0m"
echo -e "  Python Version         : \033[1;36m$PYTHON_VER\033[0m"
echo -e "  Build Preset           : \033[1;36m$PRESET\033[0m"
echo -e "\033[1;34m====================================================================\033[0m"

# Step 1: Install System Dependencies if needed
if [ "$INSTALL_SYSDEPS" = true ] || ! command -v ninja &> /dev/null || ! command -v cmake &> /dev/null; then
    echo -e "\n\033[1;33m[1/5] Installing system packages (apt)...\033[0m"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -y
        sudo apt-get install -y \
            build-essential gcc g++ gfortran git ninja-build cmake \
            pkg-config xxd automake libtool python3-dev libegl1-mesa-dev \
            libsqlite3-dev texinfo bison flex curl make ccache
    fi
else
    echo -e "\n\033[1;32m[1/5] Essential build tools (cmake, ninja, gcc) already present. Skipping apt.\033[0m"
fi

# Step 2: Ensure `uv` is installed
echo -e "\n\033[1;33m[2/5] Ensuring 'uv' is installed...\033[0m"
if ! command -v uv &> /dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "Using uv at: $(command -v uv)"

# Step 3: Setup Hierarchical Workspace Directory Structure
PY_SLUG="${PYTHON_VER//./}"
WORKSPACE_ROOT="$BASE_DIR/therock-$ROCM_VER"
SOURCE_DIR="$WORKSPACE_ROOT/TheRock"
ENV_DIR="$WORKSPACE_ROOT/py${PY_SLUG}-${PRESET}"
VENV_DIR="$ENV_DIR/.venv"
BUILD_DIR="$ENV_DIR/build"

echo -e "\n\033[1;33m[3/5] Setting up workspace directory structure...\033[0m"
echo "  Workspace Root : $WORKSPACE_ROOT"
echo "  Source Code    : $SOURCE_DIR"
echo "  Environment    : $ENV_DIR"
echo "  Virtualenv     : $VENV_DIR"
echo "  Build Output   : $BUILD_DIR"

mkdir -p "$WORKSPACE_ROOT" "$ENV_DIR"

# Step 4: Clone / Update TheRock Source Repository & Top-Level Submodules
echo -e "\n\033[1;33m[4/5] Checking TheRock source repository and submodules...\033[0m"
if [ ! -d "$SOURCE_DIR/.git" ]; then
    echo "Cloning $REPO_URL (branch: $BRANCH) into $SOURCE_DIR..."
    git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$SOURCE_DIR"
    (cd "$SOURCE_DIR" && git submodule update --init --depth 1 rocm-systems rocm-libraries third-party compiler 2>/dev/null || git submodule update --init --depth 1)
else
    echo "Existing TheRock source repository found at: $SOURCE_DIR"
    (cd "$SOURCE_DIR" && git checkout "$BRANCH" 2>/dev/null || true)
    (cd "$SOURCE_DIR" && git submodule update --init --depth 1 rocm-systems rocm-libraries third-party compiler 2>/dev/null || git submodule update --init --depth 1)
fi

# Step 5: Provision Virtual Environment and Run Build Orchestrator
echo -e "\n\033[1;33m[5/5] Provisioning Python $PYTHON_VER virtualenv and building preset '$PRESET'...\033[0m"

# Execute therock-env inside the cloned repository
cd "$SOURCE_DIR"

BUILD_ARGS=(
    "build"
    "--preset" "$PRESET"
    "--python" "$PYTHON_VER"
    "--venv-dir" "$VENV_DIR"
    "--build-dir" "$BUILD_DIR"
)

if [ "$DRY_RUN" = true ]; then
    BUILD_ARGS+=("--dry-run")
fi

# Run therock_env.py orchestrator targeting the isolated ENV_DIR / VENV_DIR
python3 "$SOURCE_DIR/build_tools/therock_env.py" "${BUILD_ARGS[@]}"

echo -e "\n\033[1;32m====================================================================\033[0m"
echo -e "\033[1;32m  Environment Successfully Provisioned & Built!\033[0m"
echo -e "  To activate this exact isolated ROCm environment:"
echo -e "    \033[1;36msource $VENV_DIR/bin/activate\033[0m"
echo -e "\033[1;32m====================================================================\033[0m\n"
