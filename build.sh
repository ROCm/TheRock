#! /bin/bash

set -euox pipefail

# ----------------------------------------------------------------------
# Verify we're in the TheRock directory
# ----------------------------------------------------------------------
CURRENT_DIR=$(basename "$PWD")
if [[ "$CURRENT_DIR" != "TheRock" ]]; then
  echo "Error: This script must be run from the TheRock directory."
  echo "Current directory: $PWD"
  echo "Please cd to the TheRock directory and run this script again."
  exit 1
fi

# ----------------------------------------------------------------------
# Variables that control build
# ----------------------------------------------------------------------
# Whether to use local path for rocm-libraries
LOCAL_ROCM_LIBRARIES=true

# rocm_agent_enumerator prints gfx number for each device, head -n 1 selects first line
THEROCK_ASIC=$(rocm_agent_enumerator | head -n 1)

# preset for fusilli-plugin build
PRESET_NAME="fusilli-plugin-build"

# Whether to apt-get build dependencies (such as patchelf)
INSTALL_DEPS=false

# Whether build TheRock
CONFIGURE=true

# Whether to install CMake preset, always true if build is true
INSTALL_PRESET=true

# ----------------------------------------------------------------------
#  Parse arguments and validate build variables
# ----------------------------------------------------------------------
function help() {
  cat <<EOF
Usage: ./build.sh [OPTIONS]

Configures TheRock development environment with pre-built options.

Options:
  -h,  --help                       Display this help message
  -i,  --install-deps               Apt get install dependencies

  --no-local-rocm-libraries         Use the GitHub URL for rocm-libraries instead
                                    of the local checkout at /home/astgeorg/Dev/c++/rocm-libraries
  --no-configure                    Don't configure TheRock, only do requisite setup
  --no-install-preset               Don't add CMakePresetsLocal.json

Examples:
  ./build.sh                        # Use local rocm-libraries
  ./build.sh -nl                    # Use GitHub URL for rocm-libraries
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  -h | --help)
    help
    exit
    ;;
  -i | --install-deps)
    INSTALL_DEPS=true
    ;;
  --no-configure)
    CONFIGURE=false
    ;;
  --no-local-rocm-libraries)
    LOCAL_ROCM_LIBRARIES=false
    ;;
  --no-build-preset)
    INSTALL_PRESET=false
    ;;
  *)
    echo "unknown flag $1"
    help
    exit 1
    ;;
  esac
  shift
done

# ----------------------------------------------------------------------
# apt-get install build dependencies (such as patchelf)
# ----------------------------------------------------------------------
if [[ $INSTALL_DEPS == true ]]; then
  sudo apt-get update && sudo apt-get install -y \
    git \
    gdb \
    gfortran \
    git-lfs \
    ninja-build \
    cmake \
    g++ \
    pkg-config \
    xxd \
    patchelf \
    automake \
    python3-venv \
    python3-dev \
    libegl1-mesa-dev \
    curl \
    unzip \
    patchelf \
    libtool \
    python3-pip \
    vim \
    rsync \
    ca-certificates \
    wget \
    jq
fi

# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
# should upgrade environment if it already exists, harmless I would assume?
python3 -m venv .venv --prompt the-rock
# setup python
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ----------------------------------------------------------------------
# Configure submodules
# ----------------------------------------------------------------------

# Fetch stock submodules
python3 ./build_tools/fetch_sources.py

# Override rocm-libraries with local path if requested
if [[ $LOCAL_ROCM_LIBRARIES == true ]]; then
  echo "Configuring rocm-libraries to use local path..."

  # Check for uncommitted changes in local rocm-libraries
  LOCAL_ROCM_PATH="/home/astgeorg/Dev/c++/rocm-libraries"
  if ! git -C "$LOCAL_ROCM_PATH" diff-index --quiet HEAD --; then
    echo "Error: Uncommitted changes detected in $LOCAL_ROCM_PATH"
    echo "Please commit or stash your changes before running this script."
    exit 1
  fi

  # Get current branch from local rocm-libraries
  LOCAL_BRANCH=$(cd "$LOCAL_ROCM_PATH" && git branch --show-current)
  echo "Local rocm-libraries branch: $LOCAL_BRANCH"

  # Push current branch to origin
  echo "Pushing $LOCAL_BRANCH to origin..."
  (cd "$LOCAL_ROCM_PATH" && git push origin "$LOCAL_BRANCH")

  # Initialize the submodule if not already done
  git submodule update --init rocm-libraries

  # Configure AaronStGeorge remote in submodule
  (cd rocm-libraries && git remote remove AaronStGeorge || true)
  (cd rocm-libraries && git remote add AaronStGeorge git@github.com:AaronStGeorge/rocm-libraries.git)

  # Fetch and checkout the branch from AaronStGeorge
  echo "Updating rocm-libraries to latest commit from remote branch: $LOCAL_BRANCH..."
  (cd rocm-libraries && git fetch AaronStGeorge && git checkout "AaronStGeorge/$LOCAL_BRANCH")

  # Verify both repositories are on the same commit
  LOCAL_COMMIT=$(cd "$LOCAL_ROCM_PATH" && git rev-parse HEAD)
  SUBMODULE_COMMIT=$(cd rocm-libraries && git rev-parse HEAD)

  if [[ "$LOCAL_COMMIT" != "$SUBMODULE_COMMIT" ]]; then
    echo "Error: Commit mismatch between local and submodule!"
    echo "Local rocm-libraries commit: $LOCAL_COMMIT"
    echo "Submodule commit: $SUBMODULE_COMMIT"
    echo "This should not happen - please investigate."
    exit 1
  fi

  # Show which commit we're now using
  ROCM_COMMIT=$(cd rocm-libraries && git rev-parse --short HEAD)
  echo "rocm-libraries now at commit: $ROCM_COMMIT (verified match with local)"
fi


# Overwrite CMakePresetsLocal.json with local configurations
cat > CMakeUserPresets.json << EOF
{
  "version": 6,
  "configurePresets": [
    {
      "name": "${PRESET_NAME}",
      "description": "Builds only fusilli-plugin, Ninja generator, default compiler, RelWithDebInfo",
      "generator": "Ninja",
      "binaryDir": "\${sourceDir}/build",
      "environment": {
        "PATH": "\${sourceDir}/.venv/bin:$PATH"
      },
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "amd-llvm_BUILD_TYPE": "Release",
        "THEROCK_ENABLE_ALL": "OFF",
        "THEROCK_ENABLE_FUSILLI_PLUGIN": "ON",
        "Python3_EXECUTABLE": "$PWD/.venv/bin/python",
        "THEROCK_SPLIT_DEBUG_INFO": "ON",
        "THEROCK_QUIET_INSTALL": "OFF",
        "CMAKE_INSTALL_PREFIX": "..",
        "THEROCK_AMDGPU_FAMILIES": "$THEROCK_ASIC"
      }
    }
  ]
}
EOF

# build
if [[ $CONFIGURE == true ]]; then
  mkdir -p build
  cd build
  cmake .. \
      --preset=$PRESET_NAME
fi

if [[ $INSTALL_PRESET == false ]]; then
  rm CMakeUserPresets.json
fi
