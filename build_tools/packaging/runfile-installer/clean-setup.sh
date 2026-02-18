#!/bin/bash

# Get the directory where this script is located
OFFLINE_SELF_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if sudo is needed
SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: $SUDO

# Parse arguments
CLEAN_SETUP=0
CLEAN_BUILD=0

usage() {
cat <<END_USAGE
Usage: $0 [options]

[options]:
    setup   = Clean only setup-installer artifacts (pulled packages)
    build   = Clean only build-installer artifacts (extracted components, build outputs)
    (no args) = Clean everything (default)

Examples:
    ./clean-setup.sh         # Clean everything
    ./clean-setup.sh setup   # Clean only pulled packages
    ./clean-setup.sh build   # Clean only build artifacts

END_USAGE
}

if [ "$#" -eq 0 ]; then
    # No arguments - clean everything
    CLEAN_SETUP=1
    CLEAN_BUILD=1
else
    # Parse arguments
    for arg in "$@"; do
        case "$arg" in
            setup)
                CLEAN_SETUP=1
                ;;
            build)
                CLEAN_BUILD=1
                ;;
            help|--help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                usage
                exit 1
                ;;
        esac
    done
fi

echo =======================
if [ $CLEAN_SETUP -eq 1 ] && [ $CLEAN_BUILD -eq 1 ]; then
    echo "Cleaning up everything..."
elif [ $CLEAN_SETUP -eq 1 ]; then
    echo "Cleaning up setup artifacts..."
elif [ $CLEAN_BUILD -eq 1 ]; then
    echo "Cleaning up build artifacts..."
fi
echo =======================

pushd "$OFFLINE_SELF_BASE"

###### Setup-related cleanup (pulled packages) ######
if [ $CLEAN_SETUP -eq 1 ]; then
    echo ""
    echo ">>> Cleaning setup artifacts..."

    if [ -d "package-puller/packages" ]; then
        echo -e "\e[93mRemoving: package-puller/packages\e[0m"
        $SUDO rm -r package-puller/packages
    fi

    if [ -d "package-puller/packages-repo" ]; then
        echo -e "\e[93mRemoving: package-puller/packages-repo\e[0m"
        $SUDO rm -r package-puller/packages-repo
    fi

    if [ -d "package-puller/logs" ]; then
        echo -e "\e[93mRemoving: package-puller/logs\e[0m"
        $SUDO rm -r package-puller/logs
    fi

    if [ -f "package-puller/epel-release-latest-10.noarch.rpm" ]; then
        echo -e "\e[93mRemoving: package-puller/epel-release-latest-10.noarch.rpm\e[0m"
        $SUDO rm package-puller/epel-release-latest-10.noarch.rpm*
    fi

    if [ -f "package-puller/epel-release-latest-9.noarch.rpm" ]; then
        echo -e "\e[93mRemoving: package-puller/epel-release-latest-9.noarch.rpm\e[0m"
        $SUDO rm package-puller/epel-release-latest-9.noarch.rpm*
    fi

    if [ -f "package-puller/epel-release-latest-8.noarch.rpm" ]; then
        echo -e "\e[93mRemoving: package-puller/epel-release-latest-8.noarch.rpm\e[0m"
        $SUDO rm package-puller/epel-release-latest-8.noarch.rpm*
    fi

    if [ -d "package-extractor/packages-amdgpu" ]; then
        echo -e "\e[93mRemoving: package-extractor/packages-amdgpu\e[0m"
        $SUDO rm -r package-extractor/packages-amdgpu
    fi

    # Remove all packages-amdgpu-* directories (distro-specific)
    for amdgpu_dir in package-extractor/packages-amdgpu-*; do
        if [ -d "$amdgpu_dir" ]; then
            echo -e "\e[93mRemoving: $amdgpu_dir\e[0m"
            $SUDO rm -r "$amdgpu_dir"
        fi
    done

    if [ -d "package-extractor/packages-rocm" ]; then
        echo -e "\e[93mRemoving: package-extractor/packages-rocm\e[0m"
        $SUDO rm -r package-extractor/packages-rocm
    fi

    if [ -d "package-extractor/packages-rocm-deb" ]; then
        echo -e "\e[93mRemoving: package-extractor/packages-rocm-deb\e[0m"
        $SUDO rm -r package-extractor/packages-rocm-deb
    fi

    if [ -d "package-extractor/packages-rocm-rpm" ]; then
        echo -e "\e[93mRemoving: package-extractor/packages-rocm-rpm\e[0m"
        $SUDO rm -r package-extractor/packages-rocm-rpm
    fi

    if [ -d "package-extractor/packages-rocdecode" ]; then
        echo -e "\e[93mRemoving: package-extractor/packages-rocdecode\e[0m"
        $SUDO rm -r package-extractor/packages-rocdecode
    fi

    if [ -d "build-config" ]; then
        echo -e "\e[93mRemoving: build-config\e[0m"
        $SUDO rm -r build-config
    fi

    if [ -d "package-puller/ubuntu-chroot" ]; then
        echo -e "\e[93mRemoving: package-puller/ubuntu-chroot\e[0m"
        $SUDO rm -r package-puller/ubuntu-chroot
    fi

    if [ -d "package-puller/package-repo-config" ]; then
        echo -e "\e[93mRemoving: package-puller/package-repo-config\e[0m"
        $SUDO rm -r package-puller/package-repo-config
    fi

    # Remove all packages-rocm-gfx* directories
    for gfx_dir in package-extractor/packages-rocm-gfx*; do
        if [ -d "$gfx_dir" ]; then
            echo -e "\e[93mRemoving: $gfx_dir\e[0m"
            $SUDO rm -r "$gfx_dir"
        fi
    done
fi

###### Build-related cleanup (extracted components, build outputs) ######
if [ $CLEAN_BUILD -eq 1 ]; then
    echo ""
    echo ">>> Cleaning build artifacts..."

    if [ -d "makeself-2.4.5" ]; then
        echo -e "\e[93mRemoving: makeself-2.4.5\e[0m"
        $SUDO rm -r makeself-2.4.5
    fi

    if [ -f "makeself-2.4.5.run" ]; then
        echo -e "\e[93mRemoving: makeself-2.4.5.run\e[0m"
        $SUDO rm makeself-2.4.5.run
    fi

    if [ -d "build-UI" ]; then
        echo -e "\e[93mRemoving: build-UI\e[0m"
        $SUDO rm -r build-UI
    fi

    if [ -d "build" ]; then
        echo -e "\e[93mRemoving: build\e[0m"
        $SUDO rm -r build
    fi

    if [ -d "package-extractor/logs" ]; then
        echo -e "\e[93mRemoving: package-extractor/logs\e[0m"
        $SUDO rm -r package-extractor/logs
    fi
    
    if [ -f "rocm-installer/epel-release-latest-8.noarch.rpm" ]; then
        echo -e "\e[93mRemoving: rocm-installer/epel-release-latest-8.noarch.rpm\e[0m"
        $SUDO rm rocm-installer/epel-release-latest-8.noarch.rpm*
    fi

    if [ -f "rocm-installer/epel-release-latest-9.noarch.rpm" ]; then
        echo -e "\e[93mRemoving: rocm-installer/epel-release-latest-9.noarch.rpm\e[0m"
        $SUDO rm rocm-installer/epel-release-latest-9.noarch.rpm*
    fi

    if [ -f "rocm-installer/epel-release-latest-10.noarch.rpm" ]; then
        echo -e "\e[93mRemoving: rocm-installer/epel-release-latest-10.noarch.rpm\e[0m"
        $SUDO rm rocm-installer/epel-release-latest-10.noarch.rpm*
    fi
    
    if [ -d "rocm-installer/component-rocm" ]; then
        echo -e "\e[93mRemoving: rocm-installer/component-rocm\e[0m"
        $SUDO rm -r rocm-installer/component-rocm
    fi

    if [ -d "rocm-installer/component-rocm-deb" ]; then
        echo -e "\e[93mRemoving: rocm-installer/component-rocm-deb\e[0m"
        $SUDO rm -r rocm-installer/component-rocm-deb
    fi

    if [ -d "rocm-installer/component-amdgpu" ]; then
        echo -e "\e[93mRemoving: rocm-installer/component-amdgpu\e[0m"
        $SUDO rm -r rocm-installer/component-amdgpu
    fi

    # Remove all component-amdgpu-* directories (distro-specific)
    for amdgpu_dir in rocm-installer/component-amdgpu-*; do
        if [ -d "$amdgpu_dir" ]; then
            echo -e "\e[93mRemoving: $amdgpu_dir\e[0m"
            $SUDO rm -r "$amdgpu_dir"
        fi
    done

    # Remove all component-rocm-gfx* directories
    for gfx_dir in rocm-installer/component-rocm-gfx*; do
        if [ -d "$gfx_dir" ]; then
            echo -e "\e[93mRemoving: $gfx_dir\e[0m"
            $SUDO rm -r "$gfx_dir"
        fi
    done

    if [ -d "rocm-installer/component_deps" ]; then
        echo -e "\e[93mRemoving: rocm-installer/component_deps\e[0m"
        $SUDO rm -r rocm-installer/component_deps
    fi

    if [ -d "rocm-installer/logs" ]; then
        echo -e "\e[93mRemoving: rocm-installer/logs\e[0m"
        $SUDO rm -r rocm-installer/logs
    fi

    if [ -d "logs" ]; then
        echo -e "\e[93mRemoving: logs\e[0m"
        $SUDO rm -r logs
    fi

    if [ -f "rocm-installer/VERSION" ]; then
        echo -e "\e[93mRemoving: rocm-installer/VERSION\e[0m"
        $SUDO rm rocm-installer/VERSION
    fi

    if [ -f "rocm-installer/rocm_ui" ]; then
        echo -e "\e[93mRemoving: rocm-installer/rocm_ui\e[0m"
        $SUDO rm rocm-installer/rocm_ui
    fi

    if [ -f "rocm-installer/deps_list.txt" ]; then
        echo -e "\e[93mRemoving: rocm-installer/deps_list.txt\e[0m"
        $SUDO rm rocm-installer/deps_list.txt
    fi

    # Test-related artifacts (created during testing/build)
    if [ -d "rocm-installer/test-logs" ]; then
        echo -e "\e[93mRemoving: rocm-installer/test-logs\e[0m"
        $SUDO rm -r rocm-installer/test-logs
    fi

    if [ -d "rocm-installer/rocm" ]; then
        echo -e "\e[93mRemoving: rocm-installer/rocm\e[0m"
        $SUDO rm -r rocm-installer/rocm
    fi

    if [ -d "rocm-installer/build" ]; then
        echo -e "\e[93mRemoving: rocm-installer/build\e[0m"
        $SUDO rm -r rocm-installer/build
    fi

    if [ -d "rocm-installer/dist" ]; then
        echo -e "\e[93mRemoving: rocm-installer/dist\e[0m"
        $SUDO rm -r rocm-installer/dist
    fi

    if [ -d "rocm-installer/amdsmi.egg-info" ]; then
        echo -e "\e[93mRemoving: rocm-installer/amdsmi.egg-info\e[0m"
        $SUDO rm -r rocm-installer/amdsmi.egg-info
    fi

    if [ -d "test/rocm-examples" ]; then
        echo -e "\e[93mRemoving: test/rocm-examples\e[0m"
        $SUDO rm -r test/rocm-examples
    fi

    if [ -d "test/shaderc" ]; then
        echo -e "\e[93mRemoving: test/shaderc\e[0m"
        $SUDO rm -r test/shaderc
    fi

    if [ -d "test/glslang" ]; then
        echo -e "\e[93mRemoving: test/glslang\e[0m"
        $SUDO rm -r test/glslang
    fi

    if [ -d "test/rocdecode-test" ]; then
        echo -e "\e[93mRemoving: test/rocdecode-test\e[0m"
        $SUDO rm -r test/rocdecode-test
    fi

    if [ -d "test/build" ]; then
        echo -e "\e[93mRemoving: test/build\e[0m"
        $SUDO rm -r test/build
    fi

    if [ -d "test/CMakeFiles" ]; then
        echo -e "\e[93mRemoving: test/CMakeFiles\e[0m"
        $SUDO rm -r test/CMakeFiles
    fi
fi

popd

echo ""
echo =======================
if [ $CLEAN_SETUP -eq 1 ] && [ $CLEAN_BUILD -eq 1 ]; then
    echo "Cleaning up...Complete"
elif [ $CLEAN_SETUP -eq 1 ]; then
    echo "Setup cleanup...Complete"
elif [ $CLEAN_BUILD -eq 1 ]; then
    echo "Build cleanup...Complete"
fi
echo =======================

