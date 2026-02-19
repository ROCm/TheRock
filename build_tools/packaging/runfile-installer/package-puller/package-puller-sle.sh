#!/bin/bash

# #############################################################################
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

# Inputs
ROCM_REPO=
AMDGPU_REPO=
GRAPHICS_REPO=

# Packaging repos
PACKAGE_REPO=$PWD/packages
SETUP_PATH=$PWD/setup

# Logs
PULL_LOGS_DIR="$PWD/logs"
PULL_CURRENT_LOG="$PULL_LOGS_DIR/pull_$(date +%s).log"

# Config
PACKAGES="rocm"
VERBOSE=0

REPO_LIST=(rocm-build.repo graphics-build.repo rocm.repo amdgpu-build.repo amdgpu.repo)

PROMPT_USER=0
DUMP_AMD_PKGS=0
DUMP_NON_AMD_PKGS=0

###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [Options] [Pull_Config] [Packages] [Output]

[Options}:
    help   = Display this help information.
    
    prompt  = Run the package puller with user prompts.
    amd     = Copy amd-specific packages out.
    other   = Copy non-amd packages out.
    verbose = Run the package puller with verbose logging.
    
[Pull_Config]:   
    config=<file_path> = <file_path> Path to a .config file with create settings in the format of create-default.config.

[Packages]:
    pkg=<package list> = <package-list> List of Package/Packages to pull
    
[Output]:
    out=<file_path>    = <file_path> Path to output directory for pulled packages

Example (pull by config):
-------------------------

    ./package_puller.sh config="config/rocm-6.2-22.04.config" out="/home/amd/package-extractor/packages" prompt amd
       
END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

        case "$ID" in
        sles)
            echo "Pulling packages for SLES $DISTRO_VER."
            ;;
        *)
            echo "$ID is not a supported OS"
            exit 1
            ;;
        esac
    else
        echo "Unsupported OS"
        exit 1
    fi
}

print_no_err() {
    local msg=$1
    echo -e "\e[32m++++++++++++++++++++++++++++++++++++\e[0m"
    echo -e "\e[32m$msg\e[0m"
    echo -e "\e[32m++++++++++++++++++++++++++++++++++++\e[0m"
}

print_err() {
    local msg=$1
    echo -e "\e[31m++++++++++++++++++++++++++++++++++++\e[0m"
    echo -e "\e[31m$msg\e[0m"
    echo -e "\e[31m++++++++++++++++++++++++++++++++++++\e[0m"
}

print_str() {
    local msg=$1
    local clr=$2
    local color_code=""

    if [[ $clr == 1 ]]; then
        color_code="\e[93m"  # yellow
    elif [[ $clr == 2 ]]; then
        color_code="\e[32m"  # green
    elif [[ $clr == 3 ]]; then
        color_code="\e[31m"  # red
    elif [[ $clr == 4 ]]; then
        color_code="\e[35m"  # purple
    elif [[ $clr == 5 ]]; then
        color_code="\e[36m"  # cyan
    fi

    echo -e "${color_code}++++++++++++++++++++++++++++++++++++\e[0m"
    echo -e "${color_code}$msg\e[0m"
    echo -e "${color_code}++++++++++++++++++++++++++++++++++++\e[0m"
}

prompt_user() {
    if [[ $PROMPT_USER == 1 ]]; then
        read -p "$1" option
    else
        option=y
    fi
}

copy_rpms() {
    echo Copying from $1 to $2
    local src_dir=$1
    local dest_dir=$2
    find $src_dir -name "*.rpm" -exec cp {} $dest_dir \;
}

install_prereqs() {
    echo ++++++++++++++++++++++++++++++++
    echo Installing prereqs...
    
    echo Updating zypper...
    $SUDO zypper install -y zypper
    
    echo Installing prereqs...Complete.
}

cleanup() {
    echo ++++++++++++++++++++++++++++++++
    echo Cleaning up...
    
    # Remove any .repo files
    for index in ${REPO_LIST[@]}; do
        if [ -f /etc/zypp/repos.d/$index ]; then
            echo -e "\e[93m=-=-=-= Removing $index =-=-=-=\e[0m"
            $SUDO rm /etc/zypp/repos.d/$index
        fi
    done
   
    # cleanup zypper cache
    $SUDO zypper clean
    $SUDO zypper refresh > /dev/null 2>&1
    
    echo Cleaning up...Complete.
}

config_create() {
    echo ++++++++++++++++++++++++++++++++
    echo Create Configure...
    
    CREATE_CONFIG_FILE=
    local CREATE_CONFIG_FILE_INPUT=$1
    
    # Check for user-modified config file (input .config to create script)
    if [[ ${CREATE_CONFIG_FILE_INPUT##*.} == "config" ]]; then
         CREATE_CONFIG_FILE=$CREATE_CONFIG_FILE_INPUT
         echo Using Create Configuration file: $CREATE_CONFIG_FILE
         
         if [[ ! -f $CREATE_CONFIG_FILE ]]; then
             echo $CREATE_CONFIG_FILE not found.
             exit 1
         fi
         
         source $CREATE_CONFIG_FILE
    else
        print_err "Fail.  No config file."
        exit 1
    fi
    
    echo Create Configure...Complete.
}

setup_rocm_repo() {
    if [[ -z $ROCM_REPO ]]; then
        return
    fi

    echo ++++++++++++++++++++++++++++++++
    echo Setting up ROCm repo...

    echo "$ROCM_REPO" | $SUDO tee /etc/zypp/repos.d/rocm-build.repo

    # cleanup the zypper cache
    $SUDO zypper clean
    $SUDO zypper --gpg-auto-import-keys refresh

    echo Setting up ROCm repo...Complete.
}

setup_amdgpu_repo() {
    if [[ -z $AMDGPU_REPO ]]; then
        return
    fi

    echo ++++++++++++++++++++++++++++++++
    echo Setting up amdgpu repo...

    echo "$AMDGPU_REPO" | $SUDO tee -a /etc/zypp/repos.d/amdgpu-build.repo

    # cleanup the zypper cache
    $SUDO zypper clean
    $SUDO zypper --gpg-auto-import-keys refresh

    echo Setting up amdgpu repo...Complete.
}

setup_graphics_repo() {
    if [[ -z $GRAPHICS_REPO ]]; then
        return
    fi

    echo ++++++++++++++++++++++++++++++++
    echo Setting up graphics repo...

    echo "$GRAPHICS_REPO" | $SUDO tee -a /etc/zypp/repos.d/graphics-build.repo

    # cleanup the zypper cache
    $SUDO zypper clean
    $SUDO zypper --gpg-auto-import-keys refresh

    echo Setting up graphics repo...Complete.
}

download_packages() {
    echo ++++++++++++++++++++++++++++++++
    echo Downloading and setting up Packaging...

    # create the package directory repo
    echo Creating packages directory: $PACKAGE_REPO
    mkdir $PACKAGE_REPO
       
    $SUDO zypper clean
    $SUDO zypper refresh > /dev/null
    
    echo =-=-=-= download packages =-=-=-=
    prompt_user "Start Download : (y/n): "
    if [[ $option == "Y" || $option == "y" ]]; then
    
        # Download the package dependencies to the dep directory
        pushd $PACKAGE_REPO
            
            $SUDO zypper install -y --download-only $PACKAGES
            ret=$?
            
        popd
        
        # check for any download errors
        if [[ $ret -ne 0 ]]; then
            print_err "Failed packages download."
            cleanup
            exit 1
        else
            print_no_err "Packages download successful."
        fi
        
    else
        cleanup
        
        echo "Exiting."
        exit 1
    fi
    
    # Copy the packages from the zypp cache to the package directory
    copy_rpms "/var/cache/zypp/packages" "$PACKAGE_REPO"
    
    # simulate/dryrun the install
    $SUDO zypper install -y --dry-run $PACKAGES
    if [ $? -ne 0 ]; then
        echo -e "\e[31m++++++++++++++++++++++++++++++++++++++++\e[0m"
        echo -e "\e[31mError occurred.  Repo validation failed.\e[0m"
        echo -e "\e[31m++++++++++++++++++++++++++++++++++++++++\e[0m"
        
        cleanup
        
        exit 1
    else
        echo -e "\e[32m+++++++++++++++++++++++++++++++++++++++\e[0m"
        echo -e "\e[32m No error.  Valid package dependencies.\e[0m"
        echo -e "\e[32m+++++++++++++++++++++++++++++++++++++++\e[0m"
    fi
    
    echo Downloading and setting up Packaging...Complete.
}

check_package_owner() {
    AMDPKG=0
    AMDGPUPKG=0
    
    local pkgName=$1
    local package=$(rpm -q --queryformat "%{NAME}" --nosignature $pkg)
    local vendor=$(rpm -qi --nosignature $pkg | grep Vendor)
    local epoch=$(rpm -q --queryformat "%{EPOCH}" --nosignature $pkg)
    
    if [[ $VERBOSE == 1 ]]; then
        rpm -qi --nosignature $pkgName
    fi
    
    if [[ $package =~ "amdgpu" || $package =~ "rocm" ]]; then
        AMDPKG=1
    else
       if [[ -n $vendor ]]; then
           if [[ $vendor =~ "Advanced Micro Devices" || $vendor =~ "AMD ROCm" ]]; then
               AMDPKG=1
           fi
       fi
    fi
    
    # for amd or amdgpu-specific packages copy to separate directories  
    if [[ $AMDPKG == 1 ]] ; then
        AMD_COUNT=$((AMD_COUNT+1))
        ROCM_PACKAGES+="$(basename $pkgName) "
        
        if [[ $DUMP_AMD_PKGS == 1 ]]; then
            if [[ ! -d $PWD/packages-amd ]]; then
                echo Creating Extraction amd directory.
                mkdir -p $PWD/packages-amd
            fi
        
            cp $pkgName $PWD/packages-amd
        fi
        
        if [[ $epoch != "(none)" ]]; then
            AMDGPUPKG=1
            AMDGPU_COUNT=$((AMDGPU_COUNT+1))
            if [[ ! -d $PWD/packages-amdgpu ]]; then
                echo Creating Extraction amdgpu directory.
                mkdir -p $PWD/packages-amdgpu
            fi
            
            cp $pkgName $PWD/packages-amdgpu
            echo -e "\e[94m++++++++++++++++++++++++++++++++++++\e[0m"
            echo -e "\e[94m$AMDGPU_COUNT: AMDGPU PACKAGE\e[0m"
            echo -e "\e[94m++++++++++++++++++++++++++++++++++++\e[0m"
        fi
        
        print_no_err "$AMD_COUNT: AMD PACKAGE"
    else
        NON_AMD_COUNT=$((NON_AMD_COUNT+1))
        OTHER_PACKAGES+="$(basename $pkgName) "
        
        if [[ $DUMP_NON_AMD_PKGS == 1 ]]; then
            if [[ ! -d $PWD/packages-other ]]; then
                echo Creating Extraction non-amd directory.
                mkdir -p $PWD/packages-other
            fi
        
            cp $pkgName $PWD/packages-other
        fi
        
        print_str "$NON_AMD_COUNT: 3rd Party PACKAGE" 4
    fi
}

dump_packages_info() {
    PKG_COUNT=0
    AMD_COUNT=0
    AMDGPU_COUNT=0
    NON_AMD_COUNT=0
    
    PACKAGES=
    ROCM_PACKAGES=
    OTHER_PACKAGES=
    
    for pkg in $PACKAGE_REPO/*; do
        if [[ $pkg == *.rpm ]]; then
            echo pkg = $pkg
            PACKAGES+="$pkg "
        fi
    done
    
    pushd $PACKAGE_REPO
        for pkg in $PACKAGES; do
            PKG_COUNT=$((PKG_COUNT+1))
        
            echo ----------------------------------------------------------------------
            echo -e "\e[93mpkg $PKG_COUNT = $(basename $pkg)\e[0m"
            check_package_owner $pkg
       done
   popd
   
   echo -----------------------------
   echo "Package Total         = $PKG_COUNT"
   echo "Package AMD           = $AMD_COUNT"
   echo "Package AMDGPU        = $AMDGPU_COUNT"
   echo "Package 3RD           = $NON_AMD_COUNT"
}


####### Main script ###############################################################

# Create the pull log directory
if [ ! -d $PULL_LOGS_DIR ]; then
    mkdir -p $PULL_LOGS_DIR
fi

exec > >(tee -a "$PULL_CURRENT_LOG") 2>&1

echo ====================
echo PACKAGE PULLER - SLE
echo ====================

PROG=${0##*/}

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: $SUDO

os_release

# parse args
while (($#))
do
    case "$1" in
    help)
        usage
        exit 0
        ;;
    prompt)
        echo "Enabling user prompts."
        PROMPT_USER=1
        shift
        ;;
    amd)
        echo "Enabling amd-only package output."
        DUMP_AMD_PKGS=1
        shift
        ;;
    other)
        echo "Enabling other (non-amd) package output."
        DUMP_NON_AMD_PKGS=1
        shift
        ;;
    config=*)
        CONFIG_FILE="${1#*=}"
        echo "Using Configuration file: $CONFIG_FILE"
        shift
        ;;
    out=*)
        PACKAGE_REPO="${1#*=}"
        echo "Using package output location: $PACKAGE_REPO"
        shift
        ;;
    pkg=*)
        PACKAGES="${1#*=}"
        echo "Downloading packages: $PACKAGES"
        shift
        ;;
    verbose)
        echo "Enabling verbose logging."
        VERBOSE=1
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Configure the creator
config_create $CONFIG_FILE

echo --------------------------------------------------
echo "PACKAGE_REPO  = $PACKAGE_REPO"
echo "PACKAGES      = $PACKAGES"
echo --------------------------------------------------
echo "DOWNLOAD_MODE = $DOWNLOAD_MODE"
echo -----------------------------------------
echo "ROCM_REPO     = $ROCM_REPO"
echo -----------------------------------------
echo "AMDGPU_REPO   = $AMDGPU_REPO"
echo -----------------------------------------
echo "GRAPHICS_REPO = $GRAPHICS_REPO"
echo -----------------------------------------
echo PACKAGES       = $PACKAGES
echo --------------------------------------------------

prompt_user "Pull Packages from repos (y/n): "
if [[ $option == "N" || $option == "n" ]]; then
    echo Exiting.
    exit 1
fi

cleanup

install_prereqs

if [ -d $PACKAGE_REPO ]; then
    echo -e "\e[93mPackage directory exists.  Removing: $PACKAGE_REPO\e[0m"
    $SUDO rm -r $PACKAGE_REPO
fi

setup_rocm_repo
setup_amdgpu_repo
setup_graphics_repo

download_packages

dump_packages_info

echo -e "\e[32m==============================\e[0m"
echo -e "\e[32mPACKAGES DOWNLOADED!\e[0m"
echo -e "\e[32m==============================\e[0m"

prompt_user "Cleanup (y/n): "
if [[ $option == "Y" || $option == "y" ]]; then
    cleanup
fi

if [[ -n $PULL_CURRENT_LOG ]]; then
    echo -e "\e[32mExtract log stored in: $PULL_CURRENT_LOG\e[0m"
fi

