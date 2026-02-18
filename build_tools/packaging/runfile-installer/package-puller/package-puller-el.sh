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

EPEL_SETUP=1

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
        DISTRO_MAJOR_VER=${DISTRO_VER%.*}
        
        case "$ID" in
        rhel|ol|rocky|almalinux)
            echo "Pulling packages for EL $DISTRO_VER."
            ;;
        amzn)
            echo "Pulling packages for Amazon $DISTRO_VER."
            echo "Disable EPEL/CRB for Amazon."
            EPEL_SETUP=0
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

prompt_user() {
    if [[ $PROMPT_USER == 1 ]]; then
        read -p "$1" option
    else
        option=y
    fi
}

update_dnf_conf() {
    $SUDO cp /etc/dnf/dnf.conf /etc/dnf/dnf.conf.bak

    # check if the dnf.conf file has fastestmirror=true
    grep -q "fastestmirror=true" /etc/dnf/dnf.conf
    if [ $? -ne 0 ]; then
        echo Enabling fastest mirror.
        echo "fastestmirror=true" | $SUDO tee -a /etc/dnf/dnf.conf
    fi
}

restore_dnf_conf() {
    # restore the dnf.conf file from the backup file

    if [ -f /etc/yum.repos.d/epel.repo ]; then
        echo Restoring dnf.conf
        if [ -f /etc/dnf/dnf.conf.bak ]; then
            $SUDO cp /etc/dnf/dnf.conf.bak /etc/dnf/dnf.conf
            $SUDO rm /etc/dnf/dnf.conf.bak
        fi
    fi
}

setup_epel_crb() {
    # Setup EPEL/crb
    local epel_pkg="epel-release-latest-$DISTRO_MAJOR_VER.noarch.rpm"
    local codeready_repo="codeready-builder-for-rhel-$DISTRO_MAJOR_VER-x86_64-rpms"
    
    if [ -f /etc/yum.repos.d/epel.repo ]; then
        echo "EPEL repo exists."
    else
        # install wget if required
        if ! rpm -qa | grep -q "wget"; then
            echo "Package: wget missing. Installing..."
            $SUDO dnf install -y wget > /dev/null 2>&1
            echo "Package: wget installed."
        fi
    
        echo "EPEL repo setup for EL $DISTRO_MAJOR_VER."
        
        wget --tries 5 https://dl.fedoraproject.org/pub/epel/$epel_pkg
        if [ $? -ne 0 ]; then
            print_err "Unsupported version for EPEL."
            exit 1
        fi
        
        $SUDO rpm -ivh "$epel_pkg"
    fi
    
    # Enable the codeready-builder repo (RHEL only)
    if [[ "$DISTRO_NAME" = "rhel" ]]; then
        if ! $SUDO dnf repolist all | grep -q "^$codeready_repo"; then
            print_err "$codeready_repo repo not configured."
            exit 1
        fi
        
        local repo_status=$(dnf repolist all | grep "^$codeready_repo" | awk '{print $NF}')
        if [[ "$repo_status" == "disabled" ]]; then
            echo "Enabling $codeready_repo."
            $SUDO dnf config-manager --enable "$codeready_repo"
        fi
    else
        $SUDO crb enable
    fi
}

install_prereqs() {
    echo ++++++++++++++++++++++++++++++++
    echo Installing prereqs...
    
    $SUDO dnf install -y dnf-plugin-config-manager
    
    if [[ $EPEL_SETUP == 1 ]]; then
        setup_epel_crb
    fi
    
    # Update the dnf.conf for faster mirrors etc.
    update_dnf_conf
    
    echo Installing prereqs...Complete.
}

cleanup() {
    echo ++++++++++++++++++++++++++++++++
    echo Cleaning up...
    
    # Remove any .repo files
    for index in ${REPO_LIST[@]}; do
        if [ -f /etc/yum.repos.d/$index ]; then
            echo -e "\e[93m=-=-=-= Removing $index =-=-=-=\e[0m"
            $SUDO rm /etc/yum.repos.d/$index
        fi
    done
    
    # cleanup dnf cache
    $SUDO dnf clean all
    $SUDO rm -r /var/cache/dnf/*
    
    restore_dnf_conf
    
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

setup_repo_priorities() {
    echo Setting up repo priorities...
    
    # Set EPEL to exclude ROCm and HIP packages and lower priority
    if [ -f /etc/yum.repos.d/epel.repo ]; then
        if ! grep -q "excludepkgs=rocm" /etc/yum.repos.d/epel.repo; then
            echo "Excluding ROCm and HIP packages from EPEL..."
            $SUDO sed -i '/\[epel\]/a excludepkgs=rocm* hip*' /etc/yum.repos.d/epel.repo
        fi
        
        if ! grep -q "priority=" /etc/yum.repos.d/epel.repo; then
            echo "Setting EPEL priority to 10..."
            $SUDO sed -i '/excludepkgs=rocm*/a priority=10' /etc/yum.repos.d/epel.repo
        fi
    fi
    
    # Set ROCm repo to higher priority and include ROCm and HIP packages
    if [ -f /etc/yum.repos.d/rocm-build.repo ]; then
        if ! grep -q "includepkgs=rocm" /etc/yum.repos.d/rocm-build.repo; then
            echo "Including ROCm and HIP packages in rocm repo..."
            $SUDO sed -i '/priority=1/a includepkgs=rocm* hip*' /etc/yum.repos.d/rocm-build.repo
        fi
    fi
    
    # Clear cache to apply changes
    $SUDO dnf clean all
    $SUDO dnf makecache > /dev/null 2>&1
    
    echo Setting up repo priorities...Complete.
}

setup_rocm_repo() {
    if [[ -z $ROCM_REPO ]]; then
        return
    fi

    echo ++++++++++++++++++++++++++++++++
    echo Setting up ROCm repo...

    # Disable rocm repo for OL 8.10 for amdgpu-dkms package pull
    if [[ "$PACKAGES" =~ "amdgpu-dkms" && "$DISTRO_NAME" == "ol" && "$DISTRO_VER" == "8.10" ]]; then
        echo "amdgpu-dkms packaged pull for OL 8.10.  Disabling rocm repo setup."
        return
    fi

    echo "$ROCM_REPO" | $SUDO tee -a /etc/yum.repos.d/rocm-build.repo

    # Setup repo priorities if required
    if [[ $DISTRO_MAJOR_VER -ge 10 ]]; then
        setup_repo_priorities
    fi

     # cleanup the dnf cache
    $SUDO dnf clean all
    $SUDO rm -rf /var/cache/dnf/*

    echo Setting up ROCm repo...Complete.
}

setup_amdgpu_repo() {
    if [[ -z $AMDGPU_REPO ]]; then
        return
    fi

    echo ++++++++++++++++++++++++++++++++
    echo Setting up amdgpu repo...

    echo "$AMDGPU_REPO" | $SUDO tee -a /etc/yum.repos.d/amdgpu-build.repo

    # cleanup the dnf cache
    $SUDO dnf clean all
    $SUDO rm -rf /var/cache/dnf/*

    echo Setting up amdgpu repo...Complete.
}

setup_graphics_repo() {
    if [[ -z $GRAPHICS_REPO ]]; then
        return
    fi

    echo ++++++++++++++++++++++++++++++++
    echo Setting up graphics repo...

    echo "$GRAPHICS_REPO" | $SUDO tee -a /etc/yum.repos.d/graphics-build.repo

    # cleanup the dnf cache
    $SUDO dnf clean all
    $SUDO rm -rf /var/cache/dnf/*

    echo Setting up graphics repo...Complete.
}

download_packages() {
    echo ++++++++++++++++++++++++++++++++
    echo Downloading and setting up Packaging...

    # create the package directory repo
    echo Creating packages directory: $PACKAGE_REPO
    mkdir $PACKAGE_REPO
       
    $SUDO dnf clean all
    $SUDO rm -rf /var/cache/dnf/*
    
    echo =-=-=-= download packages =-=-=-=
    prompt_user "Start Download : (y/n): "
    if [[ $option == "Y" || $option == "y" ]]; then
    
        # Download the package dependencies to the dep directory
        pushd $PACKAGE_REPO
            
            $SUDO dnf download --resolve --downloaddir="./" $PACKAGES
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
    
    # simulate/dryrun the install
    errorCheck=$($SUDO dnf --nogpg --assumeno install $PACKAGES)
    if  [[ $errorCheck == *"Error"* ]] || [[ $errorCheck == *"uninstallable"* ]]; then
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
        
        print_err "$NON_AMD_COUNT: 3rd Party PACKAGE"
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

echo ===================
echo PACKAGE PULLER - EL
echo ===================

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

