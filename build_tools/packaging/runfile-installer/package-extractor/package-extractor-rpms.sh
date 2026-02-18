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

# Extraction Output - supports multi-format with EXTRACT_FORMAT variable
EXTRACT_FORMAT="${EXTRACT_FORMAT:-rpm}"

# ROCm Packages Source - defaults to format-specific directory
PACKAGE_ROCM_DIR="${PACKAGE_ROCM_DIR:-$PWD/packages-rocm-${EXTRACT_FORMAT}}"

# AMDGPU Packages Source - defaults to format-specific directory
PACKAGE_AMDGPU_DIR="${PACKAGE_AMDGPU_DIR:-$PWD/packages-amdgpu-${EXTRACT_FORMAT}}"

# Extraction output directories
EXTRACT_ROCM_DIR="$PWD/component-rocm-${EXTRACT_FORMAT}"
EXTRACT_AMDGPU_DIR="$PWD/component-amdgpu-${EXTRACT_FORMAT}"

# Extra install setup
EXTRACT_TAR_DIR="$PWD/setup-install"

# Extraction Files
EXTRACT_ROCM_PKG_CONFIG_FILE="rocm-packages.config"
EXTRACT_AMDGPU_PKG_CONFIG_FILE="amdgpu-packages.config"

EXTRACT_AMDGPU_DKMS_VER_FILE="amdgpu-dkms-ver.txt"

EXTRACT_COMPO_LIST_FILE="components.txt"          # list the component version of extracted packages
EXTRACT_PACKAGE_LIST_FILE="packages.txt"          # list all extracted packages
EXTRACT_REQUIRED_DEPS_FILE="required_deps.txt"    # list only required dependencies (non-amd deps)
EXTRACT_GLOBAL_DEPS_FILE="global_deps.txt"        # list all extracted dependencies

# Extra/Installer dependencies
EXTRA_DEPS=()
INSTALLER_DEPS=(rsync wget)

# Logs
EXTRACT_LOGS_DIR="$PWD/logs"
EXTRACT_CURRENT_LOG="$EXTRACT_LOGS_DIR/extract_$(date +%s).log"

# Config
PROMPT_USER=0
ROCM_EXTRACT=0
AMDGPU_EXTRACT=0
EXTRACT_CONTENT=1
CONTENT_LIST=0

######## Build tags EXTRACT FROM ROCM meta package
ROCM_VER=

# Stats
PACKAGES=
AMD_PACKAGES=
OTHER_PACKAGES=

SCRIPLET_PREINST_COUNT=0
SCRIPLET_POSTINST_COUNT=0
SCRIPLET_PRERM_COUNT=0
SCRIPLET_POSTRM_COUNT=0
SCRIPTLET_OPT_COUNT=0
SCRIPTLET_OPT=

GLOBAL_DEPS=


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options}:
    help                    = Display this help information.
    prompt                  = Run the extractor with user prompts.
    amdgpu                  = Extract AMDGPU packages.
    rocm                    = Extract ROCm packages.

    nocontent               = Disables content extraction (deps, scriptlets will be extracted only).
    contentlist             = Lists all files extracted to content directories during extraction.

    pkgs-rocm=<file_path>   = <file_path> Path to ROCm source packages directory for extract.
    pkgs-amdgpu=<file_path> = <file_path> Path to AMDGPU source packages directory for extract.
    ext-rocm=<file_path>    = <file_path> Path to ROCm packages extraction directory.
    ext-amdgpu=<file_path>  = <file_path> Path to AMDGPU packages extraction directory.

    Example:

    ./package-extractor-rpms.sh prompt rocm ext-rocm="/extracted-rocm"

END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

        case "$ID" in
        rhel|ol|rocky|almalinux)
            echo "Extracting for EL $DISTRO_VER."
            EXTRACT_DISTRO_TYPE=el
            ;;
        sles)
            echo "Extracting for SUSE $DISTRO_VER."
            EXTRACT_DISTRO_TYPE=sle
            ;;
        amzn)
            echo "Extracting for Amazon $DISTRO_VER."
            EXTRACT_DISTRO_TYPE=el
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

install_tools() {
    echo ++++++++++++++++++++++++++++++++
    echo Installing tools...
   
    # Install rpmdevtools for dep version
    if [ $EXTRACT_DISTRO_TYPE == "el" ]; then
        if [[ "$DISTRO_NAME" = "rocky" ]]; then
            $SUDO dnf install -y cpio diffutils
        else
            $SUDO dnf install -y rpmdevtools
        fi
    elif [ $EXTRACT_DISTRO_TYPE == "sle" ]; then
        $SUDO zypper install -y rpmdevtools
    else
        echo Unsupported extract type.
        exit 1
    fi
    
    echo Installing tools...Complete.
}

dump_extract_stats() {
    echo +++++++++++++++++++++++++++++++++++++++++++++
    echo STATS
    echo -----
    
    local stat_dir=$1

    echo $stat_dir:
    echo ----------------------------
    echo "size:" 
    echo "-----"
    echo "$(du -sh $stat_dir | awk '{print $1}')"
    echo "$(du -sb $stat_dir | awk '{print $1}')" bytes
    echo "------"
    echo "types:"
    echo "------"
    echo "files = $(find $stat_dir -type f | wc -l)"
    echo "dirs  = $(find $stat_dir -type d | wc -l)"
    echo "links = $(find $stat_dir -type l | wc -l)"
    echo "        ------"
    echo "        $(find $stat_dir | wc -l)"
    echo ----------------------------
}

init_stats() {
    echo Initialize package information.
    
    PACKAGES=
    
    AMD_PACKAGES=
    OTHER_PACKAGES=
    
    GLOBAL_DEPS=
    
    SCRIPLET_PREINST_COUNT=0
    SCRIPLET_POSTINST_COUNT=0
    SCRIPLET_PRERM_COUNT=0
    SCRIPLET_POSTRM_COUNT=0
    SCRIPTLET_OPT_COUNT=0
    SCRIPTLET_OPT=
}

scriptlet_stats() {
    echo +++++++++++++++++++++++++++++++++++++++++++++
    echo Extracted Scriptlets:
    echo ---------------------
    echo "SCRIPLET_PREINST_COUNT  = $SCRIPLET_PREINST_COUNT"
    echo "SCRIPLET_POSTINST_COUNT = $SCRIPLET_POSTINST_COUNT"
    echo "SCRIPLET_PRERM_COUNT    = $SCRIPLET_PRERM_COUNT"
    echo "SCRIPLET_POSTRM_COUNT   = $SCRIPLET_POSTRM_COUNT"
    echo "SCRIPTLET_OPT_COUNT     = $SCRIPTLET_OPT_COUNT"
    echo ----------------------
    echo "Scriptlets (/opt/rocm):"  
    echo ----------------------
    echo $SCRIPTLET_OPT | tr ' ' '\n' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u
}

write_out_list() {
    local list=$1
    local file=$2
    
    echo "$list" | tr ' ' '\n' > "$file"
}

write_version() {
    echo -------------------------------------------------------------
    echo Writing version...
    
    i=0
    VERSION_FILE="../VERSION"
    
    while IFS= read -r line; do
        case $i in
            0) INSTALLER_VERSION="$line" ;;
        esac
        
        i=$((i+1))
    done < "$VERSION_FILE"
     
    if [[ -n $ROCM_VER ]]; then
        echo "INSTALLER_VERSION = $INSTALLER_VERSION"
        echo "ROCM_VER          = $ROCM_VER"
    
        # Update the version file
        echo "$INSTALLER_VERSION" > "$VERSION_FILE"
        echo "$ROCM_VER" >> "$VERSION_FILE"
    fi
}

move_opt_contents() {
    local content_dir="$1"
    local dir="$2"

    # Move all contents of the 'opt' directory to the root content directory
    mv "$dir/"* "$content_dir/"

    # Remove the empty 'opt' directory
    rmdir "$dir"
    echo "Moved contents of '$dir' to '$content_dir'."
}

move_etc_contents_rocm() {
    local content_etc_dir="$PACKAGE_DIR/content-etc"
             
    echo Creating content-etc directory: "$content_etc_dir"
    mkdir "$content_etc_dir"
     
    # Move all contents of the 'etc' directory to the content-etc directory
    mv "$dir/"* "$content_etc_dir/"
    
    # Remove the empty 'etc' directory
    rmdir "$dir"
    
    echo "Moved contents of '$dir' to '$content_etc_dir'."
}

move_usr_contents_rocm() {
    local dir="$1"

    # workaround for extra /usr content for RHEL
    if [[ -d "$dir/lib/.build-id" ]]; then
        echo -e "\e[31m$dir/lib/.build-id delete\e[0m"
        $SUDO rm -r "$dir/lib/.build-id"
        rmdir "$dir/lib"
        rmdir "$dir"
    fi
}

move_data() {
    echo -e "\e[36mMoving data...\e[0m"
    
    local content_dir="$1"
    echo "Content root: $content_dir"

    # Loop through the content directory
    for dir in "$content_dir"/*; do
        local dirname=$(basename "$dir")
        
        # Check if the current directory is the opt / etc / or usr directories
        if [[ -d "$dir" && "$dirname" == "opt" ]]; then
            echo -e "\e[93m'opt' directory detected: $dir\e[0m"
            move_opt_contents "$content_dir" "$dir"
        
        elif [[ -d "$dir" && "$dirname" == "etc" ]]; then
            echo -e "\e[93m'etc' directory detected: $dir\e[0m"
            if [[ $content_dir =~ "component-rocm" ]]; then
                move_etc_contents_rocm 
            fi 
             
        elif [[ -d "$dir" && "$dirname" == "usr" ]]; then
            echo -e "\e[93m'usr' directory detected: $dir\e[0m"
            if [[ $content_dir =~ "component-rocm" ]]; then
                move_usr_contents_rocm "$dir"
            fi
              
        else
            echo -e "\e[93m$dir not moved.\e[0m"
        fi
    done
    
    echo Moving data...Complete.
}

extract_data() {
    echo --------------------------------
    echo Extracting all data/content
    echo --------------------------------
    
    local package_dir_content="$PACKAGE_DIR/content"
    
    echo Creating content directory: $package_dir_content
    mkdir $package_dir_content
    
    echo "Extracting Data..."
    
    # Extract the rpm package file content
    pushd $package_dir_content
    
        rpm2cpio "$PACKAGE" | cpio -idmv > /dev/null 2>&1
        if [[ $? -ne 0 ]]; then
            print_err "Failed rpm2cpio"
            exit 1
        fi
        
    popd

    # List extracted content files if requested
    if [ $CONTENT_LIST -eq 1 ]; then
        echo "Content files extracted:"
        find "$package_dir_content" -type f | sort
        echo "---"
    fi

    # Move data content to the correct directories for the installer
    move_data "$package_dir_content"

    echo Extracting Data...Complete.
    echo ---------------------------
}

extract_version() {
    local pkg="$1"

    # Extract the build ci/version info from "core" packages
    if echo "$pkg" | grep -q 'amdrocm-base'; then
        echo "--------------------------------"
        echo "Extract rocm versioning..."
    
        # Extract version from package filename
        local pkg_basename=$(basename "$pkg")
        local pattern='amdrocm-base([0-9]+\.[0-9]+)-'

        if [[ $pkg_basename =~ $pattern ]]; then
            ROCM_VER="${BASH_REMATCH[1]}"
        else
            VERSION_INFO=$(rpm -qi --nosignature $pkg | grep -E 'Version' | awk '{print $3}')
            echo VERSION_INFO = $VERSION_INFO
            ROCM_VER=$(echo "$VERSION_INFO" | cut -d '.' -f 1-2)
        fi
        
        echo "ROCM_VER = $ROCM_VER"
        
        write_version
    fi
}

extract_info() {
    echo --------------------------------
    echo Extracting package info
    echo --------------------------------
    
    rpm -qi --nosignature $PACKAGE
    
    VERSION_INFO=$(rpm -qi --nosignature $PACKAGE | grep -E 'Version' | awk '{print $3}')
    
    # Check for amdgpu-based packages pulled with rocm packages
    if echo "$PACKAGE_DIR_NAME" | grep -q 'amdgpu'; then
        # write out the package/component version
        echo "$PACKAGE_DIR_NAME" >> "$EXTRACT_DIR/$EXTRACT_AMDGPU_PKG_CONFIG_FILE"
    else
        echo "$PACKAGE_DIR_NAME" >> "$EXTRACT_DIR/$EXTRACT_ROCM_PKG_CONFIG_FILE"
        
        # write out the package/component version
        printf "%-25s = %s\n" "$PACKAGE_DIR_NAME" "$VERSION_INFO" >> "$EXTRACT_DIR/$EXTRACT_COMPO_LIST_FILE"
        printf "%-25s = %s\n" "$PACKAGE_DIR_NAME" "$VERSION_INFO"
    fi
    
    echo "VERSION_INFO = $VERSION_INFO"
    echo "PACKAGE      = $PACKAGE_DIR_NAME"
    
    extract_version "$PACKAGE"
}

extract_deps() {
    echo --------------------------------
    echo Extracting all dependencies
    echo --------------------------------
    
    local package_dir_deps="$PACKAGE_DIR/deps"
    
    echo "Extracting Dependencies...: $PACKAGE to $package_dir_deps"

    if [ ! -d $package_dir_deps ]; then
        echo Creating deps directory: $package_dir_deps
        mkdir -p $package_dir_deps
    fi

    echo --------------------------------
    rpm -qpRv --nosignature $PACKAGE
    echo --------------------------------

    # Extract manual dependencies
    DEPS=$(rpm -qpRv --nosignature $PACKAGE | grep -E 'manual' | sed 's/manual: /,/')

    # Also extract important automatic shared library dependencies
    # These are typically shown as "libname.so.X()(64bit)" and we need to map them to package names
    AUTO_DEPS=$(rpm -qpR --nosignature $PACKAGE | grep -E '^lib.*\.so\.' | sed 's/(.*)//' | sed 's/\.so\..*//')

    # Add atomic library dependency if found
    if echo "$AUTO_DEPS" | grep -q "libatomic"; then
        DEPS+=", libatomic"
    fi

    # Process the depends
    if [[ -n $DEPS ]]; then
        echo "-------------"
        echo "Dependencies:"
        echo "-------------"
        echo $DEPS | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u

        # write out the dependencies
        echo $DEPS | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u > "$package_dir_deps/deps.txt"

        GLOBAL_DEPS+="$DEPS "
        echo "-------------"
    fi

    echo Extracting Dependencies...Complete.
    echo -----------------------------------
}

extract_scriptlets() {
    echo --------------------------------
    echo Extracting all scriptlets
    echo --------------------------------
    
    local package_dir_scriptlet="$PACKAGE_DIR/scriptlets"
    
    echo "Extracting Scriptlets...: $PACKAGE to $package_dir_scriptlet"
    
    if [ ! -d $package_dir_scriptlet ]; then
        echo Creating scriptlet directory: $package_dir_scriptlet
        mkdir -p $package_dir_scriptlet
    fi
    
    local scriptlets=$(rpm -qp --scripts --nosignature "$PACKAGE")
    echo +++++++++++
    echo $scriptlets
    echo +++++++++++
   
    echo "$scriptlets" | awk -v output_dir="$package_dir_scriptlet" '
    /scriptlet \(using/ {
        if (section) {
            # Remove unwanted lines from the section
            section = gensub(/postinstall program:.*|preuninstall program:.*|postuninstall program:.*|posttrans program:.*/, "", "g", section)
            print section > (output_dir "/" section_name ".sh")
            section = ""
        }
        section_name = $1
        next
    }
    {
        section = section $0 "\n"
    }
    END {
        if (section) {
            # Remove unwanted lines from the section
            section = gensub(/postinstall program:.*|preuninstall program:.*|postuninstall program:.*|posttrans program:.*/, "", "g", section)
            print section > (output_dir "/" section_name ".sh")
        }
    }
    '
    
    # Make the output scripts executable
    for scriptlet in $package_dir_scriptlet/*; do
       if [[ -s $scriptlet ]]; then
           echo ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
           echo Making scriptlet $scriptlet executable.
           chmod +x "$scriptlet"
           
           # Check the script content for /opt
           if echo "$(cat $scriptlet)" | grep -q '/opt'; then
               echo "Scriptlet contains /opt"
               SCRIPTLET_OPT_COUNT=$((SCRIPTLET_OPT_COUNT+1))
               SCRIPTLET_OPT+="$(echo "$base_name") "
           fi
           
           echo ++++++++++++++++++++++++++++
           echo $(basename $scriptlet)
           echo ++++++++++++++++++++++++++++
           cat "$scriptlet"
           echo ++++++++++++++++++++++++++++
           
           if [[ $(basename $scriptlet) == "preinstall.sh" ]]; then
               SCRIPLET_PREINST_COUNT=$((SCRIPLET_PREINST_COUNT+1))
               
               # Rename for rocm-installer
               mv "$scriptlet" "$(dirname "$scriptlet")/preinst"
               
           elif [[ $(basename $scriptlet) == "postinstall.sh" ]]; then
               SCRIPLET_POSTINST_COUNT=$((SCRIPLET_POSTINST_COUNT+1))
               
               # Rename for rocm-installer
               mv "$scriptlet" "$(dirname "$scriptlet")/postinst"
               
           elif [[ $(basename $scriptlet) == "preuninstall.sh" ]]; then
               SCRIPLET_PRERM_COUNT=$((SCRIPLET_PRERM_COUNT+1))
               
               # Rename for rocm-installer
               mv "$scriptlet" "$(dirname "$scriptlet")/prerm"
               
           elif [[ $(basename $scriptlet) == "postuninstall.sh" ]]; then
               SCRIPLET_POSTRM_COUNT=$((SCRIPLET_POSTRM_COUNT+1))
               
               # Rename for rocm-installer
               mv "$scriptlet" "$(dirname "$scriptlet")/postrm"
               
           fi
           
       else
           if [[ -f $scriptlet ]]; then
               #echo Removing empty scriptlet $(basename $scriptlet).
               rm "$scriptlet"
           fi
       fi
    done
    
    echo Extracting Scriptlets...Complete.
    echo ---------------------------------
}

extract_package() {
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo "Extracting Package...: $PACKAGE"
    
    local base_name=$(basename $PACKAGE)
    
    PACKAGE_DIR_NAME=$(echo "$base_name" | sed 's/-[0-9].*$//')
    PACKAGE_DIR=$EXTRACT_DIR/$PACKAGE_DIR_NAME
    
    echo "Package Directory Name    = $PACKAGE_DIR_NAME"
    echo "Package Extract Directory = $PACKAGE_DIR"
    
    if [ ! -d $PACKAGE_DIR ]; then
        echo Create directory $PACKAGE_DIR
        mkdir -p $PACKAGE_DIR
    fi
    
    # Extract the content from data
    if [[ $EXTRACT_CONTENT == 1 ]]; then
        extract_data
    fi
    
    # Extract package info
    extract_info
    
    # Extract the dependencies
    extract_deps
    
    # Extract the scriptlets
    extract_scriptlets
    
    # write the package list
    PACKAGE_LIST+="$PACKAGE_DIR_NAME, "
    
    # Dump the file stats on the extraction
    dump_extract_stats "$PACKAGE_DIR"
    
    echo Extracting Package...Complete.
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}

add_extra_deps() {
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo Additional Dependencies...
    
    echo "Adding Extra Dependencies."
    for pkg in "${EXTRA_DEPS[@]}"; do
        echo "    $pkg"
        GLOBAL_DEPS+=", $pkg"
    done
    
    echo "Adding Installer Dependencies."
    for pkg in "${INSTALLER_DEPS[@]}"; do
        echo "    $pkg"
        GLOBAL_DEPS+=", $pkg"
    done
    
    echo Additional Dependencies...Complete.
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}

check_package_owner() {
    AMDPKG=0
    
    local package=$(rpm -q --queryformat "%{NAME}" --nosignature $PACKAGE)
    local vendor=$(rpm -qi --nosignature $PACKAGE | grep Vendor)
    
    if [[ $package =~ "amdgpu" || $package =~ "rocm" ]]; then
        AMDPKG=1
    else
       if [[ -n $vendor ]]; then
           if [[ $vendor =~ "Advanced Micro Devices" || $vendor =~ "AMD ROCm" ]]; then
               AMDPKG=1
           fi
       fi
    fi
    
    if [[ $AMDPKG == 1 ]] ; then
        print_no_err "AMD PACKAGE"
        AMD_COUNT=$((AMD_COUNT+1))
        AMD_PACKAGES+="$(basename $PACKAGE) "
    else
        print_err "3rd Party PACKAGE"
        NON_AMD_COUNT=$((NON_AMD_COUNT+1))
        OTHER_PACKAGES+="$(basename $PACKAGE) "
    fi
}

write_package_list() {
    echo ^^^^^^^^^^^^^^^^^^^^
    echo Extracted Packages:
    echo ^^^^^^^^^^^^^^^^^^^^
    echo PKG_COUNT = $PKG_COUNT
    echo --------------------
    echo "$PACKAGE_LIST" | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u
    echo "$PACKAGE_LIST" | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u > "$EXTRACT_DIR/$EXTRACT_PACKAGE_LIST_FILE"
}

filter_deps_version() {
    echo -----------------------------
    echo Dependency Version Filter...
    
    local packages_file="$EXTRACT_DIR/$EXTRACT_PACKAGE_LIST_FILE"
    local deps_file="$EXTRACT_DIR/$EXTRACT_GLOBAL_DEPS_FILE"
    
    local deps_file_filtered="$EXTRACT_DIR/global_deps_filtered.txt"
    local reqs_file="$EXTRACT_DIR/$EXTRACT_REQUIRED_DEPS_FILE"
    
    local prev_package=""
    local prev_version=""
    local prev_line=
    
    local config_file="$EXTRACT_DIR/$EXTRACT_PKG_CONFIG_FILE"
    CONFIG_PKGS=$(<"$config_file")
    
    if [ -f "$deps_file_filtered" ]; then
        rm "$deps_file_filtered"
    fi
    
    # read the global deps file and filter to new file base on package versions
    while IFS= read -r line; do
        echo "<><><><><><><><><><><><><><><><><><><><><><>"
        echo -e "dep : \e[96m$line\e[0m"
        
        # Remove (x86-64) substrings
        line="${line//(x86-64)/}"
        
        # Remove open bracket from the start and close bracket from the end and replace " or " with "|"
        line=$(echo "$line" | sed 's/^[(]//; s/[)]$//' | sed 's/ or /|/g')
        echo "line: $line"

        # filter the versioning within brackets
        current_package=$(echo "$line" | awk -F '[()]' '{print $1}' | awk '{print $1}')
        
        # extract the current version number only
        current_version=$(echo "$line" | sed -n 's/.*[>=]\s*\(.*\)/\1/p')
        
        # init a null version to 0 (for rpmdev-vercmp)
        if [[ -z "$current_version" ]]; then
            current_version="0"
        fi
        
        echo ++++++
        echo "current  : $current_package : $current_version"
        echo "prev     : $prev_package : $prev_version"
        echo "prev_line: $prev_line"
        echo ++++++
        
        if [[ -n $prev_package ]]; then
            # check if the current and previous dep are equal.  If equal, compare the version
            if [ "$current_package" = "$prev_package" ]; then
                echo "Same package (cur = prev): comparing versions"
                if rpmdev-vercmp "$current_version" "$prev_version" | grep -q '>' ; then
                    echo "current_version > prev_version"
                    prev_version="$current_version"
                    prev_package="$current_package"
                    prev_line=$line
                else
                    echo "current_version <= prev_version"
                fi
            else
                # the packages are different, so write out the previous dep to the filter deps file
                echo "Diff package (cur != prev)"
                
                # before writing out, check for "tags" or if the dep is in the extracted package list
                if echo "$CONFIG_PKGS" | grep -qw "$prev_package"; then
                    echo -e "\e[32mConfig package: write prev_package: $prev_package\e[0m"
                    echo $prev_package >> "$deps_file_filtered"
                else
                    echo -e "\e[32mNon-Tag package: write prev_line: $prev_line\e[0m"
                    echo $prev_line >> "$deps_file_filtered"
                fi
                
                prev_package="$current_package"
                prev_line=$line
                prev_version="$current_version"
            fi
       else
            prev_line=$line
            prev_package="$current_package"
            prev_version="$current_version"
        fi
    done < "$deps_file"
    
    # write out the last line
    echo $prev_line >> "$deps_file_filtered"
    
    sort -u "$deps_file_filtered" -o "$deps_file_filtered"
    
    # diff the package list against the deps and write out deps that are not installed
    diff "$packages_file" "$deps_file_filtered" | grep '^>' | sed 's/^> //' > "$reqs_file"
    
    # remove the filtered global list
    rm "$deps_file_filtered"
    
    echo "<><><><><><><><><><><><><><><><><><><><><><>"
    echo "Required Dependencies:"
    while IFS= read -r dep; do
        echo "$dep"
    done < "$reqs_file"
    
    echo Dependency Version Filter...Complete.
}

write_global_deps() {
    echo ^^^^^^^^^^^^^^^^^^^^
    echo Global Dependencies:
    echo ^^^^^^^^^^^^^^^^^^^^
    
    echo -------------
    echo Dependencies:
    echo -------------
    echo "$GLOBAL_DEPS" | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u
    echo "$GLOBAL_DEPS" | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u > "$EXTRACT_DIR/$EXTRACT_GLOBAL_DEPS_FILE"
}

extract_rpms() {
    echo ===================================================
    echo Extracting RPMs...
    
    PKG_COUNT=0
    
    if [ -d $EXTRACT_DIR ]; then
        echo -e "\e[93mExtraction directory exists. Removing: $EXTRACT_DIR\e[0m"
        $SUDO rm -rf $EXTRACT_DIR
    fi
    
    echo Creating Extraction directory.
    mkdir -p $EXTRACT_DIR
    
    echo Extracting RPM...
    
    for pkg in $PACKAGES; do
        
        PKG_COUNT=$((PKG_COUNT+1))
        
        echo -------------------------------------------------------------------------------
        echo -e "\e[93mpkg $PKG_COUNT = $(basename $pkg)\e[0m"
        
        PACKAGE=$pkg
        
        check_package_owner
        if [[ $AMDPKG == 1 ]]; then
            extract_package
        fi
        
    done
    
    echo Extracting RPMs...Complete.
}

combine_rocm_deps() {
    echo ===================================================
    echo Combining dependencies from all component-rocm subdirectories...

    # Use the component-rocm directory for combined deps files
    local deps_dir="../rocm-installer/component-rocm"
    if [ ! -d "$deps_dir" ]; then
        echo "ERROR: $deps_dir directory does not exist!"
        return 1
    fi

    local combined_deps_file="$deps_dir/rocm_required_deps_rpm.txt"
    local gfx_deps_file="$deps_dir/rocm_required_deps_rpm_gfx.tmp"
    local gfx_deps_sorted="$deps_dir/rocm_required_deps_rpm_gfx_sorted.tmp"
    local gfx_deps_filtered="$deps_dir/rocm_required_deps_rpm_gfx_filtered.tmp"
    local temp_deps_file="$deps_dir/rocm_required_deps_rpm.tmp"

    # Remove only the output file and temporary files
    echo "Removing previous rocm_required_deps_rpm.txt if it exists..."
    rm -f "$combined_deps_file" "$gfx_deps_file" "$gfx_deps_sorted" "$gfx_deps_filtered" "$temp_deps_file"

    # First pass: Process all gfx-specific component-rocm/gfx* subdirectories
    local gfx_component_count=0
    for component_dir in ../rocm-installer/component-rocm/gfx*; do
        if [ -d "$component_dir" ]; then
            local required_deps_file="$component_dir/$EXTRACT_REQUIRED_DEPS_FILE"
            if [ -f "$required_deps_file" ]; then
                echo "Processing gfx dependencies from: $component_dir"
                cat "$required_deps_file" >> "$gfx_deps_file"
                gfx_component_count=$((gfx_component_count + 1))
            fi
        fi
    done

    # Collect all packages.txt files from all subdirectories to create comprehensive filter list
    local all_packages_file="$deps_dir/all_packages.tmp"
    rm -f "$all_packages_file"

    echo "Collecting all package names from all subdirectories for filtering..."
    for component_dir in ../rocm-installer/component-rocm/*/; do
        if [ -d "$component_dir" ]; then
            local packages_file="$component_dir/$EXTRACT_PACKAGE_LIST_FILE"
            if [ -f "$packages_file" ]; then
                cat "$packages_file" >> "$all_packages_file"
            fi
        fi
    done

    # Combine, sort and remove duplicates from gfx dependencies
    if [ -f "$gfx_deps_file" ]; then
        echo "Combining, sorting, and removing duplicates from $gfx_component_count gfx component subdirectories..."
        sort -u "$gfx_deps_file" > "$gfx_deps_sorted"
        rm -f "$gfx_deps_file"

        # Filter out AMD ROCm packages using the comprehensive package list
        if [ -f "$all_packages_file" ]; then
            echo "Filtering out AMD ROCm packages from gfx dependencies..."
            # Extract package name and filter against AMD packages list
            while IFS= read -r dep_line; do
                pkg_name=$(echo "$dep_line" | awk '{print $1}' | sed 's/[<>=()].*//')
                if ! grep -qxF "$pkg_name" "$all_packages_file"; then
                    echo "$dep_line"
                fi
            done < "$gfx_deps_sorted" > "$gfx_deps_filtered"

            local filtered_count=$(wc -l < "$gfx_deps_sorted")
            local remaining_count=$(wc -l < "$gfx_deps_filtered")
            local removed_count=$((filtered_count - remaining_count))
            echo "Filtered out $removed_count AMD ROCm package dependencies"
            echo "Remaining external dependencies: $remaining_count"

            # Use filtered file as temp
            cp "$gfx_deps_filtered" "$temp_deps_file"
            rm -f "$gfx_deps_sorted" "$gfx_deps_filtered"
        else
            echo "WARNING: No packages.txt files found, skipping filter"
            cp "$gfx_deps_sorted" "$temp_deps_file"
            rm -f "$gfx_deps_sorted"
        fi
    fi

    # Second pass: Combine with base component-rocm/base subdirectory
    local base_component_dir="../rocm-installer/component-rocm/base"
    if [ -d "$base_component_dir" ]; then
        local required_deps_file="$base_component_dir/$EXTRACT_REQUIRED_DEPS_FILE"
        if [ -f "$required_deps_file" ]; then
            echo "Processing base component dependencies from: $base_component_dir"
            cat "$required_deps_file" >> "$temp_deps_file"
        fi
    fi

    # Final sort and remove duplicates, then filter against all AMD packages
    if [ ! -f "$temp_deps_file" ]; then
        echo "WARNING: No component-rocm subdirectories with required_deps.txt found"
        rm -f "$all_packages_file"
        return 1
    fi

    echo "Final combining, sorting, and filtering out AMD packages..."
    if [ -f "$all_packages_file" ]; then
        # Filter out any AMD packages from the combined dependencies
        # Extract package name (before space or comparison operator) and check against packages list
        sort -u "$temp_deps_file" | while IFS= read -r dep_line; do
            # Extract package name (everything before first space, =, <, >, or ()
            pkg_name=$(echo "$dep_line" | awk '{print $1}' | sed 's/[<>=()].*//')
            # Check if package name is in the AMD packages list
            if ! grep -qxF "$pkg_name" "$all_packages_file"; then
                echo "$dep_line"
            fi
        done > "$combined_deps_file"
    else
        sort -u "$temp_deps_file" > "$combined_deps_file"
    fi

    rm -f "$temp_deps_file" "$all_packages_file"

    local total_deps=$(wc -l < "$combined_deps_file")
    local total_components=$((gfx_component_count + 1))
    echo "Combined dependencies from $gfx_component_count gfx component subdirectories + base component"
    echo "Total unique required dependencies: $total_deps"
    echo "Output file: $combined_deps_file"

    echo Combining dependencies...Complete.
}

extract_meta_package_deps() {
    local meta_package="$1"
    local gfx_tag="$2"
    local output_file="$3"

    echo "=========================================="
    echo "Extracting dependencies for meta package: $meta_package"
    echo "GFX tag: $gfx_tag"
    echo "=========================================="

    # Determine base directory for this gfx tag
    local base_dir="../rocm-installer/component-rocm/$gfx_tag"

    # Check if meta package directory exists
    local meta_pkg_dir="$base_dir/$meta_package"
    if [ ! -d "$meta_pkg_dir" ]; then
        echo "ERROR: Meta package directory not found: $meta_pkg_dir"
        return 1
    fi

    # Check if deps.txt exists for the meta package
    local meta_deps_file="$meta_pkg_dir/deps/deps.txt"
    if [ ! -f "$meta_deps_file" ]; then
        echo "ERROR: deps.txt not found for meta package: $meta_deps_file"
        return 1
    fi

    # Use associative array to track processed packages and avoid duplicates
    declare -A processed_packages
    declare -A all_dependencies

    # Add the meta package itself to the dependencies list
    all_dependencies["$meta_package"]=1

    # Queue for processing - start with meta package dependencies
    local -a process_queue=()

    # Read initial dependencies from meta package
    echo "Reading initial dependencies from: $meta_deps_file"
    while IFS= read -r dep_line; do
        # Skip empty lines and comments
        [[ -z "$dep_line" || "$dep_line" =~ ^# ]] && continue

        # Extract package name (remove version info after "=")
        local pkg_name=$(echo "$dep_line" | sed 's/[[:space:]]*=.*//' | xargs)

        # Only process amdrocm packages
        if [[ "$pkg_name" =~ ^amdrocm- ]]; then
            process_queue+=("$pkg_name")
            all_dependencies["$pkg_name"]=1
        fi
    done < "$meta_deps_file"

    echo "Initial dependencies found: ${#process_queue[@]} (plus meta package itself)"

    # Recursively process dependencies
    local queue_index=0
    while [ $queue_index -lt ${#process_queue[@]} ]; do
        local current_pkg="${process_queue[$queue_index]}"
        queue_index=$((queue_index + 1))

        # Skip if already processed
        if [[ -n "${processed_packages[$current_pkg]}" ]]; then
            continue
        fi

        # Mark as processed
        processed_packages["$current_pkg"]=1

        echo "Processing dependencies for: $current_pkg"

        # Determine which directory to check (base or gfxXYZ)
        local pkg_deps_file=""

        # First check in the current gfx directory
        if [ -f "$base_dir/$current_pkg/deps/deps.txt" ]; then
            pkg_deps_file="$base_dir/$current_pkg/deps/deps.txt"
        # Then check in base directory (for packages like amdrocm-base, amdrocm-runtime, etc.)
        elif [ -f "../rocm-installer/component-rocm/base/$current_pkg/deps/deps.txt" ]; then
            pkg_deps_file="../rocm-installer/component-rocm/base/$current_pkg/deps/deps.txt"
        else
            # Check in all gfx directories (for gfx-specific packages when processing base meta packages)
            local found_in_gfx=0
            for gfx_check_dir in ../rocm-installer/component-rocm/gfx*; do
                if [ -f "$gfx_check_dir/$current_pkg/deps/deps.txt" ]; then
                    pkg_deps_file="$gfx_check_dir/$current_pkg/deps/deps.txt"
                    found_in_gfx=1
                    echo "  Found in gfx directory: $(basename $gfx_check_dir)"
                    break
                fi
            done

            if [ $found_in_gfx -eq 0 ]; then
                echo "  Warning: deps.txt not found for $current_pkg, skipping"
                continue
            fi
        fi

        # Read dependencies for this package
        while IFS= read -r dep_line; do
            # Skip empty lines and comments
            [[ -z "$dep_line" || "$dep_line" =~ ^# ]] && continue

            # Extract package name (remove version info after "=")
            local dep_pkg_name=$(echo "$dep_line" | sed 's/[[:space:]]*=.*//' | xargs)

            # Only process amdrocm packages
            if [[ "$dep_pkg_name" =~ ^amdrocm- ]]; then
                # Add to dependencies if not already present
                if [[ -z "${all_dependencies[$dep_pkg_name]}" ]]; then
                    all_dependencies["$dep_pkg_name"]=1
                    process_queue+=("$dep_pkg_name")
                    echo "  Found dependency: $dep_pkg_name"
                fi
            fi
        done < "$pkg_deps_file"
    done

    echo "Total unique amdrocm dependencies found: ${#all_dependencies[@]}"

    # Write sorted dependencies to output file
    echo "Writing dependencies to: $output_file"
    printf "%s\n" "${!all_dependencies[@]}" | sort > "$output_file"

    echo "Dependency extraction complete for $meta_package"
    echo "Output: $output_file"
    echo ""

    return 0
}

extract_meta_packages() {
    echo "=========================================="
    echo "Extracting meta package configurations..."
    echo "=========================================="

    echo "Using ROCM_VER: $ROCM_VER"

    # Create meta directory in component-rocm
    local meta_dir="../rocm-installer/component-rocm/meta"
    if [ ! -d "$meta_dir" ]; then
        echo "Creating meta directory: $meta_dir"
        mkdir -p "$meta_dir"
    fi

    # Define the specific meta packages for gfxXYZ directories
    local gfx_meta_packages=(
        "amdrocm-core"
        "amdrocm-core-sdk"
        "amdrocm-core-devel"
    )

    # Define the specific meta packages for base directory
    local base_meta_packages=(
        "amdrocm-developer-tools"
    )

    # Process each gfxXYZ directory
    for gfx_dir in ../rocm-installer/component-rocm/gfx*; do
        if [ ! -d "$gfx_dir" ]; then
            continue
        fi

        local gfx_tag=$(basename "$gfx_dir")
        echo ""
        echo "Processing $gfx_tag directory..."

        # Process each specific meta package
        for meta_pkg_base in "${gfx_meta_packages[@]}"; do
            # Construct the full meta package name: e.g., amdrocm-core7.12-gfx94x
            local meta_pkg_name="${meta_pkg_base}${ROCM_VER}-${gfx_tag}"
            local meta_pkg_dir="$gfx_dir/$meta_pkg_name"

            if [ ! -d "$meta_pkg_dir" ]; then
                echo "  Meta package not found: $meta_pkg_name (skipping)"
                continue
            fi

            echo "  Found meta package: $meta_pkg_name"

            # Output file will be in the meta directory with -meta.config suffix
            local output_file="$meta_dir/${meta_pkg_name}-meta.config"

            # Extract dependencies for this meta package
            extract_meta_package_deps "$meta_pkg_name" "$gfx_tag" "$output_file"
        done
    done

    # Process base directory for non-gfx meta packages
    local base_dir="../rocm-installer/component-rocm/base"
    if [ -d "$base_dir" ]; then
        echo ""
        echo "Processing base directory for non-gfx meta packages..."

        # Process each specific meta package
        for meta_pkg_base in "${base_meta_packages[@]}"; do
            # Construct the full meta package name: e.g., amdrocm-developer-tools7.12
            local meta_pkg_name="${meta_pkg_base}${ROCM_VER}"
            local meta_pkg_dir="$base_dir/$meta_pkg_name"

            if [ ! -d "$meta_pkg_dir" ]; then
                echo "  Meta package not found: $meta_pkg_name (skipping)"
                continue
            fi

            echo "  Found base meta package: $meta_pkg_name"

            # Output file will be in the meta directory with -meta.config suffix
            local output_file="$meta_dir/${meta_pkg_name}-meta.config"

            # Extract dependencies for this meta package
            extract_meta_package_deps "$meta_pkg_name" "base" "$output_file"
        done
    fi

    echo ""
    echo "Meta package configuration extraction complete."
}

combine_rocm_deps_meta() {
    echo "=========================================="
    echo "Combining ROCm dependencies metadata..."
    echo "=========================================="

    # This function is a wrapper that calls extract_meta_packages
    # It can be extended in the future to perform additional metadata operations

    extract_meta_packages

    echo "ROCm dependencies metadata combination complete."
}

extract_rocm_rpms() {
    echo ===================================================
    echo Extracting ROCm RPMs...

    PACKAGE_DIR="$PACKAGE_ROCM_DIR"
    EXTRACT_PKG_CONFIG_FILE="$EXTRACT_ROCM_PKG_CONFIG_FILE"

    # Check if package directory exists
    if [[ ! -d "$PACKAGE_DIR" ]]; then
        echo "ERROR: Package directory not found: $PACKAGE_DIR"
        return 1
    fi

    echo "Processing packages from: $PACKAGE_DIR"
    echo "Organizing by gfx tag into component-rocm subdirectories..."
    
    PACKAGE_LIST=

    # Collect all package files and group by gfx tag
    declare -A GFX_PACKAGES
    GFX_PACKAGES["base"]=""

    for pkg_file in "$PACKAGE_DIR"/*.rpm; do
        if [[ -f "$pkg_file" ]]; then
            pkg_name=$(basename "$pkg_file")

            # Detect gfx tag from package name (e.g., amdrocm-blas7.11-gfx94x-*.rpm)
            if [[ "$pkg_name" =~ -gfx([0-9a-z]+)- ]]; then
                gfx_tag="gfx${BASH_REMATCH[1]}"
                GFX_PACKAGES["$gfx_tag"]+="$pkg_file "
            else
                # Non-gfx package goes to base
                GFX_PACKAGES["base"]+="$pkg_file "
            fi
        fi
    done

    # Process each gfx group
    for gfx_tag in "${!GFX_PACKAGES[@]}"; do
        pkg_list="${GFX_PACKAGES[$gfx_tag]}"

        # Skip if no packages for this tag
        if [[ -z "$pkg_list" ]]; then
            continue
        fi

        echo ""
        echo "=========================================="
        echo "Processing $gfx_tag packages"
        echo "=========================================="

        # Set extract directory for this gfx tag
        EXTRACT_DIR="../rocm-installer/component-rocm/$gfx_tag"

        echo "EXTRACT_DIR = $EXTRACT_DIR"
        echo -----------------------------------------

        init_stats

        # Set PKG_LIST and PACKAGES for this gfx group
        PKG_LIST=($pkg_list)
        PACKAGES="$pkg_list"
        PKG_COUNT=${#PKG_LIST[@]}

        extract_rpms

        add_extra_deps

        write_extract_info
        filter_deps_version

        echo -e "\e[93m========================================\e[0m"
        echo -e "\e[93mExtracted: $PKG_COUNT $gfx_tag packages\e[0m"
        echo -e "\e[93m========================================\e[0m"
    done

    # Combine dependencies from all component-rocm subdirectories
    echo ""
    combine_rocm_deps

    # Extract meta package configurations
    echo ""
    combine_rocm_deps_meta

    echo ""
    echo Extracting ROCm RPMs...Complete.
}

extract_amdgpu_rpms() {
    echo ===================================================
    echo Extracting AMDGPU RPMs...
    
    echo -----------------------------------------
    echo "PACKAGE_AMDGPU_DIR = $PACKAGE_AMDGPU_DIR"
    echo "EXTRACT_AMDGPU_DIR = $EXTRACT_AMDGPU_DIR"
    echo ------------------------------------------
    
    PACKAGE_DIR="$PACKAGE_AMDGPU_DIR"
    EXTRACT_DIR="$EXTRACT_AMDGPU_DIR"
    
    EXTRACT_PKG_CONFIG_FILE="$EXTRACT_AMDGPU_PKG_CONFIG_FILE"
    
    init_stats
    
    echo Getting package list...
    PACKAGE_LIST=
    
    if [ ! -d $PACKAGE_DIR ]; then
        print_err "$PACKAGE_DIR does not exist."
        exit 1
    fi
    
    for pkg in $PACKAGE_DIR/*; do
        if [[ $pkg == *.rpm ]]; then
            echo $pkg
            PACKAGES+="$pkg "
        fi
    done
    
    # Extract the amdgpu rpms
    extract_rpms
    
    echo Extracting AMDGPU RPMs...Complete.

    echo -e "\e[93m========================================\e[0m"
    echo -e "\e[93m$PKG_COUNT AMDGPU packages extracted\e[0m"
    echo -e "\e[93m========================================\e[0m"
    
    # extract the amdgpu-dkms build version
    local amdgpu_dkms_path="$EXTRACT_AMDGPU_DIR/amdgpu-dkms/content/usr/src"

    if [ -d $amdgpu_dkms_path ]; then
        AMDGPU_DKMS_BUILD_VER=$(ls $amdgpu_dkms_path)
        AMDGPU_DKMS_BUILD_VER=${AMDGPU_DKMS_BUILD_VER#amdgpu-}

        echo AMDGPU_DKMS_BUILD_VER = $AMDGPU_DKMS_BUILD_VER

        # Create root-level amdgpu-dkms-ver.txt with distro suffix removed
        local root_amdgpu_dkms_file="../rocm-installer/component-amdgpu/$EXTRACT_AMDGPU_DKMS_VER_FILE"
        # Strip distro suffix using sed to match known patterns
        # e.g., 6.16.13-2278356.24.04 -> 6.16.13-2278356
        # e.g., 6.16.13-2278356.el8 -> 6.16.13-2278356
        # e.g., 6.16.13-2278356.amzn2023 -> 6.16.13-2278356
        local clean_ver=$(echo "$AMDGPU_DKMS_BUILD_VER" | sed -E 's/\.(el[0-9]+|amzn[0-9]+|[0-9]+\.[0-9]+)$//')

        echo "Writing root AMDGPU_DKMS_VER (distro suffix removed) = $clean_ver"
        mkdir -p "$(dirname "$root_amdgpu_dkms_file")"
        echo "$clean_ver" > "$root_amdgpu_dkms_file"
    fi
    
    # reorder the amdgpu package config to ensure the order
    local config_file="$EXTRACT_DIR/$EXTRACT_AMDGPU_PKG_CONFIG_FILE"
    
    local packages=$(cat "$config_file")
    local reordered_packages=""

    # Ensure "amdgpu-dkms-firmware" is the first package
    if echo "$packages" | grep -q "^amdgpu-dkms-firmware$"; then
        reordered_packages+="amdgpu-dkms-firmware"$'\n'
        packages=$(echo "$packages" | grep -v "^amdgpu-dkms-firmware$")
    fi

    # Ensure "amdgpu-dkms" is the second package
    if echo "$packages" | grep -q "^amdgpu-dkms$"; then
        reordered_packages+="amdgpu-dkms"$'\n'
        packages=$(echo "$packages" | grep -v "^amdgpu-dkms$")
    fi

    # Append the remaining packages
    reordered_packages+="$packages"

    # Write the reordered packages back to the config file
    echo "$reordered_packages" > "$config_file"
    echo "Reordered packages written to '$config_file'."
}

extract_tar_setup() {
    echo ===================================================
    echo Extracting Tar Setup...
    
    echo -----------------------------------------
    echo "EXTRACT_TAR_DIR = $EXTRACT_TAR_DIR"
    echo ------------------------------------------
    
    SCRIPT_NAME="setup-rocm.sh"
    SCRIPT_DIR="$EXTRACT_ROCM_DIR/setup/script/rocm-$ROCM_VER"

    echo "Script           = $SCRIPT_NAME"
    echo "Script Directory = $SCRIPT_DIR"
    
    if [ -f "$EXTRACT_TAR_DIR/$SCRIPT_NAME" ]; then
        # copy the setup script into the rocm directory for untarring
        if [ ! -d $SCRIPT_DIR ]; then
            echo Create directory $SCRIPT_DIR
            mkdir -p $SCRIPT_DIR
        fi
        
        echo "Copying tar setup script to rocm directory."
        
        cp "$EXTRACT_TAR_DIR/$SCRIPT_NAME" "$SCRIPT_DIR/"
    else
        echo "Tar setup script not found."
    fi
    
    echo Extracting Tar Setup...Complete.
}

write_extract_info() {
    dump_extract_stats "$EXTRACT_DIR"
    
    write_global_deps
    write_package_list
    
    scriptlet_stats
}


####### Main script ###############################################################

# Create the extraction log directory
if [ ! -d $EXTRACT_LOGS_DIR ]; then
    mkdir -p $EXTRACT_LOGS_DIR
fi

exec > >(tee -a "$EXTRACT_CURRENT_LOG") 2>&1

echo ===============================
echo PACKAGE EXTRACTOR - RPM
echo ===============================

PROG=${0##*/}
SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)

os_release

if [ "$#" -lt 1 ]; then
   echo Missing argument
   exit 1
fi

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
    amdgpu)
        echo "Enabling amdgpu extract."
        AMDGPU_EXTRACT=1
        shift
        ;;
    rocm)
        echo "Enabling rocm extract."
        ROCM_EXTRACT=1
        shift
        ;;
    nocontent)
        echo "Disabling content/data extraction."
        EXTRACT_CONTENT=0
        shift
        ;;
    contentlist)
        echo "Enabling content file listing during extraction."
        CONTENT_LIST=1
        shift
        ;;
    pkgs-rocm=*)
        PACKAGE_ROCM_DIR="${1#*=}"
        echo "Using ROCm Packages source: $PACKAGE_ROCM_DIR"
        shift
        ;;
    pkgs-amdgpu=*)
        PACKAGE_AMDGPU_DIR="${1#*=}"
        echo "Using AMDGPU Packages source: $PACKAGE_AMDGPU_DIR"
        shift
        ;;
    ext-rocm=*)
        EXTRACT_ROCM_DIR="${1#*=}"
        EXTRACT_ROCM_DIR+="/component-rocm"
        echo "Extract ROCm output: $EXTRACT_ROCM_DIR"
        shift
        ;;
    ext-amdgpu=*)
        EXTRACT_AMDGPU_DIR="${1#*=}"
        echo "Extract AMDGPU output: $EXTRACT_AMDGPU_DIR"
        shift
        ;;
    *)
        shift
        ;;
    esac
done

prompt_user "Extract packages (y/n): "
if [[ $option == "N" || $option == "n" ]]; then
    echo "Exiting extractor."
    exit 1
fi

install_tools

if [[ $ROCM_EXTRACT == 1 ]]; then
    extract_rocm_rpms

    extract_tar_setup
fi

if [[ $AMDGPU_EXTRACT == 1 ]]; then
    extract_amdgpu_rpms
    write_extract_info
    
    filter_deps_version
fi

if [[ -n $EXTRACT_CURRENT_LOG ]]; then
    echo -e "\e[32mExtract log stored in: $EXTRACT_CURRENT_LOG\e[0m"
fi

