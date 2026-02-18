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

COMPO_DIR=
DEPS_COUNT=0

DEPS=
DEPS_ROCM=
DEPS_AMDGPU=

USE_ROCM=0
USE_AMDGPU=0

MISSING_DEPS=
MISSING_DEPS_COUNT=0
INSTALL_DEPS_COUNT=0

DEPS_LIST_ONLY=0
PROMPT_USER=0
VERBOSE=0

INSTALLABLE_PKG_CACHE=

NO_CMD_OUTPUT="> /dev/null 2>&1"

GCC_TOOLSET_PACKAGES_OL=(gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ gcc-toolset-11-gcc-gfortran gcc-toolset-11-libquadmath-devel gcc-toolset-11-libstdc++-devel gcc-toolset-11-gcc-gdb-plugin)

EPEL_SETUP=1

###### Functions ###############################################################

usage() {
cat <<END_USAGE
The dependencies installer reads in the required_deps.txt file for extracted component-<name>
and lists or installs missing dependencies based on the required deps list.

Usage: $PROG [options]

[options}:
    help         = Display this help information.
    prompt       = Prompts user prior to installing dependencies.
    amdgpu       = Check dependencies for extracted AMDGPU packages.
    rocm         = Check dependencies for extracted ROCm packages.
    list         = List the dependencies (amdgpu, rocm, amdgpu and rocm)
    install      = Install dependencies for selected packages (amdgpu, rocm, amdgpu and rocm)
    install-file = Install dependencies from a file.
    verbose      = Enable verbose logging
        
    Example:
    
    ./deps-installer.sh list amdgpu rocm    = list dependencies for both amdgpu and rocm
    ./deps-installer.sh install rocm        = install rocm dependencies
       
END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID

        case "$ID" in
        ubuntu|debian)
            DISTRO_PACKAGE_MGR="apt"
            DISTRO_CACHE_CHK="apt-cache show"
            DISTRO_VIRTUAL_CHK="apt-cache showpkg"
            PACKAGE_TYPE="deb"
            ;;
        rhel|centos|ol|rocky|almalinux|amzn)
            DISTRO_PACKAGE_MGR="dnf"
            DISTRO_CACHE_CHK="$SUDO dnf --cacheonly info"
            DISTRO_VIRTUAL_CHK="dnf provides"
            PACKAGE_TYPE="rpm"

            if [[ "$DISTRO_NAME" = "amzn" ]]; then
                echo "Disable EPEL/CRB for Amazon."
                EPEL_SETUP=0
            fi

            ;;
        sles)
            DISTRO_PACKAGE_MGR="zypper"
            DISTRO_CACHE_CHK="zypper search --type package --match-exact"
            DISTRO_VIRTUAL_CHK="zypper search --provides --match-exact"
            PACKAGE_TYPE="rpm"

            # install awk if required
            if ! rpm -qa | grep -q "awk"; then
                echo "Package: awk missing. Installing..."
                $SUDO zypper install -y awk > /dev/null 2>&1
                echo "Package: awk installed."
            fi

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
        
    DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
    DISTRO_MAJOR_VER=${DISTRO_VER%.*}
    
    # Rocky 9 support only
    if [[ $DISTRO_VER != 9* ]] && [[ "$DISTRO_NAME" = "rocky" ]]; then
        echo "$DISTRO_NAME $DISTRO_VER is not a supported OS"
        exit 1
    fi
    
    echo "Dependency install on $DISTRO_NAME $DISTRO_VER."
}

get_amdgpu_distro_tag() {
    # Map the current distro to the AMDGPU component directory name
    local distro_tag=""

    case "$DISTRO_NAME" in
        ubuntu)
            # Ubuntu: 24.04 -> ub24, 22.04 -> ub22
            distro_tag="ub${DISTRO_MAJOR_VER}"
            ;;
        debian)
            # Debian: 12 -> ub22, 13 -> ub24
            case "$DISTRO_MAJOR_VER" in
                12)
                    distro_tag="ub22"
                    ;;
                13)
                    distro_tag="ub24"
                    ;;
                *)
                    distro_tag="ub${DISTRO_MAJOR_VER}"
                    ;;
            esac
            ;;
        rhel)
            # RHEL: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        centos)
            # CentOS Stream: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        ol)
            # Oracle Linux: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        rocky)
            # Rocky: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        almalinux)
            # AlmaLinux: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        amzn)
            # Amazon Linux: 2023 -> amzn2023
            distro_tag="amzn${DISTRO_VER}"
            ;;
        sles)
            # SLES: 15.6 -> sle15, 15.5 -> sle15
            distro_tag="sle${DISTRO_MAJOR_VER}"
            ;;
        *)
            echo "Unknown distro: $DISTRO_NAME"
            return 1
            ;;
    esac

    echo "$distro_tag"
    return 0
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
    local str=$1
    local clr=$2
    
    if [[ $VERBOSE == 1 ]]; then
        if [[ $clr == 1 ]]; then
            echo -e "\e[93m$str\e[0m"   # yellow
        elif [[ $clr == 2 ]]; then
            echo -e "\e[32m$str\e[0m"   # green  
        elif [[ $clr == 3 ]]; then
            echo -e "\e[31m$str\e[0m"   # red
        elif [[ $clr == 4 ]]; then
            echo -e "\e[35m$str\e[0m"   # purple
        elif [[ $clr == 5 ]]; then
            echo -e "\e[36m$str\e[0m"   # cyan
        else
            echo "$str"                 # white
        fi
    fi
}

remove_rocky_kernel_repo() {
    if [ -f /etc/yum.repos.d/appstream-amdgpu.repo ]; then
        echo Removing Rocky AppStream repos...
        
        echo =-=-=-= Removing appstream-amdgpu.repo =-=-=-=
        $SUDO rm /etc/yum.repos.d/appstream-amdgpu.repo
        
         # Cleanup the dnf caches
        sudo dnf clean all
        sudo rm -rf /var/cache/dnf/*
        sudo dnf makecache
        
        echo Removing Rocky AppStream repos...Complete.
    fi
}

cleanup() {
    echo "------------------------------------"
    echo Cleaning up...
    
    if [[ "$DISTRO_NAME" = "rocky" ]]; then
        remove_rocky_kernel_repo
    fi
    
    echo Cleaning up...Complete.
}

check_virtual_package_deb() {
    local package_name=$1
    local installable_package=""
    local install_result=1
    
    # Check if the package is a virtual package
    if apt-cache showpkg "$package_name" | grep -q "Reverse Provides:"; then
        # Find the package that provides the virtual package
        installable_package=$($DISTRO_VIRTUAL_CHK "$package_name" | awk '
            /Reverse Provides:/ {
                getline;
                if ($1 != "" && $1 != "Reverse") {
                    print $1;
                }
            }'
        )
        
        if [[ -n $installable_package ]]; then
            echo "Package $package_name can be installed via $installable_package."
            install_result=0
        else
            echo "No providing package found for $package_name."
        fi
    fi

    # set the virtual package name
    VIRTUAL_PACKAGE="$installable_package"
    
    return $install_result
}

is_pkg_installable_deb() {
    local dep="$1"
    local install_result=0
    
    eval "$DISTRO_CACHE_CHK $dep $NO_CMD_OUTPUT"
    install_result=$?
    
    # if the dep is a virtual package - check for the virtual package
    if [[ $install_result -eq 0 ]]; then
        print_str "Checking for virtual package."
        
        if ! $DISTRO_CACHE_CHK $dep 2>&1 | grep -q "Package"; then
            echo Virtual package.
            check_virtual_package_deb "$dep"
            install_result=$?
        fi
    fi
    
    # get the version of the installable package
    if [[ $install_result -eq 0 ]]; then
        INSTALL_VER=$(apt-cache show $dep 2>/dev/null | grep -m1 "^Version:" | awk '{print $2}' | cut -d'-' -f1)
    fi
    
    return $install_result
}

get_dep_from_cache() {
    local dep="$1"
    
    if [[ -z "$dep" ]]; then
        return 1
    fi
    
    local entry=$(echo "$INSTALLABLE_PKG_CACHE" | grep "^$dep,")
    
    if [[ -n "$entry" ]]; then
        DEPS_INTS_NAME=$(echo "$entry" | cut -d',' -f1)
        DEPS_INST_VER=$(echo "$entry" | cut -d',' -f2)
        return 0
    else
        DEPS_INTS_NAME=""
        DEPS_INST_VER=""
        return 1
    fi
}

check_pkg_cache() {
    print_str "Checking: package cache for $1" 1
    
    local pkg=$1
    local pkg_cached=1
    
    if [[ -z $INSTALLABLE_PKG_CACHE ]]; then
        # No cache available
        return 1
    fi
    
    get_dep_from_cache "$pkg"
    if [[ $? -eq 0 ]]; then
        print_str "Package : $DEPS_INTS_NAME | $DEPS_INST_VER (cache)" 2
        pkg_cached=0
    else
        print_str "Package : $pkg not found in cache."
    fi
    
    return $pkg_cached
}

build_installable_pkg_cache_dnf() {
    echo "------------------------------------"
    echo "Building dnf installable cache..."
    
    PACKAGE_DEP_LIST=""
    
    if [[ -n $MISSING_DEPS ]]; then
        # Build a list of all the potential deps that are missing, removing all versioning/multiple deps etc.
        IFS=',' read -ra package_array <<< "$MISSING_DEPS"
        
        for pkg in "${package_array[@]}"; do
           testdep=$(echo "$pkg" | sed 's/^ *//')
           
           # check for a multi-option dependency
            if echo "$testdep" | grep -q '|'; then
            
                # build the list of packages removing all spaces and replacing "|" with spaces
                deplist=$(echo "$testdep" | tr -d ' ' | tr '|' ' ')
                
                for pkg in $deplist; do
                    print_str "+++++++++++++++++++++"
                    print_str "Checking: $pkg"
                    
                    # remove versioning
                    if [ $PACKAGE_TYPE == "deb" ]; then
                        dep=$(echo $pkg | sed -E 's/\([><!=]*[0-9.]*\)//g')
                        dep_version=$(echo "$pkg" | grep -oE '[0-9.]+')
                    else
                        dep=$(echo $pkg | sed -E 's/[><=!]=?[0-9.]+//g')
                        dep_version=$(echo "$pkg" | grep -oE '[0-9.]+')
                    fi
                    print_str "Name/Ver: $dep | $dep_version"
                    PACKAGE_DEP_LIST+="$dep "
                done
                
            else
                VIRTUAL_PACKAGE=""
            
                # single dependency check
                dep=$(echo "$testdep" | awk '{print $1}')
                PACKAGE_DEP_LIST+="$dep "
            fi
        done
        
        # query the dnf cache for the list of packages available for install using the full missing deps list
        INSTALLABLE_PKG_CACHE=$($DISTRO_CACHE_CHK $PACKAGE_DEP_LIST | grep -E "^(Name|Version)" | sed 'N;s/Name *: *\(.*\)\nVersion *: *\(.*\)/\1,\2/' | sort -u)
         
        echo ----------------------
        echo $INSTALLABLE_PKG_CACHE
        echo ----------------------
    fi
    
    echo "Building dnf installable cache...Complete."
}

# Extract actual package name from dnf provides output
get_package_from_provides_dnf() {
    local capability="$1"
    local providing_package=""

    # dnf provides output format:
    # package-name-version.arch : Summary
    # Repo        : repo-name
    # Matched from:
    # Provide    : capability

    # Filter out status messages (Updating, Last metadata, etc.) then look for package lines
    # Status messages to exclude: "Updating...", "Last metadata...", etc.
    providing_package=$(dnf provides "$capability" 2>/dev/null | \
        grep -vE "^(Updating|Last metadata|Repo|Matched from|Provide)" | \
        grep -E "^[a-zA-Z0-9]" | \
        head -1 | \
        awk '{print $1}' | \
        sed 's/-[0-9].*//')

    if [[ -n $providing_package ]]; then
        print_str "Capability '$capability' provided by package: $providing_package" 1
        VIRTUAL_PACKAGE="$providing_package"
        return 0
    fi

    return 1
}

# Extract actual package name from zypper provides output
get_package_from_provides_zypper() {
    local capability="$1"
    local providing_package=""

    print_str "Searching for provider of capability: $capability" 1

    # Use zypper search --provides to find packages that provide this capability
    # Not using --match-exact to allow partial matching (e.g., libatomic -> libatomic1)
    local search_output=$(zypper search --provides "$capability" 2>&1)

    # Parse the table output: S  | Name | Summary | Type
    # Extract the Name column (field 2) from data rows
    providing_package=$(echo "$search_output" | \
        grep "^[[:space:]]*|" | \
        grep -v "^S[[:space:]]*|" | \
        grep -v "^[-]*$" | \
        awk -F'|' 'NF>=3 {
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            if ($2 != "" && $2 != "Name") {
                print $2;
                exit;
            }
        }')

    # Validate and verify the package exists
    if [[ -n $providing_package ]]; then
        if zypper info "$providing_package" &>/dev/null; then
            print_str "Capability '$capability' -> package: $providing_package" 2
            VIRTUAL_PACKAGE="$providing_package"
            return 0
        fi
    fi

    print_str "No provider found for capability: $capability" 3
    return 1
}

is_pkg_installable_rpm() {
    local dep="$1"
    local install_result=0

    # first check for the dep in the installable package cache
    check_pkg_cache $dep
    install_result=$?

    # if not in the installable package cache, check the for dep directly again
    if [[ $install_result -ne 0 ]]; then
        eval "$DISTRO_CACHE_CHK $dep $NO_CMD_OUTPUT"
        install_result=$?
    fi

    # if the dep is a virtual package/capability - check for the providing package
    if [[ $install_result -ne 0 ]]; then
        if [[ -n $DISTRO_VIRTUAL_CHK ]]; then
            print_str "Base package not found.  Checking for providing package via capability."
            print_str "$DISTRO_VIRTUAL_CHK: $dep" 1

            # Use distro-specific function to extract the providing package name
            if [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
                get_package_from_provides_dnf "$dep"
                install_result=$?
            else
                get_package_from_provides_zypper "$dep"
                install_result=$?
            fi
        fi
    fi

    # get the version of the installable package
    if [[ $install_result -eq 0 ]]; then
        if [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
            if [[ -n $VIRTUAL_PACKAGE ]]; then
                INSTALL_VER=$(dnf info "$VIRTUAL_PACKAGE" 2>/dev/null | grep -m1 "^Version" | awk '{print $3}')
            else
                INSTALL_VER=$DEPS_INST_VER
            fi
        else
            if [[ -n $VIRTUAL_PACKAGE ]]; then
                INSTALL_VER=$(zypper info "$VIRTUAL_PACKAGE" 2>/dev/null | grep -m1 "^Version" | awk '{print $3}')
            else
                INSTALL_VER=$(zypper info $dep 2>/dev/null | grep -m1 "^Version" | awk '{print $3}')
            fi
        fi
    fi

    return $install_result
}

is_pkg_installable() {
    local dep="$1"
    local install_result=0
    
    if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
        is_pkg_installable_deb "$dep"
    else
        is_pkg_installable_rpm "$dep"
    fi
    
    install_result=$?
    
    return $install_result
}

is_pkg_deb_installed() {
    local dep="$1"
    local install_result=0
    
    print_str "is_pkg_deb_installed: dep = $dep" 4
    
    local status=$(dpkg-query -W -f'${Package}:${Architecture} ${Status}\n' "$dep" 2>/dev/null | grep -E ':amd64|:all' | awk '{print $4}')
    if [[ $status != "installed" ]]; then
    
        # if the package is not installed, check if it's installed via a virtual package
        dpkg-query -W -f='${Package} ${Provides}\n' '*' | grep $dep | grep -v "^$dep " > /dev/null 2>&1
        install_result=$?
        
        if [[ $install_result -eq 0 ]]; then
           echo "Virtual packages may be installed for $dep."
           
           check_virtual_package_deb "$dep"
           install_result=$?
        fi
        
    fi
    
    return $install_result
}

is_pkg_rpm_installed() {
    local dep="$1"
    local install_result=0

    print_str "is_pkg_rpm_installed : dep = $dep" 4

    # All RPM-based systems: use rpm --whatprovides for checking
    # This works reliably across RHEL, Rocky, AlmaLinux, and SLES
    local rpm_output=$(rpm -q --whatprovides "$dep" 2>&1)
    local rpm_result=$?

    # Check if the output indicates "no package provides"
    # On SLES, rpm can return exit code 0 even when no package provides the capability
    if echo "$rpm_output" | grep -qi "no package provides"; then
        print_str "rpm --whatprovides: no package provides $dep" 4
        install_result=1
    elif [[ $rpm_result -eq 0 ]] && [[ -n "$rpm_output" ]]; then
        print_str "Package found via rpm --whatprovides: $rpm_output" 2
        install_result=0
    else
        print_str "rpm --whatprovides failed or returned empty" 4
        install_result=1
    fi

    # If not found via whatprovides, try direct package name
    if [[ $install_result -ne 0 ]]; then
        print_str "Trying direct package name: rpm -q $dep" 4
        rpm -q "$dep" > /dev/null 2>&1
        install_result=$?

        if [[ $install_result -eq 0 ]]; then
            print_str "Package found via direct name" 2
        else
            # On SLES, try common naming variations for known libraries
            if [[ "$DISTRO_NAME" == "sles" ]]; then
                # Try appending '1' for library packages (libatomic -> libatomic1)
                if [[ "$dep" =~ ^lib ]]; then
                    local variant="${dep}1"
                    print_str "Trying SLES common variation: $variant" 4
                    rpm -q "$variant" > /dev/null 2>&1
                    install_result=$?

                    if [[ $install_result -eq 0 ]]; then
                        print_str "Package found via variation: $variant" 2
                    fi
                fi
            fi
        fi
    fi

    return $install_result
}

check_dep_installable() {
    echo --------------------------------------------------------------
    local testdep="$1"

    local installable=""
    local dep=""
    local dep_version=""
    local result=1

    # remove any leading space
    testdep=$(echo "$testdep" | sed 's/^ *//')
    echo -e "($INSTALL_DEPS_COUNT/$MISSING_DEPS_COUNT): Checking for available packages:\e[36m $testdep\e[0m"

    # check if the package is available for install using local cache
    if [[ -n $testdep ]]; then

        # check for a multi-option dependency
        if echo "$testdep" | grep -q '|'; then
            print_str "Processing multi-option dep..."

            # build the list of packages removing all spaces and replacing "|" with spaces
            deplist=$(echo "$testdep" | tr -d ' ' | tr '|' ' ')

            for pkg in $deplist; do
                print_str "+++++++++++++++++++++"
                print_str "Checking: $pkg"

                VIRTUAL_PACKAGE=""

                # remove versioning
                if [ $PACKAGE_TYPE == "deb" ]; then
                    dep=$(echo $pkg | sed -E 's/\([><!=]*[0-9.]*\)//g')
                    dep_version=$(echo "$pkg" | grep -oE '[0-9.]+')
                else
                    dep=$(echo $pkg | sed -E 's/[><=!]=?[0-9.]+//g')
                    dep_version=$(echo "$pkg" | grep -oE '[0-9.]+')
                fi
                print_str "Name/Ver: $dep | $dep_version"

                is_pkg_installable "$dep"
                result=$?

                if [[ $result -eq 0 ]]; then
                    echo -e "\e[32m$pkg installable : $INSTALL_VER\e[0m"
                    installable="$dep"
                    break;    # may want to select an amd package if available over any other
                else
                    echo $pkg cannot be installed.
                fi
            done

        else
            VIRTUAL_PACKAGE=""

            # single dependency check
            dep=$(echo "$testdep" | awk '{print $1}')

            is_pkg_installable "$dep"
            result=$?

            if [[ $result -eq 0 ]]; then
                echo -e "\e[32m$dep installable : $INSTALL_VER\e[0m"
                installable="$dep"
            else
                echo $pkg cannot be installed.
            fi
        fi

        # if any dependency is not installable, fail since the package will not be resolved
        if [[ $result -ne 0 ]]; then
            print_err "Dependency list cannot be installed."
            cleanup
            exit 1
        fi

        # check if a dep can be install via the virtual package/providing package
        if [[ -z $VIRTUAL_PACKAGE ]]; then
            INSTALL_LIST+="$installable "
        else
            echo Capability is provided by package: $VIRTUAL_PACKAGE. Adding.
            INSTALL_LIST+="$VIRTUAL_PACKAGE "
        fi
    else
        echo empty dep
    fi
}

compare_versions_deb() {
    local version1=$1
    local version2=$2
    
    if dpkg --compare-versions "$current_version" eq "$version"; then
        return 0  # Equal
    elif dpkg --compare-versions "$current_version" gt "$version"; then
        return 1  # Greater
    else
        return 2  # Less
    fi
}

compare_versions_rpm() {
    local version1=$1
    local version2=$2

    # Split the versions into arrays
    IFS='.' read -r -a ver1 <<< "$version1"
    IFS='.' read -r -a ver2 <<< "$version2"

    # Compare each part of the version numbers
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            # If ver2 is shorter and all previous parts are equal, ver1 is greater
            return 1
        elif ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1  # Greater
        elif ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2  # Less
        fi
    done

    # If ver1 is shorter and all previous parts are equal, ver1 is less
    if [[ ${#ver1[@]} -lt ${#ver2[@]} ]]; then
        return 2
    fi

    return 0  # Equal
}

check_dep_version_installed() {
    local dep="$1"
    local dep_status=1
    
    print_str "check_dep_version_installed: dep = $dep" 0
    
    if echo "$dep" | grep -q ' (>= [0-9]\+\.[0-9]\+\.[0-9]\+' || \
       echo "$dep" | grep -q ' (>= [0-9]\+\.[0-9]\+' || \
       echo "$dep" | grep -q ' (>= [0-9]\+' || \
       echo "$dep" | grep -q '>= [0-9]\+\.[0-9]\+\.[0-9]\+' || \
       echo "$dep" | grep -q '>= [0-9]\+\.[0-9]\+' || \
       echo "$dep" | grep -q '>= [0-9]\+'; then
       
        # check if the version dep installed        
        dep_name=$(echo "$dep" | awk '{print $1}')
        dep_ver=$(echo "$dep" | awk '{print $3}' | sed 's/)//')
        
        # first query if the dep is installed
        if [ $PACKAGE_TYPE == "deb" ]; then
            is_pkg_deb_installed "$dep_name"
        else
            # check what package provides the dep on the system
            is_pkg_rpm_installed "$dep_name"
        fi
        dep_status=$?
        
        if [ $dep_status -eq 0 ]; then
            # the dep is currently installed, get the version
            if [ $PACKAGE_TYPE == "deb" ]; then
                current_version=$(dpkg-query -W "$dep_name" | head -1 | awk '{print $2}' | cut -d '-' -f 1 | sed 's/[.+][^.+]*$//')
            else
                current_version=$(rpm -q --queryformat '%{VERSION}' "$dep_name")
            fi
            dep_status=$?
            
            # check for an error on rpm -q : an error may be due a virtual package
            if [[ $dep_status -ne 0 ]] && [[ $PACKAGE_TYPE == "rpm" ]]; then
                dep_name_v=$(rpm -q --whatprovides $dep_name)
                echo "$dep_name -> $dep_name_v <v>"
                
                # update the name to the virtual name and get the version
                dep_name=$(echo "$dep_name_v" | awk '{print $1}')
                current_version=$(rpm -q --queryformat '%{VERSION}' "$dep_name")
                dep_status=$?
            fi
            
            print_str "$dep_name is currently installed : version = $current_version" 2
            print_str "dep_name        = $dep_name" 1
            print_str "dep_status      = $dep_status" 1
            print_str "dep version     = $dep_ver " 1
            print_str "current version = $current_version " 1
                   
            # Compare the current version with the required version
            if [ $PACKAGE_TYPE == "deb" ]; then
                compare_versions_deb  "$current_version" "$dep_ver"
            else
                compare_versions_rpm "$current_version" "$dep_ver"
            fi
            comparison_result=$?

            if [[ $comparison_result -eq 0 ]]; then
                print_str "    current_version $current_version is equal"
            elif [[ $comparison_result -eq 1 ]]; then
                print_str "    current_version $current_version is greater"
            else
                 echo -e "\e[31m$dep_name current_version $current_version is less\e[0m"
                dep_status=1
            fi
        else
            print_str "$dep is not installed" 3
        fi
        
    else
        # check if non-version dep installed
        if [[ -n $dep ]]; then
            if [ $PACKAGE_TYPE == "deb" ]; then
                is_pkg_deb_installed "$dep"
            else
                is_pkg_rpm_installed "$dep"
            fi
            dep_status=$?
            
            if [ $dep_status -eq 0 ]; then
                print_str "$dep is installed" 2
            else
                print_str "$dep is not installed" 3
            fi
        fi
    fi
    
    return $dep_status
}

check_installed_kernel_packages() {
    echo "------------------------------------"
    echo Checking Kernel Packages...
    
    for pkg in $KERNEL_PACKAGES; do
        if [[ -n "$pkg" ]]; then
            check_dep_version_installed "$pkg"
            status=$?
        
            if [ $status -eq 0 ]; then
                echo -e "$pkg : \e[32mINSTALLED\e[0m"
            else
                if [[ "$MISSING_DEPS" != *"$pkg"* ]]; then
                    echo -e "$pkg : \e[31mNOT INSTALLED\e[0m"
                    MISSING_DEPS+="$pkg, "
                    MISSING_DEPS_COUNT=$((MISSING_DEPS_COUNT+1))
                fi
            fi
            DEPS_COUNT=$((DEPS_COUNT+1))
        fi
    done
    
    echo Checking Kernel Packages...Complete.
}

check_installed_dep_packages() {
    # Check if each package dep in dependency file is installed on the system
    while IFS= read -r pkg; do
        print_str "+++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print_str "$pkg : ..." 1
        
        DEPS_COUNT=$((DEPS_COUNT+1))
        DEPS+="$pkg "
        
        if [[ "$pkg" == *'|'* ]]; then
            status=1
            IFS='|' read -ra deps <<< "$pkg"
            for dep in "${deps[@]}"; do
                dep_trimmed=$(echo "$dep" | sed 's/^ *//;s/ *$//')
                check_dep_version_installed "$dep_trimmed"
                if [[ $? -eq 0 ]]; then
                    status=0
                    break
                fi
            done
            
        else
            # check a single dep
            check_dep_version_installed "$pkg"
            status=$?
        fi
        
        # Check the dep install status and add any missing deps not installed on the system to the "missing" list
        if [ $status -eq 0 ]; then
            echo -e "$pkg : \e[32mINSTALLED\e[0m"
        else
            echo -e "$pkg : \e[31mNOT INSTALLED\e[0m"
            MISSING_DEPS+="$pkg, "
            MISSING_DEPS_COUNT=$((MISSING_DEPS_COUNT+1))
        fi
    done < "$DEPS_FILE"
    
    # Check if any kernel packages are installed on the system (if required)
    check_installed_kernel_packages
}

remove_deps_duplicates() {
    echo Removing duplicate deps...
    
    if [[ -n $FILTER_PACKAGES ]]; then
        for dep in $FILTER_PACKAGES; do
            # Remove any dependency in MISSING_DEPS that matches
            if [[ "$MISSING_DEPS" == *"$dep"* ]]; then
                echo Removing: "$dep"
                MISSING_DEPS=$(echo "$MISSING_DEPS" | sed -E "s/(^|, )$dep(, |$)/\1/g")
                MISSING_DEPS_COUNT=$((MISSING_DEPS_COUNT-1))
                DEPS_COUNT=$((DEPS_COUNT-1))
            fi
        done
    fi
    
    echo Removing duplicate deps...Complete.
}

remove_deps() {
    echo Removing deps...
    
    if [[ -n $REMOVE_PACKAGES ]]; then
        for dep in $REMOVE_PACKAGES; do
            # Remove any dependency in MISSING_DEPS that contains the match
            if [[ "$MISSING_DEPS" == *"$dep"* ]]; then
                echo Removing: "$dep"
                MISSING_DEPS=$(echo "$MISSING_DEPS" | tr ',' '\n' | grep -v "$dep" | tr '\n' ',' | sed 's/,$//')
                MISSING_DEPS_COUNT=$((MISSING_DEPS_COUNT-1))
                DEPS_COUNT=$((DEPS_COUNT-1))
            fi
        done
    fi

    echo Removing deps...Complete.
}

install_rocky_kernel_packages() {
    echo Downloading Rocky kernel packages...
    
    # Set base url to rocky vault/kickstart repo and attempt to download the kernel packages
    local base_url="$1"

    # Set the kernel packages to download
    local packages=(
        "$base_url/Packages/k/kernel-headers-$KERNEL_VER.rpm"
        "$base_url/Packages/k/kernel-devel-$KERNEL_VER.rpm"
        "$base_url/Packages/k/kernel-devel-matched-$KERNEL_VER.rpm"
    )

    local failed=0
    local package_name=
    local package_list=
    
    echo --------------------------
    echo "URL: $base_url"
    echo --------------------------

    # Loop through the list of packages and attempt to download each
    
    for package in "${packages[@]}"; do
        package_name=$(basename "${package}")
        echo "Downloading: $package_name"
        wget -q "$package" -O "$package_name"
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mFailed to download kernel package: $package\e[0m"
            failed=1
            break
        fi
    done

    # If any download failed, return failure
    if [[ $failed -eq 1 ]]; then
        echo -e "\e[31mOne or more kernel packages failed to download. Exiting.\e[0m"
        return 1
    fi
    
    # Loop through the downloaded packages and install them
    echo "Installing downloaded packages..."
    
    for package in "${packages[@]}"; do
        package_list+="./$(basename "${package}") "
    done
    
    echo "Installing: $package_list"
    
    $SUDO dnf install -y $package_list
    if [[ $? -ne 0 ]]; then
        echo -e "\e[31mFailed to install kernel packages: $package_list\e[0m"
        $SUDO rm $package_list
        return 1
    fi
    
    # Clean up any downloaded packages
    $SUDO rm $package_list

    echo Downloading Rocky kernel packages...Complete.
    return 0
}

add_rocky_repo() {
    local rocky_repo_url="$1"
    local rocky_repo_name="$2"
    local rocky_repo_desc="$3"
    
    echo "Adding URL: $rocky_repo_url"
    
     # Check if the repo is accessible
    wget --spider "$rocky_repo_url/repodata" > /dev/null 2>&1
    
    if [[ $? -eq 0 ]]; then
        echo -e "\e[32mRepo $rocky_repo_desc accessible\e[0m"
cat <<EOF | $SUDO tee -a /etc/yum.repos.d/appstream-amdgpu.repo
[$rocky_repo_name]
name=Rocky Linux $DISTRO_VER - $rocky_repo_desc
baseurl=$rocky_repo_url
gpgcheck=1
enabled=1
countme=1
metadata_expire=6h
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-Rocky-$DISTRO_MAJOR_VER
EOF
    else
        echo -e "\e[31mRepo $rocky_repo_desc not accessible\e[0m"
        return 1
    fi
    
    # Cleanup the dnf caches
    $SUDO dnf clean all
    $SUDO rm -rf /var/cache/dnf/*
    $SUDO dnf makecache
    
    return 0
}

get_kernel_pacakges_repo_rocky() {
    local package="kernel-headers-$KERNEL_VER.rpm"
    
    # Repo URL list for kernel packages
    local base_urls=(
        "https://dl.rockylinux.org/pub/rocky/$DISTRO_VER/AppStream/x86_64/kickstart"
        "https://dl.rockylinux.org/vault/rocky/$DISTRO_VER/AppStream/x86_64/kickstart"
        "https://dl.rockylinux.org/vault/rocky/$DISTRO_VER/AppStream/x86_64/os"
    )
    
    local pkg_avail=0
    
    # Check each repo for the required kernel packages
    for url in "${base_urls[@]}"; do
        wget --spider "$url/Packages/k/$package" > /dev/null 2>&1
    
        if [[ $? -eq 0 ]]; then
            echo -e "Packages in $url : \e[32mAvailable.\e[0m"
            pkg_avail=1
            break
        else
            echo -e "Packages in $url : \e[93mNot Available.\e[0m"
        fi
    done
    
    if [[ $pkg_avail == 0 ]]; then
        print_err "Kernel packages not found in repos."
        return 1
    fi
    
    local dir_subtype=appstream
    local dir_type=
    local dir_package=
    
    if [[ $url =~ "vault" ]]; then
        dir_type=vault
    elif [[ $url =~ "pub" ]]; then
        dir_type=pub
    fi
    
    if [[ $url =~ "x86_64/kickstart" ]]; then
        dir_package=kickstart
    elif [[ $url =~ "x86_64/os" ]]; then
        dir_package=os
    fi
    
    # Attempt to add the repo for the kernel packages
    add_rocky_repo "$url" "$dir_subtype-$dir_type-$dir_package" "$dir_subtype $dir_type $dir_package"
    if [[ $? -eq 0 ]]; then
        KERNEL_PACKAGES_VER="-$KERNEL_VER"
    else
        echo "Repo not available. Downloading package."
        
        # The repo may not be accessible, so manually download and install from the repo
        install_rocky_kernel_packages "$url"
        if [[ $? -ne 0 ]]; then
            print_err "Unable to download and install kernel packages."
            return 1
        fi
    fi
    
    return 0
}

get_kernel_packages_rocky() {
    echo Rocky kernel packages...
    
    remove_rocky_kernel_repo
    
    dnf repoquery --available --queryformat "%{name}-%{version}-%{release}.%{arch}" | grep kernel-headers-$(uname -r)
    if [ $? -eq 0 ]; then
        echo "Kernel Packages for $KERNEL_VER are available in the AppStream repositories."
        KERNEL_PACKAGES_VER="-$KERNEL_VER"
    else
        echo -e "\e[93mKernel Packages not available in the AppStream repositories.\e[0m"
        
        # check for legacy kernel headers
    	get_kernel_pacakges_repo_rocky
    	if [ $? -eq 1 ]; then
    	    echo -e "\e[93mKernel Packages not available in the repositories.  Using defaults.\e[0m"
    	fi
    fi
    
    echo Rocky kernel packages...Complete.
}

get_kernel_packages_el() {
    echo EL kernel packages...

    if [[ $DEPS_LIST_ONLY == 0 ]]; then
        dnf list "kernel-headers-$KERNEL_VER" &> /dev/null
        if [ $? -eq 0 ]; then
            echo "Kernel Packages for $KERNEL_VER are available in the repositories."
            KERNEL_PACKAGES_VER="-$KERNEL_VER"
        else
            if [[ "$DISTRO_NAME" = "rocky" ]]; then
                get_kernel_packages_rocky
            else
                echo -e "\e[93mKernel Packages not available in the repositories.  Using defaults.\e[0m"
                KERNEL_PACKAGES_VER="-$KERNEL_VER"
            fi
        fi
    else
        KERNEL_PACKAGES_VER="-$KERNEL_VER"
    fi
    
    KERNEL_PACKAGES="kernel-headers$KERNEL_PACKAGES_VER kernel-devel$KERNEL_PACKAGES_VER "
    
    if [[ $DISTRO_VER == 9* ]]; then
        echo Adding EL9 amdgpu packages
        KERNEL_PACKAGES+="kernel-devel-matched$KERNEL_PACKAGES_VER "
    elif [[ $DISTRO_VER == 10* ]]; then
        echo Adding EL10 amdgpu packages
        KERNEL_PACKAGES+="kernel-devel-matched$KERNEL_PACKAGES_VER "
    fi
    
    FILTER_PACKAGES="kernel-devel"
    REMOVE_PACKAGES="kernel-headers"
}

get_kernel_type_ol() {
    local kernel_version=$(uname -r)

    if [[ $kernel_version == *"uek"* ]]; then
        echo "UEK"
    elif [[ $kernel_version == *"el8"* || $kernel_version == *"el9"* || $kernel_version == *"el10"* ]]; then
        echo "RHCK"
    else
        echo "UNKNOWN"
    fi
}

get_kernel_packages_ol() {
    echo OL kernel packages...

    kernel_type=$(get_kernel_type_ol)
    echo "Current kernel type: $kernel_type"

    if [[ $kernel_type == "UEK" ]]; then

        # check for the uek kernel packages
        dnf list "kernel-uek-devel-$KERNEL_VER" &> /dev/null
        if [ $? -eq 0 ]; then
            echo "Kernel Packages for UEK $KERNEL_VER are available in the repositories."
            KERNEL_PACKAGES+="kernel-uek-devel-$KERNEL_VER "
        else
            echo -e "\e[93mKernel Packages for UEK not available in the repositories.  Using defaults.\e[0m"
        fi
    elif [[ $kernel_type == "RHCK" ]]; then
        # check for the rhck kernel packages
        dnf list "kernel-headers-$KERNEL_VER" &> /dev/null
        if [[ $? -eq 0 ]]; then
            echo "Kernel Packages for RHCK $KERNEL_VER are available in the repositories."
            KERNEL_PACKAGES+="kernel-headers-$KERNEL_VER kernel-devel-$KERNEL_VER "
            if [[ $DISTRO_VER == 9* ]]; then
                echo Adding EL9 amdgpu packages
                KERNEL_PACKAGES+="kernel-devel-matched-$KERNEL_VER "
            elif [[ $DISTRO_VER == 10* ]]; then
                echo Adding EL10 amdgpu packages
                KERNEL_PACKAGES+="kernel-devel-matched-$KERNEL_VER "
            fi
        else
            echo -e "\e[93mKernel Packages for RHCK not available in the repositories.  Using defaults.\e[0m"
        fi
    else
            echo -e "\e[93mUnknown kernel type. Using defaults.\e[0m"
            KERNEL_PACKAGES+="kernel-uek-devel-$KERNEL_VER "
    fi
    
    FILTER_PACKAGES="kernel-devel"
    REMOVE_PACKAGES="kernel-headers"
    
    if [ -f "/boot/config-$(uname -r)" ]; then
        echo "Find the value of TARGET_GCC_VERSION using CONFIG_CC_VERSION_TEXT from /boot/config-$(uname -r)"
        TARGET_GCC_VERSION=$($SUDO cat /boot/config-$(uname -r) | grep CONFIG_CC_VERSION_TEXT | cut -d '=' -f2 | awk -F " " '{print $NF}' | tr -d ')' | tr -d '"')
        for gcc_package in ${GCC_TOOLSET_PACKAGES_OL[@]}; do
            # Expect full package name we want to install
            # Example: gcc-toolset-11-gcc-11.4.1-3.0.1.el8_6
            if [[ $DISTRO_VER == 8* ]]; then
                gcc_package_ver=$(sudo dnf --disablerepo="*" --enablerepo="ol8_appstream" repoquery --all --nvr | grep "$gcc_package-$TARGET_GCC_VERSION" | awk '{print $NF}' | sort | uniq | tail -1)
            elif [[ $DISTRO_VER == 9* ]]; then
                gcc_package_ver=$(sudo dnf --disablerepo="*" --enablerepo="ol9_appstream" repoquery --all --nvr | grep "$gcc_package-$TARGET_GCC_VERSION" | awk '{print $NF}' | sort | uniq | tail -1)
            elif [[ $DISTRO_VER == 10* ]]; then
                gcc_package_ver=$(sudo dnf --disablerepo="*" --enablerepo="ol10_appstream" repoquery --all --nvr | grep "$gcc_package-$TARGET_GCC_VERSION" | awk '{print $NF}' | sort | uniq | tail -1)
            fi

            if [ ! -z "$gcc_package_ver" ]; then
                echo "Install $gcc_package version $gcc_package_ver"
                KERNEL_PACKAGES+="$gcc_package_ver "
            else
                echo "Unable to gcc version $TARGET_GCC_VERSION for package $gcc_package in repo ol${DISTRO_VER_MAJ}_appstream"
            fi
        done
    fi
}

get_kernel_packages_amzn() {
    echo Amazon kernel packages...
    
    # Extract version major.minor version (x.y)
    KERNEL_MAJ_MIN=$(echo "$KERNEL_VER" | cut -d'.' -f1,2)
    
    # Extract version up to .x86_64
    KERNEL_VER_AMZN=$(echo "$KERNEL_VER" | sed 's/\.x86_64$//')
    
    echo "KERNEL_MAJ_MIN : $KERNEL_MAJ_MIN"
    echo "KERNEL_VER_AMZN: $KERNEL_VER_AMZN"

    if [[ $DEPS_LIST_ONLY == 0 ]]; then
        dnf list "kernel$KERNEL_MAJ_MIN-headers-$KERNEL_VER_AMZN"
        if [ $? -eq 0 ]; then
            echo "Kernel Packages for $KERNEL_VER_AMZN are available in the repositories."
        else
            print_err "Kernel Packages not available in the repositories."
            exit 1
        fi
    fi
    
    KERNEL_PACKAGES="kernel$KERNEL_MAJ_MIN-headers-$KERNEL_VER_AMZN kernel$KERNEL_MAJ_MIN-devel-$KERNEL_VER_AMZN "
    
    FILTER_PACKAGES="kernel-devel"
    REMOVE_PACKAGES="kernel-headers"
}

get_kernel_package_for_kernel_version() {
    print_str "--------------------------------"
    print_str "Find kernel package $1 for kernel version $KERNEL_PACKAGE_VER..."
    
    kernel_package="$1"
    
    local output=$($SUDO zypper search -s "$kernel_package" | grep $KERNEL_PACKAGE_VER)
    for col_value in ${output}; do
        grep -q $KERNEL_PACKAGE_VER <<< "$col_value"
        if [ $? -eq 0 ]; then
            NEW_KERNEL_PACKAGE_VER="$col_value"
            echo "Using $NEW_KERNEL_PACKAGE_VER for $kernel_package"
            KERNEL_PACKAGES+="$kernel_package-$NEW_KERNEL_PACKAGE_VER "
            break;
        fi
    done
}

get_kernel_packages() {
    echo "------------------------------------"

    # set the kernel headers
    KERNEL_VER=$(uname -r)
    echo "Kernel: ${KERNEL_VER}"

    echo "Availaible kernels:"
    ls -1 /boot/vmlinuz*

    # set the kernel packages
    if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
        KERNEL_PACKAGES="linux-headers-$KERNEL_VER "

    elif [ $DISTRO_PACKAGE_MGR == "dnf" ]; then

        if [[ "$DISTRO_NAME" = "rhel" || "$DISTRO_NAME" = "rocky" ]]; then
            get_kernel_packages_el
        elif [ "$DISTRO_NAME" = "ol" ]; then
            get_kernel_packages_ol
        elif [ "$DISTRO_NAME" = "amzn" ]; then
            get_kernel_packages_amzn
        else
            get_kernel_packages_el
        fi

    elif [ $DISTRO_PACKAGE_MGR == "zypper" ]; then
        if [[ $DEPS_LIST_ONLY == 0 ]]; then
            KERNEL_PACKAGE_VER="$(uname -r | sed "s/-default//") "
            $SUDO zypper search -s kernel-default-devel | grep $KERNEL_PACKAGE_VER &> /dev/null
            if [ $? -eq 0 ]; then
                echo "Kernel Packages for $KERNEL_PACKAGE_VER are available in the repositories."

                get_kernel_package_for_kernel_version "kernel-default-devel"
                get_kernel_package_for_kernel_version "kernel-syms"
                get_kernel_package_for_kernel_version "kernel-macros"
            else
                echo -e "\e[93mKernel Packages not available in the repositories.  Using defaults.\e[0m"
                KERNEL_PACKAGES="kernel-default-devel"
            fi
        else
            KERNEL_PACKAGES="kernel-default-devel"
        fi

        FILTER_PACKAGES="kernel-devel"

    else
        print_err "Unsupported OS."
        exit 1
    fi

    echo "KERNEL_PACKAGES = $KERNEL_PACKAGES"
}

print_deps() {
    if [ -f deps_list.txt ]; then
        echo Removing old missing dependency list.
        rm deps_list.txt
    fi
        
    if [[ -n $MISSING_DEPS ]]; then
        echo ==============================================================
        echo "Dependencies: $MISSING_DEPS_COUNT of $DEPS_COUNT packages require install:"
        echo $MISSING_DEPS | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u
        echo $MISSING_DEPS | tr ',' '\n' | awk 'NF' | sed 's/^[ \t]*//;s/[ \t]*$//' | sort -u > deps_list.txt
    fi
}

read_deps() {
    print_str "--------------------------------"
    print_str "Read Dependency Configuration: $DEPS_FILE ..."
    
    if [ ! -f $DEPS_FILE ]; then
        print_err "$DEPS_FILE does not exist."
        exit 1
    fi
    
    while IFS= read -r dep; do
        MISSING_DEPS="$MISSING_DEPS $dep,"
        DEPS_COUNT=$((DEPS_COUNT+1))
    done < "$DEPS_FILE"
    
    # add any kernel packages to the deps list
    if [[ -n "$KERNEL_PACKAGES" ]]; then
        for dep in $KERNEL_PACKAGES; do
             MISSING_DEPS="$MISSING_DEPS $dep,"
             DEPS_COUNT=$((DEPS_COUNT+1))
        done
    fi
    
    MISSING_DEPS_COUNT=$DEPS_COUNT
    
    print_str "Read Dependency Configuration...Complete."
}

build_dependencies_list_for_compo() {
    local compo_type=$1
    echo "------------------------------------"
    print_str "Building Dependency List for: $compo_type"

    # Set the dependency file based on component type
    if [[ "$compo_type" == "rocm" ]]; then
        # For ROCm, use the combined dependency file from appropriate component directory
        # Select the appropriate directory and dependency file based on package type
        if [ "$PACKAGE_TYPE" == "rpm" ]; then
            local deps_dir="$PWD/component-rocm"
            DEPS_FILE="$deps_dir/rocm_required_deps_rpm.txt"
        else
            # For DEB packages, check if component-rocm-deb exists (chroot mode)
            # Otherwise fall back to component-rocm (native DEB system)
            if [ -d "$PWD/component-rocm-deb" ]; then
                local deps_dir="$PWD/component-rocm-deb"
            else
                local deps_dir="$PWD/component-rocm"
            fi
            DEPS_FILE="$deps_dir/rocm_required_deps_deb.txt"
        fi

        if [ ! -d "$deps_dir" ]; then
            print_err "ROCm component directory does not exist: $deps_dir"
            exit 1
        fi

    elif [[ "$compo_type" == "amdgpu" ]]; then
        # For AMDGPU, use the distro-specific dependency file
        local amdgpu_base_dir="$PWD/component-amdgpu"

        if [ ! -d "$amdgpu_base_dir" ]; then
            print_err "component-amdgpu directory does not exist: $amdgpu_base_dir"
            exit 1
        fi

        # Get the distro tag for the current running distro
        AMDGPU_DISTRO_TAG=$(get_amdgpu_distro_tag)
        if [ $? -ne 0 ]; then
            print_err "Failed to determine AMDGPU distro tag"
            exit 1
        fi

        print_str "AMDGPU_DISTRO_TAG = $AMDGPU_DISTRO_TAG"

        DEPS_FILE="$amdgpu_base_dir/$AMDGPU_DISTRO_TAG/required_deps.txt"

    else
        print_err "Unknown component type: $compo_type"
        exit 1
    fi

    print_str "DEPS_FILE = $DEPS_FILE"

    if [ ! -f "$DEPS_FILE" ]; then
       print_err "Dependency file does not exist: $DEPS_FILE"
       exit 1
    fi

    if [[ $DEPS_LIST_ONLY == 1 ]]; then
        read_deps
    else
        # Build the the list of dependencies that are missing and require install
        check_installed_dep_packages
    fi
}

build_dependencies_list() {

     # build the list of missing deps for all rocm-based components
    if [[ $USE_ROCM == 1 ]]; then
        build_dependencies_list_for_compo "rocm"
    fi

    # build the list of missing deps for all amdgpu-based components
    if [[ $USE_AMDGPU == 1 ]]; then

        # get the list of kernel packages required by amdgpu
        get_kernel_packages

        # remove any kernel packages from the current deps list that are not required
        remove_deps

        build_dependencies_list_for_compo "amdgpu"

        # filter out any duplicate packages
        remove_deps_duplicates
    fi

    print_deps
}

setup_epel_crb() {
   # Setup for installing EL repos
    local epel_pkg="epel-release-latest-$DISTRO_MAJOR_VER.noarch.rpm"
    local codeready_repo="codeready-builder-for-rhel-$DISTRO_MAJOR_VER-x86_64-rpms"
    
    # Setup EPEL/crb if required
    if [ -f /etc/yum.repos.d/epel.repo ]; then
        echo "EPEL repo exists."
        
    else
        echo "EPEL repo setup for EL $DISTRO_MAJOR_VER."
        
        wget --tries 5 https://dl.fedoraproject.org/pub/epel/$epel_pkg
        if [ $? -ne 0 ]; then
            print_err "Unsupported version for EPEL."
            exit 1
        fi
        $SUDO rpm -ivh $epel_pkg
        
        echo "EPEL repo setup...Complete."
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

install_repos_el() {
    echo "------------------------------------"
    echo "Setting up Repos..."
    
    # Install wget if required
    if ! rpm -q "wget" > /dev/null 2>&1; then
        echo "Package: wget Installing..."
        $SUDO dnf install -y wget > /dev/null 2>&1
        echo "Package: wget installed."
    fi
    
    # Install dnf-plugins-core if required
    if ! rpm -q "dnf-plugins-core" > /dev/null 2>&1; then
        echo "Package: dnf-plugins-core Installing..."
        $SUDO dnf install -y dnf-plugins-core > /dev/null 2>&1
        echo "Package: dnf-plugins-core installed."
    fi
    
    $SUDO dnf install -y dnf-plugin-config-manager
    
    if [[ $EPEL_SETUP == 1 ]]; then
        setup_epel_crb
    fi
    
    echo "Setting up Repos...Complete."
    echo "------------------------------------"
}

install_dependencies() {
    echo ==============================================================
    echo Installing Dependencies...
    
    INSTALL_LIST=
    
    # remove trailing space
    MISSING_DEPS="${MISSING_DEPS% }"
    
    if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
        echo Updating apt cache.
        $SUDO apt-get update > /dev/null 2>&1
        
    elif [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
        # cleanup the dnf cache
        echo Clean dnf cache.
        $SUDO dnf clean all
        $SUDO rm -rf /var/cache/dnf/*
        
        # add the required repos for el
        install_repos_el
        
        echo Updating dnf cache.
        $SUDO dnf makecache > /dev/null 2>&1
        
        # build a cache of installable packages based on the missing deps
        build_installable_pkg_cache_dnf

    else
        echo Updating zypper cache.
        $SUDO zypper refresh > /dev/null 2>&1
    fi
    
    # test if each dependency is available for install 
    IFS=',' read -ra package_array <<< "$MISSING_DEPS"
    for pkg in "${package_array[@]}"; do
        INSTALL_DEPS_COUNT=$((INSTALL_DEPS_COUNT+1))
        check_dep_installable "$pkg"
    done
    
    local installopt="-y"
    
    if [[ $PROMPT_USER == 1 ]]; then
        installopt=
    fi
    
    if [[ $VERBOSE == 1 ]]; then
        echo -----------------------------------------------
        echo Installing the following based on availability:
        echo $INSTALL_LIST
        echo -----------------------------------------------
    fi
    
    if [[ -n $INSTALL_LIST ]]; then
        # install the dependent packages
        if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
            $SUDO apt-get install "$installopt" $INSTALL_LIST
        elif [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
            $SUDO dnf install $installopt $INSTALL_LIST
        else
            $SUDO zypper install --oldpackage "$installopt" $INSTALL_LIST
        fi
    fi
    
    cleanup
    
    print_no_err "Dependencies Installed."
}

install_dependencies_file() {
    echo ==============================================================
    echo Installing Dependencies : File...
    
    read_deps
    print_deps
    
    install_dependencies
    
    echo Installing Dependencies : File...Complete.
}

####### Main script ###############################################################

echo =================================
echo DEPENDENCY INSTALLER
echo =================================

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
        echo "Using amdgpu extract."
        USE_AMDGPU=1
        shift
        ;;
    rocm)
        echo "Using rocm extract."
        USE_ROCM=1
        shift
        ;;
    list)
        echo "Printing list of dependencies."
        DEPS_LIST_ONLY=1
        shift
        ;;
    install)
        echo "Enable dependency install."
        INSTALL_DEPS=1
        shift
        ;;
    install-file)
        echo "Enable dependency install."
        DEPS_FILE="$2"
        shift
        ;;
    verbose)
        echo "Enabling verbose logging."
        VERBOSE=1
        NO_CMD_OUTPUT=""
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Check if a dependency files was provided for the install
if [[ -n $DEPS_FILE ]]; then
    echo installing deps from file: "$DEPS_FILE"
    install_dependencies_file
    exit 0
fi

if [[ $USE_AMDGPU == 0 ]] && [[ $USE_ROCM == 0 ]]; then
    print_err "Missing argument for components directory."
    exit 1
fi

# Build the list of dependencies for the given components directories
build_dependencies_list

if [[ $INSTALL_DEPS == 1 ]]; then
    install_dependencies
fi

exit 0

