#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

declare -a input_args
input_args="$@"

#use readlink rather than realpath for CentOS 6.10 support
HIPBLAS_SRC_PATH=`dirname "$(readlink -m $0)"`

/bin/ln -fs ../../.githooks/pre-commit "$(dirname "$0")/.git/hooks/"

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles|opensuse-leap)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora\n"
        exit 2
        ;;
  esac
}

check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root apt-get -y --no-install-recommends install ${package_dependencies}
}

install_apt_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manager \033[0m\n"
    elevate_if_not_root apt install -y --no-install-recommends ${package_dependencies[$index]}=${package_versions[$index]}
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root yum -y --nogpgcheck install ${package_dependencies}
}

install_yum_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manage
r \033[0m\n"
    elevate_if_not_root yum -y --nogpgcheck install ${package_dependencies[$index]}-${package_versions[$index]}
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root dnf install -y ${package_dependencies}
}

install_dnf_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manage
r \033[0m\n"
    elevate_if_not_root dnf install -y ${package_dependencies[$index]}-${package_versions[$index]}
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root zypper install -y ${package_dependencies}
}

install_zypper_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manage
r \033[0m\n"
    elevate_if_not_root zypper -n --no-gpg-checks install ${package_dependencies[$index]}-${package_versions[$index]}
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  if [ -z ${build_clients+foo} ]; then
    printf "install_packages(): \$build_clients must be set\n"
    exit 2
  fi

  # dependencies needed for library and clients to build
  local library_dependencies_ubuntu=( "make" "pkg-config" )
  local library_dependencies_centos_rhel=( "epel-release" "make" "gcc-c++" "rpm-build" )
  local library_dependencies_centos_rhel_8=( "epel-release" "make" "gcc-c++" "rpm-build" )
  local library_dependencies_fedora=( "make" "gcc-c++" "libcxx-devel" "rpm-build" )
  local library_dependencies_sles=( "make" "gcc-c++" "rpm-build" )

  if [[ $HIP_PLATFORM == "nvidia" ]]; then
    # Ideally, this could be cuda-cublas-dev, but the package name has a version number in it
    library_dependencies_ubuntu+=( "" ) # removed, use --installcuda option to install cuda
  else
    # Custom rocblas installation
    # Do not install rocblas if --rocblas_path flag is set,
    # as we will be building against our own rocblas instead.
    if [[ -z ${rocblas_path+foo} ]]; then
      if [[ -z ${custom_rocblas+foo} ]]; then
        # Install base rocblas package unless -b/--rocblas flag is passed
        library_dependencies_ubuntu+=( "rocblas-dev" )
        library_dependencies_centos_rhel+=( "rocblas-devel" )
        library_dependencies_centos_rhel_8+=( "rocblas-devel" )
        library_dependencies_fedora+=( "rocblas-devel" )
        library_dependencies_sles+=( "rocblas-devel" )
      else
        # Install rocm-specific rocblas package
        library_dependencies_ubuntu+=( "${custom_rocblas}" )
        library_dependencies_centos_rhel+=( "${custom_rocblas}" )
        library_dependencies_centos_rhel_8+=( "${custom_rocblas}" )
        library_dependencies_fedora+=( "${custom_rocblas}" )
        library_dependencies_sles+=( "${custom_rocblas}" )
      fi
    fi

    # Do not install rocsolver if --rocsolver_path flag is set,
    if [[ -z ${rocsolver_path+foo} ]]; then
      if [[ "${build_solver}" == true ]]; then
        library_dependencies_ubuntu+=( "rocsolver-dev" )
        library_dependencies_centos_rhel+=( "rocsolver-devel" )
        library_dependencies_centos_rhel_8+=( "rocsolver-devel" )
        library_dependencies_fedora+=( "rocsolver-devel" )
        library_dependencies_sles+=( "rocsolver-devel" )
      fi
    fi
  fi

  # wget is needed for cmake
  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
    if $update_cmake == true; then
      library_dependencies_ubuntu+=("wget")
      library_dependencies_centos_rhel+=("wget")
      library_dependencies_centos_rhel_8+=("wget")
      library_dependencies_fedora+=("wget")
      library_dependencies_sles+=("wget")
    fi
  fi

  if [[ "${build_clients}" == true ]]; then
    library_dependencies_ubuntu+=( "gfortran" )
    library_dependencies_centos_rhel+=( "devtoolset-7-gcc-gfortran" )
    library_dependencies_centos_rhel_8+=( "gcc-gfortran" )
    library_dependencies_fedora+=( "gcc-gfortran" )
    library_dependencies_sles+=( "gcc-fortran pkg-config" "dpkg" )
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      if (( "${VERSION_ID%%.*}" >= "8" )); then
        install_yum_packages "${library_dependencies_centos_rhel_8[@]}"
      else
        install_yum_packages "${library_dependencies_centos_rhel[@]}"
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"
      ;;

    sles|opensuse-leap)
#     elevate_if_not_root zypper -y update
      if (( "${VERSION_ID%%.*}" >= "15" )); then
        library_dependencies_sles+=( "libcxxtools10" )
      else
        library_dependencies_sles+=( "libcxxtools9" )
      fi
      install_zypper_packages "${library_dependencies_sles[@]}"
      ;;
    *)
      echo "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

install_cuda_package()
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  local cuda_dependencies=("cuda")
  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_apt_packages "${cuda_dependencies[@]}"
      else
        install_apt_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    centos|rhel)
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_yum_packages "${cuda_dependencies[@]}"
      else
        install_yum_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    fedora)
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_dnf_packages "${cuda_dependencies[@]}"
      else
        install_dnf_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    sles|opensuse-leap)
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_zypper_packages "${cuda_dependencies[@]}"
      else
        install_zypper_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    *)
      echo "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# given a relative path, returns the absolute path
make_absolute_path( ) {
  (cd "$1" && pwd -P)
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: all is well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

function display_help()
{
cat <<EOF

  hipBLAS library build & installation helper script.

  Usage:
    $0 (build hipblas and put library files at ./build/hipblas-install)
    $0 <options> (modify default behavior according to the following flags)

  Options:

    -b, --rocblas <version>       Specify rocblas version (e.g. 2.42.0).

    -c, --clients                 Build the library clients benchmark and gtest.
                                  (Generated binaries will be located at builddir/clients/staging)

    --cmake_install               Install minimum cmake version if required.

    --cuda, --use-cuda            Build library for CUDA backend (deprecated).
                                  The target HIP platform is determined by \`hipconfig --platform\`.
                                  To explicitly specify a platform, set the \`HIP_PLATFORM\` environment variable.

    -d, --dependencies            Build and install external dependencies. Dependencies are to be installed in /usr/local.
                                  This should be done only once (this does not install rocBLAS, rocSolver, or cuda).

    --installcuda                 Install cuda package.

    --installcudaversion <version> Used with --installcuda, optionally specify cuda version to install.

    -g, --debug                   Build in Debug mode, equivalent to set CMAKE_BUILD_TYPE=Debug. (Default build type is Release)

    -h, --help                    Print this help message.

    -i, -install                  Generate and install library package after build.

    -k,  --relwithdebinfo         Build in release debug mode, equivalent to set CMAKE_BUILD_TYPE=RelWithDebInfo. (Default build type is Release)

    -n, --no-solver               Build hipLBAS library without rocSOLVER dependency

    --rocblas-path <blasdir>      Specify path to an existing rocBLAS install directory (e.g. /src/rocBLAS/build/release/rocblas-install).

    --rocsolver-path <solverdir>  Specify path to an existing rocSOLVER install directory (e.g. /src/rocSOLVER/build/release/rocsolver-install).

EOF
}

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
install_prefix=hipblas-install
build_clients=false
build_solver=true
build_release=true
install_cuda=false
cuda_version_install=default
cuda_path=/usr/local/cuda
rocm_path=/opt/rocm
build_release_debug=false
update_cmake=false
rmake_invoked=false
declare -a cmake_client_options

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,no-solver,dependencies,debug,relwithdebinfo,cmake_install,cuda,use-cuda,installcuda,installcudaversion:,rmake_invoked,rocblas:,rocblas-path:,rocsolver-path:,address-sanitizer:, --options :rhickndgb: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

# don't check args as rmake.py handles additional options
# if [[ $? -ne 0 ]]; then
#   echo "getopt invocation failed; could not parse the command line";
#   exit 1
# fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        rmake_cmd="python3 ./rmake.py --help"
        echo "Options provied by rmake.py script:"
        echo $rmake_cmd
        $rmake_cmd
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    -n|--no-solver)
        build_solver=false
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    -k|--relwithdebinfo)
        build_release=false
        build_release_debug=true
        shift ;;
    --cuda|--use-cuda)
        # still need this flag in install.sh to support install.sh --cuda -d for now
        echo "--cuda option is deprecated (use environment variable HIP_PLATFORM=nvidia)"
        export HIP_PLATFORM="nvidia"
        build_cuda=true
        shift ;;
    --installcuda)
      install_cuda=true
      shift ;;
    --installcudaversion)
      cuda_version_install=${2}
      shift 2 ;;
    --cmake_install)
        update_cmake=true
        shift ;;
    -b|--rocblas)
         custom_rocblas=${2}
         shift 2;;
    --rocblas-path)
        rocblas_path=${2}
        shift 2 ;;
    --rocsolver-path)
        rocsolver_path=${2}
        shift 2 ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --address-sanitizer)
        shift 2 ;;
    --rmake_invoked)
        rmake_invoked=true
        shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

set -x

build_dir=$(readlink -m ./build)
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  rm -rf ${build_dir}/release-debug
else
  rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake
cxx="g++"
cc="gcc"
fc="gfortran"

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  CMAKE_VERSION=$(${cmake_executable} --version | grep -oP '(?<=version )[^ ]*' )

  install_packages

  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
      if $update_cmake == true; then
        CMAKE_REPO="https://github.com/Kitware/CMake/releases/download/v3.16.8/"
        wget -nv ${CMAKE_REPO}/cmake-3.16.8.tar.gz
        tar -xvf cmake-3.16.8.tar.gz
        cd cmake-3.16.8
        ./bootstrap --prefix=/usr --no-system-curl --parallel=16
        make -j16
        elevate_if_not_root make install
        cd ..
        rm -rf cmake-3.16.8.tar.gz cmake-3.16.8
      else
          echo "hipBLAS requires CMake version >= 3.16.8 and CMake version ${CMAKE_VERSION} is installed. Run install.sh again with --cmake_install flag and CMake version ${CMAKE_VERSION} will be uninstalled and CMake version 3.16.8 will be installed"
          exit 2
      fi
  fi

  # The following builds googletest & lapack from source
  pushd .
    printf "\033[32mBuilding \033[33mgoogletest & lapack\033[32m from source; installing into build tree and not default \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    CXX=${cxx} CC=${cc} FC=${fc} ${cmake_executable} -DCMAKE_INSTALL_PREFIX=deps-install ${HIPBLAS_SRC_PATH}/deps
    make -j$(nproc)
    # as installing into build tree deps/deps-install rather than /usr/local won't elevate if not root
    make install
  popd
fi

if [[ "${install_cuda}" == true ]]; then
  install_cuda_package
fi

# #################################################
# configure & build
# #################################################

full_build_dir=""
if [[ "${build_release}" == true ]]; then
  full_build_dir=${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  full_build_dir=${build_dir}/release-debug
else
  full_build_dir=${build_dir}/debug
fi

# this can be removed and be done in rmake.py once --cuda flag support is gone
if [[ "${build_cuda}" != true ]]; then
  export HIP_PLATFORM="$(${rocm_path}/bin/hipconfig --platform)"
fi

if [[ "${rmake_invoked}" == false ]]; then
  pushd .

  # ensure a clean build environment
  rm -rf ${full_build_dir}

  # rmake.py at top level same as install.sh
  python3 ./rmake.py --install_invoked ${input_args} --build_dir=${build_dir} --src_path=${HIPBLAS_SRC_PATH}
  check_exit_code "$?"

  popd
else
  # only dependency install supported when called from rmake
  exit 0
fi

# #################################################
# install
# #################################################

pushd .

cd ${full_build_dir}

# installing through package manager, which makes uninstalling easy
if [[ "${install_package}" == true ]]; then
  make package
  check_exit_code "$?"

  case "${ID}" in
    ubuntu)
      elevate_if_not_root dpkg -i hipblas[-\_]*.deb
    ;;
    centos|rhel)
      elevate_if_not_root yum -y localinstall hipblas-*.rpm
    ;;
    fedora)
      elevate_if_not_root dnf install hipblas-*.rpm
    ;;
    sles|opensuse-leap)
      elevate_if_not_root zypper -n --no-gpg-checks install hipblas-*.rpm
    ;;
  esac

fi
check_exit_code "$?"

popd
