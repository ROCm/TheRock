# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

find_package(Git REQUIRED)

therock_subproject_fetch(therock-fftz-sources
  CMAKE_PROJECT
  # TODO: Mirror this archive to the rocm-third-party-deps S3 bucket.
  URL https://github.com/amd/aocl-fftz/archive/7bdcb28b96b3f2c64529105886249b55fa879ee1.tar.gz
  URL_HASH SHA256=c8dae4aa0587380fdbb3a0c3c74d7502323c707df432598a19a9f8d7c2652cff
  PATCH_COMMAND "${GIT_EXECUTABLE}" apply
    "${CMAKE_SOURCE_DIR}/patches/third-party/fftz/0001-Support-Clang-GNU-frontend-on-Windows.patch"
)

therock_cmake_subproject_declare(therock-fftz
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  # The wrapper installs fftw3.h and FFTW library names. Keep it separate
  # from FFTW until the accuracy clients support selecting this provider.
  INSTALL_DESTINATION lib/host-math/fftz
  COMPILER_TOOLCHAIN
    amd-llvm
  CMAKE_ARGS
    -DBUILD_STATIC_LIBS=OFF
    -DBUILD_THIRD_PARTY_WRAPPERS=ON
    # amd-llvm does not currently build OpenMP on Windows.
    -DENABLE_MULTI_THREADING=${THEROCK_CONDITION_IS_NON_WINDOWS}
    -DAOCL_TEST_COVERAGE=OFF
    -DBUILD_DOC=OFF
)
therock_cmake_subproject_provide_package(therock-fftz aocl-fftz lib/host-math/fftz/lib/cmake/aocl-fftz)
# Do not advertise FFTW3: the existing FFTW provider owns that package name.
therock_cmake_subproject_activate(therock-fftz)
add_dependencies(therock-third-party therock-fftz)

therock_provide_artifact(fftz
  TARGET_NEUTRAL
  DESCRIPTOR artifact-fftz.toml
  COMPONENTS
    dbg
    dev
    doc
    lib
  SUBPROJECT_DEPS therock-fftz
)
