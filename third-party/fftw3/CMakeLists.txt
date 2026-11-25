# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Patches fix bugs in upstream's own CMakeLists.txt; see patches/third-party/fftw3
# for details on each one.
therock_subproject_fetch(therock-fftw3-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://fftw.org/fftw-3.3.10.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fftw-3.3.10.tar.gz
  URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
  PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/fftw3
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_fftw3_enable_threads "ON")
else()
  set(_fftw3_enable_threads "OFF")
endif()

# FFTW leaves SIMD codelets out unless they are requested. Enable them on
# Linux x86_64. FFTW picks the ISA at runtime. Upstream's CMake build has no
# equivalent of autotools' --enable-avx512, so this is SSE2/AVX/AVX2 only;
# AVX-512 codelet support, if ever needed, is a separate follow-up patch.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _fftw3_proc)
set(_fftw3_x86_64 OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _fftw3_proc MATCHES "^(x86_64|amd64)$")
  set(_fftw3_x86_64 ON)
endif()

set(_fftw3_common_cmake_args
  -DBUILD_SHARED_LIBS=ON
  -DENABLE_THREADS=${_fftw3_enable_threads}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
if(_fftw3_x86_64)
  list(APPEND _fftw3_common_cmake_args
    -DENABLE_SSE2=ON
    -DENABLE_AVX=ON
    -DENABLE_AVX2=ON
  )
endif()

therock_cmake_subproject_declare(therock-fftw3
  USE_DIST_AMDGPU_TARGETS
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build_fftw3"
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    ${_fftw3_common_cmake_args}
)
therock_cmake_subproject_provide_package(therock-fftw3 FFTW3 lib/cmake/fftw3)
therock_cmake_subproject_activate(therock-fftw3)

add_dependencies(therock-third-party therock-fftw3)

therock_cmake_subproject_declare(therock-fftw3f
  USE_DIST_AMDGPU_TARGETS
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build_fftw3f"
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DENABLE_FLOAT=ON
    ${_fftw3_common_cmake_args}
)
therock_cmake_subproject_provide_package(therock-fftw3f FFTW3f lib/cmake/fftw3f)
therock_cmake_subproject_activate(therock-fftw3f)

add_dependencies(therock-third-party therock-fftw3f)

if(THEROCK_ENABLE_FFTW3)
  therock_provide_artifact(fftw3
    DESCRIPTOR artifact_fftw3.toml
    TARGET_NEUTRAL
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS therock-fftw3 therock-fftw3f
  )
endif()
