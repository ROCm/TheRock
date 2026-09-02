# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TODO: HIP language support not based on an /opt/rocm installed SDK
# must have these three cache variables set to avoid language setup
# error. https://github.com/ROCm/TheRock/issues/102
set(HIP_CMAKE_ARGS
  -DCMAKE_HIP_PLATFORM=amd
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${THEROCK_BINARY_DIR}/dist/rocm"
)

if(WIN32)
  # On Windows, only a unified compiler is supported (i.e. use the HIP compiler
  # for everything).
  list(APPEND HIP_CMAKE_ARGS
    "-DCMAKE_CXX_COMPILER=${THEROCK_BINARY_DIR}/dist/rocm/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  )
else()
  # On everything else, the HIP language compiler can be independent from the
  # overall CXX compiler.
  list(APPEND HIP_CMAKE_ARGS
    "-DCMAKE_HIP_COMPILER=${THEROCK_BINARY_DIR}/dist/rocm/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  )
endif()

# hipTensor is built conditionally to composable_kernel availability (CK is a hard
# build dependency). _ck_enabled, set by math-libs/CMakeLists.txt, is OFF when
# CK and hipTensor were filtered out for the selected gfx targets.
# _hiptensor_available reflect the actual built status so the example does not
# find_package() a missing hiptensor.
set(_hiptensor_available OFF)
if(_ck_enabled AND THEROCK_ENABLE_HIPTENSOR)
  set(_hiptensor_available ON)
endif()

# Examples are configured against the installed ROCm prefix. The
# cpp-sdk-user project requires HIP (LANGUAGES CXX HIP), so skip when
# HIP is not enabled (e.g. in early multi-arch CI stages) or when
# sanitizers are enabled (sanitizer builds do not install HIP).
# ADDSPV: the cpp-sdk-user smoke test does find_package(<lib> CONFIG REQUIRED)
# for every enabled feature group. In a shard where a library is arch-excluded
# (e.g. amdgcnspirv, where rocBLAS/rocSPARSE/rocSOLVER/rocFFT/rocALUTION and
# their hip* wrappers are gated out per-shard in math-libs), the package is not
# in dist, so REQUIRED fails and the packaging test errors even though the build
# is clean. Recompute the feature flags passed to the test per-shard: flip a
# flag OFF when its key subproject has no surviving gfx targets for this shard.
# Same EXCLUDE_TARGET_PROJECTS metadata / filter pattern used elsewhere. On
# non-spirv shards (gfx90a, etc.) every filter is non-empty, so flags are
# unchanged. Pairs: feature flag -> representative subproject.
set(_sdk_flag_projects
  "BLAS=rocBLAS"
  "FFT=rocFFT"
  "SPARSE=rocSPARSE"
  "SOLVER=rocSOLVER"
  "ROCALUTION=rocALUTION"
  "ROCWMMA=rocWMMA"
  "MIOPEN=MIOpen"
  "HIPTENSOR=hipTensor"
  "RCCL=rccl"
  "LIBHIPCXX=libhipcxx"
)
foreach(_pair IN LISTS _sdk_flag_projects)
  string(REPLACE "=" ";" _pair_list "${_pair}")
  list(GET _pair_list 0 _flag)
  list(GET _pair_list 1 _proj)
  set(_sdk_enable_${_flag} ${THEROCK_ENABLE_${_flag}})
  if(_sdk_enable_${_flag} AND THEROCK_AMDGPU_TARGETS
     AND NOT "${THEROCK_AMDGPU_TARGETS}" MATCHES "-NOTFOUND$")
    therock_filter_amdgpu_targets(_sdk_targets ${_proj} ${THEROCK_AMDGPU_TARGETS})
    if(NOT _sdk_targets)
      message(STATUS
        "cpp-sdk-user: ${_proj} is excluded for this shard "
        "(THEROCK_AMDGPU_TARGETS='${THEROCK_AMDGPU_TARGETS}'); disabling "
        "THEROCK_ENABLE_${_flag} for the SDK smoke test.")
      set(_sdk_enable_${_flag} OFF)
    endif()
  endif()
endforeach()

# hipTensor additionally requires composable_kernel (a hard build dependency);
# _hiptensor_available (above) is OFF when CK/hipTensor were filtered out for the
# selected gfx targets. Fold that into the per-shard flag so the smoke test does
# not find_package() a hiptensor that was not built even on non-spirv shards.
if(NOT _hiptensor_available)
  set(_sdk_enable_HIPTENSOR OFF)
endif()

if(THEROCK_ENABLE_HIP_RUNTIME AND "${THEROCK_SANITIZER}" STREQUAL "")
  add_test(
    NAME therock-examples-cpp-sdk-user
    COMMAND
      "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH=${THEROCK_BINARY_DIR}/dist/rocm" --
      "${CMAKE_COMMAND}"
          "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cpp-sdk-user"
          "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cpp-sdk-user"
          "-DGENERATOR=${CMAKE_GENERATOR}"
          "-DTHEROCK_ENABLE_BLAS=${_sdk_enable_BLAS}"
          "-DTHEROCK_ENABLE_FFT=${_sdk_enable_FFT}"
          "-DTHEROCK_ENABLE_HIP=${THEROCK_ENABLE_HIP_RUNTIME}"
          "-DTHEROCK_ENABLE_HIPDNN=${THEROCK_ENABLE_HIPDNN}"
          "-DTHEROCK_ENABLE_LIBHIPCXX=${_sdk_enable_LIBHIPCXX}"
          "-DTHEROCK_ENABLE_MIOPEN=${_sdk_enable_MIOPEN}"
          "-DTHEROCK_ENABLE_HIPTENSOR=${_sdk_enable_HIPTENSOR}"
          "-DTHEROCK_ENABLE_PRIM=${THEROCK_ENABLE_PRIM}"
          "-DTHEROCK_ENABLE_RAND=${THEROCK_ENABLE_RAND}"
          "-DTHEROCK_ENABLE_RCCL=${_sdk_enable_RCCL}"
          "-DTHEROCK_ENABLE_ROCWMMA=${_sdk_enable_ROCWMMA}"
          "-DTHEROCK_ENABLE_ROCALUTION=${_sdk_enable_ROCALUTION}"
          "-DTHEROCK_ENABLE_SOLVER=${_sdk_enable_SOLVER}"
          "-DTHEROCK_ENABLE_SPARSE=${_sdk_enable_SPARSE}"
          ${HIP_CMAKE_ARGS}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/clean_configure_test_project.cmake"
  )
endif()
