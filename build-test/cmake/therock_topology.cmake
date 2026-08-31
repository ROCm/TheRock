# Auto-generated from BUILD_TOPOLOGY.toml
# DO NOT EDIT MANUALLY

# =============================================================================
# Validation metadata
# =============================================================================

# List of all valid artifacts defined in topology
set(THEROCK_TOPOLOGY_ARTIFACTS
  sysdeps
  sysdeps-amd-mesa
  sysdeps-expat
  sysdeps-gmp
  sysdeps-mpfr
  sysdeps-ncurses
  sysdeps-libmnl
  sysdeps-util-linux
  sysdeps-libnl
  sysdeps-libpciaccess
  sysdeps-hwloc
  host-blas
  host-suite-sparse
  elfio
  fftw3
  flatbuffers
  fmt
  nlohmann-json
  spdlog
  openmpi
  base
  amd-llvm
  hipify
  core-runtime
  wsl-rocdxg
  core-amdsmi
  core-kpack
  core-hip
  core-ocl-icd
  core-ocl
  rocrtst
  kfdtest
  core-hipinfo
  core-hiptests
  blas
  rand
  fft
  prim
  sparse
  solver
  rocalution
  rocwmma
  composable-kernel
  hiptensor
  libhipcxx
  hipthreads
  support
  miopen
  hipdnn
  hipdnn-integration-tests
  miopenprovider
  hipblasltprovider
  hipdnn-samples
  hipkernelprovider
  rpp
  rocdecode
  rocjpeg
  rccl
  rocshmem
  hipfile
  aqlprofile
  rocprofiler-sdk
  rocprofiler-compute
  rocprofiler-systems
  rocprofiler-systems-examples
  rdc
  amd-dbgapi
  rocr-debug-agent
  rocr-debug-agent-tests
  rocgdb
  rocjitsu
  rocjitsu-hotswap
  mirage
)

# Mapping of artifacts to their groups
set(THEROCK_ARTIFACT_GROUP_sysdeps "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_amd_mesa "media-libs")
set(THEROCK_ARTIFACT_GROUP_sysdeps_expat "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_gmp "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_mpfr "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_ncurses "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_libmnl "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_util_linux "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_libnl "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_libpciaccess "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_sysdeps_hwloc "third-party-sysdeps")
set(THEROCK_ARTIFACT_GROUP_host_blas "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_host_suite_sparse "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_elfio "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_fftw3 "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_flatbuffers "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_fmt "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_nlohmann_json "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_spdlog "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_openmpi "third-party-libs")
set(THEROCK_ARTIFACT_GROUP_base "base")
set(THEROCK_ARTIFACT_GROUP_amd_llvm "compiler")
set(THEROCK_ARTIFACT_GROUP_hipify "compiler")
set(THEROCK_ARTIFACT_GROUP_core_runtime "core-runtime")
set(THEROCK_ARTIFACT_GROUP_wsl_rocdxg "wsl-rocdxg")
set(THEROCK_ARTIFACT_GROUP_core_amdsmi "core-amdsmi")
set(THEROCK_ARTIFACT_GROUP_core_kpack "hip-runtime")
set(THEROCK_ARTIFACT_GROUP_core_hip "hip-runtime")
set(THEROCK_ARTIFACT_GROUP_core_ocl_icd "opencl-runtime")
set(THEROCK_ARTIFACT_GROUP_core_ocl "opencl-runtime")
set(THEROCK_ARTIFACT_GROUP_rocrtst "runtime-tests")
set(THEROCK_ARTIFACT_GROUP_kfdtest "kfdtest")
set(THEROCK_ARTIFACT_GROUP_core_hipinfo "hip-runtime")
set(THEROCK_ARTIFACT_GROUP_core_hiptests "runtime-tests")
set(THEROCK_ARTIFACT_GROUP_blas "math-libs")
set(THEROCK_ARTIFACT_GROUP_rand "math-libs")
set(THEROCK_ARTIFACT_GROUP_fft "math-libs")
set(THEROCK_ARTIFACT_GROUP_prim "math-libs")
set(THEROCK_ARTIFACT_GROUP_sparse "math-libs")
set(THEROCK_ARTIFACT_GROUP_solver "math-libs")
set(THEROCK_ARTIFACT_GROUP_rocalution "math-libs")
set(THEROCK_ARTIFACT_GROUP_rocwmma "math-libs")
set(THEROCK_ARTIFACT_GROUP_composable_kernel "math-libs")
set(THEROCK_ARTIFACT_GROUP_hiptensor "math-libs")
set(THEROCK_ARTIFACT_GROUP_libhipcxx "math-libs")
set(THEROCK_ARTIFACT_GROUP_hipthreads "math-libs")
set(THEROCK_ARTIFACT_GROUP_support "math-libs")
set(THEROCK_ARTIFACT_GROUP_miopen "ml-libs")
set(THEROCK_ARTIFACT_GROUP_hipdnn "ml-libs")
set(THEROCK_ARTIFACT_GROUP_hipdnn_integration_tests "ml-libs")
set(THEROCK_ARTIFACT_GROUP_miopenprovider "ml-libs")
set(THEROCK_ARTIFACT_GROUP_hipblasltprovider "ml-libs")
set(THEROCK_ARTIFACT_GROUP_hipdnn_samples "ml-libs")
set(THEROCK_ARTIFACT_GROUP_hipkernelprovider "ml-libs")
set(THEROCK_ARTIFACT_GROUP_rpp "cv-libs")
set(THEROCK_ARTIFACT_GROUP_rocdecode "media-libs")
set(THEROCK_ARTIFACT_GROUP_rocjpeg "media-libs")
set(THEROCK_ARTIFACT_GROUP_rccl "comm-libs")
set(THEROCK_ARTIFACT_GROUP_rocshmem "comm-libs")
set(THEROCK_ARTIFACT_GROUP_hipfile "storage-libs")
set(THEROCK_ARTIFACT_GROUP_aqlprofile "profiler-core")
set(THEROCK_ARTIFACT_GROUP_rocprofiler_sdk "profiler-core")
set(THEROCK_ARTIFACT_GROUP_rocprofiler_compute "profiler-core")
set(THEROCK_ARTIFACT_GROUP_rocprofiler_systems "profiler-apps")
set(THEROCK_ARTIFACT_GROUP_rocprofiler_systems_examples "profiler-apps")
set(THEROCK_ARTIFACT_GROUP_rdc "dctools-core")
set(THEROCK_ARTIFACT_GROUP_amd_dbgapi "debug-tools")
set(THEROCK_ARTIFACT_GROUP_rocr_debug_agent "debug-tools")
set(THEROCK_ARTIFACT_GROUP_rocr_debug_agent_tests "debug-tools")
set(THEROCK_ARTIFACT_GROUP_rocgdb "debug-tools")
set(THEROCK_ARTIFACT_GROUP_rocjitsu "rocjitsu")
set(THEROCK_ARTIFACT_GROUP_rocjitsu_hotswap "rocjitsu")
set(THEROCK_ARTIFACT_GROUP_mirage "rocjitsu")

# Artifact type and split database metadata for kpack splitting
set(THEROCK_ARTIFACT_TYPE_sysdeps "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-amd-mesa "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-expat "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-gmp "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-mpfr "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-ncurses "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-libmnl "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-util-linux "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-libnl "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-libpciaccess "target-neutral")
set(THEROCK_ARTIFACT_TYPE_sysdeps-hwloc "target-neutral")
set(THEROCK_ARTIFACT_TYPE_host-blas "target-neutral")
set(THEROCK_ARTIFACT_TYPE_host-suite-sparse "target-neutral")
set(THEROCK_ARTIFACT_TYPE_elfio "target-neutral")
set(THEROCK_ARTIFACT_TYPE_fftw3 "target-neutral")
set(THEROCK_ARTIFACT_TYPE_flatbuffers "target-neutral")
set(THEROCK_ARTIFACT_TYPE_fmt "target-neutral")
set(THEROCK_ARTIFACT_TYPE_nlohmann-json "target-neutral")
set(THEROCK_ARTIFACT_TYPE_spdlog "target-neutral")
set(THEROCK_ARTIFACT_TYPE_openmpi "target-neutral")
set(THEROCK_ARTIFACT_TYPE_base "target-neutral")
set(THEROCK_ARTIFACT_TYPE_amd-llvm "target-neutral")
set(THEROCK_ARTIFACT_TYPE_hipify "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-runtime "target-neutral")
set(THEROCK_ARTIFACT_TYPE_wsl-rocdxg "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-amdsmi "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-kpack "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-hip "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-ocl-icd "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-ocl "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocrtst "target-neutral")
set(THEROCK_ARTIFACT_TYPE_kfdtest "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-hipinfo "target-neutral")
set(THEROCK_ARTIFACT_TYPE_core-hiptests "target-neutral")
set(THEROCK_ARTIFACT_TYPE_blas "target-specific")
set(THEROCK_ARTIFACT_SPLIT_DATABASES_blas "rocblas;hipblaslt;hipsparselt")
set(THEROCK_ARTIFACT_TYPE_rand "target-specific")
set(THEROCK_ARTIFACT_TYPE_fft "target-specific")
set(THEROCK_ARTIFACT_TYPE_prim "target-specific")
set(THEROCK_ARTIFACT_TYPE_sparse "target-specific")
set(THEROCK_ARTIFACT_TYPE_solver "target-specific")
set(THEROCK_ARTIFACT_TYPE_rocalution "target-specific")
set(THEROCK_ARTIFACT_TYPE_rocwmma "target-specific")
set(THEROCK_ARTIFACT_TYPE_composable-kernel "target-specific")
set(THEROCK_ARTIFACT_TYPE_hiptensor "target-specific")
set(THEROCK_ARTIFACT_TYPE_libhipcxx "target-specific")
set(THEROCK_ARTIFACT_TYPE_hipthreads "target-neutral")
set(THEROCK_ARTIFACT_TYPE_support "target-neutral")
set(THEROCK_ARTIFACT_TYPE_miopen "target-specific")
set(THEROCK_ARTIFACT_SPLIT_DATABASES_miopen "miopen")
set(THEROCK_ARTIFACT_TYPE_hipdnn "target-neutral")
set(THEROCK_ARTIFACT_TYPE_hipdnn-integration-tests "target-neutral")
set(THEROCK_ARTIFACT_TYPE_miopenprovider "target-neutral")
set(THEROCK_ARTIFACT_TYPE_hipblasltprovider "target-neutral")
set(THEROCK_ARTIFACT_TYPE_hipdnn-samples "target-specific")
set(THEROCK_ARTIFACT_TYPE_hipkernelprovider "target-specific")
set(THEROCK_ARTIFACT_SPLIT_DATABASES_hipkernelprovider "hipkernelprovider")
set(THEROCK_ARTIFACT_TYPE_rpp "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocdecode "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocjpeg "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rccl "target-specific")
set(THEROCK_ARTIFACT_SPLIT_DATABASES_rccl "hotswap_cache")
set(THEROCK_ARTIFACT_TYPE_rocshmem "target-neutral")
set(THEROCK_ARTIFACT_TYPE_hipfile "target-neutral")
set(THEROCK_ARTIFACT_TYPE_aqlprofile "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocprofiler-sdk "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocprofiler-compute "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocprofiler-systems "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocprofiler-systems-examples "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rdc "target-neutral")
set(THEROCK_ARTIFACT_TYPE_amd-dbgapi "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocr-debug-agent "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocr-debug-agent-tests "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocgdb "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocjitsu "target-neutral")
set(THEROCK_ARTIFACT_TYPE_rocjitsu-hotswap "target-neutral")
set(THEROCK_ARTIFACT_TYPE_mirage "target-neutral")

# List of artifacts in each group
set(THEROCK_GROUP_ARTIFACTS_third_party_sysdeps
  sysdeps
  sysdeps-expat
  sysdeps-gmp
  sysdeps-mpfr
  sysdeps-ncurses
  sysdeps-libmnl
  sysdeps-util-linux
  sysdeps-libnl
  sysdeps-libpciaccess
  sysdeps-hwloc
)

set(THEROCK_GROUP_ARTIFACTS_third_party_libs
  host-blas
  host-suite-sparse
  elfio
  fftw3
  flatbuffers
  fmt
  nlohmann-json
  spdlog
  openmpi
)

set(THEROCK_GROUP_ARTIFACTS_base
  base
)

set(THEROCK_GROUP_ARTIFACTS_core_runtime
  core-runtime
)

set(THEROCK_GROUP_ARTIFACTS_wsl_rocdxg
  wsl-rocdxg
)

set(THEROCK_GROUP_ARTIFACTS_core_amdsmi
  core-amdsmi
)

set(THEROCK_GROUP_ARTIFACTS_compiler
  amd-llvm
  hipify
)

set(THEROCK_GROUP_ARTIFACTS_debug_tools
  amd-dbgapi
  rocr-debug-agent
  rocr-debug-agent-tests
  rocgdb
)

set(THEROCK_GROUP_ARTIFACTS_hip_runtime
  core-kpack
  core-hip
  core-hipinfo
)

set(THEROCK_GROUP_ARTIFACTS_opencl_runtime
  core-ocl-icd
  core-ocl
)

set(THEROCK_GROUP_ARTIFACTS_runtime_tests
  rocrtst
  core-hiptests
)

set(THEROCK_GROUP_ARTIFACTS_kfdtest
  kfdtest
)

set(THEROCK_GROUP_ARTIFACTS_math_libs
  blas
  rand
  fft
  prim
  sparse
  solver
  rocalution
  rocwmma
  composable-kernel
  hiptensor
  libhipcxx
  hipthreads
  support
)

set(THEROCK_GROUP_ARTIFACTS_ml_libs
  miopen
  hipdnn
  hipdnn-integration-tests
  miopenprovider
  hipblasltprovider
  hipdnn-samples
  hipkernelprovider
)

set(THEROCK_GROUP_ARTIFACTS_comm_libs
  rccl
  rocshmem
)

set(THEROCK_GROUP_ARTIFACTS_storage_libs
  hipfile
)

set(THEROCK_GROUP_ARTIFACTS_profiler_core
  aqlprofile
  rocprofiler-sdk
  rocprofiler-compute
)

set(THEROCK_GROUP_ARTIFACTS_dctools_core
  rdc
)

set(THEROCK_GROUP_ARTIFACTS_profiler_apps
  rocprofiler-systems
  rocprofiler-systems-examples
)

set(THEROCK_GROUP_ARTIFACTS_cv_libs
  rpp
)

set(THEROCK_GROUP_ARTIFACTS_media_libs
  sysdeps-amd-mesa
  rocdecode
  rocjpeg
)

set(THEROCK_GROUP_ARTIFACTS_rocjitsu
  rocjitsu
  rocjitsu-hotswap
  mirage
)

# =============================================================================
# Feature declarations from artifacts
# =============================================================================

# Note: therock_features is already included in main CMakeLists.txt

therock_add_feature(SYSDEPS
  GROUP CORE
  DESCRIPTION "Enables sysdeps"
)

therock_add_feature(SYSDEPS_EXPAT
  GROUP CORE
  DESCRIPTION "Enables sysdeps-expat"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_GMP
  GROUP CORE
  DESCRIPTION "Enables sysdeps-gmp"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_MPFR
  GROUP CORE
  DESCRIPTION "Enables sysdeps-mpfr"
  REQUIRES SYSDEPS_GMP
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_NCURSES
  GROUP CORE
  DESCRIPTION "Enables sysdeps-ncurses"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_LIBMNL
  GROUP CORE
  DESCRIPTION "Enables sysdeps-libmnl"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_UTIL_LINUX
  GROUP CORE
  DESCRIPTION "Enables sysdeps-util-linux"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_LIBNL
  GROUP CORE
  DESCRIPTION "Enables sysdeps-libnl"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_LIBPCIACCESS
  GROUP CORE
  DESCRIPTION "Enables sysdeps-libpciaccess"
  DISABLE_PLATFORMS windows
)

therock_add_feature(SYSDEPS_HWLOC
  GROUP CORE
  DESCRIPTION "Enables sysdeps-hwloc"
  REQUIRES SYSDEPS SYSDEPS_LIBPCIACCESS
  DISABLE_PLATFORMS windows
)

therock_add_feature(BASE
  GROUP CORE
  DESCRIPTION "Enables base"
)

therock_add_feature(COMPILER
  GROUP ALL
  DESCRIPTION "Enables amd-llvm"
  REQUIRES SYSDEPS
)

therock_add_feature(HIPIFY
  GROUP ALL
  DESCRIPTION "Enables hipify"
  REQUIRES COMPILER
)

therock_add_feature(CORE_AMDSMI
  GROUP CORE
  DESCRIPTION "Enables core-amdsmi"
  REQUIRES BASE SYSDEPS SYSDEPS_LIBNL SYSDEPS_LIBMNL COMPILER
  DISABLE_PLATFORMS windows
)

string(TOLOWER "${CMAKE_SYSTEM_NAME}" _therock_system_lower)
set(_THEROCK_CORE_RUNTIME_DISABLE_PLATFORMS)
if(NOT THEROCK_FLAG_HSA_WINDOWS_SHARED_RUNTIME)
  list(APPEND _THEROCK_CORE_RUNTIME_DISABLE_PLATFORMS windows)
endif()
if(_therock_system_lower STREQUAL "windows" AND THEROCK_ENABLE_CORE_RUNTIME AND NOT THEROCK_FLAG_HSA_WINDOWS_SHARED_RUNTIME)
  message(FATAL_ERROR "CORE_RUNTIME can be built on ${CMAKE_SYSTEM_NAME} only with -DTHEROCK_FLAG_HSA_WINDOWS_SHARED_RUNTIME=ON")
endif()
if(_THEROCK_CORE_RUNTIME_DISABLE_PLATFORMS)
therock_add_feature(CORE_RUNTIME
  GROUP CORE
  DESCRIPTION "Enables core-runtime"
  REQUIRES BASE SYSDEPS COMPILER
  DISABLE_PLATFORMS ${_THEROCK_CORE_RUNTIME_DISABLE_PLATFORMS}
)
else()
therock_add_feature(CORE_RUNTIME
  GROUP CORE
  DESCRIPTION "Enables core-runtime"
  REQUIRES BASE SYSDEPS COMPILER
)
endif()
unset(_THEROCK_CORE_RUNTIME_DISABLE_PLATFORMS)
unset(_therock_system_lower)

therock_add_feature(HOST_BLAS
  GROUP HOST_MATH
  DESCRIPTION "Enables host-blas"
)

therock_add_feature(HOST_SUITE_SPARSE
  GROUP HOST_MATH
  DESCRIPTION "Enables host-suite-sparse"
  REQUIRES HOST_BLAS
  DISABLE_PLATFORMS windows
)

therock_add_feature(ELFIO
  GROUP CORE
  DESCRIPTION "Enables elfio"
  DISABLE_PLATFORMS windows
)

therock_add_feature(FFTW3
  GROUP HOST_MATH
  DESCRIPTION "Enables fftw3"
)

therock_add_feature(FLATBUFFERS
  GROUP CORE
  DESCRIPTION "Enables flatbuffers"
)

therock_add_feature(FMT
  GROUP CORE
  DESCRIPTION "Enables fmt"
)

therock_add_feature(NLOHMANN_JSON
  GROUP CORE
  DESCRIPTION "Enables nlohmann-json"
)

therock_add_feature(SPDLOG
  GROUP CORE
  DESCRIPTION "Enables spdlog"
  REQUIRES COMPILER
)

therock_add_feature(OPENMPI
  GROUP CORE
  DESCRIPTION "Enables openmpi"
)

therock_add_feature(KPACK
  GROUP CORE
  DESCRIPTION "Enables core-kpack"
  REQUIRES SYSDEPS
)

therock_add_feature(HIP_RUNTIME
  GROUP CORE
  DESCRIPTION "Enables core-hip"
  REQUIRES CORE_RUNTIME COMPILER KPACK
)

therock_add_feature(CORE_HIPINFO
  GROUP CORE
  DESCRIPTION "Enables core-hipinfo"
  REQUIRES HIP_RUNTIME
)

therock_add_feature(CORE_KFDTESTS
  GROUP CORE
  DESCRIPTION "Enables kfdtest"
  REQUIRES CORE_RUNTIME COMPILER SYSDEPS
  DISABLE_PLATFORMS windows
)

therock_add_feature(OCL_ICD
  GROUP CORE
  DESCRIPTION "Enables core-ocl-icd"
  REQUIRES CORE_RUNTIME COMPILER
)

therock_add_feature(OCL_RUNTIME
  GROUP CORE
  DESCRIPTION "Enables core-ocl"
  REQUIRES CORE_RUNTIME COMPILER OCL_ICD
)

therock_add_feature(AQLPROFILE
  GROUP PROFILER
  DESCRIPTION "Enables aqlprofile"
  REQUIRES CORE_RUNTIME
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCPROFV3
  GROUP PROFILER
  DESCRIPTION "Enables rocprofiler-sdk"
  REQUIRES CORE_RUNTIME BASE AQLPROFILE ELFIO
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCPROFILER_COMPUTE
  GROUP PROFILER
  DESCRIPTION "Enables rocprofiler-compute"
  REQUIRES ROCPROFV3 OPENMPI
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCJITSU
  GROUP EMULATION
  DESCRIPTION "Enables rocjitsu"
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCJITSU_HOTSWAP
  GROUP EMULATION
  DESCRIPTION "Enables rocjitsu-hotswap"
  REQUIRES ROCJITSU
  DISABLE_PLATFORMS windows
)

therock_add_feature(MIRAGE
  GROUP EMULATION
  DESCRIPTION "Enables mirage"
  DISABLE_PLATFORMS windows
)

therock_add_feature(WSL_ROCDXG
  GROUP WSL
  DESCRIPTION "Enables wsl-rocdxg"
  REQUIRES HIP_RUNTIME
)

therock_add_feature(CORE_RUNTIME_TESTS
  GROUP CORE
  DESCRIPTION "Enables rocrtst"
  REQUIRES CORE_RUNTIME OCL_RUNTIME CORE_AMDSMI SYSDEPS_HWLOC
  DISABLE_PLATFORMS windows
)

therock_add_feature(CORE_HIPTESTS
  GROUP CORE
  DESCRIPTION "Enables core-hiptests"
  REQUIRES HIP_RUNTIME
)

therock_add_feature(BLAS
  GROUP MATH_LIBS
  DESCRIPTION "Enables blas"
  REQUIRES CORE_RUNTIME HIP_RUNTIME CORE_AMDSMI HOST_BLAS HOST_SUITE_SPARSE ROCPROFV3 SPDLOG
)

therock_add_feature(RAND
  GROUP MATH_LIBS
  DESCRIPTION "Enables rand"
  REQUIRES CORE_RUNTIME HIP_RUNTIME
)

therock_add_feature(FFT
  GROUP MATH_LIBS
  DESCRIPTION "Enables fft"
  REQUIRES CORE_RUNTIME HIP_RUNTIME FFTW3 RAND ROCPROFV3
)

therock_add_feature(PRIM
  GROUP MATH_LIBS
  DESCRIPTION "Enables prim"
  REQUIRES CORE_RUNTIME HIP_RUNTIME RAND
)

therock_add_feature(SPARSE
  GROUP MATH_LIBS
  DESCRIPTION "Enables sparse"
  REQUIRES BLAS PRIM
)

therock_add_feature(SOLVER
  GROUP MATH_LIBS
  DESCRIPTION "Enables solver"
  REQUIRES BLAS PRIM SPARSE HOST_SUITE_SPARSE
)

therock_add_feature(ROCALUTION
  GROUP MATH_LIBS
  DESCRIPTION "Enables rocalution"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BLAS PRIM RAND SPARSE
)

therock_add_feature(ROCWMMA
  GROUP MATH_LIBS
  DESCRIPTION "Enables rocwmma"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BLAS
)

therock_add_feature(COMPOSABLE_KERNEL
  GROUP MATH_LIBS
  DESCRIPTION "Enables composable-kernel"
  REQUIRES CORE_RUNTIME HIP_RUNTIME
)

therock_add_feature(HIPTENSOR
  GROUP MATH_LIBS
  DESCRIPTION "Enables hiptensor"
  REQUIRES CORE_RUNTIME HIP_RUNTIME COMPOSABLE_KERNEL
)

therock_add_feature(LIBHIPCXX
  GROUP MATH_LIBS
  DESCRIPTION "Enables libhipcxx"
  REQUIRES CORE_RUNTIME HIP_RUNTIME COMPILER
)

therock_add_feature(HIPTHREADS
  GROUP MATH_LIBS
  DESCRIPTION "Enables hipthreads"
  REQUIRES CORE_RUNTIME HIP_RUNTIME COMPILER LIBHIPCXX
)

therock_add_feature(SUPPORT
  GROUP MATH_LIBS
  DESCRIPTION "Enables support"
)

therock_add_feature(MIOPEN
  GROUP ML_LIBS
  DESCRIPTION "Enables miopen"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BLAS COMPOSABLE_KERNEL RAND ROCPROFV3
)

therock_add_feature(HIPDNN
  GROUP ML_LIBS
  DESCRIPTION "Enables hipdnn"
  REQUIRES CORE_RUNTIME HIP_RUNTIME SPDLOG
)

therock_add_feature(HIPDNN_INTEGRATION_TESTS
  GROUP ML_LIBS
  DESCRIPTION "Enables hipdnn-integration-tests"
  REQUIRES CORE_RUNTIME HIP_RUNTIME HIPDNN RAND
)

therock_add_feature(MIOPENPROVIDER
  GROUP ML_LIBS
  DESCRIPTION "Enables miopenprovider"
  REQUIRES CORE_RUNTIME HIP_RUNTIME MIOPEN HIPDNN HIPDNN_INTEGRATION_TESTS
)

therock_add_feature(HIPBLASLTPROVIDER
  GROUP ML_LIBS
  DESCRIPTION "Enables hipblasltprovider"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BLAS HIPDNN HIPDNN_INTEGRATION_TESTS
)

therock_add_feature(HIPDNN_SAMPLES
  GROUP ML_LIBS
  DESCRIPTION "Enables hipdnn-samples"
  REQUIRES CORE_RUNTIME HIP_RUNTIME MIOPEN HIPDNN MIOPENPROVIDER
)

therock_add_feature(HIPKERNELPROVIDER
  GROUP ML_LIBS
  DESCRIPTION "Enables hipkernelprovider"
  REQUIRES CORE_RUNTIME HIP_RUNTIME HIPDNN HIPDNN_INTEGRATION_TESTS
)

therock_add_feature(RCCL
  GROUP COMM_LIBS
  DESCRIPTION "Enables rccl"
  REQUIRES CORE_RUNTIME HIP_RUNTIME HIPIFY ROCPROFV3 CORE_AMDSMI OPENMPI ROCJITSU_HOTSWAP
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCSHMEM
  GROUP COMM_LIBS
  DESCRIPTION "Enables rocshmem"
  REQUIRES CORE_RUNTIME HIP_RUNTIME SYSDEPS
  DISABLE_PLATFORMS windows
)

therock_add_feature(RPP
  GROUP CV_LIBS
  DESCRIPTION "Enables rpp"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BASE SYSDEPS
)

therock_add_feature(HIPFILE
  GROUP STORAGE_LIBS
  DESCRIPTION "Enables hipfile"
  REQUIRES CORE_RUNTIME HIP_RUNTIME SYSDEPS SYSDEPS_UTIL_LINUX
  DISABLE_PLATFORMS windows
)

therock_add_feature(AMD_DBGAPI
  GROUP DEBUG_TOOLS
  DESCRIPTION "Enables amd-dbgapi"
  REQUIRES COMPILER SYSDEPS
)

therock_add_feature(ROCR_DEBUG_AGENT
  GROUP DEBUG_TOOLS
  DESCRIPTION "Enables rocr-debug-agent"
  REQUIRES COMPILER CORE_RUNTIME AMD_DBGAPI SYSDEPS
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCR_DEBUG_AGENT_TESTS
  GROUP DEBUG_TOOLS
  DESCRIPTION "Enables rocr-debug-agent-tests"
  REQUIRES ROCR_DEBUG_AGENT HIP_RUNTIME
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCGDB
  GROUP DEBUG_TOOLS
  DESCRIPTION "Enables rocgdb"
  REQUIRES COMPILER AMD_DBGAPI SYSDEPS_GMP SYSDEPS_MPFR SYSDEPS_EXPAT SYSDEPS_NCURSES
  DISABLE_PLATFORMS windows
)

therock_add_feature(RDC
  GROUP DC_TOOLS
  DESCRIPTION "Enables rdc"
  REQUIRES HIP_RUNTIME ROCPROFV3 SYSDEPS CORE_AMDSMI
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCPROFSYS
  GROUP PROFILER
  DESCRIPTION "Enables rocprofiler-systems"
  REQUIRES COMPILER HIP_RUNTIME ROCPROFV3 CORE_AMDSMI SPDLOG OPENMPI
  DISABLE_PLATFORMS windows
  DISABLE_PROCESSORS ppc64le
)

therock_add_feature(ROCPROFILER_SYSTEMS_EXAMPLES
  GROUP PROFILER
  DESCRIPTION "Enables rocprofiler-systems-examples"
  REQUIRES ROCPROFSYS
  DISABLE_PLATFORMS windows
  DISABLE_PROCESSORS ppc64le
)

therock_add_feature(SYSDEPS_AMD_MESA
  GROUP MEDIA_LIBS
  DESCRIPTION "Enables sysdeps-amd-mesa"
  REQUIRES SYSDEPS
)

therock_add_feature(ROCDECODE
  GROUP MEDIA_LIBS
  DESCRIPTION "Enables rocdecode"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BASE SYSDEPS SYSDEPS_AMD_MESA
  DISABLE_PLATFORMS windows
)

therock_add_feature(ROCJPEG
  GROUP MEDIA_LIBS
  DESCRIPTION "Enables rocjpeg"
  REQUIRES CORE_RUNTIME HIP_RUNTIME BASE SYSDEPS SYSDEPS_AMD_MESA
  DISABLE_PLATFORMS windows
)

# =============================================================================
# Artifact targets
# =============================================================================

# Artifact: sysdeps
add_custom_target(artifact-sysdeps
  COMMENT "Building artifact sysdeps"
)

# Artifact: sysdeps-amd-mesa
add_custom_target(artifact-sysdeps-amd-mesa
  COMMENT "Building artifact sysdeps-amd-mesa"
)

# Artifact: sysdeps-expat
add_custom_target(artifact-sysdeps-expat
  COMMENT "Building artifact sysdeps-expat"
)

# Artifact: sysdeps-gmp
add_custom_target(artifact-sysdeps-gmp
  COMMENT "Building artifact sysdeps-gmp"
)

# Artifact: sysdeps-mpfr
add_custom_target(artifact-sysdeps-mpfr
  COMMENT "Building artifact sysdeps-mpfr"
)

# Artifact: sysdeps-ncurses
add_custom_target(artifact-sysdeps-ncurses
  COMMENT "Building artifact sysdeps-ncurses"
)

# Artifact: sysdeps-libmnl
add_custom_target(artifact-sysdeps-libmnl
  COMMENT "Building artifact sysdeps-libmnl"
)

# Artifact: sysdeps-util-linux
add_custom_target(artifact-sysdeps-util-linux
  COMMENT "Building artifact sysdeps-util-linux"
)

# Artifact: sysdeps-libnl
add_custom_target(artifact-sysdeps-libnl
  COMMENT "Building artifact sysdeps-libnl"
)

# Artifact: sysdeps-libpciaccess
add_custom_target(artifact-sysdeps-libpciaccess
  COMMENT "Building artifact sysdeps-libpciaccess"
)

# Artifact: sysdeps-hwloc
add_custom_target(artifact-sysdeps-hwloc
  COMMENT "Building artifact sysdeps-hwloc"
)

# Artifact: host-blas
add_custom_target(artifact-host-blas
  COMMENT "Building artifact host-blas"
)

# Artifact: host-suite-sparse
add_custom_target(artifact-host-suite-sparse
  COMMENT "Building artifact host-suite-sparse"
)

# Artifact: elfio
add_custom_target(artifact-elfio
  COMMENT "Building artifact elfio"
)

# Artifact: fftw3
add_custom_target(artifact-fftw3
  COMMENT "Building artifact fftw3"
)

# Artifact: flatbuffers
add_custom_target(artifact-flatbuffers
  COMMENT "Building artifact flatbuffers"
)

# Artifact: fmt
add_custom_target(artifact-fmt
  COMMENT "Building artifact fmt"
)

# Artifact: nlohmann-json
add_custom_target(artifact-nlohmann-json
  COMMENT "Building artifact nlohmann-json"
)

# Artifact: spdlog
add_custom_target(artifact-spdlog
  COMMENT "Building artifact spdlog"
)

# Artifact: openmpi
add_custom_target(artifact-openmpi
  COMMENT "Building artifact openmpi"
)

# Artifact: base
add_custom_target(artifact-base
  COMMENT "Building artifact base"
)

# Artifact: amd-llvm
add_custom_target(artifact-amd-llvm
  COMMENT "Building artifact amd-llvm"
)

# Artifact: hipify
add_custom_target(artifact-hipify
  COMMENT "Building artifact hipify"
)

# Artifact: core-runtime
add_custom_target(artifact-core-runtime
  COMMENT "Building artifact core-runtime"
)

# Artifact: wsl-rocdxg
add_custom_target(artifact-wsl-rocdxg
  COMMENT "Building artifact wsl-rocdxg"
)

# Artifact: core-amdsmi
add_custom_target(artifact-core-amdsmi
  COMMENT "Building artifact core-amdsmi"
)

# Artifact: core-kpack
add_custom_target(artifact-core-kpack
  COMMENT "Building artifact core-kpack"
)

# Artifact: core-hip
add_custom_target(artifact-core-hip
  COMMENT "Building artifact core-hip"
)

# Artifact: core-ocl-icd
add_custom_target(artifact-core-ocl-icd
  COMMENT "Building artifact core-ocl-icd"
)

# Artifact: core-ocl
add_custom_target(artifact-core-ocl
  COMMENT "Building artifact core-ocl"
)

# Artifact: rocrtst
add_custom_target(artifact-rocrtst
  COMMENT "Building artifact rocrtst"
)

# Artifact: kfdtest
add_custom_target(artifact-kfdtest
  COMMENT "Building artifact kfdtest"
)

# Artifact: core-hipinfo
add_custom_target(artifact-core-hipinfo
  COMMENT "Building artifact core-hipinfo"
)

# Artifact: core-hiptests
add_custom_target(artifact-core-hiptests
  COMMENT "Building artifact core-hiptests"
)

# Artifact: blas
add_custom_target(artifact-blas
  COMMENT "Building artifact blas"
)

# Artifact: rand
add_custom_target(artifact-rand
  COMMENT "Building artifact rand"
)

# Artifact: fft
add_custom_target(artifact-fft
  COMMENT "Building artifact fft"
)

# Artifact: prim
add_custom_target(artifact-prim
  COMMENT "Building artifact prim"
)

# Artifact: sparse
add_custom_target(artifact-sparse
  COMMENT "Building artifact sparse"
)

# Artifact: solver
add_custom_target(artifact-solver
  COMMENT "Building artifact solver"
)

# Artifact: rocalution
add_custom_target(artifact-rocalution
  COMMENT "Building artifact rocalution"
)

# Artifact: rocwmma
add_custom_target(artifact-rocwmma
  COMMENT "Building artifact rocwmma"
)

# Artifact: composable-kernel
add_custom_target(artifact-composable-kernel
  COMMENT "Building artifact composable-kernel"
)

# Artifact: hiptensor
add_custom_target(artifact-hiptensor
  COMMENT "Building artifact hiptensor"
)

# Artifact: libhipcxx
add_custom_target(artifact-libhipcxx
  COMMENT "Building artifact libhipcxx"
)

# Artifact: hipthreads
add_custom_target(artifact-hipthreads
  COMMENT "Building artifact hipthreads"
)

# Artifact: support
add_custom_target(artifact-support
  COMMENT "Building artifact support"
)

# Artifact: miopen
add_custom_target(artifact-miopen
  COMMENT "Building artifact miopen"
)

# Artifact: hipdnn
add_custom_target(artifact-hipdnn
  COMMENT "Building artifact hipdnn"
)

# Artifact: hipdnn-integration-tests
add_custom_target(artifact-hipdnn-integration-tests
  COMMENT "Building artifact hipdnn-integration-tests"
)

# Artifact: miopenprovider
add_custom_target(artifact-miopenprovider
  COMMENT "Building artifact miopenprovider"
)

# Artifact: hipblasltprovider
add_custom_target(artifact-hipblasltprovider
  COMMENT "Building artifact hipblasltprovider"
)

# Artifact: hipdnn-samples
add_custom_target(artifact-hipdnn-samples
  COMMENT "Building artifact hipdnn-samples"
)

# Artifact: hipkernelprovider
add_custom_target(artifact-hipkernelprovider
  COMMENT "Building artifact hipkernelprovider"
)

# Artifact: rpp
add_custom_target(artifact-rpp
  COMMENT "Building artifact rpp"
)

# Artifact: rocdecode
add_custom_target(artifact-rocdecode
  COMMENT "Building artifact rocdecode"
)

# Artifact: rocjpeg
add_custom_target(artifact-rocjpeg
  COMMENT "Building artifact rocjpeg"
)

# Artifact: rccl
add_custom_target(artifact-rccl
  COMMENT "Building artifact rccl"
)

# Artifact: rocshmem
add_custom_target(artifact-rocshmem
  COMMENT "Building artifact rocshmem"
)

# Artifact: hipfile
add_custom_target(artifact-hipfile
  COMMENT "Building artifact hipfile"
)

# Artifact: aqlprofile
add_custom_target(artifact-aqlprofile
  COMMENT "Building artifact aqlprofile"
)

# Artifact: rocprofiler-sdk
add_custom_target(artifact-rocprofiler-sdk
  COMMENT "Building artifact rocprofiler-sdk"
)

# Artifact: rocprofiler-compute
add_custom_target(artifact-rocprofiler-compute
  COMMENT "Building artifact rocprofiler-compute"
)

# Artifact: rocprofiler-systems
add_custom_target(artifact-rocprofiler-systems
  COMMENT "Building artifact rocprofiler-systems"
)

# Artifact: rocprofiler-systems-examples
add_custom_target(artifact-rocprofiler-systems-examples
  COMMENT "Building artifact rocprofiler-systems-examples"
)

# Artifact: rdc
add_custom_target(artifact-rdc
  COMMENT "Building artifact rdc"
)

# Artifact: amd-dbgapi
add_custom_target(artifact-amd-dbgapi
  COMMENT "Building artifact amd-dbgapi"
)

# Artifact: rocr-debug-agent
add_custom_target(artifact-rocr-debug-agent
  COMMENT "Building artifact rocr-debug-agent"
)

# Artifact: rocr-debug-agent-tests
add_custom_target(artifact-rocr-debug-agent-tests
  COMMENT "Building artifact rocr-debug-agent-tests"
)

# Artifact: rocgdb
add_custom_target(artifact-rocgdb
  COMMENT "Building artifact rocgdb"
)

# Artifact: rocjitsu
add_custom_target(artifact-rocjitsu
  COMMENT "Building artifact rocjitsu"
)

# Artifact: rocjitsu-hotswap
add_custom_target(artifact-rocjitsu-hotswap
  COMMENT "Building artifact rocjitsu-hotswap"
)

# Artifact: mirage
add_custom_target(artifact-mirage
  COMMENT "Building artifact mirage"
)

# =============================================================================
# Artifact group targets
# =============================================================================

# Artifact group: third-party-sysdeps
add_custom_target(artifact-group-third-party-sysdeps
  COMMENT "Building artifact group third-party-sysdeps"
  DEPENDS
    artifact-sysdeps
    artifact-sysdeps-expat
    artifact-sysdeps-gmp
    artifact-sysdeps-mpfr
    artifact-sysdeps-ncurses
    artifact-sysdeps-libmnl
    artifact-sysdeps-util-linux
    artifact-sysdeps-libnl
    artifact-sysdeps-libpciaccess
    artifact-sysdeps-hwloc
)

# Artifact group: third-party-libs
add_custom_target(artifact-group-third-party-libs
  COMMENT "Building artifact group third-party-libs"
  DEPENDS
    artifact-host-blas
    artifact-host-suite-sparse
    artifact-elfio
    artifact-fftw3
    artifact-flatbuffers
    artifact-fmt
    artifact-nlohmann-json
    artifact-spdlog
    artifact-openmpi
)

# Artifact group: base
add_custom_target(artifact-group-base
  COMMENT "Building artifact group base"
  DEPENDS
    artifact-base
)

# Artifact group: core-runtime
add_custom_target(artifact-group-core-runtime
  COMMENT "Building artifact group core-runtime"
  DEPENDS
    artifact-core-runtime
)

# Artifact group: wsl-rocdxg
add_custom_target(artifact-group-wsl-rocdxg
  COMMENT "Building artifact group wsl-rocdxg"
  DEPENDS
    artifact-wsl-rocdxg
)

# Artifact group: core-amdsmi
add_custom_target(artifact-group-core-amdsmi
  COMMENT "Building artifact group core-amdsmi"
  DEPENDS
    artifact-core-amdsmi
)

# Artifact group: compiler
add_custom_target(artifact-group-compiler
  COMMENT "Building artifact group compiler"
  DEPENDS
    artifact-amd-llvm
    artifact-hipify
)

# Artifact group: debug-tools
add_custom_target(artifact-group-debug-tools
  COMMENT "Building artifact group debug-tools"
  DEPENDS
    artifact-amd-dbgapi
    artifact-rocr-debug-agent
    artifact-rocr-debug-agent-tests
    artifact-rocgdb
)

# Artifact group: hip-runtime
add_custom_target(artifact-group-hip-runtime
  COMMENT "Building artifact group hip-runtime"
  DEPENDS
    artifact-core-kpack
    artifact-core-hip
    artifact-core-hipinfo
)

# Artifact group: opencl-runtime
add_custom_target(artifact-group-opencl-runtime
  COMMENT "Building artifact group opencl-runtime"
  DEPENDS
    artifact-core-ocl-icd
    artifact-core-ocl
)

# Artifact group: runtime-tests
add_custom_target(artifact-group-runtime-tests
  COMMENT "Building artifact group runtime-tests"
  DEPENDS
    artifact-rocrtst
    artifact-core-hiptests
)

# Artifact group: kfdtest
add_custom_target(artifact-group-kfdtest
  COMMENT "Building artifact group kfdtest"
  DEPENDS
    artifact-kfdtest
)

# Artifact group: math-libs
add_custom_target(artifact-group-math-libs
  COMMENT "Building artifact group math-libs"
  DEPENDS
    artifact-blas
    artifact-rand
    artifact-fft
    artifact-prim
    artifact-sparse
    artifact-solver
    artifact-rocalution
    artifact-rocwmma
    artifact-composable-kernel
    artifact-hiptensor
    artifact-libhipcxx
    artifact-hipthreads
    artifact-support
)

# Artifact group: ml-libs
add_custom_target(artifact-group-ml-libs
  COMMENT "Building artifact group ml-libs"
  DEPENDS
    artifact-miopen
    artifact-hipdnn
    artifact-hipdnn-integration-tests
    artifact-miopenprovider
    artifact-hipblasltprovider
    artifact-hipdnn-samples
    artifact-hipkernelprovider
)

# Artifact group: comm-libs
add_custom_target(artifact-group-comm-libs
  COMMENT "Building artifact group comm-libs"
  DEPENDS
    artifact-rccl
    artifact-rocshmem
)

# Artifact group: storage-libs
add_custom_target(artifact-group-storage-libs
  COMMENT "Building artifact group storage-libs"
  DEPENDS
    artifact-hipfile
)

# Artifact group: profiler-core
add_custom_target(artifact-group-profiler-core
  COMMENT "Building artifact group profiler-core"
  DEPENDS
    artifact-aqlprofile
    artifact-rocprofiler-sdk
    artifact-rocprofiler-compute
)

# Artifact group: dctools-core
add_custom_target(artifact-group-dctools-core
  COMMENT "Building artifact group dctools-core"
  DEPENDS
    artifact-rdc
)

# Artifact group: profiler-apps
add_custom_target(artifact-group-profiler-apps
  COMMENT "Building artifact group profiler-apps"
  DEPENDS
    artifact-rocprofiler-systems
    artifact-rocprofiler-systems-examples
)

# Artifact group: cv-libs
add_custom_target(artifact-group-cv-libs
  COMMENT "Building artifact group cv-libs"
  DEPENDS
    artifact-rpp
)

# Artifact group: media-libs
add_custom_target(artifact-group-media-libs
  COMMENT "Building artifact group media-libs"
  DEPENDS
    artifact-sysdeps-amd-mesa
    artifact-rocdecode
    artifact-rocjpeg
)

# Artifact group: rocjitsu
add_custom_target(artifact-group-rocjitsu
  COMMENT "Building artifact group rocjitsu"
  DEPENDS
    artifact-rocjitsu
    artifact-rocjitsu-hotswap
    artifact-mirage
)

# =============================================================================
# Build stage targets
# =============================================================================

# Build stage: compiler-runtime
# Type: generic
# Description: Compiler, runtimes, and core profiling
add_custom_target(stage-compiler-runtime
  COMMENT "Building stage compiler-runtime"
  DEPENDS
    artifact-group-third-party-sysdeps
    artifact-group-base
    artifact-group-compiler
    artifact-group-core-amdsmi
    artifact-group-core-runtime
    artifact-group-third-party-libs
    artifact-group-hip-runtime
    artifact-group-kfdtest
    artifact-group-opencl-runtime
    artifact-group-profiler-core
    artifact-group-rocjitsu
)

# Build stage: wsl-rocdxg
# Type: generic
# Description: WSL-only ROCDXG bridge library
add_custom_target(stage-wsl-rocdxg
  COMMENT "Building stage wsl-rocdxg"
  DEPENDS
    artifact-group-wsl-rocdxg
)

# Build stage: runtime-tests
# Type: generic
# Description: Runtime tests (parallel to math-libs, not on critical path)
add_custom_target(stage-runtime-tests
  COMMENT "Building stage runtime-tests"
  DEPENDS
    artifact-group-runtime-tests
)

# Build stage: math-libs
# Type: per-arch
# Description: Math and ML libraries per architecture
add_custom_target(stage-math-libs
  COMMENT "Building stage math-libs"
  DEPENDS
    artifact-group-math-libs
    artifact-group-ml-libs
)

# Build stage: comm-libs
# Type: generic
# Description: Communication libraries (generic, single job parallel to math-libs)
add_custom_target(stage-comm-libs
  COMMENT "Building stage comm-libs"
  DEPENDS
    artifact-group-comm-libs
)

# Build stage: cv-libs
# Type: generic
# Description: Computer Vision Libraries
add_custom_target(stage-cv-libs
  COMMENT "Building stage cv-libs"
  DEPENDS
    artifact-group-cv-libs
)

# Build stage: storage-libs
# Type: generic
# Description: Storage libraries (generic, single job parallel to math-libs)
add_custom_target(stage-storage-libs
  COMMENT "Building stage storage-libs"
  DEPENDS
    artifact-group-storage-libs
)

# Build stage: debug-tools
# Type: generic
# Description: ROCm debug tools
add_custom_target(stage-debug-tools
  COMMENT "Building stage debug-tools"
  DEPENDS
    artifact-group-debug-tools
)

# Build stage: dctools-core
# Type: generic
# Description: Data center tools with minimal dependencies
add_custom_target(stage-dctools-core
  COMMENT "Building stage dctools-core"
  DEPENDS
    artifact-group-dctools-core
)

# Build stage: profiler-apps
# Type: generic
# Description: Profiler applications (depends on profiler-core)
add_custom_target(stage-profiler-apps
  COMMENT "Building stage profiler-apps"
  DEPENDS
    artifact-group-profiler-apps
)

# Build stage: media-libs
# Type: generic
# Description: Media Libraries
add_custom_target(stage-media-libs
  COMMENT "Building stage media-libs"
  DEPENDS
    artifact-group-media-libs
)

# =============================================================================
# Dependency information
# =============================================================================

# Stage compiler-runtime - produced artifacts
set(THEROCK_STAGE_COMPILER_RUNTIME_ARTIFACTS
  amd-llvm
  aqlprofile
  base
  core-amdsmi
  core-hip
  core-hipinfo
  core-kpack
  core-ocl
  core-ocl-icd
  core-runtime
  elfio
  fftw3
  flatbuffers
  fmt
  hipify
  host-blas
  host-suite-sparse
  kfdtest
  mirage
  nlohmann-json
  openmpi
  rocjitsu
  rocjitsu-hotswap
  rocprofiler-compute
  rocprofiler-sdk
  spdlog
  sysdeps
  sysdeps-expat
  sysdeps-gmp
  sysdeps-hwloc
  sysdeps-libmnl
  sysdeps-libnl
  sysdeps-libpciaccess
  sysdeps-mpfr
  sysdeps-ncurses
  sysdeps-util-linux
)

# Stage compiler-runtime - inbound artifacts
set(THEROCK_STAGE_COMPILER_RUNTIME_DEPS
)

# Stage wsl-rocdxg - produced artifacts
set(THEROCK_STAGE_WSL_ROCDXG_ARTIFACTS
  wsl-rocdxg
)

# Stage wsl-rocdxg - inbound artifacts
set(THEROCK_STAGE_WSL_ROCDXG_DEPS
  amd-llvm
  base
  core-hip
  core-kpack
  core-runtime
  sysdeps
)

# Stage runtime-tests - produced artifacts
set(THEROCK_STAGE_RUNTIME_TESTS_ARTIFACTS
  core-hiptests
  rocrtst
)

# Stage runtime-tests - inbound artifacts
set(THEROCK_STAGE_RUNTIME_TESTS_DEPS
  amd-llvm
  base
  core-amdsmi
  core-hip
  core-hipinfo
  core-kpack
  core-ocl
  core-ocl-icd
  core-runtime
  sysdeps
  sysdeps-hwloc
  sysdeps-libmnl
  sysdeps-libnl
  sysdeps-libpciaccess
)

# Stage math-libs - produced artifacts
set(THEROCK_STAGE_MATH_LIBS_ARTIFACTS
  blas
  composable-kernel
  fft
  hipblasltprovider
  hipdnn
  hipdnn-integration-tests
  hipdnn-samples
  hipkernelprovider
  hiptensor
  hipthreads
  libhipcxx
  miopen
  miopenprovider
  prim
  rand
  rocalution
  rocwmma
  solver
  sparse
  support
)

# Stage math-libs - inbound artifacts
set(THEROCK_STAGE_MATH_LIBS_DEPS
  amd-llvm
  aqlprofile
  base
  core-amdsmi
  core-hip
  core-hipinfo
  core-kpack
  core-runtime
  elfio
  fftw3
  host-blas
  host-suite-sparse
  openmpi
  rocprofiler-compute
  rocprofiler-sdk
  spdlog
  sysdeps
  sysdeps-libmnl
  sysdeps-libnl
)

# Stage comm-libs - produced artifacts
set(THEROCK_STAGE_COMM_LIBS_ARTIFACTS
  rccl
  rocshmem
)

# Stage comm-libs - inbound artifacts
set(THEROCK_STAGE_COMM_LIBS_DEPS
  amd-llvm
  aqlprofile
  base
  core-amdsmi
  core-hip
  core-hipinfo
  core-kpack
  core-runtime
  elfio
  hipify
  openmpi
  rocjitsu
  rocjitsu-hotswap
  rocprofiler-sdk
  sysdeps
  sysdeps-libmnl
  sysdeps-libnl
)

# Stage cv-libs - produced artifacts
set(THEROCK_STAGE_CV_LIBS_ARTIFACTS
  rpp
)

# Stage cv-libs - inbound artifacts
set(THEROCK_STAGE_CV_LIBS_DEPS
  amd-llvm
  base
  core-hip
  core-hipinfo
  core-kpack
  core-runtime
  sysdeps
  sysdeps-expat
  sysdeps-gmp
  sysdeps-hwloc
  sysdeps-libmnl
  sysdeps-libnl
  sysdeps-libpciaccess
  sysdeps-mpfr
  sysdeps-ncurses
  sysdeps-util-linux
)

# Stage storage-libs - produced artifacts
set(THEROCK_STAGE_STORAGE_LIBS_ARTIFACTS
  hipfile
)

# Stage storage-libs - inbound artifacts
set(THEROCK_STAGE_STORAGE_LIBS_DEPS
  amd-llvm
  base
  core-hip
  core-hipinfo
  core-kpack
  core-runtime
  sysdeps
  sysdeps-util-linux
)

# Stage debug-tools - produced artifacts
set(THEROCK_STAGE_DEBUG_TOOLS_ARTIFACTS
  amd-dbgapi
  rocgdb
  rocr-debug-agent
  rocr-debug-agent-tests
)

# Stage debug-tools - inbound artifacts
set(THEROCK_STAGE_DEBUG_TOOLS_DEPS
  amd-llvm
  base
  core-hip
  core-kpack
  core-runtime
  hipify
  sysdeps
  sysdeps-expat
  sysdeps-gmp
  sysdeps-hwloc
  sysdeps-libmnl
  sysdeps-libnl
  sysdeps-libpciaccess
  sysdeps-mpfr
  sysdeps-ncurses
  sysdeps-util-linux
)

# Stage dctools-core - produced artifacts
set(THEROCK_STAGE_DCTOOLS_CORE_ARTIFACTS
  rdc
)

# Stage dctools-core - inbound artifacts
set(THEROCK_STAGE_DCTOOLS_CORE_DEPS
  amd-llvm
  aqlprofile
  base
  core-amdsmi
  core-hip
  core-kpack
  core-runtime
  elfio
  openmpi
  rocprofiler-compute
  rocprofiler-sdk
  sysdeps
  sysdeps-libmnl
  sysdeps-libnl
)

# Stage profiler-apps - produced artifacts
set(THEROCK_STAGE_PROFILER_APPS_ARTIFACTS
  rocprofiler-systems
  rocprofiler-systems-examples
)

# Stage profiler-apps - inbound artifacts
set(THEROCK_STAGE_PROFILER_APPS_DEPS
  amd-llvm
  aqlprofile
  base
  core-amdsmi
  core-hip
  core-kpack
  core-runtime
  elfio
  hipify
  openmpi
  rocprofiler-compute
  rocprofiler-sdk
  spdlog
  sysdeps
  sysdeps-libmnl
  sysdeps-libnl
)

# Stage media-libs - produced artifacts
set(THEROCK_STAGE_MEDIA_LIBS_ARTIFACTS
  rocdecode
  rocjpeg
  sysdeps-amd-mesa
)

# Stage media-libs - inbound artifacts
set(THEROCK_STAGE_MEDIA_LIBS_DEPS
  amd-llvm
  base
  core-hip
  core-hipinfo
  core-kpack
  core-runtime
  sysdeps
  sysdeps-expat
  sysdeps-gmp
  sysdeps-hwloc
  sysdeps-libmnl
  sysdeps-libnl
  sysdeps-libpciaccess
  sysdeps-mpfr
  sysdeps-ncurses
  sysdeps-util-linux
)

# =============================================================================
# Build order
# =============================================================================

# Stages in dependency order:
#   1. compiler-runtime
#   2. wsl-rocdxg
#   3. runtime-tests
#   4. math-libs
#   5. comm-libs
#   6. cv-libs
#   7. storage-libs
#   8. debug-tools
#   9. dctools-core
#   10. profiler-apps
#   11. media-libs

set(THEROCK_BUILD_ORDER
  compiler-runtime
  wsl-rocdxg
  runtime-tests
  math-libs
  comm-libs
  cv-libs
  storage-libs
  debug-tools
  dctools-core
  profiler-apps
  media-libs
)

