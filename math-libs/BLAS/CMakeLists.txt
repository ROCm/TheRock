# Many things need to be configured with paths to additional LLVM tools.
# If a project is using an amd-hip or amd-llvm toolchain, then it will already
# have an implicit dep on the toolchain, so it is safe to reference binaries
# here without worrying about build deps.
therock_cmake_subproject_dist_dir(_blas_hack_hip_dir hip-clr)
cmake_path(APPEND _blas_hack_hip_dir lib/llvm OUTPUT_VARIABLE _toolchain_dir)

# Most libraries need to depend on the profiler, but is conditionally only used
# on Posix.
set(optional_profiler_deps)
if(NOT WIN32)
  list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
endif()


##############################################################################
# hipBLAS-common
##############################################################################
set(_blas_subproject_names)

therock_cmake_subproject_declare(hipBLAS-common
EXTERNAL_SOURCE_DIR "hipBLAS-common"
BACKGROUND_BUILD
COMPILER_TOOLCHAIN
  amd-hip
BUILD_DEPS
  rocm-cmake
RUNTIME_DEPS
  hip-clr
)
therock_cmake_subproject_glob_c_sources(hipBLAS-common
SUBDIRS
    library
)
therock_cmake_subproject_provide_package(hipBLAS-common hipblas-common lib/cmake/hipblas-common)
therock_cmake_subproject_activate(hipBLAS-common)
list(APPEND _blas_subproject_names hipBLAS-common)


##############################################################################
# rocRoller
##############################################################################

if(WIN32)
  set(_enable_rocRoller "OFF")
else()
  set(_enable_rocRoller "ON")
endif()

if(_enable_rocRoller)
  therock_cmake_subproject_declare(rocRoller
    EXTERNAL_SOURCE_DIR "rocRoller"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCROLLER_ENABLE_YAML_CPP=ON
      -DROCROLLER_ENABLE_CLIENT=OFF
      -DROCROLLER_BUILD_TESTING=${THEROCK_BUILD_TESTING}
      -DROCROLLER_BUILD_SHARED_LIBS=ON
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      mxDataGenerator
      rocm-cmake
      therock-boost
      therock-catch2
      therock-fmt
      therock-googletest
      therock-libdivide
      therock-msgpack-cxx
      therock-spdlog
      therock-yaml-cpp
      # therock-cl11 # will need this dep to enable client
    RUNTIME_DEPS
      therock-host-blas
      hip-clr
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_glob_c_sources(rocRoller
  SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocRoller rocroller lib/cmake/rocroller)
  therock_cmake_subproject_activate(rocRoller)
  list(APPEND _blas_subproject_names rocRoller)
endif()


##############################################################################
# hipBLASLt
##############################################################################

set(hipBLASLt_optional_deps)
if(NOT WIN32)
  # hipBLASLt is hard-coded to not expect rocm-smi on Windows.
  list(APPEND hipBLASLt_optional_deps
    rocm_smi_lib
  )
endif()

set(hipBLASLt_rocRoller_deps)
if(_enable_rocRoller)
  list(APPEND hipBLASLt_rocRoller_build_deps
    mxDataGenerator
    therock-boost
    therock-fmt
    therock-yaml-cpp
  )
  list(APPEND hipBLASLt_rocRoller_runtime_deps
    rocRoller
  )
endif()

therock_cmake_subproject_declare(hipBLASLt
  EXTERNAL_SOURCE_DIR "hipBLASLt"
  BACKGROUND_BUILD
  CMAKE_LISTS_RELPATH "next-cmake"
  CMAKE_ARGS
    -DHIP_PLATFORM=amd
    -DHIPBLASLT_ENABLE_BLIS=OFF  # TODO: Evaluate
    -DBUILD_TESTING=${THEROCK_BUILD_TESTING}
    -DHIPBLASLT_BUILD_TESTING=${THEROCK_BUILD_TESTING}
    -DHIPBLASLT_ENABLE_ROCROLLER=${_enable_rocRoller}
  CMAKE_INCLUDES
    therock_explicit_finders.cmake
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    hipBLAS-common
    rocm-cmake
    therock-googletest
    therock-msgpack-cxx
    ${hipBLASLt_rocRoller_build_deps}
  RUNTIME_DEPS
    hip-clr
    therock-host-blas
    ${hipBLASLt_rocRoller_runtime_deps}
    ${hipBLASLt_optional_deps}
    ${optional_profiler_deps}
)
therock_cmake_subproject_glob_c_sources(hipBLASLt
SUBDIRS
    .
)
therock_cmake_subproject_provide_package(hipBLASLt hipblaslt lib/cmake/hipblaslt)
therock_cmake_subproject_activate(hipBLASLt)
list(APPEND _blas_subproject_names hipBLASLt)


##############################################################################
# rocBLAS
##############################################################################

set(rocBLAS_optional_runtime_deps)
if(NOT WIN32)
  # rocBLAS is hard-coded to not expect rocm-smi.
  list(APPEND rocBLAS_optional_runtime_deps rocm_smi_lib)
elseif(THEROCK_BUILD_TESTING)
  list(APPEND rocBLAS_optional_runtime_deps therock-host-blas)
endif()

therock_cmake_subproject_declare(rocBLAS
  EXTERNAL_SOURCE_DIR "rocBLAS"
  BACKGROUND_BUILD
  CMAKE_ARGS
    -DHIP_PLATFORM=amd
    -DROCM_PATH=
    -DROCM_DIR=
    -DBUILD_WITH_TENSILE=ON
    -DBUILD_WITH_HIPBLASLT=ON
    # TODO: With `Tensile_TEST_LOCAL_PATH` set, the resulting build path is ${Tensile_TEST_LOCAL_PATH}/build.
    "-DTensile_TEST_LOCAL_PATH=${CMAKE_CURRENT_SOURCE_DIR}/Tensile"
    # Since using a local Tensile vs fetched, unset TENSILE_VERSION to avoid
    # doing a strict check of exact version.
    -DTENSILE_VERSION=
    -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
    -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
    -DLINK_BLIS=OFF
  CMAKE_INCLUDES
    therock_explicit_finders.cmake
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    hipBLAS-common
    rocm-cmake
    therock-googletest
    therock-msgpack-cxx
  RUNTIME_DEPS
    hip-clr
    hipBLASLt
    ${rocBLAS_optional_runtime_deps}
    ${optional_profiler_deps}
)
therock_cmake_subproject_glob_c_sources(rocBLAS
SUBDIRS
    .
)
therock_cmake_subproject_provide_package(rocBLAS rocblas lib/cmake/rocblas)
therock_cmake_subproject_activate(rocBLAS)
list(APPEND _blas_subproject_names rocBLAS)


if(THEROCK_ENABLE_SPARSE)
  ##############################################################################
  # rocSPARSE
  ##############################################################################
  set(_sparse_subproject_names)

  therock_cmake_subproject_declare(rocSPARSE
    EXTERNAL_SOURCE_DIR "rocSPARSE"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_WITH_ROCBLAS=ON
      -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
      -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
      -DBUILD_CLIENTS_SAMPLES=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      rocBLAS
      rocPRIM
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_glob_c_sources(rocSPARSE
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocSPARSE rocsparse lib/cmake/rocsparse)
  therock_cmake_subproject_activate(rocSPARSE)
  list(APPEND _sparse_subproject_names rocSPARSE)

  ##############################################################################
  # hipSPARSE
  ##############################################################################

  therock_cmake_subproject_declare(hipSPARSE
    EXTERNAL_SOURCE_DIR "hipSPARSE"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
      -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
      -DBUILD_CLIENTS_SAMPLES=OFF
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      rocSPARSE
      therock-googletest
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(hipSPARSE
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipSPARSE hipsparse lib/cmake/hipsparse)
  therock_cmake_subproject_activate(hipSPARSE)
  list(APPEND _blas_subproject_names hipSPARSE)
endif(THEROCK_ENABLE_SPARSE)


if(THEROCK_ENABLE_SOLVER)
  ##############################################################################
  # rocSOLVER
  ##############################################################################

  set(rocSOLVER_optional_deps)
  if(THEROCK_BUILD_TESTING)
    list(APPEND rocSOLVER_optional_deps therock-host-blas)
  endif()

  therock_cmake_subproject_declare(rocSOLVER
      EXTERNAL_SOURCE_DIR "rocSOLVER"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
        -DROCM_DIR=
        -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
        -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
        -DROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        rocm-cmake
        rocPRIM
        therock-fmt
        therock-googletest
      RUNTIME_DEPS
        hip-clr
        rocBLAS
        ${rocSOLVER_optional_deps}
  )
  therock_cmake_subproject_glob_c_sources(rocSOLVER
      SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(rocSOLVER rocsolver lib/cmake/rocsolver)
  therock_cmake_subproject_activate(rocSOLVER)
  list(APPEND _solver_subproject_names rocSOLVER)

  ##############################################################################
  # hipSOLVER
  ##############################################################################

  set(hipSOLVER_optional_deps)
  if(THEROCK_BUILD_TESTING)
    list(APPEND hipSOLVER_optional_deps therock-host-blas)
  endif()

  if(WIN32)
    set(hipSOLVER_build_with_sparse "OFF")
  else()
    # Note: BUILD_WITH_SPARSE converts rocSPARSE and cholmod to build time vs
    # runtime/dlopen deps.
    set(hipSOLVER_build_with_sparse "ON")
    list(APPEND hipSOLVER_optional_deps therock-SuiteSparse)
  endif()

  therock_cmake_subproject_declare(hipSOLVER
      EXTERNAL_SOURCE_DIR "hipSOLVER"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
        -DROCM_DIR=
        -DBUILD_HIPSPARSE_TESTS=OFF
        -DBUILD_WITH_SPARSE=${hipSOLVER_build_with_sparse}
        -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
        -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
        -DHIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        rocm-cmake
        therock-googletest
      RUNTIME_DEPS
        hip-clr
        rocBLAS
        rocSOLVER
        rocSPARSE
        ${hipSOLVER_optional_deps}
  )
  therock_cmake_subproject_glob_c_sources(hipSOLVER
      SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(hipSOLVER hipsolver lib/cmake/hipsolver)
  therock_cmake_subproject_activate(hipSOLVER)
  list(APPEND _blas_subproject_names hipSOLVER)
endif(THEROCK_ENABLE_SOLVER)


##############################################################################
# hipBLAS
##############################################################################

set(hipBLAS_optional_deps)
if(THEROCK_ENABLE_SOLVER)
  list(APPEND hipBLAS_optional_deps rocSOLVER)
endif()
if(THEROCK_BUILD_TESTING)
  list(APPEND hipBLAS_optional_deps therock-host-blas)
endif()

therock_cmake_subproject_declare(hipBLAS
  EXTERNAL_SOURCE_DIR "hipBLAS"
  BACKGROUND_BUILD
  CMAKE_ARGS
    -DHIP_PLATFORM=amd
    -DROCM_PATH=
    -DROCM_DIR=
    "-DBUILD_WITH_SOLVER=${THEROCK_ENABLE_SOLVER}"
    -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
    -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
    -DBUILD_CLIENTS_SAMPLES=OFF
    -DLINK_BLIS=OFF
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    hipBLAS-common
    rocm-cmake
    therock-googletest
  RUNTIME_DEPS
    hip-clr
    rocBLAS
    ${hipBLAS_optional_deps}
)
therock_cmake_subproject_glob_c_sources(hipBLAS
SUBDIRS
    .
)
therock_cmake_subproject_provide_package(hipBLAS hipblas lib/cmake/hipblas)
therock_cmake_subproject_activate(hipBLAS)
list(APPEND _blas_subproject_names hipBLAS)


##############################################################################
# blas artifact
##############################################################################

therock_provide_artifact(blas
  DESCRIPTOR artifact-blas.toml
  COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
  SUBPROJECT_DEPS ${_blas_subproject_names}
)
