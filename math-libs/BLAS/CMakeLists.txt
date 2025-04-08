# Many things need to be configured with paths to additional LLVM tools.
# If a project is using an amd-hip or amd-llvm toolchain, then it will already
# have an implicit dep on the toolchain, so it is safe to reference binaries
# here without worrying about build deps.
therock_cmake_subproject_dist_dir(_blas_hack_hip_dir hip-clr)
cmake_path(APPEND _blas_hack_hip_dir lib/llvm OUTPUT_VARIABLE _toolchain_dir)

##############################################################################
# hipBLAS-common
##############################################################################
set(_blas_subproject_names)

therock_cmake_subproject_declare(hipBLAS-common
EXTERNAL_SOURCE_DIR "hipBLAS-common"
BACKGROUND_BUILD
CMAKE_ARGS
  -DROCM_PATH=
  -DROCM_DIR=
COMPILER_TOOLCHAIN
  amd-hip
IGNORE_PACKAGES
  # The current version of rccl needs to download a 2y old version of rocm-cmake
  # to work and it will only do so if the system resolver reports it not found
  # without any other error (which due to the signature, our resolver will
  # hard fail). Once fixed, `rocm-core` should be added to the BUILD_DEPS
  # and this removed: https://github.com/ROCm/TheRock/issues/18
  ROCM
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
# hipBLASLt
##############################################################################

set(hipBLASLt_optional_deps)
if(THEROCK_BUILD_TESTING)
  list(APPEND hipBLASLt_optional_deps
    therock-host-blas
    rocm_smi_lib
  )
endif()

therock_cmake_subproject_declare(hipBLASLt
  EXTERNAL_SOURCE_DIR "hipBLASLt"
  BACKGROUND_BUILD
  CMAKE_ARGS
    -DHIP_PLATFORM=amd
    -DROCM_PATH=
    -DROCM_DIR=
    -DTensile_LOGIC=
    -DTensile_CPU_THREADS=
    -DTensile_LIBRARY_FORMAT=msgpack
    # TODO(#55): Enable once roctracer is added
    -DHIPBLASLT_ENABLE_MARKER=OFF
    -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
    -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    hipBLAS-common
    therock-msgpack-cxx
  RUNTIME_DEPS
    hip-clr
    ${hipBLASLt_optional_deps}
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

set(rocBLAS_optional_deps)
if(THEROCK_BUILD_TESTING)
  list(APPEND rocBLAS_optional_deps rocm_smi_lib)
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
  IGNORE_PACKAGES
      ROCM
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    therock-msgpack-cxx
    hipBLAS-common
  RUNTIME_DEPS
    hip-clr
    hipBLASLt
    ${rocBLAS_optional_deps}
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
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocBLAS
      rocPRIM
    RUNTIME_DEPS
      hip-clr
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
      rocSPARSE
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
        rocPRIM
        therock-fmt
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

  therock_cmake_subproject_declare(hipSOLVER
      EXTERNAL_SOURCE_DIR "hipSOLVER"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
        -DROCM_DIR=
        -DBUILD_HIPSPARSE_TESTS=OFF
        # BUILD_WITH_SPARSE converts rocsparse and cholmod to build time vs
        # runtime/dlopen deps.
        -DBUILD_WITH_SPARSE=ON
        -DBUILD_CLIENTS_BENCHMARKS=${THEROCK_BUILD_TESTING}
        -DBUILD_CLIENTS_TESTS=${THEROCK_BUILD_TESTING}
        -DHIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF
      COMPILER_TOOLCHAIN
        amd-hip
      RUNTIME_DEPS
        hip-clr
        rocBLAS
        rocSOLVER
        rocSPARSE
        therock-SuiteSparse
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
