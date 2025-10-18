therock_subproject_fetch(therock-SuiteSparse-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.8.3.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/SuiteSparse-7.8.3.tar.gz
  URL_HASH SHA256=ce39b28d4038a09c14f21e02c664401be73c0cb96a9198418d6a98a7db73a259
)

therock_cmake_subproject_declare(therock-SuiteSparse
  TARGET_NEUTRAL
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  INSTALL_DESTINATION "lib/host-math"
  # RPATH logic needs to know that executables/libs for this project are in
  # a non-default location.
  INSTALL_RPATH_EXECUTABLE_DIR "lib/host-math/bin"
  INSTALL_RPATH_LIBRARY_DIR "lib/host-math/lib"
  # We both have inter-library deps and need to advertise that consumers need
  # their RPATH to include our locations.
  INSTALL_RPATH_DIRS "lib/host-math/lib"
  INTERFACE_INSTALL_RPATH_DIRS "lib/host-math/lib"
  CMAKE_ARGS
    "-DSUITESPARSE_ENABLE_PROJECTS=\"suitesparse_config;cholmod\""
    -DBUILD_STATIC_LIBS=OFF
    -DSUITESPARSE_USE_CUDA=OFF
    -DSUITESPARSE_USE_OPENMP=OFF
    -DSUITESPARSE_USE_FORTRAN=OFF
    -DSUITESPARSE_USE_64BIT_BLAS=ON
  RUNTIME_DEPS
    therock-host-blas
)
therock_cmake_subproject_provide_package(
  therock-SuiteSparse AMD lib/host-math/lib/cmake/AMD)
therock_cmake_subproject_provide_package(
  therock-SuiteSparse CAMD lib/host-math/lib/cmake/CAMD)
therock_cmake_subproject_provide_package(
  therock-SuiteSparse CCOLAMD lib/host-math/lib/cmake/CCOLAMD)
therock_cmake_subproject_provide_package(
  therock-SuiteSparse CHOLMOD lib/host-math/lib/cmake/CHOLMOD)
therock_cmake_subproject_provide_package(
  therock-SuiteSparse COLAMD lib/host-math/lib/cmake/COLAMD)
therock_cmake_subproject_provide_package(
  therock-SuiteSparse SuiteSparse_config lib/host-math/lib/cmake/SuiteSparse_config)
therock_cmake_subproject_activate(therock-SuiteSparse)

therock_test_validate_shared_lib(
  PATH dist/lib/host-math/lib
  LIB_NAMES
    libamd.so libcamd.so libccolamd.so libcholmod.so libcolamd.so
    libsuitesparseconfig.so
)

therock_provide_artifact(host-suite-sparse
  DESCRIPTOR artifact-host-suite-sparse.toml
  TARGET_NEUTRAL
  COMPONENTS
    dbg
    dev
    doc
    lib
    run
  SUBPROJECT_DEPS therock-SuiteSparse
)
