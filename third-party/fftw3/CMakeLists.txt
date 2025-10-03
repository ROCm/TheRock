therock_subproject_fetch(therock-fftw3-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://fftw.org/fftw-3.3.10.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fftw-3.3.10.tar.gz
  URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_fftw3_enable_threads "ON")
else()
  set(_fftw3_enable_threads "OFF")
endif()

therock_cmake_subproject_declare(therock-fftw3
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build_fftw3"
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DENABLE_THREADS=${_fftw3_enable_threads}
)
therock_cmake_subproject_provide_package(therock-fftw3 FFTW3 lib/cmake/fftw3)
therock_cmake_subproject_activate(therock-fftw3)

add_dependencies(therock-third-party therock-fftw3)

therock_cmake_subproject_declare(therock-fftw3f
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build_fftw3f"
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DENABLE_FLOAT=ON
    -DBUILD_SHARED_LIBS=ON
    -DENABLE_THREADS=${_fftw3_enable_threads}
)
therock_cmake_subproject_provide_package(therock-fftw3f FFTW3f lib/cmake/fftw3f)
therock_cmake_subproject_activate(therock-fftw3f)

add_dependencies(therock-third-party therock-fftw3f)

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
