therock_subproject_fetch(therock-fftwf-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://fftw.org/fftw-3.3.10.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fftw-3.3.10.tar.gz
  URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
)

therock_cmake_subproject_declare(therock-fftwf
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DENABLE_FLOAT=ON
    -DENABLE_DOUBLE=ON
    -DBUILD_SHARED_LIBS=ON
)
therock_cmake_subproject_provide_package(therock-fftwf FFTW3f lib/cmake/fftw3f)
therock_cmake_subproject_activate(therock-fftwf)

add_dependencies(therock-third-party therock-fftwf)

therock_provide_artifact(fftwf
  DESCRIPTOR artifact.toml
  TARGET_NEUTRAL
  COMPONENTS
    dbg
    dev
    doc
    lib
    run
  SUBPROJECT_DEPS therock-fftwf
)
