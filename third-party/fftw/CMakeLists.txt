therock_subproject_fetch(therock-fftw-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://fftw.org/fftw-3.3.10.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fftw-3.3.10.tar.gz
  URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
)

therock_cmake_subproject_declare(therock-fftw
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)
therock_cmake_subproject_provide_package(therock-fftw FFTW3 lib/cmake/fftw3)
therock_cmake_subproject_activate(therock-fftw)

add_dependencies(therock-third-party therock-fftw)

therock_provide_artifact(fftw
  DESCRIPTOR artifact.toml
  TARGET_NEUTRAL
  COMPONENTS
    dbg
    dev
    doc
    lib
    run
  SUBPROJECT_DEPS therock-fftw
)
