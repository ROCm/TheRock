therock_subproject_fetch(therock-fusilli-sources
  CMAKE_PROJECT
  SOURCE_DIR "/home/astgeorg/Dev/python/shark-ai/sharkfuser"
)

therock_cmake_subproject_declare(therock-fusilli
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "/home/astgeorg/Dev/python/shark-ai/sharkfuser"
  BUILD_DEPS
    therock-iree-runtime
    therock-catch2
  CMAKE_ARGS
    -DFUSILLI_BUILD_TESTS=OFF
    -DFUSILLI_BUILD_BENCHMARKS=OFF
    -DFUSILLI_SYSTEMS_AMDGPU=ON
    -DFUSILLI_CODE_COVERAGE=OFF
    -DFUSILLI_ENABLE_LOGGING=OFF
    -DFUSILLI_ENABLE_CLANG_TIDY=OFF
)
therock_cmake_subproject_provide_package(therock-fusilli Fusilli lib/cmake/fusilli)
therock_cmake_subproject_activate(therock-fusilli)

add_dependencies(therock-third-party therock-fusilli)