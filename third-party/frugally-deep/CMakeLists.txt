therock_subproject_fetch(therock-frugally-deep-sources
  CMAKE_PROJECT
  # Use frugally-deep 0.15.x for MIOpen until the following issue is fixed
  # https://github.com/ROCm/MIOpen/issues/3588
  # Originally mirrored from: https://github.com/Dobiasd/frugally-deep/archive/refs/tags/v0.15.31.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/frugally-deep-0.15.31.tar.gz
  URL_HASH SHA256=49bf5e30ad2d33e464433afbc8b6fe8536fc959474004a1ce2ac03d7c54bc8ba
)

therock_cmake_subproject_declare(therock-frugally-deep
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BUILD_DEPS
    therock-eigen
    therock-nlohmann-json
    therock-FunctionalPlus
  CMAKE_ARGS
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
therock_cmake_subproject_provide_package(
  therock-frugally-deep frugally-deep lib/cmake/frugally-deep)
therock_cmake_subproject_activate(therock-frugally-deep)

add_dependencies(therock-third-party therock-frugally-deep)
