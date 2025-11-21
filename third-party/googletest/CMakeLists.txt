therock_subproject_fetch(therock-googletest-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/googletest-1.16.0.tar.gz
  URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
)

therock_cmake_subproject_declare(therock-googletest
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -Dgtest_force_shared_crt=ON
)
therock_cmake_subproject_provide_package(therock-googletest GTest lib/cmake/GTest)
therock_cmake_subproject_activate(therock-googletest)
