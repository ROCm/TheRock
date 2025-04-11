therock_subproject_fetch(therock-googletest-sources
  CMAKE_PROJECT
  URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip  # v1.16.0
  URL_HASH SHA256=adfdfad18a802c8d00ef32e0587cac3ae8786910688024e0264d619e2c266718
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

add_dependencies(therock-third-party therock-googletest)
