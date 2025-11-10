if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, we download sources and set up the sub-project.
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-grpc-sources
    SOURCE_DIR "${_source_dir}"
    GIT_REPOSITORY "https://github.com/grpc/grpc.git"
    GIT_TAG "v1.76.0"
    GIT_SHALLOW ON
    GIT_SUBMODULES_RECURSE ON
    TOUCH "${_download_stamp}"
  )

  therock_cmake_subproject_declare(therock-grpc
    EXTERNAL_SOURCE_DIR .
    BINARY_DIR build
    NO_MERGE_COMPILE_COMMANDS
    BACKGROUND_BUILD
    OUTPUT_ON_FAILURE
    CMAKE_ARGS
      "-DSOURCE_DIR=${_source_dir}"
      "-DPATCHELF=${PATCHELF}"
      "-DTHEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
    RUNTIME_DEPS
      therock-zlib
    INSTALL_DESTINATION
      lib/rocm_sysdeps
    INTERFACE_INCLUDE_DIRS
      lib/rocm_sysdeps/include
    INTERFACE_LINK_DIRS
      lib/rocm_sysdeps/lib
    INTERFACE_INSTALL_RPATH_DIRS
      lib/rocm_sysdeps/lib
    INTERFACE_PKG_CONFIG_DIRS
      lib/rocm_sysdeps/lib/pkgconfig
    EXTRA_DEPENDS
      "${_download_stamp}"
  )
  therock_cmake_subproject_provide_package(therock-grpc gRPC lib/rocm_sysdeps/lib/cmake/grpc)
  therock_cmake_subproject_provide_package(therock-grpc protobuf lib/rocm_sysdeps/lib/cmake/protobuf)
  therock_cmake_subproject_activate(therock-grpc)

  # Validate that key static libraries are built correctly
  foreach(lib_name libgrpc++.a libgrpc.a libprotobuf.a)
    add_test(
      NAME therock-validate-static-lib-${lib_name}
      COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/validate_static_library.sh"
          "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/${lib_name}"
    )
  endforeach()
  return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(GRPC_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

find_package(ZLIB REQUIRED CONFIG)

if(NOT TARGET ZLIB::ZLIB)
  message(FATAL_ERROR "ZLIB::ZLIB target not found. Please ensure therock-zlib is built first.")
endif()

get_target_property(ZLIB_INCLUDE_DIR ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(ZLIB_LIBRARY ZLIB::ZLIB IMPORTED_LOCATION)

get_filename_component(ZLIB_LIB_DIR "${ZLIB_LIBRARY}" DIRECTORY)
get_filename_component(ZLIB_ROOT "${ZLIB_LIB_DIR}/.." ABSOLUTE)

message(STATUS "Found ZLIB via find_package()")
message(STATUS "  ZLIB_ROOT: ${ZLIB_ROOT}")
message(STATUS "  ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")
message(STATUS "  ZLIB_LIBRARY: ${ZLIB_LIBRARY}")

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s" "${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}"
      "-G${CMAKE_GENERATOR}"
      "-S${CMAKE_CURRENT_BINARY_DIR}/s"
      "-B${CMAKE_CURRENT_BINARY_DIR}/b"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_CXX_STANDARD=17"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
      # Use already-built zlib instead of rebuilding
      "-DgRPC_ZLIB_PROVIDER=package"
      "-DZLIB_ROOT=${ZLIB_ROOT}"
      "-DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIR}"
      "-DZLIB_LIBRARY=${ZLIB_LIBRARY}"
      # Build from submodules
      "-DgRPC_ABSL_PROVIDER=module"
      "-DgRPC_CARES_PROVIDER=module"
      "-DgRPC_RE2_PROVIDER=module"
      "-DgRPC_SSL_PROVIDER=module"
      "-DgRPC_PROTOBUF_PROVIDER=module"
      # Build options
      "-DgRPC_BUILD_TESTS=OFF"
      "-DgRPC_BUILD_CSHARP_EXT=OFF"
      "-DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF"
      "-DgRPC_INSTALL=ON"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      # Set RPATH for installed libraries and executables
      "-DCMAKE_INSTALL_RPATH=\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../../../../../../zlib/build/stage/lib/rocm_sysdeps/lib"
      "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
      # Add build-time RPATH to find zlib during gRPC build/test phase
      # This is separate from install RPATH and only used during build
      "-DCMAKE_BUILD_RPATH=${ZLIB_ROOT}/lib"
      "-DCMAKE_SKIP_BUILD_RPATH=OFF"
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/b" -j "${PAR_JOBS}"
  COMMAND
    "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
  VERBATIM
)
