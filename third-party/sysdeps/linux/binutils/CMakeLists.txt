# if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
#     # When included in TheRock, we download sources and set up the sub-project.
#     set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
#     set(_download_stamp "${_source_dir}/download.stamp")

#     therock_subproject_fetch(therock-elfutils-sources
#     SOURCE_DIR "${_source_dir}"
#     # Originally mirrored from: http://ftpmirror.gnu.org/gnu/binutils/binutils-2.45.tar.gz
#     URL http://ftpmirror.gnu.org/gnu/binutils/binutils-2.42.tar.gz
#     URL_HASH "SHA256=5d2a6c1d49686a557869caae08b6c2e83699775efd27505e01b2f4db1a024ffc"
#     TOUCH "${_download_stamp}"
#   )

#   therock_cmake_subproject_declare(therock-elfutils
#     EXTERNAL_SOURCE_DIR .
#     BINARY_DIR build
#     NO_MERGE_COMPILE_COMMANDS
#     BACKGROUND_BUILD
#     OUTPUT_ON_FAILURE
#     CMAKE_ARGS
#         "-DSOURCE_DIR=${_source_dir}"
#         "-DPATCHELF=${PATCHELF}"
#         "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
#     INSTALL_DESTINATION
#         lib/rocm_sysdeps
#     INTERFACE_LINK_DIRS
#         lib/rocm_sysdeps/lib
#     INTERFACE_INSTALL_RPATH_DIRS
#         lib/rocm_sysdeps/lib
#     INTERFACE_PKG_CONFIG_DIRS
#         lib/rocm_sysdeps/lib/pkgconfig
#     EXTRA_DEPENDS
#         "${_download_stamp}"
#     )
#     therock_cmake_subproject_provide_package(therock-binutils libiberty lib/rocm_sysdeps/lib/cmake/libiberty)
#     therock_cmake_subproject_activate(therock-binutils)

#     add_dependencies(therock-third-party therock-binutils)
#     return()
# endif()

# # Otherwise, this is the sub-project build.
# cmake_minimum_required(VERSION 3.25)
# project(BINUTILS_BUILD)

# include(ProcessorCount)
# ProcessorCount(PAR_JOBS)

# if(PATCHELF)
#   message(FATAL_ERROR "Missing PATCHELF from super-project")
# endif()

# set(EXTRA_CPPFLAGS)
# string(APPEND EXTRA_CPPFLAGS " -fPIC")
# message(STATUS "EXTRA_CPPFLAGS=${EXTRA_CPPFLAGS}")

# set(CONFIG_FLAGS "--with-system-zlib=yes --without-zstd")
# message(STATUS "CONFIG_FLAGS=${CONFIG_FLAGS}")

# add_custom_target(
#   build ALL
#   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#   COMMAND
#     "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
#   COMMAND
#     "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
#   COMMAND
#     "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
#   COMMAND
#     # CMake copy directory does not preserve timestamps. If the stamp-po is
#     # out of date, the build will try to regenerate docs, which requires a bunch
#     # of extra tools and fails mysteriously if not present with something like:
#     #   mv: cannot stat 't-de.gmo': No such file or directory
#     # Ensuring it has the most current timestamp post copy/patch disables doc
#     # regneration.
#     "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/s/po/stamp-po"
#   COMMAND
#     "${CMAKE_COMMAND}" -E env
#       "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
#       # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
#       # then escaped single quotes to make it to the linker command line.
#       "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\'"
#       "CPPFLAGS=${EXTRA_CPPFLAGS}"
#       --
#     "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
#       --prefix "${CMAKE_INSTALL_PREFIX}"
#       "${CONFIG_FLAGS}"
#   COMMAND
#     make -j "${PAR_JOBS}" V=1
#   COMMAND
#     make install
#   COMMAND
#     "${CMAKE_COMMAND}" -E env
#       "PATCHELF=${PATCHELF}"
#       "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
#       "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
#     "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
#   DEPENDS
#     "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh"
#     "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
# )
