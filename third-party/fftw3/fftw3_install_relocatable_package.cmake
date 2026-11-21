# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install third-party/fftw3/FFTW3Config.cmake.in into the subproject prefix.
# Included from the therock-fftw3 and therock-fftw3f post hooks, so it runs
# for the upstream CMake build and for the optional Linux autotools build.
#
# Upstream configure_file() writes CMAKE_INSTALL_FULL_INCLUDEDIR into
# FFTW3Config.cmake. That install() is registered first. install(FILES) then
# sees the upstream file as newer and leaves the absolute path in place.
# cmake -E copy always replaces it. FFTW3LibraryDepends.cmake is removed so
# nothing loads the export this config no longer includes.

if(ENABLE_FLOAT)
  set(_package_name FFTW3f)
  set(_library_name fftw3f)
else()
  set(_package_name FFTW3)
  set(_library_name fftw3)
endif()

# Upstream installs libfftw3_threads.so without an EXPORT name. Publish
# FFTW3::fftw3_threads (or the float name) only where that library is built.
set(_threads_library_name "")
if(ENABLE_THREADS AND NOT WIN32)
  set(_threads_library_name "${_library_name}_threads")
endif()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/FFTW3Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/therock-fftw-package/${_package_name}Config.cmake"
  @ONLY
)

# Each install(CODE) script is written as normal CMake, then filled in.
# @NAME@ is replaced now, while this file is being configured. ${name} is
# left alone and runs later, during cmake --install, when _fftw_cmake_dir
# exists. The bracket quotes keep the ${...} from expanding too early.
# Replacing the variable in place is safe: the input is read before the
# result is stored.
set(_fftw_install_config_code [=[
  set(_fftw_cmake_dir "@CMAKE_INSTALL_PREFIX@/lib/cmake/@_library_name@")
  file(MAKE_DIRECTORY "${_fftw_cmake_dir}")
  execute_process(
    COMMAND "@CMAKE_COMMAND@" -E copy
      "@CMAKE_CURRENT_BINARY_DIR@/therock-fftw-package/@_package_name@Config.cmake"
      "${_fftw_cmake_dir}/@_package_name@Config.cmake"
    COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB _fftw_absolute_exports "${_fftw_cmake_dir}/FFTW3LibraryDepends*.cmake")
  if(_fftw_absolute_exports)
    file(REMOVE ${_fftw_absolute_exports})
  endif()
  set(_fftw_config "${_fftw_cmake_dir}/@_package_name@Config.cmake")
  if(NOT EXISTS "${_fftw_config}")
    message(FATAL_ERROR "FFTW @_package_name@ relocatable config was not installed")
  endif()
  file(READ "${_fftw_config}" _fftw_config_text)
  string(FIND "${_fftw_config_text}" "@CMAKE_INSTALL_PREFIX@" _fftw_prefix_at)
  if(NOT _fftw_prefix_at EQUAL -1)
    message(FATAL_ERROR
      "FFTW @_package_name@ config still contains the install prefix @CMAKE_INSTALL_PREFIX@")
  endif()
]=])
string(CONFIGURE "${_fftw_install_config_code}" _fftw_install_config_code @ONLY)
# Copy our config over FFTW's, delete FFTW3LibraryDepends*.cmake, and fail
# the install if the build-machine prefix is still written in the config.
install(CODE "${_fftw_install_config_code}")

if(_threads_library_name)
  set(_fftw_install_threads_code [=[
    if(NOT EXISTS "@CMAKE_INSTALL_PREFIX@/lib/@CMAKE_SHARED_LIBRARY_PREFIX@@_threads_library_name@@CMAKE_SHARED_LIBRARY_SUFFIX@")
      message(FATAL_ERROR
        "FFTW @_package_name@ install is missing lib/@CMAKE_SHARED_LIBRARY_PREFIX@@_threads_library_name@@CMAKE_SHARED_LIBRARY_SUFFIX@")
    endif()
  ]=])
  string(CONFIGURE "${_fftw_install_threads_code}" _fftw_install_threads_code @ONLY)
  # Linux only. Fail the install if libfftw3_threads.so or libfftw3f_threads.so
  # was not produced. The CMake name for that library is created in the config.
  install(CODE "${_fftw_install_threads_code}")
endif()
