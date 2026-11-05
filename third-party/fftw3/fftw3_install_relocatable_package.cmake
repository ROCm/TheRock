# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install third-party/fftw3/FFTW3Config.cmake.in into the subproject prefix.
# Included from the therock-fftw3 and therock-fftw3f post hooks, so it runs
# for the upstream CMake build and for the optional Linux autotools build.
#
# Upstream configure_file() writes CMAKE_INSTALL_FULL_INCLUDEDIR into
# FFTW3Config.cmake. That install() is registered first. install(FILES) then
# sees the upstream file as newer and leaves the absolute path in place.
# file(COPY) always replaces it. FFTW3LibraryDepends.cmake is removed so
# nothing loads the export this config no longer includes.
#
# FFTW3::fftw3f searches only the float prefix's include directory. Copy
# api/fftw3.h from the fetched source into that directory. The double prefix
# is not consulted.

if(ENABLE_FLOAT)
  set(_package_name FFTW3f)
  set(_library_name fftw3f)
else()
  set(_package_name FFTW3)
  set(_library_name fftw3)
endif()

# The float build does not know the double prefix. The header has to be in
# this install's include directory, taken from the shared FFTW source tree.
set(_fftw_header "")
if(ENABLE_FLOAT)
  if(DEFINED SOURCE_DIR AND EXISTS "${SOURCE_DIR}/api/fftw3.h")
    set(_fftw_header "${SOURCE_DIR}/api/fftw3.h")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/api/fftw3.h")
    set(_fftw_header "${CMAKE_CURRENT_SOURCE_DIR}/api/fftw3.h")
  endif()
  if(NOT _fftw_header)
    message(FATAL_ERROR
      "FFTW api/fftw3.h was not found; the FFTW3f install cannot ship a header")
  endif()
endif()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/FFTW3Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/therock-fftw-package/${_package_name}Config.cmake"
  @ONLY
)

install(CODE "
  set(_fftw_cmake_dir \"${CMAKE_INSTALL_PREFIX}/lib/cmake/${_library_name}\")
  file(MAKE_DIRECTORY \"\${_fftw_cmake_dir}\")
  execute_process(
    COMMAND \"${CMAKE_COMMAND}\" -E copy
      \"${CMAKE_CURRENT_BINARY_DIR}/therock-fftw-package/${_package_name}Config.cmake\"
      \"\${_fftw_cmake_dir}/${_package_name}Config.cmake\"
    COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB _fftw_absolute_exports \"\${_fftw_cmake_dir}/FFTW3LibraryDepends*.cmake\")
  if(_fftw_absolute_exports)
    file(REMOVE \${_fftw_absolute_exports})
  endif()
  set(_fftw_config \"\${_fftw_cmake_dir}/${_package_name}Config.cmake\")
  if(NOT EXISTS \"\${_fftw_config}\")
    message(FATAL_ERROR \"FFTW ${_package_name} relocatable config was not installed\")
  endif()
  file(READ \"\${_fftw_config}\" _fftw_config_text)
  string(FIND \"\${_fftw_config_text}\" \"${CMAKE_INSTALL_PREFIX}\" _fftw_prefix_at)
  if(NOT _fftw_prefix_at EQUAL -1)
    message(FATAL_ERROR
      \"FFTW ${_package_name} config still contains the install prefix ${CMAKE_INSTALL_PREFIX}\")
  endif()
")

if(_fftw_header)
  install(CODE "
    file(MAKE_DIRECTORY \"${CMAKE_INSTALL_PREFIX}/include\")
    execute_process(
      COMMAND \"${CMAKE_COMMAND}\" -E copy
        \"${_fftw_header}\"
        \"${CMAKE_INSTALL_PREFIX}/include/fftw3.h\"
      COMMAND_ERROR_IS_FATAL ANY)
    if(NOT EXISTS \"${CMAKE_INSTALL_PREFIX}/include/fftw3.h\")
      message(FATAL_ERROR
        \"FFTW ${_package_name} install is missing include/fftw3.h under ${CMAKE_INSTALL_PREFIX}\")
    endif()
  ")
endif()
