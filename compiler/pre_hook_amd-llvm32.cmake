# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for the 32-bit LLVM build. Produces only the static libraries
# that comgr32 needs (LLVM, Clang, LLD, device-libs). No Flang, no offload
# runtimes, no tests, no tools beyond the minimum.

include("${THEROCK_SOURCE_DIR}/compiler/amd-llvm/cmake/Modules/LLVMVersion.cmake")

set(BUILD_SHARED_LIBS OFF)
set(LLVM_BUILD_LLVM_DYLIB OFF)
set(LLVM_LINK_LLVM_DYLIB OFF)
set(LIBUNWIND_ENABLE_SHARED OFF)
set(LIBUNWIND_ENABLE_STATIC ON)
set(LLVM_ENABLE_LIBCXX OFF)
set(LLVM_ENABLE_RUNTIMES "" CACHE STRING "Enabled runtimes" FORCE)
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "Enable LLVM projects" FORCE)

set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF)
set(LLVM_TARGETS_TO_BUILD "AMDGPU;SPIRV" CACHE STRING "Enable LLVM Targets" FORCE)

set(PACKAGE_VENDOR "AMD" CACHE STRING "Vendor" FORCE)

# Device-libs and spirv-llvm-translator as external projects.
set(LLVM_EXTERNAL_ROCM_DEVICE_LIBS_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/amd-llvm/amd/device-libs")
set(LLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/spirv-llvm-translator")
set(LLVM_EXTERNAL_PROJECTS "rocm-device-libs;spirv-llvm-translator" CACHE STRING "Enable extra projects" FORCE)

# Helper function to disable LLVM/Clang tools not in the required list.
function(therock_set_implicit_llvm_options type tools_dir required_tool_names)
  file(GLOB subdirs "${tools_dir}/*")
  foreach(dir ${subdirs})
    if(NOT IS_DIRECTORY "${dir}" OR NOT EXISTS "${dir}/CMakeLists.txt")
      continue()
    endif()
    cmake_path(GET dir FILENAME toolname)
    string(REPLACE "-" "_" toolname "${toolname}")
    string(TOUPPER "${toolname}" toolname)
    set(_option_name "${type}_TOOL_${toolname}_BUILD")
    set(_option_value OFF)
    if("${toolname}" IN_LIST required_tool_names)
      set(_option_value ON)
    endif()
    message(STATUS "Implicit tool option: ${_option_name} = ${_option_value}")
    set(${_option_name} "${_option_value}" CACHE BOOL "Implicit disable ${type} tool" FORCE)
  endforeach()
endfunction()

# Disable all optional LLVM/Clang tools — only build the minimum required set.
block()
  set(_llvm_required_tools
    LLVM_AR
    LLVM_AS
    LLVM_CONFIG
    LLVM_COV
    LLVM_CXXFILT
    LLVM_DIS
    LLVM_DWARFDUMP
    LLVM_LINK
    LLVM_MC
    LLVM_NM
    LLVM_OFFLOAD_BINARY
    LLVM_PROFDATA
    LLVM_SHLIB
    LLVM_OBJCOPY
    LLVM_OBJDUMP
    LLVM_READOBJ
    LLVM_SYMBOLIZER
    OPT
    YAML2OBJ
  )
  if(WIN32)
    list(APPEND _llvm_required_tools "LLVM_DLLTOOL" "LLVM_LIB" "LLVM_RANLIB")
  endif()
  therock_set_implicit_llvm_options(LLVM "${CMAKE_CURRENT_SOURCE_DIR}/tools" "${_llvm_required_tools}")

  set(_clang_required_tools
    CLANG_OFFLOAD_BUNDLER CLANG_OFFLOAD_PACKAGER CLANG_SHLIB DRIVER LIBCLANG
  )
  if(WIN32)
    list(APPEND _clang_required_tools "CLANG_SCAN_DEPS")
  endif()
  therock_set_implicit_llvm_options(CLANG "${CMAKE_CURRENT_SOURCE_DIR}/../clang/tools" "${_clang_required_tools}")
endblock()
