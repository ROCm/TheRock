# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include("${THEROCK_SOURCE_DIR}/compiler/amd-llvm/cmake/Modules/LLVMVersion.cmake")

set(BUILD_SHARED_LIBS OFF)
if(WIN32)
  set(LLVM_BUILD_LLVM_DYLIB OFF)
  set(LLVM_LINK_LLVM_DYLIB OFF)
  set(LIBUNWIND_ENABLE_SHARED OFF)
  set(LIBUNWIND_ENABLE_STATIC ON)
  set(LLVM_ENABLE_LIBCXX OFF)
  set(LLVM_ENABLE_RUNTIMES "compiler-rt" CACHE STRING "Enabled runtimes" FORCE)
  set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra" CACHE STRING "Enable LLVM projects" FORCE)
else()
  set(LLVM_BUILD_LLVM_DYLIB ON)
  set(LLVM_LINK_LLVM_DYLIB ON)
  set(LLVM_ENABLE_LIBCXX ON)
  set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra" CACHE STRING "Enable LLVM projects" FORCE)
  set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi" CACHE STRING "Enabled runtimes" FORCE)
  set(LLVM_BUILTIN_TARGETS "default;amdgcn-amd-amdhsa" CACHE STRING "Enabled compiler-rt builtin targets" FORCE)
  set(BUILTINS_amdgcn-amd-amdhsa_CACHE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../compiler-rt/cmake/caches/GPU.cmake"
    CACHE STRING "AMDGPU compiler-rt builtin cache files" FORCE)
  set(CLANG_LINK_FLANG OFF)
endif()

if(THEROCK_BUILD_LLVM_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "Enable building LLVM tests" FORCE)
else()
  set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
endif()

if(THEROCK_BUILD_LLVM_TESTS OR THEROCK_BUILD_LLVM_TOOLS OR THEROCK_BUILD_COMGR_TESTS)
  set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools required for tests" FORCE)
  set(LLVM_INSTALL_UTILS ON CACHE BOOL "Install LLVM utility binaries like FileCheck" FORCE)

  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/bin/llvm-lit" DESTINATION bin)
  set(_lit_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../llvm/utils/lit")
  install(DIRECTORY "${_lit_source_dir}/lit"
    DESTINATION "lib/python"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
  )

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/llvm-lit-wrapper" [=[#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../lib/python:${PYTHONPATH}"
exec "${SCRIPT_DIR}/llvm-lit.real" "$@"
]=])
  install(CODE "
    file(RENAME \"\${CMAKE_INSTALL_PREFIX}/bin/llvm-lit\" \"\${CMAKE_INSTALL_PREFIX}/bin/llvm-lit.real\" )
    file(COPY \"${CMAKE_CURRENT_BINARY_DIR}/llvm-lit-wrapper\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/bin\")
    file(RENAME \"\${CMAKE_INSTALL_PREFIX}/bin/llvm-lit-wrapper\" \"\${CMAKE_INSTALL_PREFIX}/bin/llvm-lit\")
    file(CHMOD \"\${CMAKE_INSTALL_PREFIX}/bin/llvm-lit\" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  ")
endif()

set(LLVM_INCLUDE_BENCHMARKS OFF)
set(LLVM_TARGETS_TO_BUILD "AMDGPU;X86" CACHE STRING "Enable LLVM Targets" FORCE)
set(PACKAGE_VENDOR "AMD" CACHE STRING "Vendor" FORCE)

set(LLVM_EXTERNAL_ROCM_DEVICE_LIBS_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/amd-llvm/amd/device-libs")
set(LLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/spirv-llvm-translator")
set(LLVM_EXTERNAL_PROJECTS "rocm-device-libs;spirv-llvm-translator" CACHE STRING "Enable extra projects" FORCE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../../../lib;$ORIGIN/../../rocm_sysdeps/lib")
endif()

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

if(NOT THEROCK_BUILD_LLVM_TESTS AND NOT THEROCK_BUILD_LLVM_TOOLS AND NOT THEROCK_BUILD_COMGR_TESTS)
  block()
    set(_llvm_required_tools
      LLVM_AR
      LLVM_AS
      LLVM_CONFIG
      LLVM_DIS
      LLVM_DWARFDUMP
      LLVM_LINK
      LLVM_MC
      LLVM_NM
      LLVM_OFFLOAD_BINARY
      LLVM_SHLIB
      LLVM_OBJCOPY
      LLVM_OBJDUMP
      LLVM_READOBJ
      LLVM_SYMBOLIZER
      OPT
      YAML2OBJ
    )
    if(WIN32)
      list(APPEND _llvm_required_tools
        LLVM_DLLTOOL
        LLVM_LIB
        LLVM_RANLIB
      )
    endif()
    therock_set_implicit_llvm_options(LLVM "${CMAKE_CURRENT_SOURCE_DIR}/tools" "${_llvm_required_tools}")

    set(_clang_required_tools
      CLANG_HIP
      CLANG_OFFLOAD_BUNDLER
      CLANG_OFFLOAD_PACKAGER
      CLANG_OFFLOAD_WRAPPER
      CLANG_LINKER_WRAPPER
      CLANG_SHLIB
      DRIVER
      LIBCLANG
      OFFLOAD_ARCH
    )
    if(WIN32)
      list(APPEND _clang_required_tools CLANG_SCAN_DEPS)
    endif()
    therock_set_implicit_llvm_options(CLANG "${CMAKE_CURRENT_SOURCE_DIR}/../clang/tools" "${_clang_required_tools}")
  endblock()
endif()
