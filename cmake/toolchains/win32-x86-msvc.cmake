# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Toolchain file for cross-compiling 32-bit (x86) binaries using MSVC on a
# 64-bit Windows host. Discovers the x86 cl.exe from the existing x64
# installation via VCToolsInstallDir or by inspecting CMAKE_CXX_COMPILER.
#
# Usage from TheRock: subprojects marked CROSS_COMPILE_32BIT automatically
# include this file in their generated toolchain.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86)
set(CMAKE_SIZEOF_VOID_P 4)

# Skip compiler verification to avoid manifest issues in try_compile
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ---------------------------------------------------------------------------
# Locate VCToolsInstallDir
# ---------------------------------------------------------------------------
if(NOT DEFINED _THEROCK_VCTOOLS_INSTALL_DIR)
  if(DEFINED ENV{VCToolsInstallDir})
    file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _THEROCK_VCTOOLS_INSTALL_DIR)
  elseif(DEFINED THEROCK_HOST_CMAKE_CXX_COMPILER)
    # Derive from the host compiler path:
    #   .../VC/Tools/MSVC/<ver>/bin/Hostx64/x64/cl.exe
    #   We need .../VC/Tools/MSVC/<ver>/
    get_filename_component(_cl_bin_dir "${THEROCK_HOST_CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_host_arch_dir "${_cl_bin_dir}" DIRECTORY)
    get_filename_component(_bin_dir "${_host_arch_dir}" DIRECTORY)
    get_filename_component(_THEROCK_VCTOOLS_INSTALL_DIR "${_bin_dir}" DIRECTORY)
  endif()
endif()

if(NOT _THEROCK_VCTOOLS_INSTALL_DIR)
  message(FATAL_ERROR
    "Cannot locate MSVC tools for 32-bit cross-compilation.\n"
    "Set VCToolsInstallDir or ensure CMAKE_CXX_COMPILER points to "
    "the MSVC cl.exe under the standard VC/Tools/MSVC/<ver>/ layout.")
endif()

# ---------------------------------------------------------------------------
# Resolve x86 compiler, linker, and tools
# ---------------------------------------------------------------------------
# Prefer Hostx64/x86 (cross-compile from x64 host to x86 target).
# Fall back to Hostx86/x86 if the cross tools aren't installed.
set(_x86_cross_bin "${_THEROCK_VCTOOLS_INSTALL_DIR}/bin/Hostx64/x86")
set(_x86_native_bin "${_THEROCK_VCTOOLS_INSTALL_DIR}/bin/Hostx86/x86")

if(EXISTS "${_x86_cross_bin}/cl.exe")
  set(_x86_bin "${_x86_cross_bin}")
elseif(EXISTS "${_x86_native_bin}/cl.exe")
  set(_x86_bin "${_x86_native_bin}")
else()
  message(FATAL_ERROR
    "Could not find x86 cl.exe in:\n"
    "  ${_x86_cross_bin}\n"
    "  ${_x86_native_bin}\n"
    "Ensure the MSVC x86 build tools are installed.")
endif()

set(CMAKE_C_COMPILER "${_x86_bin}/cl.exe")
set(CMAKE_CXX_COMPILER "${_x86_bin}/cl.exe")
set(CMAKE_LINKER "${_x86_bin}/link.exe")
set(CMAKE_AR "${_x86_bin}/lib.exe")
set(CMAKE_MT "${_x86_bin}/mt.exe")

# Resource compiler is architecture-neutral; prefer the host-native version already on PATH.
find_program(_therock_rc_compiler rc)
if(_therock_rc_compiler)
  set(CMAKE_RC_COMPILER "${_therock_rc_compiler}")
endif()

# ---------------------------------------------------------------------------
# x86 library search paths
# ---------------------------------------------------------------------------
# MSVC libraries
set(_x86_lib_dir "${_THEROCK_VCTOOLS_INSTALL_DIR}/lib/x86")
if(EXISTS "${_x86_lib_dir}")
  list(APPEND CMAKE_LIBRARY_PATH "${_x86_lib_dir}")
  link_directories("${_x86_lib_dir}")
endif()

# MSVC include directories — use CMAKE_*_STANDARD_INCLUDE_DIRECTORIES only,
# not include_directories(), to avoid leaking host paths into device-libs
# OpenCL compilation (which reads INCLUDE_DIRECTORIES from directory scope).
set(_x86_include_dir "${_THEROCK_VCTOOLS_INSTALL_DIR}/include")
if(EXISTS "${_x86_include_dir}")
  list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_x86_include_dir}")
  list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_x86_include_dir}")
endif()

# ATL/MFC libraries (needed by DebugInfoPDB)
get_filename_component(_vc_install_dir "${_THEROCK_VCTOOLS_INSTALL_DIR}" DIRECTORY)
get_filename_component(_vc_install_dir "${_vc_install_dir}" DIRECTORY)
get_filename_component(_vc_install_dir "${_vc_install_dir}" DIRECTORY)
set(_x86_atlmfc_lib "${_vc_install_dir}/ATLMFC/lib/x86")
if(EXISTS "${_x86_atlmfc_lib}")
  list(APPEND CMAKE_LIBRARY_PATH "${_x86_atlmfc_lib}")
  link_directories("${_x86_atlmfc_lib}")
endif()
set(_atlmfc_include "${_vc_install_dir}/ATLMFC/include")
if(EXISTS "${_atlmfc_include}")
  list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_atlmfc_include}")
  list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_atlmfc_include}")
endif()

# Windows SDK libraries and includes (ucrt, um, shared)
if(DEFINED ENV{WindowsSdkDir} AND DEFINED ENV{WindowsSDKVersion})
  file(TO_CMAKE_PATH "$ENV{WindowsSdkDir}" _winsdk_dir)
  set(_winsdk_ver "$ENV{WindowsSDKVersion}")
  # Remove trailing backslash from version
  string(REGEX REPLACE "[/\\\\]$" "" _winsdk_ver "${_winsdk_ver}")
else()
  # Fallback: search for Windows Kits
  file(TO_CMAKE_PATH "C:/Program Files (x86)/Windows Kits/10" _winsdk_dir)
  if(EXISTS "${_winsdk_dir}/Lib")
    file(GLOB _sdk_versions LIST_DIRECTORIES true "${_winsdk_dir}/Lib/10.0.*")
    list(SORT _sdk_versions)
    list(REVERSE _sdk_versions)
    list(GET _sdk_versions 0 _winsdk_lib_dir)
    get_filename_component(_winsdk_ver "${_winsdk_lib_dir}" NAME)
  endif()
endif()

if(_winsdk_dir AND _winsdk_ver)

  # Libraries
  set(_ucrt_x86 "${_winsdk_dir}/Lib/${_winsdk_ver}/ucrt/x86")
  set(_um_x86 "${_winsdk_dir}/Lib/${_winsdk_ver}/um/x86")
  if(EXISTS "${_ucrt_x86}")
    list(APPEND CMAKE_LIBRARY_PATH "${_ucrt_x86}")
    link_directories("${_ucrt_x86}")
  endif()
  if(EXISTS "${_um_x86}")
    list(APPEND CMAKE_LIBRARY_PATH "${_um_x86}")
    link_directories("${_um_x86}")
  endif()

  # Include directories
  set(_ucrt_include "${_winsdk_dir}/Include/${_winsdk_ver}/ucrt")
  set(_um_include "${_winsdk_dir}/Include/${_winsdk_ver}/um")
  set(_shared_include "${_winsdk_dir}/Include/${_winsdk_ver}/shared")
  set(_winrt_include "${_winsdk_dir}/Include/${_winsdk_ver}/winrt")
  set(_cppwinrt_include "${_winsdk_dir}/Include/${_winsdk_ver}/cppwinrt")

  if(EXISTS "${_ucrt_include}")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_ucrt_include}")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_ucrt_include}")
  endif()
  if(EXISTS "${_um_include}")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_um_include}")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_um_include}")
  endif()
  if(EXISTS "${_shared_include}")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_shared_include}")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_shared_include}")
  endif()
  if(EXISTS "${_winrt_include}")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_winrt_include}")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_winrt_include}")
  endif()
  if(EXISTS "${_cppwinrt_include}")
    list(APPEND CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_cppwinrt_include}")
    list(APPEND CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_cppwinrt_include}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Force linker to use x86 library paths and machine type.
# Paths are quoted individually to handle spaces (e.g. "Program Files (x86)").
# We rely on CMake's built-in MSVC link rules rather than overriding
# CMAKE_*_LINK_EXECUTABLE / CMAKE_*_CREATE_SHARED_LIBRARY, since the
# built-in rules handle response files and quoting correctly.
# ---------------------------------------------------------------------------
set(_link_flags "/machine:X86")
if(DEFINED _x86_lib_dir AND EXISTS "${_x86_lib_dir}")
  string(APPEND _link_flags " \"/LIBPATH:${_x86_lib_dir}\"")
endif()
if(DEFINED _x86_atlmfc_lib AND EXISTS "${_x86_atlmfc_lib}")
  string(APPEND _link_flags " \"/LIBPATH:${_x86_atlmfc_lib}\"")
endif()
if(DEFINED _ucrt_x86 AND EXISTS "${_ucrt_x86}")
  string(APPEND _link_flags " \"/LIBPATH:${_ucrt_x86}\"")
endif()
if(DEFINED _um_x86 AND EXISTS "${_um_x86}")
  string(APPEND _link_flags " \"/LIBPATH:${_um_x86}\"")
endif()

set(CMAKE_EXE_LINKER_FLAGS_INIT "${_link_flags}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_link_flags}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_link_flags}")
set(CMAKE_STATIC_LINKER_FLAGS_INIT "")
