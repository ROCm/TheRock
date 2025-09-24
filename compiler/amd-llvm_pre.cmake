# Get access to LLVM_VERSION_MAJOR
include("${THEROCK_SOURCE_DIR}/compiler/amd-llvm/cmake/Modules/LLVMVersion.cmake")

# Build LLVM and the comgr dependency.
# Note that in LLVM "BUILD_SHARED_LIBS" enables an unsupported development mode.
# The flag you want for a shared library build is LLVM_BUILD_LLVM_DYLIB.
set(BUILD_SHARED_LIBS OFF)
if(WIN32)
  set(LLVM_BUILD_LLVM_DYLIB OFF)
  set(LLVM_LINK_LLVM_DYLIB OFF)
  set(LIBUNWIND_ENABLE_SHARED OFF)
  set(LIBUNWIND_ENABLE_STATIC ON)
  # TODO(#36): Enable libunwind, libcxx, and libcxxabi on Windows?
  #     Should they be supported? What depends on them?
  set(LLVM_ENABLE_LIBCXX OFF)
  set(LLVM_ENABLE_RUNTIMES "compiler-rt" CACHE STRING "Enabled runtimes" FORCE)
  set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra" CACHE STRING "Enable LLVM projects" FORCE)
else()
  set(LLVM_BUILD_LLVM_DYLIB ON)
  set(LLVM_LINK_LLVM_DYLIB ON)
  set(LLVM_ENABLE_LIBCXX ON)
  set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra;flang" CACHE STRING "Enable LLVM projects" FORCE)
  set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi;openmp;offload" CACHE STRING "Enabled runtimes" FORCE)
  if("offload" IN_LIST LLVM_ENABLE_RUNTIMES)
    set(OPENMP_ENABLE_LIBOMPTARGET ON)
    set(LIBOMPTARGET_BUILD_DEVICE_FORTRT ON)
    set(LIBOMPTARGET_ENABLE_DEBUG ON)
    set(LIBOMPTARGET_NO_SANITIZER_AMDGPU ON)
    set(LIBOMP_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../../lib:\$ORIGIN/../../../lib")
    set(LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH "${THEROCK_SOURCE_DIR}/rocm-systems/projects/rocr-runtime")
    set(OFFLOAD_EXTERNAL_PROJECT_UNIFIED_ROCR ON)
    # There is an issue with finding the zstd config built by TheRock when zstd
    # is searched for in the llvm config. LLVM has a FindZSTD.cmake that is
    # found in module mode, which ultimately fails to locate the library.
    # For now we will switch the priorty for find_package to first search in
    # CONFIG mode.
    set(RUNTIMES_CMAKE_ARGS "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")
  endif()
  # Setting "LIBOMP_COPY_EXPORTS" to `OFF` "aids parallel builds to not interfere
  # with each other" as libomp and generated headers are copied into the original
  # source otherwise. Defaults to `ON`.
  set(LIBOMP_COPY_EXPORTS OFF)
endif()

# Set the LLVM_ENABLE_PROJECTS variable before including LLVM's CMakeLists.txt
set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
set(LLVM_TARGETS_TO_BUILD "AMDGPU;X86" CACHE STRING "Enable LLVM Targets" FORCE)

# Packaging.
set(PACKAGE_VENDOR "AMD" CACHE STRING "Vendor" FORCE)

# Build the device-libs as part of the core compiler so that clang works by
# default (as opposed to other components that are *users* of the compiler).
set(LLVM_EXTERNAL_ROCM_DEVICE_LIBS_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/amd-llvm/amd/device-libs")
set(LLVM_EXTERNAL_PROJECTS "rocm-device-libs" CACHE STRING "Enable extra projects" FORCE)

# TODO2: This mechanism has races in certain situations, failing to create a
# symlink. Revisit once devicemanager code is made more robust.
# TODO: Arrange for the devicelibs to be installed to the clange resource dir
# by default. This corresponds to the layout for ROCM>=7. However, not all
# code (specifically the AMDDeviceLibs.cmake file) has adapted to the new
# location, so we have to also make them available at amdgcn. There are cache
# options to manage this transition but they require knowing the clange resource
# dir. In order to avoid drift, we just fixate that too. This can all be
# removed in a future version.
# set(CLANG_RESOURCE_DIR "../lib/clang/${LLVM_VERSION_MAJOR}" CACHE STRING "Resource dir" FORCE)
# set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW "lib/clang/${LLVM_VERSION_MAJOR}/amdgcn" CACHE STRING "New devicelibs loc" FORCE)
# set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD "amdgcn" CACHE STRING "Old devicelibs loc" FORCE)

# Setup the install rpath (let CMake handle build RPATH per usual):
# * Executables and libraries can always search their adjacent lib directory
#   (which may be the same as the origin for libraries).
# * Files in lib/llvm/(bin|lib) should search the project-wide lib/ directory
#   so that dlopen of runtime files from the compiler can work.
# * One might think that only EXEs need to be build this way, but the dlopen
#   utilities can be compiled into libLLVM, in which case, that RUNPATH is
#   primary.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../../../lib;$ORIGIN/../../rocm_sysdeps/lib")
endif()

# Disable all implicit LLVM tools by default so that we can allow-list just what
# we want. It is unfortunate that LLVM doesn't have a global option to do this
# bulk disabling. In the absence of that, we manually generate options using
# the same logic as `create_llvm_tool_options` in `AddLLVM.cmake`. If this
# ever drifts, we will build extra tools and (presumably) someone will notice
# the bloat.

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

block()
  # This list contains the minimum tooling that must be enabled to build LLVM.
  # It is empically derived (either configure or ninja invocation will fail
  # on a missing tool).
  set(_llvm_required_tools
    LLVM_CONFIG
    LLVM_DWARFDUMP
    LLVM_JITLINK
    LLVM_LINK
    LLVM_MC
    LLVM_NM
    LLVM_SHLIB
    LLVM_OBJCOPY
    LLVM_OBJDUMP
    OPT
    YAML2OBJ
  )
  if(WIN32)
    # These can be provided by the "C++ Clang tools for Windows" in MSVC, but
    # we might as well build them from source ourselves.
    list(APPEND _llvm_required_tools "LLVM_AR")
    list(APPEND _llvm_required_tools "LLVM_DLLTOOL")
    list(APPEND _llvm_required_tools "LLVM_LIB")
    list(APPEND _llvm_required_tools "LLVM_RANLIB")
  endif()
  therock_set_implicit_llvm_options(LLVM "${CMAKE_CURRENT_SOURCE_DIR}/tools" "${_llvm_required_tools}")

  # Clang tools that are required.
  set(_clang_required_tools
    AMDGPU_ARCH
    CLANG_HIP
    CLANG_OFFLOAD_BUNDLER
    CLANG_OFFLOAD_PACKAGER
    CLANG_OFFLOAD_WRAPPER
    CLANG_SHLIB
    DRIVER
  )
  if(WIN32)
    # These can be provided by the "C++ Clang tools for Windows" in MSVC, but
    # we might as well build them from source ourselves.
    list(APPEND _clang_required_tools "CLANG_SCAN_DEPS")
  endif()
  therock_set_implicit_llvm_options(CLANG "${CMAKE_CURRENT_SOURCE_DIR}/../clang/tools" "${_clang_required_tools}")
endblock()
