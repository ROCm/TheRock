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
else()
  set(LLVM_BUILD_LLVM_DYLIB ON)
  set(LLVM_LINK_LLVM_DYLIB ON)
  set(LLVM_ENABLE_LIBCXX ON)
  set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi;openmp" CACHE STRING "Enabled runtimes" FORCE)
  # Settinng "LIBOMP_COPY_EXPORTS" to `OFF` "aids parallel builds to not interfere
  # with each other" as libomp and generated headers are copied into the original
  # source otherwise. Defaults to `ON`.
  set(LIBOMP_COPY_EXPORTS OFF)
endif()

# Set the LLVM_ENABLE_PROJECTS variable before including LLVM's CMakeLists.txt
set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
set(LLVM_ENABLE_PROJECTS "clang;lld;clang-tools-extra" CACHE STRING "Enable LLVM projects" FORCE)
set(LLVM_TARGETS_TO_BUILD "AMDGPU;X86" CACHE STRING "Enable LLVM Targets" FORCE)

# Packaging.
set(PACKAGE_VENDOR "AMD" CACHE STRING "Vendor" FORCE)

# Build the device-libs as part of the core compiler so that clang works by
# default (as opposed to other components that are *users* of the compiler).
set(LLVM_EXTERNAL_AMDDEVICE_LIBS_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/amd-llvm/amd/device-libs")
set(LLVM_EXTERNAL_PROJECTS "amddevice-libs" CACHE STRING "Enable extra projects" FORCE)

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
