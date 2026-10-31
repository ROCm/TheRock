# TODO: Arrange for the devicelibs to be installed to the clange resource dir
# by default. This corresponds to the layout for ROCM>=7. However, not all
# code (specifically the AMDDeviceLibs.cmake file) has adapted to the new
# location, so we have to also make them available at amdgcn. There are cache
# options to manage this transition but they require knowing the clange resource
# dir. In order to avoid drift, we just fixate that too. This can all be
# removed in a future version.
set(CLANG_RESOURCE_DIR "../lib/clang/${LLVM_VERSION_MAJOR}" CACHE STRING "Resource dir" FORCE)
set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW "lib/clang/${LLVM_VERSION_MAJOR}/amdgcn" CACHE STRING "New devicelibs loc" FORCE)
set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD "amdgcn" CACHE STRING "Old devicelibs loc" FORCE)