# Script run to install the amdgpu_runtime_dev component.
# This runs at install time and has access to the staging tree files.
message(STATUS "Custom installing amdgpu-runtime-dev to: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Staging install dir: ${STAGING_INSTALL_DIR}")
if(NOT IS_DIRECTORY "${STAGING_INSTALL_DIR}")
  message(FATAL_ERROR "Did not get STAGING_INSTALL_DIR from super-project")
endif()

set(LIB_PREFIX "lib")
set(SO_SUFFIX ".so")

# Assemble file lists.
# Note that this presumes we did not link against libLLVM.so. If we do (we
# should for this slice of the compiler), then we also need to include it
# from the lib directory.
file(
  GLOB_RECURSE LLVM_FILES
  LIST_DIRECTORIES FALSE
  # hipcc expects clang++ to be under llvm/bin, not just bin.
  # but hipcc is in bin.
  RELATIVE ${STAGING_INSTALL_DIR}
  ${STAGING_INSTALL_DIR}/bin/hipcc*
  ${STAGING_INSTALL_DIR}/bin/hipconfig*
  ${STAGING_INSTALL_DIR}/bin/rocm_agent_enumerator
  ${STAGING_INSTALL_DIR}/bin/rocminfo
  ${STAGING_INSTALL_DIR}/llvm/amdgcn/*
  ${STAGING_INSTALL_DIR}/llvm/bin/*lld*
  ${STAGING_INSTALL_DIR}/llvm/bin/amdgpu*
  ${STAGING_INSTALL_DIR}/llvm/bin/clang*
  ${STAGING_INSTALL_DIR}/llvm/bin/offload-arch*
)

foreach(_relpath ${LLVM_FILES} ${HIPCC_FILES})
  cmake_path(GET _relpath PARENT_PATH _parent_rel_path)
  file(
    INSTALL ${STAGING_INSTALL_DIR}/${_relpath}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${_parent_rel_path}
    USE_SOURCE_PERMISSIONS
  )
endforeach()
