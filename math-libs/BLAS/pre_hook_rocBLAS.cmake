# Tensile just uses the system path to find most of its tools and it does this
# in the build phase. Rather than tunneling everything through manually, we
# just explicitly set up the path to include our toolchain ROCM and LLVM
# tools. This kind of reacharound is not great but the project is old, so
# c'est la vie.

block(SCOPE_FOR VARIABLES)
  if(NOT THEROCK_TOOLCHAIN_ROOT)
    message(FATAL_ERROR "As a sub-project, THEROCK_TOOLCHAIN_ROOT should have been defined and was not")
  endif()
  if(WIN32)
    set(PS ";")
  else()
    set(PS ":")
  endif()
  set(CURRENT_PATH "$ENV{PATH}")
  set(ENV{PATH} "${THEROCK_TOOLCHAIN_ROOT}/bin${PS}${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin${PS}${CURRENT_PATH}")
  message(STATUS "Augmented toolchain PATH=$ENV{PATH}")
endblock()

# rocBLAS has deprecated dependencies on roctracer. We apply a patch to move these
# to rocprofiler-sdk via the compat library until it can be caught up.
# See: https://github.com/ROCm/TheRock/issues/364
find_package(roctracer-compat REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_subproject_utils)

function(_rocblas_patch_roctracer)
  therock_get_all_targets(all_targets "${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "Patching rocblas targets: ${all_targets}")
  foreach(target ${all_targets})
    get_target_property(link_libs "${target}" LINK_LIBRARIES)
    if("-lroctx64" IN_LIST link_libs)
      list(REMOVE_ITEM link_libs "-lroctx64")
      list(APPEND link_libs "roctracer-compat::roctx")
      set_target_properties("${target}" PROPERTIES LINK_LIBRARIES "${link_libs}")
      message(WARNING "target ${target} depends on deprecated -lroctx64. redirecting: ${link_libs}")
    endif()
  endforeach()
endfunction()

cmake_language(DEFER CALL _rocblas_patch_roctracer)
