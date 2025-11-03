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

  # Add paths to rocm_agent_enumerator (Linux) and hipInfo (Windows) too.
  # TODO: The build system shouldn't depend on device enumeration and the
  #   `--no-enumerate` flag to TensileCreateLibrary.py should be used in
  #   `TensileConfig.cmake`. If we do end up keeping this for whatever reason,
  #   these tools could be found with explicit paths instead of walks up
  #   from THEROCK_TOOLCHAIN_ROOT and back down into subproject dist folders.
  set(DEVICE_ENUMERATOR_PATH "")
  if(WIN32)
    set(DEVICE_ENUMERATOR_PATH "${THEROCK_TOOLCHAIN_ROOT}/../../hipInfo/dist/bin")
  else()
    set(DEVICE_ENUMERATOR_PATH "${THEROCK_TOOLCHAIN_ROOT}/../../rocminfo/dist/bin")
  endif()

  set(ENV{PATH} "${THEROCK_TOOLCHAIN_ROOT}/bin${PS}${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin${PS}${DEVICE_ENUMERATOR_PATH}${PS}${CURRENT_PATH}")
  message(STATUS "Augmented toolchain PATH=$ENV{PATH}")
endblock()

# Tensile is using msgpack and will pull in Boost otherwise.
add_compile_definitions(MSGPACK_NO_BOOST)

if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # rocBLAS has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64 REQUIRED)
  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
endif()
