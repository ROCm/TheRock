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
