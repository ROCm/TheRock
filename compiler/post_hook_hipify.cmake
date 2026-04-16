# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# hipify is declared NO_INSTALL_RPATH, and the submodule's CMakeLists.txt
# manually bakes hipify-clang's RPATH via LINK_FLAGS:
#
#   -Wl,--enable-new-dtags -Wl,--rpath,\$ORIGIN/../lib
#
# That leaves no spare .dynstr space for py_packaging's later
# `patchelf --set-rpath` pass. When patchelf grows .dynstr to fit the new
# string, it prepends a writable PT_LOAD segment at a non-standard base
# address and the resulting ELF crashes execve() on RHEL 8.10 / EL 4.18
# kernels (see https://github.com/ROCm/TheRock/issues/4271 and the long
# comment in cmake/therock_subproject_utils.cmake).
#
# Appending the TheRock pad to the linker's --rpath argument here reserves
# ~1KB of .dynstr up front so the subsequent patchelf overwrite fits without
# mutating program headers. We do this from the super-project post-hook so
# the fix ships without a submodule patch.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET hipify-clang)
  get_target_property(_therock_hipify_link_flags hipify-clang LINK_FLAGS)
  if(NOT _therock_hipify_link_flags)
    set(_therock_hipify_link_flags "")
  endif()
  # The pad value contains literal $ORIGIN; escape $ so Ninja/Make pass it
  # through to the linker unchanged, matching the submodule's existing
  # \\$ORIGIN/../lib escaping.
  string(REPLACE "$" "\\$" _therock_hipify_escaped_pad
    "${THEROCK_INSTALL_RPATH_PAD_COLON}")
  # Only patch once and only if the submodule still uses the unpadded
  # -Wl,--rpath,$ORIGIN/../lib form we audited (issue #4271). If hipify
  # upstream grows richer rpath handling in the future, the pad-marker CI
  # gate in rocm_sdk.tests.core_test will flag any regression.
  if(_therock_hipify_link_flags MATCHES "--rpath,\\\\\\$ORIGIN/\\.\\./lib"
     AND NOT _therock_hipify_link_flags MATCHES "${THEROCK_INSTALL_RPATH_PAD_MARKER}")
    set_target_properties(hipify-clang PROPERTIES
      LINK_FLAGS "${_therock_hipify_link_flags}:${_therock_hipify_escaped_pad}")
    message(STATUS "Appended TheRock RPATH pad to hipify-clang LINK_FLAGS")
  endif()
endif()
