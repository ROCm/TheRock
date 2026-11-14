# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_install_windows_dlls.cmake
# Copy a sub-project's installed DLLs to the stage's top-level bin/.
#
# Run as a script, after the sub-project has installed:
#
#   cmake -DPREFIX=<CMAKE_INSTALL_PREFIX>
#         -DSTAGE_DIR=<THEROCK_STAGE_INSTALL_ROOT>
#         -P cmake/therock_install_windows_dlls.cmake
#
# THEROCK_STAGE_INSTALL_ROOT is the stage root the super-project passes to
# every sub-project.  A sub-project cannot derive it from its own
# CMAKE_INSTALL_PREFIX: INSTALL_DESTINATION nests that prefix some number of
# directories below the stage root, and only the super-project knows how deep.
#
# Why this is needed:
#
# A sub-project that sets INSTALL_DESTINATION installs under a private prefix,
# lib/rocm_sysdeps for the bundled sysdeps, so its DLLs land in
# lib/rocm_sysdeps/bin.  Windows resolves a DLL from the directory of the
# executable that loads it, from the system directories, or from PATH, and a
# private directory inside the install tree is none of those.  No distribution
# channel puts it on PATH either: the MSI adds only InstallDir\bin (see
# docs/packaging/windows_msi_packaging.md), and someone unpacking the tarball
# points PATH at the same place by hand.  An executable installed in that
# top-level bin therefore cannot load the DLL, and fails to start with the
# loader reporting nothing more than an exit status.
#
# Why this copies the DLL instead of moving it:
#
#   - A sysdep may install its own tools beside the DLL, as liblzma does with
#     xz.exe and friends, and those load it from their own directory, exactly
#     as an executable in the top-level bin does.
#   - The package's exported CMake targets record an IMPORTED_LOCATION under
#     that prefix, so removing the file makes find_package() fail for every
#     consumer with "the imported target references the file ... but this file
#     does not exist".
#
# So on Windows a DLL has to exist in every directory that holds something which
# loads it.  That is the cost of having no $ORIGIN: on Linux these same sysdeps
# keep one copy in their private directory and reach it through an
# $ORIGIN-relative RPATH, which is what the Linux post-install patching sets up.
# Windows offers no such indirection, so the file is duplicated instead.
#
# Duplicating the file costs less than the alternative of static linking, which
# copies the same code into every binary that uses it.  There are far more
# binaries than there are directories holding one.

if(NOT PREFIX OR NOT STAGE_DIR)
  message(FATAL_ERROR "PREFIX and STAGE_DIR are both required")
endif()

file(GLOB _dlls "${PREFIX}/bin/*.dll")
if(NOT _dlls)
  # Not an error: a sysdep may build static-only on this platform.
  return()
endif()

foreach(_dll ${_dlls})
  get_filename_component(_name "${_dll}" NAME)
  file(COPY "${_dll}" DESTINATION "${STAGE_DIR}/bin")
  message(STATUS "Copied ${_name} to ${STAGE_DIR}/bin")
endforeach()
