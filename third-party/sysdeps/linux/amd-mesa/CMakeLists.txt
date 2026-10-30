# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # setup submodule project
    set(_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../common/mesa-fork")

    therock_cmake_subproject_declare(therock-amd-mesa
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DMESON_BUILD=${MESON_BUILD}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      RUNTIME_DEPS
        therock-libdrm
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_INCLUDE_DIRS
        lib/rocm_sysdeps/include
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
    )

    therock_cmake_subproject_activate(therock-amd-mesa)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libgallium_drv_video.so radeonsi_drv_video.so libva.so libva-drm.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(MESA_BUILD)
if(NOT PATCHELF OR NOT MESON_BUILD)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
therock_get_meson_compiler_env(_meson_cc _meson_cxx)

set(_meson_ldflags "-Wl,-rpath='$$ORIGIN'")
set(_meson_sanitizer_env)
set(_meson_runtime_env)
if(THEROCK_SANITIZER STREQUAL "DASAN")
  # Meson does not consume CMake language flags or add_link_options().
  list(APPEND _meson_sanitizer_env
    "CFLAGS=${CMAKE_C_FLAGS}"
    "CXXFLAGS=${CMAKE_CXX_FLAGS}"
  )
  string(APPEND _meson_ldflags " ${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=daddress -shared-libsan")
  # Meson's compiler sanity checks and generated build tools execute before
  # installation, where the SDK-relative RPATH below cannot locate the runtime.
  # Supply the selected compiler's runtime only to the build environment; keep
  # installed libraries relocatable instead of embedding an absolute build path.
  execute_process(
    COMMAND "${CMAKE_C_COMPILER}" -print-file-name=libclang_rt.dasan.so
    OUTPUT_VARIABLE _dasan_runtime
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _dasan_runtime_result
  )
  if(NOT _dasan_runtime_result EQUAL 0 OR
     NOT IS_ABSOLUTE "${_dasan_runtime}" OR NOT EXISTS "${_dasan_runtime}")
    message(FATAL_ERROR "DASan Mesa build requires the selected compiler's libclang_rt.dasan.so")
  endif()
  get_filename_component(_dasan_runtime_dir "${_dasan_runtime}" DIRECTORY)
  set(_meson_ld_library_path "${_dasan_runtime_dir}")
  if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    string(APPEND _meson_ld_library_path ":$ENV{LD_LIBRARY_PATH}")
  endif()
  list(APPEND _meson_runtime_env "LD_LIBRARY_PATH=${_meson_ld_library_path}")
  # Mesa's libraries install below lib/rocm_sysdeps/lib in the unified SDK.
  # Keep the shared sanitizer discoverable after relocating that SDK.
  foreach(_runtime_dir IN LISTS THEROCK_PRIVATE_INSTALL_RPATH_DIRS)
    if(_runtime_dir MATCHES "^lib/llvm/lib/clang/")
      file(RELATIVE_PATH _runtime_relative
        "/prefix/lib/rocm_sysdeps/lib" "/prefix/${_runtime_dir}")
      string(APPEND _meson_ldflags " -Wl,-rpath='$$ORIGIN/${_runtime_relative}'")
    endif()
  endforeach()
endif()

set(PATCH_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../patch_source")

add_custom_target(
  meson_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # We have to patch the sources so make a fresh copy. Use cp -a rather than
    # cmake -E copy_directory so symlinks are preserved instead of dereferenced:
    # mesa ships a dangling symlink that would otherwise abort the copy.
    cp -a "${SOURCE_DIR}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 1: Initial meson setup to download libva-2.22.0 subproject
    # This is required before patching because patch_source.sh modifies files in the libva subproject
    "${CMAKE_COMMAND}" -E env
      "LDFLAGS=${_meson_ldflags}"
      ${_meson_sanitizer_env}
      ${_meson_runtime_env}
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      --
      "${MESON_BUILD}" setup
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dva-libs-path="/lib"
        -Dallow-fallback-for=libva # NOTE: Use local build of VA library
        -Dforce_fallback_for=libva # NOTE: Enforce local build, if public VA found
        -Dpkg_config_path="${CMAKE_CURRENT_BINARY_DIR}"/../../../libdrm/build/dist/lib/rocm_sysdeps/lib/pkgconfig
        -Degl=disabled
        -Dgallium-drivers=radeonsi
        -Dgallium-rusticl=false
        -Dgallium-va=enabled
        -Dgbm=disabled
        -Dgles1=disabled
        -Dgles2=disabled
        -Dglvnd=disabled
        -Dglx=disabled
        -Dllvm=disabled
        -Dopengl=false
        -Dvideo-codecs=h264dec,h265dec,av1dec,vp9dec,jpegdec
        -Dvulkan-drivers=
        -Dplatforms=
        -Dlibva:with_glx=no
        -Dlibva:with_wayland=no
        -Dlibva:with_x11=no
        -Dvalgrind=disabled
        -Dspirv-tools=disabled
        -Dzlib=disabled
        -Dzstd=disabled
        -Dshader-cache=disabled
        -Dexpat=disabled
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 2: Patch source files including libva driver path configuration
    # Modifies library names and driver search paths for ROCm sysdeps integration
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 3: Reconfigure meson to build from patched source
    # This ensures the build system uses the modified source files
    "${CMAKE_COMMAND}" -E env
      "LDFLAGS=${_meson_ldflags}"
      ${_meson_sanitizer_env}
      ${_meson_runtime_env}
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      --
      "${MESON_BUILD}" setup
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dva-libs-path="/lib"
        -Dallow-fallback-for=libva # NOTE: Use local build of VA library
        -Dforce_fallback_for=libva # NOTE: Enforce local build, if public VA found
        -Dpkg_config_path="${CMAKE_CURRENT_BINARY_DIR}"/../../../libdrm/build/dist/lib/rocm_sysdeps/lib/pkgconfig
        -Degl=disabled
        -Dgallium-drivers=radeonsi
        -Dgallium-rusticl=false
        -Dgallium-va=enabled
        -Dgbm=disabled
        -Dgles1=disabled
        -Dgles2=disabled
        -Dglvnd=disabled
        -Dglx=disabled
        -Dllvm=disabled
        -Dopengl=false
        -Dvideo-codecs=h264dec,h265dec,av1dec,vp9dec,jpegdec
        -Dvulkan-drivers=
        -Dplatforms=
        -Dlibva:with_glx=no
        -Dlibva:with_wayland=no
        -Dlibva:with_x11=no
        -Dvalgrind=disabled
        -Dspirv-tools=disabled
        -Dzlib=disabled
        -Dzstd=disabled
        -Dshader-cache=disabled
        -Dexpat=disabled
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PATCH_SOURCE_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E env ${_meson_runtime_env} --
      "${MESON_BUILD}" compile --verbose
  COMMAND
    "${CMAKE_COMMAND}" -E env "DESTDIR=${CMAKE_INSTALL_PREFIX}" ${_meson_runtime_env} --
      "${MESON_BUILD}" install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
)
