# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_ROCPROFV3)
  set(_openmpi_optional_dep)
  if(THEROCK_ENABLE_MPI)
    set(_openmpi_optional_dep therock-openmpi)
  endif()

  ##############################################################################
  # rocprof-trace-decoder
  # The trace decoder is responsible for decoding advanced thread traces from the
  # GPU and is used at runtime by `rocprofv3 --att`.
  # See: https://github.com/ROCm/rocprof-trace-decoder/
  ##############################################################################
  therock_cmake_subproject_declare(rocprof-trace-decoder
    # The build infrastructure requires generic stages to pass through
    # DIST_AMDGPU_TARGETS even if the list isn't used by the underlying project.
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprof-trace-decoder"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprof-trace-decoder"
    BACKGROUND_BUILD
  )
  therock_cmake_subproject_activate(rocprof-trace-decoder)

  ##############################################################################
  # aqlprofile
  ##############################################################################
  therock_cmake_subproject_declare(aqlprofile
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/aqlprofile"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aqlprofile"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DAQLPROFILE_BUILD_TESTS=${THEROCK_BUILD_TESTING}
      -DHIP_PLATFORM="amd"
    COMPILER_TOOLCHAIN
      amd-hip
    INTERFACE_LINK_DIRS
      lib  # So that dependents can find the aqlprofile library via find_library
    BUILD_DEPS
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      amd-llvm
      ROCR-Runtime
  )
  therock_cmake_subproject_glob_c_sources(aqlprofile
    SUBDIRS .
  )
  therock_cmake_subproject_activate(aqlprofile)

  therock_provide_artifact(aqlprofile
    TARGET_NEUTRAL
    DESCRIPTOR artifact-aqlprofile.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      aqlprofile
  )

  ##############################################################################
  # rocprofiler-sdk
  ##############################################################################
  # Detect available Default Python3 version on the system or
  # use explicitly provided Python3 versions
  therock_detect_python_versions(_detected_python_executables _detected_python_versions)

  # Use detected versions or fall back to default if none found
  if(NOT _detected_python_versions)
    message(FATAL_ERROR "No Python versions detected for rocprofiler-sdk")
    set(_detected_python_versions_str "")
  else()
    message(STATUS "Detected Python versions for rocprofiler-sdk python bindings: ${_detected_python_versions}")
    message(STATUS "Python executables: ${_detected_python_executables}")
    # Join all detected Python versions into a single string separated by semicolons.
    # This is a workaround because therock_cmake_subproject_declare does not support passing lists directly.
    list(JOIN _detected_python_versions "\;" _detected_python_versions_str)
  endif()

  therock_cmake_subproject_declare(rocprofiler-sdk
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-sdk"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-sdk"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCPROFILER_BUILD_CI=OFF
      -DROCPROFILER_BUILD_DEVELOPER=OFF
      -DROCPROFILER_BUILD_TESTS=${THEROCK_BUILD_TESTING}
      -DROCPROFILER_BUILD_INTEGRATION_TESTS=OFF
      -DROCPROFILER_BUILD_SAMPLES=OFF
      -DCMAKE_PREFIX_PATH="${THEROCK_BINARY_DIR}/dist/rocm"
      -DROCM_PATH="${THEROCK_BINARY_DIR}/dist/rocm"
      -DROCPROFILER_PYTHON_VERSIONS:STRING="${_detected_python_versions_str}"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    IGNORE_PACKAGES
      rocprofiler-sdk
      rocprofiler-sdk-roctx
      rocprofiler-sdk-rocpd
      rocprofiler-sdk-tests
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      aqlprofile
      hip-clr
      rocprof-trace-decoder
      rocprofiler-register
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_SQLITE3}
    INTERFACE_INCLUDE_DIRS
      include
    INTERFACE_LINK_DIRS
      lib
      lib/rocm_sysdeps/lib
    INTERFACE_INSTALL_RPATH_DIRS
      lib
      lib/rocm_sysdeps/lib
  )
  therock_cmake_subproject_glob_c_sources(rocprofiler-sdk
    SUBDIRS .
  )
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk lib/cmake/rocprofiler-sdk)
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk-roctx lib/cmake/rocprofiler-sdk-roctx)
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk-rocpd lib/cmake/rocprofiler-sdk-rocpd)
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk-rocattach lib/cmake/rocprofiler-sdk-rocattach)
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk-tests lib/cmake/rocprofiler-sdk-tests)
  therock_cmake_subproject_activate(rocprofiler-sdk)

  therock_test_validate_shared_lib(PATH ${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-sdk/dist/lib
    LIB_NAMES
      librocprofiler-sdk.so
      librocprofiler-sdk-roctx.so
      librocprofiler-sdk-rocpd.so
      librocprofiler-sdk-rocattach.so
  )
  therock_test_validate_shared_lib(PATH ${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-sdk/dist/lib/rocprofiler-sdk
    LIB_NAMES
      libgotcha.so
      librocprofiler-sdk-tool-kokkosp.so
      librocprofv3-list-avail.so
  )

  ##############################################################################
  # rocprofiler-compute
  ##############################################################################
  therock_cmake_subproject_declare(rocprofiler-compute
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-compute"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-compute"
    BACKGROUND_BUILD
    DEFAULT_GPU_TARGETS
      gfx942
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DENABLE_TESTS=${THEROCK_BUILD_TESTING}
      -DINSTALL_TESTS=${THEROCK_BUILD_TESTING}
      -DTEST_FROM_INSTALL=${THEROCK_BUILD_TESTING}
      -DCHECK_PYTHON_DEPS=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      rocprofiler-sdk
      aqlprofile
      hip-clr
      rocprofiler-register
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_SQLITE3}
      ${_openmpi_optional_dep}
    )
    therock_cmake_subproject_glob_c_sources(rocprofiler-compute
      SUBDIRS .
    )

    therock_cmake_subproject_activate(rocprofiler-compute)

  ##############################################################################
  # roctracer
  # This is a very old, deprecated library with a number of quirks. Long term,
  # everything is supposed to migrate to rocprofiler-sdk, which has runtime
  # support to intercept any old roctracer clients when tracing is active.
  # To prepare for removal, anything that depends on roctracer should also
  # depend on rocprofiler-sdk.
  ##############################################################################

  therock_cmake_subproject_declare(roctracer
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/roctracer"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/roctracer"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN
      # Must build with the HIP compiler.
      amd-hip
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
    INTERFACE_INCLUDE_DIRS
      # All old clients of roctx64 expect to just be able to find its include
      # with no further qualification. Bad design, but also deprecated, so meh.
      include
    INTERFACE_LINK_DIRS
      # So that dependents can find the roctx library via find_library()
      lib
    RUNTIME_DEPS
      hip-clr
      ROCR-Runtime
  )
  therock_cmake_subproject_glob_c_sources(roctracer
    SUBDIRS .
  )
  therock_cmake_subproject_activate(roctracer)

  ##############################################################################
  # Artifacts
  ##############################################################################

  therock_provide_artifact(rocprofiler-sdk
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocprofiler-sdk.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      rocprofiler-sdk
      roctracer
      rocprof-trace-decoder
  )

  therock_provide_artifact(rocprofiler-compute
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocprofiler-compute.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      aqlprofile
      rocprofiler-sdk
      rocprofiler-compute
  )

  ##############################################################################
  # rocprofiler-systems
  ##############################################################################
  # rocprofiler-systems requires GCC, but sanitizer builds use Clang.
  # Skip enabling rocprofiler-systems when building with sanitizers.
  if (THEROCK_ENABLE_ROCPROFSYS AND THEROCK_SANITIZER STREQUAL "")
    # Escape semicolons in GPU targets list to prevent shell interpretation
    if (THEROCK_TEST_AMDGPU_TARGETS STREQUAL "THEROCK_TEST_AMDGPU_TARGETS-NOTFOUND")
      set(_ROCPROFSYS_GFX_TARGETS_ESCAPED "")
    else()
      string(REPLACE ";" "\\;" _ROCPROFSYS_GFX_TARGETS_ESCAPED "${THEROCK_TEST_AMDGPU_TARGETS}")
    endif()

    therock_detect_python_versions(_rocprofsys_detected_python_executables _rocprofsys_detected_python_versions)
    if(NOT _rocprofsys_detected_python_versions OR NOT _rocprofsys_detected_python_executables)
      message(STATUS "No python versions detected for rocprofiler-systems, disabling Python support")
      set(_ROCPROFSYS_USE_PYTHON OFF)
      set(_ROCPROFSYS_PYTHON_ROOT_DIRS_STR "")
      set(_ROCPROFSYS_PYTHON_VERSIONS_STR "")
    else()
      set(_ROCPROFSYS_USE_PYTHON ON)
      # Build root dirs from detected executables.
      set(_ROCPROFSYS_PYTHON_ROOT_DIRS)
      foreach(_exe IN LISTS _rocprofsys_detected_python_executables)
        cmake_path(GET _exe PARENT_PATH _bin_dir)
        cmake_path(GET _bin_dir PARENT_PATH _root_dir)
        list(APPEND _ROCPROFSYS_PYTHON_ROOT_DIRS "${_root_dir}")
      endforeach()
      list(JOIN _ROCPROFSYS_PYTHON_ROOT_DIRS "\;" _ROCPROFSYS_PYTHON_ROOT_DIRS_STR)
      list(JOIN _rocprofsys_detected_python_versions "\;" _ROCPROFSYS_PYTHON_VERSIONS_STR)
      message(STATUS "Detected Python versions for rocprofiler-systems: ${_ROCPROFSYS_PYTHON_VERSIONS_STR}")
      message(STATUS "Detected Python root dirs for rocprofiler-systems: ${_ROCPROFSYS_PYTHON_ROOT_DIRS_STR}")
    endif()

    therock_cmake_subproject_declare(rocprofiler-systems
      USE_TEST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-systems"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-systems"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCPROFSYS_BUILD_ELFUTILS=ON
        -DROCPROFSYS_BUILD_TBB=ON
        -DROCPROFSYS_BUILD_LIBIBERTY=ON
        -DROCPROFSYS_BUILD_BOOST=ON
        -DROCPROFSYS_BUILD_DYNINST=ON
        -DROCPROFSYS_BUILD_FOR_THEROCK=ON # TODO: Will be deprecated
        -DROCPROFSYS_INSTALL_TESTING="${THEROCK_BUILD_TESTING}"
        -DROCPROFSYS_INSTALL_EXAMPLES="${THEROCK_BUILD_TESTING}"
        -DROCPROFSYS_GFX_TARGETS="${_ROCPROFSYS_GFX_TARGETS_ESCAPED}"
        -DROCPROFSYS_BUILD_SPDLOG=OFF
        -DROCPROFSYS_BUILD_SQLITE3=OFF
        -DROCPROFSYS_USE_PYTHON="${_ROCPROFSYS_USE_PYTHON}"
        -DROCPROFSYS_PYTHON_VERSIONS:STRING="${_ROCPROFSYS_PYTHON_VERSIONS_STR}"
        -DROCPROFSYS_PYTHON_ROOT_DIRS:STRING="${_ROCPROFSYS_PYTHON_ROOT_DIRS_STR}"
        "-DROCPROFSYS_USE_MPI=${THEROCK_ENABLE_MPI}"
      INSTALL_RPATH_DIRS
        "lib"
        "lib/rocprofiler-systems"
        "llvm/lib"
      CMAKE_INCLUDES
        therock_explicit_finders.cmake
      BUILD_DEPS
        therock-spdlog
        therock-fmt
      RUNTIME_DEPS
        amdsmi
        aqlprofile
        hip-clr
        rocprofiler-register
        rocprofiler-sdk
        ${THEROCK_BUNDLED_LIBDRM}
        ${THEROCK_BUNDLED_SQLITE3}
        ${_openmpi_optional_dep}
    )
    therock_cmake_subproject_glob_c_sources(rocprofiler-systems
      SUBDIRS .
    )

    therock_cmake_subproject_activate(rocprofiler-systems)

    therock_provide_artifact(rocprofiler-systems
      TARGET_NEUTRAL
      DESCRIPTOR artifact-rocprofiler-systems.toml
      COMPONENTS
        dbg
        dev
        doc
        lib
        run
        test
      SUBPROJECT_DEPS
        amdsmi
        rocprofiler-sdk
        rocprofiler-systems
      )
  endif(THEROCK_ENABLE_ROCPROFSYS AND THEROCK_SANITIZER STREQUAL "")
endif(THEROCK_ENABLE_ROCPROFV3)
