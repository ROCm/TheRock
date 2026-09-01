# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_RCCL)
  # Most libraries need to depend on the profiler, but is conditionally only used
  # on Posix.
  set(optional_profiler_deps)

  if(THEROCK_FLAG_INCLUDE_PROFILER)
    list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
  endif()

  # RCCL can optionally use amdsmi/rocm_smi_lib on Linux, but those may not be
  # enabled/declared depending on feature selection.
  set(_rccl_optional_runtime_deps)
  if(NOT WIN32)
    if(TARGET amdsmi)
      list(APPEND _rccl_optional_runtime_deps amdsmi)
    endif()
    if(TARGET rocm_smi_lib)
      list(APPEND _rccl_optional_runtime_deps rocm_smi_lib)
    endif()
  endif()

  ##############################################################################
  # rccl
  ##############################################################################
  set(_rccl_subproject_names)

  therock_cmake_subproject_declare(rccl
    # Build all GPU targets in a single target-neutral artifact as long as the
    # host code isn't fully target agnostic.
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rccl"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl"
    # High latency LTO link of a single library.
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      # TODO(#56): Enable once reworked.
      -DENABLE_MSCCLPP=OFF
      -DENABLE_MSCCL_KERNEL=OFF
      -DBUILD_TESTS=${THEROCK_BUILD_TESTING}
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-fmt
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      hipify
      rocprofiler-register
      ${_rccl_optional_runtime_deps}
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_glob_c_sources(rccl
    SUBDIRS
      src
  )
  therock_cmake_subproject_provide_package(rccl RCCL lib/cmake/rccl)
  therock_cmake_subproject_activate(rccl)
  list(APPEND _rccl_subproject_names rccl)

  # rccl-translate unbundles gfx1250 device code from librccl.so. Under ASAN,
  # GPU_TARGETS is filtered to xnack+ arches only (gfx942, gfx950), so gfx1250
  # is never compiled into librccl.so and the unbundle step fails with
  # "No code objects for architecture 'gfx1250'".
  if(THEROCK_ENABLE_ROCJITSU_HOTSWAP AND THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS
     AND NOT THEROCK_SANITIZER)
    therock_cmake_subproject_dist_dir(_rccl_dist_dir rccl)
    therock_cmake_subproject_dist_dir(_rocjitsu_dist_dir rocjitsu)
    therock_cmake_subproject_dist_dir(_rocjitsu_hotswap_dist_dir rocjitsu-hotswap)

    # Generate the install-owned translation tier from RCCL before KPACK splits
    # its fat binary. The splitter then assigns this tree only to gfx1250.
    therock_cmake_subproject_declare(rccl-translate
      DISABLE_AMDGPU_TARGETS
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      FPRINT_SOURCE_HASH
      EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rccl-translate"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl-translate"
      CMAKE_ARGS
        -DRCCL_ROOT=${_rccl_dist_dir}
        -DROCJITSU_ROOT=${_rocjitsu_dist_dir}
        -DROCJITSU_HOTSWAP_ROOT=${_rocjitsu_hotswap_dist_dir}
        -DROCJITSU_PRETRANSLATE_DRIVER=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/emulation/rocjitsu/scripts/rocjitsu-pretranslate.py
        -DKPACK_PYTHON_SOURCE=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python
        # TODO: Set ON once RocJitsu supports every RCCL gfx1250 code object.
        -DRCCL_TRANSLATE_FAIL_ON_SKIPPED=OFF
      EXTRA_DEPENDS
        ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/emulation/rocjitsu/scripts/rocjitsu-pretranslate.py
        ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python/rocm_kpack/binutils.py
        ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python/rocm_kpack/ccob_parser.py
        ${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python/rocm_kpack/tools/bulk_unbundle.py
      BUILD_DEPS
        rccl
        rocjitsu
        rocjitsu-hotswap
        rocm-kpack
    )
    therock_cmake_subproject_activate(rccl-translate)
    list(APPEND _rccl_subproject_names rccl-translate)
  endif()

  if(THEROCK_BUILD_TESTING)

    set(_rccl_tests_optional_deps)
    if(THEROCK_ENABLE_MPI)
      set(_rccl_tests_optional_deps "therock-openmpi")
    endif()

    therock_cmake_subproject_declare(rccl-tests
      USE_TEST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rccl-tests"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl-tests"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
#        -DROCM_DIR=
        "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
        -DUSE_MPI=${THEROCK_ENABLE_MPI}
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        rocm-cmake
      RUNTIME_DEPS
        hip-clr
        hipify
        rccl
        ${_rccl_tests_optional_deps}
    )
    therock_cmake_subproject_glob_c_sources(rccl-tests
      SUBDIRS
        src
    )
    therock_cmake_subproject_activate(rccl-tests)
    list(APPEND _rccl_subproject_names rccl-tests)
  endif(THEROCK_BUILD_TESTING)

  therock_provide_artifact(rccl
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rccl.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      ${_rccl_subproject_names}
  )
endif()

# Note that rocSHMEM is *not* declared here: it must be configured before
# profiler/ (rocprofiler-sdk optionally traces its API) whereas rccl above
# depends on the profiler. It lives in the rocshmem/ subdirectory, which the
# root CMakeLists.txt adds ahead of profiler/.
