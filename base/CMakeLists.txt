################################################################################
# aux-overlay
################################################################################

therock_cmake_subproject_declare(therock-aux-overlay
  EXTERNAL_SOURCE_DIR "aux-overlay"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
)
therock_cmake_subproject_activate(therock-aux-overlay)

################################################################################
# rocm-cmake
################################################################################

therock_cmake_subproject_declare(rocm-cmake
  EXTERNAL_SOURCE_DIR "rocm-cmake"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
)
therock_cmake_subproject_provide_package(rocm-cmake
  ROCmCMakeBuildTools share/rocmcmakebuildtools/cmake)
therock_cmake_subproject_provide_package(rocm-cmake
  ROCM share/rocm/cmake)
therock_cmake_subproject_activate(rocm-cmake)


################################################################################
# rocm-core
################################################################################

if(WIN32)
  set(_shared_libs_arg "OFF")
else()
  set(_shared_libs_arg "ON")
endif()

therock_cmake_subproject_declare(rocm-core
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocm-core"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocm-core"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  CMAKE_ARGS
    "-DBUILD_SHARED_LIBS=${_shared_libs_arg}"
    "-DROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
)
therock_cmake_subproject_glob_c_sources(rocm-core
  SUBDIRS .
)
therock_cmake_subproject_provide_package(rocm-core rocm-core lib/cmake/rocm-core)
therock_cmake_subproject_activate(rocm-core)


################################################################################
# rocm_smi_lib
################################################################################

if(NOT WIN32)  # TODO(#36): Enable on Windows and/or make subproject inclusion generally optional

therock_cmake_subproject_declare(rocm_smi_lib
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocm-smi-lib"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocm_smi_lib"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  CMAKE_ARGS
    # See the post_hook, which needs to advertise install interface directories
    # in this case.
    "-DTHEROCK_HAS_BUNDLED_LIBDRM=${THEROCK_BUNDLED_LIBDRM}"
  INTERFACE_LINK_DIRS
    "lib"
  INSTALL_RPATH_DIRS
    "lib"
  INTERFACE_INSTALL_RPATH_DIRS
    "lib"
  INTERFACE_PROGRAM_DIRS
    "bin"
  RUNTIME_DEPS
    ${THEROCK_BUNDLED_LIBDRM}
)
therock_cmake_subproject_glob_c_sources(rocm_smi_lib
  SUBDIRS
    include
    src
)
therock_cmake_subproject_provide_package(rocm_smi_lib rocm_smi lib/cmake/rocm_smi)
therock_cmake_subproject_activate(rocm_smi_lib)

endif()


################################################################################
# rocprofiler-register
# This is a stub that helps runtime libraries and profiles rendezvous
################################################################################

if(NOT WIN32)  # TODO(#36): Enable on Windows?

therock_cmake_subproject_declare(rocprofiler-register
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-register"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-register"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  INTERFACE_LINK_DIRS
    "lib"
  INTERFACE_INSTALL_RPATH_DIRS
    "lib"
)
therock_cmake_subproject_glob_c_sources(rocprofiler-register
  SUBDIRS
    source/include
    source/lib
)
therock_cmake_subproject_provide_package(rocprofiler-register
  rocprofiler-register lib/cmake/rocprofiler-register)
therock_cmake_subproject_activate(rocprofiler-register)

endif()

################################################################################
# rocm-half
################################################################################

therock_cmake_subproject_declare(rocm-half
  EXTERNAL_SOURCE_DIR "half"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  INTERFACE_INCLUDE_DIRS
    # No find_package support (just a naked include).
    include
  BUILD_DEPS
    rocm-cmake
)
therock_cmake_subproject_glob_c_sources(rocm-half
  SUBDIRS
    include
)
therock_cmake_subproject_activate(rocm-half)


################################################################################
# Artifacts
################################################################################

set(_optional_subproject_deps)
if(NOT WIN32)
  # TODO(#36): Enable on Windows and/or make subproject inclusion generally optional
  list(APPEND _optional_subproject_deps rocm_smi_lib)
  list(APPEND _optional_subproject_deps rocprofiler-register)
endif()

therock_provide_artifact(base
  TARGET_NEUTRAL
  DESCRIPTOR artifact.toml
  COMPONENTS
    dbg
    dev
    doc
    lib
    run
    test
  SUBPROJECT_DEPS
    ${_optional_subproject_deps}
    rocm-cmake
    rocm-core
    rocm-half
    therock-aux-overlay
)
