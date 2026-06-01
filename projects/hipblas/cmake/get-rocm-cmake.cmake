# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install

include(FetchContent)

find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS "${ROCM_PATH}")
if(NOT ROCmCMakeBuildTools_FOUND)
  find_package(ROCM 0.11.0 CONFIG QUIET PATHS "${ROCM_PATH}") # deprecated fallback
  if(NOT ROCM_FOUND)
    message(STATUS "ROCmCMakeBuildTools not found. Fetching...")
    set(rocm_cmake_tag "rocm-6.4.0" CACHE STRING "rocm-cmake tag to download")
    FetchContent_Declare(
      rocm-cmake
      GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
      GIT_TAG        ${rocm_cmake_tag}
      SOURCE_SUBDIR "DISABLE ADDING TO BUILD" # We don't really want to consume the build and test targets of ROCm CMake.
    )
    FetchContent_MakeAvailable(rocm-cmake)
    find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
  endif()
endif()
