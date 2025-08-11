cmake_minimum_required(VERSION 3.10)
project(hipInfo CXX)

# Prefer config packages (TheRock provides hip via config files)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Use TheRock-provided HIP toolchain (paths injected by TheRock)
find_package(hip REQUIRED CONFIG)

# --- Important for Windows: force hipcc as the compiler to avoid empty rules ---
if (WIN32 AND DEFINED HIP_HIPCC_EXECUTABLE)
  # hipcc.bat doesn't pass CMake's probe reliably; mark compilers as working
  set(CMAKE_C_COMPILER_WORKS 1)
  set(CMAKE_CXX_COMPILER_WORKS 1)
  set(CMAKE_CXX_COMPILER "${HIP_HIPCC_EXECUTABLE}")
  set(CMAKE_CXX_LINKER   "${HIP_HIPCC_EXECUTABLE}")
  # Help Ninja when command lines get long
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
endif()

# Locate the upstream sources relative to this wrapper directory.
# Layout: core/hipInfo-ds (this file) and core/hip-tests are siblings.
set(HIP_TESTS_DIR "${CMAKE_CURRENT_LIST_DIR}/../hip-tests")
set(HIPINFO_SRC "${HIP_TESTS_DIR}/samples/1_Utils/hipInfo/hipInfo.cpp")
set(HIPINFO_INC "${HIP_TESTS_DIR}/samples/common")

if(NOT EXISTS "${HIPINFO_SRC}")
  message(FATAL_ERROR "hipInfo.cpp not found at: ${HIPINFO_SRC}")
endif()

add_executable(hipInfo "${HIPINFO_SRC}")
target_include_directories(hipInfo PRIVATE "${HIPINFO_INC}")
target_link_libraries(hipInfo PRIVATE hip::host)

# Optional: make platform explicit (usually not required)
set(HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)

# Install so TheRock can package from stage/bin
include(GNUInstallDirs)
install(TARGETS hipInfo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
