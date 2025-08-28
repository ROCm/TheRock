cmake_minimum_required(VERSION 3.10)
project(hipInfo CXX)

# Prefer config packages (TheRock provides hip via config files)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# Use TheRock-provided HIP
find_package(hip REQUIRED CONFIG)

# Locate sources (hip-tests is a subproject under rocm-systems)
set(HIP_TESTS_DIR CACHE STRING "")  # Set by the superproject.
set(HIPINFO_SRC "${HIP_TESTS_DIR}/samples/1_Utils/hipInfo/hipInfo.cpp")
set(HIPINFO_INC "${HIP_TESTS_DIR}/samples/common")

if(NOT EXISTS "${HIPINFO_SRC}")
  message(FATAL_ERROR "hipInfo.cpp not found at: ${HIPINFO_SRC}")
endif()

add_executable(hipInfo "${HIPINFO_SRC}")
target_include_directories(hipInfo PRIVATE "${HIPINFO_INC}")
target_link_libraries(hipInfo PRIVATE hip::host)

include(GNUInstallDirs)
install(TARGETS hipInfo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
