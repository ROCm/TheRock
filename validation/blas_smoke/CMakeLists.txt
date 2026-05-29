cmake_minimum_required(VERSION 3.24)

project(blas_macos_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROCM_ROOT "" CACHE PATH "Path to the ROCm distribution under test")
if(NOT ROCM_ROOT)
  message(FATAL_ERROR "Pass -DROCM_ROOT=/path/to/rocm")
endif()

find_library(HIPBLAS_LIBRARY NAMES hipblas PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HIP_LIBRARY NAMES amdhip64 PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(hipblas_saxpy_smoke hipblas_saxpy_smoke.cpp)
target_include_directories(hipblas_saxpy_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(hipblas_saxpy_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(hipblas_saxpy_smoke PRIVATE "${HIPBLAS_LIBRARY}" "${HIP_LIBRARY}")
set_target_properties(hipblas_saxpy_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")

add_executable(hipblas_gemm_smoke hipblas_gemm_smoke.cpp)
target_include_directories(hipblas_gemm_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(hipblas_gemm_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(hipblas_gemm_smoke PRIVATE "${HIPBLAS_LIBRARY}" "${HIP_LIBRARY}")
set_target_properties(hipblas_gemm_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")
