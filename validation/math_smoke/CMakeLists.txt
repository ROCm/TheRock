cmake_minimum_required(VERSION 3.24)

project(math_macos_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROCM_ROOT "" CACHE PATH "Path to the ROCm distribution under test")
if(NOT ROCM_ROOT)
  message(FATAL_ERROR "Pass -DROCM_ROOT=/path/to/rocm")
endif()

find_library(HIP_LIBRARY NAMES amdhip64 PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HIPRAND_LIBRARY NAMES hiprand PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HIPFFT_LIBRARY NAMES hipfft PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HIPSPARSE_LIBRARY NAMES hipsparse PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HIPSOLVER_LIBRARY NAMES hipsolver PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HIPBLAS_LIBRARY NAMES hipblas PATHS "${ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(hiprand_uniform_smoke hiprand_uniform_smoke.cpp)
target_include_directories(hiprand_uniform_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(hiprand_uniform_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(hiprand_uniform_smoke PRIVATE "${HIPRAND_LIBRARY}" "${HIP_LIBRARY}")
set_target_properties(hiprand_uniform_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")

add_executable(hipfft_c2c_smoke hipfft_c2c_smoke.cpp)
target_include_directories(hipfft_c2c_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(hipfft_c2c_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(hipfft_c2c_smoke PRIVATE "${HIPFFT_LIBRARY}" "${HIP_LIBRARY}")
set_target_properties(hipfft_c2c_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")

add_executable(hipsparse_spmv_smoke hipsparse_spmv_smoke.cpp)
target_include_directories(hipsparse_spmv_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(hipsparse_spmv_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(hipsparse_spmv_smoke PRIVATE "${HIPSPARSE_LIBRARY}" "${HIP_LIBRARY}")
set_target_properties(hipsparse_spmv_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")

add_executable(hipsolver_potrf_smoke hipsolver_potrf_smoke.cpp)
target_include_directories(hipsolver_potrf_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(hipsolver_potrf_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(hipsolver_potrf_smoke PRIVATE "${HIPSOLVER_LIBRARY}" "${HIP_LIBRARY}")
set_target_properties(hipsolver_potrf_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")

add_executable(rocm_combined_stack_smoke rocm_combined_stack_smoke.cpp)
target_include_directories(rocm_combined_stack_smoke PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(rocm_combined_stack_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(rocm_combined_stack_smoke PRIVATE
  "${HIPFFT_LIBRARY}"
  "${HIPSOLVER_LIBRARY}"
  "${HIPSPARSE_LIBRARY}"
  "${HIPRAND_LIBRARY}"
  "${HIPBLAS_LIBRARY}"
  "${HIP_LIBRARY}")
set_target_properties(rocm_combined_stack_smoke PROPERTIES
  BUILD_RPATH "${ROCM_ROOT}/lib")
