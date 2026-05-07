
if (DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
elseif (DEFINED ENV{HIP_DIR})
  file(TO_CMAKE_PATH "$ENV{HIP_DIR}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
else()
  set(HIP_DIR "C:/hip")
  set(rocm_bin "C:/hip/bin")
endif()

set(CMAKE_CXX_COMPILER "${rocm_bin}/clang++.exe")

# working
#set(CMAKE_Fortran_COMPILER "C:/Strawberry/c/bin/gfortran.exe")

#set(CMAKE_Fortran_PREPROCESS_SOURCE "<CMAKE_Fortran_COMPILER> -E <INCLUDES> <FLAGS> -cpp <SOURCE> -o <PREPROCESSED_SOURCE>")

# TODO remove, just to speed up slow cmake
#set(CMAKE_CXX_COMPILER_WORKS 1)
#set(CMAKE_Fortran_COMPILER_WORKS 1)

if (NOT HIPBLAS_TOOLCHAIN_VARS_APPENDED)
  set(HIPBLAS_TOOLCHAIN_VARS_APPENDED True)

  # our usage flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32 -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING")

  # flags for clang direct use

  # -Wno-ignored-attributes to avoid warning: __declspec attribute 'dllexport' is not supported [-Wignored-attributes] which is used by msvc compiler
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIP_CLANG_HCC_COMPAT_MODE=1")

  # args also in hipcc.bat
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions -fms-compatibility -D__HIP_ROCclr__=1 -D__HIP_PLATFORM_AMD__=1")
endif()

# paths required for client testing on windows
if (NOT ROCBLAS_PATH)
  if (ROCBLAS_CUSTOM_PATH)
    file(TO_CMAKE_PATH "${ROCBLAS_CUSTOM_PATH}" ROCBLAS_PATH )
  else()
    set(ROCBLAS_PATH "C:/hipSDK")
  endif()
endif()

# paths required for client testing on windows
if (NOT ROCSOLVER_PATH)
  if (ROCSOLVER_CUSTOM_PATH)
    file(TO_CMAKE_PATH "${ROCSOLVER_CUSTOM_PATH}" ROCSOLVER_PATH )
  else()
    set(ROCSOLVER_PATH "C:/hipSDK")
  endif()
endif()

if (DEFINED ENV{LAPACK_DIR})
  file(TO_CMAKE_PATH "$ENV{LAPACK_DIR}" LAPACK_DIR)
else()
  set(LAPACK_DIR "C:/lapack/build")
endif()

if (DEFINED ENV{BLIS_DIR})
  file(TO_CMAKE_PATH "$ENV{BLIS_DIR}" BLIS_DIR)
else()
  set(BLIS_DIR "C:/blis/blis-0.8.1-h8d14728_1/Library")
endif()

if (DEFINED ENV{CBLAS_DIR})
  file(TO_CMAKE_PATH "$ENV{CBLAS_DIR}" CBLAS_DIR)
else()
  set(CBLAS_DIR "C:/lapack/build")
endif()

if (DEFINED ENV{OPENBLAS_DIR})
  file(TO_CMAKE_PATH "$ENV{OPENBLAS_DIR}" OPENBLAS_DIR)
else()
  set(OPENBLAS_DIR "C:/OpenBLAS/OpenBLAS-0.3.18-x64")
endif()

if (DEFINED ENV{VCPKG_PATH})
  file(TO_CMAKE_PATH "$ENV{VCPKG_PATH}" VCPKG_PATH)
else()
  set(VCPKG_PATH "C:/github/vcpkg")
endif()
include("${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_STATIC_LIBRARY_PREFIX "static_")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
