# Build with:
#   cmake -GNinja -S. -Bbuild -DCMAKE_PREFIX_PATH=path/to/rocm
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory.")
endif()

cmake_minimum_required(VERSION 3.25)

project(rocm-cpp-sdk-user LANGUAGES CXX HIP)
enable_testing()

set(CMAKE_CXX_STANDARD 17)

include(CMakeDependentOption)

# If defining a new option, ensure it is in the outer CMakeLists.txt and
# the clean_configure_test_project.cmake to ensure it is passed from the
# parent when testing.
option(THEROCK_ENABLE_BLAS "Whether hipblas/hipblaslt/rocblas are available" ON)
option(THEROCK_ENABLE_FFT "Whether the FFT libraries are available" ON)
option(THEROCK_ENABLE_HIP "Whether the HIP runtime and compiler are available" ON)
option(THEROCK_ENABLE_HIPDNN "Whether hipDNN is available" ON)
option(THEROCK_ENABLE_MIOPEN "Whether MIOpen is available" ON)
option(THEROCK_ENABLE_PRIM "Whether rocprim is available" ON)
option(THEROCK_ENABLE_RAND "Whether rocrand/hiprand are available" ON)
option(THEROCK_ENABLE_RCCL "Whether rccl is available" ON)
option(THEROCK_ENABLE_SOLVER "Whether rocsolver/hipsolver are available" ON)
option(THEROCK_ENABLE_SPARSE "Whether rocsparse/hipsparse are available" ON)
option(ENABLE_DEVICE_TEST "Whether to enable testing that requires a device" OFF)

set(TEST_PATH)
if(WIN32)
    # On Windows, when not installed as a global assembly, the ROCM bin dir
    # must be on the PATH in order to find DLLs.
    set(TEST_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}/bin")
    string(APPEND TEST_PATH ";")
    string(APPEND TEST_PATH "$ENV{PATH}")
    message(STATUS "Using test path: ${TEST_PATH}")
endif()

if(THEROCK_ENABLE_HIP)
    # TODO: Don't require HIP_PLATFORM https://github.com/ROCm/TheRock/issues/68
    set(HIP_PLATFORM "amd")
    find_package(hip CONFIG REQUIRED)
    message(STATUS "**** HIP Configuration:")
    message(STATUS "  VERSION: ${hip_VERSION}")
    message(STATUS "  INCLUDE: ${hip_INCLUDE_DIRS}")
    message(STATUS "  LIBRARIES: ${hip_LIBRARIES}")
    message(STATUS "  GPU_TARGETS: ${GPU_TARGETS}")
endif()

if(THEROCK_ENABLE_RAND)
    find_package(hiprand CONFIG REQUIRED)
    find_package(rocrand CONFIG REQUIRED)
    message(STATUS "hiprand version: ${hiprand_VERSION}")
    message(STATUS "rocrand version: ${rocrand_VERSION}")
endif()

if(THEROCK_ENABLE_SPARSE)
    find_package(hipsparse CONFIG REQUIRED)
    find_package(rocsparse CONFIG REQUIRED)
    message(STATUS "hipsparse version: ${hipsparse_VERSION}")
    message(STATUS "rocsparse version: ${rocsparse_VERSION}")
endif()

if(THEROCK_ENABLE_PRIM)
    find_package(rocprim CONFIG REQUIRED)
    find_package(rocthrust CONFIG REQUIRED)
    find_package(hipcub CONFIG REQUIRED)
    message(STATUS "rocprim version: ${rocprim_VERSION}")
    message(STATUS "rocthrust version: ${rocthrust_VERSION}")
    message(STATUS "hipcub version: ${hipcub_VERSION}")
endif()

if(THEROCK_ENABLE_SOLVER)
    find_package(hipsolver CONFIG REQUIRED)
    find_package(rocsolver CONFIG REQUIRED)
    message(STATUS "hipsolver version: ${hipsolver_VERSION}")
    message(STATUS "rocsolver version: ${rocsolver_VERSION}")
endif()

if(THEROCK_ENABLE_RCCL)
    find_package(rccl CONFIG REQUIRED)
    message(STATUS "rccl version: ${rccl_VERSION}")
endif()

if(THEROCK_ENABLE_BLAS)
    find_package(hipblaslt CONFIG REQUIRED)
    find_package(hipblas CONFIG REQUIRED)
    find_package(rocblas CONFIG REQUIRED)
    message(STATUS "hipblaslt version: ${hipblaslt_VERSION}")
    message(STATUS "hipblas version: ${hipblas_VERSION}")
    message(STATUS "rocblas version: ${rocblas_VERSION}")
endif()

if(THEROCK_ENABLE_MIOPEN)
    find_package(miopen CONFIG REQUIRED)
    message(STATUS "miopen version: ${miopen_VERSION}")
endif()

if(THEROCK_ENABLE_FFT)
    find_package(hipfft CONFIG REQUIRED)
    find_package(rocfft CONFIG REQUIRED)
    message(STATUS "hipfft version: ${hipfft_VERSION}")
    message(STATUS "rocfft version: ${rocfft_VERSION}")
endif()

if(THEROCK_ENABLE_HIP)
    add_executable(
        hip-host-test
        hip-host-test.cpp
    )
    target_link_libraries(hip-host-test PRIVATE hip::host)
    set(_DEVICE_ORDINAL)
    if(ENABLE_DEVICE_TEST)
        set(_DEVICE_ORDINAL 0)
    endif()
    add_test(NAME hip-host-test
        COMMAND hip-host-test ${_DEVICE_ORDINAL}
    )
    if(TEST_PATH)
        set_tests_properties(
            hip-host-test
            PROPERTIES ENVIRONMENT "PATH=${TEST_PATH}"
        )
    endif()

    add_executable(sample_kernel sample_kernel.hip)
    set_source_files_properties(sample_kernel PROPERTIES LANGUAGE HIP)
    if(ENABLE_DEVICE_TEST)
        add_test(NAME sample-kernel-test COMMAND sample_kernel)
        if(TEST_PATH)
            set_tests_properties(
                sample-kernel-test
                PROPERTIES ENVIRONMENT "PATH=${TEST_PATH}"
            )
        endif()
    endif()
endif()

if(THEROCK_ENABLE_HIPDNN)
    find_package(hipdnn_backend CONFIG REQUIRED)
    message(STATUS "hipdnn version: ${hipdnn_VERSION}")
endif()
