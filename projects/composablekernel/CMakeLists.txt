# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0140)
  # policies CMP0140 not known to CMake until 3.25
  cmake_policy(SET CMP0140 NEW)
endif()

get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(_GENERATOR_IS_MULTI_CONFIG)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING
            "Available build types (configurations) on multi-config generators")
else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

# Allow user to specify the C++ standard.
# We must support C++17 builds until downstream users are migrated to C++20, but we default to C++20.
set(CK_CXX_STANDARD "20" CACHE STRING "C++ standard to use (e.g. 17 or 20)")
set(valid_cxx_standards 17 20)
set_property(CACHE CK_CXX_STANDARD PROPERTY STRINGS ${valid_cxx_standards})
if(NOT CK_CXX_STANDARD IN_LIST valid_cxx_standards)
    message(FATAL_ERROR "CK_CXX_STANDARD must be one of ${valid_cxx_standards}")
endif()

# Default installation path
if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

set(version 1.2.0)
# Check support for CUDA/HIP in Cmake
project(composable_kernel VERSION ${version} LANGUAGES CXX HIP)
include(CTest)

option(ENABLE_CLANG_CPP_CHECKS "Enables clang tidy, cppcheck" ON)
option(MIOPEN_REQ_LIBS_ONLY "Build only the MIOpen required libraries" OFF)
option(CK_EXPERIMENTAL_BUILDER "Enable experimental builder" OFF)
option(BUILD_MHA_LIB "Build the static library for flash attention" OFF)
option(FORCE_DISABLE_XDL "Skip compiling XDL specific instances (even if supported GPUs are included in GPU_TARGETS)" OFF)
option(FORCE_DISABLE_WMMA "Skip compiling WMMA specific instances (even if supported GPUs are included in GPU_TARGETS)" OFF)

if(CK_EXPERIMENTAL_BUILDER)
    add_definitions(-DCK_EXPERIMENTAL_BUILDER)
    include_directories(${PROJECT_SOURCE_DIR}/experimental/builder/include)
endif()

# Usage: for customized Python location cmake -DCK_USE_ALTERNATIVE_PYTHON="/opt/Python-3.8.13/bin/python3.8"
# CK Codegen requires dataclass which is added in Python 3.7
# Python version 3.8 is required for general good practice as it is default for Ubuntu 20.04
if(NOT CK_USE_ALTERNATIVE_PYTHON)
   find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)
else()
   message(STATUS "Using alternative python version")
   set(EXTRA_PYTHON_PATH)
   # this is overly restrictive, we may need to be more flexible on the following
   string(REPLACE "/bin/python3.8" "" EXTRA_PYTHON_PATH "${CK_USE_ALTERNATIVE_PYTHON}")
   message(STATUS "alternative python path is: ${EXTRA_PYTHON_PATH}")
   find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)
   add_definitions(-DPython3_EXECUTABLE="${CK_USE_ALTERNATIVE_PYTHON}")
   set(Python3_EXECUTABLE "${CK_USE_ALTERNATIVE_PYTHON}")
   set(PYTHON_EXECUTABLE "${CK_USE_ALTERNATIVE_PYTHON}")
   set(ENV{LD_LIBRARY_PATH} "${EXTRA_PYTHON_PATH}/lib:$ENV{LD_LIBRARY_PATH}")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if (DTYPES)
    add_definitions(-DDTYPES)
    if (DTYPES MATCHES "int8")
        add_definitions(-DCK_ENABLE_INT8)
        set(CK_ENABLE_INT8 "ON")
    endif()
    if (DTYPES MATCHES "fp8")
        add_definitions(-DCK_ENABLE_FP8)
        set(CK_ENABLE_FP8 "ON")
    endif()
    if (DTYPES MATCHES "bf8")
        add_definitions(-DCK_ENABLE_BF8)
        set(CK_ENABLE_BF8 "ON")
    endif()
    if (DTYPES MATCHES "fp16")
        add_definitions(-DCK_ENABLE_FP16)
        set(CK_ENABLE_FP16 "ON")
    endif()
    if (DTYPES MATCHES "fp32")
        add_definitions(-DCK_ENABLE_FP32)
        set(CK_ENABLE_FP32 "ON")
    endif()
    if (DTYPES MATCHES "tf32")
        # definition will be added based on the GPU target in the following section
        set(CK_ENABLE_TF32 "ON")
    endif()
    if (DTYPES MATCHES "fp64")
        add_definitions(-DCK_ENABLE_FP64)
        set(CK_ENABLE_FP64 "ON")
    endif()
    if (DTYPES MATCHES "bf16")
        add_definitions(-DCK_ENABLE_BF16)
        set(CK_ENABLE_BF16 "ON")
    endif()
    message(STATUS "DTYPES macro set to ${DTYPES}")
else()
    add_definitions(-DCK_ENABLE_INT8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_BF16 -DCK_ENABLE_FP8 -DCK_ENABLE_BF8)
    set(CK_ENABLE_INT8 "ON")
    set(CK_ENABLE_FP16 "ON")
    set(CK_ENABLE_FP32 "ON")
    set(CK_ENABLE_TF32 "ON")
    set(CK_ENABLE_FP64 "ON")
    set(CK_ENABLE_BF16 "ON")
    set(CK_ENABLE_FP8 "ON")
    set(CK_ENABLE_BF8 "ON")
endif()

#for f8/bf8_t type
add_compile_options(-Wno-bit-int-extension)
add_compile_options(-Wno-pass-failed)
add_compile_options(-Wno-switch-default)
add_compile_options(-Wno-unique-object-duplication)

# add -Og -gdwarf64 for debug builds
add_compile_options(
    "$<$<CONFIG:Debug>:-Og>"
    "$<$<CONFIG:Debug>:-gdwarf64>"
)

# Recent change in compiler makes this warning ON by default, which led to compile errors.
add_compile_options(-Wno-nrvo)

if(NOT DISABLE_DL_KERNELS AND GPU_TARGETS MATCHES "gfx101|gfx103|gfx10-1|gfx10-3")
    add_definitions(-DDL_KERNELS)
    set(DL_KERNELS "ON")
    set(CK_ENABLE_DL_KERNELS "ON")
endif()
if(NOT DISABLE_DPP_KERNELS)
    add_definitions(-DDPP_KERNELS)
    set(DPP_KERNELS "ON")
    set(CK_ENABLE_DPP_KERNELS "ON")
endif()
option(CK_USE_CODEGEN "Enable codegen library" OFF)
if(CK_USE_CODEGEN)
   add_definitions(-DCK_USE_CODEGEN)
endif()

option(CK_TIME_KERNEL "Enable kernel time tracking" ON)
if(CK_TIME_KERNEL)
    add_definitions(-DCK_TIME_KERNEL=1)
else()
    add_definitions(-DCK_TIME_KERNEL=0)
endif()

include(getopt)

# CK version file to record release version as well as git commit hash
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD OUTPUT_VARIABLE COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(include/ck/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck/version.h)

set(ROCM_SYMLINK_LIBS OFF)
find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)
include(ROCMCheckTargetIds)
include(TargetFlags)

rocm_setup_version(VERSION ${version})

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}")

message(STATUS "GPU_TARGETS= ${GPU_TARGETS}")
message(STATUS "GPU_ARCHS= ${GPU_ARCHS}")
if(GPU_ARCHS)
    #disable GPU_TARGETS to avoid conflicts, this needs to happen before we call hip package
    unset(GPU_TARGETS CACHE)
    unset(AMDGPU_TARGETS CACHE)
endif()
if(GPU_TARGETS)
    set(USER_GPU_TARGETS 1)
else()
    set(USER_GPU_TARGETS 0)
endif()
find_package(hip REQUIRED)
# No assumption that HIP kernels are launched with uniform block size for backward compatibility
# SWDEV-413293 and https://reviews.llvm.org/D155213
math(EXPR hip_VERSION_FLAT "(${hip_VERSION_MAJOR} * 1000 + ${hip_VERSION_MINOR}) * 100000 + ${hip_VERSION_PATCH}")
message(STATUS "hip_version_flat=${hip_VERSION_FLAT}")

message(STATUS "checking which targets are supported")
#In order to build just the CK library (without tests and examples) for all supported GPU targets
#use -D GPU_ARCHS="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
#the GPU_TARGETS flag will be reset in this case in order to avoid conflicts.
#
#In order to build CK along with all tests and examples it should be OK to set GPU_TARGETS to just 1 or 2 similar architectures.
if(NOT ENABLE_ASAN_PACKAGING)
    if(NOT WIN32 AND ${hip_VERSION_FLAT} LESS 600300000)
        # WORKAROUND: compiler does not yet fully support gfx12 targets, need to fix version above
        set(CK_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102")
    elseif(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER_EQUAL 600300000 AND ${hip_VERSION_FLAT} LESS 600400000)
        set(CK_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201")
    elseif(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER_EQUAL 600400000 AND ${hip_VERSION_FLAT} LESS 600443483)
        set(CK_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx950")
    elseif(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER_EQUAL 600443483)
        set(CK_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx950;gfx10-3-generic;gfx11-generic;gfx12-generic")
    endif()
else()
    #build CK only for xnack-supported targets when using ASAN
    set(CK_GPU_TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+")
endif()

#if user set GPU_ARCHS on the cmake command line, overwrite default target list with user's list
#otherwise, if user set GPU_TARGETS, use that set of targets
if(GPU_ARCHS)
    set(CK_GPU_TARGETS ${GPU_ARCHS})
else()
    if(USER_GPU_TARGETS)
        set(CK_GPU_TARGETS ${GPU_TARGETS})
    endif()
endif()
#if the user did not set GPU_TARGETS, delete whatever was set by HIP package
if(NOT USER_GPU_TARGETS)
    set(GPU_TARGETS "")
endif()
#make sure all the targets on the list are actually supported by the current compiler
rocm_check_target_ids(SUPPORTED_GPU_TARGETS
        TARGETS ${CK_GPU_TARGETS})

message(STATUS "Building CK for the following targets: ${SUPPORTED_GPU_TARGETS}")

# Cache SUPPORTED_GPU_TARGETS for debug
set(SUPPORTED_GPU_TARGETS "${SUPPORTED_GPU_TARGETS}" CACHE STRING "List of supported GPU targets")

if (SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx11|gfx12" AND NOT FORCE_DISABLE_XDL)
    message(STATUS "Enabling XDL instances")
    add_definitions(-DCK_USE_XDL)
    set(CK_USE_XDL "ON")
endif()
if ((SUPPORTED_GPU_TARGETS MATCHES "gfx94" OR SUPPORTED_GPU_TARGETS MATCHES "gfx95") AND NOT FORCE_DISABLE_XDL)
    message(STATUS "Enabling XDL FP8 gemms on native architectures")
    add_definitions(-DCK_USE_GFX94)
    set(CK_USE_GFX94 "ON")
endif()

# new macro CK_TILE_USE_WMMA in order to separately compile examples for MFMA/WMMA
set(CK_TILE_USE_WMMA 0)

if (SUPPORTED_GPU_TARGETS MATCHES "gfx10")
    add_definitions(-DCK_GFX1030_SUPPORT)
endif()

if ((SUPPORTED_GPU_TARGETS MATCHES "gfx11" OR SUPPORTED_GPU_TARGETS MATCHES "gfx12") AND NOT FORCE_DISABLE_WMMA)
    message(STATUS "Enabling WMMA instances")
    add_definitions(-DCK_USE_WMMA)
    set(CK_USE_WMMA "ON")
    set(CK_TILE_USE_WMMA 1)
endif()

# define the macro with the current value (0 or 1)
add_definitions(-DCK_TILE_USE_WMMA=${CK_TILE_USE_WMMA})

if (SUPPORTED_GPU_TARGETS MATCHES "gfx12" AND NOT FORCE_DISABLE_WMMA)
    message(STATUS "Enabling WMMA FP8 gemms on native architectures")
    add_definitions(-DCK_USE_WMMA_FP8)
    set(CK_USE_WMMA_FP8 "ON")
endif()
if (SUPPORTED_GPU_TARGETS MATCHES "gfx12" OR SUPPORTED_GPU_TARGETS MATCHES "gfx950")
    add_definitions(-DCK_USE_OCP_FP8)
    set(CK_USE_OCP_FP8 "ON")
endif()
if (SUPPORTED_GPU_TARGETS MATCHES "gfx90a" OR SUPPORTED_GPU_TARGETS MATCHES "gfx94")
    add_definitions(-DCK_USE_FNUZ_FP8)
    set(CK_USE_FNUZ_FP8 "ON")
endif()
if (SUPPORTED_GPU_TARGETS MATCHES "gfx950")
    add_definitions(-DCK_USE_NATIVE_MX_SUPPORT)
    set(CK_USE_NATIVE_MX_SUPPORT "ON")
    add_definitions(-DCK_GFX950_SUPPORT)
    set(CK_GFX950_SUPPORT "ON")
endif()

if ((SUPPORTED_GPU_TARGETS MATCHES "gfx942" OR SUPPORTED_GPU_TARGETS MATCHES "gfx95") AND CK_ENABLE_TF32)
    add_definitions(-DCK_ENABLE_TF32)
    set(CK_ENABLE_TF32 "ON")
else()
    message(STATUS "Disabling TF32 instances")
    remove_definitions(-DCK_ENABLE_TF32)
    set(CK_ENABLE_TF32 "OFF")
endif()

option(CK_USE_FP8_ON_UNSUPPORTED_ARCH "Enable FP8 GEMM instances on older architectures" OFF)
if(CK_USE_FP8_ON_UNSUPPORTED_ARCH AND (SUPPORTED_GPU_TARGETS MATCHES "gfx90a" OR SUPPORTED_GPU_TARGETS MATCHES "gfx908"))
    add_definitions(-DCK_USE_FP8_ON_UNSUPPORTED_ARCH)
    set(CK_USE_FP8_ON_UNSUPPORTED_ARCH "ON")
endif()

# CK config file to record supported datatypes, etc.
configure_file(include/ck/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/ck/config.h)

if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 500723302)
  check_cxx_compiler_flag("-fno-offload-uniform-block" HAS_NO_OFFLOAD_UNIFORM_BLOCK)
  if(HAS_NO_OFFLOAD_UNIFORM_BLOCK)
    message(STATUS "Adding the fno-offload-uniform-block compiler flag")
    add_compile_options(-fno-offload-uniform-block)
  endif()
endif()
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 500500000)
  check_cxx_compiler_flag("-mllvm --lsr-drop-solution=1" HAS_LSR_DROP_SOLUTION)
  if(HAS_LSR_DROP_SOLUTION)
    message(STATUS "Adding the lsr-drop-solution=1 compiler flag")
    add_compile_options("SHELL: -mllvm --lsr-drop-solution=1")
  endif()
endif()
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600140090)
  check_cxx_compiler_flag("-mllvm -enable-post-misched=0" HAS_ENABLE_POST_MISCHED)
  if(HAS_ENABLE_POST_MISCHED)
    message(STATUS "Adding the enable-post-misched=0 compiler flag")
    add_compile_options("SHELL: -mllvm -enable-post-misched=0")
  endif()
endif()
set(check-coerce)
check_cxx_compiler_flag(" -mllvm -amdgpu-coerce-illegal-types=1" check-coerce)
if(NOT WIN32 AND check-coerce AND ${hip_VERSION_FLAT} GREATER 600241132)
   message(STATUS "Adding the amdgpu-coerce-illegal-types=1")
   add_compile_options("SHELL: -mllvm -amdgpu-coerce-illegal-types=1")
endif()
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
   message(STATUS "Adding -amdgpu-early-inline-all=true and -amdgpu-function-calls=false")
   add_compile_options("SHELL: -mllvm -amdgpu-early-inline-all=true")
   add_compile_options("SHELL: -mllvm -amdgpu-function-calls=false")
endif()
#
# Seperate linking jobs from compiling
# Too many concurrent linking jobs can break the build
# Copied from LLVM
set(CK_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(CK_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${CK_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
  endif()
elseif(CK_PARALLEL_LINK_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()
# Similar for compiling
set(CK_PARALLEL_COMPILE_JOBS "" CACHE STRING
  "Define the maximum number of concurrent compile jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(CK_PARALLEL_COMPILE_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${CK_PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
  endif()
elseif(CK_PARALLEL_COMPILE_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()


option(USE_BITINT_EXTENSION_INT4 "Whether to enable clang's BitInt extension to provide int4 data type." OFF)
option(ENABLE_ASM_DUMP "Whether to enable assembly dump for kernels." OFF)
option(ENABLE_JSON_DUMP "Whether to enable json dump for examples." OFF)

if(USE_BITINT_EXTENSION_INT4)
    add_compile_definitions(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    add_compile_options(-Wno-bit-int-extension)
    message(STATUS "CK compiled with USE_BITINT_EXTENSION_INT4 set to ${USE_BITINT_EXTENSION_INT4}")
endif()

if(ENABLE_ASM_DUMP)
    add_compile_options(--save-temps) 
    add_compile_options(-Wno-gnu-line-marker)
    message("CK compiled with ENABLE_ASM_DUMP set to ${ENABLE_ASM_DUMP}")
endif()

if (ENABLE_JSON_DUMP)
    add_compile_definitions(CK_ENABLE_JSON_DUMP)
    message("CK compiled with ENABLE_JSON_DUMP set to ${ENABLE_JSON_DUMP}")
endif()

## Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

## C++
set(CMAKE_CXX_STANDARD ${CK_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_macros.html
# _GLIBCXX_ASSERTIONS
# Undefined by default. When defined, enables extra error checking in the form of
# precondition assertions, such as bounds checking in strings and null pointer
# checks when dereferencing smart pointers
option(USE_GLIBCXX_ASSERTIONS "Turn on additional c++ library checks." OFF)
if(USE_GLIBCXX_ASSERTIONS)
  add_compile_options(-Wp,-D_GLIBCXX_ASSERTIONS)
endif()

## HIP
set(CMAKE_HIP_PLATFORM amd)
set(CMAKE_HIP_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_HIP_EXTENSIONS ON)
message(STATUS "CMAKE_HIP_COMPILER: ${CMAKE_HIP_COMPILER}")

## OpenMP
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message(STATUS "OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message(STATUS "OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message(STATUS "OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

## HIP
# Override HIP version in config.h, if necessary.
# The variables set by find_package() can't be overwritten,
# therefore let's use intermediate variables.
set(CK_HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
set(CK_HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
set(CK_HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
if( DEFINED CK_OVERRIDE_HIP_VERSION_MAJOR )
    set(CK_HIP_VERSION_MAJOR "${CK_OVERRIDE_HIP_VERSION_MAJOR}")
    message(STATUS "CK_HIP_VERSION_MAJOR overriden with ${CK_OVERRIDE_HIP_VERSION_MAJOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_MINOR )
    set(CK_HIP_VERSION_MINOR "${CK_OVERRIDE_HIP_VERSION_MINOR}")
    message(STATUS "CK_HIP_VERSION_MINOR overriden with ${CK_OVERRIDE_HIP_VERSION_MINOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_PATCH )
    set(CK_HIP_VERSION_PATCH "${CK_OVERRIDE_HIP_VERSION_PATCH}")
    message(STATUS "CK_HIP_VERSION_PATCH overriden with ${CK_OVERRIDE_HIP_VERSION_PATCH}")
endif()
message(STATUS "Build with HIP ${HIP_VERSION}")
link_libraries(hip::device)
if(CK_hip_VERSION VERSION_GREATER_EQUAL 6.0.23494)
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
else()
    add_compile_definitions(__HIP_PLATFORM_HCC__=1)
endif()

include(EnableCompilerWarnings)
## tidy
set(CK_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
set(CK_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(CK_BACKEND STREQUAL "HIP" OR CK_BACKEND STREQUAL "HIPNOGPU")
set(CK_TIDY_ERRORS ALL)
endif()


if(ENABLE_CLANG_CPP_CHECKS)
    include(ClangTidy)
    enable_clang_tidy(
        CHECKS
            *
            -abseil-*
            -android-cloexec-fopen
            # Yea we shouldn't be using rand()
            -cert-msc30-c
            -bugprone-exception-escape
            -bugprone-macro-parentheses
            -cert-env33-c
            -cert-msc32-c
            -cert-msc50-cpp
            -cert-msc51-cpp
            -cert-dcl37-c
            -cert-dcl51-cpp
            -clang-analyzer-alpha.core.CastToStruct
            -clang-analyzer-optin.performance.Padding
            -clang-diagnostic-deprecated-declarations
            -clang-diagnostic-extern-c-compat
            -clang-diagnostic-unused-command-line-argument
            -cppcoreguidelines-avoid-c-arrays
            -cppcoreguidelines-avoid-magic-numbers
            -cppcoreguidelines-explicit-virtual-functions
            -cppcoreguidelines-init-variables
            -cppcoreguidelines-macro-usage
            -cppcoreguidelines-non-private-member-variables-in-classes
            -cppcoreguidelines-pro-bounds-array-to-pointer-decay
            -cppcoreguidelines-pro-bounds-constant-array-index
            -cppcoreguidelines-pro-bounds-pointer-arithmetic
            -cppcoreguidelines-pro-type-member-init
            -cppcoreguidelines-pro-type-reinterpret-cast
            -cppcoreguidelines-pro-type-union-access
            -cppcoreguidelines-pro-type-vararg
            -cppcoreguidelines-special-member-functions
            -fuchsia-*
            -google-explicit-constructor
            -google-readability-braces-around-statements
            -google-readability-todo
            -google-runtime-int
            -google-runtime-references
            -hicpp-vararg
            -hicpp-braces-around-statements
            -hicpp-explicit-conversions
            -hicpp-named-parameter
            -hicpp-no-array-decay
            # We really shouldn't use bitwise operators with signed integers, but
            # opencl leaves us no choice
            -hicpp-avoid-c-arrays
            -hicpp-signed-bitwise
            -hicpp-special-member-functions
            -hicpp-uppercase-literal-suffix
            -hicpp-use-auto
            -hicpp-use-equals-default
            -hicpp-use-override
            -llvm-header-guard
            -llvm-include-order
            #-llvmlibc-*
            -llvmlibc-restrict-system-libc-headers
            -llvmlibc-callee-namespace
            -llvmlibc-implementation-in-namespace
            -llvm-else-after-return
            -llvm-qualified-auto
            -misc-misplaced-const
            -misc-non-private-member-variables-in-classes
            -misc-no-recursion
            -modernize-avoid-bind
            -modernize-avoid-c-arrays
            -modernize-pass-by-value
            -modernize-use-auto
            -modernize-use-default-member-init
            -modernize-use-equals-default
            -modernize-use-trailing-return-type
            -modernize-use-transparent-functors
            -performance-unnecessary-value-param
            -readability-braces-around-statements
            -readability-else-after-return
            # we are not ready to use it, but very useful
            -readability-function-cognitive-complexity
            -readability-isolate-declaration
            -readability-magic-numbers
            -readability-named-parameter
            -readability-uppercase-literal-suffix
            -readability-convert-member-functions-to-static
            -readability-qualified-auto
            -readability-redundant-string-init
            # too many narrowing conversions in our code
            -bugprone-narrowing-conversions
            -cppcoreguidelines-narrowing-conversions
            -altera-struct-pack-align
            -cppcoreguidelines-prefer-member-initializer
            ${CK_TIDY_CHECKS}
            ${CK_TIDY_ERRORS}
        HEADER_FILTER
            "\.hpp$"
        EXTRA_ARGS
            -DCK_USE_CLANG_TIDY
    )

    include(CppCheck)
    enable_cppcheck(
        CHECKS
            warning
            style
            performance
            portability
        SUPPRESS
            ConfigurationNotChecked
            constStatement
            duplicateCondition
            noExplicitConstructor
            passedByValue
            preprocessorErrorDirective
            shadowVariable
            unusedFunction
            unusedPrivateFunction
            unusedStructMember
            unmatchedSuppression
        FORCE
        SOURCES
            library/src
        INCLUDE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/library/include
        DEFINE
            CPPCHECK=1
            __linux__=1
    )
else()
    function(clang_tidy_check TARGET)
        # stub out empty function if clang tidy is not enabled
    endfunction()
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# set CK project include directories
include_directories(BEFORE
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
    ${HIP_INCLUDE_DIRS}
)

SET(BUILD_DEV ON CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    add_compile_options(-Werror)
    add_compile_options(-Weverything)
endif()
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_compile_options(-fcolor-diagnostics)
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    add_compile_options(-fdiagnostics-color=always)
endif()

if(NOT MIOPEN_REQ_LIBS_ONLY)
    # make check runs the entire set of examples and tests
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} USES_TERMINAL)
    # make smoke runs the tests and examples that runs within 30 seconds on gfx90a
    add_custom_target(smoke COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "SMOKE_TEST" USES_TERMINAL)
    # make regression runs the tests and examples that runs for more 30 seconds on gfx90a
    add_custom_target(regression COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "REGRESSION_TEST" USES_TERMINAL)
endif()



file(GLOB_RECURSE INSTANCE_FILES "${PROJECT_SOURCE_DIR}/*/device_*_instance.cpp")
file(GLOB dir_list RELATIVE ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/*)
set(CK_DEVICE_INSTANCES)
FOREACH(subdir_path ${dir_list})
set(target_dir)
IF(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/${subdir_path}")
    set(cmake_instance)
    file(READ "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/${subdir_path}/CMakeLists.txt" cmake_instance)
    set(add_inst 0)
    if(("${cmake_instance}" MATCHES "fp8" OR "${cmake_instance}" MATCHES "_f8") AND DTYPES MATCHES "fp8")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "bf8" OR "${cmake_instance}" MATCHES "_b8") AND DTYPES MATCHES "bf8")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "fp16" OR "${cmake_instance}" MATCHES "_f16") AND DTYPES MATCHES "fp16")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "fp32" OR "${cmake_instance}" MATCHES "_f32") AND DTYPES MATCHES "fp32")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "tf32" OR "${cmake_instance}" MATCHES "_tf32") AND DTYPES MATCHES "tf32")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "fp64" OR "${cmake_instance}" MATCHES "_f64") AND DTYPES MATCHES "fp64")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "bf16" OR "${cmake_instance}" MATCHES "_b16") AND DTYPES MATCHES "bf16")
        set(add_inst 1)
    endif()
    if(("${cmake_instance}" MATCHES "int8" OR "${cmake_instance}" MATCHES "_i8") AND DTYPES MATCHES "int8")
        set(add_inst 1)
    endif()
    if(NOT "${cmake_instance}" MATCHES "DTYPES")
        set(add_inst 1)
    endif()
    if(add_inst EQUAL 1 OR NOT DEFINED DTYPES)
        list(APPEND CK_DEVICE_INSTANCES device_${subdir_path}_instance)
    endif()
ENDIF()
ENDFOREACH()

add_custom_target(instances DEPENDS utility;${CK_DEVICE_INSTANCES}  SOURCES ${INSTANCE_FILES})

option(MIOPEN_REQ_LIBS_ONLY "Build only the MIOpen required libraries" OFF)
option(DISABLE_OFFLOAD_COMPRESS "Disable offload compress compiler flag when building instances" OFF)
option(BUILD_MHA_LIB "Build the static library for flash attention" OFF)

add_subdirectory(library)

if(NOT GPU_ARCHS AND USER_GPU_TARGETS AND NOT MIOPEN_REQ_LIBS_ONLY)
   rocm_package_setup_component(tests
        LIBRARY_NAME composablekernel
        PACKAGE_NAME tests # Prevent -static suffix on package name
   )

   rocm_package_setup_component(examples
        LIBRARY_NAME composablekernel
        PACKAGE_NAME examples
   )
   add_subdirectory(example)

   add_subdirectory(tutorial)
   rocm_package_setup_component(tutorials
        LIBRARY_NAME composablekernel
        PACKAGE_NAME tutorials
   )
   add_subdirectory(tile_engine)
   if(BUILD_TESTING)
       add_subdirectory(test)
   endif()
endif()

if (NOT MIOPEN_REQ_LIBS_ONLY)
    rocm_package_setup_component(profiler
        LIBRARY_NAME composablekernel
        PACKAGE_NAME ckprofiler
    )
    add_subdirectory(profiler)
endif()

if (CK_EXPERIMENTAL_BUILDER)
  add_subdirectory(experimental/builder)
endif()

if(CK_USE_CODEGEN AND (SUPPORTED_GPU_TARGETS MATCHES "gfx9" OR GPU_ARCHS))
  add_subdirectory(codegen)
endif()

#Create an interface target for the include only files and call it "composablekernels"
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

# Install CK version and configuration files
rocm_install(FILES
    ${PROJECT_BINARY_DIR}/include/ck/version.h
    ${PROJECT_BINARY_DIR}/include/ck/config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck/
)

if(CK_EXPERIMENTAL_BUILDER)
    rocm_install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/experimental/builder/include/ck_tile/builder
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck_tile
    )

    set(CK_TILE_SRC_FOLDER ${CMAKE_SOURCE_DIR}/include/ck_tile/)
    rocm_install(DIRECTORY ${CK_TILE_SRC_FOLDER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck_tile)
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
    NAME composablekernel
    DESCRIPTION "High Performance Composable Kernel for AMD GPUs"
    MAINTAINER "MIOpen Kernels Dev Team <dl.MIOpen@amd.com>"
    LDCONFIG
    HEADER_ONLY
)
