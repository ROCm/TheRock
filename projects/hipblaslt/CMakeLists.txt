# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Generate compile commands json file." ON)

project(hipblaslt LANGUAGES CXX C ASM)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    set(HIPBLASLT_IS_SUBPROJECT FALSE)
else()
    set(HIPBLASLT_IS_SUBPROJECT TRUE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)

include(dependencies)
include(hipblaslt_python)
include(hipblaslt_target_configure_sanitizers)
include(tensilelite_supported_architectures)

set(HIPBLASLT_PROJECT_VERSION "1.2.0" CACHE STRING "Semantic version string.")

set(PROJECT_VERSION "${HIPBLASLT_PROJECT_VERSION}")
string(REPLACE "." ";" VERSION_LIST ${PROJECT_VERSION})
list(GET VERSION_LIST 0 PROJECT_VERSION_MAJOR)
list(GET VERSION_LIST 1 PROJECT_VERSION_MINOR)
list(GET VERSION_LIST 2 PROJECT_VERSION_PATCH)
rocm_get_git_commit_tag(PROJECT_VERSION_TWEAK)

set(hipblaslt_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(hipblaslt_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(hipblaslt_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(hipblaslt_VERSION_TWEAK "${PROJECT_VERSION_TWEAK}")

set(hipblaslt_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
      set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
      set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

option(HIPBLASLT_ENABLE_DEVICE "Build hipBLASLt device libraries." ON)
option(HIPBLASLT_ENABLE_CLIENT "Build hipBLASLt client apps." ON)
option(HIPBLASLT_ENABLE_HOST "Build hipBLASLt host library." ON)
option(HIPBLASLT_ENABLE_FETCH "Fetch dependencies using fetch content" OFF)
option(HIPBLASLT_BUNDLE_PYTHON_DEPS "Build python dependencies requied for device code generation." ON)
option(HIPBLASLT_ENABLE_COVERAGE "Build coverage support." OFF)
option(HIPBLASLT_ENABLE_THEROCK "Build hipBLASLt for TheRock." OFF)

option(TENSILELITE_ENABLE_HOST "Build the tensilelite host library." ON)
option(TENSILELITE_BUILD_TESTING "Build client when building tests" OFF)
option(TENSILELITE_ENABLE_CLIENT "Build tensilelite client" ${TENSILELITE_BUILD_TESTING})
option(TENSILELITE_ENABLE_AUTOBUILD "Generate scripts for wrapping tensile python scripts" OFF)

if(HIPBLASLT_ENABLE_CLIENT)
    option(HIPBLASLT_BUILD_TESTING "Build hipblaslt client tests." ON)
    option(HIPBLASLT_ENABLE_SAMPLES "Build client samples." ON)
    # rocm-smi is not presently available on Windows so we do not require it.
    cmake_dependent_option(HIPBLASLT_ENABLE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
    option(HIPBLASLT_ENABLE_BLIS "Enable BLIS support." ON)
endif()

if(HIPBLASLT_ENABLE_HOST OR TENSILELITE_ENABLE_HOST)
    option(HIPBLASLT_ENABLE_ASAN "Build hipBLASLt with address sanitizer enabled." OFF)
    option(HIPBLASLT_ENABLE_LAZY_LOAD "Enable lazy loading of runtime code oject files to reduce ram usage." ON)
    option(HIPBLASLT_ENABLE_MSGPACK "Use msgpack for parsing configuration files." ON)
    option(HIPBLASLT_ENABLE_OPENMP "Use OpenMP to improve performance." ON)
    option(HIPBLASLT_ENABLE_LLVM "Use msgpack for parsing configuration files." OFF)
endif()

if(HIPBLASLT_ENABLE_HOST)
    option(HIPBLASLT_BUILD_SHARED_LIBS "Build the hipblaslt library as shared vs static" ON)
    option(HIPBLASLT_ENABLE_ROCROLLER "Use RocRoller library." ON)
    cmake_dependent_option(HIPBLASLT_ENABLE_MARKER "Use the marker library." ON "NOT WIN32" OFF)
    option(HIPBLASLT_ENABLE_HIPBLAS_DIRECT "Use the hipblas header directly." OFF)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE CACHE BOOL "Skip build RPATH")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE CACHE BOOL "Build with install RPATH")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../llvm/lib" CACHE STRING "Install RPATH")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE CACHE BOOL "Use link path for RPATH")

mark_as_advanced(CMAKE_SKIP_BUILD_RPATH)
mark_as_advanced(CMAKE_BUILD_WITH_INSTALL_RPATH)
mark_as_advanced(CMAKE_INSTALL_RPATH)
mark_as_advanced(CMAKE_INSTALL_RPATH_USE_LINK_PATH)

set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
    tensilelite_get_base_architectures(GPU_TARGETS)
else()
    tensilelite_validate_gpu_targets("${GPU_TARGETS}")
endif()
message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

find_package(hip REQUIRED)
if(HIPBLASLT_ENABLE_HOST)
    if(HIPBLASLT_ENABLE_HIPBLAS_DIRECT)
        set(hipblas_target hipblas)
    else()
        set(hipblas_target hipblas-common)
    endif()
    if(NOT ROCM_LIBS_SUPERBUILD)
        find_package(${hipblas_target} REQUIRED)
    endif()
    if(HIPBLASLT_ENABLE_MARKER)
        find_library(rocTracer roctx64 REQUIRED)
        get_filename_component(ROCTRACER_LIB_DIR ${rocTracer} DIRECTORY)
        get_filename_component(ROCTRACER_ROOT_DIR ${ROCTRACER_LIB_DIR} DIRECTORY)
        find_path(ROCTRACER_INCLUDE_DIR "roctracer/roctx.h"
            HINTS ${ROCTRACER_ROOT_DIR}
            PATH_SUFFIXES include
            REQUIRED)
    else()
        set(rocTracer)
    endif()
endif()

if(HIPBLASLT_ENABLE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()

if(HIPBLASLT_ENABLE_CLIENT)
    if (NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
        set(CMAKE_Fortran_COMPILER  "gfortran")
    endif()
    enable_language(Fortran)

    if(HIPBLASLT_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
    endif()

    # There is an implicit find_package(BLAS) when finding lapack
    find_package(LAPACK REQUIRED)
endif()

if(HIPBLASLT_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(HIPBLASLT_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    if(NOT WIN32 AND (HIPBLASLT_ENABLE_ASAN OR THEROCK_SANITIZER STREQUAL "ASAN"))
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
            OUTPUT_VARIABLE ASAN_LIB_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
        # If ASAN_LIB_PATH is libclang_rt.asan-x86_64.so
        # rather than /path/to/libclang_rt.asan-x86_64.so then
        # we failed to locate it and HAS_PARENT_PATH is false.
        cmake_path(HAS_PARENT_PATH ASAN_LIB_PATH result)
        if(NOT result)
            message(FATAL_ERROR "Failed to locate libclang_rt.asan-x86_64.so ")
        endif()
        # Disable a few asan options to get builds going but these should be addressed
        set(asan_opts "LD_PRELOAD=${ASAN_LIB_PATH}" "ASAN_OPTIONS=detect_leaks=0,new_delete_type_mismatch=0")
    endif()
endif()

if(HIPBLASLT_BUNDLE_PYTHON_DEPS)
    set(HIPBLASLT_PYTHON_DEPS "rocisa")
    set(hipblaslt_python_dev Development.Module)
    hipblaslt_find_python("${hipblaslt_python_dev}")
    hipblaslt_configure_bundled_python_command("${CMAKE_CURRENT_BINARY_DIR}/tensilelite/rocisa/lib" "${asan_opts}")
else()
    hipblaslt_find_python("${hipblaslt_python_dev}")
    set(HIPBLASLT_PYTHON_COMMAND "${Python_EXECUTABLE}")
endif()

if(HIPBLASLT_ENABLE_ROCROLLER)
    if(NOT ROCM_LIBS_SUPERBUILD)
        if(HIPBLASLT_ENABLE_THEROCK)
            find_package(rocroller REQUIRED)
        else()
            block(SCOPE_FOR VARIABLES)
                set(ROCROLLER_ENABLE_FETCH ON)
                set(ROCROLLER_BUILD_TESTING OFF)
                set(ROCROLLER_ENABLE_CLIENT OFF)
                option(YAML_CPP_INSTALL "" ON)
                set(BUILD_TESTING OFF)
                set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--exclude-libs,ALL")
                set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
                add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../shared/rocroller" "${CMAKE_CURRENT_BINARY_DIR}/rocroller")
            endblock()
        endif()
    endif()
endif()

if(HIPBLASLT_ENABLE_MSGPACK)
    # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
    # Prefer 6.x (msgpack-cxx)
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
        message(STATUS "Found msgpack-cxx (>=6.x)")
        set(hipblaslt_msgpack_target "msgpack-cxx")
    else()
        # TODO: set minimum version requied for *-cxx library
        find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
        set(hipblaslt_msgpack_target "msgpackc-cxx")
        message(STATUS "Found msgpack (<=5.x)")
    endif()
endif()

if(NOT ROCM_LIBS_SUPERBUILD)
    # TODO: This is a temporary solution to prevent the installation of the origami license files.
    block(SCOPE_FOR VARIABLES)
        set(ORIGAMI_ENABLE_INSTALL OFF)
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../shared/origami" origami)
    endblock()
endif()

add_subdirectory(tensilelite)

if(HIPBLASLT_ENABLE_HOST)
    if(HIPBLASLT_BUILD_SHARED_LIBS)
        add_library(hipblaslt SHARED)
    else()
        add_library(hipblaslt STATIC)
        target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_STATIC_LIB)
    endif()
    if(HIPBLASLT_ENABLE_ROCROLLER)
        target_compile_definitions(hipblaslt PUBLIC HIPBLASLT_USE_ROCROLLER)
    endif()
    if(HIPBLASLT_ENABLE_MSGPACK)
        target_link_libraries(hipblaslt PRIVATE ${hipblaslt_msgpack_target})
    endif()
    rocm_set_soversion(hipblaslt "${hipblaslt_SOVERSION}")
    add_library(roc::hipblaslt ALIAS hipblaslt)

    target_link_libraries(hipblaslt
        PUBLIC
            roc::${hipblas_target}
        PRIVATE
            hip::device
            tensilelite::tensilelite-host
            ${CMAKE_DL_LIBS}
            ${rocTracer}
    )

    if(HIPBLASLT_ENABLE_ASAN)
        hipblaslt_target_configure_sanitizers(hipblaslt PRIVATE)
    endif()

    target_compile_features(hipblaslt PUBLIC cxx_std_17)
    target_compile_definitions(hipblaslt
        PUBLIC
            $<BUILD_INTERFACE:ROCM_USE_FLOAT16>
            $<BUILD_INTERFACE:__HIP_HCC_COMPAT_MODE__=1> # if we don't have this we fail tests on 942_304cu
    )

    if(HIPBLASLT_ENABLE_HIPBLAS_DIRECT)
        # This definition shows up in a public header
        target_compile_definitions(hipblaslt PUBLIC LEGACY_HIPBLAS_DIRECT)
    endif()

    target_compile_definitions(hipblaslt PRIVATE
        ROCBLASLT_TENSILE_LAZY_LOAD=$<BOOL:${HIPBLASLT_ENABLE_LAZY_LOAD}>
    )

    if(HIPBLASLT_ENABLE_MARKER)
        target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_MARKER)
        target_include_directories(hipblaslt PRIVATE ${ROCTRACER_INCLUDE_DIR})
    endif()

    target_include_directories(hipblaslt
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/library/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/library/src/amd_detail/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include"
    )

    set_target_properties(hipblaslt
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/library"
    )
    add_subdirectory(library)
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    add_subdirectory(device-library)
endif()

if(HIPBLASLT_ENABLE_CLIENT)
    add_subdirectory(clients)
endif()

if(HIPBLASLT_ENABLE_HOST)
    # work around code object stripping failure if using /usr/bin/strip
    find_program(LLVM_STRIP
        NAMES llvm-strip
        PATH_SUFFIXES lib/llvm/bin llvm/bin bin
    )
    if(LLVM_STRIP)
        set(CPACK_RPM_SPEC_MORE_DEFINE "%define __strip ${LLVM_STRIP}")
    else()
        message(WARNING "llvm-strip not found, using default strip tool. This may cause stripping to fail during packaging.")
    endif()

    get_target_property(hipblaslt_public_headers roc::hipblaslt INTERFACE_SOURCES)

    install(
        FILES ${hipblaslt_public_headers}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt"
        COMPONENT devel
    )
    # temporary workaround to avoid updating downstream components
    # TODO: remove the follow file install once rocblas is updated
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/library/include/hipblaslt/hipblaslt-export.h"
            "${CMAKE_CURRENT_BINARY_DIR}/library/include/hipblaslt/hipblaslt-version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
    )

    rocm_install_targets(TARGETS hipblaslt)

    rocm_export_targets(
        TARGETS roc::hipblaslt
        DEPENDS PACKAGE hip
        DEPENDS PACKAGE ${hipblas_target}
        NAMESPACE roc::
    )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipblaslt-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config.cmake"
        COPYONLY
    )

    if( LEGACY_HIPBLAS_DIRECT )
        rocm_package_add_dependencies(DEPENDS "hipblas >= 0.50.0")
    else()
        set(hipblas_common_minimum 1.0.0)
        if(HIPBLASLT_BUILD_SHARED_LIBS)
            rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-dev >= ${hipblas_common_minimum}")
            rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-devel >= ${hipblas_common_minimum}")
        else()
            rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-dev >= ${hipblas_common_minimum}")
            rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-devel >= ${hipblas_common_minimum}")
        endif()
    endif()
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    if(WIN32)
        # Put files used at runtime adjacent to '.dll's and '.exe's in bin/.
        set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_BINDIR}/hipblaslt" CACHE PATH "path to tensile library")
    else()
        # Put files used at runtime adjacent to shared libraries ('.so') in
        # lib/, separate from executables (no extension) in bin/.
        set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipblaslt" CACHE PATH "path to tensile library")
    endif()
    rocm_install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Tensile/library"
        DESTINATION ${HIPBLASLT_TENSILE_LIBRARY_DIR}
        COMPONENT runtime
    )
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

include(ROCMCreatePackage)
include(ROCMClients)

if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
endif()

if(HIPBLASLT_ENABLE_CLIENT)
    rocm_install(
        TARGETS
            hipblaslt-bench
            hipblaslt-api-overhead
            hipblaslt-bench-groupedgemm-fixed-mk
            hipblaslt-bench-extop-layernorm
            hipblaslt-bench-extop-matrixtransform
            hipblaslt-bench-extop-softmax
            hipblaslt-bench-extop-amax
        COMPONENT benchmarks
    )
    if(HIPBLASLT_ENABLE_LLVM)
        rocm_install(
            TARGETS
                hipblaslt-sequence
            COMPONENT benchmarks
        )
    endif()
    rocm_install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/clients/tests/data/hipblaslt_common.yaml"
            "${CMAKE_CURRENT_SOURCE_DIR}/clients/tests/data/hipblaslt_template.yaml"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients-common
    )
    rocm_install(
        PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/clients/tests/hipblaslt_gentest.py"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients-common
    )
    rocm_install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/clients/bench/sequence.yaml"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT benchmarks
    )

    if(HIPBLASLT_ENABLE_MARKER)
        rocm_package_add_dependencies(DEPENDS "roctracer >= 1.0.0")
    endif()

    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()

    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
    rocm_package_setup_client_component(benchmarks
        DEPENDS
        COMPONENT clients-common
    )

endif()

if(HIPBLASLT_ENABLE_CLIENT AND (HIPBLASLT_BUILD_TESTING OR BUILD_TESTING))
    rocm_install(
        TARGETS hipblaslt-test
        COMPONENT tests
    )
    rocm_install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/clients/hipblaslt_gtest.data"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT tests
    )
    rocm_package_setup_client_component(tests
        DEPENDS
        COMPONENT clients-common
    )
endif()

if(HIPBLASLT_ENABLE_CLIENT AND HIPBLASLT_ENABLE_SAMPLES)
    install(
        TARGETS
            sample_hipblaslt_gemm
            sample_hipblaslt_gemm_ext
            sample_hipblaslt_gemm_batched
            sample_hipblaslt_gemm_batched_ext
            sample_hipblaslt_gemm_tuning_splitk_ext
            sample_hipblaslt_gemm_bias
            sample_hipblaslt_gemm_bias_ext
            sample_hipblaslt_gemm_get_all_algos
            sample_hipblaslt_gemm_get_all_algos_ext
            sample_hipblaslt_gemm_get_algo_by_index_ext
            sample_hipblaslt_gemm_alphavec_ext
            sample_hipblaslt_gemm_gelu_aux_bias
            sample_hipblaslt_gemm_gelu_aux_bias_ext
            sample_hipblaslt_gemm_amax
            sample_hipblaslt_gemm_amax_ext
            sample_hipblaslt_gemm_amax_with_scale
            sample_hipblaslt_gemm_amax_with_scale_ext
            sample_hipblaslt_gemm_bgradb
            sample_hipblaslt_gemm_ext_bgradb
            sample_hipblaslt_gemm_dgelu_bgrad
            sample_hipblaslt_gemm_dgelu_bgrad_ext
            sample_hipblaslt_gemm_is_tuned_ext
            sample_hipblaslt_gemm_tuning_wgm_ext
            sample_hipblaslt_gemm_with_scale_a_b
            sample_hipblaslt_gemm_with_scale_a_b_ext
            sample_hipblaslt_groupedgemm_ext
            sample_hipblaslt_groupedgemm_fixed_mk_ext
            sample_hipblaslt_groupedgemm_get_all_algos_ext
            sample_hipblaslt_gemm_mix_precision
            sample_hipblaslt_gemm_mix_precision_ext
            sample_hipblaslt_gemm_mix_precision_with_amax_ext
            sample_hipblaslt_gemm_attr_tciA_tciB
            sample_hipblaslt_ext_op_layernorm
            sample_hipblaslt_ext_op_amax
            sample_hipblaslt_gemm_with_TF32
            sample_hipblaslt_gemm_swizzle_a
            sample_hipblaslt_gemm_bias_swizzle_a_ext
            sample_hipblaslt_weight_swizzle_padding
            sample_hipblaslt_gemm_swish_bias
        RUNTIME
        COMPONENT samples
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/hipblaslt-samples"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/clients/samples"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/hipblaslt/samples
        COMPONENT samples
        FILES_MATCHING
        PATTERN "*.cpp|*.h"
        PATTERN "CMakeLists.txt" EXCLUDE
    )
    rocm_package_setup_client_component(samples)
endif()

if(ROCM_LIBS_SUPERBUILD OR NOT HIPBLASLT_IS_SUBPROJECT)
    # rocm_create_package makes decisions on what type of package to produce
    # based on BUILD_SHARED_LIBS. If we don't set this here we will only get
    # a dev package.
    set(BUILD_SHARED_LIBS ${HIPBLASLT_BUILD_SHARED_LIBS})
    set(HIPBLASLT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")
    rocm_create_package(
        NAME hipblaslt
        DESCRIPTION "HIP library for GEMM operations with extended functionality"
        MAINTAINER "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>"
        LDCONFIG
        LDCONFIG_DIR ${HIPBLASLT_CONFIG_DIR}
    )
endif()

if(HIPBLASLT_ENABLE_COVERAGE)
    target_compile_options(hipblaslt PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt PUBLIC -fprofile-instr-generate)

    target_compile_options(hipblaslt-clients-common PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-clients-common PUBLIC -fprofile-instr-generate)

    set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
    add_custom_target(
        code_cov_tests
        DEPENDS hipblaslt-test
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
        COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_%p.profraw"
                                        GTEST_LISTENER=NO_PASS_LINE_IN_LOG
                                        $<TARGET_FILE:hipblaslt-test>
                                        --precompile=hipblaslt-test-precompile.db
        COMMENT "Running hipblaslt testcases for coverage report"
      )

      find_program(
        LLVM_PROFDATA
        llvm-profdata
        REQUIRED
        HINTS ${ROCM_PATH}/llvm/bin
        PATHS /opt/rocm/llvm/bin
      )
      find_program(
        LLVM_COV
        llvm-cov
        REQUIRED
        HINTS ${ROCM_PATH}/llvm/bin
        PATHS /opt/rocm/llvm/bin
      )
      add_custom_target(
        coverage
        DEPENDS code_cov_tests
        COMMAND ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/hipblaslt-coverage_*.profraw" -o "${coverage_dir}/hipblaslt.profdata"
        COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipblaslt> -instr-profile="${coverage_dir}/hipblaslt.profdata" --ignore-filename-regex=".*Tensile.*|.*origami.*"
        COMMAND ${LLVM_COV} show -object $<TARGET_FILE:hipblaslt> -instr-profile="${coverage_dir}/hipblaslt.profdata" --ignore-filename-regex=".*Tensile.*|.*origami.*" -format=html -output-dir="${coverage_dir}"
        COMMAND ${LLVM_COV} export -object $<TARGET_FILE:hipblaslt> -instr-profile="${coverage_dir}/hipblaslt.profdata" --ignore-filename-regex=".*Tensile.*|.*origami.*" -format=lcov > "${coverage_dir}/coverage.info"
        COMMENT "Generating code coverage report"
      )
endif()
