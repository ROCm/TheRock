# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipblaslt-bench)
add_library(hipblaslt-clients-common STATIC)

target_include_directories(hipblaslt-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite"
)

add_library(hipblaslt::hipblaslt-clients-common ALIAS hipblaslt-clients-common)

if(HIPBLASLT_ENABLE_BLIS)
    target_link_libraries(hipblaslt-clients-common PUBLIC BLIS::BLIS)
endif()
target_link_libraries(hipblaslt-clients-common
    PUBLIC
        roc::hipblaslt
        OpenMP::OpenMP_CXX
        ${LAPACK_LIBRARIES}
    PRIVATE
        hip::device
)

if(HIPBLASLT_ENABLE_ROCROLLER)
    if(NOT ROCM_LIBS_SUPERBUILD)
        if(HIPBLASLT_ENABLE_THEROCK)
            find_package(mxDataGenerator REQUIRED)
        else()
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../shared/mxdatagenerator" "${CMAKE_CURRENT_BINARY_DIR}/mxdatagenerator")
        endif()
    endif()
    target_compile_definitions(hipblaslt-clients-common PRIVATE HIPBLASLT_USE_ROCROLLER)
    target_link_libraries(hipblaslt-clients-common PRIVATE roc::mxDataGenerator)
    target_compile_features(hipblaslt-clients-common PRIVATE cxx_std_20)
endif()

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(hipblaslt-clients-common PUBLIC)
endif()

target_link_libraries(hipblaslt-bench
    PRIVATE
        hip::device
        hipblaslt::hipblaslt-clients-common
        OpenMP::OpenMP_CXX
)

target_include_directories(hipblaslt-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bench/include")

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(hipblaslt-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(common)
add_subdirectory(bench)

add_executable(hipblaslt-api-overhead "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_api_overhead.cpp")
target_link_libraries(hipblaslt-api-overhead PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-api-overhead PRIVATE ROCBLASLT_INTERNAL_API)
target_include_directories(hipblaslt-api-overhead PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/samples/common")

add_executable(hipblaslt-bench-groupedgemm-fixed-mk "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_groupedgemm_fixed_mk.cpp")
target_link_libraries(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-layernorm "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_layernorm.cpp")
target_link_libraries(hipblaslt-bench-extop-layernorm PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-layernorm PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-matrixtransform "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_matrixtransform.cpp")
target_link_libraries(hipblaslt-bench-extop-matrixtransform PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-matrixtransform PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-softmax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_softmax.cpp")
target_link_libraries(hipblaslt-bench-extop-softmax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-softmax PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-amax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_amax.cpp")
target_link_libraries(hipblaslt-bench-extop-amax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-amax PRIVATE ROCBLASLT_INTERNAL_API)

if(HIPBLASLT_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
    add_executable(hipblaslt-sequence "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_sequence.cpp")
    target_link_libraries(hipblaslt-sequence PRIVATE hipblaslt-clients-common hip::device)
    target_compile_definitions(hipblaslt-sequence PRIVATE ROCBLASLT_INTERNAL_API)
    target_link_libraries(hipblaslt-sequence PRIVATE LLVMObjectYAML)
    target_include_directories(hipblaslt-sequence PRIVATE ${LLVM_INCLUDE_DIRS})
endif()
if(BUILD_TESTING OR HIPBLASLT_BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(hipblaslt-test)
    if(rocm_smi_FOUND)
        target_link_libraries(hipblaslt-test PRIVATE rocm_smi64)
    endif()
    target_link_libraries(hipblaslt-test
        PRIVATE
            hip::device
            hipblaslt::hipblaslt-clients-common
            GTest::gtest
            OpenMP::OpenMP_CXX
            ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(hipblaslt-test PRIVATE GOOGLE_TEST)
    if(HIPBLASLT_ENABLE_ROCROLLER)
        target_compile_definitions(hipblaslt-test PRIVATE HIPBLASLT_USE_ROCROLLER)
    endif()
    add_subdirectory(tests)
endif()

if(HIPBLASLT_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/hipblaslt_gentest.py" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_common.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_template.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/bench/sequence.yaml" "${CMAKE_CURRENT_BINARY_DIR}/sequence.yaml" COPYONLY)
