# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Keep in sync with example/ck_tile/01_fmha/CMakeLists.txt
if(NOT SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx12")
    return()
endif()

set(FMHA_BWD_INSTANCES "tile_fmha_bwd_instances")
set(FMHA_FWD_INSTANCES "tile_fmha_fwd_instances")

set(TEST_NAME "test_ck_tile_fmha")

function(add_gtest_fwd test_group)
    set(V_TYPES "fp16" "bf16" "fp8bf16" "fp32")
    set(CPP_TYPE_fp16 "FmhaFwdFp16")
    set(CPP_TYPE_bf16 "FmhaFwdBf16")
    set(CPP_TYPE_fp8bf16 "FmhaFwdFp8Bf16")
    set(CPP_TYPE_fp32 "FmhaFwdFp32")

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        add_gtest_executable(${name} test_fmha_fwd.cpp)
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group}")
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        target_link_libraries(${name} PRIVATE ${FMHA_FWD_INSTANCES})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "FMHA FWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()

function(add_gtest_bwd test_group)
    set(V_TYPES "fp16" "bf16" "fp32")
    set(CPP_TYPE_fp16 "FmhaBwdFp16")
    set(CPP_TYPE_bf16 "FmhaBwdBf16")
    set(CPP_TYPE_fp32 "FmhaBwdFp32")

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        add_gtest_executable(${name} test_fmha_bwd.cpp)
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group}")
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        target_link_libraries(${name} PRIVATE ${FMHA_BWD_INSTANCES})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "FMHA BWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()


add_gtest_fwd(${TEST_NAME}_fwd)
add_gtest_bwd(${TEST_NAME}_bwd)
add_custom_target(${TEST_NAME} DEPENDS ${TEST_NAME}_fwd ${TEST_NAME}_bwd)
