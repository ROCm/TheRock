# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/profiler/include
)

include(gtest)

add_custom_target(tests)

# list of tests that are labelled as REGRESSION_TEST for make regression (runtime more than 30 seconds)
# all other tests are labelled as SMOKE_TEST
set(REGRESSION_TESTS
    test_gemm_standalone_xdl_fp16
    test_gemm_fp16
    test_gemm_splitk
    test_gemm_universal_wmma_fp16
    test_gemm_universal_xdl_fp16
    test_gemm_universal_streamk_fp16
    test_gemm_universal_streamk_bf16
    test_gemm_universal_streamk_fp8
    test_batched_gemm_softmax_gemm_fp16
    test_batched_gemm_softmax_gemm_permute_fp16
    test_batched_gemm_bias_softmax_gemm_permute_fp16
    test_batched_gemm_softmax_gemm_permute_bf16
    test_batched_gemm_bias_softmax_gemm_permute_bf16
    test_grouped_gemm_splitk
    test_batched_gemm_b_scale_wmma
    test_reduce_no_index
    test_reduce_with_index
    test_convnd_fwd
    test_convnd_bwd_data
    test_grouped_convnd_fwd
    test_grouped_convnd_fwd_scaleadd_ab
    test_grouped_convnd_bwd_weight
    test_softmax_rank3
    test_softmax_rank4
    test_batchnorm_fwd_rank_4
    test_batchnorm_bwd_rank_4
    test_grouped_convnd_bwd_data_xdl
    test_grouped_convnd_bwd_data_wmma
    test_grouped_convnd_bwd_data_wmma_large_cases
    test_grouped_conv_bwd_data_wmma_scale
    test_conv_tensor_rearrange
    test_gemm_mx
    test_ck_tile_batched_transpose
    test_ck_tile_fmha_bwd_fp32
    test_ck_tile_fmha_bwd_bf16
    test_ck_tile_fmha_bwd_fp16
    test_ck_tile_fmha_fwd_fp32
    test_ck_tile_fmha_fwd_bf16
    test_ck_tile_fmha_fwd_fp16
    test_ck_tile_fmha_fwd_fp8
    test_ck_tile_streamk_extended
)

function(add_test_executable TEST_NAME)
    message(DEBUG "adding test ${TEST_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
        foreach(source IN LISTS ARGN)
            get_filename_component(source_name ${source} NAME)
            set(test 0)
            if((source_name MATCHES "_fp16|_f16") AND NOT "fp16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp32|_f32") AND NOT "fp32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_tf32|_tf32") AND NOT "tf32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp64|_f64") AND NOT "fp64" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp8|_f8") AND NOT "fp8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf8|_bf8") AND NOT "bf8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf16|_b16") AND NOT "bf16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_int8|_i8") AND NOT "int8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if(test EQUAL 1)
                message(DEBUG "removing test ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endforeach()
    endif()

    set(TEST_TARGETS ${SUPPORTED_GPU_TARGETS})

    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        if(NOT DEFINED DPP_KERNELS AND source_name MATCHES "_dpp")
            message(DEBUG "removing dpp test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT DEFINED DL_KERNELS AND source_name MATCHES "_dl")
            message(DEBUG "removing dl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx9|gfx11|gfx12" AND source_name MATCHES "xdl")
            message(DEBUG "removing xdl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx11|gfx12" AND source_name MATCHES "wmma")
            message(DEBUG "removing wmma test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()
    #only continue if there are some source files left on the list
    set(source_name_list "")
    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        list(APPEND source_name_list ${source_name})
    endforeach()
    if(ARGN)
        if(source_name_list MATCHES "_xdl")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx10-3-generic)
        elseif(source_name_list MATCHES "_wmma")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx950)
        elseif(source_name_list MATCHES "_smfmac")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx908 gfx90a gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic)
        endif()
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
        add_executable(${TEST_NAME} ${ARGN})
        set_property(TARGET ${TEST_NAME} PROPERTY HIP_ARCHITECTURES ${TEST_TARGETS} )
        target_link_libraries(${TEST_NAME} PRIVATE getopt::getopt)
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
        add_dependencies(tests ${TEST_NAME})
        add_dependencies(check ${TEST_NAME})
        rocm_install(TARGETS ${TEST_NAME} COMPONENT tests)
        set(result 0)
    endif()
    message(DEBUG "add_test returns ${result}")
    set(result ${result} PARENT_SCOPE)
    if(result EQUAL 0 AND NOT "${TEST_NAME}" IN_LIST REGRESSION_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "SMOKE_TEST")
        add_dependencies(smoke ${TEST_NAME})
    elseif(result EQUAL 0 AND "${TEST_NAME}" IN_LIST REGRESSION_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "REGRESSION_TEST")
        add_dependencies(regression ${TEST_NAME})
    endif()
endfunction()

function(add_gtest_executable TEST_NAME)
    message(DEBUG "adding gtest ${TEST_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
        foreach(source IN LISTS ARGN)
            get_filename_component(source_name ${source} NAME)
            set(test 0)
            if((source_name MATCHES "_fp16|_f16") AND NOT "fp16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp32|_f32") AND NOT "fp32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_tf32|_tf32") AND NOT "tf32" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp64|_f64") AND NOT "fp64" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_fp8|_f8") AND NOT "fp8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf8|_bf8") AND NOT "bf8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_bf16|_b16") AND NOT "bf16" IN_LIST DTYPES)
                set(test 1)
            endif()
            if((source_name MATCHES "_int8|_i8") AND NOT "int8" IN_LIST DTYPES)
                set(test 1)
            endif()
            if(test EQUAL 1)
                message(DEBUG "removing gtest ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endforeach()
    endif()

    set(TEST_TARGETS ${SUPPORTED_GPU_TARGETS})

    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        if(NOT DEFINED DL_KERNELS AND source_name MATCHES "_dl")
            message(DEBUG "removing dl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx9|gfx11|gfx12" AND source_name MATCHES "xdl")
            message(DEBUG "removing xdl test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx95" AND source_name MATCHES "mx_")
            message(DEBUG "removing microscaling test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        if(NOT TEST_TARGETS MATCHES "gfx11|gfx12" AND source_name MATCHES "wmma")
            message(DEBUG "removing wmma test ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
    endforeach()

    #only continue if there are some source files left on the list
    set(source_name_list "")
    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)
        list(APPEND source_name_list ${source_name})
    endforeach()
    if(ARGN)
        if(source_name_list MATCHES "_xdl")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx10-3-generic)
        elseif(source_name_list MATCHES "_wmma")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx950)
        elseif(source_name_list MATCHES "_smfmac")
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx908 gfx90a gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic)
        elseif(source_name_list MATCHES "_mx") #only build mx example for gfx950
             list(REMOVE_ITEM TEST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic)
        endif()
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
        add_executable(${TEST_NAME} ${ARGN})
        set_property(TARGET ${TEST_NAME} PROPERTY HIP_ARCHITECTURES ${TEST_TARGETS} )
        add_dependencies(tests ${TEST_NAME})
        add_dependencies(check ${TEST_NAME})

        # suppress gtest warnings
        target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
        target_link_libraries(${TEST_NAME} PRIVATE gtest_main getopt::getopt)
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
        rocm_install(TARGETS ${TEST_NAME} COMPONENT tests)
        set(result 0)
    endif()
    message(DEBUG "add_gtest returns ${result}")
    set(result ${result} PARENT_SCOPE)
    if(result EQUAL 0 AND NOT "${TEST_NAME}" IN_LIST REGRESSION_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "SMOKE_TEST")
        add_dependencies(smoke ${TEST_NAME})
    elseif(result EQUAL 0 AND "${TEST_NAME}" IN_LIST REGRESSION_TESTS)
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "REGRESSION_TEST")
        add_dependencies(regression ${TEST_NAME})
    endif()
endfunction()

add_compile_options(-Wno-c++20-extensions)
add_subdirectory(ck_tile)
add_subdirectory(magic_number_division)
add_subdirectory(space_filling_curve)
add_subdirectory(conv_util)
add_subdirectory(reference_conv_fwd)
add_subdirectory(gemm)
add_subdirectory(gemm_add)
add_subdirectory(gemm_bias_add_reduce)
add_subdirectory(gemm_blockscale_wp)
add_subdirectory(gemm_layernorm)
add_subdirectory(gemm_multi_abd)
add_subdirectory(gemm_multiply_multiply_wp)
add_subdirectory(gemm_split_k)
add_subdirectory(gemm_universal)
add_subdirectory(gemm_universal_preshuffle)
add_subdirectory(gemm_ab_scale)
add_subdirectory(gemm_b_scale)
add_subdirectory(gemm_universal_streamk)
add_subdirectory(gemm_reduce)
add_subdirectory(gemm_universal_reduce)
add_subdirectory(batched_gemm)
add_subdirectory(batched_gemm_reduce)
add_subdirectory(batched_gemm_gemm)
add_subdirectory(batched_gemm_softmax_gemm)
add_subdirectory(batched_gemm_softmax_gemm_permute)
add_subdirectory(batched_gemm_b_scale)
add_subdirectory(grouped_gemm)
add_subdirectory(reduce)
add_subdirectory(convnd_fwd)
add_subdirectory(convnd_bwd_data)
add_subdirectory(grouped_convnd_fwd)
add_subdirectory(grouped_convnd_fwd_activation)
add_subdirectory(grouped_convnd_bwd_weight)
add_subdirectory(block_to_ctile_map)
add_subdirectory(softmax)
add_subdirectory(normalization_fwd)
add_subdirectory(normalization_bwd_data)
add_subdirectory(normalization_bwd_gamma_beta)
add_subdirectory(data_type)
add_subdirectory(elementwise_normalization)
add_subdirectory(batchnorm)
add_subdirectory(contraction)
add_subdirectory(pool)
add_subdirectory(batched_gemm_multi_d)
add_subdirectory(grouped_convnd_bwd_data)
add_subdirectory(conv_tensor_rearrange)
add_subdirectory(transpose)
add_subdirectory(permute_scale)
add_subdirectory(wrapper)
add_subdirectory(quantization)
add_subdirectory(device_memory)
if(SUPPORTED_GPU_TARGETS MATCHES "gfx11")
    add_subdirectory(wmma_op)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx942" OR SUPPORTED_GPU_TARGETS MATCHES "gfx950") # smfmac needs ROCm6.2
    add_subdirectory(smfmac_op)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx950") 
    add_subdirectory(mx_mfma_op)
    add_subdirectory(gemm_mx)
endif()
if(SUPPORTED_GPU_TARGETS MATCHES "gfx12")
    add_subdirectory(s_prefetch_op)
endif()
add_subdirectory(position_embedding)
add_subdirectory(scatter_gather)
add_subdirectory(gpu_reference)
add_subdirectory(util)
