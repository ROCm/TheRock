# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(mx_flatmm_instance_generate FILE_LIST)
    set(C_DATA_TYPE FP16)
    set(A_LAYOUT ROW)
    set(B_LAYOUT COL)
    set(C_LAYOUT ROW)
    set(FLATMM_CONFIG_FP4xFP4 "MXfp4_FlatmmConfig16")
    set(FLATMM_CONFIG_FP8xFP8 "MXfp8_FlatmmConfig16")
    set(FLATMM_CONFIG_FP8xFP4 "MXf8f4_FlatmmConfig16")
    set(FLATMM_CONFIG_FP4xFP8 "MXf4f8_FlatmmConfig16")

    # foreach(PERSISTENT false true)
    # TODO: Persistent kernels are disabled due to compilation failures with some LLVM versions.  
    foreach(PERSISTENT false)
        foreach(DATA_TYPE FP4xFP4 FP8xFP8 FP8xFP4 FP4xFP8)
            set(FLATMM_CONFIG ${FLATMM_CONFIG_${DATA_TYPE}})
            string(REPLACE "x" ";" DATA_TYPE_AB ${DATA_TYPE})
            list(GET DATA_TYPE_AB 0 A_DATA_TYPE)
            list(GET DATA_TYPE_AB 1 B_DATA_TYPE)

            foreach(SPLIT_K false true)
                foreach(HAS_HOT_LOOP false true)
                    foreach(TAIL_NUMBER ODD EVEN)
                        set(KERNEL_FILE mxgemm/mx_flatmm_instance_${PERSISTENT}_${DATA_TYPE}_${SPLIT_K}_${HAS_HOT_LOOP}_${TAIL_NUMBER}.cpp)
                        string(TOLOWER ${KERNEL_FILE} KERNEL_FILE)
                        configure_file(
                            ${CMAKE_CURRENT_SOURCE_DIR}/mxgemm/mx_flatmm_instance.cpp.in
                            ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE}
                            @ONLY)
                        list(APPEND ${FILE_LIST} ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE})
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    set(${FILE_LIST} ${${FILE_LIST}} PARENT_SCOPE)
endfunction()
