# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# This file creates targets of the form ExtOp<ID>_<arch> where ID is:
#   - Obj: targets that map *.s into *.o
#   - Library: target associated creation of .dat

set(ops_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite")
set(output_dir "${CMAKE_CURRENT_BINARY_DIR}")
string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${GPU_TARGETS}")
list(REMOVE_DUPLICATES archs)
message(STATUS "Architectures to process: ${archs}")

# Serializing extop-library targets: Generating hipblasltExtOpLibrary.dat
# per architecture is fast compared to other build tasks and not on the
# critical path. Concurrent access to intermediate files is not well
# supported on certain file systems or operating systems. Therefore, we
# serialize by adding dependencies between each target and the previous one.
set(previous_target "")

foreach(arch IN LISTS archs)
    #  Start of arch configuration
    message(STATUS "hipblaslt device library extops: Configuring target for architecture ${arch} for hipblasltExtOpLibrary")
    set(WAVEFRONT "-mwavefrontsize64")
    if(arch MATCHES "^gfx1[12][0-9][0-9]$")
        set(WAVEFRONT "-mno-wavefrontsize64")
    endif()
    add_library(extop-obj-${arch} OBJECT)
    target_compile_options(extop-obj-${arch}
        PRIVATE
            -Wno-unused-command-line-argument -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=${arch} ${WAVEFRONT} -c
    )
    target_sources(extop-obj-${arch}
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s"
    )
    add_custom_command(
        OUTPUT
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s"
            "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s"
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/LayerNormGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_1_${arch}.s" -w 256 -c 4 --sweep-once 1 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/LayerNormGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/L_256_4_0_${arch}.s" -w 256 -c 4 --sweep-once 0 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/SoftmaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/S_8_32_${arch}.s" -m 8 -n 32 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/SoftmaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/S_16_16_${arch}.s" -m 16 -n 16 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/SoftmaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/S_4_64_${arch}.s" -m 4 -n 64 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/SoftmaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/S_2_128_${arch}.s" -m 2 -n 128 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/SoftmaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/S_1_256_${arch}.s" -m 1 -n 256 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/AMaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_S_256_4_${arch}.s" -t S -d S -w 256 -c 4 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/AMaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/A_H_H_256_4_${arch}.s" -t H -d H -w 256 -c 4 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/AMaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/A_H_S_256_4_${arch}.s" -t H -d S -w 256 -c 4 --arch ${arch}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/AMaxGenerator.py" -o "${CMAKE_CURRENT_BINARY_DIR}/A_S_H_256_4_${arch}.s" -t S -d H -w 256 -c 4 --arch ${arch}
        COMMENT "Creating Layer Norm, Softmax and Amax Assembly for ${arch}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS}
    )

    set(output_code_object_file "${CMAKE_CURRENT_BINARY_DIR}/extop_${arch}.co")
    add_custom_command(
        DEPENDS extop-obj-${arch}
        OUTPUT "${output_code_object_file}"
        COMMAND ${CMAKE_CXX_COMPILER};-target;amdgcn-amdhsa;-Xlinker;$<TARGET_OBJECTS:extop-obj-${arch}>;-o;${output_code_object_file}
        COMMAND ${CMAKE_COMMAND} -E copy "${output_code_object_file}" "${HIPBLASLT_TENSILE_LIBPATH}/library"
        COMMENT "Creating extop_${arch}"
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(extop-library-${arch} ALL
        DEPENDS
            "${output_code_object_file}"
            ${HIPBLASLT_PYTHON_DEPS}
            ${previous_target}
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${ops_dir}/ExtOpCreateLibrary.py" --src=${CMAKE_CURRENT_BINARY_DIR} --co=${output_code_object_file} --output=${output_dir} --arch=${arch}
        COMMAND ${CMAKE_COMMAND} -E copy "${output_dir}/hipblasltExtOpLibrary.dat" "${HIPBLASLT_TENSILE_LIBPATH}/library"
        COMMENT "Creating hipblasltExtOpLibrary.dat for ${arch}"
    )
    # Update previous target for next iteration
    set(previous_target "extop-library-${arch}")
endforeach()
