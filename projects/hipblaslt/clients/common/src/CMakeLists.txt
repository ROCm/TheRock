# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-clients-common
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/efficiency_monitor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_parse_data.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_arguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_random.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_init_device.cpp"
)

if(HIPBLASLT_ENABLE_ROCROLLER)
    target_sources(hipblaslt-clients-common PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/mxDataGen.cpp")
endif()

if(HIPBLASLT_ENABLE_BLIS)
    target_sources(hipblaslt-clients-common
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/blis_interface.cpp"
    )
endif()

if(rocm_smi_FOUND)
    target_link_libraries(hipblaslt-clients-common PRIVATE rocm_smi64)
endif()
