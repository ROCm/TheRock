# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-clients-common
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/allclose.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/datatype_interface.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/device_vector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/d_vector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/efficiency_monitor.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/flops.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_arguments.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_data.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_datatype2string.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_init.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_math.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_parse_data.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_random.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/host_vector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipBuffer.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mxDataGen.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/near.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/norm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/singletons.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDataManipulation.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_cleanup.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/testing_auxiliary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/testing_matmul.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/unit.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.hpp"
)
