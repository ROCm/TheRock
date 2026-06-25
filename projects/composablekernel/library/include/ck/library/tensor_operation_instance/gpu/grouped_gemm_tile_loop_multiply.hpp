// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_mnpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

void add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          Multiply>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
                                                            BLayout,
                                                            ck::Tuple<D0Layout>,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            ck::Tuple<D0DataType>,
                                                            EDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            Multiply>>
{
    using DeviceOp = DeviceGroupedGemmTileLoop<ALayout,
                                               BLayout,
                                               ck::Tuple<D0Layout>,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               ck::Tuple<D0DataType>,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               Multiply>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_mnkpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_mnpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_comp_kpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_mnkpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_mnpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v1_kpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_mnkpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_mnpadding_instances(
                    op_ptrs);
                add_device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

void add_device_grouped_gemm_xdl_tile_loop_multiply_fastgelu_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyFastGelu>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
                                                            BLayout,
                                                            ck::Tuple<D0Layout>,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            ck::Tuple<D0DataType>,
                                                            EDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            MultiplyFastGelu>>
{
    using DeviceOp = DeviceGroupedGemmTileLoop<ALayout,
                                               BLayout,
                                               ck::Tuple<D0Layout>,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               ck::Tuple<D0DataType>,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               MultiplyFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_tile_loop_multiply_fastgelu_bf16_i8_bf16_mk_kn_mn_instances(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

void add_device_grouped_gemm_xdl_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
                                                            BLayout,
                                                            ck::Tuple<D0Layout, D1Layout>,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            ck::Tuple<D0DataType, D1DataType>,
                                                            EDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            MultiplyAdd>>
{
    using DeviceOp = DeviceGroupedGemmTileLoop<ALayout,
                                               BLayout,
                                               ck::Tuple<D0Layout, D1Layout>,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               ck::Tuple<D0DataType, D1DataType>,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               MultiplyAdd>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instances(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

void add_device_grouped_gemm_xdl_tile_loop_multiply_bias_fastgelu_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          BF16_BF16_Tuple,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAddFastGelu>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
                                                            BLayout,
                                                            ck::Tuple<D0Layout, D1Layout>,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            ck::Tuple<D0DataType, D1DataType>,
                                                            EDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            MultiplyAddFastGelu>>
{
    using DeviceOp = DeviceGroupedGemmTileLoop<ALayout,
                                               BLayout,
                                               ck::Tuple<D0Layout, D1Layout>,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               ck::Tuple<D0DataType, D1DataType>,
                                               EDataType,
                                               PassThrough,
                                               PassThrough,
                                               MultiplyAddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_grouped_gemm_xdl_tile_loop_multiply_bias_fastgelu_bf16_i8_bf16_mk_kn_mn_instances(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
