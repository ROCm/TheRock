// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_km_kn_mn_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_bias_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                 BsLayout,
                                                                 ck::Tuple<D0Layout>,
                                                                 ELayout,
                                                                 AsDataType,
                                                                 BsDataType,
                                                                 ck::Tuple<D0DataType>,
                                                                 EDataType,
                                                                 AElementOp,
                                                                 BElementOp,
                                                                 AddFastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances<
            ck::Tuple<D0Layout>,
            ck::Tuple<D0DataType>,
            AddFastGelu,
            GemmMNKPadding>{});
}

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_bias_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                 BsLayout,
                                                                 ck::Tuple<D0Layout>,
                                                                 ELayout,
                                                                 AsDataType,
                                                                 BsDataType,
                                                                 ck::Tuple<D0DataType>,
                                                                 EDataType,
                                                                 AElementOp,
                                                                 BElementOp,
                                                                 Add>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances<
            ck::Tuple<D0Layout>,
            ck::Tuple<D0DataType>,
            Add,
            GemmMNKPadding>{});
}

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                 BsLayout,
                                                                 ck::Tuple<>,
                                                                 ELayout,
                                                                 AsDataType,
                                                                 BsDataType,
                                                                 ck::Tuple<>,
                                                                 EDataType,
                                                                 AElementOp,
                                                                 BElementOp,
                                                                 PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances<
            ck::Tuple<>,
            ck::Tuple<>,
            PassThrough,
            GemmMNKPadding>{});
}

void add_device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_gelu_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                 BsLayout,
                                                                 ck::Tuple<>,
                                                                 ELayout,
                                                                 AsDataType,
                                                                 BsDataType,
                                                                 ck::Tuple<>,
                                                                 EDataType,
                                                                 AElementOp,
                                                                 BElementOp,
                                                                 FastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_km_kn_mn_instances<
            ck::Tuple<>,
            ck::Tuple<>,
            FastGelu,
            GemmMNKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
