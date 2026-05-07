// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_wmma_cshuffle_v3.hpp"

#include "device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_common.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABDSplitK<AsLayout,
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
        device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_comp_instances<ck::Tuple<D0Layout>,
                                                                        ck::Tuple<D0DataType>,
                                                                        AddFastGelu,
                                                                        GemmMNKPadding,
                                                                        Interwave>{});
}

void add_device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABDSplitK<AsLayout,
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
        device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_comp_instances<ck::Tuple<D0Layout>,
                                                                        ck::Tuple<D0DataType>,
                                                                        Add,
                                                                        GemmMNKPadding,
                                                                        Interwave>{});
}

void add_device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABDSplitK<AsLayout,
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
        device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_comp_instances<ck::Tuple<>,
                                                                        ck::Tuple<>,
                                                                        PassThrough,
                                                                        GemmMNKPadding,
                                                                        Interwave>{});
}

void add_device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABDSplitK<AsLayout,
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
        device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_comp_instances<ck::Tuple<>,
                                                                        ck::Tuple<>,
                                                                        FastGelu,
                                                                        GemmMNKPadding,
                                                                        Interwave>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
