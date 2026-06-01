// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_wmma_universal_bf16_i8_bf16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I8          = int8_t;
using BF16        = bhalf_t;
using Row         = tensor_layout::gemm::RowMajor;
using PassThrough = element_wise::PassThrough;

void add_device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               BF16,
                                               I8,
                                               DsDataType,
                                               BF16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances)
{
    if(ck::is_gfx12_supported())
    {
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_instances<GemmDefault,
                                                                              DsLayout,
                                                                              DsDataType>{});
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_instances<GemmKPadding,
                                                                              DsLayout,
                                                                              DsDataType>{});
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_instances<GemmMNPadding,
                                                                              DsLayout,
                                                                              DsDataType>{});
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_bf16_i8_bf16_mk_kn_mn_instances<GemmMNKPadding,
                                                                              DsLayout,
                                                                              DsDataType>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
