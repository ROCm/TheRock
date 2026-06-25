// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16         = half_t;
using Row         = tensor_layout::gemm::RowMajor;
using PassThrough = element_wise::PassThrough;
using Add         = element_wise::Add;

using DsLayout_F16   = ck::Tuple<>;
using DsDataType_F16 = ck::Tuple<>;

void add_device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout_F16,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType_F16,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances)
{
    if(ck::is_gfx12_supported())
    {
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_instances<GemmDefault,
                                                                             DsLayout_F16,
                                                                             DsDataType_F16>{});
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_instances<GemmKPadding,
                                                                             DsLayout_F16,
                                                                             DsDataType_F16>{});
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_instances<GemmMNPadding>{});
        add_device_operation_instances(
            instances,
            device_gemm_wmma_universal_reduce_f16_f16_f16_mk_kn_mn_instances<GemmMNKPadding>{});
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
