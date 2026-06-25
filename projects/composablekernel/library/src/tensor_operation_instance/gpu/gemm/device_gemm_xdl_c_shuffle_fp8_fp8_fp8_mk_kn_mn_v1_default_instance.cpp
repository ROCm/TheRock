// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_v1_instance.hpp"

#ifdef CK_ENABLE_FP8
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

void add_device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_default_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_xdl_c_shuffle_f8_f8_f8_mk_kn_mn_v1_instances<GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
