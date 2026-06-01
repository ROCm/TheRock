// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pool_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto ReduceOpId = ck::ReduceTensorOp::AVG;

void add_device_pool3d_fwd_ndhwc_bf16_instances(
    std::vector<
        std::unique_ptr<DevicePoolFwd<5, 3, BF16, BF16, I32, NDHWC, NDHWC, ReduceOpId, false>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_pool3d_fwd_ndhwc_instances<BF16, BF16, I32, F32, ReduceOpId, false>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
