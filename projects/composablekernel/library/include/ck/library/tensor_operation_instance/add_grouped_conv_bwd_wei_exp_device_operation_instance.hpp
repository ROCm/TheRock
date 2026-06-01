// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <type_traits>

#include "ck/utility/functional2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_explicit.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename DeviceGemmV3Ops,
          typename BaseOp>
void add_explicit_gemm_device_operation_instances(
    std::vector<std::unique_ptr<BaseOp>>& op_instances)
{
    ck::static_for<0, std::tuple_size_v<DeviceGemmV3Ops>, 1>{}([&](auto i) {
        using DeviceGemmOp = std::tuple_element_t<i, DeviceGemmV3Ops>;

        using NewOpInstance = DeviceGroupedConvBwdWeight_Explicit<NDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  OutLayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  InElementwiseOperation,
                                                                  WeiElementwiseOperation,
                                                                  OutElementwiseOperation,
                                                                  DeviceGemmOp>;

        static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                      "wrong! NewOpInstance should be derived from BaseOp");

        op_instances.push_back(std::make_unique<NewOpInstance>(NewOpInstance{}));
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
