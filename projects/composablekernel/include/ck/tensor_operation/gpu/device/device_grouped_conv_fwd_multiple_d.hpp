// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/**
 * \brief Grouped Convolution Forward
 *
 * \note This structure is deprecated (left for backwards compatibility). Please use
 *       DeviceGroupedConvFwdMultipleABD.
 *
 * \tparam NDimSpatial Number of spatial dimensions.
 * \tparam ALayout Input layout (also for a1, a2...).
 * \tparam BLayout Weight layout (also for b1, b2...).
 * \tparam DsLayout Ds layouts.
 * \tparam ELayout Output layout.
 * \tparam ADataType Input data type. Pass tuple if there is multiple A.
 * \tparam BDataType Weight data type. Pass tuple if there is multiple B.
 * \tparam DsDataType D data types.
 * \tparam EDataType Output data type.
 * \tparam AElementwiseOperation A elementwise operation.
 * \tparam BElementwiseOperation B elementwise operation.
 * \tparam CDEElementwiseOperation CDE elementwise operation.
 * \tparam ComputeType Compute data type (default: ADataType, first if tuple passed).
 */
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputeType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>())> // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
using DeviceGroupedConvFwdMultipleD = DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                      ALayout,
                                                                      BLayout,
                                                                      DsLayout,
                                                                      ELayout,
                                                                      ADataType,
                                                                      BDataType,
                                                                      DsDataType,
                                                                      EDataType,
                                                                      AElementwiseOperation,
                                                                      BElementwiseOperation,
                                                                      CDEElementwiseOperation,
                                                                      ComputeType>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
