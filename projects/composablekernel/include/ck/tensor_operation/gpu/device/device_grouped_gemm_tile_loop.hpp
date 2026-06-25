// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/// @brief Grouped GEMM kernel using output Tile Looping algorithm
///
/// @par This kernel does not require any knowledge about input data sizes (GEMM M/N/K)
///       It requires only the number of groups to launch. Other information like
///       data pointers and GEMM sizes, packed into gemm kernel args may be all dynamic
///       (known only at kernel run-time).
///
/// @note This kernel does not support SplitK.

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedGemmTileLoop : public DeviceGroupedGemm<ALayout,
                                                            BLayout,
                                                            DsLayout,
                                                            ELayout,
                                                            ADataType,
                                                            BDataType,
                                                            DsDataType,
                                                            EDataType,
                                                            AElementwiseOperation,
                                                            BElementwiseOperation,
                                                            CDEElementwiseOperation>
{
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
