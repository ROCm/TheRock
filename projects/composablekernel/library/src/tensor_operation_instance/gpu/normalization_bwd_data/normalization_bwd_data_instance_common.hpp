// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_bwd_data_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_data_f16_instances =
    // clang-format off
    std::tuple <
        // DYDataType, XDataType, GammaDataType, MeanInvStdDataType, ComputeDataType, DXDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, IsDYFastestDimReduced, DYSrcVectorSize, IsXFastestDimReduced, XSrcVectorSize, IsGammaFastestDimReduced, GammaSrcVectorSize, IsMeanInvStdFastestDimReduced, MeanInvStdSrcVectorSize, IsDXFastestDimReduced, DXDstVectorSize>
        DeviceNormalizationBwdDataImpl<F16, F16, F16, F16, F32, F16, Rank, Reduce, 256, 1, 256, 1, 2, true, 2, true, 2, true, 2, false, 1, true, 2>,
        DeviceNormalizationBwdDataImpl<F16, F16, F16, F16, F32, F16, Rank, Reduce, 256, 1, 256, 1, 4, true, 4, true, 4, true, 4, false, 1, true, 4>,
        DeviceNormalizationBwdDataImpl<F16, F16, F16, F16, F32, F16, Rank, Reduce, 256, 1, 256, 1, 8, true, 8, true, 8, true, 8, false, 1, true, 8>
        // clang-format on
        >;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_data_f16_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationBwdDataImpl<F16, F16, F16, F16, F32, F16, Rank, Reduce, 64, 1, 64, 1, 1, true, 1, true, 1, true, 1, false, 1, true, 1>
    // clang-format on
    >;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_data_f32_instances =
    // clang-format off
    std::tuple <
        // DYDataType, XDataType, GammaDataType, MeanInvStdDataType, ComputeDataType, DXDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, IsDYFastestDimReduced, DYSrcVectorSize, IsXFastestDimReduced, XSrcVectorSize, IsGammaFastestDimReduced, GammaSrcVectorSize, IsMeanInvStdFastestDimReduced, MeanInvStdSrcVectorSize, IsDXFastestDimReduced, DXDstVectorSize>
        DeviceNormalizationBwdDataImpl<F32, F32, F32, F32, F32, F32, Rank, Reduce, 256, 1, 256, 1, 2, true, 2, true, 2, true, 2, false, 1, true, 2>,
        DeviceNormalizationBwdDataImpl<F32, F32, F32, F32, F32, F32, Rank, Reduce, 256, 1, 256, 1, 4, true, 4, true, 4, true, 4, false, 1, true, 4>
        // clang-format on
        >;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_data_f32_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationBwdDataImpl<F32, F32, F32, F32, F32, F32, Rank, Reduce, 64, 1, 64, 1, 1, true, 1, true, 1, true, 1, false, 1, true, 1>
    // clang-format on
    >;

using device_groupnorm_bwd_data_f32_instances =
    // clang-format off
    std::tuple <
        // DYDataType, XDataType, GammaDataType, MeanInvStdDataType, ComputeDataType, DXDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, IsDYFastestDimReduced, DYSrcVectorSize, IsXFastestDimReduced, XSrcVectorSize, IsGammaFastestDimReduced, GammaSrcVectorSize, IsMeanInvStdFastestDimReduced, MeanInvStdSrcVectorSize, IsDXFastestDimReduced, DXDstVectorSize>
        DeviceNormalizationBwdDataImpl<F32, F32, F32, F32, F32, F32, 5, 3, 256, 1, 256, 1, 2, true, 2, true, 2, true, 2, false, 1, true, 2>,
        DeviceNormalizationBwdDataImpl<F32, F32, F32, F32, F32, F32, 5, 3, 256, 1, 256, 1, 4, true, 4, true, 4, true, 4, false, 1, true, 4>
        // clang-format on
        >;

using device_groupnorm_bwd_data_f32_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationBwdDataImpl<F32, F32, F32, F32, F32, F32, 5, 3, 64, 1, 64, 1, 1, true, 1, true, 1, true, 1, false, 1, true, 1>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
