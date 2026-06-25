// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_bwd_gamma_beta_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_gamma_beta_f16_instances =
    // clang-format off
    std::tuple <
        // DYDataType, XDataType, MeanInvStdDataType, ComputeDataType, DGammaDataType, DBetaDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, IsDYFastestDimReduced, DYSrcVectorSize, IsXFastestDimReduced, XSrcVectorSize, IsMeanInvStdFastestDimReduced, MeanInvStdSrcVectorSize, DGammaDstVectorSize, DBetaDstVectorSize>
        DeviceNormalizationBwdGammaBetaImpl<F16, F16, F16, F32, F16, F16, Rank, Reduce, 256, 1, 256, 2, 1, false, 2, false, 2, true, 1, 2, 2>,
        DeviceNormalizationBwdGammaBetaImpl<F16, F16, F16, F32, F16, F16, Rank, Reduce, 256, 1, 256, 4, 1, false, 4, false, 4, true, 1, 4, 4>,
        DeviceNormalizationBwdGammaBetaImpl<F16, F16, F16, F32, F16, F16, Rank, Reduce, 256, 1, 256, 8, 1, false, 8, false, 8, true, 1, 8, 8>
        // clang-format on
        >;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_gamma_beta_f16_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationBwdGammaBetaImpl<F16, F16, F16, F32, F16, F16, Rank, Reduce, 64, 1, 64, 1, 1, false, 1, false, 1, true, 1, 1, 1>
    // clang-format on
    >;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_gamma_beta_f32_instances =
    // clang-format off
    std::tuple <
        // DYDataType, XDataType, MeanInvStdDataType, ComputeDataType, DGammaDataType, DBetaDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, IsDYFastestDimReduced, DYSrcVectorSize, IsXFastestDimReduced, XSrcVectorSize, IsMeanInvStdFastestDimReduced, MeanInvStdSrcVectorSize, DGammaDstVectorSize, DBetaDstVectorSize>
        DeviceNormalizationBwdGammaBetaImpl<F32, F32, F32, F32, F32, F32, Rank, Reduce, 256, 1, 256, 2, 1, false, 2, false, 2, true, 1, 2, 2>,
        DeviceNormalizationBwdGammaBetaImpl<F32, F32, F32, F32, F32, F32, Rank, Reduce, 256, 1, 256, 4, 1, false, 4, false, 4, true, 1, 4, 4>
        // clang-format on
        >;

template <index_t Rank, index_t Reduce>
using device_layernorm_bwd_gamma_beta_f32_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationBwdGammaBetaImpl<F32, F32, F32, F32, F32, F32, Rank, Reduce, 64, 1, 64, 1, 1, false, 1, false, 1, true, 1, 1, 1>
    // clang-format on
    >;

using device_groupnorm_bwd_gamma_beta_f32_instances =
    // clang-format off
    std::tuple <
        // DYDataType, XDataType, MeanInvStdDataType, ComputeDataType, DGammaDataType, DBetaDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, IsDYFastestDimReduced, DYSrcVectorSize, IsXFastestDimReduced, XSrcVectorSize, IsMeanInvStdFastestDimReduced, MeanInvStdSrcVectorSize, DGammaDstVectorSize, DBetaDstVectorSize>
        DeviceNormalizationBwdGammaBetaImpl<F32, F32, F32, F32, F32, F32, 5, 3, 256, 1, 256, 2, 1, false, 2, false, 2, false, 1, 2, 2>,
        DeviceNormalizationBwdGammaBetaImpl<F32, F32, F32, F32, F32, F32, 5, 3, 256, 1, 256, 4, 1, false, 4, false, 4, false, 1, 4, 4>
        // clang-format on
        >;

using device_groupnorm_bwd_gamma_beta_f32_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationBwdGammaBetaImpl<F32, F32, F32, F32, F32, F32, 5, 3, 64, 1, 64, 1, 1, false, 1, false, 1, false, 1, 1, 1>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
