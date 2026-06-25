// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp"

namespace ck_tile {

template <bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f16_f16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16, kTransC>>;

template <bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf16_bf16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16, kTransC>>;

template <bool kTransC = false>
using WarpGemmWmma_i32_16x16x16_i8_i8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_i32_16x16x16_i8_i8, kTransC>>;

template <bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_f8, kTransC>>;

template <bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8, kTransC>>;

template <bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_bf8, kTransC>>;

template <bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf8_f8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_f8, kTransC>>;

} // namespace ck_tile
