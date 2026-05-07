// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum class FusedMoeGemmWeightPermuteEnum
{
    // permute_b_n0_k0_n1_k1_n2_k2 = 0, // 0,1,4,2,5,3,6
    // permute_b_n0_n1_k0_k1_n2_k2 = 1, // 0,1,2,4,5,3,6
    no_permute          = 0,
    b_nr_kr_kw_nw_kv    = 1, // 0,1,3,4,2,5
    b_nr_kr_waveflatten = b_nr_kr_kw_nw_kv,
};

template <bool IsGateOnly_,
          bool UseSmoothQuant_,
          index_t OAtomic_, // 0-no atomic, 1-atomic-pk-f16/bf16, 2-atomic-f32
          FusedMoeGemmWeightPermuteEnum PermuteEnum_ =
              FusedMoeGemmWeightPermuteEnum::b_nr_kr_waveflatten,
          bool PadHiddenSize_       = false,
          bool PadIntermediateSize_ = false,
          bool PipeInterleave_      = true>
struct FusedMoeGemmTraits
{
    // Gate+Up or Gate only
    static constexpr bool IsGateOnly                           = IsGateOnly_;
    static constexpr bool UseSmoothQuant                       = UseSmoothQuant_;
    static constexpr index_t OAtomic                           = OAtomic_;
    static constexpr FusedMoeGemmWeightPermuteEnum PermuteEnum = PermuteEnum_;
    static constexpr bool PadHiddenSize                        = PadHiddenSize_;
    static constexpr bool PadIntermediateSize                  = PadIntermediateSize_;
    static constexpr bool PipeInterleave                       = PipeInterleave_;
};

// Note: this need to be a bit mask
enum class FusedMoeGemmPipelineSequencerEnum
{
    SLD_A = 1 << 0, // shared load a
    SLD_B = 1 << 1,
    GLD_A = 1 << 2, // global load a
    GLD_B = 1 << 3,
    SST_A = 1 << 4, // shared store a
    SST_B = 1 << 5,
    GST_O = 1 << 6, // global store out
};
} // namespace ck_tile
