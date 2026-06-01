// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename RandValOutputDataType>
CK_TILE_HOST void
reference_batched_dropout_randval(HostTensor<RandValOutputDataType>& randval_b_m_n,
                                  index_t batch,
                                  uint64_t drop_seed,
                                  uint64_t drop_offset)
{
    const index_t nhead         = randval_b_m_n.mDesc.get_lengths()[0];
    const index_t real_seqlen_q = randval_b_m_n.mDesc.get_lengths()[1];
    const index_t real_seqlen_k = randval_b_m_n.mDesc.get_lengths()[2];

    static_assert(std::is_same_v<RandValOutputDataType, uint8_t>);

    // BlockDropout generates random numbers by 32x32 tiles. Even when warp gemm 16x16 is used, the
    // order of values in the bigger 32x32 tile must be the same because fwd and bwd may use
    // different warp gemms (16x16 or 32x32).
    // To compute 32x32 tiles, WarpGemmMfmaF16F16F32M32N32K16SwizzleA is used. It is
    // WarpGemmAttributeMfmaImplF16F16F32M32N32K8 with SFactor = 2 (swizzling factor).
    // Matrix element to register mapping for WarpGemmAttributeMfmaImplF16F16F32M32N32K8:
    // C i:  (8 * floor(GPR_num / 4) % 32) + 4 * floor(lane / 32) + (GPR_num % 4)
    // C j: (lane % 32)
    // With SFactor = 2 it becomes:
    // C i: (16 * floor(GPR_num / 8) % 32) + 8 * floor(lane / 32) + (GPR_num % 8)
    // C j: (lane % 32)
    // See ck_tile/ops/fmha/block/block_dropout.hpp for more details.

    // The number of Philox 4x32 results required to fill 32x32 tile of 8-bit values
    constexpr index_t philox_per_tile = 64;
    constexpr index_t warp_gemm_mn    = 32;

    const index_t rows = integer_divide_ceil(real_seqlen_q, warp_gemm_mn);
    const index_t cols = integer_divide_ceil(real_seqlen_k, warp_gemm_mn);

    auto f = [&](index_t i_h, index_t row, index_t col) {
        uint2 rowcol = make_uint2(row, col);
        for(index_t lane = 0; lane < philox_per_tile; lane++)
        {
            const uint64_t ph_head_offset = drop_offset + (batch * nhead + i_h) * philox_per_tile;
            const index_t ph_offset       = lane;
            philox ph(drop_seed, ph_head_offset + ph_offset);

            uint8_t random_uint8_t[16];
            ph.get_random_16x8(random_uint8_t, reinterpret_cast<unsigned long long&>(rowcol));

            for(auto r = 0; r < 16; r++)
            {
                index_t i = (16 * (r / 8) % 32) + 8 * (lane / 32) + (r % 8);
                index_t j = (lane % 32);
                index_t m = row * warp_gemm_mn + i;
                index_t n = col * warp_gemm_mn + j;

                if(m < real_seqlen_q && n < real_seqlen_k)
                {
                    randval_b_m_n(i_h, m, n) = random_uint8_t[r];
                }
            }
        }
    };

    make_ParallelTensorFunctor(f, nhead, rows, cols)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
