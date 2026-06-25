// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp"

namespace ck_tile {

// A is block window on shared memory
// BQ (scale tensor) is block distributed tensor.
// Consecutive QuantGroupSize elements of B are quantized with a separate scale.
// B is block window on block distributed tensor.
// C is block distributed tensor
template <typename Problem_, typename BlockPolicy_>
struct BlockGemmWeightPreshuffleBQuantARegBRegCReg
{
    using Problem         = remove_cvref_t<Problem_>;
    using BlockPolicy     = remove_cvref_t<BlockPolicy_>;
    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape
    using QuantGroupSize  = remove_cvref_t<typename Problem::BQuantGroupSize>;

    static_assert(QuantGroupSize::kM == 1, "only N/K blocks for BQuant preshuffle kernel!");

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr auto config = BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    static constexpr auto warp_size = get_warp_size();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp =
        BlockTile::at(idxN) / (WarpTile::at(idxN) * BlockWarps::at(idxN));
    static constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    static constexpr auto MIter_2nd_last =
        (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;

    static constexpr index_t KPerBlockBQ = KPerBlock / QuantGroupSize::kK;

    static constexpr index_t QScalesPerBlockRow =
        integer_divide_ceil(KPerBlock, QuantGroupSize::kK);
    static constexpr index_t QScalesPerWarpGemmRow =
        integer_divide_ceil(WG::kK, QuantGroupSize::kK);

    static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;
    static constexpr index_t DsReadPreload  = 2; // default 2, preload 2 ds read

    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

    template <typename T>
    CK_TILE_DEVICE static float cvt_scale_to_fp32(T& scale)
    {
        float scale_reg_f = 0.f;
        if constexpr(std::is_same_v<BQDataType, ck_tile::fp8_t>)
        {
            scale_reg_f = __builtin_amdgcn_cvt_f32_fp8(static_cast<uint32_t>(scale), 0);
        }
        else if constexpr(std::is_same_v<BQDataType, ck_tile::bf8_t>)
        {
            scale_reg_f = __builtin_amdgcn_cvt_f32_bf8(static_cast<uint32_t>(scale), 0);
        }
        else if constexpr(std::is_same_v<BQDataType, float>)
        {
            scale_reg_f = ck_tile::bit_cast<float>(scale);
        }
        else
        {
            static_assert(false, "BQDataType must be float, fp8_t or bf8_t.");
        }
        return scale_reg_f;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return BlockGemmQuantCommon<CDataType, WG, MIterPerWarp, MWarp, NIterPerWarp, NWarp>::
            MakeCBlockTile();
    }

    // C += A * B
    template <typename CBlockTensor,
              typename ABlockTensor,
              typename BFlatBlockTensor,
              typename BQBlockTensor,
              typename ABlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   ABlockTensor& a_warp_tensor,
                                   BFlatBlockTensor& b_warp_tensor,
                                   BQBlockTensor& bq_block_tensor,
                                   ABlockWindow& a_warp_windows) const
    {
        using CWarpDstr = typename WG::CWarpDstr;
        using AccTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        statically_indexed_array<statically_indexed_array<AccTensor, NIterPerWarp>, MIterPerWarp>
            c_acc;

        auto zero_accumulators = [&] {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    static_for<0, (WG::kM * WG::kN) / warp_size, 1>{}([&](auto i) {
                        c_acc(mIter)(nIter).get_thread_buffer()[i] = 0.0f;
                    }); // make sure WG::CWarpTensor exposes a clear/zero
                });
            });
        };
        static_for<0, QScalesPerBlockRow, 1>{}([&](auto kQScale) {
            zero_accumulators();
            static_for<0, KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                constexpr auto kIter = kQScale * KIterPerQScale + kIterInQScale;
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // warp GEMM
                        WG{}(c_acc(mIter)(nIter),
                             a_warp_tensor(number<AwarpIter>{}),
                             b_warp_tensor(nIter)(number<kIter>{}));
                    });
                    __builtin_amdgcn_sched_barrier(0x7F6);
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) =
                            load_tile(a_warp_windows(number<AmIter>{})(number<AkIter>{}));
                    }
                    // barrier
                    // Could be deleted
                    if constexpr((mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    constexpr auto tbuf_offset =
                        number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                   merge_sequences(sequence<mIter, nIter>{},
                                                   c_warp_y_index_zeros)) /
                               CBlockTensor::PackedSize>{};

                    if constexpr(PreshuffleQuant)
                    {
                        constexpr index_t reg_offset = nIter;
                        auto pull_from_lane = (__lane_id() & (WG::kN - 1)) * KPerBlockBQ + kQScale;
                        auto& scale_reg     = bq_block_tensor.get_thread_buffer()[reg_offset];
                        // cross lane ops
                        uint32_t scale_reg_dword;

                        if constexpr(std::is_same_v<BQDataType, float>)
                        {
                            scale_reg_dword = ck_tile::bit_cast<uint32_t>(scale_reg);
                        }
                        else
                        {
                            scale_reg_dword = static_cast<uint32_t>(scale_reg);
                        }

                        // cross lane ops to get the value of scale_reg.
                        int gathered_scale_reg = __builtin_amdgcn_ds_bpermute(
                            pull_from_lane << 2, __builtin_bit_cast(int, scale_reg_dword));

                        float scale_reg_f = cvt_scale_to_fp32(gathered_scale_reg);

                        static_for<0, WG::kM * WG::kN / warp_size, 1>{}([&](auto c_row) {
                            auto& c_ref = c_block_tensor.get_thread_buffer()[tbuf_offset + c_row];
                            const auto acc_val = c_acc(mIter)(nIter).get_thread_buffer()[c_row];
                            c_ref              = c_ref + acc_val * scale_reg_f;
                        });
                    }
                    else
                    {
                        index_t reg_offset = [&]() {
                            if constexpr(QuantGroupSize::kN >= (NWarp * WG::kN))
                            {
                                return (nIter * NWarp * WG::kN) / QuantGroupSize::kN * KPerBlockBQ +
                                       kQScale;
                            }
                            else
                            {
                                return nIter * KPerBlockBQ + kQScale;
                            }
                        }();
                        auto& scale_reg   = bq_block_tensor.get_thread_buffer()[reg_offset];
                        float scale_reg_f = cvt_scale_to_fp32(scale_reg);

                        static_for<0, WG::kM * WG::kN / warp_size, 1>{}([&](auto c_row) {
                            auto& c_ref = c_block_tensor.get_thread_buffer()[tbuf_offset + c_row];
                            const auto acc_val = c_acc(mIter)(nIter).get_thread_buffer()[c_row];
                            c_ref              = c_ref + acc_val * scale_reg_f;
                        });
                    }
                });
            });
        });
    }
};

} // namespace ck_tile
