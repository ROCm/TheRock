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
struct BlockGemmWeightPreshuffleABQuantARegBRegCReg
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem         = remove_cvref_t<PipelineProblem_>;
        using Policy          = remove_cvref_t<GemmPolicy_>;
        using ADataType       = remove_cvref_t<typename Problem::ADataType>;
        using AQDataType      = remove_cvref_t<typename Problem::AQDataType>;
        using BDataType       = remove_cvref_t<typename Problem::BDataType>;
        using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
        using BQLayout        = remove_cvref_t<typename Problem::BQLayout>;
        using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
        using CDataType       = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
        using AQuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;
        using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        // Threadblock GEMM tile size
        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr index_t NQPerBlock = NPerBlock / BQuantGroupSize::kN;
        static constexpr index_t KQPerBlock = KPerBlock / BQuantGroupSize::kK;
        static constexpr index_t AQPerBlock = KPerBlock / AQuantGroupSize::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        // number of warps along M and N for threadblock's GEMM problem size
        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();

        using I0 = number<0>;
        using I1 = number<1>;

        static_assert(MWarp == BlockGemmShape::BlockWarps::at(I0{}),
                      "Error! WarpGemm's MWarp is not consistent with BlockGemmShape!");
        static_assert(NWarp == BlockGemmShape::BlockWarps::at(I1{}),
                      "Error! WarpGemm's NWarp is not consistent with BlockGemmShape!");
        static_assert(WarpGemm::kM == BlockGemmShape::WarpTile::at(I0{}),
                      "Error! WarpGemm's M is not consistent with BlockGemmShape!");
        static_assert(WarpGemm::kN == BlockGemmShape::WarpTile::at(I1{}),
                      "Error! WarpGemm's N is not consistent with BlockGemmShape!");

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;

        static constexpr index_t QScalesPerBlockRow =
            integer_divide_ceil(KPerBlock, BQuantGroupSize::kK);
        static constexpr index_t QScalesPerWarpGemmRow =
            integer_divide_ceil(WarpGemm::kK, BQuantGroupSize::kK);

        static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

        static_assert(BQuantGroupSize::kK % WarpGemm::kK == 0,
                      "Error! WarpGemm::kK should be a multiple of QuantGroupSize");
        static_assert(QScalesPerWarpGemmRow == 1,
                      "Error! QuantGroupSize shouldn't be smaller than WarpGemm::kK");
        static_assert(KIterPerWarp % QScalesPerBlockRow == 0,
                      "Error! KItersPerWarp should be a multiple of QscalesPerBlockRow");

        static_assert(KPerBlock / BQuantGroupSize::kK > 0,
                      "Error! Each row of blockgemm should have a separate scale");

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        // Currently tested combinations (A, B, BQ)
        // 1. fp8, fp8, fp32 -> f32
        // 2. bf8, bf8, fp32 -> f32
        // 3. i4,  fp8, (fp8/fp32) -> f32
        // 4. i4,  bf8, (fp8/fp32) -> f32
        static_assert(
            (std::is_same_v<ADataType, fp8_t> || std::is_same_v<ADataType, bf8_t> ||
             std::is_same_v<ADataType, ck_tile::pk_int4_t>) &&
            (std::is_same_v<BDataType, fp8_t> || std::is_same_v<BDataType, bf8_t> ||
             std::is_same_v<BDataType, ck_tile::pk_int4_t>) &&
            (std::is_same_v<AQDataType, float> || std::is_same_v<AQDataType, ck_tile::fp8_t> ||
             std::is_same_v<AQDataType, ck_tile::bf8_t>) &&
            (std::is_same_v<BQDataType, float> || std::is_same_v<BQDataType, ck_tile::fp8_t> ||
             std::is_same_v<BQDataType, ck_tile::bf8_t>) &&
            (std::is_same_v<ComputeDataType, fp8_t> || std::is_same_v<ComputeDataType, bf8_t>) &&
            std::is_same_v<CDataType, fp32_t>);

        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPack      = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;
        static constexpr bool TransposeC    = Problem::TransposeC;
    };

    public:
    using Traits          = GemmTraits_<Problem_, BlockPolicy_>;
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

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM); // 128 / (1 * 16) = 8
    static constexpr index_t NIterPerWarp =
        BlockTile::at(idxN) / (WarpTile::at(idxN) * BlockWarps::at(idxN)); // 128 / (4 * 16) = 2
    static constexpr index_t KIterPerWarp = KPerBlock / WG::kK;            // 128 / 16 = 8
    static constexpr auto MIter_2nd_last =
        (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;

    static constexpr index_t KPerBlockBQ = KPerBlock / QuantGroupSize::kK;

    static constexpr index_t QScalesPerBlockRow =
        integer_divide_ceil(KPerBlock, QuantGroupSize::kK); // 128 / 128 = 1
    static constexpr index_t QScalesPerWarpGemmRow =
        integer_divide_ceil(WG::kK, QuantGroupSize::kK);

    static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow; // 8 / 1 = 8
    static constexpr index_t DsReadPreload  = 2; // default 2, preload 2 ds read

    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return BlockGemmQuantCommon<CDataType, WG, MIterPerWarp, MWarp, NIterPerWarp, NWarp>::
            MakeCBlockTile();
    }

    // C += A * B
    template <typename CBlockTensor,
              typename ABlockTensor,
              typename BFlatBlockTensor,
              typename AQBlockTensor,
              typename BQBlockTensor,
              typename ABlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   ABlockTensor& a_warp_tensor,
                                   BFlatBlockTensor& b_warp_tensor,
                                   AQBlockTensor& aq_block_tensor,
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
                AQPickerCommon<AQBlockTensor, Traits, mIter, kQScale> aq_picker(aq_block_tensor);
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    constexpr auto tbuf_offset =
                        number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                   merge_sequences(sequence<mIter, nIter>{},
                                                   c_warp_y_index_zeros)) /
                               CBlockTensor::PackedSize>{};

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
                    auto& scale_reg = bq_block_tensor.get_thread_buffer()[reg_offset];
                    float b_scale_reg_f =
                        aq_picker.template cvt_scale_to_fp32<BQDataType>(scale_reg);

                    static_for<0, WG::kM * WG::kN / warp_size, 1>{}([&](auto c_row) {
                        float a_scale_reg_f = aq_picker.template pick<c_row>();
                        auto& c_ref = c_block_tensor.get_thread_buffer()[tbuf_offset + c_row];
                        const auto acc_val = c_acc(mIter)(nIter).get_thread_buffer()[c_row];
                        c_ref              = c_ref + acc_val * b_scale_reg_f * a_scale_reg_f;
                    });
                });
            });
        });
    }
};

} // namespace ck_tile
