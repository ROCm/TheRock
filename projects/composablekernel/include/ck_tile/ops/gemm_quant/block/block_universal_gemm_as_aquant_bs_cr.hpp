// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp"

namespace ck_tile {

// A is block window on shared memory
// AQ (scale tensor) is block distributed tensor.
// Consecutive QuantGroupSize elements of A are quantized with a separate scale.
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_,
          typename Policy_     = BlockGemmASmemBSmemCRegV1DefaultPolicy,
          index_t UnaryOpSize_ = 8>
struct AQuantBlockUniversalGemmAsBsCr
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
        using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
        using CDataType       = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
        using QuantGroupSize  = remove_cvref_t<typename Problem::AQuantGroupSize>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        // Threadblock GEMM tile size
        static constexpr index_t MPerBlock  = BlockGemmShape::kM;
        static constexpr index_t NPerBlock  = BlockGemmShape::kN;
        static constexpr index_t KPerBlock  = BlockGemmShape::kK;
        static constexpr index_t AQPerBlock = KPerBlock / QuantGroupSize::kK;

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

        static constexpr index_t QScalesPerBlockRow =
            integer_divide_ceil(KPerBlock, QuantGroupSize::kK);
        static constexpr index_t QScalesPerWarpGemmRow =
            integer_divide_ceil(WarpGemm::kK, QuantGroupSize::kK);

        static constexpr index_t KIterPerQScale = KIterPerWarp / QScalesPerBlockRow;

        static_assert(QuantGroupSize::kK % WarpGemm::kK == 0,
                      "Error! WarpGemm::kK should be a multiple of QuantGroupSize");
        static_assert(QScalesPerWarpGemmRow == 1,
                      "Error! QuantGroupSize shouldn't be smaller than WarpGemm::kK");
        static_assert(KIterPerWarp % QScalesPerBlockRow == 0,
                      "Error! KItersPerWarp should be a multiple of QscalesPerBlockRow");

        static_assert(KPerBlock / QuantGroupSize::kK > 0,
                      "Error! Each row of blockgemm should have a separate scale");

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        // Currently tested combinations (A, AQ, B)
        // 1. fp8, fp32, fp8 -> f32
        // 2. bf8, fp32, bf8 -> f32
        // 3. i4, (fp8/fp32) fp8 -> f32
        // 4. i4, (fp8/fp32) bf8 -> f32
        static_assert((std::is_same_v<ADataType, pk_int4_t> || std::is_same_v<ADataType, fp8_t> ||
                       std::is_same_v<ADataType, bf8_t>) &&
                      (std::is_same_v<BDataType, fp8_t> || std::is_same_v<BDataType, bf8_t>) &&
                      (std::is_same_v<AQDataType, float> ||
                       std::is_same_v<AQDataType, ck_tile::fp8_t> ||
                       std::is_same_v<AQDataType, ck_tile::bf8_t>) &&
                      (std::is_same_v<ComputeDataType, fp8_t> ||
                       std::is_same_v<ComputeDataType, bf8_t>) &&
                      std::is_same_v<CDataType, fp32_t>);

        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        static constexpr index_t KPack      = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;

        static constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;
        static constexpr bool TransposeC      = Problem::TransposeC;
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType       = remove_cvref_t<typename Traits::ADataType>;
    using AQDataType      = remove_cvref_t<typename Traits::AQDataType>;
    using BDataType       = remove_cvref_t<typename Traits::BDataType>;
    using ComputeDataType = remove_cvref_t<typename Traits::ComputeDataType>;
    using CDataType       = remove_cvref_t<typename Traits::CDataType>;

    using WarpGemm = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;

    static constexpr auto Scheduler = Traits::Scheduler;

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    static_assert(std::is_same_v<typename WarpGemm::CDataType, float>);

    static constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    static constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    static constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using I0 = number<0>;
    using I1 = number<1>;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;

        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);

        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, KIterSeq>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return a_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;

        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, KIterSeq>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return b_block_dstr_encode;
    }

    private:
    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {
        public:
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        template <typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window,
                                          bool_constant<ALoadTranspose> = {},
                                          bool_constant<BLoadTranspose> = {})
        {
            // while ADatatype might not be the same as BDataType at the time of problem
            // initialization, we can safely use BDataType here because when A would be int4 we will
            // ensure A is converted to BDataType prior to loading
            load_int4_tile<BDataType, ComputeDataType, UnaryOpSize_, ALoadTranspose>(
                a_warp_tile_, a_block_window);
            load_int4_tile<BDataType, ComputeDataType, UnaryOpSize_, BLoadTranspose>(
                b_warp_tile_, b_block_window);
        }

        // C += A * B
        template <typename CBlockTensor,
                  typename AQBlockTensor,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       AQBlockTensor& aq_block_tensor,
                                       [[maybe_unused]] ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as corresponding "
                          "C block tensor data type!");
            constexpr auto warp_size = get_warp_size();

            // hot loop:
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    CWarpTensor c_warp_tensor;

                    static_for<0, Traits::QScalesPerBlockRow, 1>{}([&](auto kQScale) {
                        static_for<0, Traits::KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                            constexpr auto kIter = kQScale * Traits::KIterPerQScale + kIterInQScale;

                            AWarpTensor a_warp_tensor;
                            a_warp_tensor.get_thread_buffer() =
                                a_warp_tile_.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                            BWarpTensor b_warp_tensor;
                            b_warp_tensor.get_thread_buffer() =
                                b_warp_tile_.get_y_sliced_thread_data(
                                    merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                            if constexpr(kIterInQScale == 0)
                            {
                                c_warp_tensor = WarpGemm{}(a_warp_tensor, b_warp_tensor);
                            }
                            else
                            {
                                WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                            }
                        });

                        constexpr auto tbuf_offset =
                            number<typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
                                       merge_sequences(sequence<mIter, nIter>{},
                                                       c_warp_y_index_zeros)) /
                                   CBlockTensor::PackedSize>{};

                        AQPickerCommon<AQBlockTensor, Traits, mIter, kQScale> aq_picker(
                            aq_block_tensor);

                        static_for<0, WarpGemm::kM * WarpGemm::kN / warp_size, 1>{}(
                            [&](auto c_row) {
                                float scale_reg_f = aq_picker.template pick<c_row>();

                                c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
                                    (c_warp_tensor.get_thread_buffer()[c_row] * scale_reg_f);
                            });
                    });
                });
            });
        }
    };

    public:
    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return BlockGemmQuantCommon<CDataType, WarpGemm, MIterPerWarp, MWarp, NIterPerWarp, NWarp>::
            MakeCBlockTile();
    }

    template <typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              bool ALoadTranspose = false,
              bool BLoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window,
                                      bool_constant<ALoadTranspose> a_load_tr = {},
                                      bool_constant<BLoadTranspose> b_load_tr = {})
    {
        block_gemm_impl_.LocalPrefetch(a_block_window, b_block_window, a_load_tr, b_load_tr);
    }

    // C += A * B
    template <typename CBlockTensor,
              typename AQBlockTensor,
              typename ASmemBlockWindow,
              typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   AQBlockTensor& aq_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_(c_block_tensor, aq_block_tensor, a_block_window, b_block_window);
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
