// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/load_interleaved_pk_type.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_,
          typename Policy_     = BlockGemmASmemBSmemCRegV1DefaultPolicy,
          index_t UnaryOpSize_ = 8>
struct BlockUniversalGemmAsBsCr
{
    private:
    // TODO: This should be in Policy - UniversalGemmPolicyBase ?
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem         = remove_cvref_t<PipelineProblem_>;
        using Policy          = remove_cvref_t<GemmPolicy_>;
        using ADataType       = remove_cvref_t<typename Problem::ADataType>;
        using BDataType       = remove_cvref_t<typename Problem::BDataType>;
        using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
        using CDataType       = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();

        using I0 = number<0>;
        using I1 = number<1>;

        static_assert(MWarp == BlockGemmShape::BlockWarps::at(I0{}),
                      "Error! WarpGemm's MWarp is not consisten with BlockGemmShape!");
        static_assert(NWarp == BlockGemmShape::BlockWarps::at(I1{}),
                      "Error! WarpGemm's NWarp is not consisten with BlockGemmShape!");
        static_assert(WarpGemm::kM == BlockGemmShape::WarpTile::at(I0{}),
                      "Error! WarpGemm's M is not consisten with BlockGemmShape!");
        static_assert(WarpGemm::kN == BlockGemmShape::WarpTile::at(I1{}),
                      "Error! WarpGemm's N is not consisten with BlockGemmShape!");

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        static constexpr index_t MPerBlockPerIter = MWarp * WarpGemm::kM;
        static constexpr index_t NPerBlockPerIter = NWarp * WarpGemm::kN;
        static constexpr index_t KPerBlockPerIter = WarpGemm::kK;

        // Controls how many MAC clusters (MFMA blocks) we have per wave
        // Ie if
        // InterWaveSchedulingMacClusters = 1;
        // KPerBlock == 32
        // WarpGemm::kK = 8
        // Then we would group all 4 WarpGemms into single MAC cluster.
        // But if we would set InterWaveSchedulingMacClusters = 2, then we would
        // split those 4 warp gemms into two groups.
        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        // should be at least equal to: WarpGemm::Impl::kABKPerLane
        static constexpr index_t KPack      = WarpGemm::kKPerThread;
        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType       = remove_cvref_t<typename Traits::ADataType>;
    using BDataType       = remove_cvref_t<typename Traits::BDataType>;
    using ComputeDataType = remove_cvref_t<typename Traits::ComputeDataType>;
    using CDataType       = remove_cvref_t<typename Traits::CDataType>;

    using ATypeToUse =
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
    using BTypeToUse = std::conditional_t<std::is_same_v<BDataType, pk_int4_t> ||
                                              std::is_same_v<BDataType, pk_fp4_raw_t>,
                                          ADataType,
                                          BDataType>;

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

    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Default, GemmTraits>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ATypeToUse>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<BTypeToUse>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        // C += A * B
        template <typename CBlockTensor,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BSmemBlockWindow& b_block_window,
                                       bool_constant<ALoadTranspose> = {},
                                       bool_constant<BLoadTranspose> = {})
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");
            static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                              std::is_same_v<BDataType, typename BSmemBlockWindow::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            load_int4_tile<ADataType, ATypeToUse, UnaryOpSize_, ALoadTranspose>(a_warp_tile_,
                                                                                a_block_window);
            load_int4_tile<BDataType, BTypeToUse, UnaryOpSize_, BLoadTranspose>(b_warp_tile_,
                                                                                b_block_window);
            // hot loop:
            static_for<0, GemmTraits::KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // read C warp tensor from C block tensor-
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ATypeToUse>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<BTypeToUse>(BLdsTileDistr));

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
            load_int4_tile<ADataType, ATypeToUse, UnaryOpSize_, ALoadTranspose>(a_warp_tile_,
                                                                                a_block_window);
            load_int4_tile<BDataType, BTypeToUse, UnaryOpSize_, BLoadTranspose>(b_warp_tile_,
                                                                                b_block_window);
        }

        // C += A * B
        template <typename CBlockTensor,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow&,
                                       const BSmemBlockWindow&,
                                       bool_constant<ALoadTranspose> = {},
                                       bool_constant<BLoadTranspose> = {})
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits>
    {
        static constexpr index_t KPerThread     = GemmTraits::KPerThread;
        static constexpr index_t NumMacClusters = GemmTraits::InterWaveSchedulingMacClusters;
        static constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        static constexpr index_t KRepeat        = KPerThread / KPerInnerLoop;
        static constexpr index_t KInnerLoopIter = KPerInnerLoop / WarpGemm::kKPerThread;

        static constexpr auto ALdsTileDistr =
            make_static_tile_distribution(MakeABlockDistributionEncode());
        static constexpr auto BLdsTileDistr =
            make_static_tile_distribution(MakeBBlockDistributionEncode());

        using ALdsTile = decltype(make_static_distributed_tensor<ATypeToUse>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<BTypeToUse>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        template <index_t KIdx,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window,
                                          bool_constant<ALoadTranspose> = {},
                                          bool_constant<BLoadTranspose> = {})
        {
            constexpr auto a_lds_load_distr = [&]() {
                if constexpr(ALoadTranspose)
                    return make_static_tile_distribution(typename InputTileDistributionTraits<
                                                         decltype(MakeABlockDistributionEncode()),
                                                         ADataType>::TransposedDstrEncode{});
                else
                    return make_static_tile_distribution(MakeABlockDistributionEncode());
            }();
            constexpr auto b_lds_load_distr = [&]() {
                if constexpr(BLoadTranspose)
                    return make_static_tile_distribution(typename InputTileDistributionTraits<
                                                         decltype(MakeBBlockDistributionEncode()),
                                                         BDataType>::TransposedDstrEncode{});
                else
                    return make_static_tile_distribution(MakeBBlockDistributionEncode());
            }();
            constexpr auto a_lds_shape = []() {
                if constexpr(ALoadTranspose)
                    return make_tuple(number<KPerInnerLoop>{}, number<GemmTraits::MPerBlock>{});
                else
                    return make_tuple(number<GemmTraits::MPerBlock>{}, number<KPerInnerLoop>{});
            }();
            constexpr auto b_lds_shape = []() {
                if constexpr(BLoadTranspose)
                    return make_tuple(number<KPerInnerLoop>{}, number<GemmTraits::NPerBlock>{});
                else
                    return make_tuple(number<GemmTraits::NPerBlock>{}, number<KPerInnerLoop>{});
            }();
            constexpr auto k_idx_offset = KIdx * KPerInnerLoop;
            constexpr auto a_offset =
                ALoadTranspose ? multi_index<2>{k_idx_offset, 0} : multi_index<2>{0, k_idx_offset};
            constexpr auto b_offset =
                BLoadTranspose ? multi_index<2>{k_idx_offset, 0} : multi_index<2>{0, k_idx_offset};

            auto a_lds_gemm_window = make_tile_window(
                a_block_window.get_bottom_tensor_view(), a_lds_shape, a_offset, a_lds_load_distr);
            auto b_lds_gemm_window = make_tile_window(
                b_block_window.get_bottom_tensor_view(), b_lds_shape, b_offset, b_lds_load_distr);

            load_int4_tile<ADataType, ATypeToUse, UnaryOpSize_, ALoadTranspose>(a_warp_tile_,
                                                                                a_lds_gemm_window);
            load_int4_tile<BDataType, BTypeToUse, UnaryOpSize_, BLoadTranspose>(b_warp_tile_,
                                                                                b_lds_gemm_window);
        }

        // C += A * B
        template <typename CBlockTensor,
                  typename ASmemBlockWindow,
                  typename BSmemBlockWindow,
                  bool ALoadTranspose = false,
                  bool BLoadTranspose = false>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BSmemBlockWindow& b_block_window,
                                       bool_constant<ALoadTranspose> a_load_tr = {},
                                       bool_constant<BLoadTranspose> b_load_tr = {})
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KRepeat, 1>{}([&](auto kIter) {
                LocalPrefetch<kIter.value>(a_block_window, b_block_window, a_load_tr, b_load_tr);
                __builtin_amdgcn_sched_barrier(0);
                // NOTE: Synchronize threads in a workgroup at the start of each MAC
                // cluster, but except the first, as we can shorten non-MAC cluster a bit
                // and there's no observable negative impact. The desired effect is waves in
                // a workgroup executing MAC in sync. This avoids some out-of-sync waves
                // hijacking MAC resource from other workgroups and reducing the chance of
                // latency hiding by waiting for the rest of the workgroup at the eventual
                // sync point.
                if constexpr(kIter.value != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }

                static_for<0, KInnerLoopIter, 1>{}([&](auto kInnerIter) {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        // read A warp tensor from A block tensor
                        AWarpTensor a_warp_tensor;

                        a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, kInnerIter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read B warp tensor from B block tensor
                            BWarpTensor b_warp_tensor;

                            b_warp_tensor.get_thread_buffer() =
                                b_warp_tile_.get_y_sliced_thread_data(
                                    merge_sequences(sequence<nIter, kInnerIter>{},
                                                    b_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                            // read C warp tensor from C block tensor-
                            CWarpTensor c_warp_tensor;

                            c_warp_tensor.get_thread_buffer() =
                                c_block_tensor.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                            // The block_sync_lds() here performs double duty:
                            // A) safeguard against data hazard because barrier from
                            // blockwise_gemm is moved here B) reduce VMEM FIFO congestion
                            // by applying small delays to different wavefronts It is
                            // performed near the end of MAC cluster to minimize lgkmcnt
                            // penalty
                            if constexpr(kIter.value == KRepeat - 1 &&
                                         kInnerIter.value == KInnerLoopIter - 1 &&
                                         mIter.value == MIterPerWarp - 1 &&
                                         nIter.value == NIterPerWarp - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            // warp GEMM
                            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                            // write C warp tensor into C block tensor
                            c_block_tensor.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());

                            if constexpr(kInnerIter.value == 0 && mIter.value == 0 &&
                                         nIter.value == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    };

    public:
    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);

        return c_block_tensor;
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
              typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              bool ALoadTranspose = false,
              bool BLoadTranspose = false>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window,
                                   bool_constant<ALoadTranspose> a_load_tr = {},
                                   bool_constant<BLoadTranspose> b_load_tr = {})
    {
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window, a_load_tr, b_load_tr);
    }

    // C = A * B
    template <typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              bool ALoadTranspose = false,
              bool BLoadTranspose = false>
    CK_TILE_DEVICE auto operator()(const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window,
                                   bool_constant<ALoadTranspose> a_load_tr = {},
                                   bool_constant<BLoadTranspose> b_load_tr = {})
    {
        auto c_block_tensor = MakeCBlockTile();
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window, a_load_tr, b_load_tr);
        return c_block_tensor;
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
