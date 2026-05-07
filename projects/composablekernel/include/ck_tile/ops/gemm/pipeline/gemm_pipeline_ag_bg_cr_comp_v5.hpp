// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5_default_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {
// A Tile Window: global memory
// B Tile Window: global memory
// C Distributed Tensor: register

template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV5
{
    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t) { return true; }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t)
    {
        return TailNumber::Empty;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func, bool, TailNumber)
    {
        return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Empty>{});
    }
};

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV5DefaultPolicy>
struct GemmPipelineAgBgCrCompV5 : public BaseGemmPipelineAgBgCrCompV5<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV5<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType      = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType      = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AElementWise = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise = remove_cvref_t<typename Problem::BElementWise>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr index_t NumWarps  = BlockGemmShape::NumWarps;
    static constexpr index_t KTileSize = BlockGemmShape::WarpTile::at(I2{});

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_V5";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompV5", BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(),  GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Policy::template IsTransposeC<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BsDramBlockWindowTmp,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem_0) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

            static_assert(
                KPerBlock % ((NumWarps / 2) * KTileSize) == 0,
                "Ping Pong Warps, TileSize and Block Size for K dimensions does not match.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            index_t warp_id = get_warp_id();
            index_t operation_id =
                amd_wave_read_first_lane(get_warp_id()); // 0 - Memory read, 1 - block-gemm

            auto a_offset = (warp_id == 0) ? make_array(0, 0) : make_array(0, KPerBlock);
            auto b_offset = (warp_id == 0) ? make_array(0, 0) : make_array(0, KPerBlock);

            auto tensor_views =
                Base::GetABLdsTensorViews(static_cast<void*>(static_cast<char*>(p_smem_0)));
            auto& a_lds_block = tensor_views.get(number<0>{});
            auto& b_lds_block = tensor_views.get(number<1>{});

            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            auto a_windows = Base::GetAWindows(
                a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr, a_offset);
            auto& a_copy_dram_window = a_windows.get(number<0>{});
            auto& a_copy_lds_window  = a_windows.get(number<1>{});
            auto& a_lds_window       = a_windows.get(number<2>{});

            auto b_windows = Base::GetBWindows(
                b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr, b_offset);
            auto& b_copy_dram_window = b_windows.get(number<0>{});
            auto& b_copy_lds_window  = b_windows.get(number<1>{});
            auto& b_lds_window       = b_windows.get(number<2>{});

            // DRAM window steps.
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock * NumWarps, 0)
                               : make_array(0, KPerBlock * NumWarps);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock * NumWarps, 0)
                               : make_array(0, KPerBlock * NumWarps);

            constexpr auto AGemmTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto BGemmTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            using AGemmTile = decltype(make_static_distributed_tensor<ADataType>(AGemmTileDistr));
            using BGemmTile = decltype(make_static_distributed_tensor<BDataType>(BGemmTileDistr));
            AGemmTile a_tile_0, a_tile_1;
            BGemmTile b_tile_0, b_tile_1;

            // Register tile for A and B.
            using ABlockTileDistr =
                decltype(a_copy_dram_window[number<0>{}].get_tile_distribution());
            using BBlockTileDistr =
                decltype(b_copy_dram_window[number<0>{}].get_tile_distribution());
            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));
            ABlockTile elementwise_As_res;
            BBlockTile elementwise_Bs_res;

            // Block GEMM
            auto block_gemm     = BlockGemm();
            auto c_block_tile_0 = block_gemm.MakeCBlockTile();
            auto c_block_tile_1 = block_gemm.MakeCBlockTile();

            CDataType* __restrict__ p_c_lds = static_cast<CDataType*>(p_smem_0);
            auto c_lds_block_0 =
                make_naive_tensor_view<address_space_enum::lds>(p_c_lds,
                                                                make_tuple(MPerBlock, NPerBlock),
                                                                make_tuple(NPerBlock, 1),
                                                                number<BlockGemm::Traits::KPack>{},
                                                                number<1>{});
            auto c_window_0 = make_tile_window(c_lds_block_0,
                                               make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                               {0, 0},
                                               c_block_tile_1.get_tile_distribution());

            // initialize C
            if(warp_id == 0)
            {
                tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile_0);
            }
            else
            {
                tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile_1);
            }

            // define ping, pong steps here as lambda functions.
            auto MemoryOpsStep = [&](auto idx) {
                // Memory read half here.

                // Load tile — during value loading, an elementwise function is executed for each
                // A0, A1, … AN. The values A0, A1, … AN are read by the same thread.
                elementwise_As_res = load_tile_with_elementwise(a_copy_dram_window, a_element_func);

                // Move each A — the enhanced function move_tile_window is executed, which takes a
                // tuple as input.
                move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                // Load tile — during value loading, an elementwise function is executed for each
                // B0, B1, … BN. The values B0, B1, … BN are read by the same thread.
                elementwise_Bs_res = load_tile_with_elementwise(b_copy_dram_window, b_element_func);

                // Move each B — the enhanced function move_tile_window is executed, which takes a
                // tuple as input.
                move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                if constexpr(is_a_col_major)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                }

                if constexpr(is_b_row_major)
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                }

                if(idx == 0)
                {
                    Base::LocalPrefetch(a_tile_0, a_lds_window);
                    Base::LocalPrefetch(b_tile_0, b_lds_window);
                }
                else
                {
                    Base::LocalPrefetch(a_tile_1, a_lds_window);
                    Base::LocalPrefetch(b_tile_1, b_lds_window);
                }
            };

            auto ComputeStep = [&](auto idx) {
                if(idx == 0)
                {
                    block_gemm(c_block_tile_0, a_tile_0, b_tile_0);
                }
                else
                {
                    block_gemm(c_block_tile_1, a_tile_1, b_tile_1);
                }
            };

            if(operation_id == 0)
            {
                MemoryOpsStep(warp_id);
            }

            index_t num_compute_steps = amd_wave_read_first_lane(num_loop);
            while(num_compute_steps > 1)
            {
                block_sync_lds();
                operation_id = (operation_id + 1) % NumWaveGroups;

                if(operation_id == 0)
                {
                    MemoryOpsStep(warp_id);
                }
                else
                {
                    ComputeStep(warp_id);
                }
                num_compute_steps -= 1;
            }
            block_sync_lds();

            if(operation_id == 0)
            {
                ComputeStep(warp_id);
            }
            block_sync_lds();

            if(warp_id == 1)
            {
                store_tile(c_window_0, c_block_tile_1);
            }
            block_sync_lds();

            if(warp_id == 0)
            {
                load_tile(c_block_tile_1, c_window_0);

                constexpr auto s_spans = decltype(c_block_tile_0)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        auto idx2 = make_tuple(idx0, idx1);
                        c_block_tile_0(idx2) += c_block_tile_1(idx2);
                    });
                });
            }
            return c_block_tile_0;
        }
    };

    public:
    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem_0) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                num_loop,
                p_smem_0);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem_0) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                [](auto& e, const ADataType& a) { e = a; },
                b_dram_block_window_tmp,
                [](auto& e, const BDataType& b) { e = b; },
                num_loop,
                p_smem_0);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem_0) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          a_element_func,
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          b_element_func,
                          num_loop,
                          p_smem_0);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem_0) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          num_loop,
                          p_smem_0);
    }
};

} // namespace ck_tile
