// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_wmma(const ADataType* __restrict__ p_a_grid,
                     const BDataType* __restrict__ p_b_grid,
                     CDataType* __restrict__ p_c_grid,
                     const AGridDesc a_grid_desc,
                     const BGridDesc b_grid_desc,
                     const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                         c_grid_desc_mblock_mperblock_nblock_nperblock,
                     const AElementwiseOperation a_element_op,
                     const BElementwiseOperation b_element_op,
                     const CElementwiseOperation c_element_op,
                     const Block2CTileMap block_2_ctile_map)
{
#if(defined(__gfx11__) || defined(__gfx12__))
    __shared__ char p_shared[GridwiseGemm::SharedMemTrait::lds_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared,
                                                  a_grid_desc,
                                                  b_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc;
    ignore = b_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx11__))
}

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename CDataType,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDesc_M_N,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool AEnableLds,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BEnableLds,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          index_t NumGemmKPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemm_Wmma
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // FIX ME: To be deprecated
    static constexpr auto K1 = Number<K1Value>{};

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);
    static constexpr auto WmmaK  = K1 == 16 ? 32 : 16;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe =
        remove_cvref_t<decltype(GridwiseGemmPipeline_Selector<PipelineVer,
                                                              NumGemmKPrefetchStage,
                                                              LoopSched,
                                                              AEnableLds,
                                                              BEnableLds>())>;

    // Describe how data store to (LDS/VGPR) buffer from Global memory
    __host__ __device__ static constexpr auto MakeABlockDescriptor()
    {
        constexpr auto a_block_desc = [&]() {
            if constexpr(AEnableLds)
            {
                // K0->M->K1 Per Block
                constexpr auto K0PerBlock    = KPerBlock / K1;
                constexpr auto max_lds_align = K1;

                if constexpr(ABlockLdsExtraM)
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                        make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
                }
                else
                {
                    return make_naive_tensor_descriptor_aligned(
                        make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
                }
            }
            else
            {
                constexpr auto A_KRow        = I2;
                constexpr auto KWmmaPerblock = KPerBlock / WmmaK;
                constexpr auto K0PerWmma     = WmmaK / A_KRow / K1;
                // KWmma->MRepeat->MWave->K0PerWmma->KRow->MPerWmma->K1 Per Thread
                return make_naive_tensor_descriptor(
                    make_tuple(Number<KWmmaPerblock>{},
                               Number<MRepeat>{},
                               I1,
                               Number<K0PerWmma>{},
                               I1,
                               I1,
                               K1),
                    make_tuple(Number<MRepeat>{} * Number<K0PerWmma>{} * K1,
                               Number<K0PerWmma>{} * K1,
                               Number<K0PerWmma>{} * K1,
                               K1,
                               K1,
                               K1,
                               I1));
            }
        }();

        return a_block_desc;
    }

    __host__ __device__ static constexpr auto MakeBBlockDescriptor()
    {
        constexpr auto b_block_desc = [&]() {
            if constexpr(BEnableLds)
            {
                // K0->N->K1 Per Block
                constexpr auto K0PerBlock    = KPerBlock / K1;
                constexpr auto max_lds_align = K1;

                if constexpr(BBlockLdsExtraN)
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                        make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
                }
                else
                {
                    return make_naive_tensor_descriptor_aligned(
                        make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
                }
            }
            else
            {

                constexpr auto B_KRow        = I2;
                constexpr auto KWmmaPerblock = KPerBlock / WmmaK;
                constexpr auto K0PerWmma     = WmmaK / B_KRow / K1;
                // KWmma->NRepeat->MWave->K0PerWmma->KRow->MPerWmma->K1 Per Thread
                return make_naive_tensor_descriptor(
                    make_tuple(Number<KWmmaPerblock>{},
                               Number<NRepeat>{},
                               I1,
                               Number<K0PerWmma>{},
                               I1,
                               I1,
                               K1),
                    make_tuple(Number<NRepeat>{} * Number<K0PerWmma>{} * K1,
                               Number<K0PerWmma>{} * K1,
                               Number<K0PerWmma>{} * K1,
                               K1,
                               K1,
                               K1,
                               I1));
            }
        }();

        return b_block_desc;
    }

    __host__ __device__ static constexpr auto MakeABlockSliceCopyStep()
    {
        constexpr auto a_block_copy_step = [&]() {
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = KPerBlock / K1;

                return make_multi_index(K0PerBlock, 0, 0);
            }
            else
            {
                constexpr auto KWmmaPerBlock = KPerBlock / WmmaK;

                return make_multi_index(KWmmaPerBlock, 0, 0, 0, 0, 0, 0);
            }
        }();

        return a_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeBBlockSliceCopyStep()
    {
        constexpr auto b_block_copy_step = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto K0PerBlock = KPerBlock / K1;

                return make_multi_index(K0PerBlock, 0, 0);
            }
            else
            {
                constexpr auto KWmmaPerBlock = KPerBlock / WmmaK;

                return make_multi_index(KWmmaPerBlock, 0, 0, 0, 0, 0, 0);
            }
        }();

        return b_block_copy_step;
    }

    // Describe how data read from (LDS/VGPR) buffer
    template <typename ABlockDesc_>
    __host__ __device__ static constexpr auto MakeAWaveDescriptor(const ABlockDesc_&)
    {

        constexpr auto a_wave_desc = [&]() {
            if constexpr(AEnableLds)
            {
                // AK0_M_AK1 -> AK0_MRepeat_Mwaves_AKRow_MPerWmma_AK1
                constexpr auto A_K0 = ABlockDesc_{}.GetLength(I0);
                constexpr auto A_K1 = ABlockDesc_{}.GetLength(I2);
#ifdef __gfx12__
                constexpr auto A_KRow = I2;
#else
                constexpr auto A_KRow = I1;
#endif

                return transform_tensor_descriptor(
                    ABlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(Number<A_K0 / A_KRow>{}, A_KRow)),
                               make_unmerge_transform(make_tuple(
                                   Number<MRepeat>{}, Number<MWaves>{}, Number<MPerWmma>{})),
                               make_pass_through_transform(Number<A_K1>{})),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3>{}, Sequence<1, 2, 4>{}, Sequence<5>{}));
            }
            else
            {
                // KWmma_MRepeat_MWave_K0PerWmma_KRow_MPerWmma_K1 -> K0_MRepeat_Mwaves_MPerWmma_K1
                constexpr auto KWmma     = ABlockDesc_{}.GetLength(I0);
                constexpr auto K0PerWmma = ABlockDesc_{}.GetLength(I3);
                constexpr auto A_KRow    = ABlockDesc_{}.GetLength(I4);
                constexpr auto A_K1      = ABlockDesc_{}.GetLength(I6);

                // Err: merge transform cause non-constexpr issue

                // return transform_tensor_descriptor(
                //     ABlockDesc_{},
                //     make_tuple(make_merge_transform(make_tuple(Number<KWmma>{}, I1)),
                //                make_pass_through_transform(Number<MRepeat>{}),
                //                make_pass_through_transform(I1),
                //                make_pass_through_transform(I1),
                //                make_pass_through_transform(Number<A_K1>{})),
                //     make_tuple(Sequence<0, 3>{},
                //                Sequence<1>{},
                //                Sequence<2>{},
                //                Sequence<4>{},
                //                Sequence<5>{}),
                //     make_tuple(
                //         Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{},
                //         Sequence<4>{}));

                // Workaround, Freeze transform
                return make_naive_tensor_descriptor_packed(make_tuple(Number<KWmma * K0PerWmma>{},
                                                                      Number<MRepeat>{},
                                                                      I1,
                                                                      Number<A_KRow>{},
                                                                      I1,
                                                                      Number<A_K1>{}));
            }
        }();

        return a_wave_desc;
    }

    template <typename BBlockDesc_>
    __host__ __device__ static constexpr auto MakeBWaveDescriptor(const BBlockDesc_&)
    {
        constexpr auto b_wave_desc = [&]() {
            if constexpr(BEnableLds)
            {
                // BK0_N_BK1 -> BK0_NRepeat_Nwaves_NPerWmma_BK1
                constexpr auto B_K0 = BBlockDesc_{}.GetLength(I0);
                constexpr auto B_K1 = BBlockDesc_{}.GetLength(I2);
#ifdef __gfx12__
                constexpr auto B_KRow = I2;
#else
                constexpr auto B_KRow = I1;
#endif
                return transform_tensor_descriptor(
                    BBlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(Number<B_K0 / B_KRow>{}, B_KRow)),
                               make_unmerge_transform(make_tuple(
                                   Number<NRepeat>{}, Number<NWaves>{}, Number<NPerWmma>{})),
                               make_pass_through_transform(Number<B_K1>{})),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3>{}, Sequence<1, 2, 4>{}, Sequence<5>{}));
            }
            else
            {
                // KWmma_MRepeat_MWave_K0PerWmma_KRow_MPerWmma_K1 -> K0_MRepeat_Mwaves_MPerWmma_K1
                constexpr auto KWmma     = BBlockDesc_{}.GetLength(I0);
                constexpr auto K0PerWmma = BBlockDesc_{}.GetLength(I3);
                constexpr auto B_KRow    = BBlockDesc_{}.GetLength(I4);
                constexpr auto B_K1      = BBlockDesc_{}.GetLength(I6);

                // Workaround, Freeze transform
                return make_naive_tensor_descriptor_packed(make_tuple(Number<KWmma * K0PerWmma>{},
                                                                      Number<NRepeat>{},
                                                                      I1,
                                                                      Number<B_KRow>{},
                                                                      I1,
                                                                      Number<B_K1>{}));
            }
        }();

        return b_wave_desc;
    }

    __host__ __device__ static constexpr auto
    // *Caution Here repeat is shuffle repeat
    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
    {
        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMRepeatPerShuffle * MWaves * MPerWmma>{},
                           I1,
                           Number<CShuffleNRepeatPerShuffle * NWaves * NPerWmma>{}));

        return c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AGridDesc& a_grid_desc,
                                                            const BGridDesc& b_grid_desc,
                                                            const CGridDesc_M_N& c_grid_desc_m_n,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
                          (NPerBlock % (NRepeat * NPerWmma)) == 0,
                      "Invalid tuning param!");

        const auto GetAProblemsizeMK = [&]() {
            if constexpr(AEnableLds)
            {
                return make_tuple(a_grid_desc.GetLength(I1),
                                  a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I2));
            }
            else
            {
                return make_tuple(a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I2) *
                                      a_grid_desc.GetLength(I5),
                                  a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I3) *
                                      a_grid_desc.GetLength(I4) * a_grid_desc.GetLength(I6));
            }
        };

        const auto GetBProblemsizeNK = [&]() {
            if constexpr(BEnableLds)
            {
                return make_tuple(b_grid_desc.GetLength(I1),
                                  b_grid_desc.GetLength(I0) * b_grid_desc.GetLength(I2));
            }
            else
            {
                return make_tuple(b_grid_desc.GetLength(I1) * b_grid_desc.GetLength(I2) *
                                      b_grid_desc.GetLength(I5),
                                  b_grid_desc.GetLength(I0) * b_grid_desc.GetLength(I3) *
                                      b_grid_desc.GetLength(I4) * b_grid_desc.GetLength(I6));
            }
        };

        const auto M = GetAProblemsizeMK()[I0];
        const auto N = GetBProblemsizeNK()[I0];
        const auto K = GetAProblemsizeMK()[I1];

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K == GetBProblemsizeNK()[I1]))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("A: MxK = %d x %d, B: NxK = %d x %d, C: MxN = %d x %d\n",
                       GetAProblemsizeMK()[I0],
                       GetAProblemsizeMK()[I1],
                       GetBProblemsizeNK()[I0],
                       GetBProblemsizeNK()[I1],
                       c_grid_desc_m_n.GetLength(I0),
                       c_grid_desc_m_n.GetLength(I1));
                printf("GridwiseOp err: ProblemSize check");
            }
            return false;
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("GridwiseOp err: ProblemSize division");
            }
            return false;
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = K / KPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("GridwiseOp err: Pipeline not support this k_loop");
            }
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
             b_grid_desc.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB))
        {
            return false;
        }
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const CGridDesc_M_N& c_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment

        static constexpr auto max_lds_align = K1;

        static constexpr auto a_block_space_size_aligned =
            AEnableLds ? math::integer_least_multiple(MakeABlockDescriptor().GetElementSpaceSize(),
                                                      max_lds_align)
                       : 0;
        static constexpr auto b_block_space_size_aligned =
            BEnableLds ? math::integer_least_multiple(MakeBBlockDescriptor().GetElementSpaceSize(),
                                                      max_lds_align)
                       : 0;

        static constexpr auto a_block_space_offset = 0;
        static constexpr auto b_block_space_offset = a_block_space_size_aligned;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_space_size =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
                .GetElementSpaceSize();

        static constexpr auto c_shuffle_block_space_offset = 0;

        static constexpr auto lds_size =
            math::max(c_shuffle_block_space_size * sizeof(CShuffleDataType),
                      a_block_space_size_aligned * sizeof(ADataType) +
                          b_block_space_size_aligned * sizeof(BDataType));
    };

    using CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}))>;
    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1))>;

    template <bool HasMainKBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const AGridDesc& a_grid_desc,
                               const BGridDesc& b_grid_desc,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CElementwiseOperation& c_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
        // clang-format off
/*******************************************************************************/
// Memory buffer zone.
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

/*******************************************************************************/
// BlockIdx.x -> [BlockId.m, BlockId.n]
        const auto block_work_idx = block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        { return; }

        // Store BlockId into SGPR
        const index_t m_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

/*******************************************************************************/
// BlockLevel, A/B Matrix ThreadMapping in WMMA Source buffer, As Destinaion of BlockWise_Copy
        const auto K = [&](){
            if constexpr(AEnableLds){
                return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I2);
            }
            else{
                return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I3) 
                        * a_grid_desc.GetLength(I4) * a_grid_desc.GetLength(I6);
            }
        }();

        constexpr auto a_block_desc = MakeABlockDescriptor();
        constexpr auto b_block_desc = MakeBBlockDescriptor();
        
        auto a_block_trait = [&](){
            // A matrix blockwise copy
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = KPerBlock/ K1;
                auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<ADataType*>(p_shared), 
                    SharedMemTrait::a_block_space_size_aligned);        

                auto a_blockwise_copy =
                    ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
/* typename SrcElementwiseOperation,              */    AElementwiseOperation,
/* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */    Sequence<K0PerBlock, MPerBlock, K1>,
/* typename ThreadClusterLengths,                 */    ABlockTransferThreadClusterLengths_K0_M_K1,
/* typename ThreadClusterArrangeOrder,            */    ABlockTransferThreadClusterArrangeOrder,
/* typename SrcData,                              */    ADataType,
/* typename DstData,                              */    ADataType,
/* typename SrcDesc,                              */    decltype(a_grid_desc),
/* typename DstDesc,                              */    decltype(a_block_desc),
/* typename SrcDimAccessOrder,                    */    ABlockTransferSrcAccessOrder,
/* typename DstDimAccessOrder,                    */    Sequence<0, 1, 2>,
/* index_t SrcVectorDim,                          */    ABlockTransferSrcVectorDim,
/* index_t DstVectorDim,                          */    2,
/* index_t SrcScalarPerVector,                    */    ABlockTransferSrcScalarPerVector,
/* index_t DstScalarPerVector,                    */    ABlockTransferDstScalarPerVector_K1,
/* index_t SrcScalarStrideInVector,               */    1,
/* index_t DstScalarStrideInVector,               */    1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */    true,
                                                        NumGemmKPrefetchStage>(
                a_grid_desc,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

                return make_tuple(a_block_buf, a_blockwise_copy);
            }
            else
            {
                // Thread-wise copy
                // KPerBlock/WmmaK -> MRepeat -> MWaves -> K0PerWmma -> KRow -> MPerWmma -> K1
                constexpr auto KWmmaPerBlock = KPerBlock / WmmaK;
                constexpr auto K0PerWmma     = WmmaK/2/K1Value;
                auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ADataType>(
                    a_block_desc.GetElementSpaceSize());
                
                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto a_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<ADataType,
                                                     ADataType,
                                                     decltype(a_grid_desc),
                                                     decltype(a_block_desc),
                                                     Sequence<Number<KWmmaPerBlock>{},
                                                              Number<MRepeat>{},
                                                              I1,
                                                              Number<K0PerWmma>{},
                                                              I1,
                                                              I1,
                                                              Number<K1Value>{}>,
                                                     Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                     6,
                                                     ABlockTransferSrcScalarPerVector,
                                                     AThreadTransferSrcResetCoordinateAfterRun,
                                                     true>(
                    a_grid_desc,
                    make_multi_index(0, 
                                     m_block_data_idx_on_grid/(MWaves * MPerWmma), 
                                     get_thread_local_1d_id() / 32,
                                     0,
                                     (get_thread_local_1d_id() % 32 )/ 16, 
                                     get_thread_local_1d_id() % 16,
                                     0));
                                
                return make_tuple(a_block_buf, a_blockwise_copy);
            }
        };

        auto b_block_trait = [&](){
            if constexpr(BEnableLds)
            {
                constexpr auto K0PerBlock = KPerBlock/ K1;
                auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<BDataType*>(p_shared) + SharedMemTrait::b_block_space_offset, 
                    SharedMemTrait::b_block_space_size_aligned);

                auto b_blockwise_copy =
                    ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                        BElementwiseOperation,
                                                        ck::tensor_operation::element_wise::PassThrough,
                                                        InMemoryDataOperationEnum::Set,
                                                        Sequence<K0PerBlock, NPerBlock, K1>,
                                                        BBlockTransferThreadClusterLengths_K0_N_K1,
                                                        BBlockTransferThreadClusterArrangeOrder,
                                                        BDataType,
                                                        BDataType,
                                                        decltype(b_grid_desc),
                                                        decltype(b_block_desc),
                                                        BBlockTransferSrcAccessOrder,
                                                        Sequence<0, 1, 2>,
                                                        BBlockTransferSrcVectorDim,
                                                        2,
                                                        BBlockTransferSrcScalarPerVector,
                                                        BBlockTransferDstScalarPerVector_K1,
                                                        1,
                                                        1,
                                                        BThreadTransferSrcResetCoordinateAfterRun,
                                                        true,
                                                        NumGemmKPrefetchStage>(
                    b_grid_desc,
                    make_multi_index(0, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});
                
                return make_tuple(b_block_buf, b_blockwise_copy);
            }
            else
            {
                // Thread-wise copy
                // KPerBlock/WmmaK -> NRepeat -> NWaves -> WmmaK/K1 -> NPerWmma -> K1
                constexpr auto KWmmaPerBlock = KPerBlock / WmmaK;
                constexpr auto K0PerWmma     = WmmaK/2/K1Value;
                auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
                    b_block_desc.GetElementSpaceSize());
                
                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto b_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<BDataType,
                                                     BDataType,
                                                     decltype(b_grid_desc),
                                                     decltype(b_block_desc),
                                                     Sequence<Number<KWmmaPerBlock>{},
                                                              Number<NRepeat>{},
                                                              I1,
                                                              Number<K0PerWmma>{},
                                                              I1,
                                                              I1,
                                                              Number<K1Value>{}>,
                                                     Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                     6,
                                                     BBlockTransferSrcScalarPerVector,
                                                     BThreadTransferSrcResetCoordinateAfterRun,
                                                     true>(
                    b_grid_desc,
                    make_multi_index(0, 
                                     n_block_data_idx_on_grid/(NWaves * NPerWmma), 
                                     get_thread_local_1d_id() / 32,
                                     0,
                                     (get_thread_local_1d_id() % 32 )/ 16, 
                                     get_thread_local_1d_id() % 16,
                                     0));
                                
                return make_tuple(b_block_buf, b_blockwise_copy);
            }
        };

        auto a_block_buf       = a_block_trait()[I0];
        auto a_blockwise_copy  = a_block_trait()[I1];

        auto b_block_buf       = b_block_trait()[I0];
        auto b_blockwise_copy  = b_block_trait()[I1];
/*******************************************************************************/
        // GEMM
        constexpr auto KPack = math::integer_least_multiple(K1, WmmaK);

        auto blockwise_gemm =
            BlockwiseGemmWMMA<BlockSize,
                              ADataType,
                              BDataType,
                              AccDataType,
                              decltype(MakeAWaveDescriptor(a_block_desc)),
                              decltype(MakeBWaveDescriptor(b_block_desc)),
                              MPerBlock,
                              NPerBlock,
                              KPerBlock,
                              MPerWmma,
                              NPerWmma,
                              MRepeat,
                              NRepeat,
                              KPack,
                              AEnableLds,
                              BEnableLds>{};

        // Prepare Register for C matrix
        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

/*******************************************************************************/        
        // Shift Per SUB_K
        constexpr auto a_block_slice_copy_step = MakeABlockSliceCopyStep();
        constexpr auto b_block_slice_copy_step = MakeBBlockSliceCopyStep();

        // gridwise GEMM pipeline
        const index_t KBlockMainLoop = __builtin_amdgcn_readfirstlane(K / KPerBlock);
        GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc,
                                                          a_block_desc,
                                                          a_blockwise_copy,
                                                          a_grid_buf,
                                                          a_block_buf,
                                                          a_block_slice_copy_step,
                                                          b_grid_desc,
                                                          b_block_desc,
                                                          b_blockwise_copy,
                                                          b_grid_buf,
                                                          b_block_buf,
                                                          b_block_slice_copy_step,
                                                          blockwise_gemm,
                                                          c_thread_buf,
                                                          KBlockMainLoop);
/*******************************************************************************/
        // write out to C, implement shuffle
        {
            // C mapping in single thread.
            constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =  
            blockwise_gemm.GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

            // C mapping in single block
            constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
                blockwise_gemm.GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

            constexpr auto MWave              = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I1);
            constexpr auto MSubGroup          = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I2);
            constexpr auto NWave              = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I4);
            constexpr auto NThreadPerSubGroup = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I5);
            constexpr auto MAccVgprs          = c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp.GetLength(I6);

            // LDS descriptor, shuffle and write out in MRepeat x NRepeat times
            constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
                GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<CShuffleDataType*>(p_shared) + SharedMemTrait::c_shuffle_block_space_offset, 
                SharedMemTrait::c_shuffle_block_space_size);

            constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs = transform_tensor_descriptor(
                c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMRepeatPerShuffle>{}, // MRepeat per shuffle repeat
                        MWave,                               // MWave
                        MSubGroup,                           // MSubGroup * MAccVgprs = MPerWmma
                        MAccVgprs)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNRepeatPerShuffle>{}, // NRepeat per shuffle repeat
                        NWave,                               // NWave
                        NThreadPerSubGroup))),               // NThreadPerSubGroup = NPerWmma
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<>{}, Sequence<0, 1, 2, 6>{}, Sequence<>{}, Sequence<3, 4, 5>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block = blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(MRepeat, MWave, MSubGroup, MAccVgprs))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(NRepeat, NWave, NThreadPerSubGroup))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));
            
            const auto m_thread_data_on_block_idx = m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor.CalculateBottomIndex(
                make_multi_index(m_thread_data_on_block));
            
            const auto n_thread_data_on_block_idx = n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   CShuffleDataType,
                                                   decltype(c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
                                                   decltype(c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            CShuffleNRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            MAccVgprs>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                   6,
                                                   1, // vector write pixel
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                    make_multi_index(0,
                                     m_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     0,
                                     n_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                CShuffleDataType,        // typename SrcData,
                CDataType,               // typename DstData,
                decltype(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for local reg & global memory
            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MRepeat, 1, 1, NRepeat, 1, 1, MAccVgprs>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6>,
                                  Sequence<CShuffleMRepeatPerShuffle,
                                           1,
                                           1,
                                           CShuffleNRepeatPerShuffle,
                                           1,
                                           1,
                                           MAccVgprs>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                                           1,
                                           CShuffleNRepeatPerShuffle * NWave * NPerWmma>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
        // clang-format on
    }
};

} // namespace ck
