// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_waveletmodel.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename ABDataType,
          typename FloatGemmAcc,
          typename EDataTypeShuffle,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename EElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_M_K,
          typename BGridDesc_N_K,
          typename EGridDesc_M_N,
          index_t NumGemmKPrefetchStage,
          index_t TileLoadThreadGroupSize,
          index_t TileMathThreadGroupSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdl_waveletmodel_cshuffle
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto AK1         = Number<AK1Value>{};
    static constexpr auto BK1         = Number<BK1Value>{};
    static constexpr auto AK0PerBlock = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0PerBlock = Number<KPerBlock / BK1Value>{};
    static constexpr auto BlockSize   = math::max(TileLoadThreadGroupSize, TileMathThreadGroupSize);

    struct TileLoadThreadGroup
    {
        __device__ static constexpr index_t GetNumOfThread() { return TileLoadThreadGroupSize; }

        __device__ static constexpr bool IsBelong()
        {
            return (get_thread_local_1d_id() >= TileLoadThreadGroupSize);
        }

        __device__ static index_t GetThreadId()
        {
            return get_thread_local_1d_id() - TileMathThreadGroupSize;
        }
    };

    struct TileMathThreadGroup
    {
        __device__ static constexpr index_t GetNumOfThread() { return TileMathThreadGroupSize; }

        __device__ static constexpr bool IsBelong()
        {
            return get_thread_local_1d_id() < TileMathThreadGroupSize;
        }

        __device__ static index_t GetThreadId() { return get_thread_local_1d_id(); }
    };

    using CShuffleBlockTransferThreadGroup = ThisThreadBlock<TileMathThreadGroupSize>;

    // load and math+store Wave pipelines.
    // TODO: build pipelines blocks scheduling parallel tasks
    using GridwiseGemmLoad = GridwiseGemmLoadWave<TileLoadThreadGroup, NumGemmKPrefetchStage>;
    using GridwiseGemmMath = GridwiseGemmMathWave<TileMathThreadGroup, NumGemmKPrefetchStage>;

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0PerBlock, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0PerBlock, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max((a_block_space_size_aligned + b_block_space_size_aligned) *
                             sizeof(ABDataType),
                         c_block_size * sizeof(EDataTypeShuffle));
    }

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        return ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            EDataType,
            CGlobalMemoryDataOperation>();
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2ETileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_M_K& a_grid_desc_m_k,
                  const BGridDesc_N_K& b_grid_desc_n_k,
                  const EGridDesc_M_N& e_grid_desc_m_n,
                  const Block2ETileMap& /*block_2_etile_map*/)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        // check consistency of desc
        if(!(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1) &&
             K == b_grid_desc_n_k.GetLength(I1)))
        {
            return false;
        }

        // check tile size
        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
        {
            return false;
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = K / KPerBlock;

        if(!GridwiseGemmMath::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)

        // check tensor size: cannot be larger than 2GB each
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc_m_k.GetElementSpaceSize() * sizeof(ABDataType) <= TwoGB &&
             b_grid_desc_n_k.GetElementSpaceSize() * sizeof(ABDataType) <= TwoGB &&
             e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmMath::CalculateHasMainLoop(num_loop);
    }

    // return block_id to E matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        constexpr auto M01 = I1;
        constexpr auto N01 = I1;

        const auto m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(M0, M01)),
                           make_unmerge_transform(make_tuple(N0, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

        const auto cblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, N0, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const index_t grid_size = (M / MPerBlock) * (N / NPerBlock);

        return grid_size;
    }

    // A desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // B desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultBGridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // E desc for destination in blockwise copy
    template <typename EGridDescriptor_M_N>
    __host__ __device__ static constexpr auto MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const EGridDescriptor_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    using EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    using DefaultBlock2ETileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    template <bool HasMainKBlockLoop,
              typename AGridDesc_AK0_M_AK1,
              typename BGridDesc_BK0_N_BK1,
              typename Block2ETileMap>
    __device__ static void Run(const ABDataType* __restrict__ p_a_grid,
                               const ABDataType* __restrict__ p_b_grid,
                               EDataType* __restrict__ p_e_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const EElementwiseOperation& e_element_op,
                               const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                               const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2ETileMap& block_2_etile_map)
    {
        // build loadWave and MathWave pipelines
        // loadWave and MathWave synchronized through LDS

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ABDataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ABDataType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        if(TileLoadThreadGroup::IsBelong())
        {

            // LoadWave
            const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
            const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

            // A matrix blockwise copy
            auto a_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<TileLoadThreadGroup,
                                                    AElementwiseOperation,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<AK0PerBlock, MPerBlock, AK1>,
                                                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                    ABlockTransferThreadClusterArrangeOrder,
                                                    ABDataType,
                                                    ABDataType,
                                                    decltype(a_grid_desc_ak0_m_ak1),
                                                    decltype(a_block_desc_ak0_m_ak1),
                                                    ABlockTransferSrcAccessOrder,
                                                    Sequence<1, 0, 2>,
                                                    ABlockTransferSrcVectorDim,
                                                    2,
                                                    ABlockTransferSrcScalarPerVector,
                                                    ABlockTransferDstScalarPerVector_AK1,
                                                    1,
                                                    1,
                                                    AThreadTransferSrcResetCoordinateAfterRun,
                                                    true,
                                                    NumGemmKPrefetchStage>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(0, m_block_data_idx_on_grid, 0),
                    a_element_op,
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

            // B matrix blockwise copy
            auto b_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<TileLoadThreadGroup,
                                                    BElementwiseOperation,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<BK0PerBlock, NPerBlock, BK1>,
                                                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                    BBlockTransferThreadClusterArrangeOrder,
                                                    ABDataType,
                                                    ABDataType,
                                                    decltype(b_grid_desc_bk0_n_bk1),
                                                    decltype(b_block_desc_bk0_n_bk1),
                                                    BBlockTransferSrcAccessOrder,
                                                    Sequence<1, 0, 2>,
                                                    BBlockTransferSrcVectorDim,
                                                    2,
                                                    BBlockTransferSrcScalarPerVector,
                                                    BBlockTransferDstScalarPerVector_BK1,
                                                    1,
                                                    1,
                                                    BThreadTransferSrcResetCoordinateAfterRun,
                                                    true,
                                                    NumGemmKPrefetchStage>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(0, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

            GridwiseGemmLoad::template RunLoadWavePipeline<HasMainKBlockLoop>(
                a_grid_desc_ak0_m_ak1,
                a_block_desc_ak0_m_ak1,
                a_blockwise_copy,
                a_grid_buf,
                a_block_buf,
                a_block_slice_copy_step,
                b_grid_desc_bk0_n_bk1,
                b_block_desc_bk0_n_bk1,
                b_blockwise_copy,
                b_grid_buf,
                b_block_buf,
                b_block_slice_copy_step,
                num_k_block_main_loop);

            block_sync_lds();
            block_sync_lds();
        }
        else if(TileMathThreadGroup::IsBelong())
        {
            // branch early for math wave
            constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
            constexpr bool is_single_rate_mfma =
                (((is_same<ABDataType, half_t>::value || is_same<ABDataType, bhalf_t>::value) &&
                  lcm_AK1_BK1 <= 4) ||
                 (is_same<ABDataType, int8_t>::value && lcm_AK1_BK1 <= 8) ||
                 ((is_same<ABDataType, f8_t>::value || is_same<ABDataType, bf8_t>::value) &&
                  lcm_AK1_BK1 < 32))
                    ? true
                    : false;
            constexpr auto is_scale_mfma = false;
            constexpr index_t KPack =
                math::max(lcm_AK1_BK1,
                          MfmaSelector<ABDataType,
                                       MPerXdl,
                                       NPerXdl,
                                       ABDataType,
                                       is_single_rate_mfma,
                                       is_scale_mfma>::selected_mfma.k_per_blk);

            auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<
                TileMathThreadGroupSize,
                ABDataType,
                ABDataType,
                FloatGemmAcc,
                decltype(a_block_desc_ak0_m_ak1),
                decltype(b_block_desc_bk0_n_bk1),
                MPerXdl,
                NPerXdl,
                MXdlPerWave,
                NXdlPerWave,
                KPack>{};

            auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();
            auto c_grid_buf   = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            // TODO re-architect LDS+math stages
            // Writing data to GMEM: only math wave is doing the work in cshuffle
            GridwiseGemmMath::template RunMathWavePipeline<HasMainKBlockLoop>(
                a_block_buf, b_block_buf, blockwise_gemm, c_thread_buf, num_k_block_main_loop);

            // GEMM definition
            //   c_mtx += transpose(a_mtx) * b_mtx
            //     a_mtx[K0PerBlock, MPerBlock] is in LDS
            //     b_mtx[K0PerBlock, NPerBlock] is in LDS
            //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
            //       register
            // sanity check

            // shuffle C and write out
            {
                static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                                  NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                              "wrong!");

                constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
                constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

                // TODO: hacky, fix it!
                constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                    blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                // TODO: hacky, fix it!
                // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                    blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
                constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
                constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

                constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

                auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<EDataTypeShuffle*>(p_shared),
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    make_tuple(
                        make_freeze_transform(I0),
                        make_unmerge_transform(make_tuple(
                            Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                            M1,                                      // M1 = MWave
                            M2,                                      // M2 * M3 * M4 = MPerXdl
                            M3,
                            M4)),
                        make_freeze_transform(I0),
                        make_unmerge_transform(make_tuple(
                            Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                            N1,                                      // N1 = NWave
                            N2))),                                   // N2 = NPerXdl
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<>{},
                               Sequence<0, 2, 4, 5, 6>{},
                               Sequence<>{},
                               Sequence<1, 3, 7>{}));

                // calculate origin of thread output tensor on global memory
                // blockwise GEMM c matrix starting index
                const auto c_thread_mtx_on_block =
                    blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

                const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
                const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

                const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                        make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                        make_tuple(Sequence<0>{}));

                const auto m_thread_data_on_block_idx =
                    m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                        make_multi_index(m_thread_data_on_block));

                const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                        make_tuple(Sequence<0, 1, 2>{}),
                        make_tuple(Sequence<0>{}));

                const auto n_thread_data_on_block_idx =
                    n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                        make_multi_index(n_thread_data_on_block));

                // shuffle: threadwise copy C from VGPR to LDS
                auto c_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
                    FloatGemmAcc,
                    EDataTypeShuffle,
                    decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                    decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                    ck::tensor_operation::element_wise::PassThrough,
                    Sequence<CShuffleMXdlPerWavePerShuffle,
                             CShuffleNXdlPerWavePerShuffle,
                             I1,
                             I1,
                             M2,
                             I1,
                             M4,
                             I1>,
                    Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                    7,
                    1,
                    InMemoryDataOperationEnum::Set,
                    1,
                    true>{c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                          make_multi_index(0,
                                           0,
                                           m_thread_data_on_block_idx[I1],
                                           n_thread_data_on_block_idx[I1],
                                           m_thread_data_on_block_idx[I2],
                                           m_thread_data_on_block_idx[I3],
                                           m_thread_data_on_block_idx[I4],
                                           n_thread_data_on_block_idx[I2]),
                          ck::tensor_operation::element_wise::PassThrough{}};

                // shuffle: blockwise copy C from LDS to global
                auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                    CShuffleBlockTransferThreadGroup, // ThreadGroup
                    EElementwiseOperation,            // ElementwiseOperation,
                    CGlobalMemoryDataOperation,       // DstInMemOp,
                    Sequence<1,
                             CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                             1,
                             CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    EDataTypeShuffle,     // typename SrcData,
                    EDataType,            // typename DstData,
                    decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                    decltype(e_grid_desc_mblock_mperblock_nblock_nperblock),
                    Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                    3,                                              // index_t VectorDim,
                    CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                    true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                    false> // bool ThreadTransferDstResetCoordinateAfterRun>
                    {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                     make_multi_index(0, 0, 0, 0),
                     e_grid_desc_mblock_mperblock_nblock_nperblock,
                     make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0),
                     e_element_op};

                // space filling curve for threadwise C in VGPR
                constexpr auto sfc_c_vgpr =
                    SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                      Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                      Sequence<CShuffleMXdlPerWavePerShuffle,
                                               CShuffleNXdlPerWavePerShuffle,
                                               1,
                                               1,
                                               M2,
                                               1,
                                               M4,
                                               1>>{};

                // space filling curve for shuffled blockwise C in global mem
                constexpr auto sfc_c_global =
                    SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                      Sequence<0, 2, 1, 3>,
                                      Sequence<1,
                                               CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                               1,
                                               CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

                constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

                static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

                // Different way of getting coalesced writes:
                // We can get rid of doing cshuffle. Instead of reading A rows in contiguous manner
                // do it interleaved, then mfma can have nice c-mat layout as below:
                //
                // TODO
                //      We do not need to do LDS swizzle to align global writes writing cache lines:
                //         v_mfma  cmat, amat, bmat, cmat   - c-mat register layout   are 1xN
                //                                            elments  (N is vertical or strided
                //                                            dimension)
                //         v_mfma  cmat, bmat, amat, cmat   - c-mat register layout   are Mx1
                //         elments  (M is coalescing
                //                                            dimension) by enumerating M index in
                //                                            amat, bmat you can align cmat
                //                                            register(s) to contiguous M elements
                //                                            for example
                //              1st mfma instruction  output space : 0 4 8  12 16 ....
                //              2nd mfma instruction  output space : 1 5 9  13 17 ....
                //              3rd mfma instruction  output space : 2 6 10 14 18 ....
                //              4th mfma instruction  output space : 3 7 11 15 19 ....
                //              you can pack 4 registers output space into 2WORD and do global write
                //              (no LDS swizzling required)

                static_for<0, num_access, 1>{}([&](auto access_id) {
                    // make sure it's safe to write to LDS
                    block_sync_lds();

                    // each thread write its data from VGPR to LDS
                    c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                  sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                                  c_thread_buf,
                                                  c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                  c_shuffle_block_buf);
                    // make sure it's safe to read from LDS
                    block_sync_lds();

                    // each block copy its data from LDS to global
                    c_shuffle_block_copy_lds_to_global.Run(
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                        c_shuffle_block_buf,
                        e_grid_desc_mblock_mperblock_nblock_nperblock,
                        c_grid_buf);

                    if constexpr(access_id < num_access - 1)
                    {
                        constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                        // move on C
                        c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                            e_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                    }
                });
            }
        }
    }
};

} // namespace ck
