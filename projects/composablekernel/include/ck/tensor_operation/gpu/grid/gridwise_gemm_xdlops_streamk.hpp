// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1r2.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v3.hpp"
#include "ck/utility/workgroup_barrier.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"

namespace ck {

template <typename GridwiseGemm>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdlops_streamk(const typename GridwiseGemm::FloatAB* p_a_grid,
                               const typename GridwiseGemm::FloatAB* p_b_grid,
                               typename GridwiseGemm::FloatC* p_c_grid,
                               void* p_workspace,
                               index_t M,
                               index_t N,
                               index_t K,
                               index_t StrideA,
                               index_t StrideB,
                               index_t StrideC,
                               typename GridwiseGemm::Block2CTileMap block_mapping)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();

        __shared__ uint8_t p_shared[shared_size];

        GridwiseGemm::Run(p_a_grid,
                          p_b_grid,
                          p_c_grid,
                          p_workspace,
                          M,
                          N,
                          K,
                          StrideA,
                          StrideB,
                          StrideC,
                          block_mapping,
                          static_cast<void*>(p_shared));
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = p_workspace;
    ignore = M;
    ignore = N;
    ignore = K;
    ignore = StrideA;
    ignore = StrideB;
    ignore = StrideC;
    ignore = block_mapping;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <index_t BlockSize,
          typename Block2CTileMap_,
          typename FloatAB_,
          typename FloatAcc_,
          typename FloatC_,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
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
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>
struct GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_streamk
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
    static constexpr auto K1        = Number<K1Value>{};
    static constexpr auto M01       = 1;
    static constexpr auto N01       = 1;
    static constexpr auto KPerBlock = K0PerBlock * K1;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;
    using FloatAcc        = FloatAcc_;
    using FloatCShuffle   = FloatAcc;

    using Block2CTileMap = Block2CTileMap_;
    using FloatAB        = FloatAB_;
    using FloatC         = FloatC_;

    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        const FloatAB* p_a_grid;
        const FloatAB* p_b_grid;
        FloatC* p_c_grid;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        Block2CTileMap block_mapping;

        Argument(const FloatAB* p_a_grid_,
                 const FloatAB* p_b_grid_,
                 FloatC* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 uint32_t num_cu,
                 uint32_t occupancy,
                 uint32_t num_sk_blocks_)
            : p_a_grid(p_a_grid_),
              p_b_grid(p_b_grid_),
              p_c_grid(p_c_grid_),
              M(M_),
              N(N_),
              K(K_),
              StrideA(StrideA_),
              StrideB(StrideB_),
              StrideC(StrideC_),
              block_mapping(M, N, K, num_cu, occupancy, num_sk_blocks_)
        {
        }

        void Print() const
        {
            std::cout << "arg {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << std::endl;
        }
    };

    __host__ __device__ static auto CalculateGridSize(const Argument& karg)
    {
        return std::make_tuple(math::integer_divide_ceil(karg.N, NPerBlock),
                               math::integer_divide_ceil(karg.M, MPerBlock),
                               karg.k_batch);
    }

    __host__ __device__ static auto CalculateK0(index_t KPad) { return KPad / K1; }

    __host__ __device__ static auto
    MakeAGridDescriptor_K0_M_K1(index_t M, index_t MPad, index_t K, index_t KPad, index_t StrideA)
    {
        const index_t K0 = CalculateK0(KPad);

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(a_grid_desc_m_kpad,
                                           make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                      make_right_pad_transform(M, MPad - M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    __host__ __device__ static auto
    MakeBGridDescriptor_K0_N_K1(index_t K, index_t KPad, index_t N, index_t NPad, index_t StrideB)
    {
        const index_t K0 = CalculateK0(KPad);

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
            b_grid_desc_k_n,
            make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(b_grid_desc_kpad_n,
                                           make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    __host__ __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        return transform_tensor_descriptor(c_grid_desc_m_n,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * K1, K1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * K1, K1, I1));
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = K1;

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_k0_m_k1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_k0_n_k1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        constexpr auto a_block_space_size_aligned =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned =
            math::integer_least_multiple(b_block_desc_k0_n_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto c_block_size =
            GetCBlockDescriptor_MBlock_MPerShuffle_NBlock_NPerShuffle().GetElementSpaceSize();

        return math::max((a_block_space_size_aligned + b_block_space_size_aligned) *
                             sizeof(FloatAB),
                         c_block_size * sizeof(FloatCShuffle));
    }

    static constexpr index_t MXdlPerWave = MRepeat;
    static constexpr index_t NXdlPerWave = NRepeat;
    IS_VALID_COMPILATION_PARAMETER_IMPL(FloatC)

    __host__ __device__ static constexpr bool CheckValidity(const Argument& karg)
    {
        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
                return false;
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
                return false;
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
                return false;
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
                return false;
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
                return false;
        }
        else
        {
            if(karg.M % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
                return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0)
    {
        const bool has_main_k0_block_loop = K0 > K0PerBlock;

        return has_main_k0_block_loop;
    }

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto
    MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        return transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    template <typename CGridDesc>
    __host__ __device__ static constexpr auto MakeCBlockClusterAdaptor(
        const CGridDesc& c_m_n_grid_desc, index_t /* M01 */, index_t /* N01 */, index_t KBatch)
    {
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc>(
            c_m_n_grid_desc, 8, KBatch);
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MBlock_MPerShuffle_NBlock_NPerShuffle()
    {
        constexpr index_t MWave = MPerBlock / (MRepeat * MPerXdl);
        constexpr index_t NWave = NRepeat * NPerXdl == 0 ? 1 : NPerBlock / (NRepeat * NPerXdl);

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1,
                       Number<CShuffleMRepeatPerShuffle * MWave * MPerXdl>{},
                       I1,
                       Number<CShuffleNRepeatPerShuffle * NWave * NPerXdl>{}));
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MShuffleRepeat_MPerShuffle_NShuffleRepeat_NPerShuffle()
    {
        constexpr index_t MWave = MPerBlock / (MRepeat * MPerXdl);
        constexpr index_t NWave = NRepeat * NPerXdl == 0 ? 1 : NPerBlock / (NRepeat * NPerXdl);

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat / CShuffleMRepeatPerShuffle>{},
                       Number<CShuffleMRepeatPerShuffle * MWave * MPerXdl>{},
                       Number<NRepeat / CShuffleNRepeatPerShuffle>{},
                       Number<CShuffleNRepeatPerShuffle * NWave * NPerXdl>{}));
    }

    __host__ __device__ static constexpr auto GetClusterLengthReduction()
    {
        // TODO: assume C is row major
        // TODO: we always first loop over N, then M
        constexpr auto NPerBlockPow2 = math::next_power_of_two<NPerBlock>();
        constexpr auto NPerBlockReduction =
            NPerBlockPow2 / CBlockTransferScalarPerVector_NWaveNPerXDL;
        constexpr auto MPerBlockReduction =
            (BlockSize + NPerBlockReduction - 1) / NPerBlockReduction;
        return Sequence<MPerBlockReduction, NPerBlockReduction>{};
    }

    __host__ __device__ static constexpr auto GetPartialAccBlockDescriptor()
    {
        const auto c_partial_acc_block_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MPerBlock, NPerBlock),
                                                    make_tuple(NPerBlock, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MPerBlock, NPerBlock),
                                                    make_tuple(I1, MPerBlock));
            }
        }();
        return c_partial_acc_block_m_n;
    }

    using CGridDesc_M_N = remove_cvref_t<decltype(MakeCGridDescriptor_M_N(1, 1, 1, 1, 1))>;

    __device__ static void Run(const FloatAB* p_a_grid,
                               const FloatAB* p_b_grid,
                               FloatC* p_c_grid,
                               void* p_workspace,
                               index_t M,
                               index_t N,
                               index_t K,
                               index_t StrideA,
                               index_t StrideB,
                               index_t StrideC,
                               Block2CTileMap block_mapping,
                               void* __restrict__ p_shared_block)
    {
        uint32_t m        = M;
        uint32_t n        = N;
        uint32_t k        = K;
        uint32_t pad_m    = (m + MPerBlock - 1) / MPerBlock * MPerBlock;
        uint32_t pad_n    = (n + NPerBlock - 1) / NPerBlock * NPerBlock;
        uint32_t pad_k    = (k + KPerBlock - 1) / KPerBlock * KPerBlock;
        uint32_t stride_a = StrideA;
        uint32_t stride_b = StrideB;
        uint32_t stride_c = StrideC;

        const auto a_k0_m_k1_grid_desc = MakeAGridDescriptor_K0_M_K1(m, pad_m, k, pad_k, stride_a);
        const auto b_k0_n_k1_grid_desc = MakeBGridDescriptor_K0_N_K1(k, pad_k, n, pad_n, stride_b);
        const auto c_grid_desc_m_n     = MakeCGridDescriptor_M_N(m, pad_m, n, pad_n, stride_c);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);
        const AElementwiseOperation a_element_op = AElementwiseOperation{};
        const BElementwiseOperation b_element_op = BElementwiseOperation{};
        const CElementwiseOperation c_element_op = CElementwiseOperation{};

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_k0_m_k1_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_k0_n_k1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0_m_k1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0_n_k1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        auto blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_k0_m_k1),
                                                                decltype(b_block_desc_k0_n_k1),
                                                                MPerXdl,
                                                                NPerXdl,
                                                                MRepeat,
                                                                NRepeat,
                                                                K1>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block = static_cast<FloatAB*>(p_shared_block);
        FloatAB* p_b_block = static_cast<FloatAB*>(p_shared_block) + a_block_space_size;

        constexpr auto a_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block, a_block_desc_k0_m_k1.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_b_block, b_block_desc_k0_n_k1.GetElementSpaceSize());

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline = GridwiseGemmPipeline_v3();

        uint32_t block_idx = block_mapping.get_block_idx();
        bool is_sk_block   = block_idx < block_mapping.sk_num_blocks;
        bool is_dp_block   = block_idx >= block_mapping.dp_start_block_idx &&
                           block_idx < block_mapping.reduction_start_block_idx;
        bool is_reduction_block = block_idx >= block_mapping.reduction_start_block_idx;
        bool is_padding_block   = block_idx >= block_mapping.sk_num_blocks &&
                                block_idx < block_mapping.dp_start_block_idx;
        uint32_t iter_start, iter_end;
        block_mapping.get_block_itr(block_idx, iter_start, iter_end);
        uint32_t total_iter_length = iter_end - iter_start;

        if(is_padding_block)
            return;

        uint32_t* p_semaphore =
            reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(p_workspace) +
                                        block_mapping.get_workspace_size_for_acc(sizeof(FloatAcc)));

        if constexpr(Block2CTileMap::ReductionStrategy == StreamKReductionStrategy::Reduction)
        {
            if(is_reduction_block)
            {
                // descriptors
                constexpr auto cluster_length_reduce = GetClusterLengthReduction();
                constexpr auto reduce_desc = make_cluster_descriptor(cluster_length_reduce);
                const auto reduce_thread_cluster_idx =
                    reduce_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));
                const auto thread_m_cluster_id = reduce_thread_cluster_idx[I0];
                const auto thread_n_cluster_id = reduce_thread_cluster_idx[I1];

                constexpr auto MReduceIters =
                    math::integer_divide_ceil(Number<MPerBlock>{}, cluster_length_reduce.At(I0));
                constexpr auto NReduceIters = math::integer_divide_ceil(
                    Number<NPerBlock>{},
                    cluster_length_reduce.At(I1) *
                        Number<CBlockTransferScalarPerVector_NWaveNPerXDL>{});

                constexpr auto acc_thread_buf_load_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(I1, Number<CBlockTransferScalarPerVector_NWaveNPerXDL>{}));
                constexpr auto acc_thread_buf_store_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(I1, I1, I1, Number<CBlockTransferScalarPerVector_NWaveNPerXDL>{}));

                constexpr auto c_partial_acc_block_m_n = GetPartialAccBlockDescriptor();

                constexpr auto partial_acc_load_step_n = make_multi_index(
                    0, cluster_length_reduce.At(I1) * CBlockTransferScalarPerVector_NWaveNPerXDL);
                constexpr auto partial_acc_load_step_n_reverse =
                    make_multi_index(0,
                                     -1 * cluster_length_reduce.At(I1).value * (NReduceIters - 1) *
                                         CBlockTransferScalarPerVector_NWaveNPerXDL);
                constexpr auto partial_acc_load_step_m =
                    make_multi_index(cluster_length_reduce.At(I0), 0);

                constexpr auto partial_acc_store_step_n = make_multi_index(
                    0,
                    0,
                    0,
                    cluster_length_reduce.At(I1) * CBlockTransferScalarPerVector_NWaveNPerXDL);
                constexpr auto partial_acc_store_step_n_reverse =
                    make_multi_index(0,
                                     0,
                                     0,
                                     -1 * cluster_length_reduce.At(I1).value * (NReduceIters - 1) *
                                         CBlockTransferScalarPerVector_NWaveNPerXDL);
                constexpr auto partial_acc_store_step_m =
                    make_multi_index(0, cluster_length_reduce.At(I0), 0, 0);

                StaticBuffer<AddressSpaceEnum::Vgpr,
                             FloatAcc,
                             CBlockTransferScalarPerVector_NWaveNPerXDL,
                             true>
                    parcial_acc_buf;
                StaticBuffer<AddressSpaceEnum::Vgpr,
                             FloatAcc,
                             CBlockTransferScalarPerVector_NWaveNPerXDL,
                             true>
                    acc_buf;

                // start to compute
                auto reduction_idx = blockIdx.x - block_mapping.reduction_start_block_idx;
                auto spatial_idx   = block_mapping.tile_to_spatial(reduction_idx, m, n);

                workgroup_barrier wg_barrier(p_semaphore);

                uint32_t tile_acc_offset_start =
                    block_mapping.get_acc_buffer_offset_from_tile(reduction_idx);
                uint32_t tile_acc_offset_end =
                    block_mapping.get_acc_buffer_offset_from_tile(reduction_idx + 1);

                auto acc_load = ThreadwiseTensorSliceTransfer_v2<
                    FloatAcc,                                                // SrcData,
                    FloatAcc,                                                // DstData,
                    decltype(c_partial_acc_block_m_n),                       // SrcDesc,
                    decltype(acc_thread_buf_load_desc),                      // DstDesc,
                    Sequence<1, CBlockTransferScalarPerVector_NWaveNPerXDL>, // SliceLengths,
                    Sequence<0, 1>,                                          // DimAccessOrder,
                    1,                                                       // SrcVectorDim,
                    CBlockTransferScalarPerVector_NWaveNPerXDL,              // SrcScalarPerVector,
                    1,    // SrcScalarStrideInVector,
                    false // SrcResetCoordinateAfterRun,
                    >{c_partial_acc_block_m_n,
                      make_multi_index(thread_m_cluster_id,
                                       thread_n_cluster_id *
                                           CBlockTransferScalarPerVector_NWaveNPerXDL)};

                auto acc_store = ThreadwiseTensorSliceTransfer_v1r3<
                    FloatAcc,                                                // SrcData,
                    FloatC,                                                  // DstData,
                    decltype(acc_thread_buf_store_desc),                     // SrcDesc,
                    decltype(c_grid_desc_mblock_mperblock_nblock_nperblock), // DstDesc,
                    CElementwiseOperation, // ElementwiseOperation,
                    Sequence<1, 1, 1, CBlockTransferScalarPerVector_NWaveNPerXDL>, // SliceLengths,
                    Sequence<0, 1, 2, 3>,                       // DimAccessOrder,
                    3,                                          // DstVectorDim,
                    CBlockTransferScalarPerVector_NWaveNPerXDL, // DstScalarPerVector,
                    InMemoryDataOperationEnum::Set, // InMemoryDataOperationEnum DstInMemOp,
                    1,                              // DstScalarStrideInVector,
                    false                           // DstResetCoordinateAfterRun,
                    >{c_grid_desc_mblock_mperblock_nblock_nperblock,
                      make_multi_index(__builtin_amdgcn_readfirstlane(spatial_idx[I0]),
                                       thread_m_cluster_id,
                                       __builtin_amdgcn_readfirstlane(spatial_idx[I1]),
                                       thread_n_cluster_id *
                                           CBlockTransferScalarPerVector_NWaveNPerXDL),
                      CElementwiseOperation{}};

                // block synchronization
                wg_barrier.wait_eq(reduction_idx, tile_acc_offset_end - tile_acc_offset_start);

#if 0
                if(threadIdx.x == 0) {
                    printf("bid:%d, rid:%d, os:%d,%d, spatial:%d,%d\n", static_cast<int>(blockIdx.x),
                        reduction_idx, __builtin_amdgcn_readfirstlane(tile_acc_offset_start), __builtin_amdgcn_readfirstlane(tile_acc_offset_end),
                        __builtin_amdgcn_readfirstlane(spatial_idx[I0]),
                        __builtin_amdgcn_readfirstlane(spatial_idx[I1]));
                }
#endif

                using Accumulation = ck::detail::
                    AccumulateWithNanCheck<false /*PropagateNan*/, reduce::Add, FloatAcc>;

                for(int i_m = 0; i_m < MReduceIters; i_m++)
                {
                    static_for<0, NReduceIters, 1>{}([&](auto i_n_reduce) {
                        acc_buf.Clear();
                        for(auto i = tile_acc_offset_start; i < tile_acc_offset_end; i++)
                        {
                            auto c_partial_acc_buf =
                                make_dynamic_buffer<AddressSpaceEnum::Global,
                                                    AmdBufferCoherenceEnum::GLC>(
                                    reinterpret_cast<FloatAcc*>(p_workspace) +
                                        i * c_partial_acc_block_m_n.GetElementSpaceSize(),
                                    c_partial_acc_block_m_n.GetElementSpaceSize());

                            acc_load.Run(c_partial_acc_block_m_n,
                                         c_partial_acc_buf,
                                         acc_thread_buf_load_desc,
                                         make_tuple(I0, I0),
                                         parcial_acc_buf);

                            static_for<0, CBlockTransferScalarPerVector_NWaveNPerXDL, 1>{}(
                                [&](auto i_vec) {
                                    constexpr auto offset =
                                        acc_thread_buf_load_desc.CalculateOffset(
                                            make_tuple(0, i_vec));
                                    Accumulation::Calculate(acc_buf(Number<offset>{}),
                                                            parcial_acc_buf[Number<offset>{}]);
                                });
                        }

                        if(thread_n_cluster_id * CBlockTransferScalarPerVector_NWaveNPerXDL <
                           NPerBlock)
                        {
                            acc_store.Run(acc_thread_buf_store_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          acc_buf,
                                          c_grid_desc_mblock_mperblock_nblock_nperblock,
                                          c_grid_buf);
                        }
                        if constexpr(NReduceIters != 1)
                        {
                            if constexpr(i_n_reduce != (NReduceIters - 1))
                            {
                                acc_load.MoveSrcSliceWindow(c_partial_acc_block_m_n,
                                                            partial_acc_load_step_n);
                                acc_store.MoveDstSliceWindow(
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    partial_acc_store_step_n);
                            }
                            else
                            {
                                acc_load.MoveSrcSliceWindow(c_partial_acc_block_m_n,
                                                            partial_acc_load_step_n_reverse);
                                acc_store.MoveDstSliceWindow(
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    partial_acc_store_step_n_reverse);
                            }
                        }
                    });
                    {
                        acc_load.MoveSrcSliceWindow(c_partial_acc_block_m_n,
                                                    partial_acc_load_step_m);
                        acc_store.MoveDstSliceWindow(c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                     partial_acc_store_step_m);
                    }
                }
                return;
            }
        }

        // offset for last acc buffer of this block
        uint32_t block_acc_offset =
            (block_mapping.get_acc_buffer_offset_from_block(block_idx + 1) - 1) * MPerBlock *
            NPerBlock;

        while(true)
        {
            uint32_t current_iter_length = __builtin_amdgcn_readfirstlane(
                block_mapping.get_current_iter_length(iter_start, iter_end, total_iter_length));
            uint32_t tile_idx, iter_offset;
            block_mapping.get_tile_idx_with_offset(iter_end - 1, tile_idx, iter_offset);
            iter_offset = __builtin_amdgcn_readfirstlane(iter_offset - current_iter_length + 1);
            auto spatial_idx = block_mapping.tile_to_spatial(tile_idx, m, n);

            const index_t m_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(spatial_idx[I0] * MPerBlock);

            const index_t n_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(spatial_idx[I1] * NPerBlock);

            const index_t k0_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(iter_offset * K0PerBlock);

            // A matrix blockwise copy
            auto a_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                    AElementwiseOperation,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<K0PerBlock, MPerBlock, K1>,
                                                    ABlockTransferThreadClusterLengths_K0_M_K1,
                                                    ABlockTransferThreadClusterArrangeOrder,
                                                    FloatAB,
                                                    FloatAB,
                                                    decltype(a_k0_m_k1_grid_desc),
                                                    decltype(a_block_desc_k0_m_k1),
                                                    ABlockTransferSrcAccessOrder,
                                                    Sequence<1, 0, 2>,
                                                    ABlockTransferSrcVectorDim,
                                                    2,
                                                    ABlockTransferSrcScalarPerVector,
                                                    ABlockTransferDstScalarPerVector_K1,
                                                    1,
                                                    1,
                                                    AThreadTransferSrcResetCoordinateAfterRun,
                                                    true>(
                    a_k0_m_k1_grid_desc,
                    make_multi_index(k0_block_data_idx_on_grid, m_block_data_idx_on_grid, 0),
                    a_element_op,
                    a_block_desc_k0_m_k1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

            // B matrix blockwise copy
            auto b_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                    BElementwiseOperation,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    InMemoryDataOperationEnum::Set,
                                                    Sequence<K0PerBlock, NPerBlock, K1>,
                                                    BBlockTransferThreadClusterLengths_K0_N_K1,
                                                    BBlockTransferThreadClusterArrangeOrder,
                                                    FloatAB,
                                                    FloatAB,
                                                    decltype(b_k0_n_k1_grid_desc),
                                                    decltype(b_block_desc_k0_n_k1),
                                                    BBlockTransferSrcAccessOrder,
                                                    Sequence<1, 0, 2>,
                                                    BBlockTransferSrcVectorDim,
                                                    2,
                                                    BBlockTransferSrcScalarPerVector,
                                                    BBlockTransferDstScalarPerVector_K1,
                                                    1,
                                                    1,
                                                    BThreadTransferSrcResetCoordinateAfterRun,
                                                    true>(
                    b_k0_n_k1_grid_desc,
                    make_multi_index(k0_block_data_idx_on_grid, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc_k0_n_k1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

            const index_t num_k_block_main_loop = current_iter_length;

            gridwise_gemm_pipeline.Run(a_k0_m_k1_grid_desc,
                                       a_block_desc_k0_m_k1,
                                       a_blockwise_copy,
                                       a_grid_buf,
                                       a_block_buf,
                                       a_block_slice_copy_step,
                                       b_k0_n_k1_grid_desc,
                                       b_block_desc_k0_n_k1,
                                       b_blockwise_copy,
                                       b_grid_buf,
                                       b_block_buf,
                                       b_block_slice_copy_step,
                                       blockwise_gemm,
                                       c_thread_buf,
                                       num_k_block_main_loop);

            // output: register to global memory
            {
                constexpr index_t MWave = MPerBlock / (MRepeat * MPerXdl);
                constexpr index_t NWave = NPerBlock / (NRepeat * NPerXdl);

                constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc =
                    blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc =
                    blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                constexpr auto M0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I0);
                constexpr auto N0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I1);
                constexpr auto M1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I2);
                constexpr auto N1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I3);
                constexpr auto M2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I4);
                constexpr auto M3 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I5);
                constexpr auto M4 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I6);
                constexpr auto N2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I7);

                constexpr auto c_block_desc_mblock_mpershuffle_nblock_npershuffle =
                    GetCBlockDescriptor_MBlock_MPerShuffle_NBlock_NPerShuffle();

                constexpr auto c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle =
                    GetCBlockDescriptor_MShuffleRepeat_MPerShuffle_NShuffleRepeat_NPerShuffle();

                auto c_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    reinterpret_cast<FloatCShuffle*>(p_shared_block),
                    c_block_desc_mblock_mpershuffle_nblock_npershuffle.GetElementSpaceSize());

                auto c_partial_acc_buf =
                    make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::GLC>(
                        reinterpret_cast<FloatAcc*>(p_workspace) + block_acc_offset,
                        c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle
                            .GetElementSpaceSize());

                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                    c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                    make_tuple(make_freeze_transform(I0), // freeze mblock
                               make_unmerge_transform(
                                   make_tuple(CShuffleMRepeatPerShuffle,
                                              M1,
                                              M2,
                                              M3,
                                              M4)),       // M1 = MWave, M2 * M3 * M4 = MPerXdl
                               make_freeze_transform(I0), // freeze nblock
                               make_unmerge_transform(
                                   make_tuple(CShuffleNRepeatPerShuffle,
                                              N1,
                                              N2))), // M1 = MWave, M2 * M3 * M4 = MPerXdl
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<>{},
                               Sequence<0, 2, 4, 5, 6>{},
                               Sequence<>{},
                               Sequence<1, 3, 7>{}));

                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
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

                // VGPR to LDS
                auto c_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
                    FloatAcc,
                    FloatCShuffle,
                    decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc),
                    decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                    ck::tensor_operation::element_wise::PassThrough,
                    Sequence<CShuffleMRepeatPerShuffle,
                             CShuffleNRepeatPerShuffle,
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

                // LDS to global
                auto c_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1r2<
                    ThisThreadBlock,       // index_t BlockSize,
                    CElementwiseOperation, // ElementwiseOperation,
                                           // InMemoryDataOperationEnum::Set, // DstInMemOp,
                    Sequence<1,
                             CShuffleMRepeatPerShuffle * MWave * MPerXdl,
                             1,
                             CShuffleNRepeatPerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                    CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    FloatCShuffle,        // typename SrcData,
                    FloatC,               // typename DstData,
                    decltype(c_block_desc_mblock_mpershuffle_nblock_npershuffle),
                    decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                    Sequence<0, 1, 2, 3>,                       // typename DimAccessOrder,
                    3,                                          // index_t VectorDim,
                    CBlockTransferScalarPerVector_NWaveNPerXDL, // index_t ScalarPerVector,
                    false, // bool ThreadTransferSrcResetCoordinateAfterRun,
                    false> // bool ThreadTransferDstResetCoordinateAfterRun
                    {c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                     make_multi_index(0, 0, 0, 0),
                     c_grid_desc_mblock_mperblock_nblock_nperblock,
                     make_multi_index(__builtin_amdgcn_readfirstlane(spatial_idx[I0]),
                                      0,
                                      __builtin_amdgcn_readfirstlane(spatial_idx[I1]),
                                      0),
                     c_element_op};

                // LDS to global partial acc
                auto c_block_copy_lds_to_partial_acc = ThreadGroupTensorSliceTransfer_v6r1r2<
                    ThisThreadBlock,       // index_t BlockSize,
                    CElementwiseOperation, // ElementwiseOperation,
                                           // InMemoryDataOperationEnum::Set, // DstInMemOp,
                    Sequence<1,
                             CShuffleMRepeatPerShuffle * MWave * MPerXdl,
                             1,
                             CShuffleNRepeatPerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                    CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    FloatCShuffle,        // typename SrcData,
                    FloatCShuffle,        // typename DstData,
                    decltype(c_block_desc_mblock_mpershuffle_nblock_npershuffle),
                    decltype(c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle),
                    Sequence<0, 1, 2, 3>,                       // typename DimAccessOrder,
                    3,                                          // index_t VectorDim,
                    CBlockTransferScalarPerVector_NWaveNPerXDL, // index_t ScalarPerVector,
                    false, // bool ThreadTransferSrcResetCoordinateAfterRun, => need to be false,
                           // othre wise has scratch
                    false> // bool ThreadTransferDstResetCoordinateAfterRun, => need to be false,
                           // othre wise has scratch
                    {c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                     make_multi_index(0, 0, 0, 0),
                     c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                     make_multi_index(0, 0, 0, 0),
                     c_element_op};

                constexpr auto mxdlperwave_forward_step =
                    make_multi_index(0, CShuffleMRepeatPerShuffle * MWave * MPerXdl, 0, 0);
                constexpr auto nxdlperwave_forward_step =
                    make_multi_index(0, 0, 0, CShuffleNRepeatPerShuffle * NWave * NPerXdl);
                constexpr auto nxdlperwave_backward_step =
                    make_multi_index(0, 0, 0, -CShuffleNRepeatPerShuffle * NWave * NPerXdl);

                static_for<0, MRepeat, CShuffleMRepeatPerShuffle>{}([&](auto mxdlperwave_iter) {
                    constexpr auto mxdlperwave = mxdlperwave_iter;

                    static_for<0, NRepeat, CShuffleNRepeatPerShuffle>{}([&](auto nxdlperwave_iter) {
                        constexpr bool nxdlperwave_forward_sweep =
                            (mxdlperwave % (2 * CShuffleMRepeatPerShuffle) == 0);

                        constexpr index_t nxdlperwave_value =
                            nxdlperwave_forward_sweep
                                ? nxdlperwave_iter
                                : (NRepeat - nxdlperwave_iter - CShuffleNRepeatPerShuffle);

                        constexpr auto nxdlperwave = Number<nxdlperwave_value>{};

                        // make sure it's safe to do ds_write
                        block_sync_lds();

                        // VGPR to LDS
                        c_thread_copy_vgpr_to_lds.Run(
                            c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc,
                            make_tuple(mxdlperwave, nxdlperwave, I0, I0, I0, I0, I0, I0),
                            c_thread_buf,
                            c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                            c_block_buf);

                        // make sure it's safe to do ds_read
                        block_sync_lds();

                        c_block_copy_lds_to_global.SetSrcSliceOrigin(
                            c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                            make_tuple(0, 0, 0, 0));

                        // LDS to global
                        if(is_dp_block)
                            c_block_copy_lds_to_global.template Run<decltype(c_block_buf),
                                                                    decltype(c_grid_buf),
                                                                    InMemoryDataOperationEnum::Set>(
                                c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                                c_block_buf,
                                c_grid_desc_mblock_mperblock_nblock_nperblock,
                                c_grid_buf);
                        else if(is_sk_block)
                        {
                            if constexpr(Block2CTileMap::ReductionStrategy ==
                                         StreamKReductionStrategy::Reduction)
                            {
                                // constexpr offset
                                c_block_copy_lds_to_partial_acc.SetSrcSliceOrigin(
                                    c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                                    make_tuple(0, 0, 0, 0));

                                c_block_copy_lds_to_partial_acc.SetDstSliceOrigin(
                                    c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                                    make_tuple(mxdlperwave.value, 0, nxdlperwave.value, 0));

                                c_block_copy_lds_to_partial_acc
                                    .template Run<decltype(c_block_buf),
                                                  decltype(c_partial_acc_buf),
                                                  InMemoryDataOperationEnum::Set>(
                                        c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                                        c_block_buf,
                                        c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                                        c_partial_acc_buf);
                            }
                            else if constexpr(Block2CTileMap::ReductionStrategy ==
                                              StreamKReductionStrategy::Atomic)
                            {
                                c_block_copy_lds_to_global
                                    .template Run<decltype(c_block_buf),
                                                  decltype(c_grid_buf),
                                                  InMemoryDataOperationEnum::AtomicAdd>(
                                        c_block_desc_mblock_mpershuffle_nblock_npershuffle,
                                        c_block_buf,
                                        c_grid_desc_mblock_mperblock_nblock_nperblock,
                                        c_grid_buf);
                            }
                        }

                        // move on nxdlperwave dimension
                        if constexpr(nxdlperwave_forward_sweep &&
                                     (nxdlperwave < NRepeat - CShuffleNRepeatPerShuffle))
                        {
                            c_block_copy_lds_to_global.MoveDstSliceWindow(
                                c_grid_desc_mblock_mperblock_nblock_nperblock,
                                nxdlperwave_forward_step);
                        }
                        else if constexpr((!nxdlperwave_forward_sweep) && (nxdlperwave > 0))
                        {
                            c_block_copy_lds_to_global.MoveDstSliceWindow(
                                c_grid_desc_mblock_mperblock_nblock_nperblock,
                                nxdlperwave_backward_step);
                        }
                    });

                    // move on mxdlperwave dimension
                    if constexpr(mxdlperwave < MRepeat - CShuffleMRepeatPerShuffle)
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            mxdlperwave_forward_step);
                    }
                });

                if constexpr(Block2CTileMap::ReductionStrategy ==
                             StreamKReductionStrategy::Reduction)
                {
                    if(is_sk_block)
                    {
                        // increase the counter for this tile
                        workgroup_barrier wg_barrier(p_semaphore);
                        wg_barrier.inc(tile_idx);
                    }
                }
            }

            // exit condition
            iter_end -= current_iter_length;
            if(iter_end <= iter_start)
                break;

            if constexpr(Block2CTileMap::ReductionStrategy == StreamKReductionStrategy::Reduction)
            {
                block_acc_offset -= MPerBlock * NPerBlock;
            }
            // make sure next loop LDS is ready for use
            block_sync_lds();
        }
    }

    template <typename Layout>
    struct LStr
    {
        static std::string Get() { return ""; }
    };

    template <>
    struct LStr<ck::tensor_layout::gemm::RowMajor>
    {
        static std::string Get() { return "R"; }
    };

    template <>
    struct LStr<ck::tensor_layout::gemm::ColumnMajor>
    {
        static std::string Get() { return "C"; }
    };

    static std::string GetTypeString()
    {
        auto str = std::stringstream();

        // clang-format off
        str << "GemmXdlStreamK_"
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << "_"
            << "B" << BlockSize << "_"
            << "Vec" << ABlockTransferSrcScalarPerVector << "x"
            << BBlockTransferSrcScalarPerVector << "x"
            << CBlockTransferScalarPerVector_NWaveNPerXDL << "_"
            << MPerBlock << "x"
            << NPerBlock << "x"
            << K0PerBlock << "x"
            << K1 ;
        // clang-format on

        return str.str();
    }
};

} // namespace ck
