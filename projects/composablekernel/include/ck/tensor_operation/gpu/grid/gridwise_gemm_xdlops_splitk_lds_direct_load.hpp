// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/amd_lds.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_direct_load.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

namespace ck {

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename Block2CTileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdlops_splitk_lds_direct_load(typename GridwiseGemm::Argument karg,
                                              const Block2CTileMap& b2c_map,
                                              const AElementwiseOperation a_element_op,
                                              const BElementwiseOperation b_element_op,
                                              const CElementwiseOperation c_element_op)
{
#if defined(__gfx9__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();

        __shared__ uint8_t p_shared[shared_size];

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
            karg, static_cast<void*>(p_shared), b2c_map, a_element_op, b_element_op, c_element_op);
    }
#else
    ignore = karg;
    ignore = b2c_map;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx9__))
}

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename FloatC,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v4,
          typename ComputeType        = FloatC>
struct GridwiseGemm_xdlops_splitk_lds_direct_load
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
    static constexpr auto KPerBlock = Number<K1Value * K0PerBlock>{};
    static constexpr auto M01       = 1;
    static constexpr auto N01       = 1;

    static constexpr auto gemm_padder =
        tensor_operation::device::GemmPadder<GemmSpec, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, K1* K0PerBlock};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        const FloatA* p_a_grid;
        const FloatB* p_b_grid;
        FloatC* p_c_grid;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t KPadded;
        index_t K0Padded;
        index_t k_batch;

        Argument(const FloatA* p_a_grid_,
                 const FloatB* p_b_grid_,
                 FloatC* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 index_t MPadded_,
                 index_t NPadded_,
                 index_t KPadded_,
                 index_t K0Padded_,
                 index_t k_batch_)
            : p_a_grid(p_a_grid_),
              p_b_grid(p_b_grid_),
              p_c_grid(p_c_grid_),
              M(M_),
              N(N_),
              K(K_),
              StrideA(StrideA_),
              StrideB(StrideB_),
              StrideC(StrideC_),
              MPadded(MPadded_),
              NPadded(NPadded_),
              KPadded(KPadded_),
              K0Padded(K0Padded_),
              k_batch(k_batch_)
        {
        }

        void Print() const
        {
            std::cout << "arg {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KP:" << KPadded << ", " << "K0Padded:" << K0Padded << ", "
                      << "KB:" << k_batch << "}" << std::endl;
        }
    };

    __host__ __device__ static auto CalculateGridSize(const Argument& karg)
    {
        return std::make_tuple(math::integer_divide_ceil(karg.N, NPerBlock),
                               math::integer_divide_ceil(karg.M, MPerBlock),
                               karg.k_batch);
    }

    // prefer this to be called on host
    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateK0Padded(index_t K, index_t K_Batch = 1)
    {
        // k_batch * k0 * k0_per_block * k1
        auto K_t = K_Batch * K0PerBlock * K1;
        return (K + K_t - 1) / K_t * K0PerBlock;
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K0Padded = CalculateK0Padded(K, K_Batch);
        return K_Batch * K0Padded * K1;
    }

    __host__ __device__ static auto MakeAGridDescriptor_KBatch_K0_M_K1(index_t M,
                                                                       index_t MPad,
                                                                       index_t K,
                                                                       index_t StrideA,
                                                                       index_t KBatch,
                                                                       index_t K0Padded,
                                                                       index_t KPad)
    {
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

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {

            const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                          GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_KBatch_K0_N_K1(index_t K,
                                                                       index_t NPad,
                                                                       index_t N,
                                                                       index_t StrideB,
                                                                       index_t KBatch,
                                                                       index_t K0Padded,
                                                                       index_t KPad)
    {
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

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {

            const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                          GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    __host__ __device__ static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
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

        return gemm_padder.PadCDescriptor_M_N(c_grid_desc_m_n);
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k0_m_k1_block_desc = [&]() {
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
        }();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k0_n_k1_block_desc = [&]() {
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
        }();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size =
            math::integer_least_multiple(b_k0_n_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto c_block_size =
            GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock().GetElementSpaceSize();

        return math::max(NumGemmKPrefetchStage * (a_block_space_size + b_block_space_size) *
                             sizeof(ComputeType),
                         c_block_size * sizeof(FloatC));
    }

    static constexpr index_t MXdlPerWave = MRepeat;
    static constexpr index_t NXdlPerWave = NRepeat;
    IS_VALID_COMPILATION_PARAMETER_IMPL(FloatC)

    __host__ __device__ static constexpr bool CheckValidity(const Argument& karg)
    {
        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(karg.M % MPerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(karg.N % NPerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = karg.k_batch * K0PerBlock * K1;
            if(!(karg.K % K_t == 0))
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
            {
                return false;
            }
        }
        else
        {
            if(karg.M % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
            {
                return false;
            }
        }

        const auto num_k_loop = karg.K0Padded / K0PerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static auto GetKPad(index_t K, index_t KBatch)
    {
        const index_t K0Padded =
            math::integer_divide_ceil(K, K1 * K0PerBlock * KBatch) * K0PerBlock;
        const index_t KPad = KBatch * K0Padded * K1;
        return KPad;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0Padded)
    {
        const index_t num_loop = K0Padded / K0PerBlock;
        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
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

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MRepeat * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NRepeat * NPerXdl);

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1,
                       Number<CShuffleMRepeatPerShuffle * MWave * MPerXdl>{},
                       I1,
                       Number<CShuffleNRepeatPerShuffle * NWave * NPerXdl>{}));
    }

    // return block_id to C matrix tile idx (m0, n0, k_split) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap()
    {
        return BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>();
    }

    using CGridDesc_M_N         = remove_cvref_t<decltype(MakeCGridDescriptor_M_N(1, 1, 1))>;
    using DefaultBlock2CTileMap = remove_cvref_t<decltype(MakeDefaultBlock2CTileMap())>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              typename Block2CTileMap>
    __device__ static void Run(const Argument& karg,
                               void* __restrict__ p_shared_block,
                               const Block2CTileMap& block_2_ctile_map,
                               const AElementwiseOperation a_element_op = AElementwiseOperation{},
                               const BElementwiseOperation b_element_op = BElementwiseOperation{},
                               const CElementwiseOperation c_element_op = CElementwiseOperation{})
    {
        // Elementwise operations are not supported for A and B, arguments left only for the API
        // consistency.
        (void)a_element_op;
        (void)b_element_op;

        const FloatA* p_a_grid           = karg.p_a_grid;
        const FloatB* p_b_grid           = karg.p_b_grid;
        FloatC* p_c_grid                 = karg.p_c_grid;
        const auto a_b_k0_m_k1_grid_desc = MakeAGridDescriptor_KBatch_K0_M_K1(
            karg.M, karg.MPadded, karg.K, karg.StrideA, karg.k_batch, karg.K0Padded, karg.KPadded);
        const auto b_b_k0_n_k1_grid_desc = MakeBGridDescriptor_KBatch_K0_N_K1(
            karg.K, karg.NPadded, karg.N, karg.StrideB, karg.k_batch, karg.K0Padded, karg.KPadded);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_b_k0_m_k1_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_b_k0_n_k1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // divide block work by [KBatch, M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I2]);
        const index_t k_batch_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k0_m_k1_block_desc = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(K1, Number<KPerBlock>{}, I1));
            }
        }();

        constexpr auto a_b_k0_m_k1_block_desc = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<K0PerBlock>{} * Number<MPerBlock + 1>{} * K1,
                               Number<MPerBlock + 1>{} * K1,
                               K1,
                               I1));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(
                        Number<KPerBlock>{} * Number<MPerBlock>{}, K1, Number<KPerBlock>{}, I1));
            }
        }();
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k0_n_k1_block_desc = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(K1, Number<KPerBlock>{}, I1));
            }
        }();

        constexpr auto b_b_k0_n_k1_block_desc = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<K0PerBlock>{} * Number<NPerBlock + 1>{} * K1,
                               Number<NPerBlock + 1>{} * K1,
                               K1,
                               I1));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<1>{}, Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(
                        Number<KPerBlock>{} * Number<NPerBlock>{}, K1, Number<KPerBlock>{}, I1));
            }
        }();

        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_DirectLoad<ThisThreadBlock,
                                                      Sequence<1, K0PerBlock, MPerBlock, K1>,
                                                      ABlockTransferThreadClusterLengths_K0_M_K1,
                                                      ABlockTransferSrcAccessOrder,
                                                      FloatA,
                                                      ComputeType,
                                                      decltype(a_b_k0_m_k1_grid_desc),
                                                      decltype(a_b_k0_m_k1_block_desc),
                                                      ABlockTransferSrcAccessOrder,
                                                      ABlockTransferSrcVectorDim,
                                                      3,
                                                      ABlockTransferSrcScalarPerVector>(
                a_b_k0_m_k1_grid_desc,
                make_multi_index(k_batch_id, 0, m_block_data_idx_on_grid, 0),
                a_b_k0_m_k1_block_desc,
                make_multi_index(0, 0, 0, 0));

        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_DirectLoad<ThisThreadBlock,
                                                      Sequence<1, K0PerBlock, NPerBlock, K1>,
                                                      BBlockTransferThreadClusterLengths_K0_N_K1,
                                                      BBlockTransferSrcAccessOrder,
                                                      FloatB,
                                                      ComputeType,
                                                      decltype(b_b_k0_n_k1_grid_desc),
                                                      decltype(b_b_k0_n_k1_block_desc),
                                                      BBlockTransferSrcAccessOrder,
                                                      BBlockTransferSrcVectorDim,
                                                      3,
                                                      BBlockTransferSrcScalarPerVector>(
                b_b_k0_n_k1_grid_desc,
                make_multi_index(k_batch_id, 0, n_block_data_idx_on_grid, 0),
                b_b_k0_n_k1_block_desc,
                make_multi_index(0, 0, 0, 0));

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ComputeType, // ComputeType A
            ComputeType, // ComputeType B
            FloatAcc,
            decltype(a_k0_m_k1_block_desc),
            decltype(b_k0_n_k1_block_desc),
            MPerXdl,
            NPerXdl,
            MRepeat,
            NRepeat,
            K1,
            LoopSched>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto a_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);

        const auto a_buffers_offset = 0;
        auto a_block_buffers =
            ck::lds_utils::AllocateLdsBuffers<ComputeType, NumGemmKPrefetchStage>(
                p_shared_block,
                a_b_k0_m_k1_block_desc.GetElementSpaceSize(),
                a_buffers_offset,
                max_lds_align);
        const auto b_buffers_offset = a_block_space_size * NumGemmKPrefetchStage;
        auto b_block_buffers =
            ck::lds_utils::AllocateLdsBuffers<ComputeType, NumGemmKPrefetchStage>(
                p_shared_block,
                b_b_k0_n_k1_block_desc.GetElementSpaceSize(),
                b_buffers_offset,
                max_lds_align);

        // gridwise GEMM pipeline
        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_b_k0_m_k1_grid_desc.GetLength(I1) * a_b_k0_m_k1_grid_desc.GetLength(I3)) /
            (K0PerBlock * K1));

        const auto gridwise_gemm_pipeline = GridwiseGemmPipe{};

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_b_k0_m_k1_grid_desc,
                                                               a_b_k0_m_k1_block_desc,
                                                               a_blockwise_copy,
                                                               a_grid_buf,
                                                               a_block_buffers,
                                                               a_block_slice_copy_step,
                                                               b_b_k0_n_k1_grid_desc,
                                                               b_b_k0_n_k1_block_desc,
                                                               b_blockwise_copy,
                                                               b_grid_buf,
                                                               b_block_buffers,
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

            constexpr auto c_block_desc_mblock_mperblock_nblock_nperblock =
                GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatC*>(p_shared_block),
                c_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0), // freeze mblock
                    make_unmerge_transform(make_tuple(CShuffleMRepeatPerShuffle,
                                                      M1,
                                                      M2,
                                                      M3,
                                                      M4)), // M1 = MWave, M2 * M3 * M4 = MPerXdl
                    make_freeze_transform(I0),              // freeze nblock
                    make_unmerge_transform(make_tuple(CShuffleNRepeatPerShuffle,
                                                      N1,
                                                      N2))), // M1 = MWave, M2 * M3 * M4 = MPerXdl
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

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
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatC,
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
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
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
            auto c_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // index_t BlockSize,
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatC,               // typename SrcData,
                FloatC,               // typename DstData,
                decltype(c_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                       // typename DimAccessOrder,
                3,                                          // index_t VectorDim,
                CBlockTransferScalarPerVector_NWaveNPerXDL, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun
                {c_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_m_id, 0, block_n_id, 0),
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

                    // LDS to global
                    c_block_copy_lds_to_global.Run(c_block_desc_mblock_mperblock_nblock_nperblock,
                                                   c_block_buf,
                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                   c_grid_buf);

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
                        c_grid_desc_mblock_mperblock_nblock_nperblock, mxdlperwave_forward_step);
                }
            });
        }
    }
};

} // namespace ck
