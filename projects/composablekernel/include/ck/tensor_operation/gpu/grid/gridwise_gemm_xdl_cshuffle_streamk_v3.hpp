// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1r2.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/workgroup_barrier.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"

namespace ck {

// Currently we do not have a elegant way to put single lds buffer & double lds buffer pipe in same
// kernel function Blockers:
// 1. Two separted declaration of __shared__ pointer is the key to make sure data access operate on
// two lds chunks.
// 2. Occupied __shared__ won't release until whole shader end, a.k.a AB and C may not use same lds
// buffer when we declare __shared__ inside blkgemmpipe
template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_xdl_cshuffle_v3(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid, karg.p_b_grid, karg.p_c_grid, p_shared, karg, karg.p_workspace_);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_xdl_cshuffle_v3_2lds(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid,
            karg.p_b_grid,
            karg.p_c_grid,
            p_shared_0,
            p_shared_1,
            karg,
            karg.p_workspace_);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
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
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v4,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct GridwiseGemm_xdl_cshuffle_streamk_v3
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
    static constexpr auto AK0Number = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0Number = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1Number = Number<AK1Value>{};
    static constexpr auto BK1Number = Number<BK1Value>{};

    static constexpr auto lcm_AK1_BK1 = math::lcm(AK1Number, BK1Number);
    static constexpr bool is_single_rate_mfma =
        (((is_same<ComputeTypeA, half_t>::value || is_same<ComputeTypeA, bhalf_t>::value) &&
          lcm_AK1_BK1 <= 4) ||
         (is_same<ComputeTypeA, int8_t>::value && lcm_AK1_BK1 <= 8) ||
         ((is_same<ComputeTypeA, f8_t>::value || is_same<ComputeTypeA, bf8_t>::value) &&
          lcm_AK1_BK1 < 32))
            ? true
            : false;
    static constexpr auto is_scale_mfma = false;
    static constexpr index_t KPack =
        math::max(lcm_AK1_BK1,
                  MfmaSelector<ComputeTypeA,
                               MPerXdl,
                               NPerXdl,
                               ComputeTypeA,
                               is_single_rate_mfma,
                               is_scale_mfma>::selected_mfma.k_per_blk);

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;
    __host__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ static auto CalculateAK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }

    __host__ static auto CalculateBK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }

    __host__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
    }

    __host__ static auto CalculateKRead(index_t K, index_t K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }

    __host__ static auto CalculateMBlock(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock);
    }

    __host__ static auto CalculateNBlock(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock);
    }

    template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
    __host__ __device__ static constexpr auto MakeGemmMmaTileDescriptor(const TileDesc_K0_MN_K1&)
    {
        constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
        constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

        return transform_tensor_descriptor(
            TileDesc_K0_MN_K1{},
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                       make_unmerge_transform(make_tuple(
                           Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
        index_t M, index_t MPad, index_t K, index_t KPad, index_t StrideA, index_t AK0)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        // Pad both M and K to be multiples of the block sizes
        const auto a_grid_desc_m_k =
            transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                        make_tuple(make_right_pad_transform(M, MPad - M),
                                                   make_right_pad_transform(K, KPad - K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                       make_pass_through_transform(MPad)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_ak0_m_ak1;
#if 0
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(M, MPad - M),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(MPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
#endif
    }

    __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
        index_t K, index_t KPad, index_t N, index_t NPad, index_t StrideB, index_t BK0)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(StrideB, I1));
            }
        }();

        // Pad both N and K to be multiples of the block sizes
        const auto b_grid_desc_n_k =
            transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                        make_tuple(make_right_pad_transform(N, NPad - N),
                                                   make_right_pad_transform(K, KPad - K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
            b_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                       make_pass_through_transform(NPad)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_bk0_n_bk1;
#if 0     
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(N, NPad - N),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(NPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(N), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
#endif
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeAMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor<MXdlPerWave, MWaves, MPerXdl>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeBMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor<NXdlPerWave, NWaves, NPerXdl>(BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        // Pad both M and N to be multiples of the block sizes
        return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
#if 0
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(M, MPad - M),
                                                          make_right_pad_transform(N, NPad - N)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(M, MPad - M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
#endif
    }

    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         index_t StrideA_,
                         index_t StrideB_,
                         index_t StrideC_,
                         index_t Streamk_sel_,
                         index_t Grid_size_,
                         StreamKReductionStrategy reduction_strategy_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              Streamk_sel{Streamk_sel_},
              Grid_size{Grid_size_},
              reduction_strategy{reduction_strategy_}, // Initialize the member variable
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              KRead{CalculateKRead(K_, 1)},
              KPadded{CalculateKPadded(K_, 1)},
              AK0{CalculateAK0Padded(K_, 1)},
              BK0{CalculateBK0Padded(K_, 1)},
              MBlock{CalculateMBlock(M_)},
              NBlock{CalculateNBlock(N_)}

        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KRead:" << KRead << ", " << "KP:" << KPadded << ", " << "AK0:" << AK0
                      << ", " << "BK0:" << BK0 << ", " << "MBlock: " << MBlock << ", "
                      << "NBlock: " << NBlock << ", " << "Stream-K Selection:" << Streamk_sel
                      << ", " << "Grid size:" << Grid_size << ", " << "Reduction Strategy:"
                      << (reduction_strategy == StreamKReductionStrategy::Atomic ? "Atomic"
                                                                                 : "Reduction")
                      << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t Streamk_sel;
        mutable index_t Grid_size;
        StreamKReductionStrategy reduction_strategy;
        index_t MPadded;
        index_t NPadded;
        index_t KRead;
        index_t KPadded;
        index_t AK0;
        index_t BK0;
        index_t MBlock;
        index_t NBlock;
    };

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t Streamk_sel_,
                          index_t Grid_size_,
                          StreamKReductionStrategy reduction_strategy_)
            : Problem{M_,
                      N_,
                      K_,
                      StrideA_,
                      StrideB_,
                      StrideC_,
                      Streamk_sel_,
                      Grid_size_,
                      reduction_strategy_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_},
              block_2_ctile_map_streamk(M_,
                                        N_,
                                        AK0Number * CalculateKPadded(K_, 1),
                                        Grid_size_,
                                        Streamk_sel_,
                                        reduction_strategy_)

        {
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;
        BlockToCTileMap_GemmStreamK_v2<MPerBlock,
                                       NPerBlock,
                                       KPerBlock,
                                       StreamKReductionStrategy::Atomic,
                                       8,
                                       4>
            block_2_ctile_map_streamk;
    };

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(Problem& problem, unsigned int kbatch_id, unsigned int orig_K)
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = kbatch_id * problem.KRead;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = kbatch_id * problem.KRead * problem.M;
            }

            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = kbatch_id * problem.KRead * problem.N;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = kbatch_id * problem.KRead;
            }

            if(kbatch_id < static_cast<uint32_t>(problem.KBatch - 1))
            {
                problem.K = problem.KRead;
            }
            else
            {
                problem.K = orig_K - problem.KRead * (problem.KBatch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
    };

    __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave    = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWave * NWave);
        // A matrix in LDS memory, dst of blockwise copy
        if constexpr(ABlockLdsExtraM)
        {
            return make_naive_tensor_descriptor(
                make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                make_tuple(AK1Number, Number<KPerBlock + ABlockLdsExtraM>{}, I1));
        }
        // xor tensor transformation request more unnecessary vgpr usage, would cause register spill
        // in some cases.
        else if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            constexpr auto MLdsLayer        = 32 * 4 / KPerBlock / sizeof(ADataType) < 1
                                                  ? 1
                                                  : 32 * 4 / KPerBlock / sizeof(ADataType);
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    AK0Number * Number<MLdsLayer>{}, Number<MPerBlock / MLdsLayer>{}, AK1Number),
                make_tuple(AK1Number, Number<KPerBlock * MLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<MPerBlock / MLdsLayer>{}, Number<AK0Number * MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto a_lds_block_desc_ak0_mldslayer_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(AK0Number, Number<MLdsLayer>{})),
                           make_pass_through_transform(Number<MPerBlock / MLdsLayer>{}),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_ak0_mldslayer_m_ak1,
                make_tuple(make_pass_through_transform(AK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<MPerBlock / MLdsLayer>{}, Number<MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
        else // ColumnMajor A
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto M0 = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
            constexpr auto M1 = MPerBlock / M0;

            constexpr auto KThreadWrite     = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
            constexpr auto K0PerThreadWrite = AK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / MPerXdl;
            constexpr auto K0PerThreadRead  = AK0Number / KThreadRead;

            constexpr auto kfold = (AK1Number * M0 * sizeof(ADataType) > 128)
                                       ? 1
                                       : 128 / (AK1Number * M0 * sizeof(ADataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=n0
            constexpr auto mpair = (AK1Number * MPerXdl * sizeof(ADataType) > 128)
                                       ? 1
                                       : ((128 / (AK1Number * MPerXdl * sizeof(ADataType))) > M0
                                              ? M0
                                              : 128 / (AK1Number * MPerXdl * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * M1>{},
                           Number<kfold * M0 / mpair>{},
                           Number<mpair>{},
                           AK1Number));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * M1>{}, Number<kfold * M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<M1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<M0 / mpair>{}, Number<mpair>{}, Number<M1>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
    }

    __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave    = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWave * NWave);
        // B matrix in LDS memory, dst of blockwise copy
        if constexpr(BBlockLdsExtraN)
        {
            return make_naive_tensor_descriptor(
                make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                make_tuple(BK1Number, Number<KPerBlock + BBlockLdsExtraN>{}, I1));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
        {
            // NLdsLayer * K0 as logical Bank
            constexpr auto NLdsLayer = 32 * 4 / KPerBlock / sizeof(BDataType) < 1
                                           ? 1
                                           : 32 * 4 / KPerBlock / sizeof(BDataType);
            ;
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    BK0Number * Number<NLdsLayer>{}, Number<NPerBlock / NLdsLayer>{}, BK1Number),
                make_tuple(BK1Number, Number<KPerBlock * NLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<NPerBlock / NLdsLayer>{}, Number<BK0Number * NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(BK0Number, Number<NLdsLayer>{})),
                           make_pass_through_transform(Number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_bk0_nldslayer_n_bk1,
                make_tuple(make_pass_through_transform(BK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<NPerBlock / NLdsLayer>{}, Number<NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
        else // RowMajor B
        {
            constexpr auto N0 = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I1);
            constexpr auto N1 = NPerBlock / N0;

            constexpr auto KThreadWrite     = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I0);
            constexpr auto K0PerThreadWrite = BK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / NPerXdl;
            constexpr auto K0PerThreadRead  = BK0Number / KThreadRead;

            constexpr auto kfold = (BK1Number * N0 * sizeof(BDataType) > 128)
                                       ? 1
                                       : 128 / (BK1Number * N0 * sizeof(BDataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=n0
            constexpr auto npair = (BK1Number * NPerXdl * sizeof(BDataType) > 128)
                                       ? 1
                                       : ((128 / (BK1Number * NPerXdl * sizeof(BDataType))) > N0
                                              ? N0
                                              : 128 / (BK1Number * NPerXdl * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * N1>{},
                           Number<kfold * N0 / npair>{},
                           Number<npair>{},
                           BK1Number));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * N1>{}, Number<kfold * N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<N1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<N0 / npair>{}, Number<npair>{}, Number<N1>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
    }

    __device__ static constexpr auto GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
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

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MShuffle_MPerShuffle_NShuffle_NPerShuffle()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MXdlPerWave / CShuffleMXdlPerWavePerShuffle>{},
                       Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                       Number<NXdlPerWave / CShuffleNXdlPerWavePerShuffle>{},
                       Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));
    }

    using BlockwiseGemmPipe =
        remove_cvref_t<decltype(BlockGemmPipeline_Selector<
                                BlkGemmPipelineVer,
                                BlkGemmPipeSched,
                                BlockSize,
                                ADataType,
                                BDataType,
                                ComputeTypeA,
                                AccDataType,
                                decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()),
                                decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()),
                                decltype(MakeAMmaTileDescriptor_M0_M1_M2_K(
                                    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1())),
                                decltype(MakeBMmaTileDescriptor_N0_N1_N2_K(
                                    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1())),
                                ABlockTransferSrcScalarPerVector,
                                BBlockTransferSrcScalarPerVector,
                                MPerBlock,
                                NPerBlock,
                                KPerBlock,
                                MPerXdl,
                                NPerXdl,
                                MXdlPerWave,
                                NXdlPerWave,
                                KPack>())>;

    __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max((a_block_space_size_aligned * sizeof(ADataType) +
                          b_block_space_size_aligned * sizeof(BDataType)),
                         c_block_size * sizeof(CShuffleDataType));
    }

    IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType)

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ static constexpr bool CheckValidity(const Argument& karg)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     !(is_same<tensor_layout::gemm::RowMajor, ALayout>::value))
        {
            if(!(karg.M % MPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     (is_same<tensor_layout::gemm::RowMajor, BLayout>::value))
        {
            if(!(karg.N % NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = KPerBlock;
            if(!(karg.K % K_t == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                              << karg.K << " " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {

            if(karg.K <= 0)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }

                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }

                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of "
                                 "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }

                return false;
            }
        }
        else
        {
            if(karg.M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of "
                                 "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }

                return false;
            }
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = karg.AK0 / (KPerBlock / AK1Value);

        if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
        {
            if(num_k_loop <= BlockwiseGemmPipe::PrefetchStages)
            {
                return false;
            }
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    template <typename CGridDesc>
    __device__ static constexpr auto MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const CGridDesc& c_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr auto GetClusterLengthReduction()
    {
        // TODO: assume C is row major
        // TODO: we always first loop over N, then M
        constexpr auto NPerBlockPow2 = math::next_power_of_two<NPerBlock>();
        constexpr auto NPerBlockReduction =
            NPerBlockPow2 / CShuffleBlockTransferScalarPerVector_NPerBlock;
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
    using Block2CTileMap_streamk = BlockToCTileMap_GemmStreamK_v2<MPerBlock,
                                                                  NPerBlock,
                                                                  KPerBlock,
                                                                  StreamKReductionStrategy::Atomic,
                                                                  8,
                                                                  4>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               CDataType* p_c_grid,
                               void* p_shared,
                               Problem& problem,
                               void* p_workspace)
    {
        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideB, problem.BK0);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        Block2CTileMap_streamk block_2_ctile_map_streamk(problem.M,
                                                         problem.N,
                                                         AK0Number * problem.KPadded,
                                                         problem.Grid_size,
                                                         problem.Streamk_sel,
                                                         problem.reduction_strategy);
        uint32_t iter_start, iter_end;
        bool is_sk_block, is_dp_block, is_reduction_block;
        index_t num_k_block_main_loop;
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        uint32_t* p_semaphore = reinterpret_cast<uint32_t*>(
            reinterpret_cast<char*>(p_workspace) +
            block_2_ctile_map_streamk.get_workspace_size_for_acc(sizeof(AccDataType)));

        for(auto block_idx = get_block_1d_id();
            block_idx < block_2_ctile_map_streamk.get_grid_dims();
            block_idx += gridDim.x)
        {

            is_sk_block =
                static_cast<uint32_t>(block_idx) < block_2_ctile_map_streamk.sk_num_blocks;
            is_dp_block =
                static_cast<uint32_t>(block_idx) >= block_2_ctile_map_streamk.dp_start_block_idx &&
                static_cast<uint32_t>(block_idx) <
                    block_2_ctile_map_streamk.reduction_start_block_idx;

            block_2_ctile_map_streamk.get_block_itr(block_idx, iter_start, iter_end);
            num_k_block_main_loop = iter_end - iter_start;

            if(problem.reduction_strategy == StreamKReductionStrategy::Reduction)
            {
                is_reduction_block = static_cast<uint32_t>(block_idx) >=
                                     block_2_ctile_map_streamk.reduction_start_block_idx;
                if(is_reduction_block)
                {
                    // descriptors
                    constexpr auto cluster_length_reduce = GetClusterLengthReduction();
                    constexpr auto reduce_desc = make_cluster_descriptor(cluster_length_reduce);
                    const auto reduce_thread_cluster_idx =
                        reduce_desc.CalculateBottomIndex(make_multi_index(block_idx));
                    const auto thread_m_cluster_id = reduce_thread_cluster_idx[I0];
                    const auto thread_n_cluster_id = reduce_thread_cluster_idx[I1];

                    constexpr auto MReduceIters = math::integer_divide_ceil(
                        Number<MPerBlock>{}, cluster_length_reduce.At(I0));
                    constexpr auto NReduceIters = math::integer_divide_ceil(
                        Number<NPerBlock>{},
                        cluster_length_reduce.At(I1) *
                            Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{});

                    constexpr auto acc_thread_buf_load_desc = make_naive_tensor_descriptor_packed(
                        make_tuple(I1, Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{}));
                    constexpr auto acc_thread_buf_store_desc =
                        make_naive_tensor_descriptor_packed(make_tuple(
                            I1, I1, I1, Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{}));

                    constexpr auto c_partial_acc_block_m_n = GetPartialAccBlockDescriptor();

                    constexpr auto partial_acc_load_step_n =
                        make_multi_index(0,
                                         cluster_length_reduce.At(I1) *
                                             CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_load_step_n_reverse = make_multi_index(
                        0,
                        -1 * cluster_length_reduce.At(I1).value * (NReduceIters - 1) *
                            CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_load_step_m =
                        make_multi_index(cluster_length_reduce.At(I0), 0);

                    constexpr auto partial_acc_store_step_n =
                        make_multi_index(0,
                                         0,
                                         0,
                                         cluster_length_reduce.At(I1) *
                                             CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_store_step_n_reverse = make_multi_index(
                        0,
                        0,
                        0,
                        -1 * cluster_length_reduce.At(I1).value * (NReduceIters - 1) *
                            CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_store_step_m =
                        make_multi_index(0, cluster_length_reduce.At(I0), 0, 0);

                    StaticBuffer<AddressSpaceEnum::Vgpr,
                                 AccDataType,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock,
                                 true>
                        parcial_acc_buf;
                    StaticBuffer<AddressSpaceEnum::Vgpr,
                                 AccDataType,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock,
                                 true>
                        acc_buf;

                    // start to compute
                    auto reduction_idx =
                        block_idx - block_2_ctile_map_streamk.reduction_start_block_idx;
                    auto spatial_idx = block_2_ctile_map_streamk.tile_to_spatial(
                        reduction_idx, problem.M, problem.N);

                    workgroup_barrier wg_barrier(p_semaphore);

                    uint32_t tile_acc_offset_start =
                        block_2_ctile_map_streamk.get_acc_buffer_offset_from_tile(reduction_idx);
                    uint32_t tile_acc_offset_end =
                        block_2_ctile_map_streamk.get_acc_buffer_offset_from_tile(reduction_idx +
                                                                                  1);
                    __syncthreads();

                    auto acc_load = ThreadwiseTensorSliceTransfer_v2<
                        AccDataType,                        // SrcData,
                        AccDataType,                        // DstData,
                        decltype(c_partial_acc_block_m_n),  // SrcDesc,
                        decltype(acc_thread_buf_load_desc), // DstDesc,
                        Sequence<1,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock>, // SliceLengths,
                        Sequence<0, 1>,                                           // DimAccessOrder,
                        1,                                                        // SrcVectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // SrcScalarPerVector,
                        1,                                              // SrcScalarStrideInVector,
                        false // SrcResetCoordinateAfterRun,
                        >{c_partial_acc_block_m_n,
                          make_multi_index(thread_m_cluster_id,
                                           thread_n_cluster_id *
                                               CShuffleBlockTransferScalarPerVector_NPerBlock)};

                    auto acc_store = ThreadwiseTensorSliceTransfer_v1r3<
                        AccDataType,                                             // SrcData,
                        CDataType,                                               // DstData,
                        decltype(acc_thread_buf_store_desc),                     // SrcDesc,
                        decltype(c_grid_desc_mblock_mperblock_nblock_nperblock), // DstDesc,
                        CElementwiseOperation, // ElementwiseOperation,
                        Sequence<1,
                                 1,
                                 1,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock>, // SliceLengths,
                        Sequence<0, 1, 2, 3>,                                     // DimAccessOrder,
                        3,                                                        // DstVectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // DstScalarPerVector,
                        InMemoryDataOperationEnum::Set, // InMemoryDataOperationEnum DstInMemOp,
                        1,                              // DstScalarStrideInVector,
                        false                           // DstResetCoordinateAfterRun,
                        >{c_grid_desc_mblock_mperblock_nblock_nperblock,
                          make_multi_index(__builtin_amdgcn_readfirstlane(spatial_idx[I0]),
                                           thread_m_cluster_id,
                                           __builtin_amdgcn_readfirstlane(spatial_idx[I1]),
                                           thread_n_cluster_id *
                                               CShuffleBlockTransferScalarPerVector_NPerBlock),
                          CElementwiseOperation{}};

                    wg_barrier.wait_eq(reduction_idx, tile_acc_offset_end - tile_acc_offset_start);

                    if(threadIdx.x == 0)
                    {
                        p_semaphore[reduction_idx] = 0;
                    }
                    using Accumulation = ck::detail::
                        AccumulateWithNanCheck<false /*PropagateNan*/, reduce::Add, AccDataType>;

                    for(int i_m = 0; i_m < MReduceIters; i_m++)
                    {
                        static_for<0, NReduceIters, 1>{}([&](auto i_n_reduce) {
                            acc_buf.Clear();
                            for(auto i = tile_acc_offset_start; i < tile_acc_offset_end; i++)
                            {
                                auto c_partial_acc_buf =
                                    make_dynamic_buffer<AddressSpaceEnum::Global,
                                                        AmdBufferCoherenceEnum::GLC>(
                                        reinterpret_cast<AccDataType*>(p_workspace) +
                                            i * c_partial_acc_block_m_n.GetElementSpaceSize(),
                                        c_partial_acc_block_m_n.GetElementSpaceSize());

                                acc_load.Run(c_partial_acc_block_m_n,
                                             c_partial_acc_buf,
                                             acc_thread_buf_load_desc,
                                             make_tuple(I0, I0),
                                             parcial_acc_buf);

                                static_for<0, CShuffleBlockTransferScalarPerVector_NPerBlock, 1>{}(
                                    [&](auto i_vec) {
                                        constexpr auto offset =
                                            acc_thread_buf_load_desc.CalculateOffset(
                                                make_tuple(0, i_vec));
                                        Accumulation::Calculate(acc_buf(Number<offset>{}),
                                                                parcial_acc_buf[Number<offset>{}]);
                                    });
                            }

                            if(thread_n_cluster_id *
                                   CShuffleBlockTransferScalarPerVector_NPerBlock <
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
                            acc_store.MoveDstSliceWindow(
                                c_grid_desc_mblock_mperblock_nblock_nperblock,
                                partial_acc_store_step_m);
                        }
                    }

                    continue;
                }
            }

            // offset for last acc buffer of this block
            uint32_t block_acc_offset =
                (block_2_ctile_map_streamk.get_acc_buffer_offset_from_block(block_idx + 1) - 1) *
                MPerBlock * NPerBlock;
            while(true)
            {
                uint32_t current_iter_length = __builtin_amdgcn_readfirstlane(
                    block_2_ctile_map_streamk.get_current_iter_length(
                        iter_start, iter_end, num_k_block_main_loop));
                uint32_t tile_idx, iter_offset;
                block_2_ctile_map_streamk.get_tile_idx_with_offset(
                    iter_end - 1, tile_idx, iter_offset);
                iter_offset = __builtin_amdgcn_readfirstlane(iter_offset - current_iter_length + 1);

                auto block_work_idx =
                    block_2_ctile_map_streamk.tile_to_spatial(tile_idx, problem.M, problem.N);

                const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
                const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

                // HACK: this force m/n_block_data_idx_on_grid into SGPR
                const index_t m_block_data_idx_on_grid =
                    __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

                const index_t n_block_data_idx_on_grid =
                    __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

                const index_t k0_block_data_idx_on_grid =
                    __builtin_amdgcn_readfirstlane(iter_offset * AK0Number);

                // lds max alignment
                constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

                // A matrix in LDS memory, dst of blockwise copy
                constexpr auto a_block_desc_ak0_m_ak1 =
                    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

                // B matrix in LDS memory, dst of blockwise copy
                constexpr auto b_block_desc_bk0_n_bk1 =
                    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

                // A matrix blockwise copy
                auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    AElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ADataType,
                    ADataType,
                    decltype(a_grid_desc_ak0_m_ak1),
                    decltype(a_block_desc_ak0_m_ak1),
                    ABlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    1,
                    1,
                    AThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(k0_block_data_idx_on_grid, m_block_data_idx_on_grid, 0),
                    a_element_op,
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

                // B matrix blockwise copy
                auto b_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    BElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BDataType,
                    BDataType,
                    decltype(b_grid_desc_bk0_n_bk1),
                    decltype(b_block_desc_bk0_n_bk1),
                    BBlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    1,
                    1,
                    BThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(k0_block_data_idx_on_grid, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

                // LDS allocation for A and B: be careful of alignment
                constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
                    a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

                // Cast after lds
                auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<ADataType*>(p_shared),
                    a_block_desc_ak0_m_ak1.GetElementSpaceSize());

                auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<BDataType*>(p_shared) +
                        a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
                    b_block_desc_bk0_n_bk1.GetElementSpaceSize());

                constexpr auto a_block_slice_copy_step =
                    make_multi_index(KPerBlock / AK1Number, 0, 0);
                constexpr auto b_block_slice_copy_step =
                    make_multi_index(KPerBlock / BK1Number, 0, 0);

                // Blockwise GEMM pipeline
                static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
                auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
                auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

                num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
                    (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
                    KPerBlock);

                blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(
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
                    c_thread_buf,
                    num_k_block_main_loop);

                // shuffle C and write out
                {
                    static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                                      NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                                  "wrong!");

                    constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
                    constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

                    // TODO: hacky, fix it!
                    constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                        blockwise_gemm_pipeline.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                    // TODO: hacky, fix it!
                    // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
                    constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                        blockwise_gemm_pipeline.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

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

                    constexpr auto c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle =
                        GetCBlockDescriptor_MShuffle_MPerShuffle_NShuffle_NPerShuffle();

                    auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                        static_cast<CShuffleDataType*>(p_shared),
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock
                            .GetElementSpaceSize());

                    auto c_partial_acc_buf =
                        make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::GLC>(
                            reinterpret_cast<AccDataType*>(p_workspace) + block_acc_offset,
                            c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle
                                .GetElementSpaceSize());

                    constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                        transform_tensor_descriptor(
                            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                            make_tuple(
                                make_freeze_transform(I0),
                                make_unmerge_transform(make_tuple(
                                    Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per
                                                                             // shuffle
                                    M1,                                      // M1 = MWave
                                    M2, // M2 * M3 * M4 = MPerXdl
                                    M3,
                                    M4)),
                                make_freeze_transform(I0),
                                make_unmerge_transform(make_tuple(
                                    Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per
                                                                             // shuffle
                                    N1,                                      // N1 = NWave
                                    N2))),                                   // N2 = NPerXdl
                            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                            make_tuple(Sequence<>{},
                                       Sequence<0, 2, 4, 5, 6>{},
                                       Sequence<>{},
                                       Sequence<1, 3, 7>{}));

                    // calculate origin of thread output tensor on global memory
                    //     blockwise GEMM c matrix starting index
                    const auto c_thread_mtx_on_block =
                        blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

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
                        AccDataType,
                        CShuffleDataType,
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
                    auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1r2<
                        ThisThreadBlock,       // ThreadGroup
                        CElementwiseOperation, // ElementwiseOperation,
                        // CGlobalMemoryDataOperation, // DstInMemOp,
                        Sequence<1,
                                 CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                 1,
                                 CShuffleNXdlPerWavePerShuffle * NWave *
                                     NPerXdl>, // BlockSliceLengths,
                        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                        Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                        CShuffleDataType,     // typename SrcData,
                        CDataType,            // typename DstData,
                        decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                        decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                        Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                        3,                                              // index_t VectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                        false, // bool ThreadTransferSrcResetCoordinateAfterRun,
                        false> // bool ThreadTransferDstResetCoordinateAfterRun>
                        {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                         make_multi_index(0, 0, 0, 0),
                         c_grid_desc_mblock_mperblock_nblock_nperblock,
                         make_multi_index(block_m_id, 0, block_n_id, 0),
                         c_element_op};
                    // LDS to global partial acc
                    auto c_block_copy_lds_to_partial_acc = ThreadGroupTensorSliceTransfer_v6r1r2<
                        ThisThreadBlock,       // index_t BlockSize,
                        CElementwiseOperation, // ElementwiseOperation,
                                               // InMemoryDataOperationEnum::Set, // DstInMemOp,
                        Sequence<1,
                                 CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                 1,
                                 CShuffleNXdlPerWavePerShuffle * NWave *
                                     NPerXdl>, // BlockSliceLengths,
                        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                        Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                        CShuffleDataType,     // typename SrcData,
                        AccDataType,          // typename DstData,
                        decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                        decltype(c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle),
                        Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                        3,                                              // index_t VectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                        false, // bool ThreadTransferSrcResetCoordinateAfterRun, => need to be
                               // false, othre wise has scratch
                        false> // bool ThreadTransferDstResetCoordinateAfterRun, => need to be
                               // false, othre wise has scratch
                        {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                         make_multi_index(0, 0, 0, 0),
                         c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                         make_multi_index(0, 0, 0, 0),
                         c_element_op};
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
                    constexpr auto sfc_c_global = SpaceFillingCurve<
                        Sequence<1, MPerBlock, 1, NPerBlock>,
                        Sequence<0, 2, 1, 3>,
                        Sequence<1,
                                 CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                 1,
                                 CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

                    constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

                    static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

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
                        c_shuffle_block_copy_lds_to_global.SetSrcSliceOrigin(
                            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                            make_tuple(0, 0, 0, 0));

                        if(is_dp_block)
                        {
                            // each block copy its data from LDS to global
                            c_shuffle_block_copy_lds_to_global
                                .template Run<decltype(c_shuffle_block_buf),
                                              decltype(c_grid_buf),
                                              InMemoryDataOperationEnum::Set>(
                                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                    c_shuffle_block_buf,
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    c_grid_buf);
                        }
                        else if(is_sk_block)
                        {
                            if(problem.reduction_strategy == StreamKReductionStrategy::Atomic)
                            {
                                // each block copy its data from LDS to global
                                c_shuffle_block_copy_lds_to_global
                                    .template Run<decltype(c_shuffle_block_buf),
                                                  decltype(c_grid_buf),
                                                  InMemoryDataOperationEnum::AtomicAdd>(
                                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                        c_shuffle_block_buf,
                                        c_grid_desc_mblock_mperblock_nblock_nperblock,
                                        c_grid_buf);
                            }
                            else if(problem.reduction_strategy ==
                                    StreamKReductionStrategy::Reduction)
                            {
                                // constexpr offset
                                c_block_copy_lds_to_partial_acc.SetSrcSliceOrigin(
                                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                    make_tuple(0, 0, 0, 0));

                                c_block_copy_lds_to_partial_acc.SetDstSliceOrigin(
                                    c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                                    make_tuple(MXdlPerWave, 0, NXdlPerWave, 0));

                                c_block_copy_lds_to_partial_acc
                                    .template Run<decltype(c_shuffle_block_buf),
                                                  decltype(c_partial_acc_buf),
                                                  InMemoryDataOperationEnum::Set>(
                                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                        c_shuffle_block_buf,
                                        c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                                        c_partial_acc_buf);
                            }
                        }

                        if constexpr(access_id < num_access - 1)
                        {
                            constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                            // move on C
                            c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                                c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                        }
                    });

                    if(problem.reduction_strategy == StreamKReductionStrategy::Reduction)
                    {
                        if(is_sk_block)
                        {
                            // increase the counter for this tile
                            workgroup_barrier wg_barrier(p_semaphore);
                            wg_barrier.inc(tile_idx);
                        }
                    }
                } // shuffle c and write-out end

                // exit condition
                iter_end -= current_iter_length;
                if(iter_end <= iter_start)
                    break;
                if(problem.reduction_strategy == StreamKReductionStrategy::Reduction)
                {
                    block_acc_offset -= MPerBlock * NPerBlock;
                }
                // make sure next loop LDS is ready for use
                block_sync_lds();
            } // while loop

        } // for loop
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(const ADataType* p_a_grid,
                                    const BDataType* p_b_grid,
                                    CDataType* p_c_grid,
                                    void* p_shared_0,
                                    void* p_shared_1,
                                    Problem& problem,
                                    void* p_workspace)
    {

        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideB, problem.BK0);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        uint32_t iter_start, iter_end;
        bool is_sk_block, is_dp_block, is_reduction_block;
        index_t num_k_block_main_loop;

        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        Block2CTileMap_streamk block_2_ctile_map_streamk(problem.M,
                                                         problem.N,
                                                         AK0Number * problem.KPadded,
                                                         problem.Grid_size,
                                                         problem.Streamk_sel,
                                                         problem.reduction_strategy);
        for(auto block_idx = get_block_1d_id();
            block_idx < block_2_ctile_map_streamk.get_grid_dims();
            block_idx += gridDim.x)
        {
            is_sk_block =
                static_cast<uint32_t>(block_idx) < block_2_ctile_map_streamk.sk_num_blocks;
            is_dp_block =
                static_cast<uint32_t>(block_idx) >= block_2_ctile_map_streamk.dp_start_block_idx &&
                static_cast<uint32_t>(block_idx) <
                    block_2_ctile_map_streamk.reduction_start_block_idx;

            block_2_ctile_map_streamk.get_block_itr(block_idx, iter_start, iter_end);
            num_k_block_main_loop = iter_end - iter_start;

            uint32_t* p_semaphore = reinterpret_cast<uint32_t*>(
                reinterpret_cast<char*>(p_workspace) +
                block_2_ctile_map_streamk.get_workspace_size_for_acc(sizeof(AccDataType)));

            if(problem.reduction_strategy == StreamKReductionStrategy::Reduction)
            {
                is_reduction_block = static_cast<uint32_t>(block_idx) >=
                                     block_2_ctile_map_streamk.reduction_start_block_idx;
                if(is_reduction_block)
                {
                    // descriptors
                    constexpr auto cluster_length_reduce = GetClusterLengthReduction();
                    constexpr auto reduce_desc = make_cluster_descriptor(cluster_length_reduce);
                    const auto reduce_thread_cluster_idx =
                        reduce_desc.CalculateBottomIndex(make_multi_index(block_idx));
                    const auto thread_m_cluster_id = reduce_thread_cluster_idx[I0];
                    const auto thread_n_cluster_id = reduce_thread_cluster_idx[I1];

                    constexpr auto MReduceIters = math::integer_divide_ceil(
                        Number<MPerBlock>{}, cluster_length_reduce.At(I0));
                    constexpr auto NReduceIters = math::integer_divide_ceil(
                        Number<NPerBlock>{},
                        cluster_length_reduce.At(I1) *
                            Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{});

                    constexpr auto acc_thread_buf_load_desc = make_naive_tensor_descriptor_packed(
                        make_tuple(I1, Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{}));
                    constexpr auto acc_thread_buf_store_desc =
                        make_naive_tensor_descriptor_packed(make_tuple(
                            I1, I1, I1, Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{}));

                    constexpr auto c_partial_acc_block_m_n = GetPartialAccBlockDescriptor();

                    constexpr auto partial_acc_load_step_n =
                        make_multi_index(0,
                                         cluster_length_reduce.At(I1) *
                                             CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_load_step_n_reverse = make_multi_index(
                        0,
                        -1 * cluster_length_reduce.At(I1).value * (NReduceIters - 1) *
                            CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_load_step_m =
                        make_multi_index(cluster_length_reduce.At(I0), 0);

                    constexpr auto partial_acc_store_step_n =
                        make_multi_index(0,
                                         0,
                                         0,
                                         cluster_length_reduce.At(I1) *
                                             CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_store_step_n_reverse = make_multi_index(
                        0,
                        0,
                        0,
                        -1 * cluster_length_reduce.At(I1).value * (NReduceIters - 1) *
                            CShuffleBlockTransferScalarPerVector_NPerBlock);
                    constexpr auto partial_acc_store_step_m =
                        make_multi_index(0, cluster_length_reduce.At(I0), 0, 0);

                    StaticBuffer<AddressSpaceEnum::Vgpr,
                                 AccDataType,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock,
                                 true>
                        parcial_acc_buf;
                    StaticBuffer<AddressSpaceEnum::Vgpr,
                                 AccDataType,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock,
                                 true>
                        acc_buf;

                    // start to compute
                    auto reduction_idx =
                        block_idx - block_2_ctile_map_streamk.reduction_start_block_idx;
                    auto spatial_idx = block_2_ctile_map_streamk.tile_to_spatial(
                        reduction_idx, problem.M, problem.N);

                    workgroup_barrier wg_barrier(p_semaphore);

                    uint32_t tile_acc_offset_start =
                        block_2_ctile_map_streamk.get_acc_buffer_offset_from_tile(reduction_idx);
                    uint32_t tile_acc_offset_end =
                        block_2_ctile_map_streamk.get_acc_buffer_offset_from_tile(reduction_idx +
                                                                                  1);

                    uint32_t expected_count = tile_acc_offset_end - tile_acc_offset_start;

                    if(threadIdx.x == 0)
                    {
                        p_semaphore[reduction_idx] = 0;
                    }

                    __syncthreads();

                    auto acc_load = ThreadwiseTensorSliceTransfer_v2<
                        AccDataType,                        // SrcData,
                        AccDataType,                        // DstData,
                        decltype(c_partial_acc_block_m_n),  // SrcDesc,
                        decltype(acc_thread_buf_load_desc), // DstDesc,
                        Sequence<1,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock>, // SliceLengths,
                        Sequence<0, 1>,                                           // DimAccessOrder,
                        1,                                                        // SrcVectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // SrcScalarPerVector,
                        1,                                              // SrcScalarStrideInVector,
                        false // SrcResetCoordinateAfterRun,
                        >{c_partial_acc_block_m_n,
                          make_multi_index(thread_m_cluster_id,
                                           thread_n_cluster_id *
                                               CShuffleBlockTransferScalarPerVector_NPerBlock)};

                    auto acc_store = ThreadwiseTensorSliceTransfer_v1r3<
                        AccDataType,                                             // SrcData,
                        CDataType,                                               // DstData,
                        decltype(acc_thread_buf_store_desc),                     // SrcDesc,
                        decltype(c_grid_desc_mblock_mperblock_nblock_nperblock), // DstDesc,
                        CElementwiseOperation, // ElementwiseOperation,
                        Sequence<1,
                                 1,
                                 1,
                                 CShuffleBlockTransferScalarPerVector_NPerBlock>, // SliceLengths,
                        Sequence<0, 1, 2, 3>,                                     // DimAccessOrder,
                        3,                                                        // DstVectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // DstScalarPerVector,
                        InMemoryDataOperationEnum::Set, // InMemoryDataOperationEnum DstInMemOp,
                        1,                              // DstScalarStrideInVector,
                        false                           // DstResetCoordinateAfterRun,
                        >{c_grid_desc_mblock_mperblock_nblock_nperblock,
                          make_multi_index(__builtin_amdgcn_readfirstlane(spatial_idx[I0]),
                                           thread_m_cluster_id,
                                           __builtin_amdgcn_readfirstlane(spatial_idx[I1]),
                                           thread_n_cluster_id *
                                               CShuffleBlockTransferScalarPerVector_NPerBlock),
                          CElementwiseOperation{}};

#if 0
                if(threadIdx.x == 0) {
                    printf("bid:%d, rid:%d, os:%d,%d, spatial:%d,%d\n", static_cast<int>(blockIdx.x),
                        reduction_idx, __builtin_amdgcn_readfirstlane(tile_acc_offset_start), __builtin_amdgcn_readfirstlane(tile_acc_offset_end),
                        __builtin_amdgcn_readfirstlane(spatial_idx[I0]),
                        __builtin_amdgcn_readfirstlane(spatial_idx[I1]));
                }
#endif
                    if(threadIdx.x == 0)
                    {
                        atomicAdd(&p_semaphore[reduction_idx], 1);
                    }

                    wg_barrier.wait_eq(p_semaphore[reduction_idx], expected_count);
                    using Accumulation = ck::detail::
                        AccumulateWithNanCheck<false /*PropagateNan*/, reduce::Add, AccDataType>;

                    for(int i_m = 0; i_m < MReduceIters; i_m++)
                    {
                        static_for<0, NReduceIters, 1>{}([&](auto i_n_reduce) {
                            acc_buf.Clear();
                            for(auto i = tile_acc_offset_start; i < tile_acc_offset_end; i++)
                            {
                                auto c_partial_acc_buf =
                                    make_dynamic_buffer<AddressSpaceEnum::Global,
                                                        AmdBufferCoherenceEnum::GLC>(
                                        reinterpret_cast<AccDataType*>(p_workspace) +
                                            i * c_partial_acc_block_m_n.GetElementSpaceSize(),
                                        c_partial_acc_block_m_n.GetElementSpaceSize());

                                acc_load.Run(c_partial_acc_block_m_n,
                                             c_partial_acc_buf,
                                             acc_thread_buf_load_desc,
                                             make_tuple(I0, I0),
                                             parcial_acc_buf);

                                static_for<0, CShuffleBlockTransferScalarPerVector_NPerBlock, 1>{}(
                                    [&](auto i_vec) {
                                        constexpr auto offset =
                                            acc_thread_buf_load_desc.CalculateOffset(
                                                make_tuple(0, i_vec));
                                        Accumulation::Calculate(acc_buf(Number<offset>{}),
                                                                parcial_acc_buf[Number<offset>{}]);
                                    });
                            }

                            if(thread_n_cluster_id *
                                   CShuffleBlockTransferScalarPerVector_NPerBlock <
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
                            acc_store.MoveDstSliceWindow(
                                c_grid_desc_mblock_mperblock_nblock_nperblock,
                                partial_acc_store_step_m);
                        }
                    }

                    continue;
                }
            }

            // offset for last acc buffer of this block
            uint32_t block_acc_offset =
                (block_2_ctile_map_streamk.get_acc_buffer_offset_from_block(block_idx + 1) - 1) *
                MPerBlock * NPerBlock;
            while(true)
            {

                uint32_t current_iter_length = __builtin_amdgcn_readfirstlane(
                    block_2_ctile_map_streamk.get_current_iter_length(
                        iter_start, iter_end, num_k_block_main_loop));
                uint32_t tile_idx, iter_offset;
                block_2_ctile_map_streamk.get_tile_idx_with_offset(
                    iter_end - 1, tile_idx, iter_offset);
                iter_offset = __builtin_amdgcn_readfirstlane(iter_offset - current_iter_length + 1);

                auto block_work_idx =
                    block_2_ctile_map_streamk.tile_to_spatial(tile_idx, problem.M, problem.N);

                const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
                const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

                // HACK: this force m/n_block_data_idx_on_grid into SGPR
                const index_t m_block_data_idx_on_grid =
                    __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

                const index_t n_block_data_idx_on_grid =
                    __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);
                const index_t k0_block_data_idx_on_grid =
                    __builtin_amdgcn_readfirstlane(iter_offset * AK0Number);

                // lds max alignment
                constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

                // A matrix in LDS memory, dst of blockwise copy
                constexpr auto a_block_desc_ak0_m_ak1 =
                    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

                // B matrix in LDS memory, dst of blockwise copy
                constexpr auto b_block_desc_bk0_n_bk1 =
                    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

                // A matrix blockwise copy
                auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    AElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ADataType,
                    ADataType,
                    decltype(a_grid_desc_ak0_m_ak1),
                    decltype(a_block_desc_ak0_m_ak1),
                    ABlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    1,
                    1,
                    AThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(k0_block_data_idx_on_grid, m_block_data_idx_on_grid, 0),
                    a_element_op,
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

                // B matrix blockwise copy
                auto b_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    BElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BDataType,
                    BDataType,
                    decltype(b_grid_desc_bk0_n_bk1),
                    decltype(b_block_desc_bk0_n_bk1),
                    BBlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    1,
                    1,
                    BThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(k0_block_data_idx_on_grid, n_block_data_idx_on_grid, 0),
                    b_element_op,
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

                // LDS allocation for A and B: be careful of alignment
                constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
                    a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

                auto a_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<ADataType*>(p_shared_0),
                    a_block_desc_ak0_m_ak1.GetElementSpaceSize());

                auto b_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<BDataType*>(p_shared_0) +
                        a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
                    b_block_desc_bk0_n_bk1.GetElementSpaceSize());

                auto a_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<ADataType*>(p_shared_1),
                    a_block_desc_ak0_m_ak1.GetElementSpaceSize());

                auto b_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<BDataType*>(p_shared_1) +
                        a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
                    b_block_desc_bk0_n_bk1.GetElementSpaceSize());

                auto a_block_bufs = make_tuple(a_block_buf_ping, a_block_buf_pong);
                auto b_block_bufs = make_tuple(b_block_buf_ping, b_block_buf_pong);

                constexpr auto a_block_slice_copy_step =
                    make_multi_index(KPerBlock / AK1Number, 0, 0);
                constexpr auto b_block_slice_copy_step =
                    make_multi_index(KPerBlock / BK1Number, 0, 0);

                // Blockwise GEMM pipeline
                static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
                auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
                auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

                num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
                    (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
                    KPerBlock);

                blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(
                    a_grid_desc_ak0_m_ak1,
                    a_block_desc_ak0_m_ak1,
                    a_blockwise_copy,
                    a_grid_buf,
                    a_block_bufs,
                    a_block_slice_copy_step,
                    b_grid_desc_bk0_n_bk1,
                    b_block_desc_bk0_n_bk1,
                    b_blockwise_copy,
                    b_grid_buf,
                    b_block_bufs,
                    b_block_slice_copy_step,
                    c_thread_buf,
                    num_k_block_main_loop);

                // shuffle C and write out
                {
                    static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                                      NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                                  "wrong!");

                    constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
                    constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

                    // TODO: hacky, fix it!
                    constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                        blockwise_gemm_pipeline.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                    // TODO: hacky, fix it!
                    // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
                    constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                        blockwise_gemm_pipeline.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

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

                    constexpr auto c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle =
                        GetCBlockDescriptor_MShuffle_MPerShuffle_NShuffle_NPerShuffle();

                    auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                        static_cast<CShuffleDataType*>(p_shared_0),
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock
                            .GetElementSpaceSize());

                    auto c_partial_acc_buf =
                        make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::GLC>(
                            reinterpret_cast<AccDataType*>(p_workspace) + block_acc_offset,
                            c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle
                                .GetElementSpaceSize());

                    constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                        transform_tensor_descriptor(
                            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                            make_tuple(
                                make_freeze_transform(I0),
                                make_unmerge_transform(make_tuple(
                                    Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per
                                                                             // shuffle
                                    M1,                                      // M1 = MWave
                                    M2, // M2 * M3 * M4 = MPerXdl
                                    M3,
                                    M4)),
                                make_freeze_transform(I0),
                                make_unmerge_transform(make_tuple(
                                    Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per
                                                                             // shuffle
                                    N1,                                      // N1 = NWave
                                    N2))),                                   // N2 = NPerXdl
                            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                            make_tuple(Sequence<>{},
                                       Sequence<0, 2, 4, 5, 6>{},
                                       Sequence<>{},
                                       Sequence<1, 3, 7>{}));

                    // calculate origin of thread output tensor on global memory
                    //     blockwise GEMM c matrix starting index
                    const auto c_thread_mtx_on_block =
                        blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

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
                        AccDataType,
                        CShuffleDataType,
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
                    auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1r2<
                        ThisThreadBlock,       // ThreadGroup
                        CElementwiseOperation, // ElementwiseOperation,
                        // CGlobalMemoryDataOperation, // DstInMemOp,
                        Sequence<1,
                                 CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                 1,
                                 CShuffleNXdlPerWavePerShuffle * NWave *
                                     NPerXdl>, // BlockSliceLengths,
                        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                        Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                        CShuffleDataType,     // typename SrcData,
                        CDataType,            // typename DstData,
                        decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                        decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                        Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                        3,                                              // index_t VectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                        false, // bool ThreadTransferSrcResetCoordinateAfterRun,
                        false> // bool ThreadTransferDstResetCoordinateAfterRun>
                        {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                         make_multi_index(0, 0, 0, 0),
                         c_grid_desc_mblock_mperblock_nblock_nperblock,
                         make_multi_index(block_m_id, 0, block_n_id, 0),
                         c_element_op};

                    // LDS to global partial acc
                    auto c_block_copy_lds_to_partial_acc = ThreadGroupTensorSliceTransfer_v6r1r2<
                        ThisThreadBlock,       // index_t BlockSize,
                        CElementwiseOperation, // ElementwiseOperation,
                                               // InMemoryDataOperationEnum::Set, // DstInMemOp,
                        Sequence<1,
                                 CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                 1,
                                 CShuffleNXdlPerWavePerShuffle * NWave *
                                     NPerXdl>, // BlockSliceLengths,
                        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                        Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                        CShuffleDataType,     // typename SrcData,
                        AccDataType,          // typename DstData,
                        decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                        decltype(c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle),
                        Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                        3,                                              // index_t VectorDim,
                        CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                        false, // bool ThreadTransferSrcResetCoordinateAfterRun, => need to be
                               // false, othre wise has scratch
                        false> // bool ThreadTransferDstResetCoordinateAfterRun, => need to be
                               // false, othre wise has scratch
                        {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                         make_multi_index(0, 0, 0, 0),
                         c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                         make_multi_index(0, 0, 0, 0),
                         c_element_op};

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
                    constexpr auto sfc_c_global = SpaceFillingCurve<
                        Sequence<1, MPerBlock, 1, NPerBlock>,
                        Sequence<0, 2, 1, 3>,
                        Sequence<1,
                                 CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                 1,
                                 CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

                    constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

                    static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

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
                        c_shuffle_block_copy_lds_to_global.SetSrcSliceOrigin(
                            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                            make_tuple(0, 0, 0, 0));

                        if(is_dp_block)
                        {
                            // each block copy its data from LDS to global
                            c_shuffle_block_copy_lds_to_global
                                .template Run<decltype(c_shuffle_block_buf),
                                              decltype(c_grid_buf),
                                              InMemoryDataOperationEnum::Set>(
                                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                    c_shuffle_block_buf,
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    c_grid_buf);
                        }
                        else if(is_sk_block)
                        {
                            if(problem.reduction_strategy == StreamKReductionStrategy::Atomic)
                            {
                                // each block copy its data from LDS to global
                                c_shuffle_block_copy_lds_to_global
                                    .template Run<decltype(c_shuffle_block_buf),
                                                  decltype(c_grid_buf),
                                                  InMemoryDataOperationEnum::AtomicAdd>(
                                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                        c_shuffle_block_buf,
                                        c_grid_desc_mblock_mperblock_nblock_nperblock,
                                        c_grid_buf);
                            }
                            else if(problem.reduction_strategy ==
                                    StreamKReductionStrategy::Reduction)
                            {
                                // constexpr offset
                                c_block_copy_lds_to_partial_acc.SetSrcSliceOrigin(
                                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                    make_tuple(0, 0, 0, 0));

                                c_block_copy_lds_to_partial_acc.SetDstSliceOrigin(
                                    c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                                    make_tuple(MXdlPerWave, 0, NXdlPerWave, 0));

                                c_block_copy_lds_to_partial_acc
                                    .template Run<decltype(c_shuffle_block_buf),
                                                  decltype(c_partial_acc_buf),
                                                  InMemoryDataOperationEnum::Set>(
                                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                                        c_shuffle_block_buf,
                                        c_block_desc_mshuffle_mpershuffle_nshuffle_npershuffle,
                                        c_partial_acc_buf);
                            }
                        }
                        if constexpr(access_id < num_access - 1)
                        {
                            constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                            // move on C
                            c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                                c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                        }
                    });
                }
                // exit condition
                iter_end -= current_iter_length;
                if(iter_end <= iter_start)
                    break;
                if(problem.reduction_strategy == StreamKReductionStrategy::Reduction)
                {
                    block_acc_offset -= MPerBlock * NPerBlock;
                }
                // make sure next loop LDS is ready for use
                block_sync_lds();
            }
            if(problem.reduction_strategy == StreamKReductionStrategy::Reduction)
            {
                if(is_sk_block)
                {
                    // increase the counter for this tile
                    workgroup_barrier wg_barrier(p_semaphore);
                    wg_barrier.inc(0);
                }
            }
        }
    }
};

} // namespace ck
