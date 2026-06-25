// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename AScaleType,
          typename BsDataType,
          typename BScaleType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t ScaleBlockM,
          index_t ScaleBlockN, // scale N
          index_t ScaleBlockK, // scale K
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
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
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA,
          typename ComputeTypeB,
          bool PermuteA,
          bool PermuteB,
          bool IsBPreShuffled   = false,
          typename AScaleLayout = ALayout,
          typename BScaleLayout = BLayout>
struct GridwiseGemm_wmma_cshuffle_v3_ab_scale
    : GridwiseGemm_wmma_cshuffle_v3_base<
          ALayout,
          BLayout,
          DsLayout,
          ELayout,
          AsDataType,
          BsDataType,
          AccDataType,
          CShuffleDataType,
          DsDataType,
          EDataType,
          AElementwiseOperation,
          BElementwiseOperation,
          CDEElementwiseOperation,
          GemmSpec,
          BlockSize,
          MPerBlock,
          NPerBlock,
          KPerBlock,
          AK1Value,
          BK1Value,
          MPerWmma,
          NPerWmma,
          MRepeat,
          NRepeat,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_AK1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraM,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_BK1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraN,
          CShuffleMRepeatPerShuffle,
          CShuffleNRepeatPerShuffle,
          CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CDEShuffleBlockTransferScalarPerVectors,
          BlkGemmPipeSched,
          BlkGemmPipelineVer,
          ComputeTypeA,
          ComputeTypeB,
          PermuteA,
          PermuteB,
          IsBPreShuffled,
          true>
{
    using Base = GridwiseGemm_wmma_cshuffle_v3_base<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        AsDataType,
        BsDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1Value,
        BK1Value,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB,
        IsBPreShuffled,
        true>;

    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using Base::I4;
    using Base::I5;
    using Base::I6;
    using Base::I7;

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;

    using Base::APackedSize;
    using Base::BPackedSize;

    using Base::CalculateAK0Padded;
    using Base::CalculateBK0Padded;
    using Base::CalculateKPadded;
    using Base::CalculateKRead;
    using Base::CalculateMBlock;
    using Base::CalculateMPadded;
    using Base::CalculateNBlock;
    using Base::CalculateNPadded;
    using Base::MakeAsGridDescriptor_AK0_M_AK1;
    using Base::MakeBsGridDescriptor_BK0_N_BK1;
    using Base::MakeDEGridDescriptor_M_N;
    using Base::MakeDsGridDescriptor_M_N;
    using Base::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock;

    using Base::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using Base::NumATensor;
    using Base::NumBTensor;
    using Base::NumDTensor;
    using typename Base::AsGridPointer;
    using typename Base::BsGridPointer;
    using typename Base::DsGridPointer;
    using AsDataType_ = AsDataType;
    using BsDataType_ = BsDataType;

    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         std::array<index_t, NumATensor> StrideAs_,
                         std::array<index_t, NumBTensor> StrideBs_,
                         std::array<index_t, NumDTensor> StrideDs_,
                         index_t StrideE_,
                         index_t StrideScaleA_,
                         index_t StrideScaleB_,
                         index_t KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideAs{StrideAs_},
              StrideBs{StrideBs_},
              StrideDs{StrideDs_},
              StrideE{StrideE_},
              StrideScaleA{StrideScaleA_},
              StrideScaleB{StrideScaleB_},
              KBatch{KBatch_},
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              KRead{CalculateKRead(K_, KBatch_)},
              KPadded{CalculateKPadded(K_, KBatch_)},
              AK0{CalculateAK0Padded(K_, KBatch_)},
              BK0{CalculateBK0Padded(K_, KBatch_)},
              MBlock{CalculateMBlock(M_)},
              NBlock{CalculateNBlock(N_)},
              Kt{K_}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SAs: {";
            static_for<0, NumATensor, 1>{}([&](auto i) {
                std::cout << StrideAs[i] << (i.value < NumATensor - 1 ? ", " : "");
            });
            std::cout << "}, " << "SBs: {";
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                std::cout << StrideBs[i] << (i.value < NumBTensor - 1 ? ", " : "");
            });
            std::cout << "}, ";
            if constexpr(NumDTensor > 0)
            {
                std::cout << "SDs: { ";
                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    std::cout << StrideDs[i] << (i.value < NumDTensor - 1 ? ", " : "");
                });
                std::cout << " }, ";
            }
            std::cout << "SE:" << StrideE << ", " << "SScaleA:" << StrideScaleA << ", "
                      << "SScaleB:" << StrideScaleB << ", " << "MP:" << MPadded << ", "
                      << "NP:" << NPadded << ", " << "KRead:" << KRead << ", " << "KP:" << KPadded
                      << ", " << "AK0:" << AK0 << ", " << "BK0:" << BK0 << ", "
                      << "MBlock: " << MBlock << ", " << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        std::array<index_t, NumATensor> StrideAs;
        std::array<index_t, NumBTensor> StrideBs;
        std::array<index_t, NumDTensor> StrideDs;
        index_t StrideE;
        index_t StrideScaleA;
        index_t StrideScaleB;
        index_t KBatch;
        index_t MPadded;
        index_t NPadded;
        index_t KRead;
        index_t KPadded;
        index_t AK0;
        index_t BK0;
        index_t MBlock;
        index_t NBlock;
        index_t Kt;
    };

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument(std::array<const void*, NumATensor> p_as_grid_,
                          std::array<const void*, NumBTensor> p_bs_grid_,
                          std::array<const void*, NumDTensor> p_ds_grid_,
                          EDataType* p_e_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          std::array<index_t, NumATensor> StrideAs_,
                          std::array<index_t, NumBTensor> StrideBs_,
                          std::array<index_t, NumDTensor> StrideDs_,
                          index_t StrideE_,
                          index_t StrideScaleA_,
                          index_t StrideScaleB_,
                          const AScaleType* p_a_scale_grid_,
                          const BScaleType* p_b_scale_grid_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CDEElementwiseOperation cde_element_op_,
                          bool is_reduce_ = false)
            : Problem{M_,
                      N_,
                      K_,
                      StrideAs_,
                      StrideBs_,
                      StrideDs_,
                      StrideE_,
                      StrideScaleA_,
                      StrideScaleB_,
                      k_batch_},
              p_as_grid{},
              p_bs_grid{},
              p_ds_grid{},
              p_e_grid{p_e_grid_},
              p_a_scale_grid{p_a_scale_grid_},
              p_b_scale_grid{p_b_scale_grid_},
              a_element_op{a_element_op_},
              b_element_op{b_element_op_},
              cde_element_op{cde_element_op_},
              is_reduce(is_reduce_)
        {
            // populate pointer, desc for As
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

                // A pointer
                p_as_grid(i) = static_cast<const ADataType_*>(p_as_grid_[i]);
            });

            // populate pointer, desc for Bs
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

                // B pointer
                p_bs_grid(i) = static_cast<const BDataType_*>(p_bs_grid_[i]);
            });

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                p_ds_grid(i) = static_cast<const DDataType*>(p_ds_grid_[i]);
            });
        }

        __host__ __device__ inline bool IsReduceAdd() const
        {
            return (Problem::KBatch > 1) && is_reduce;
        }

        __host__ __device__ inline bool IsAtomicAdd() const
        {
            return (Problem::KBatch > 1) && (!is_reduce);
        }

        AsGridPointer p_as_grid;
        BsGridPointer p_bs_grid;
        DsGridPointer p_ds_grid;
        EDataType* p_e_grid;

        const AScaleType* p_a_scale_grid;
        const BScaleType* p_b_scale_grid;
        const AElementwiseOperation a_element_op;
        const BElementwiseOperation b_element_op;
        const CDEElementwiseOperation cde_element_op;
        bool is_reduce;
    };

    struct SplitKBatchOffset
    {

        __device__ SplitKBatchOffset(Argument& karg, index_t k_id)
        {
            // Note: in xdl implementation multiple AB supports one layout
            // but multiple strides, so we create an array of offsets with
            // the same values.
            // It should be fixed later on. Once we will have a thread transfer
            // more flexible.
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                static_for<0, NumATensor, 1>{}(
                    [&](auto i) { a_k_split_offset[i] = k_id * karg.KRead / APackedSize; });
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                static_for<0, NumATensor, 1>{}(
                    [&](auto i) { a_k_split_offset[i] = k_id * karg.KRead * karg.StrideAs[i]; });
            }

            if constexpr(IsBPreShuffled)
            {
                static_for<0, NumBTensor, 1>{}([&](auto i) { b_k_split_offset[i] = 0; });
            }
            else
            {
                if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
                {
                    static_for<0, NumBTensor, 1>{}([&](auto i) {
                        b_k_split_offset[i] = k_id * karg.KRead * karg.StrideBs[i];
                    });
                }
                else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
                {
                    if constexpr(!PermuteB)
                    {
                        static_for<0, NumBTensor, 1>{}(
                            [&](auto i) { b_k_split_offset[i] = k_id * karg.KRead / BPackedSize; });
                    }
                    else
                    {
                        const int k0_offset = karg.KRead * karg.N;
                        static_for<0, NumBTensor, 1>{}(
                            [&](auto i) { b_k_split_offset[i] = k_id * k0_offset / BPackedSize; });
                    }
                }
            }

            // Calculate A scale offset
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, AScaleLayout>)
            {
                scale_a_k_split_offset = k_id * (karg.KRead / ScaleBlockK);
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, AScaleLayout>)
            {
                scale_a_k_split_offset = k_id * (karg.KRead / ScaleBlockK) * karg.StrideScaleA;
            }

            // Calculate B scale offset
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BScaleLayout>)
            {
                scale_b_k_split_offset = k_id * (karg.KRead / ScaleBlockK) * karg.StrideScaleB;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BScaleLayout>)
            {
                scale_b_k_split_offset = k_id * (karg.KRead / ScaleBlockK);
            }

            if(k_id < karg.KBatch - 1)
            {
                karg.K = karg.KRead;
            }
            else
            {
                karg.K = karg.K - karg.KRead * (karg.KBatch - 1);
            }

            if(karg.IsReduceAdd())
            {
                c_reduce_offset = k_id * karg.M * karg.N;
            }
            else
            {
                c_reduce_offset = 0;
            }
        }

        std::array<index_t, NumATensor> a_k_split_offset;
        std::array<index_t, NumBTensor> b_k_split_offset;
        index_t scale_a_k_split_offset; // A scale matrix offset
        index_t scale_b_k_split_offset; // B scale matrix offset
        index_t c_reduce_offset;
    };

    using BlockwiseGemmPipe = typename Base::BlockwiseGemmPipe;

    // return block_id to C matrix tile idx (m0, n0) mapping
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;

    __device__ static constexpr auto
    MakeAScaleGridDesciptor_M_K(index_t M, index_t K, index_t StrideScaleA)
    {
        const auto BM = math::integer_divide_ceil(M, ScaleBlockM);
        const auto BK = math::integer_divide_ceil(K, ScaleBlockK);
        if constexpr(is_same<tensor_layout::gemm::RowMajor, AScaleLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(BM, BK), make_tuple(StrideScaleA, I1));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, AScaleLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(BM, BK), make_tuple(I1, StrideScaleA));
        }
    }

    template <index_t NumberOfBuffers>
    __device__ static auto
    MakeAScale(const Problem& problem, const AScaleType* p_a_scale_grid, index_t block_m_id)
    {
        if constexpr(ck::is_same_v<AScaleType, void>)
        {
            using AScale = typename BlockwiseGemmPipe::Empty;
            return AScale{};
        }
        else
        {
            static_assert(
                ScaleBlockK >=
                    WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::
                        selected_wmma.k_per_wmma,
                "ScaleBlockK must be greater equal than KPerWmma");

            const auto a_scale_grid_desc_am_ak =
                MakeAScaleGridDesciptor_M_K(problem.M, problem.K, problem.StrideScaleA);

            const auto a_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_a_scale_grid, a_scale_grid_desc_am_ak.GetElementSpaceSize());

            constexpr auto wmma =
                WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>{};
            constexpr auto RegSizePerWmmaFull =
                wmma.selected_wmma.num_acc_vgprs_per_wave * wmma.selected_wmma.acc_pack_number;
            constexpr auto RegSizePerWmma =
                math::integer_divide_ceil(RegSizePerWmmaFull, ScaleBlockM);

            constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);
            constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

            constexpr auto ScaleSliceSizeM =
                ScaleBlockM < MPerWmma ? MRepeat * RegSizePerWmma
                                       : math::integer_divide_ceil(MPerBlock, ScaleBlockM);
            constexpr auto ScaleSliceStrideM =
                math::integer_divide_ceil(MWaves * MPerWmma, ScaleBlockM);
            constexpr auto ScaleSliceSizeK = math::integer_divide_ceil(KPerBlock, ScaleBlockK);

            constexpr auto a_scale_thread_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<ScaleSliceSizeM>{}, Number<ScaleSliceSizeK>{}));

            auto a_thread_offset_m =
                ((get_thread_local_1d_id() % 32) / MPerWmma * RegSizePerWmma) /
                    math::integer_divide_ceil(ScaleBlockM, RegSizePerWmmaFull) +
                (get_thread_local_1d_id() / 32) / NWaves * MPerWmma / ScaleBlockM;

            constexpr index_t VectorDim =
                is_same<tensor_layout::gemm::ColumnMajor, AScaleLayout>::value ? 0 : 1;
            constexpr index_t VectorSize =
                is_same<tensor_layout::gemm::ColumnMajor, AScaleLayout>::value ? RegSizePerWmma
                                                                               : ScaleSliceSizeK;

            auto a_scale_thread_copy =
                ThreadwiseTensorSliceTransfer_v2<AScaleType,
                                                 AScaleType,
                                                 decltype(a_scale_grid_desc_am_ak),
                                                 decltype(a_scale_thread_desc),
                                                 Sequence<RegSizePerWmma, ScaleSliceSizeK>,
                                                 Sequence<0, 1>,
                                                 VectorDim,
                                                 VectorSize,
                                                 1,
                                                 true>(
                    a_scale_grid_desc_am_ak,
                    make_multi_index(block_m_id * MPerBlock / ScaleBlockM + a_thread_offset_m, 0));

            auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AScaleType>(
                a_scale_thread_desc.GetElementSpaceSize());

            using AScale =
                typename BlockwiseGemmPipe::template ABScale<ScaleSliceSizeM,
                                                             ScaleSliceStrideM,
                                                             ScaleSliceSizeK,
                                                             NumberOfBuffers,
                                                             RegSizePerWmma,
                                                             decltype(a_scale_grid_desc_am_ak),
                                                             decltype(a_scale_thread_copy),
                                                             decltype(a_scale_grid_buf),
                                                             decltype(a_scale_thread_buf),
                                                             decltype(a_scale_thread_desc)>;

            return AScale{a_scale_grid_desc_am_ak, a_scale_thread_copy, a_scale_grid_buf};
        }
    }

    __device__ static constexpr auto
    MakeBScaleGridDesciptor_N_K(index_t N, index_t K, index_t StrideScaleB)
    {
        const auto BN = math::integer_divide_ceil(N, ScaleBlockN);
        const auto BK = math::integer_divide_ceil(K, ScaleBlockK);
        if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BScaleLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(BN, BK), make_tuple(StrideScaleB, I1));
        }
        else if constexpr(is_same<tensor_layout::gemm::RowMajor, BScaleLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(BN, BK), make_tuple(I1, StrideScaleB));
        }
    }

    template <index_t NumberOfBuffers>
    __device__ static auto
    MakeBScale(const Problem& problem, const BScaleType* p_b_scale_grid, index_t block_n_id)
    {
        if constexpr(ck::is_same_v<BScaleType, void>)
        {
            using BScale = typename BlockwiseGemmPipe::Empty;
            return BScale{};
        }
        else
        {
            static_assert(
                ScaleBlockK >=
                    WmmaSelector<ComputeTypeA, ComputeTypeB, AccDataType, MPerWmma, NPerWmma>::
                        selected_wmma.k_per_wmma,
                "ScaleBlockK must be greater equal than KPerWmma");

            const auto b_scale_grid_desc_bn_ak =
                MakeBScaleGridDesciptor_N_K(problem.N, problem.K, problem.StrideScaleB);

            const auto b_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_b_scale_grid, b_scale_grid_desc_bn_ak.GetElementSpaceSize());

            constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

            constexpr auto ScaleSliceSizeN =
                ScaleBlockN < NPerWmma ? NRepeat
                                       : math::integer_divide_ceil(NPerBlock, ScaleBlockN);
            constexpr auto ScaleSliceStrideN =
                math::integer_divide_ceil(NWaves * NPerWmma, ScaleBlockN);
            constexpr auto ScaleSliceSizeK = math::integer_divide_ceil(KPerBlock, ScaleBlockK);

            constexpr auto b_scale_thread_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<ScaleSliceSizeN>{}, Number<ScaleSliceSizeK>{}));

            auto b_thread_offset_n = (get_thread_local_1d_id() % NPerWmma +
                                      (get_thread_local_1d_id() / 32) % NWaves * NPerWmma) /
                                     ScaleBlockN;

            constexpr index_t VectorDim =
                is_same<tensor_layout::gemm::RowMajor, BScaleLayout>::value ? 0 : 1;
            constexpr index_t VectorSize =
                is_same<tensor_layout::gemm::RowMajor, BScaleLayout>::value ? 1 : ScaleSliceSizeK;

            auto b_scale_thread_copy =
                ThreadwiseTensorSliceTransfer_v2<BScaleType,
                                                 BScaleType,
                                                 decltype(b_scale_grid_desc_bn_ak),
                                                 decltype(b_scale_thread_desc),
                                                 Sequence<1, ScaleSliceSizeK>,
                                                 Sequence<0, 1>,
                                                 VectorDim,
                                                 VectorSize,
                                                 1,
                                                 true>(
                    b_scale_grid_desc_bn_ak,
                    make_multi_index(block_n_id * NPerBlock / ScaleBlockN + b_thread_offset_n, 0));

            auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BScaleType>(
                b_scale_thread_desc.GetElementSpaceSize());

            using BScale =
                typename BlockwiseGemmPipe::template ABScale<ScaleSliceSizeN,
                                                             ScaleSliceStrideN,
                                                             ScaleSliceSizeK,
                                                             NumberOfBuffers,
                                                             1,
                                                             decltype(b_scale_grid_desc_bn_ak),
                                                             decltype(b_scale_thread_copy),
                                                             decltype(b_scale_grid_buf),
                                                             decltype(b_scale_thread_buf),
                                                             decltype(b_scale_thread_desc)>;

            return BScale{b_scale_grid_desc_bn_ak, b_scale_thread_copy, b_scale_grid_buf};
        }
    }

    __device__ static index_t GetKBlockPerScale()
    {
        if constexpr(ck::is_same_v<AScaleType, void> && ck::is_same_v<BScaleType, void>)
        {
            return 0;
        }
        else
        {
            return (ScaleBlockK + KPerBlock - 1) / KPerBlock;
        }
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename EpilogueArgument>
    __device__ static void Run(AsGridPointer& p_as_grid,
                               BsGridPointer& p_bs_grid,
                               DsGridPointer& p_ds_grid,
                               EDataType* p_e_grid,
                               const AScaleType* p_a_scale_grid,
                               const BScaleType* p_b_scale_grid,
                               void* p_shared,
                               const Problem& problem,
                               AElementwiseOperation a_element_op,
                               BElementwiseOperation b_element_op,
                               CDEElementwiseOperation cde_element_op,
                               EpilogueArgument& epilogue_args,
                               const index_t A_k_id = 0,
                               const index_t B_k_id = 0)
    {
        const auto as_grid_desc_ak0_m_ak1 = MakeAsGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideAs, problem.AK0);
        const index_t K_b                 = IsBPreShuffled ? problem.Kt : problem.K;
        const auto bs_grid_desc_bk0_n_bk1 = MakeBsGridDescriptor_BK0_N_BK1(
            K_b, problem.KPadded, problem.N, problem.NPadded, problem.StrideBs, problem.BK0);
        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideDs);
        const auto e_grid_desc_m_n = Base::template MakeDEGridDescriptor_M_N<ELayout>(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideE);
        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_grid_desc_m_n, problem.MBlock, problem.NBlock);
        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                e_grid_desc_m_n, problem.MBlock, problem.NBlock);

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // AScale struct
        auto a_scale_struct = MakeAScale<1>(problem, p_a_scale_grid, block_m_id);

        // BScale struct
        auto b_scale_struct = MakeBScale<1>(problem, p_b_scale_grid, block_n_id);

        const index_t num_k_block_per_scale = GetKBlockPerScale();

        Base::template Run<decltype(as_grid_desc_ak0_m_ak1),
                           decltype(bs_grid_desc_bk0_n_bk1),
                           decltype(ds_grid_desc_mblock_mperblock_nblock_nperblock),
                           decltype(e_grid_desc_mblock_mperblock_nblock_nperblock),
                           decltype(a_scale_struct),
                           decltype(b_scale_struct),
                           decltype(epilogue_args),
                           HasMainKBlockLoop,
                           EGlobalMemoryDataOperation,
                           TailNum>(p_as_grid,
                                    p_bs_grid,
                                    p_ds_grid,
                                    p_e_grid,
                                    p_shared,
                                    as_grid_desc_ak0_m_ak1,
                                    bs_grid_desc_bk0_n_bk1,
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                    e_grid_desc_mblock_mperblock_nblock_nperblock,
                                    a_element_op,
                                    b_element_op,
                                    cde_element_op,
                                    block_m_id,
                                    block_n_id,
                                    num_k_block_per_scale,
                                    a_scale_struct,
                                    b_scale_struct,
                                    epilogue_args,
                                    A_k_id,
                                    B_k_id);
    }

    // NOTE: Wrapper function to have __global__ function in common
    // between gemm_universal, b_scale, ab_scale, etc.
    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename EpilogueArgument>
    __device__ static void Run(void* p_shared,
                               const SplitKBatchOffset& splitk_batch_offset,
                               Argument& karg,
                               EpilogueArgument& epilogue_args,
                               const index_t A_k_id = 0,
                               const index_t B_k_id = 0)
    {
        // shift A matrices pointer for splitk
        AsGridPointer p_as_grid_splitk;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType_    = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            p_as_grid_splitk(i) = static_cast<const ADataType_*>(karg.p_as_grid[i]) +
                                  splitk_batch_offset.a_k_split_offset[i];
        });

        // shift B matrices pointer for splitk
        BsGridPointer p_bs_grid_splitk;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType_    = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            p_bs_grid_splitk(i) = static_cast<const BDataType_*>(karg.p_bs_grid[i]) +
                                  splitk_batch_offset.b_k_split_offset[i];
        });

        const AScaleType* p_a_scale_grid_ptr;
        if constexpr(ck::is_same_v<AScaleType, void>)
        {
            p_a_scale_grid_ptr = karg.p_a_scale_grid;
        }
        else
        {
            p_a_scale_grid_ptr = karg.p_a_scale_grid + splitk_batch_offset.scale_a_k_split_offset;
        }

        const BScaleType* p_b_scale_grid_ptr;
        if constexpr(ck::is_same_v<BScaleType, void>)
        {
            p_b_scale_grid_ptr = karg.p_b_scale_grid;
        }
        else
        {
            p_b_scale_grid_ptr = karg.p_b_scale_grid + splitk_batch_offset.scale_b_k_split_offset;
        }

        Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_as_grid_splitk,
            p_bs_grid_splitk,
            karg.p_ds_grid,
            karg.p_e_grid + splitk_batch_offset.c_reduce_offset,
            p_a_scale_grid_ptr,
            p_b_scale_grid_ptr,
            p_shared,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.cde_element_op,
            epilogue_args,
            A_k_id,
            B_k_id);
    }
};

} // namespace ck
