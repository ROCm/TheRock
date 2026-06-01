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
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {

/// @brief \"Universal\" GEMM kernel with SplitK support.
///
/// @par Overview
///         This GEMM kernel is carrying out following mathematical equation:
///         E{M,N} = CDE_op(A_op(A{M,K}) * B_op(B{K,N}), Ds{M,N}...)
///         Where A, B, Ds are input tensors and E is the output tensor. The A/B are elementwise
//          operations that could be applied on each tensor respectively. The CDE_op is an
//          elementwise operation applied to the C and all D tensors.
///         The \"universal\" gemm comes with multiple pipelines optimized for different usage
///         scenarios. That's why it's called \"universal\". It's universal through it's design
///         and versatilty.
///
/// @note   This Kernel implementation supports SplitK algorithm. It can be configured
///         to split the dot product accumulated over the K dimension into multiple working groups.
///         The partial products of different workgroups are then reduced using the AtomicAdd
///         operation.
///
/// @tparam ALayout     A tensor data layout.
/// @tparam BLayout     B tensor data layout.
/// @tparam DsLayout    D tensors data layouts.
/// @tparam ELayout     E tensor data layout.
/// @tparam AsDataType  A tensors data types.
/// @tparam BsDataType  B tensors data types.
/// @tparam AccDataType The accumulation data type related to the hardware
///                         matrix-multiplication instruction.
/// @tparam CShuffleDataType The data type used to store matrix-multiplication results into
///                          LDS memory during \"CShuffle\" data layout optimization.
/// @tparam DsDataType  D tensors data types.
/// @tparam EDataType   E tensor data type.
/// @tparam AElementwiseOperation   Elementwise operation applied to the A input tensor elements.
/// @tparam BElementwiseOperation   Elementwise operation applied to the B input tensor elements.
/// @tparam CDEElementwiseOperation Elementwise operation applied to the C output tensor (after
///                                 GEMM) and D input tensors.
/// @tparam GemmSpec    Determines used "padding" version.
/// @tparam BlockSize   The number of threads within workgroup.
/// @tparam MPerBlock   The input/output data tile size in the M dimension.
/// @tparam NPerBlock   The input/output data tile size in the N dimension.
/// @tparam KPerBlock   The input data tile size in the K dimension.
/// @tparam AK1Value    The vector load size from global memory for A tensor.
/// @tparam BK1Value    The vector load size from global memory for B tensor.
/// @tparam MPerWmma    M size of Wave Matrix Multiply Accumulate (WMMA) instruction.
/// @tparam NPerWmma    N size of Wave Matrix Multiply Accumulate (WMMA) instruction.
/// @tparam MRepeat     The number of iterations in the M dimension over output tile per wavefront.
/// @tparam NRepeat     The number of iterations in the N dimension over output tile per wavefront.
/// @tparam ABlockTransferThreadClusterLengths_AK0_M_AK1 Spatial thread distribution over the input
///                                                      data. Can be interpreted as the answer
///                                                      to the question, "How many threads can be
///                                                      arranged on each input data axis?"
/// @tparam ABlockTransferThreadClusterArrangeOrder The order of thread spatial distribution over
///                                                 the input tensor dimension. Can be interpreted
///                                                 as the answer to the question: "In which
///                                                 order to spread threads through tensor axes?".
/// @tparam ABlockTransferSrcAccessOrder The order of accessing input tensor axes. Can be
///                                      interpreted as the answer to the question "Which dimension
///                                      to read first? And which next?" etc.
/// @tparam ABlockTransferSrcVectorDim   The index of axis on which we could do vectorized memory
///                                      access - the one with contiguous memory.
/// @tparam ABlockTransferSrcScalarPerVector The size of vector access instruction - the number of
///                                          elements accessed per thread per instruction.
/// @tparam ABlockTransferDstScalarPerVector_AK1 The size of vectorized store into LDS memory.
/// @tparam AThreadTransferSrcResetCoordinateAfterRun   Decides whether we reset thread coordinate
///                          (return back to the window origin) after all thread finish data copy.
/// @tparam ABlockLdsExtraM                      Whether to use padding for LDS or not. With
///                                              universal GEMM there's no need for padding.
/// @tparam BBlockTransferThreadClusterLengths_BK0_N_BK1 Spatial thread distribution over the input
///                                                      data. Can be interpreted as the answer
///                                                      to the question: "How many threads to
///                                                      arrange on each input data axis?"
/// @tparam BBlockTransferThreadClusterArrangeOrder The order of thread spatial distribution over
///                                                 the input tensor dimension. Can be interpreted
///                                                 as the answer to the question: "In which
///                                                 order to spread threads through tensor axes?".
/// @tparam BBlockTransferSrcAccessOrder he order of accessing input tensor axes. Can be
///                                      interpreted as the answer to the question "Which dimension
///                                      to read first? And which next?" etc.
/// @tparam BBlockTransferSrcVectorDim  The index of axis on which we could do vectorized memory
///                                      access - the one with contiguous memory.
/// @tparam BBlockTransferSrcScalarPerVector The size of vector access instruction - the number of
///                                          elements accessed per thread per instruction.
/// @tparam BBlockTransferDstScalarPerVector_BK1 The size of vectorized store into LDS memory.
/// @tparam BThreadTransferSrcResetCoordinateAfterRun   Decides whether we reset thread coordinate
///                          (return back to the window origin) after all thread finish data copy.
/// @tparam BBlockLdsExtraN             Whether to use padding for LDS or not. With universal GEMM
///                                         there's no need for padding.
/// @tparam CShuffleMRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in M dimension.
/// @tparam CShuffleNRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in N dimension.
/// @tparam CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock The spatial
///                                         thread distribution used for storing data into output
///                                         tensor across output data layout dimensions.
/// @tparam CDEShuffleBlockTransferScalarPerVectors The size of vectorized memory access.
///                                         Used when loading data from D tensors and storing data
///                                         to output tensor.
/// @tparam BlkGemmPipeSched    The version of blockwise-gemm pipeline scheduler (interwave or
///                             intrawave).
/// @tparam BlkGemmPipelineVer  The version of blockwise-gemm pipeline.
/// @tparam ComputeTypeA    Data type used for A input of hardware matrix-multiplication
///                         instructions.
/// @tparam ComputeTypeB    Data type used for B input of hardware matrix-multiplication
///                         instructions.
/// @tparam PermuteA            Whether the A input tensor has gridwise-gemm friendly data layout
///                             in global memory. Currently not supported!
/// @tparam PermuteB            Whether the B input tensor has gridwise-gemm friendly data layout
///                             in global memory (pre-shuffled).
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
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
          bool IsBPreShuffled          = false,
          bool ForceThreadTileTransfer = false>
struct GridwiseGemm_wmma_cshuffle_v3
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
          ForceThreadTileTransfer>
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
        ForceThreadTileTransfer>;

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
        __host__ Problem() = default;
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         std::array<index_t, NumATensor> StrideAs_,
                         std::array<index_t, NumBTensor> StrideBs_,
                         std::array<index_t, NumDTensor> StrideDs_,
                         index_t StrideE_,
                         index_t KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideAs{StrideAs_},
              StrideBs{StrideBs_},
              StrideDs{StrideDs_},
              StrideE{StrideE_},
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
            std::cout << "SE:" << StrideE << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded
                      << ", " << "KRead:" << KRead << ", " << "KP:" << KPadded << ", "
                      << "AK0:" << AK0 << ", " << "BK0:" << BK0 << ", " << "MBlock: " << MBlock
                      << ", " << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        std::array<index_t, NumATensor> StrideAs;
        std::array<index_t, NumBTensor> StrideBs;
        std::array<index_t, NumDTensor> StrideDs;
        index_t StrideE;
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
        __host__ Argument() = default;
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
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CDEElementwiseOperation cde_element_op_,
                          bool is_reduce_ = false)
            : Problem{M_, N_, K_, StrideAs_, StrideBs_, StrideDs_, StrideE_, k_batch_},
              p_as_grid{},
              p_bs_grid{},
              p_ds_grid{},
              p_e_grid{p_e_grid_},
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

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
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

        AElementwiseOperation a_element_op;
        BElementwiseOperation b_element_op;
        CDEElementwiseOperation cde_element_op;

        // TODO: it can be used with SplitK+reduction but currently only used with SplitK+atomicAdd
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
        index_t c_reduce_offset;
    };

    using BlockwiseGemmPipe = typename Base::BlockwiseGemmPipe;

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    // using Block2CTileMap = BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>;

    __device__ static index_t GetKBlockPerScale() { return 1; }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename Block2CTileMap,
              typename EpilogueArgument,
              int BlockMapMBlockIndex = 0,
              int BlockMapNBlockIndex = 1>
    __device__ static void Run(AsGridPointer& p_as_grid,
                               BsGridPointer& p_bs_grid,
                               DsGridPointer& p_ds_grid,
                               EDataType* p_e_grid,
                               void* p_shared,
                               const Problem& problem,
                               const Block2CTileMap& block_2_ctile_map,
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

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id =
            __builtin_amdgcn_readfirstlane(block_work_idx[Number<BlockMapMBlockIndex>{}]);
        const index_t block_n_id =
            __builtin_amdgcn_readfirstlane(block_work_idx[Number<BlockMapNBlockIndex>{}]);

        // BScale struct (Empty)
        using Scale         = typename BlockwiseGemmPipe::Empty;
        auto a_scale_struct = Scale{};
        auto b_scale_struct = Scale{};

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

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename EpilogueArgument>
    __device__ static void Run(AsGridPointer& p_as_grid,
                               BsGridPointer& p_bs_grid,
                               DsGridPointer& p_ds_grid,
                               EDataType* p_e_grid,
                               void* p_shared,
                               const Problem& problem,
                               AElementwiseOperation a_element_op,
                               BElementwiseOperation b_element_op,
                               CDEElementwiseOperation cde_element_op,
                               EpilogueArgument& epilogue_args)
    {
        Run<HasMainKBlockLoop,
            EGlobalMemoryDataOperation,
            TailNum,
            Block2CTileMap,
            EpilogueArgument>(p_as_grid,
                              p_bs_grid,
                              p_ds_grid,
                              p_e_grid,
                              p_shared,
                              problem,
                              DefaultBlock2CTileMap(problem),
                              a_element_op,
                              b_element_op,
                              cde_element_op,
                              epilogue_args);
    }

    // Wrapper function to have __global__ function in common
    // between gemm_universal, b_scale, ab_scale, etc.
    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename Block2CTileMap,
              typename EpilogueArgument,
              int BlockMapMBlockIndex = 0,
              int BlockMapNBlockIndex = 1>
    __device__ static void Run(void* p_shared,
                               const SplitKBatchOffset& splitk_batch_offset,
                               Argument& karg,
                               const Block2CTileMap& block_2_ctile_map,
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

        Run<HasMainKBlockLoop,
            EGlobalMemoryDataOperation,
            TailNum,
            Block2CTileMap,
            EpilogueArgument,
            BlockMapMBlockIndex,
            BlockMapNBlockIndex>(p_as_grid_splitk,
                                 p_bs_grid_splitk,
                                 karg.p_ds_grid,
                                 karg.p_e_grid + splitk_batch_offset.c_reduce_offset,
                                 p_shared,
                                 karg,
                                 block_2_ctile_map,
                                 karg.a_element_op,
                                 karg.b_element_op,
                                 karg.cde_element_op,
                                 epilogue_args,
                                 A_k_id,
                                 B_k_id);
    }

    // Wrapper function to have __global__ function in common
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
        Run<HasMainKBlockLoop,
            EGlobalMemoryDataOperation,
            TailNum,
            Block2CTileMap,
            EpilogueArgument>(p_shared,
                              splitk_batch_offset,
                              karg,
                              DefaultBlock2CTileMap(karg),
                              epilogue_args,
                              A_k_id,
                              B_k_id);
    }

    __device__ static auto DefaultBlock2CTileMap(const Problem& problem)
    {
        return Block2CTileMap{problem.M, problem.N, 4};
    }

    // Run method for convolution for bwd_data (grid descriptors are passed as arguments,
    // not generated internally)
    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename Block2CTileMapExt,
              typename ComputePtrOffsetOfBatch,
              typename ComputePtrOffsetOfN,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              bool CTranspose,
              TailNumber TailNum,
              typename EpilogueArgument>
    __device__ static void Run(void* p_shared,
                               const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
                               const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2CTileMapExt& block_2_ctile_map,
                               const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                               const ComputePtrOffsetOfN compute_ptr_offset_of_n,
                               const index_t num_k_per_block,
                               Argument& karg,
                               EpilogueArgument& epilogue_args)
    {
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z / karg.KBatch);
        const index_t k_idx =
            __builtin_amdgcn_readfirstlane((blockIdx.z - n_idx * karg.KBatch) * num_k_per_block);

        // offset base pointer for each work-group
        const long_index_t a_batch_offset =
            CTranspose ? amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx))
                       : amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            CTranspose ? amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx))
                       : amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t e_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

        const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

        const long_index_t a_n_offset =
            CTranspose ? 0 : amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
        const long_index_t b_n_offset =
            CTranspose ? amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx)) : 0;
        const long_index_t e_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

        AsGridPointer p_as_grid_;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            p_as_grid_(i) =
                static_cast<const ADataType_*>(karg.p_as_grid[i]) + a_batch_offset + a_n_offset;
        });

        BsGridPointer p_bs_grid_;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            p_bs_grid_(i) =
                static_cast<const BDataType_*>(karg.p_bs_grid[i]) + b_batch_offset + b_n_offset;
        });

        DsGridPointer p_ds_grid_grp;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_batch_offset[i]; });

        // Currently supporting one A and one B
        const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
            [&](auto i) {
                ignore = i;
                return a_grid_desc_ak0_m_ak1;
            },
            Number<NumATensor>{});

        const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
            [&](auto i) {
                ignore = i;
                return b_grid_desc_bk0_n_bk1;
            },
            Number<NumBTensor>{});

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

        // AScale struct (Empty)
        using AScale        = typename BlockwiseGemmPipe::Empty;
        auto a_scale_struct = AScale{};

        // BScale struct (Empty)
        using BScale        = typename BlockwiseGemmPipe::Empty;
        auto b_scale_struct = BScale{};

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
                           TailNum>(p_as_grid_,
                                    p_bs_grid_,
                                    p_ds_grid_grp,
                                    karg.p_e_grid + e_batch_offset + e_n_offset,
                                    p_shared,
                                    as_grid_desc_ak0_m_ak1,
                                    bs_grid_desc_bk0_n_bk1,
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                    e_grid_desc_mblock_mperblock_nblock_nperblock,
                                    karg.a_element_op,
                                    karg.b_element_op,
                                    karg.cde_element_op,
                                    block_m_id,
                                    block_n_id,
                                    num_k_block_per_scale,
                                    a_scale_struct,
                                    b_scale_struct,
                                    epilogue_args,
                                    k_idx,
                                    k_idx,
                                    karg.KBatch);
    }

    // Run method for convolution (grid descriptors are passed as arguments,
    // not generated internally)
    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename ComputePtrOffsetOfBatch,
              index_t NumGroupsToMerge,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename EpilogueArgument>
    __device__ static void Run(void* p_shared,
                               const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
                               const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                               const index_t num_k_per_block,
                               Argument& karg,
                               EpilogueArgument& epilogue_args)
    {
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z * NumGroupsToMerge);
        const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y * num_k_per_block);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t e_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

        AsGridPointer p_as_grid_;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            p_as_grid_(i)    = static_cast<const ADataType_*>(karg.p_as_grid[i]) + a_batch_offset;
        });

        BsGridPointer p_bs_grid_;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            p_bs_grid_(i)    = static_cast<const BDataType_*>(karg.p_bs_grid[i]) + b_batch_offset;
        });

        const auto ds_grid_desc_m_n =
            MakeDsGridDescriptor_M_N(karg.M, karg.MPadded, karg.N, karg.NPadded, karg.StrideDs);

        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_grid_desc_m_n, karg.MBlock, karg.NBlock);

        const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
            [&](auto i) {
                ignore = i;
                return a_grid_desc_ak0_m_ak1;
            },
            Number<NumATensor>{});

        const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
            [&](auto i) {
                ignore = i;
                return b_grid_desc_bk0_n_bk1;
            },
            Number<NumBTensor>{});

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{karg.M, karg.N, 4};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // Scale structs (Empty)
        using Scale         = typename BlockwiseGemmPipe::Empty;
        auto b_scale_struct = Scale{};
        auto a_scale_struct = Scale{};

        const index_t num_k_block_per_scale = GetKBlockPerScale();

        Base::template Run<decltype(as_grid_desc_ak0_m_ak1),
                           decltype(bs_grid_desc_bk0_n_bk1),
                           decltype(ds_grid_desc_mblock_mperblock_nblock_nperblock),
                           decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                           decltype(a_scale_struct),
                           decltype(b_scale_struct),
                           decltype(epilogue_args),
                           HasMainKBlockLoop,
                           CGlobalMemoryDataOperation,
                           TailNum>(p_as_grid_,
                                    p_bs_grid_,
                                    karg.p_ds_grid,
                                    karg.p_e_grid + e_batch_offset,
                                    p_shared,
                                    as_grid_desc_ak0_m_ak1,
                                    bs_grid_desc_bk0_n_bk1,
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    karg.a_element_op,
                                    karg.b_element_op,
                                    karg.cde_element_op,
                                    block_m_id,
                                    block_n_id,
                                    num_k_block_per_scale,
                                    a_scale_struct,
                                    b_scale_struct,
                                    epilogue_args,
                                    k_idx,
                                    k_idx,
                                    karg.KBatch);
    }

    // Run method for convolution fwd (grid descriptors are passed as arguments,
    // not generated internally)
    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename ComputePtrOffsetOfBatch,
              typename ComputePtrOffsetOfN,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename EpilogueArgument>
    __device__ static void Run(void* p_shared,
                               const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                               const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const ComputePtrOffsetOfBatch& compute_ptr_offset_of_batch,
                               const ComputePtrOffsetOfN& compute_ptr_offset_of_n,
                               [[maybe_unused]] const index_t num_k_per_block,
                               Argument& karg,
                               EpilogueArgument& epilogue_args)
    {
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z / karg.KBatch);
        // offset base pointer for each work-group
        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t e_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

        const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

        const long_index_t a_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
        const long_index_t b_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetBPtrOffset(n_idx));
        const long_index_t e_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

        const auto ds_n_offset = compute_ptr_offset_of_n.GetDsPtrOffset(n_idx);

        AsGridPointer p_as_grid_;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            p_as_grid_(i) =
                static_cast<const ADataType_*>(karg.p_as_grid[i]) + a_batch_offset + a_n_offset;
        });

        BsGridPointer p_bs_grid_;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            p_bs_grid_(i) =
                static_cast<const BDataType_*>(karg.p_bs_grid[i]) + b_batch_offset + b_n_offset;
        });

        DsGridPointer p_ds_grid_grp;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType_ = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            p_ds_grid_grp(i) = static_cast<const DDataType_*>(karg.p_ds_grid[i]) +
                               ds_batch_offset[i] + ds_n_offset[i];
        });

        // Currently supporting one A and one B
        const auto as_grid_desc_ak0_m_ak1 = generate_tuple(
            [&](auto i) {
                ignore = i;
                return a_grid_desc_ak0_m_ak1;
            },
            Number<NumATensor>{});

        const auto bs_grid_desc_bk0_n_bk1 = generate_tuple(
            [&](auto i) {
                ignore = i;
                return b_grid_desc_bk0_n_bk1;
            },
            Number<NumBTensor>{});

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{karg.M, karg.N, 4};

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

        // AScale struct (Empty)
        using AScale        = typename BlockwiseGemmPipe::Empty;
        auto a_scale_struct = AScale{};

        // BScale struct (Empty)
        using BScale        = typename BlockwiseGemmPipe::Empty;
        auto b_scale_struct = BScale{};

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
                           TailNum>(p_as_grid_,
                                    p_bs_grid_,
                                    p_ds_grid_grp,
                                    karg.p_e_grid + e_batch_offset + e_n_offset,
                                    p_shared,
                                    as_grid_desc_ak0_m_ak1,
                                    bs_grid_desc_bk0_n_bk1,
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                    e_grid_desc_mblock_mperblock_nblock_nperblock,
                                    karg.a_element_op,
                                    karg.b_element_op,
                                    karg.cde_element_op,
                                    block_m_id,
                                    block_n_id,
                                    num_k_block_per_scale,
                                    a_scale_struct,
                                    b_scale_struct,
                                    epilogue_args);
    }
};

} // namespace ck
