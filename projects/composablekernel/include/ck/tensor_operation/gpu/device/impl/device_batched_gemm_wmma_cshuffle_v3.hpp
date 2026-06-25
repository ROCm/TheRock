// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename ComputePtrOffsetOfStridedBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_batched_gemm_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg, // This works for now but it actually receives a
                                              // DeviceBatchedGemm_Wmma_CShuffleV3::Argument
                                              // argument through implicit conversion to base class!
        const ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using c_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<c_data_type, ck::half_t> ||
                    std::is_same_v<c_data_type, ck::bhalf_t>)))
    {
#endif
        // The normal approach to batching would be to increase the grid size by just stretching out
        // the grid Z dimension (which is the outermost dimension), but this depends on lower level
        // functions not directly using the Z dimension for other calculations. As it turns out, k
        // batching does rely directly on blockIdx.Z through SplitKBatchOffset. Therefore, for now
        // we will use the grid Y dimension for batching. This may be a bit fragile.
        const index_t g_idx = amd_wave_read_first_lane(blockIdx.y);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t c_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

        constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueCShuffle>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        // shift A matrices pointer for splitk
        typename GridwiseGemm::AsGridPointer p_as_grid_shift;
        static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
            using ADataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::AsDataType_>>;
            p_as_grid_shift(i) = static_cast<const ADataType_*>(karg.p_as_grid[i]) +
                                 splitk_batch_offset.a_k_split_offset[i] + a_batch_offset;
        });

        // shift B matrices pointer for splitk
        typename GridwiseGemm::BsGridPointer p_bs_grid_shift;
        static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
            using BDataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::BsDataType_>>;
            p_bs_grid_shift(i) = static_cast<const BDataType_*>(karg.p_bs_grid[i]) +
                                 splitk_batch_offset.b_k_split_offset[i] + b_batch_offset;
        });

        auto epilogue_args = typename GridwiseGemm::EpilogueCShuffle{};

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            p_as_grid_shift,
            p_bs_grid_shift,
            karg.p_ds_grid,
            karg.p_e_grid + splitk_batch_offset.c_reduce_offset + c_batch_offset,
            p_shared,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.cde_element_op,
            epilogue_args);
#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = compute_ptr_offset_of_batch;
#endif
}

/// @brief \"Universal\" Batched GEMM operation without SplitK support.
///
/// @par Overview
///         This GEMM operation implements the following mathematical equation:
///         C{G,M,N} = C_op(A_op(A{G,M,K}) * B_op(B{G,K,N}))
///         Where A, B are input tensors and C is the output tensor. The A/B/C_op are
///         elementwise operations applied to the A, B, and C tensors, respectively.
///         The \"universal\" gemm comes with multiple pipelines optimized for different usage
///         scenarios. That's why it's called \"universal\". It's universal through its design
///         and versatilty.
///
/// @note   This Kernel implementation currently does not support the SplitK algorithm.
///
/// @tparam ALayout     A tensor data layout.
/// @tparam BLayout     B tensor data layout.
/// @tparam CLayout     C tensor data layout.
/// @tparam ADataType   A tensor data type.
/// @tparam BDataType   B tensor data type.
/// @tparam CDataType   C tensor data type.
/// @tparam AccDataType The accumulation data type related to the hardware
///                         matrix-multiplication instruction.
/// @tparam CShuffleDataType The data type used to store matrix-multiplication results into
///                          LDS memory during \"CShuffle\" data layout optimization.
/// @tparam AElementwiseOperation Elementwise operation applied to the A input tensor elements.
/// @tparam BElementwiseOperation Elementwise operation applied to the B input tensor elements.
/// @tparam CElementwiseOperation Elementwise operation applied to the C output tensor
///                               (after GEMM).
/// @tparam GemmSpec    Determines used "padding" version.
/// @tparam BlockSize   The number of threads within workgroup.
/// @tparam MPerBlock   The input/output data tile size in the M dimension.
/// @tparam NPerBlock   The input/output data tile size in the N dimension.
/// @tparam KPerBlock   The input data tile size in the K dimension.
/// @tparam AK1         The vector load size from global memory for A tensor.
/// @tparam BK1         The vector load size from global memory for B tensor.
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
/// @tparam BBlockLdsExtraN                      Whether to use padding for LDS or not. With
///                                              universal GEMM there's no need for padding.
/// @tparam CShuffleMRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in M dimension.
/// @tparam CShuffleNRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in N dimension.
/// @tparam CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock The spatial
///                                         thread distribution used for storing data into output
///                                         tensor across output data layout dimensions.
/// @tparam CShuffleBlockTransferScalarPerVector_NPerBlock The size of vectorized memory access.
///                                         Used when storing data to output tensor.
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
///                             in global memory (pre-shuffled). Currently not supported!
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceBatchedGemm_Wmma_CShuffleV3 : public DeviceBatchedGemm<ALayout,
                                                                    BLayout,
                                                                    CLayout,
                                                                    ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    AElementwiseOperation,
                                                                    BElementwiseOperation,
                                                                    CElementwiseOperation>
{
    // We are inheriting from DeviceBatchedGemm and this base class does not support permuteA and
    // permuteB arguments so for now we are not including this functionality.
    static_assert(PermuteA == false,
                  "Permute A functionality not supported by DeviceBatchedGemm operations.\n");
    static_assert(PermuteB == false,
                  "Permute B functionality not supported by DeviceBatchedGemm operations.\n");

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA), BatchStrideB_(BatchStrideB), BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        Tuple<>, // DsLayout
        CLayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        Tuple<>, // DsDataType
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
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
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,  // PermuteA not supported by DeviceBatchedGemm base class.
        false>; // PermuteB not supported by DeviceBatchedGemm base class.

    // Argument
    struct Argument : public GridwiseGemm::Argument
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
                          index_t BatchStrideA_,
                          index_t BatchStrideB_,
                          index_t BatchStrideC_,
                          index_t Batch_,
                          index_t k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation cde_element_op_,
                          bool is_reduce_ = false)
            : GridwiseGemm::Argument(std::array<const void*, 1>{p_a_grid_},
                                     std::array<const void*, 1>{p_b_grid_},
                                     std::array<const void*, 0>{}, // p_ds_grid_
                                     p_c_grid_,
                                     M_,
                                     N_,
                                     K_,
                                     std::array<index_t, 1>{StrideA_},
                                     std::array<index_t, 1>{StrideB_},
                                     std::array<index_t, 0>{}, // StrideDs_
                                     StrideC_,
                                     k_batch_,
                                     a_element_op_,
                                     b_element_op_,
                                     cde_element_op_,
                                     is_reduce_),
              Batch(Batch_),
              compute_ptr_offset_of_batch{BatchStrideA_, BatchStrideB_, BatchStrideC_}
        {
        }

        index_t Batch;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch;
    };

    /// @brief  Helper structure responsible for kernel invocation.
    ///
    /// @paragraph  The `Invoker` class is responsible for preparation and invocation of actual GPU
    ///             kernel function. It usually determines the launched grid size prepares kernel
    ///             arguments as well as perform specific kernel configuration selection based on
    ///             runtime arguments.
    ///
    /// @note       If appropriately configured it may measure kernel execution time.
    ///
    struct Invoker : public BaseInvoker
    {
        /// @brief  This function issues GPU kernel execution.
        /// @param arg           The GPU kernel arguments.
        /// @param stream_config The HIP stream configuration helper structure.
        /// @return              The kernel's average execution time (if time measurement is
        ///                      enabled).
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
                GridwiseGemm::BlockwiseGemmPipe::HotLoopInstList::Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            // The normal approach to batching would be to increase the grid size by just stretching
            // out the grid Z dimension (which is the outermost dimension), but this depends on
            // lower level functions not directly using the Z dimension for other calculations. As
            // it turns out, k batching does rely directly on blockIdx.Z through SplitKBatchOffset.
            // Therefore, for now we will use the grid Y dimension for batching. This may be a bit
            // fragile.
            gdy *= arg.Batch;

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    Argument arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAsGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideAs, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBsGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideBs, arg_.BK0);

                    // Packed sizes are 1 for all implemented data types but we include it anyway
                    // for future compatibility.
                    // Note: the grid descriptors and size_a / size_b do *not* take batching into
                    // account, so we have to manually multiply overall buffer sizes for rotating
                    // memory by batch.
                    std::array<std::size_t, 1> size_as_buffers;
                    size_as_buffers[0] = a_grid_desc_ak0_m_ak1[Number<0>{}].GetElementSpaceSize() *
                                         sizeof(ADataType) / GridwiseGemm::APackedSize * arg_.Batch;

                    std::array<std::size_t, 1> size_bs_buffers;
                    size_bs_buffers[0] = b_grid_desc_bk0_n_bk1[Number<0>{}].GetElementSpaceSize() *
                                         sizeof(BDataType) / GridwiseGemm::BPackedSize * arg_.Batch;

                    ck::utility::RotatingMemWrapperMultiABD<Argument,
                                                            Tuple<ADataType>,
                                                            Tuple<BDataType>,
                                                            Tuple<>>
                        rotating_mem(arg_,
                                     stream_config.rotating_count,
                                     size_as_buffers,
                                     size_bs_buffers,
                                     std::array<std::size_t, 0>{});
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            // Note: we multiply by batch since we want to clear the C matrix for
                            // the whole batch. Untested since we don't have k batching ATM.
                            // Note: This seems incorrect for non-contiguous memory layouts for C
                            // (padding, gaps).
                            HIP_CHECK_ERROR(
                                hipMemsetAsync(arg_.p_e_grid,
                                               0,
                                               arg_.Batch * arg_.M * arg_.N * sizeof(CDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_,
                        arg_.compute_ptr_offset_of_batch);
                }
                else
                {
                    auto clear_workspace = [&]() {
                        // clear c mem
                        if(arg.KBatch > 1)
                            // Note: we multiply by batch since we want to clear the C matrix for
                            // the whole batch. Untested since we don't have k batching ATM.
                            // Note: This seems incorrect for non-contiguous memory layouts for C
                            // (padding, gaps).
                            HIP_CHECK_ERROR(
                                hipMemsetAsync(arg.p_e_grid,
                                               0,
                                               arg.Batch * arg.M * arg.N * sizeof(CDataType),
                                               stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        clear_workspace,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg,
                        arg.compute_ptr_offset_of_batch);
                }
            };

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<ComputePtrOffsetOfStridedBatch>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
                else
                {
                    // TODO: Implement
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.KBatch > 1)
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            ComputePtrOffsetOfStridedBatch,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_batched_gemm_wmma_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<ComputePtrOffsetOfStridedBatch>,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }

        if constexpr(std::is_same_v<CDataType, ck::half_t> ||
                     std::is_same_v<CDataType, ck::bhalf_t>)
        {
            if(arg.KBatch > 1 && ck::is_gfx11_supported())
            {
                // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
                return false;
            }
        }

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    // TODO: This is not part of the DeviceBatchedGemm base class but it was part of
    // DeviceBatchedGemmV2. Remove?
    // index_t GetKPerBlock() override { return KPerBlock; }
    // bool GetPermuteA() override { return PermuteA; }
    // bool GetPermuteB() override { return PermuteB; }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             index_t BatchStrideC,
                             index_t Batch,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideC,
                        Batch,
                        1, /* KBatch */
                        AElementwiseOperation{},
                        BElementwiseOperation{},
                        CElementwiseOperation{}};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      index_t BatchStrideA,
                                                      index_t BatchStrideB,
                                                      index_t BatchStrideC,
                                                      index_t Batch,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideC,
                                          Batch,
                                          1,
                                          AElementwiseOperation{},
                                          BElementwiseOperation{},
                                          CElementwiseOperation{}); // KBatch
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceBatchedGemm_Wmma_CShuffleV3"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock << "x" << NPerBlock << "x" << KPerBlock << ", "
            << "WaveTile: "
            << MPerWmma << "x"<<NPerWmma << ", "
            << "WaveMap: "
            << MRepeat << "x" << NRepeat << ", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector << "x" << BBlockTransferSrcScalarPerVector << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages << ", "
            << "KPack: "
            << GridwiseGemm::KPack;
        // clang-format on

        return str.str();
    }
    REGISTER_EXTRA_PRINTING_METHODS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
