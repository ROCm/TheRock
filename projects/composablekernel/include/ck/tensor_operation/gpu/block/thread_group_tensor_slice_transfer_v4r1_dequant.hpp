// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v3r1_dequant.hpp"

namespace ck {

/**
 * @brief Blockwise data transfer with dequantization
 *
 * RunRead  would load low-precision data and scale data.
 * RunWrite would process dequantization process.
 * Assume Scale is identical along K-dimension
 *
 * This version does following things to avoid scratch memory issue
 * 1. Use StaticallyIndexedArray instead of C array for thread buffer
 * 2. ThreadwiseTensorSliceTransfer_v3 does not keep reference to tensor descriptor
 * 3. ThreadwiseTensorSliceTransfer_v3::Run() does not construct new tensor coordinate
 *
 */
template <typename ThreadGroup,
          typename SrcElementwiseOperation,
          typename ScaleElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename BlockSliceLengths,
          typename BlockScaleSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename ScaleData,
          typename DstData,
          typename SrcDesc,
          typename ScaleDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t ScaleScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t ScaleScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool ThreadTransferSrcResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun,
          index_t NumThreadScratch = 1>
struct ThreadGroupTensorSliceTransfer_v4r1_dequant
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();

    static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};
    static constexpr auto scale_thread_slice_lengths =
        BlockScaleSliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    __device__ constexpr ThreadGroupTensorSliceTransfer_v4r1_dequant(
        const SrcDesc& src_desc,
        const Index& src_block_slice_origin,
        const SrcElementwiseOperation& src_element_op,
        const ScaleDesc& scale_desc,
        const Index& scale_block_slice_origin,
        const ScaleElementwiseOperation& scale_element_op,
        const DstDesc& dst_desc,
        const Index& dst_block_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : threadwise_transfer_(src_desc,
                               make_zero_multi_index<nDim>(),
                               src_element_op,
                               scale_desc,
                               make_zero_multi_index<nDim>(),
                               scale_element_op,
                               dst_desc,
                               make_zero_multi_index<nDim>(),
                               dst_element_op)

    {
        static_assert(nDim == remove_cvref_t<SrcDesc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<ScaleDesc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<DstDesc>::GetNumOfDimension() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{} &&
                is_same<BlockScaleSliceLengths,
                        decltype(scale_thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "wrong! ThreadGroup::GetNumOfThread() too small");

        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(ThreadGroup::GetThreadId()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            threadwise_transfer_.SetSrcSliceOrigin(src_desc,
                                                   src_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetScaleSliceOrigin(
                scale_desc, scale_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    template <typename SrcBuffer, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcBuffer& src_buf,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunRead(src_desc, src_buf, thread_scratch_id);
        }
    }

    // With the assumption, scale scratch is always one
    template <typename ScaleBuffer>
    __device__ void RunScaleRead(const ScaleDesc& scale_desc, const ScaleBuffer& scale_buf)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunScaleRead(scale_desc, scale_buf);
        }
    }

    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunWrite(dst_desc, dst_buf, thread_scratch_id);
        }
    }

    // We don't prefer use this API directly
    /*
    template <typename SrcBuffer, typename DstBuffer, index_t ThreadScratchId>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf,
                        Number<ThreadScratchId> thread_scratch_id)
    {
        RunRead(src_desc, src_buf, thread_scratch_id);
        RunWrite(dst_desc, dst_buf, thread_scratch_id);
    }
    */

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(src_desc, step);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    // With the assumption, scale buffer don't need move slice window method

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v3r1_dequant<decltype(thread_slice_lengths),
                                                   decltype(scale_thread_slice_lengths),
                                                   SrcElementwiseOperation,
                                                   ScaleElementwiseOperation,
                                                   DstElementwiseOperation,
                                                   DstInMemOp,
                                                   SrcData,
                                                   ScaleData,
                                                   DstData,
                                                   SrcDesc,
                                                   ScaleDesc,
                                                   DstDesc,
                                                   SrcDimAccessOrder,
                                                   DstDimAccessOrder,
                                                   SrcVectorDim,
                                                   DstVectorDim,
                                                   SrcScalarPerVector,
                                                   ScaleScalarPerVector,
                                                   DstScalarPerVector,
                                                   SrcScalarStrideInVector,
                                                   ScaleScalarStrideInVector,
                                                   DstScalarStrideInVector,
                                                   ThreadTransferSrcResetCoordinateAfterRun,
                                                   ThreadTransferDstResetCoordinateAfterRun,
                                                   NumThreadScratch>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
