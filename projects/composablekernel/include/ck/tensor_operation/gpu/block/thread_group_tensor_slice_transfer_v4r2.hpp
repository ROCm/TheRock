// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v3r2.hpp"

namespace ck {

/**
 * @brief Blockwise data transfer
 *
 * This version does following things to avoid scratch memory issue
 * 1. Use StaticallyIndexedArray instead of C array for thread buffer
 * 2. ThreadwiseTensorSliceTransfer_v3 does not keep reference to tensor descriptor
 * 3. ThreadwiseTensorSliceTransfer_v3::Run() does not construct new tensor coordinate
 *
 */
template <typename ThreadGroup,
          typename ElementwiseOperation,
          typename DstInMemOps, // Sequence
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDatas,
          typename DstDatas,
          typename SrcDescs,
          typename DstDescs,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          typename SrcsScalarPerVector,                       // Sequence
          typename DstsScalarPerVector,                       // Sequence
          typename SrcsScalarStrideInVector,                  // Sequence
          typename DstsScalarStrideInVector,                  // Sequence
          typename ThreadTransferSrcsResetCoordinateAfterRun, // Sequence
          typename ThreadTransferDstsResetCoordinateAfterRun, // Sequence
          index_t NumThreadScratch = 1>
struct ThreadGroupTensorSliceTransfer_v4r2
{
    static constexpr index_t nDim =
        remove_reference_t<tuple_element_t<0, SrcDescs>>::GetNumOfDimension();
    static constexpr index_t nSrc = SrcDescs::Size();
    static constexpr index_t nDst = DstDescs::Size();

    static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    __device__ constexpr ThreadGroupTensorSliceTransfer_v4r2(
        const SrcDescs& src_descs,
        const StaticallyIndexedArray<Index, nSrc>& src_block_slice_origins,
        const DstDescs& dst_descs,
        const StaticallyIndexedArray<Index, nDst>& dst_block_slice_origins,
        const ElementwiseOperation& element_op)
        : threadwise_transfer_(src_descs,
                               StaticallyIndexedArray<Index, nSrc>{},
                               dst_descs,
                               StaticallyIndexedArray<Index, nDst>{},
                               element_op)

    {
        static_assert(nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == SrcDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_for<0, nSrc, 1>{}([&](auto src_i) {
            static_assert(nDim ==
                              remove_cvref_t<tuple_element_t<src_i, SrcDescs>>::GetNumOfDimension(),
                          "wrong! nDim not consistent");
        });

        static_for<0, nDst, 1>{}([&](auto dst_i) {
            static_assert(nDim ==
                              remove_cvref_t<tuple_element_t<dst_i, DstDescs>>::GetNumOfDimension(),
                          "wrong! nDim not consistent");
        });

        static_assert(
            is_same<BlockSliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "wrong! ThreadGroup::GetNumOfThread() too small");

        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(ThreadGroup::GetThreadId()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            const auto src_thread_slice_origins = generate_tuple(
                [&](auto i) { return src_block_slice_origins[i] + thread_data_idx_begin; },
                Number<nSrc>{});

            const auto dst_thread_slice_origins = generate_tuple(
                [&](auto i) { return dst_block_slice_origins[i] + thread_data_idx_begin; },
                Number<nDst>{});

            threadwise_transfer_.SetSrcSliceOrigins(src_descs, src_thread_slice_origins);
            threadwise_transfer_.SetDstSliceOrigins(dst_descs, dst_thread_slice_origins);
        }
    }

    template <typename SrcBuffers, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDescs& src_descs,
                            const SrcBuffers& src_bufs,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunRead(src_descs, src_bufs, thread_scratch_id);
        }
    }

    template <typename DstBuffers, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDescs& dst_descs,
                             DstBuffers& dst_bufs,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunWrite(dst_descs, dst_bufs, thread_scratch_id);
        }
    }

    template <typename SrcBuffer, typename DstBuffer, index_t ThreadScratchId>
    __device__ void Run(const SrcDescs& src_descs,
                        const SrcBuffer& src_bufs,
                        const DstDescs& dst_descs,
                        DstBuffer& dst_bufs,
                        Number<ThreadScratchId> thread_scratch_id)
    {
        RunRead(src_descs, src_bufs, thread_scratch_id);
        RunWrite(dst_descs, dst_bufs, thread_scratch_id);
    }

    __device__ void MoveSrcSliceWindow(const SrcDescs& src_descs, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(src_descs, step);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDescs& dst_descs, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_descs, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v3r2<decltype(thread_slice_lengths),
                                           ElementwiseOperation,
                                           DstInMemOps,
                                           SrcDatas,
                                           DstDatas,
                                           SrcDescs,
                                           DstDescs,
                                           SrcDimAccessOrder,
                                           DstDimAccessOrder,
                                           SrcVectorDim,
                                           DstVectorDim,
                                           SrcsScalarPerVector,
                                           DstsScalarPerVector,
                                           SrcsScalarStrideInVector,
                                           DstsScalarStrideInVector,
                                           ThreadTransferSrcsResetCoordinateAfterRun,
                                           ThreadTransferDstsResetCoordinateAfterRun,
                                           NumThreadScratch>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
