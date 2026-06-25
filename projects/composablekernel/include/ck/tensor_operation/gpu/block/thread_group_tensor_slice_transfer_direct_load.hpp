// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

/**
 * Transfer that uses direct load instructions to copy data from global to LDS memory.
 *
 * Traditional loads first copy data from global to registers, and then from registers to LDS.
 * Direct loads do not need an intermediate step, data is copied directly from global to LDS,
 * without the use of additional registers.
 *
 * However, the instruction has limitations:
 * - each thread must copy exactly a single DWORD - 4 bytes;
 * - threads within a single wavefront must write consecutive DWORDS into LDS,
 *   (data in global do not need to be contiguous, each thread might have its own offset).
 *
 * To make sure that all the transfers finished, the `waitcnt` instruction must be used with
 * `vmcnt` instead of `lgkmcnt`.
 *
 * Limitations of the transfer class:
 * - `SrcData` must be the same as `DstData` - no possibility to convert the data type in flight;
 * - `DstVectorDim` must be the last dimension;
 * - `SrcVectorDim` must be the last dimension if `ScalarPerVector` is greater than 1;
 * - `ScalarPerVector` times the number of bytes of `DstData` must be equal to a single DWORD = 4B
 *   (for examlpe if `DstData` is fp32, then `ScalarPerVector` must be 1; if `DstData` is fp16,
 *   `ScalarPerVector` must be 2);
 * - if `ScalarPerVector` is greater than 1, the contiguous dimension in src and dst must be
 *   the same dimension;
 * - threads in a wavefront must write contiguous data to LDS (when wavefront size is 64,
 *   they must write 64 contiguous DWORDs) - `ThreadClusterLengths` must be prepared in such a way
 *   to guarantee that.
 */
template <typename ThreadGroup,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t ScalarPerVector>
struct ThreadGroupTensorSliceTransfer_DirectLoad
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr auto block_slice_lengths    = BlockSliceLengths{};
    static constexpr auto thread_cluster_lengths = ThreadClusterLengths{};

    static constexpr auto thread_single_load_size = generate_sequence(
        detail::lambda_scalar_per_access<DstVectorDim, ScalarPerVector>{}, Number<nDim>{});
    // After a load, each thread moves by `thread_steps` instead of loading the next elements.
    // It makes the whole wavefront load contiguous memory, what is required for direct loads.
    static constexpr auto thread_steps         = thread_cluster_lengths * thread_single_load_size;
    static constexpr auto thread_slice_lengths = block_slice_lengths / thread_steps;

    static __device__ constexpr bool AreThreadClusterLengthsValid()
    {
        // Make sure that ThreadClusterLengths are set in a way that allows for contiguous writes to
        // LDS by the threads from a single wavefront.
        // Examples (assuming 64 threads in a wavefront, 128 in a thread block):
        // 1. BlockSliceLengths = [K0PerBlock, MPerBlock, K1PerBlock] = [4, 128, 8],
        //    data type = fp32 -> ScalarPerVector = 1
        //    INVALID: ThreadClusterLengths = [4, 4, 8] since in the first iteration, threads 0-31
        //             write [0, 0, 0] - [0, 3, 7] and thread 32 writes [1, 0, 0] instead of
        //             [0, 4, 0].
        //    VALID: ThreadClusterLengths = [2, 8, 8] or [1, 16, 8] since in the first iteration,
        //           threads 0-63 write [0, 0, 0] - [0, 7, 7] -> 64 consecutive elements (DWORDs).
        // 2. BlockSliceLengths = [K0PerBlock, MPerBlock, K1PerBlock] = [4, 128, 8],
        //    data type = fp16 -> ScalarPerVector = 2
        //    NOTE: ThreadClusterLengths must take into account that each thread writes two
        //          elements (single DWORD) along the contiguous dimension.
        //    INVALID: ThreadClusterLengths = [4, 4, 8] since each 8 threads would try to write
        //             8 * 2 elements of K1PerBlock and there are only 8;
        //             ThreadClusterLengths = [4, 8, 4] since in the first iteration, threads 0-31
        //             write [0, 0, 0] - [0, 7, 7] (7 since each writes 2 elements) and thread 32
        //             writes [1, 0, 0] instead of [0, 8, 0].
        //    VALID: ThreadClusterLengths = [4, 16, 4] or [2, 32, 4] or [1, 64, 4] since in the
        //           first iteration, threads 0-63 write [0, 0, 0] -  [0, 15, 7] -> 128 consecutive
        //           elements = 64 consecutive DWORDs.
#if defined(__gfx950__)
        int num_contiguous_dwords = 4;
#else
        int num_contiguous_dwords = 1;
#endif
        bool is_contiguous = true;
        static_for<0, nDim, 1>{}([&](auto i) {
            if(is_contiguous)
            {
                num_contiguous_dwords *= thread_cluster_lengths[nDim - i - 1];
            }
            if(thread_slice_lengths[nDim - i - 1] > 1)
            {
                is_contiguous = false;
            }
        });
        constexpr index_t wavefront_size = get_warp_size();
        const bool wave_contiguous       = num_contiguous_dwords % wavefront_size == 0;

        bool thread_slice_lengths_correct = true;
        static_for<0, nDim, 1>{}([&](auto i) {
            if(thread_slice_lengths[i] <= 0)
            {
                thread_slice_lengths_correct = false;
            }
        });

        return wave_contiguous && thread_slice_lengths_correct;
    }

    __device__ constexpr ThreadGroupTensorSliceTransfer_DirectLoad(
        const SrcDesc& src_desc,
        const Index& src_block_slice_origin,
        const DstDesc& dst_desc,
        const Index& dst_block_slice_origin)

    {
        static_assert(ck::is_same_v<SrcData, DstData>,
                      "Direct load transfer does not support datatypes conversion. Source and "
                      "destination data types must be the same.");

        static_assert(
            DstVectorDim == nDim - 1,
            "Direct load transfer requires the destination vector dimension to be the last one.");

        static_assert(ScalarPerVector == 1 || SrcVectorDim == DstVectorDim,
                      "When loading more than one element per thread at once, the contiguous "
                      "dimension must be the same between source and destination.");

        // constexpr auto dword_bytes           = 4;
        // constexpr auto bytes_per_thread_load = ScalarPerVector * sizeof(SrcData);
        // static_assert(bytes_per_thread_load == dword_bytes,
        //               "Direct load transfer requires each thread to load exactly a single "
        //               "DWORD of data.");

        static_assert(nDim == remove_cvref_t<SrcDesc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<DstDesc>::GetNumOfDimension() &&
                          nDim == ThreadClusterLengths::Size(),
                      "Inconsistent number of dimensions across lengths and descriptors.");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "The number of threads cannot be less than the number of elements in "
                      "thread cluster lengths.");

        // static_assert(
        //     AreThreadClusterLengthsValid(),
        //     "Thread cluster lengths are incorrect. They must be set in a way that allows a single
        //     " "wavefront to write contiguous DWORDs into LDS memory. ");

        const auto thread_cluster_idx =
            thread_cluster_desc_.CalculateBottomIndex(make_multi_index(ThreadGroup::GetThreadId()));

        constexpr auto wave_cluster_lengths = generate_sequence_v2(
            [&](auto i) {
                // FIXME: wave parallelism is not always in that dimension.
                // The ThreadClusterLengths{} must be bigger than wave_num;
                if constexpr(ThreadClusterArrangeOrder{}.At(i) == (nDim - 3))
                {
                    return Number<ThreadGroup::GetNumOfThread() / 64>{};
                }
                else
                {
                    return I1;
                }
            },
            Number<nDim>{});

        constexpr auto wave_thread_cluster_lengths = ThreadClusterLengths{} / wave_cluster_lengths;
        constexpr auto wave_single_load_size =
            wave_thread_cluster_lengths * thread_single_load_size;
        constexpr auto wave_cluster_desc_ =
            make_cluster_descriptor(wave_cluster_lengths, ThreadClusterArrangeOrder{});

        const auto wave_cluster_idx = wave_cluster_desc_.CalculateBottomIndex(
            make_multi_index(ThreadGroup::GetThreadId() / 64));

        const auto thread_data_idx_begin = thread_cluster_idx * thread_single_load_size;
        const auto wave_data_idx_begin   = wave_cluster_idx * wave_single_load_size;

        SetSrcSliceOrigin(src_desc, src_block_slice_origin + thread_data_idx_begin);
        // We don't need threadwise offset for lds since it was calculate by HW
        // We still need input the wavewise offset.
        SetDstSliceOrigin(dst_desc, dst_block_slice_origin + wave_data_idx_begin);
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_idx);
        src_slice_origin_ = src_slice_origin_idx;
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_        = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
        dst_slice_origin_ = dst_slice_origin_idx;
    }

    __device__ void ResetDstSliceWindow(const DstDesc& dst_desc)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_);
    }

    template <typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global,
                      "Source data must come from a global memory buffer.");
        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "Destination data must be stored in an LDS memory buffer.");

        static_assert(
            ck::is_same_v<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>,
            "SrcBuffer and SrcData data types must be consistent.");
        static_assert(
            ck::is_same_v<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>,
            "DstBuffer and DstData data types must be consistent.");

        constexpr auto dst_access_lengths = thread_slice_lengths;

        const auto dst_forward_steps  = generate_steps(dst_desc, 1);
        const auto dst_backward_steps = generate_steps(dst_desc, -1);
        const auto src_forward_steps  = generate_steps(src_desc, 1);
        const auto src_backward_steps = generate_steps(src_desc, -1);

        // Loop over the destination block and copy data.
        static_ford<decltype(dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            const auto src_offset = src_coord_.GetOffset();
            const auto dst_offset = __builtin_amdgcn_readfirstlane(dst_coord_.GetOffset());

            // Check if src data is not in the logic padding area.
            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            src_buf.template DirectCopyToLds<remove_cvref_t<decltype(dst_buf)>, ScalarPerVector>(
                dst_buf, src_offset, dst_offset, is_src_valid);

            constexpr auto move_on_dim = [&]() constexpr {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_dst_access_idx[i] < dst_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &= ordered_dst_access_idx[j] == dst_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }();

            // Decide whether to move forward or backward.
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_dst_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * dst_access_lengths[j] + ordered_dst_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(dst_desc, dst_coord_, dst_forward_steps[i]);
                        move_tensor_coordinate(src_desc, src_coord_, src_forward_steps[i]);
                    }
                    else
                    {
                        move_tensor_coordinate(dst_desc, dst_coord_, dst_backward_steps[i]);
                        move_tensor_coordinate(src_desc, src_coord_, src_backward_steps[i]);
                    }
                }
            });
        });

        // Reset the destination slice since the entire buffer has been already filled.
        ResetDstSliceWindow(dst_desc);
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        src_slice_origin_ = src_slice_origin_ + step;
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_);
    }

    template <typename DescType>
    __device__ auto generate_steps(const DescType& desc, int sign)
    {
        return generate_tuple(
            [&](auto i) {
                Index step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    step_idx(j) = (i.value == j.value) ? sign * thread_steps[i] : 0;
                });

                return make_tensor_coordinate_step(desc, step_idx);
            },
            Number<nDim>{});
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    SrcCoord src_coord_;
    DstCoord dst_coord_;
    Index src_slice_origin_;
    Index dst_slice_origin_;
};

} // namespace ck
