// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/wrapper/utils/tensor_utils.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

/**
 * \brief Perform optimized copy between two tensors partitions (threadwise copy).
 * Tensors must have the same size.
 *
 * \tparam DimAccessOrderTuple Tuple with dimension access order.
 * \tparam VectorDim Dimension for vectorized read and write.
 * \tparam ScalarPerVector Number of scalar per vectorized read and write.
 * \param src_tensor Source tensor.
 * \param dst_tensor Destination tensor.
 */
template <typename DimAccessOrderTuple,
          index_t VectorDim,
          index_t ScalarPerVector,
          typename SrcTensorType,
          typename DstTensorType>
__device__ void copy(const SrcTensorType& src_tensor, DstTensorType& dst_tensor)
{
    static_assert(is_detected<is_tuple, DimAccessOrderTuple>::value);
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const auto& in_grid_desc  = layout(src_tensor).GetUnrolledDescriptor();
    const auto& out_grid_desc = layout(dst_tensor).GetUnrolledDescriptor();

    using SrcShapeType         = remove_cvref_t<decltype(shape(src_tensor))>;
    constexpr index_t num_dims = SrcShapeType::Size();

    constexpr auto thread_slice_lengths =
        generate_sequence_v2([](auto I) { return size(SrcShapeType{}.At(I)); }, Number<num_dims>{});
    constexpr auto dim_access_order = generate_sequence_v2(
        [](auto I) { return DimAccessOrderTuple{}.At(I); }, Number<num_dims>{});

    if constexpr(SrcTensorType::IsDynamicBuffer && DstTensorType::IsDynamicBuffer)
    {
        // Perform a copy between DynamicBuffers
        auto transfer = ThreadwiseTensorSliceTransfer_v7<
            Tuple<typename SrcTensorType::TensorElementType>,
            Tuple<typename DstTensorType::TensorElementType>,
            decltype(tie(in_grid_desc)),
            decltype(tie(out_grid_desc)),
            tensor_operation::element_wise::PassThrough,
            Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
            decltype(thread_slice_lengths),
            decltype(dim_access_order),
            VectorDim,
            ScalarPerVector,
            Sequence<true>,
            Sequence<true>>{in_grid_desc,
                            make_tuple(src_tensor.GetMultiIdxOffsets()),
                            out_grid_desc,
                            make_tuple(dst_tensor.GetMultiIdxOffsets()),
                            tensor_operation::element_wise::PassThrough{}};

        transfer.Run(tie(in_grid_desc),
                     tie(src_tensor.GetBuffer()),
                     tie(out_grid_desc),
                     tie(dst_tensor.GetBuffer()));
    }
    else if constexpr(!SrcTensorType::IsDynamicBuffer && DstTensorType::IsDynamicBuffer)
    {
        // Perform copy from StaticBuffer to DynamicBuffer
        const auto src_slice_origin_idxs =
            generate_tuple([&](auto) { return I0; }, Number<num_dims>{});

        auto transfer =
            ThreadwiseTensorSliceTransfer_v1r3<typename SrcTensorType::TensorElementType,
                                               typename DstTensorType::TensorElementType,
                                               remove_cvref_t<decltype(in_grid_desc)>,
                                               remove_cvref_t<decltype(out_grid_desc)>,
                                               tensor_operation::element_wise::PassThrough,
                                               decltype(thread_slice_lengths),
                                               decltype(dim_access_order),
                                               VectorDim,
                                               ScalarPerVector,
                                               InMemoryDataOperationEnum::Set,
                                               I1,
                                               true>{out_grid_desc,
                                                     dst_tensor.GetMultiIdxOffsets(),
                                                     tensor_operation::element_wise::PassThrough{}};

        transfer.Run(in_grid_desc,
                     src_slice_origin_idxs,
                     src_tensor.GetBuffer(),
                     out_grid_desc,
                     dst_tensor.GetBuffer());
    }
    else if constexpr(SrcTensorType::IsDynamicBuffer && !DstTensorType::IsDynamicBuffer)
    {
        // Perform copy from DynamicBuffer to StaticBuffer
        const auto dst_slice_origin_idxs =
            generate_tuple([&](auto) { return I0; }, Number<num_dims>{});
        auto transfer = ThreadwiseTensorSliceTransfer_v2<
            std::remove_const_t<typename SrcTensorType::TensorElementType>,
            std::remove_const_t<typename DstTensorType::TensorElementType>,
            remove_cvref_t<decltype(in_grid_desc)>,
            remove_cvref_t<decltype(out_grid_desc)>,
            decltype(thread_slice_lengths),
            decltype(dim_access_order),
            VectorDim,
            ScalarPerVector,
            I1,
            false,
            false>{in_grid_desc, src_tensor.GetMultiIdxOffsets()};

        transfer.Run(in_grid_desc,
                     src_tensor.GetBuffer(),
                     out_grid_desc,
                     dst_slice_origin_idxs,
                     dst_tensor.GetBuffer());
    }
    else
    {
        // Perform copy between StaticBuffers
        static_for<0, SrcShapeType::Size(), 1>{}([&](auto i) { dst_tensor(i) = src_tensor(i); });
    }
}

/**
 * \brief Perform generic copy between two tensors partitions (threadwise copy).
 *  Tensors must have the same size.
 *
 * \param src_tensor Source tensor.
 * \param dst_tensor Destination tensor.
 */
template <typename SrcTensorType, typename DstTensorType>
__host__ __device__ void copy(const SrcTensorType& src_tensor, DstTensorType& dst_tensor)
{
    // Generate default params
    using SrcShapeType         = remove_cvref_t<decltype(shape(src_tensor))>;
    constexpr index_t num_dims = SrcShapeType::Size();
    // Incrementing dims 0, 1, 2 ... num_dims - 1
    constexpr auto dim_access_order_tuple =
        generate_tuple([](auto i) { return Number<i>{}; }, Number<num_dims>{});
    constexpr index_t vector_dim        = num_dims - 1;
    constexpr index_t scalar_per_vector = 1;
    copy<decltype(dim_access_order_tuple), vector_dim, scalar_per_vector>(src_tensor, dst_tensor);
}

/**
 * \brief Perform optimized blockwise copy between two tensors. Tensors must have the
 *  same size.
 *
 * \note At now Vgpr and Sgpr are not supported.
 *
 * \tparam DimAccessOrderTuple Tuple with dimension access order.
 * \tparam VectorDim Dimension for vectorize read and write.
 * \tparam ScalarPerVector Number of scalar per vectorize read and write.
 * \param src_tensor Source tensor.
 * \param dst_tensor Destination tensor.
 * \param thread_layout Thread layout per each dimension for copy.
 */
template <typename DimAccessOrderTuple,
          index_t VectorDim,
          index_t ScalarPerVector,
          typename SrcTensorType,
          typename DstTensorType,
          typename ThreadShape,
          typename ThreadUnrolledDesc>
__device__ void
blockwise_copy(const SrcTensorType& src_tensor,
               DstTensorType& dst_tensor,
               [[maybe_unused]] const Layout<ThreadShape, ThreadUnrolledDesc>& thread_layout)
{
    static_assert(SrcTensorType::IsDynamicBuffer && DstTensorType::IsDynamicBuffer);
    static_assert(is_detected<is_tuple, DimAccessOrderTuple>::value);

    const auto& in_grid_desc  = layout(src_tensor).GetUnrolledDescriptor();
    const auto& out_grid_desc = layout(dst_tensor).GetUnrolledDescriptor();

    using SrcShapeType         = remove_cvref_t<decltype(shape(src_tensor))>;
    constexpr index_t num_dims = SrcShapeType::Size();

    constexpr auto tile_lengths_seq =
        generate_sequence_v2([](auto I) { return size(SrcShapeType{}.At(I)); }, Number<num_dims>{});
    constexpr auto thread_layout_seq =
        generate_sequence_v2([](auto I) { return size<I>(ThreadShape{}); }, Number<num_dims>{});
    constexpr auto dim_access_order = generate_sequence_v2(
        [](auto I) { return DimAccessOrderTuple{}.At(I); }, Number<num_dims>{});

    using ThisThreadBlock = ThisThreadBlock<size(ThreadShape{})>;

    // Perform copy between DynamicBuffers
    auto transfer = ThreadGroupTensorSliceTransfer_v7<
        ThisThreadBlock,
        Tuple<typename SrcTensorType::TensorElementType>,
        Tuple<typename DstTensorType::TensorElementType>,
        decltype(tie(in_grid_desc)),
        decltype(tie(out_grid_desc)),
        tensor_operation::element_wise::PassThrough,
        Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
        std::remove_const_t<decltype(tile_lengths_seq)>,
        std::remove_const_t<decltype(thread_layout_seq)>,
        std::remove_const_t<decltype(dim_access_order)>,
        std::remove_const_t<decltype(dim_access_order)>,
        VectorDim,
        ScalarPerVector,
        Sequence<true>,
        Sequence<true>>{in_grid_desc,
                        make_tuple(src_tensor.GetMultiIdxOffsets()),
                        out_grid_desc,
                        make_tuple(dst_tensor.GetMultiIdxOffsets()),
                        tensor_operation::element_wise::PassThrough{}};

    transfer.Run(tie(in_grid_desc),
                 tie(src_tensor.GetBuffer()),
                 tie(out_grid_desc),
                 tie(dst_tensor.GetBuffer()));
}

} // namespace wrapper
} // namespace ck
