// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "tensor_utils.hpp"
#include "layout_utils.hpp"

#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace {

namespace detail {

/**
 * \brief Calculate shape for partition based on number of threads per each dim and
 * previous shape
 *
 * \param shape Base tensor shape.
 * \param thread_lengths Tuple of thread lengths.
 * \return Partition shape.
 */
template <typename... Ts, typename... Ls>
__host__ __device__ constexpr auto CalculateLocalPartitionShape(const Tuple<Ts...>& shape,
                                                                const Tuple<Ls...>& thread_lengths)
{
    static_assert(Tuple<Ts...>::Size() == Tuple<Ls...>::Size(), "Wrong thread_lengths shape.");
    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            const auto slice_len =
                ck::math::integer_divide_ceil(size<num_i>(shape), thread_lengths.At(num_i));
            return slice_len;
        },
        Number<Tuple<Ls...>::Size()>{});
}

/**
 * \brief Apply projection.
 *
 * \param base_tuple Tuple to apply projection.
 * \param projection Projection is used to remove selected dim from
 * partitioning. Use `slice(X)` to remove dimension, where X is dim
 * size. Use `Number<1>{}` to keep it.
 * \return Multi index after projection.
 */
template <typename MultiIndex, typename ProjectionTuple>
__host__ __device__ constexpr auto
ApplyProjection([[maybe_unused]] const MultiIndex& base_tuple,
                [[maybe_unused]] const ProjectionTuple& projection)
{
    if constexpr(is_same_v<ProjectionTuple, Tuple<>>)
    {
        return Tuple<>{};
    }
    else
    {
        auto base_tuple_after_projection = generate_tuple(
            [&](auto i) {
                const auto i_num = Number<i.value>{};
                static_assert(
                    is_detected<is_slice, tuple_element_t<i_num, ProjectionTuple>>::value ||
                    is_same_v<tuple_element_t<i_num, ProjectionTuple>, Number<1>>);
                if constexpr(is_detected<is_slice, tuple_element_t<i_num, ProjectionTuple>>::value)
                {
                    // When slice (to remove), then insert empty tuple (will be removed in next
                    // step).
                    return Tuple<>{};
                }
                else
                {
                    return make_tuple(base_tuple.At(i_num));
                }
            },
            Number<MultiIndex::Size()>{});
        // Remove empty tuples
        return UnrollNestedTuple<0, 1>(base_tuple_after_projection);
    }
}

/**
 * \brief Calculate shape with dims from projection.
 *
 * \param shape Base tensor shape.
 * \param projection Projection is used to remove selected dim from
 * partitioning. Use `slice(X)` to remove dimension, where X is dim
 * size. Use `Number<1>{}` to keep it.
 * \return Shape with dims from projection
 */
template <typename... Ts, typename... Ps>
__host__ __device__ constexpr auto CalculateShapeWithProjection(const Tuple<Ts...>& shape,
                                                                const Tuple<Ps...>& projection)
{
    return generate_tuple(
        [&](auto i) {
            if constexpr(is_detected<is_slice, tuple_element_t<i, Tuple<Ps...>>>::value)
            {
                return size<i>(projection).to_;
            }
            else
            {
                // number of shape element in actual fragment of shape and projection (method to
                // calculate shape idx)
                constexpr index_t shape_i =
                    detail::ApplyProjection(TupleSlice<0, i>(Tuple<Ts...>{}),
                                            TupleSlice<0, i>(Tuple<Ps...>{}))
                        .Size();
                return size<shape_i>(shape);
            }
        },
        Number<Tuple<Ps...>::Size()>{});
}

/**
 * \brief Calculate total number of blocks.
 *
 * \param shape Base tensor shape.
 * \param tile_shape Tile shape.
 * \return Tuple with blocks number.
 */
template <typename... Ts, typename... Ls, typename... Ps>
__host__ __device__ constexpr auto CalculateGridSize(const Tuple<Ts...>& shape,
                                                     const Tuple<Ls...>& tile_shape)
{
    return generate_tuple(
        [&](auto i) { return ck::math::integer_divide_ceil(size<i>(shape), size<i>(tile_shape)); },
        Number<Tuple<Ls...>::Size()>{});
}

/**
 * \brief Calculate scaled offset for new partition/tile.
 *
 * \param thread_idxs Thread 1d id.
 * \param partition_lengths_seq Sequence of partition shape.
 * \param old_offset_idxs Multi index offset from base tensor to shift values.
 * \return Partition shape.
 */
template <typename ThreadIdxs, typename PartitionLengthsSeq, typename OldOffsetIdxs>
__host__ __device__ constexpr auto
CalculateOffsetMultiIdxs(const ThreadIdxs& thread_idxs,
                         const PartitionLengthsSeq& partition_lengths_seq,
                         const OldOffsetIdxs& old_offset_idxs)
{
    return thread_idxs * partition_lengths_seq + old_offset_idxs;
}

/**
 * \brief Select dims to partition (skip if slice).
 *
 * \param block_idxs Input block indexes.
 * \return Partitioned dims.
 */
template <typename BlockIdxs>
__host__ __device__ constexpr auto GetDimsToPartition([[maybe_unused]] const BlockIdxs& block_idxs)
{
    const auto dims_to_partition = generate_tuple(
        [&](auto i) {
            if constexpr(!is_detected<is_slice, tuple_element_t<i, BlockIdxs>>::value)
            {
                return Number<i>{};
            }
            else
            {
                return Tuple<>{};
            }
        },
        Number<BlockIdxs::Size()>{});
    // Remove empty tuples
    return UnrollNestedTuple<0, 1>(dims_to_partition);
}

/**
 * \brief Replace slices with zeros (Slice dims are not partitioned).
 *
 * \param block_idxs Input block indexes.
 * \return Parsed dims.
 */
template <typename BlockIdxs>
__host__ __device__ constexpr auto ReplaceSlicesWithZeros(const BlockIdxs& block_idxs)
{
    return generate_tuple(
        [&](auto i) {
            if constexpr(!is_detected<is_slice, tuple_element_t<i, BlockIdxs>>::value)
            {
                return block_idxs.At(i);
            }
            else
            {
                return Number<0>{};
            }
        },
        Number<BlockIdxs::Size()>{});
}

/**
 * \brief Calculate default projection.
 *
 * \param tile_shape Tile shape.
 * \return Default projection (filled with Number<1>{}).
 */
template <typename TileShape>
__host__ __device__ constexpr auto
GenerateDefaultProjection([[maybe_unused]] const TileShape tile_shape)
{
    return generate_tuple([&](auto) { return Number<1>{}; }, Number<TileShape::Size()>{});
}

/**
 * \brief Calculate thread multi index from 1d thread index.
 *
 * \param thread_layout Layout of threads (could not be nested).
 * \param thread_id Thread index represented as integer.
 * \return Multi index.
 */
template <typename ThreadShape, typename ThreadUnrolledDesc>
__host__ __device__ constexpr auto CalculateThreadMultiIdx(
    [[maybe_unused]] const Layout<ThreadShape, ThreadUnrolledDesc>& thread_layout,
    const index_t thread_id)
{
    static_assert(ThreadUnrolledDesc::GetNumOfTransform() == 1,
                  "Thread layout should not be transformed.");
    constexpr auto embed_transform = ThreadUnrolledDesc{}.GetTransforms().At(Number<0>{});
    constexpr auto shape           = ThreadShape{};
    constexpr auto strides         = embed_transform.coefficients_;

    return generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            return (thread_id / strides.At(num_i)) % shape.At(num_i);
        },
        Number<ThreadShape::Size()>{});
}
} // namespace detail
} // namespace
/// @endcond

/**
 * \brief Create local partition for thread (At now only packed partition
 * is supported).
 *
 * \param tensor Tensor for partition.
 * \param thread_layout Layout of threads (could not be transformed).
 * \param thread_id Thread index represented as integer.
 * \param projection Projection is used to remove selected dim from
 * partitioning. Use `slice(X)` to remove dimension, where X is dim
 * size. Use `Number<1>{}` to keep it.
 * \return Partition tensor.
 */
template <typename TensorType,
          typename ThreadShape,
          typename ThreadUnrolledDesc,
          typename ProjectionTuple>
__host__ __device__ constexpr auto
make_local_partition(TensorType& tensor,
                     [[maybe_unused]] const Layout<ThreadShape, ThreadUnrolledDesc>& thread_layout,
                     const index_t thread_id,
                     const ProjectionTuple& projection)
{
    static_assert(!IsNestedTuple(ThreadShape{}));
    // Calculate new partition shape
    const auto& tensor_shape = shape(tensor);
    // Calculate projected thread lengths
    constexpr auto projected_thread_lengths =
        detail::ApplyProjection(ThreadShape{}, ProjectionTuple{});
    constexpr auto partition_shape =
        detail::CalculateLocalPartitionShape(decltype(tensor_shape){}, projected_thread_lengths);
    constexpr auto partition_shape_seq =
        generate_sequence_v2([&](auto I) { return size<I>(partition_shape); },
                             Number<decltype(partition_shape)::Size()>{});
    // Calculate thread idxs and offsets
    const auto thread_idxs = detail::CalculateThreadMultiIdx(thread_layout, thread_id);
    // Apply projection on thread idxs to remove not needed idxs
    const auto projected_thread_idxs = detail::ApplyProjection(thread_idxs, projection);
    const auto offset_multi_idxs     = detail::CalculateOffsetMultiIdxs(
        projected_thread_idxs, partition_shape_seq, tensor.GetMultiIdxOffsets());
    // Create new layout and tensor
    auto& unrolled_desc = layout(tensor).GetUnrolledDescriptor();
    // Slice descriptor
    const auto transforms = generate_tuple(
        [&](auto i) {
            return make_slice_transform(partition_shape.At(i),
                                        offset_multi_idxs.At(i),
                                        partition_shape.At(i) + offset_multi_idxs.At(i));
        },
        Number<remove_reference_t<decltype(tensor_shape)>::Size()>{});
    const auto lower_upper_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; },
                       Number<remove_reference_t<decltype(tensor_shape)>::Size()>{});
    auto sliced_desc =
        transform_tensor_descriptor(unrolled_desc, transforms, lower_upper_dims, lower_upper_dims);
    // Create layout
    const auto partition_layout =
        Layout<remove_reference_t<decltype(partition_shape)>, decltype(sliced_desc)>(
            partition_shape, sliced_desc);
    auto partition_tensor =
        make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), partition_layout);
    // Apply offsets
    return partition_tensor;
}

/**
 * \brief Create local partition for thread (At now only packed partition
 * is supported).
 *
 * \param tensor Tensor for partition.
 * \param thread_lengths Layout of threads (could not be nested).
 * \param thread_id Thread index represented as integer.
 * \return Partition tensor.
 */
template <typename TensorType, typename ThreadShape, typename ThreadUnrolledDesc>
__host__ __device__ constexpr auto
make_local_partition(TensorType& tensor,
                     const Layout<ThreadShape, ThreadUnrolledDesc>& thread_lengths,
                     const index_t thread_id)
{
    const auto projection = detail::GenerateDefaultProjection(ThreadShape{});
    return make_local_partition(tensor, thread_lengths, thread_id, projection);
}

/**
 * \brief Create local tile for thread block. (At now only packed tile
 * is supported).
 *
 * \note Temporary to gain the best performance use 2d
 * tile_shape.
 *
 *
 * \param tensor Tensor for partition.
 * \param tile_shape Shapes of requested tile.
 * \param block_idxs Tuple of block indexes represented as integer. If slice,
 * then get whole dim.
 * \param projection Projection is used to remove selected dim from
 * partitioning. Use `slice(X)` to remove dimension, where X is dim
 * size. Use `Number<1>{}` to keep it.
 * \return Tile tensor.
 */
template <typename TensorType,
          typename BlockShapeTuple,
          typename BlockIdxs,
          typename ProjectionTuple>
__host__ __device__ constexpr auto make_local_tile(const TensorType& tensor,
                                                   const BlockShapeTuple& tile_shape,
                                                   const BlockIdxs& block_idxs,
                                                   const ProjectionTuple& projection)
{
    static_assert(!IsNestedTuple(BlockShapeTuple{}));
    static_assert(!IsNestedTuple(BlockIdxs{}));

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    auto& aligned_desc = layout(tensor).GetMergedNestingDescriptor();

    constexpr auto projected_tile_shape =
        detail::ApplyProjection(BlockShapeTuple{}, ProjectionTuple{});
    // Number of dims which are partitioned
    constexpr auto dims_to_partition = detail::GetDimsToPartition(BlockIdxs{});
    const auto parsed_block_idxs     = detail::ReplaceSlicesWithZeros(block_idxs);
    if constexpr(decltype(dims_to_partition)::Size() == I2)
    {
        const auto shape_with_projection_dims =
            detail::CalculateShapeWithProjection(shape(tensor), projection);
        // Set Value for M, N partition
        const auto M             = shape_with_projection_dims.At(dims_to_partition.At(I0));
        const auto N             = shape_with_projection_dims.At(dims_to_partition.At(I1));
        constexpr auto MPerBlock = BlockShapeTuple{}.At(dims_to_partition.At(I0));
        constexpr auto NPerBlock = BlockShapeTuple{}.At(dims_to_partition.At(I1));
        auto m_n_desc            = make_naive_tensor_descriptor_packed(make_tuple(M, N));
        // Get 1D block id
        const auto grid_size = detail::CalculateGridSize(shape_with_projection_dims, tile_shape);
        const auto block_lengths_desc = make_naive_tensor_descriptor_packed(grid_size);
        const index_t block_id_1d     = block_lengths_desc.CalculateOffset(parsed_block_idxs);
        // Optimized version for 2d tile shape [MxN]
        const auto block_2_tile_map =
            BlockToCTileMap_M00_N0_M01Adapt<MPerBlock,
                                            NPerBlock,
                                            remove_cvref_t<decltype(m_n_desc)>>(m_n_desc);
        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(block_id_1d));
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);
        // Apply 0 for non partitioned dims
        const auto offset_multi_idxs = generate_tuple(
            [&](auto i) {
                if constexpr(i == dims_to_partition.At(I0))
                {
                    return m_block_data_idx_on_grid;
                }
                else if constexpr(i == dims_to_partition.At(I1))
                {
                    return n_block_data_idx_on_grid;
                }
                else
                {
                    return Number<0>{};
                }
            },
            Number<BlockShapeTuple::Size()>{});
        const auto projected_offset_multi_idxs =
            detail::ApplyProjection(offset_multi_idxs, projection);
        // Create new layout and tensor
        const auto tile_layout =
            Layout<remove_reference_t<decltype(projected_tile_shape)>, decltype(aligned_desc)>(
                projected_tile_shape, aligned_desc);
        auto tile_tensor =
            make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), tile_layout);
        // Apply offsets
        tile_tensor.SetMultiIdxOffset(to_multi_index(projected_offset_multi_idxs));
        return tile_tensor;
    }
    else
    {
        // Calculate offsets
        // Sequence with data to process per block
        using ProjectedTileShapeTuple = decltype(projected_tile_shape);
        constexpr auto projected_tile_shape_seq =
            generate_sequence_v2([](auto I) { return ProjectedTileShapeTuple{}.At(I); },
                                 Number<ProjectedTileShapeTuple::Size()>{});
        // Tuple with number of blocks
        const auto projected_block_idxs =
            to_multi_index(detail::ApplyProjection(parsed_block_idxs, projection));
        const auto offset_multi_idxs = detail::CalculateOffsetMultiIdxs(
            projected_block_idxs, projected_tile_shape_seq, tensor.GetMultiIdxOffsets());
        // Create new layout and tensor
        const auto tile_layout =
            Layout<remove_reference_t<ProjectedTileShapeTuple>, decltype(aligned_desc)>(
                projected_tile_shape, aligned_desc);
        auto tile_tensor =
            make_tensor<TensorType::TensorBufferAddressSpace>(tensor.GetPointer(), tile_layout);
        // Apply offsets
        tile_tensor.SetMultiIdxOffset(to_multi_index(offset_multi_idxs));
        return tile_tensor;
    }
}

/**
 * \brief Create local tile for thread block. (At now only packed tile
 * is supported).
 *
 * \note Currently to get the best performance please use 2d shape.
 *
 * \param tensor Tensor for partition.
 * \param tile_shape Shapes of requested tile.
 * \param block_idxs Tuple of block indexes represented as integer. If slice,
 * then get whole dim.
 * \return Tile tensor.
 */
template <typename TensorType, typename BlockShapeTuple, typename BlockIdxs>
__host__ __device__ constexpr auto make_local_tile(const TensorType& tensor,
                                                   const BlockShapeTuple& tile_shape,
                                                   const BlockIdxs& block_idxs)
{
    const auto projection = detail::GenerateDefaultProjection(BlockShapeTuple{});
    return make_local_tile(tensor, tile_shape, block_idxs, projection);
}

} // namespace wrapper
} // namespace ck
