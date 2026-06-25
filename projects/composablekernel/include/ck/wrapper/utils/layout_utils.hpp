// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/is_detected.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

// Disable from doxygen docs generation
/// @cond INTERNAL
// forward declaration
template <typename Shape, typename UnrolledDescriptorType>
struct Layout;

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

namespace {
namespace detail {
/**
 * \brief Generate packed (column-major) strides if not passed
 *
 * \param shape Tensor shape.
 * \return Generated column-major strides.
 */
template <typename... Ts>
__host__ __device__ constexpr static auto
GenerateColumnMajorPackedStrides(const Tuple<Ts...>& shape)
{
    const auto unrolled_shape = UnrollNestedTuple(shape);
    return generate_tuple(
        [&](auto i) {
            if constexpr(i.value == 0)
            {
                return Number<1>{};
            }
            else
            {
                return TupleReduce<Number<0>{}.value, i.value>([](auto x, auto y) { return x * y; },
                                                               unrolled_shape);
            }
        },
        Number<decltype(unrolled_shape)::Size()>{});
}

/**
 * \brief Create naive tensor descriptor from nested shape.
 *
 * \param shape Tensor shape.
 * \param strides Tensor strides.
 * \return Unrolled descriptor
 */
template <typename LayoutShape, typename LayoutStrides>
__host__ __device__ constexpr auto MakeUnrolledDescriptor(const LayoutShape& shape,
                                                          const LayoutStrides& strides)
{
    const auto unrolled_shape = UnrollNestedTuple(shape);
    if constexpr(is_same_v<LayoutStrides, Tuple<>>)
    {
        // if not passed, then generate
        const auto unrolled_strides = GenerateColumnMajorPackedStrides(unrolled_shape);
        static_assert(unrolled_shape.Size() == unrolled_strides.Size(),
                      "Size of strides and shape are not consistent.");
        return make_naive_tensor_descriptor(unrolled_shape, unrolled_strides);
    }
    else
    {
        const auto unrolled_strides = UnrollNestedTuple(strides);
        static_assert(unrolled_shape.Size() == unrolled_strides.Size(),
                      "Size of strides and shape are not consistent.");
        return make_naive_tensor_descriptor(unrolled_shape, unrolled_strides);
    }
}
} // namespace detail
} // namespace

/// @endcond

// make_*
/**
 * \brief Make layout function.
 *
 * \tparam Shape Shape for layout.
 * \tparam Strides Strides for layout.
 * \return Constructed layout.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr auto make_layout(const Shape& shape, const Strides& strides)
{
    using UnrolledDescriptorType = decltype(detail::MakeUnrolledDescriptor(Shape{}, Strides{}));
    return Layout<Shape, UnrolledDescriptorType>(shape,
                                                 detail::MakeUnrolledDescriptor(shape, strides));
}

/**
 * \brief Make layout function with packed strides
 *        (column-major).
 *
 * \tparam Shape Shape for layout.
 * \return Constructed layout.
 */
template <typename Shape>
__host__ __device__ constexpr auto make_layout(const Shape& shape)
{
    using UnrolledDescriptorType = decltype(detail::MakeUnrolledDescriptor(Shape{}, Tuple<>{}));
    return Layout<Shape, UnrolledDescriptorType>(shape,
                                                 detail::MakeUnrolledDescriptor(shape, Tuple<>{}));
}
// Layout helpers
// get
/**
 * \private
 * \brief Get dim.
 *
 * \param dim Dimension.
 * \return Returned the same dimension.
 */
template <typename T>
__host__ __device__ T constexpr get(const T& dim)
{
    return dim;
}

/**
 * \brief Get element from tuple (Shape/Strides/Idxs).
 *
 * \tparam idx Index to lookup.
 * \param tuple Tuple to lookup.
 * \return Requsted element.
 */
template <index_t idx, typename... Dims>
__host__ __device__ constexpr auto get(const Tuple<Dims...>& tuple)
{
    return tuple.At(Number<idx>{});
}

/**
 * \brief Get sub layout.
 *
 * \tparam idx Index to lookup.
 * \param layout Layout to create sub layout.
 * \return Requsted sub layout.
 */
template <index_t idx, typename Shape, typename UnrolledDesc>
__host__ __device__ constexpr auto get(const Layout<Shape, UnrolledDesc>& layout)
{
    const auto& shape    = layout.GetShape();
    const auto new_shape = get<idx>(shape);
    static_assert(is_detected<is_tuple, decltype(new_shape)>::value,
                  "Shape of sub layout must be tuple");

    constexpr auto old_shape_dims = decltype(UnrollNestedTuple(shape))::Size();
    constexpr auto new_shape_dims = decltype(UnrollNestedTuple(new_shape))::Size();
    constexpr auto shape_offset   = decltype(UnrollNestedTuple(TupleSlice<0, idx>(shape)))::Size();

    const auto unrolled_shape = UnrollNestedTuple(shape);
    const auto transforms     = generate_tuple(
        [&](auto i) {
            // Compare Idx with shape
            if constexpr(i < shape_offset || i >= shape_offset + new_shape_dims)
            {
                // Remove dimension
                return make_freeze_transform(Number<0>{});
            }
            else
            {
                return make_pass_through_transform(unrolled_shape.At(i));
            }
        },
        Number<old_shape_dims>{});

    const auto lower_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<old_shape_dims>{});
    const auto upper_dims = generate_tuple(
        [&](auto i) {
            if constexpr(i < shape_offset || i >= shape_offset + new_shape_dims)
                return Sequence<>{};

            else
            {
                return Sequence<i.value - shape_offset>{};
            }
        },
        Number<old_shape_dims>{});

    const auto& flatten_desc = layout.GetUnrolledDescriptor();
    auto new_desc = transform_tensor_descriptor(flatten_desc, transforms, lower_dims, upper_dims);
    return Layout<decltype(new_shape), decltype(new_desc)>(new_shape, new_desc);
}

/**
 * \brief Hierarchical get.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted element.
 */
template <index_t Idx, index_t... Idxs, typename T>
__host__ __device__ constexpr auto get(const T& elem)
{
    return get<Idxs...>(get<Idx>(elem));
}

// size
/**
 * \private
 * \brief Get size.
 *
 * \param dim Size.
 * \return Returned the same size.
 */
template <typename T>
__host__ __device__ T constexpr size(const T& dim)
{
    return dim;
}

/**
 * \brief Length get (product if tuple).
 *
 * \tparam idx Index to lookup.
 * \param layout Layout to get Shape of.
 * \return Requsted length.
 */
template <index_t idx, typename Shape, typename UnrolledDescriptorType>
__host__ __device__ constexpr auto size(const Layout<Shape, UnrolledDescriptorType>& layout)
{
    return layout.template GetLength<idx>();
}

/**
 * \brief Shape size (product of dims).
 *
 * \param shape Shape to lookup.
 * \return Requsted size.
 */
template <typename... ShapeDims>
__host__ __device__ constexpr auto size(const Tuple<ShapeDims...>& shape)
{
    const auto unrolled_shape = UnrollNestedTuple(shape);
    return TupleReduce<0, unrolled_shape.Size()>([](auto x, auto y) { return x * y; },
                                                 unrolled_shape);
}

/**
 * \brief Layout size (product of dims).
 *
 * \param layout Layout to calculate shape size.
 * \return Requsted size.
 */
template <typename Shape, typename UnrolledDescriptorType>
__host__ __device__ constexpr auto size(const Layout<Shape, UnrolledDescriptorType>& layout)
{
    return layout.GetLengths();
}

/**
 * \brief Length get from tuple (product if tuple).
 *
 * \tparam idx Index to lookup.
 * \param tuple Tuple to lookup.
 * \return Requsted length.
 */
template <index_t idx, typename... Ts>
__host__ __device__ constexpr auto size(const Tuple<Ts...>& tuple)
{
    return size(tuple.At(Number<idx>{}));
}

/**
 * \brief Hierarchical size.
 *
 * \tparam Idx First index to lookup (to avoid empty Idxs).
 * \tparam Idxs Next indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted element.
 */
template <index_t Idx, index_t... Idxs, typename T>
__host__ __device__ constexpr auto size(const T& elem)
{
    return size(get<Idx, Idxs...>(elem));
}

// rank
/**
 * \brief Get layout rank (num elements in shape).
 *
 * \param layout Layout to calculate rank.
 * \return Requsted rank.
 */
template <typename Shape, typename UnrolledDescriptorType>
__host__ __device__ constexpr auto
rank([[maybe_unused]] const Layout<Shape, UnrolledDescriptorType>& layout)
{
    return Shape::Size();
}

/**
 * \brief Get tuple rank (num elements in tuple).
 *        Return 1 if scalar passed.
 *
 * \param tuple Tuple to calculate rank.
 * \return Requsted rank.
 */
template <typename... Dims>
__host__ __device__ constexpr auto rank([[maybe_unused]] const Tuple<Dims...>& tuple)
{
    return Tuple<Dims...>::Size();
}

/**
 * \private
 * \brief Rank for scalar
 *
 * \param dim Dimension scalar.
 * \return Returned 1.
 */
template <index_t IDim>
__host__ __device__ constexpr index_t rank([[maybe_unused]] const Number<IDim>& dim)
{
    return 1;
}

/**
 * \private
 * \brief Rank for scalar
 *
 * \param dim Dimension scalar.
 * \return Returned 1.
 */
__host__ __device__ constexpr index_t rank([[maybe_unused]] const index_t& dim) { return 1; }

/**
 * \brief Hierarchical rank.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted rank.
 */
template <index_t... Idxs, typename T>
__host__ __device__ constexpr auto rank(const T& elem)
{
    return rank(get<Idxs...>(elem));
}

// depth
/**
 * \brief Get depth of the layout shape (return 0 if scalar).
 *
 * \param layout Layout to calculate depth.
 * \return Requsted depth.
 */
template <typename Shape, typename UnrolledDescriptorType>
__host__ __device__ constexpr auto depth(const Layout<Shape, UnrolledDescriptorType>& layout)
{
    const auto& shape = layout.GetShape();
    return TupleDepth(shape);
}

/**
 * \brief Get depth of the tuple. (return 0 if scalar)
 *
 * \param tuple Tuple to calculate depth.
 * \return Requsted depth.
 */
template <typename... Dims>
__host__ __device__ constexpr auto depth(const Tuple<Dims...>& tuple)
{
    return TupleDepth(tuple);
}

/**
 * \private
 * \brief Depth for scalar
 *
 * \param dim Scalar.
 * \return Returned 0.
 */
template <index_t IDim>
__host__ __device__ constexpr index_t depth([[maybe_unused]] const Number<IDim>& dim)
{
    return 0;
}

/**
 * \private
 * \brief Depth for scalar
 *
 * \param dim Scalar.
 * \return Returned 0.
 */
__host__ __device__ constexpr index_t depth([[maybe_unused]] const index_t& dim) { return 0; }

/**
 * \brief Hierarchical depth.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted depth.
 */
template <index_t... Idxs, typename T>
__host__ __device__ constexpr auto depth(const T& elem)
{
    return depth(get<Idxs...>(elem));
}

/**
 * \brief Get Layout shape.
 *
 * \param layout Layout to get shape from.
 * \return Requsted shape.
 */
template <typename LayoutType>
__host__ __device__ constexpr const auto& shape(const LayoutType& layout)
{
    return layout.GetShape();
}

// pad
/**
 * \brief Pad layout shapes to be adjusted to tile lengths.
 *
 *
 * \param layout Layout to pad.
 * \param tile_lengths Tile lengths to align layout shape.
 * \return Padded layout.
 */
template <typename Shape, typename UnrolledDesc, typename TileLengths>
__host__ __device__ constexpr auto pad(const Layout<Shape, UnrolledDesc>& layout,
                                       const TileLengths& tile_lengths)
{
    auto& unrolled_desc = layout.GetUnrolledDescriptor();
    // Generate sequence with ones to mark that all dims will be padded
    constexpr auto do_pads_seq =
        generate_sequence_v2([](auto) { return Number<1>{}; }, Number<Shape::Size()>{});
    // Create descriptor with padding
    auto padded_desc =
        tensor_operation::device::PadTensorDescriptor(unrolled_desc, tile_lengths, do_pads_seq);
    // Generate padded shape
    const auto padded_shape = generate_tuple(
        [&](auto i) { return padded_desc.GetLength(Number<i>{}); }, Number<TileLengths::Size()>{});
    // Create layout
    return Layout<decltype(padded_shape), decltype(padded_desc)>(padded_shape, padded_desc);
}

// unmerge
/**
 * \brief Unmerge selected dim in layout.
 *
 * \tparam Idx Index to dimension being unmerged.
 * \param layout Layout to pad.
 * \param new_lengths Dimensions into which the indicated dimension will be divided.
 * \param new_indexes Indexes to shuffle dims. Dims for unmerged dim should be nested.
 * \return Unmerged layout.
 */
template <index_t Idx, typename Shape, typename UnrolledDesc, typename NewLengths, typename NewIdxs>
__host__ __device__ constexpr auto unmerge(const Layout<Shape, UnrolledDesc>& layout,
                                           const NewLengths& new_lengths,
                                           [[maybe_unused]] const NewIdxs& new_indexes)
{
    const auto& layout_shape = shape(layout);
    auto& unrolled_desc      = layout.GetUnrolledDescriptor();
    constexpr auto dims      = Shape::Size();
    // Generate transforms
    const auto transforms = generate_tuple(
        [&](auto i) {
            if constexpr(i == Idx)
            {
                return make_unmerge_transform(new_lengths);
            }
            else
            {
                return make_pass_through_transform(layout_shape.At(i));
            }
        },
        Number<dims>{});

    constexpr auto lower_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<dims>{});
    constexpr auto upper_dims = generate_tuple(
        [&](auto i) {
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, NewIdxs>>::value)
            {
                constexpr auto idxs_tuple = tuple_element_t<i.value, NewIdxs>{};
                return to_sequence(idxs_tuple);
            }
            else
            {
                constexpr index_t index = tuple_element_t<i.value, NewIdxs>{};
                return Sequence<index>{};
            }
        },
        Number<dims>{});

    const auto unmerged_desc =
        transform_tensor_descriptor(unrolled_desc, transforms, lower_dims, upper_dims);
    const auto unmerged_shape =
        generate_tuple([&](auto i) { return unmerged_desc.GetLength(Number<i>{}); },
                       Number<decltype(unmerged_desc)::GetNumOfVisibleDimension()>{});

    // Create layout
    return Layout<decltype(unmerged_shape), decltype(unmerged_desc)>(unmerged_shape, unmerged_desc);
}

} // namespace wrapper
} // namespace ck
