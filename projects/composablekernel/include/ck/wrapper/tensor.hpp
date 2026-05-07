// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/tensor_utils.hpp"
#include "utils/tensor_partition.hpp"
#include "utils/layout_utils.hpp"

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
 * \brief Check if Tuple contains Slice object
 *
 * \return True if tuple contains Slice object.
 */
template <typename T>
__host__ __device__ constexpr bool HasSlice(T&&)
{
    return is_detected<is_slice, T>::value;
}
template <typename... Ts>
__host__ __device__ constexpr bool HasSlice(Tuple<Ts...>&&)
{
    return (HasSlice(Ts{}) || ...);
}

/**
 * \brief Calculate new shape after slice from parent shape.
 *
 * \param idxs Tuple of indexes defining slice ranges.
 * \param shape Shape which will be sliced.
 * \return New tensor shape.
 */
template <typename... Ts, typename SlicedShape>
__host__ __device__ constexpr auto GetSlicedShape(const Tuple<Ts...>& idxs,
                                                  const SlicedShape& shape)
{
    // Pack each value in tuple to remove empty tuples after generation
    auto new_shape = generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
            {
                if constexpr(!detail::HasSlice(tuple_element_t<i.value, Tuple<Ts...>>{}))
                {
                    // if tuple does not have any slice then we can remove dimension
                    return Tuple<>{};
                }
                else
                {
                    // if tuple then recurrence
                    return make_tuple(GetSlicedShape(idxs.At(num_i), shape.At(num_i)));
                }
            }
            else if constexpr(is_detected<is_slice, tuple_element_t<i.value, Tuple<Ts...>>>::value)
            {
                // calculate new dimension
                const auto& dim = size(shape.At(num_i));
                const auto val  = idxs.At(num_i).range(dim);
                return make_tuple(val);
            }
            else
            {
                // remove dimension for just value
                return Tuple<>{};
            }
        },
        Number<Tuple<Ts...>::Size()>{});
    // Remove empty tuples (deleted elements) and return
    return UnrollNestedTuple<0, 1>(new_shape);
}

/**
 * \brief Generate Freeze for each of nested shape.
 *
 * \param idx Tuple of start indices for slice.
 * \param shape Shape which will be freezed.
 * \return Generated freeze transforms.
 */
template <typename T, typename Shape>
__host__ __device__ constexpr auto GenerateMultipleFreeze(T idx, const Shape& shape)
{
    const auto unrolled_shape = UnrollNestedTuple(shape);
    return generate_tuple(
        [&](auto i) {
            // dimension offset from idx
            const auto dim     = unrolled_shape.At(Number<i>{});
            const auto dim_idx = idx % dim;
            idx /= dim;
            return make_freeze_transform(dim_idx);
        },
        Number<decltype(unrolled_shape)::Size()>{});
}

/**
 * \brief Generate transforms for slice tensor.
 *
 * \param idx Tuple of start indices for slice.
 * \param shape Shape which will be sliced.
 * \return Generated transforms.
 */
template <typename... Ts, typename Shape>
__host__ __device__ constexpr auto GenerateSliceTransforms(const Tuple<Ts...>& idx,
                                                           const Shape& shape)
{
    // Pack each value in tuple to remove empty tuples after generation
    auto transforms = generate_tuple(
        [&](auto i) {
            constexpr auto num_i = Number<i>{};
            if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
            {
                return GenerateSliceTransforms(idx.At(num_i), shape.At(num_i));
            }
            else if constexpr(is_detected<is_slice, tuple_element_t<i.value, Tuple<Ts...>>>::value)
            {

                const auto from  = idx.At(num_i).from_;
                const auto dim   = size<num_i>(shape);
                const auto range = idx.At(num_i).range(dim);
                return make_slice_transform(range, from, from + range);
            }
            else
            {
                // remove dimension for just value
                return GenerateMultipleFreeze(idx.At(num_i), shape.At(num_i));
            }
        },
        Number<Tuple<Ts...>::Size()>{});
    // Remove empty tuples (deleted elements) and return
    return UnrollNestedTuple(transforms);
}

template <index_t i, typename LowerIndex>
__host__ __device__ constexpr auto GetSequenceVal(const ck::Freeze<LowerIndex>&)
{
    // There is no output for Freeze transform
    return Sequence<>{};
}

template <index_t i, typename LowLength, typename SliceBegin, typename SliceEnd>
__host__ __device__ constexpr auto GetSequenceVal(const ck::Slice<LowLength, SliceBegin, SliceEnd>&)
{
    return Sequence<i>{};
}

template <index_t i>
__host__ __device__ constexpr auto GenerateUpperDims(const Tuple<>&)
{
    return Tuple<>{};
}

template <index_t i, typename... Transforms>
__host__ __device__ constexpr auto GenerateUpperDims(const Tuple<Transforms...>& transforms)
{
    constexpr auto num_transforms = Tuple<Transforms...>::Size();
    // Deduce Sequence element for specific transform
    const auto current_elem = GetSequenceVal<i>(transforms.At(Number<0>{}));
    if constexpr(is_same_v<decltype(current_elem), const Sequence<>>)
    {
        const auto next_tuple = GenerateUpperDims<i>(TupleSlice<1, num_transforms>(transforms));
        return concat_tuple(make_tuple(current_elem), next_tuple);
    }
    else
    {
        // Increase i if current_elem is Slice transform
        const auto next_tuple = GenerateUpperDims<i + 1>(TupleSlice<1, num_transforms>(transforms));
        return concat_tuple(make_tuple(current_elem), next_tuple);
    }
}

template <typename... Ts, typename Shape, typename UnrolledDescriptor>
__host__ __device__ constexpr auto GenerateSlicedDescriptor(const Tuple<Ts...>& idx,
                                                            const Shape& shape,
                                                            const UnrolledDescriptor& flatten_desc)
{
    constexpr auto old_shape_dims = decltype(UnrollNestedTuple(shape))::Size();

    const auto transforms     = GenerateSliceTransforms(idx, shape);
    using TransformsTupleType = decltype(transforms);

    const auto lower_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<old_shape_dims>{});
    const auto upper_dims = decltype(GenerateUpperDims<0>(TransformsTupleType{})){};
    return transform_tensor_descriptor(flatten_desc, transforms, lower_dims, upper_dims);
}
} // namespace detail
} // namespace
/// @endcond

/**
 * \brief Tensor wrapper that performs static and dynamic buffer logic.
 * The tensor is based on a descriptor stored in the Layout. Additionally,
 * tensor can be sliced or shifted using multi-index offset.
 *
 * \tparam BufferAddressSpace Memory type (Generic, Global, LDS, VGPR, SGPR).
 * \tparam ElementType Element data type.
 * \tparam Shape Tensor shape (layout component).
 * \tparam UnrolledDescriptorType Flatten descriptor (layout component).
 */
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
struct Tensor
{
    public:
    using ElementSpaceSize  = decltype(Layout<Shape, UnrolledDescriptorType>{
        Shape{}, UnrolledDescriptorType{}}.GetElementSpaceSize()); // SpaceSize type for buffer
    using TensorElementType = std::conditional_t<
        is_scalar_type<ElementType>::value,
        ElementType,
        typename scalar_type<std::remove_const_t<ElementType>>::type>; // DataType

    static constexpr MemoryTypeEnum TensorBufferAddressSpace = BufferAddressSpace;
    static constexpr bool IsDynamicBuffer = !(BufferAddressSpace == MemoryTypeEnum ::Sgpr ||
                                              BufferAddressSpace == MemoryTypeEnum ::Vgpr);

    __host__ __device__ Tensor() = delete;
    __host__ __device__ constexpr Tensor(ElementType* pointer,
                                         const Layout<Shape, UnrolledDescriptorType>& layout)
        : layout_(layout),
          buffer_(make_dynamic_buffer<BufferAddressSpace>(pointer, layout.GetElementSpaceSize())),
          multi_idx_offset_(make_zero_multi_index<Shape::Size()>()),
          base_offset_(0)
    {
        static_assert(IsDynamicBuffer, "Wrong BufferAddressSpace for register.");
    }

    __host__ __device__ constexpr Tensor(const Layout<Shape, UnrolledDescriptorType>& layout)
        : layout_(layout),
          multi_idx_offset_(make_zero_multi_index<Shape::Size()>()),
          base_offset_(0)
    {
        static_assert(!IsDynamicBuffer, "Wrong BufferAddressSpace for register.");
    }

    __host__ __device__ constexpr const Layout<Shape, UnrolledDescriptorType>& GetLayout() const
    {
        return layout_;
    }

    /**
     * \brief Get the new sliced tensor.
     *
     * \param idx Tuple of indices: slice(from,to) or scalar.
     * \return Sliced tensor.
     */
    template <typename... Ts, enable_if_t<detail::HasSlice(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator[](const Tuple<Ts...>& idx)
    {
        static_assert(IsDynamicBuffer, "Register slice is not supported");
        const auto& shape = layout_.GetShape();
        auto new_shape    = detail::GetSlicedShape(idx, shape);

        const auto& flatten_desc = layout_.GetUnrolledDescriptor();
        auto new_desc            = detail::GenerateSlicedDescriptor(idx, shape, flatten_desc);
        const auto new_layout =
            Layout<decltype(new_shape), decltype(new_desc)>(new_shape, new_desc);
        // Update embed offset
        base_offset_ -= new_layout(make_tuple(Number<0>{}));
        return make_tensor<BufferAddressSpace>(buffer_.p_data_, new_layout);
    }

    template <typename... Ts, enable_if_t<detail::HasSlice(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator()(const Tuple<Ts...>& idx)
    {
        return this->operator[](idx);
    }

    template <typename... Idxs, enable_if_t<detail::HasSlice(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ auto operator()(Idxs... idxs)
    {
        return this->operator[](make_tuple(idxs...));
    }

    /**
     * \brief Getter of the tensor's const value reference.
     *
     * \param idx Tuple of indices.
     * \return Requested value.
     */
    template <typename... Ts, enable_if_t<!detail::HasSlice(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ const TensorElementType& operator[](const Tuple<Ts...>& idx) const
    {
        if constexpr(IsDynamicBuffer)
        {
            const index_t offset = layout_(idx) + base_offset_;
            return buffer_[offset];
        }
        else
        {
            constexpr index_t index_offset = Layout<Shape, UnrolledDescriptorType>{
                Shape{},
                UnrolledDescriptorType{}}.template operator()<Tuple<Ts...>>();
            // Calculate and apply base offset in compile-time
            constexpr index_t base_offset = Layout<Shape, UnrolledDescriptorType>{
                Shape{},
                UnrolledDescriptorType{}}.template operator()<MultiIndex<Shape::Size()>>();
            return buffer_[Number<index_offset + base_offset>{}];
        }
    }

    template <typename... Ts, enable_if_t<!detail::HasSlice(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ const TensorElementType& operator()(const Tuple<Ts...>& idx) const
    {
        return this->operator[](idx);
    }

    template <typename... Idxs, enable_if_t<!detail::HasSlice(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ const TensorElementType& operator()(Idxs... idxs) const
    {
        return this->operator[](make_tuple(idxs...));
    }

    /**
     * \brief Getter of tensor value reference.
     *
     * \param idx Tuple of indices.
     * \return Requested value.
     */
    template <typename... Ts, enable_if_t<!detail::HasSlice(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ TensorElementType& operator[](const Tuple<Ts...>& idx)
    {
        if constexpr(IsDynamicBuffer)
        {
            const index_t offset = layout_(idx) + base_offset_;
            return buffer_(offset);
        }
        else
        {
            constexpr index_t index_offset = Layout<Shape, UnrolledDescriptorType>{
                Shape{},
                UnrolledDescriptorType{}}.template operator()<Tuple<Ts...>>();
            // Apply embed offset (calculate in compiletime)
            constexpr index_t base_offset = Layout<Shape, UnrolledDescriptorType>{
                Shape{},
                UnrolledDescriptorType{}}.template operator()<MultiIndex<Shape::Size()>>();
            return buffer_(Number<index_offset + base_offset>{});
        }
    }

    template <typename... Ts, enable_if_t<!detail::HasSlice(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ TensorElementType& operator()(const Tuple<Ts...>& idx)
    {
        return this->operator[](idx);
    }

    template <typename... Idxs, enable_if_t<!detail::HasSlice(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ TensorElementType& operator()(Idxs... idxs)
    {
        return this->operator[](make_tuple(idxs...));
    }

    /**
     * \brief Get descriptor with all nested dimensions merged.
     *
     * \return Merged nests descriptor.
     */
    __host__ __device__ constexpr auto GetMergedNestingDescriptor()
    {
        return layout_.GetMergedNestingDescriptor();
    }

    /**
     * \brief Get pointer to the data.
     *
     * \return Pointer.
     */
    __host__ __device__ TensorElementType* GetPointer() const { return buffer_.p_data_; }

    __host__ __device__ constexpr auto& GetBuffer() { return buffer_; }
    __host__ __device__ constexpr auto& GetBuffer() const { return buffer_; }

    /**
     * \brief Get multi index offset to the data.
     *
     * \return Multi index offset.
     */
    __host__ __device__ constexpr auto& GetMultiIdxOffsets() const { return multi_idx_offset_; }

    /**
     * \brief Apply multi index offset on the tensor.
     *
     * \param multi_idx_offset Multi index offset.
     */
    template <typename MultiIdxOffsets>
    __host__ __device__ constexpr void SetMultiIdxOffset(const MultiIdxOffsets multi_idx_offset)
    {
        multi_idx_offset_ = multi_idx_offset;
        base_offset_ += layout_(multi_idx_offset);
    }

    private:
    // Disable from doxygen docs generation
    /// @cond INTERNAL
    using DynamicBufferType = DynamicBuffer<BufferAddressSpace,
                                            ElementType,
                                            ElementSpaceSize,
                                            true /*InvalidElementUseNumericalZeroValue*/>;
    using StaticBufferType  = std::conditional_t<
         is_scalar_type<ElementType>::value,
         StaticBuffer<BufferAddressSpace,
                      ElementType,
                      size(Shape{}),
                      true /*InvalidElementUseNumericalZeroValue*/>,
         StaticBufferTupleOfVector<BufferAddressSpace,
                                   TensorElementType,
                                   size(Shape{}) /
                                       scalar_type<std::remove_const_t<ElementType>>::vector_size,
                                   scalar_type<std::remove_const_t<ElementType>>::vector_size,
                                   true /*InvalidElementUseNumericalZeroValue*/>>;
    // If register use static buffer, else use dynamic buffer
    using Buffer = std::conditional_t<IsDynamicBuffer, DynamicBufferType, StaticBufferType>;

    const Layout<Shape, UnrolledDescriptorType> layout_;
    Buffer buffer_;
    // We use multi_idx_offset_ to enable the creation of a descriptor in
    // compile time for partitions or tiles if tile shape and thread layout
    // is known at compile time (We can use the same descriptor for each
    // thread). Additionally, the copy between the static and dynamic buffer
    // requires a descriptor known at compile time, so we can shift data using
    // such multi_idx_offset_.
    MultiIndex<Shape::Size()> multi_idx_offset_;
    // Base offset and multi index offset are corresponding to exactly the
    // same element in tensor ( and in physical memory ). Multi index offset
    // is multi dimensional index. However base offset is calculated using
    // tensor descriptor (thus all it's transforms) and is linear (1D).
    // We store base_offset_ to avoid multiple recalculations.
    index_t base_offset_;
    /// @endcond
};

} // namespace wrapper
} // namespace ck
