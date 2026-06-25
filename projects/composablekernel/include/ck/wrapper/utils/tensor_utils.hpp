// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

#include "ck/utility/data_type.hpp"
#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/utility/dynamic_buffer.hpp"
#include "ck/utility/amd_address_space.hpp"
#include "ck/utility/multi_index.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

/**
 * \brief Memory type, allowed members:
 * - Generic,
 * - Global,
 * - Lds,
 * - Sgpr,
 * - Vgpr,
 */
using MemoryTypeEnum = AddressSpaceEnum;

// Disable from doxygen docs generation
/// @cond INTERNAL
// forward declarations
template <typename Shape, typename UnrolledDescriptorType>
struct Layout;
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
struct Tensor;

template <typename FromType, typename ToType>
struct Slice
{
    __host__ __device__ constexpr Slice() : from_(), to_() {}
    __host__ __device__ constexpr Slice(FromType from, ToType to) : from_(from), to_(to) {}

    /**
     * \brief Calculate slice range.
     *
     * \param dim Dimension size.
     * \return Slice range.
     */
    template <typename T>
    __host__ __device__ constexpr auto range(const T& dim) const
    {
        if constexpr(is_same_v<FromType, index_t> || is_same_v<ToType, index_t> ||
                     is_same_v<std::remove_const_t<T>, index_t>)
        {
            if(to_ < 0)
            {
                return dim - from_ + to_ + 1;
            }
            else
            {
                // workaround if one end of the interval is index_t and the second one is Number
                return static_cast<index_t>(to_) - static_cast<index_t>(from_);
            }
        }
        else
        {
            static_assert(T{} >= ToType{} && FromType{} >= Number<0>{} &&
                              (ToType{} < 0 || ToType{} > FromType{}),
                          "Invalid range");
            if constexpr(ToType{} < 0)
            {
                return dim - from_ + to_ + Number<1>{};
            }
            else
            {
                return to_ - from_;
            }
        }
    }

    __host__ __device__ static constexpr bool IsSlice() { return true; }

    const FromType from_;
    const ToType to_;
};

template <typename T>
using is_slice = decltype(std::declval<T&>().IsSlice());

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());
/// @endcond

/**
 * \brief Make tensor function.
 *
 * \tparam MemoryType Type of memory.
 * \param pointer Pointer to the memory.
 * \param layout Tensor layout.
 * \return Constructed tensor.
 */
template <MemoryTypeEnum MemoryType,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
constexpr auto make_tensor(ElementType* pointer,
                           const Layout<Shape, UnrolledDescriptorType>& layout)
{
    return Tensor<MemoryType, ElementType, Shape, UnrolledDescriptorType>(pointer, layout);
}

/**
 * \brief Make SGPR or VGPR tensor function.
 *
 * \tparam MemoryType Type of memory.
 * \tparam ElementType Memory data type.
 * \return Constructed tensor.
 */
template <MemoryTypeEnum MemoryType,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
constexpr auto make_register_tensor(const Layout<Shape, UnrolledDescriptorType>& layout)
{
    return Tensor<MemoryType, ElementType, Shape, UnrolledDescriptorType>(layout);
}

/**
 * \brief Clear tensor. (Only for Vpgr/Sgpr)
 *
 * \param tensor Tensor to be cleared.
 */
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
__host__ __device__ void
clear(Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>& tensor)
{
    static_assert(
        !Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>::IsDynamicBuffer);
    return tensor.GetBuffer().Clear();
}

/**
 * \brief Get Tensor Layout.
 *
 * \param tensor Tensor to get layout of.
 * \return Requsted layout.
 */
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
__host__ __device__ constexpr const auto&
layout(const Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>& tensor)
{
    return tensor.GetLayout();
}

/**
 * \brief Product of tensor shape dims.
 *
 * \tparam Idxs Indexes to access specific shape dim (optional).
 * \param tensor Tensor to get Shape of.
 * \return Requsted size.
 */
template <index_t... Idxs,
          MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
__host__ __device__ constexpr auto
size(const Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>& tensor)
{
    return size<Idxs...>(tensor.GetLayout());
}

/**
 * \brief Rank of Shape tuple.
 *
 * \tparam Idxs Indexes to access specific shape dim (optional).
 * \param tensor Tensor to get rank of.
 * \return Requsted rank.
 */
template <index_t... Idxs,
          MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
__host__ __device__ constexpr auto
rank(const Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>& tensor)
{
    return rank<Idxs...>(tensor.GetLayout());
}

/**
 * \brief Depth of Shape tuple.
 *
 * \tparam Idxs Indexes to access specific shape dim (optional).
 * \param tensor Tensor to get depth of.
 * \return Requsted depth.
 */
template <index_t... Idxs,
          MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
__host__ __device__ constexpr auto
depth(const Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>& tensor)
{
    return depth<Idxs...>(tensor.GetLayout());
}

/**
 * \brief Get Tensor shape.
 *
 * \param tensor Tensor to get shape from.
 * \return Requsted shape.
 */
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnrolledDescriptorType>
__host__ __device__ constexpr const auto&
shape(const Tensor<BufferAddressSpace, ElementType, Shape, UnrolledDescriptorType>& tensor)
{
    return shape(tensor.GetLayout());
}

/**
 * \brief Get dim slice.
 *
 * \param from Beginning of the interval.
 * \param to End of the interval. (could be also negative to index from the end)
 * \return Requested slice. Could be used to create sliced tensor from other tensor.
 */
template <typename FromType, typename ToType>
constexpr auto slice(const FromType from, const ToType to)
{
    return Slice<FromType, ToType>(from, to);
}

/**
 * \brief Get dim slice. (Assumed that from is equal to 1)
 *
 * \param to End of the interval. (could be also negative to index from the end)
 * \return Requested slice. Could be used to create sliced tensor from other tensor.
 */
template <typename ToType>
constexpr auto slice(const ToType to)
{
    if constexpr(is_same_v<ToType, index_t>)
    {
        return Slice<index_t, ToType>(0, to);
    }
    else
    {
        return Slice<Number<0>, ToType>(Number<0>{}, to);
    }
}

/**
 * \brief Get whole dim slice (from = 0, to = -1).
 *
 * \return Requested slice. Could be used to create sliced tensor from other tensor.
 */
constexpr auto slice() { return Slice<Number<0>, Number<-1>>(Number<0>{}, Number<-1>{}); }

} // namespace wrapper
} // namespace ck
