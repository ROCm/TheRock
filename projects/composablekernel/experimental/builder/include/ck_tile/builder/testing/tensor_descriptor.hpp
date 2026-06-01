// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <array>
#include <vector>
#include <sstream>
#include <iosfwd>
#include <concepts>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/host/host_tensor.hpp"

/// This file deals with tensor memory layout. The `TensorDescriptor` is the
/// main item, which is a type that describes (but not manages!) the layout
/// of tensor memory. There are also some related utilities.

namespace ck_tile::builder::test {

/// @brief Tensor dimensions type
///
/// An Extent describes size in tensor space, usually either the tensor lengths
/// (conceptual size) or the tensor strides (memory layout). This type is mainly
/// used by the `TensorDescriptor`. This type is based on `std::array<size_t, RANK>`
/// and supports all relevant operations on that.
///
/// @note In practical terms, this type is not just an alias of `std::array` for
/// two reasons: First, writing a separate type allows us to write a custom
/// CTAD deduction guideline. This allows users to write `Extent{1, 2, 3}` and
/// get an instance of the correct type, whereas `std::array{1, 2, 3}` yields an
/// instance of `std::array<int, 3>`. This, in turn, allows inferring the rank
/// from the instance (useful in combination with `make_descriptor`), as it alows
/// us to write `function(Extent{1, 2, 3})`. Note that `function({1, 2, 3})` is
/// not valid before C++26 because `{1, 2, 3}` is an initializer list (even if
/// `function` accepts an instance of `Extent`), which does not have a known size
/// at compile time. Second, creating a separate struct for the `Extent` allows
/// additional (static) member functions.
///
/// @tparam RANK The rank (number of spatial dimensions) of the tensor that this
/// extent describes a size of.
///
/// @see TensorDescriptor
/// @see make_descriptor
template <size_t RANK>
struct Extent : std::array<size_t, RANK>
{
    using Base = std::array<size_t, RANK>;
    // Note: Default constructor inherited from std::array.

    /// @brief Construct an extent from an `std::vector`.
    ///
    /// This function can be used to turn an `std::vector` into an `Extent`.
    /// Because this code is mainly intended for testing, the vector's size is
    /// checked. If its not equal to `RANK`, an exception is thrown.
    ///
    /// @throws std::runtime_error if the size of `extent` is not equal to `RANK`.
    static Extent from_vector(const std::vector<size_t>& extent)
    {
        if(extent.size() != RANK)
        {
            std::stringstream msg;
            msg << "invalid rank! expected: " << RANK << ", got: " << extent.size();
            throw std::runtime_error(msg.str());
        }

        Extent result;
        std::copy_n(extent.begin(), RANK, result.begin());
        return result;
    }

    // Note: std::array doesn't like generating indexing code when the RANK
    // is zero. Looks like there is a missing __device__ overload in ROCm 7.1
    // at least. Its not terribly important, but just override the default
    // operator[] to fix it.

    /// @brief Array indexing operator
    ///
    /// `std::array` has issues with this operator when RANK=0, this version
    /// fixes that.
    ///
    /// @param i The index to index the array with.
    ///
    /// @see std::array::operator[]
    __device__ __host__ size_t operator[](size_t i) const
    {
        if constexpr(RANK > 0)
        {
            return Base::operator[](i);
        }
        else
        {
            __builtin_unreachable();
        }
    }

    /// @brief Array indexing operator
    ///
    /// `std::array` has issues with this operator when RANK=0, this version
    /// fixes that.
    ///
    /// @param i The index to index the array with.
    ///
    /// @see std::array::operator[]
    __device__ __host__ size_t& operator[](size_t i)
    {
        if constexpr(RANK > 0)
        {
            return Base::operator[](i);
        }
        else
        {
            __builtin_unreachable();
        }
    }
};

// This is a deduction guideline necessary to resolve `Extent{1, 2, 3}` to the
// correct type. This definition is practically the same as that of `std::array`.
template <typename... T>
Extent(T...) -> Extent<sizeof...(T)>;

/// @brief Extent printer
///
/// This function implements an ostream printing overload for `Extent`, so that
/// they can be printed in the usual `stream << extent` fashion.
///
/// @tparam RANK Rank (number of spatial dimensions) of the extent.
///
/// @param stream The stream to print the extent to.
/// @param extent The extent to print to the stream.
template <size_t RANK>
std::ostream& operator<<(std::ostream& stream, const Extent<RANK>& extent)
{
    stream << '[';
    bool first = true;
    for(const auto x : extent)
    {
        if(first)
            first = false;
        else
            stream << ", ";

        stream << x;
    }

    return stream << ']';
}

/// @brief Concept for automatically deriving tensor memory layout.
///
/// A `TensorStridesGenerator` is a type which can be used to automatically
/// derive the strides (memory layout) of a tensor, given the tensor lengths.
/// This is mainly used to avoid manually computing strides.
///
/// Implementors of this concept are required to implement `operator()`,
/// which accepts an instance of `Extent<RANK>` (the tensor lengths) and
/// yields another instance of `Extent<RANK>` (the tensor strides). Note
/// that the returned strides are expected to be "pre-scanned", meaning
/// that the offset in memory of a tensor can be computed as
/// `dot(index * strides)` (where `*` is element-wise multiplication).
///
/// @see TensorDescriptor
/// @see PackedRightLayout
/// @see PackedLeftLayout
template <typename G, int RANK>
concept TensorStridesGenerator = requires(const G& generator, const Extent<RANK>& lengths) {
    { generator(lengths) } -> std::convertible_to<Extent<RANK>>;
};

/// @brief Layout generator where right-most dimension has stride 1 and
/// all dimensions are packed.
///
/// This structure implements a `TensorStridesGenerator` which generates
/// a memory layout which has the right-most dimension equal to 1, and
/// all other strides increase right-to-left as a products of the extent.
/// This corresponds with a row-major layout.
///
/// @see TensorStridesGenerator
/// @see TensorDescriptor
struct PackedRightLayout
{
    /// @brief Stride generation implementation.
    ///
    /// This is the main function which implements the stride generation
    ///
    /// @tparam RANK The rank of the tensor.
    ///
    /// @param lengths The lengths of the tensor.
    ///
    /// @returns The tensor's memory layout according to the definition
    /// of `PackedRightLayout`.
    ///
    /// @see TensorStridesGenerator
    template <size_t RANK>
    Extent<RANK> operator()(const Extent<RANK>& lengths) const
    {
        Extent<RANK> strides = {};
        size_t numel         = 1;

        for(size_t i = RANK; i > 0; --i)
        {
            strides[i - 1] = numel;
            numel *= lengths[i - 1];
        }

        return strides;
    }
};
static_assert(TensorStridesGenerator<PackedRightLayout, 3>,
              "PackedRightLayout should be a TensorStridesGenerator!");

/// @brief Layout generator where left-most dimension has stride 1 and
/// all dimensions are packed.
///
/// This structure implements a `TensorStridesGenerator` which generates
/// a memory layout which has the left-most dimension equal to 1, and
/// all other strides increase left-to-right as a products of the extent.
/// This corresponds with a column-major layout.
///
/// @see TensorStridesGenerator
/// @see TensorDescriptor
struct PackedLeftLayout
{
    /// @brief Stride generation implementation.
    ///
    /// This is the main function which implements the stride generation
    ///
    /// @tparam RANK The rank of the tensor.
    ///
    /// @param lengths The lengths of the tensor.
    ///
    /// @returns The tensor's memory layout according to the definition
    /// of `PackedLeftLayout`.
    ///
    /// @see TensorStridesGenerator
    template <size_t RANK>
    Extent<RANK> operator()(const Extent<RANK>& lengths) const
    {
        Extent<RANK> strides = {};
        size_t numel         = 1;

        for(size_t i = 0; i < RANK; ++i)
        {
            strides[i] = numel;
            numel *= lengths[i];
        }

        return strides;
    }
};
static_assert(TensorStridesGenerator<PackedLeftLayout, 3>,
              "PackedLeftLayout should be a TensorStridesGenerator!");

/// @brief Type managing tensor data layout in memory.
///
/// This structure describes a tensor in memory. It does not actually hold any
/// reference to memory, it just describes how the memory should be laid out if it
/// were.
///
/// @note This type is very much like ck_tile::HostTensorDescriptor, except that it
/// also  includes the data type of the elements of htis tensor. This is mainly to
/// make the descriptor a _complete_ description of a tensor rather than just the
/// dimensions in strides, which helps in reducing clutter in uses of this type.
///
/// @note All strides are still in _elements_.
///
/// @tparam DT The conceptual data type of the tensor elements. This need not be the
/// type that the data is actually stored as in memory.
/// @tparam RANK The tensor "rank": the number of conceptial spatial dimensions that
/// the tensor covers.
template <DataType DT, size_t RANK>
struct TensorDescriptor
{
    // For now, the implementation of this type is based on
    // `ck_tile::HostTensorDescriptor`, so that we can prototype without
    // reimplementing the `HostTensorDescriptor` for the 3rd time. You can regard
    // the use of `ck_tile::HostTensorDescriptor` here as an implementation detail.

    /// @brief Tensor extent alias
    ///
    /// This alias represents a std::array which holds tensor dimensions. There is one
    /// item for each dimension in the tensor, and each item corresponds with the
    /// value for that dimension.
    using Extent = ::ck_tile::builder::test::Extent<RANK>;

    /// The conceptual data type of the tensor elements. This need not be the type
    /// that the data is actually stored as in memory.
    constexpr static DataType data_type = DT;

    /// The tensor "rank": the number of conceptial spatial dimensions that the
    /// tensor covers.
    constexpr static size_t rank = RANK;

    /// @brief Create a tensor descriptor from lengths and strides.
    ///
    /// @param lengths A sequence of tensor lengths, the conceptial dimensions of
    /// the tensor in  elements.
    /// @param strides A sequence of in-memory strides of the tensor, measured in
    /// elements. Each element of `strides`` corresponds to one at the same index
    /// in `lengths`, the amount of elements to skip in memory to find the next
    /// element along that axis.
    TensorDescriptor(const Extent& lengths, const Extent& strides)
        : inner_descriptor_(lengths, strides)
    {
        // TODO: Validation of strides? For now we just delegate the details of the
        // construction to the CK Tile HostTensorDescriptor.
    }

    /// @brief Create a tensor descriptor with lengths and automatic layout.
    ///
    /// This function initializes a tensor descriptor using lengths, and by deriving
    /// the memory layout from the layout generator `Generator`. The tensor will be
    /// initialized with the strides yielded from `Generator`.
    ///
    /// @tparam Generator The generator type to generate the strides with. For example,
    /// `PackedRightLayout` or `PackedLeftLayout`.
    ///
    /// @param lengths A sequence of tensor lengths, the conceptial dimensions of
    /// the tensor in  elements.
    /// @param gen An instance of `Generator` to generate the strides with.
    ///
    /// @see TensorStridesGenerator
    /// @see PackedLeftLayout
    /// @see PackedRightLayout
    template <typename Generator>
        requires TensorStridesGenerator<Generator, RANK>
    TensorDescriptor(const Extent& lengths, const Generator& gen)
        : TensorDescriptor(lengths, gen(lengths))
    {
    }

    /// Query the conceptual dimensions of the tensor.
    ///
    /// @returns A span of tensor dimensions, one for every axis. Note that the order
    /// does *not* correspond with memory layout, query the in-memory strides for that.
    ///
    /// @see get_strides()
    Extent get_lengths() const
    {
        // TODO: This is ugly for now. We should ditch the HostTensorDescriptor, and
        // after that this can just be `return lengths_;` (and make it const Extent&).
        Extent result;
        std::copy_n(inner_descriptor_.get_lengths().begin(), RANK, result.begin());
        return result;
    }

    /// Query the in-memory strides of the tensor.
    ///
    /// @returns A span of tensor dimensions, one for every axis. Each element
    /// corresponds directly with the stride in elements at the same index in the
    /// tensor dimensions.
    ///
    /// @see get_lengths()
    Extent get_strides() const
    {
        // TODO: This is ugly for now. We should ditch the HostTensorDescriptor, and
        // after that this can just be `return strides_;` (and make it const Extent&).
        Extent result;
        std::copy_n(inner_descriptor_.get_strides().begin(), RANK, result.begin());
        return result;
    }

    /// @brief Compute conceptual tensor size in elements.
    ///
    /// This function returns the size of the tensor in elements. This function only
    /// takes the lengths into account, not the strides. In order to allocate memory
    /// for the tensor, use `get_element_space_size()`.
    ///
    /// @see get_lengths
    /// @see get_element_space_size
    size_t get_element_size() const { return inner_descriptor_.get_element_size(); }

    /// @brief Compute total tensor space size in elements.
    ///
    /// This function returns the total size of the memory backing a tensor with
    /// this descriptor in *elements*, including required extra size for strides.
    ///
    /// @see get_element_space_size_in_bytes()
    size_t get_element_space_size() const { return inner_descriptor_.get_element_space_size(); }

    /// @brief Compute total tensor size in bytes.
    ///
    /// This function is like `get_element_space_size()`, except that the returned
    /// value is measured in *bytes* rather than *elements*. Use this function for
    /// figuring out how much memory needs to be allocated for a particular tensor.
    ///
    /// @see get_element_space_size()
    size_t get_element_space_size_in_bytes() const
    {
        // For now, the backing type is the naive C++-type that represents the data
        // type. When we are going to support packed types such as i4 and fp6, this
        // is going to become more complicated.
        return get_element_space_size() * data_type_sizeof(DT);
    }

    /// @brief Check if a tensor is packed in memory.
    ///
    /// This function checks whether the tensor memory is "packed", that is, whether
    /// all elements are continuous in memory with no gaps.
    bool is_packed() const
    {
        // First sort by stride, then check if they match the scan of the
        // sizes.
        const auto& lengths = inner_descriptor_.get_lengths();
        const auto& strides = inner_descriptor_.get_strides();

        std::array<size_t, RANK> indices;
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](auto i, auto j) {
            return strides[i] < strides[j];
        });

        size_t x = 1;
        for(size_t i = 0; i < RANK; ++i)
        {
            if(strides[indices[i]] != x)
                return false;

            x *= lengths[indices[i]];
        }

        return true;
    }

    /// @brief Get a tensor descriptor for the space backing a tensor.
    ///
    /// This function returns a tensor descriptor which represents the buffer space
    /// required to a tensor with this descriptor. This is mainly useful to process
    /// buffers with functions which normally operate over tensor descriptors. The
    /// resulting tensor descriptor describes a 1D tensor with the same number of
    /// elements as in the space.
    ///
    /// @see get_element_space_size()
    TensorDescriptor<DT, 1> get_space_descriptor() const
    {
        ck_tile::builder::test::Extent<1> lengths = {this->get_element_space_size()};
        ck_tile::builder::test::Extent<1> strides = {1};
        return TensorDescriptor<DT, 1>(lengths, strides);
    }

    private:
    ck_tile::HostTensorDescriptor inner_descriptor_;
};

/// @brief Tensor descriptor construction helper.
///
/// This function can be used to create a tensor descriptor. It accepts the same
/// parameters as the constructor of `TensorDescriptor`, that is, a sequence of
/// lengths and a sequence of strides (or a generator to generate the strides).
/// The main use of this function is that it allows automatic inference of the `RANK`
/// parameter. C++ constructors do not allow partial specification of type parameters,
/// and so its impossible to write  `TensorDescriptor<DT> x(Extent{1, 2, 3}, ...)`
/// and have the `RANK` be automatically inferred. Functions do allow this though,
/// so this function can be used to write `make_descriptor(Extent{1, 2, 3}, ...)`
///
/// @tparam DT The conceptual data type of the tensor elements. This need not be the
/// type that the data is actually stored as in memory.
/// @tparam RANK The tensor "rank": the number of conceptial spatial dimensions that
/// the tensor covers.
///
/// @param lengths A sequence of tensor lengths, the conceptial dimensions of
/// the tensor in  elements.
/// @param strides A sequence of in-memory strides of the tensor, or a generator
/// to generate those strides from the tensor lengths.
///
/// @see TensorDescriptor
template <DataType DT, size_t RANK>
TensorDescriptor<DT, RANK> make_descriptor(const Extent<RANK>& lengths, const auto& strides)
{
    return TensorDescriptor<DT, RANK>(lengths, strides);
}

/// @brief Allocate automatically managed GPU memory corresponding to a tensor descriptor.
///
/// This function is similar to `alloc_buffer()`, except that the required size is
/// derived automatically from a tensor descriptor. The returned buffer is valid for
/// tensors with that layout. Strides are also taken into account when computing the
/// required size.
///
/// @tparam DT The conceptual datatype of the elements of the tensor.
/// @tparam RANK The conceptual rank (number of dimensions) of the tensor.
///
/// @param descriptor A descriptor of the memory layout of the tensor to allocate.
///
/// @throws OutOfDeviceMemoryError if memory allocation failed.
///
/// @see TensorDescriptor
/// @see DeviceBuffer
/// @see OutOfDeviceMemoryError
/// @see hipMalloc()
template <DataType DT, size_t RANK>
DeviceBuffer alloc_tensor_buffer(const TensorDescriptor<DT, RANK>& descriptor)
{
    return alloc_buffer(descriptor.get_element_space_size_in_bytes());
}

} // namespace ck_tile::builder::test
