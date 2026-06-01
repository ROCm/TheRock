// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"

namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array;
    using data_type = TData;

    TData mData[NSize];

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    __host__ __device__ constexpr const TData& operator[](index_t i) const { return At(i); }

    __host__ __device__ constexpr TData& operator()(index_t i) { return At(i); }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }
    __host__ __device__ constexpr const TData* begin() const { return &mData[0]; }
    __host__ __device__ constexpr const TData* end() const { return &mData[NSize]; }
    __host__ __device__ constexpr TData* begin() { return &mData[0]; }
    __host__ __device__ constexpr TData* end() { return &mData[NSize]; }
};

// empty Array
template <typename TData>
struct Array<TData, 0>
{
    using type      = Array;
    using data_type = TData;

    __host__ __device__ static constexpr index_t Size() { return 0; }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(X&& x, Xs&&... xs)
{
    using data_type = remove_cvref_t<X>;
    return Array<data_type, sizeof...(Xs) + 1>{ck::forward<X>(x), ck::forward<Xs>(xs)...};
}

// make empty array
template <typename X>
__host__ __device__ constexpr auto make_array()
{
    return Array<X, 0>{};
}

} // namespace ck
#endif
