// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/utility/data_type.hpp"

namespace ck {

// vector_type
template <typename T, index_t N, typename Enable = void>
struct vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

template <typename T>
struct vector_type<T, 1, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    using type = d1_t;

    union
    {
        T d1_;
        StaticallyIndexedArray<T, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }
};

__device__ int static err = 0;
template <typename T>
struct vector_type<T, 2, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 3, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));

    using type = d3_t;

    union
    {
        d3_t d3_;
        StaticallyIndexedArray<d1_t, 3> d1x3_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
        StaticallyIndexedArray<d3_t, 1> d3x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 5, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d5_t __attribute__((ext_vector_type(5)));

    using type = d5_t;

    union
    {
        d5_t d5_;
        StaticallyIndexedArray<d1_t, 5> d1x5_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d5_t, 1> d5x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 6, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d6_t __attribute__((ext_vector_type(6)));

    using type = d6_t;

    union
    {
        d6_t d6_;
        StaticallyIndexedArray<d1_t, 6> d1x6_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d3_t, 2> d3x2_;
        StaticallyIndexedArray<d6_t, 1> d6x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 7, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d7_t __attribute__((ext_vector_type(7)));

    using type = d7_t;

    union
    {
        d7_t d7_;
        StaticallyIndexedArray<d1_t, 7> d1x7_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d7_t, 1> d7x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 13, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d13_t __attribute__((ext_vector_type(13)));

    using type = d13_t;

    union
    {
        d13_t d13_;
        StaticallyIndexedArray<d1_t, 13> d1x13_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
        StaticallyIndexedArray<d13_t, 1> d13x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_ = {d32_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    // __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    // __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 128, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_ = {d128_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 256, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_ = {d256_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N, typename Enable = void>
struct non_native_vector_base;

template <typename T>
struct nnvb_data_t_selector
{
    using type = unsigned _BitInt(8 * sizeof(T));
};

template <>
struct nnvb_data_t_selector<f8_ocp_t>
{
    using type = f8_ocp_t::data_type;
};

template <>
struct nnvb_data_t_selector<bf8_ocp_t>
{
    using type = bf8_ocp_t::data_type;
};

#ifndef CK_CODE_GEN_RTC
template <>
struct nnvb_data_t_selector<f8_fnuz_t>
{
    using type = f8_fnuz_t::data_type;
};

template <>
struct nnvb_data_t_selector<bf8_fnuz_t>
{
    using type = bf8_fnuz_t::data_type;
};

template <>
struct nnvb_data_t_selector<e8m0_bexp_t>
{
    using type = e8m0_bexp_t::type;
};
#endif

template <>
struct nnvb_data_t_selector<f6x16_pk_t>
{
    using type = f6x16_pk_t::storage_type;
};

template <>
struct nnvb_data_t_selector<f6x32_pk_t>
{
    using type = f6x32_pk_t::storage_type;
};

template <>
struct nnvb_data_t_selector<bf6x16_pk_t>
{
    using type = bf6x16_pk_t::storage_type;
};

template <>
struct nnvb_data_t_selector<bf6x32_pk_t>
{
    using type = bf6x32_pk_t::storage_type;
};

template <>
struct nnvb_data_t_selector<pk_i4_t>
{
    using type = pk_i4_t::type;
};

template <>
struct nnvb_data_t_selector<f4x2_pk_t>
{
    using type = f4x2_pk_t::type;
};

template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>
{
    using data_t = typename nnvb_data_t_selector<T>::type; // select data_t based on the size of T
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    using data_v = data_t __attribute__((ext_vector_type(N)));
    using type   = non_native_vector_base<T, N>;

    union alignas(next_pow2(N * sizeof(T)))
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray<data_t, N> dxN;
        StaticallyIndexedArray<T, N> dTxN;
        StaticallyIndexedArray<data_v, 1> dNx1;
    } data_;

    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{data_v(a)} {}
    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }
    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};
    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }
    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }
    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return data_.dTxN; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }
};

// implementation for f6x16 and f6x32
template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 16 || sizeof(T) == 24 || sizeof(T) == 32>>
{
    using data_t =
        typename nnvb_data_t_selector<T>::type; // select data_t based on declared base type
    using element_t = typename T::element_type; // select element_t based on declared element type
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    static constexpr size_t size_factor = sizeof(data_t) / sizeof(element_t);
    using data_v = element_t __attribute__((ext_vector_type(N * size_factor)));
    using type   = non_native_vector_base<T, N>;

    union alignas(next_pow2(N * sizeof(T)))
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray<data_t, N> dxN;
        StaticallyIndexedArray<T, N> dTxN;
        StaticallyIndexedArray<data_v, 1> dNx1;
    } data_;

    // Broadcast single value to vector
    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{}
    {
        // TODO: consider removing initialization similar to vector_type<T, 256>

        ck::static_for<0, N, 1>{}([&](auto i) {
            data_.dxN(i) = a; // broadcast value to all elements
        });
    }

    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }

    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};

    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr non_native_vector_base(element_t v) : data_{data_v(v)} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }

    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return err; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, data_v> || is_same_v<X, T>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>>
{
    using type                           = typename non_native_vector_base<T, N>::data_t;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 16 || sizeof(T) == 24 || sizeof(T) == 32>>>
{
    using type                           = typename non_native_vector_base<T, N>::element_t;
    static constexpr index_t vector_size = N * non_native_vector_base<T, N>::size_factor;
};

// non-native vector_type implementation
template <typename T>
struct vector_type<T, 1, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using type     = d1_nnv_t;

    union alignas(next_pow2(1 * sizeof(T)))
    {
        d1_t d1_;
        StaticallyIndexedArray<d1_t, 1> d1x1_;
        d1_nnv_t d1_nnv_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{d1_t{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 2, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;

    using type = d2_t;

    union alignas(next_pow2(2 * sizeof(T)))
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;

    using type = d4_t;

    union alignas(next_pow2(4 * sizeof(T)))
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;
    using d8_t     = non_native_vector_base<T, 8>;

    using type = d8_t;

    union alignas(next_pow2(8 * sizeof(T)))
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;
    using d8_t     = non_native_vector_base<T, 8>;
    using d16_t    = non_native_vector_base<T, 16>;

    using type = d16_t;

    union alignas(next_pow2(16 * sizeof(T)))
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;

    using type = d32_t;

    union alignas(next_pow2(32 * sizeof(T)))
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;
    using d64_t = non_native_vector_base<T, 64>;

    using type = d64_t;

    union alignas(next_pow2(64 * sizeof(T)))
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x6_t  = typename vector_type<int32_t, 6>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// f8
using f8x2_fnuz_t  = typename vector_type<f8_fnuz_t, 2>::type;
using f8x4_fnuz_t  = typename vector_type<f8_fnuz_t, 4>::type;
using f8x8_fnuz_t  = typename vector_type<f8_fnuz_t, 8>::type;
using f8x16_fnuz_t = typename vector_type<f8_fnuz_t, 16>::type;
using f8x32_fnuz_t = typename vector_type<f8_fnuz_t, 32>::type;
using f8x64_fnuz_t = typename vector_type<f8_fnuz_t, 64>::type;

// bf8
using bf8x2_fnuz_t  = typename vector_type<bf8_fnuz_t, 2>::type;
using bf8x4_fnuz_t  = typename vector_type<bf8_fnuz_t, 4>::type;
using bf8x8_fnuz_t  = typename vector_type<bf8_fnuz_t, 8>::type;
using bf8x16_fnuz_t = typename vector_type<bf8_fnuz_t, 16>::type;
using bf8x32_fnuz_t = typename vector_type<bf8_fnuz_t, 32>::type;
using bf8x64_fnuz_t = typename vector_type<bf8_fnuz_t, 64>::type;

// f8
using f8x2_ocp_t  = typename vector_type<f8_ocp_t, 2>::type;
using f8x4_ocp_t  = typename vector_type<f8_ocp_t, 4>::type;
using f8x8_ocp_t  = typename vector_type<f8_ocp_t, 8>::type;
using f8x16_ocp_t = typename vector_type<f8_ocp_t, 16>::type;
using f8x32_ocp_t = typename vector_type<f8_ocp_t, 32>::type;
using f8x64_ocp_t = typename vector_type<f8_ocp_t, 64>::type;

// bf8
using bf8x2_ocp_t  = typename vector_type<bf8_ocp_t, 2>::type;
using bf8x4_ocp_t  = typename vector_type<bf8_ocp_t, 4>::type;
using bf8x8_ocp_t  = typename vector_type<bf8_ocp_t, 8>::type;
using bf8x16_ocp_t = typename vector_type<bf8_ocp_t, 16>::type;
using bf8x32_ocp_t = typename vector_type<bf8_ocp_t, 32>::type;
using bf8x64_ocp_t = typename vector_type<bf8_ocp_t, 64>::type;

#if CK_FP8_TYPE_OCP
// f8
using f8x2_t  = f8x2_ocp_t;
using f8x4_t  = f8x4_ocp_t;
using f8x8_t  = f8x8_ocp_t;
using f8x16_t = f8x16_ocp_t;
using f8x32_t = f8x32_ocp_t;
using f8x64_t = f8x64_ocp_t;

// bf8
using bf8x2_t  = bf8x2_ocp_t;
using bf8x4_t  = bf8x4_ocp_t;
using bf8x8_t  = bf8x8_ocp_t;
using bf8x16_t = bf8x16_ocp_t;
using bf8x32_t = bf8x32_ocp_t;
using bf8x64_t = bf8x64_ocp_t;
#elif CK_FP8_TYPE_FNUZ
// f8
using f8x2_t  = f8x2_fnuz_t;
using f8x4_t  = f8x4_fnuz_t;
using f8x8_t  = f8x8_fnuz_t;
using f8x16_t = f8x16_fnuz_t;
using f8x32_t = f8x32_fnuz_t;
using f8x64_t = f8x64_fnuz_t;

// bf8
using bf8x2_t  = bf8x2_fnuz_t;
using bf8x4_t  = bf8x4_fnuz_t;
using bf8x8_t  = bf8x8_fnuz_t;
using bf8x16_t = bf8x16_fnuz_t;
using bf8x32_t = bf8x32_fnuz_t;
using bf8x64_t = bf8x64_fnuz_t;
#endif

// u8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;
using uint8x32_t = typename vector_type<uint8_t, 32>::type;
using uint8x64_t = typename vector_type<uint8_t, 64>::type;

// f4
using f4x2_t  = typename vector_type<f4x2_pk_t, 1>::type;
using f4x4_t  = typename vector_type<f4x2_pk_t, 2>::type;
using f4x8_t  = typename vector_type<f4x2_pk_t, 4>::type;
using f4x16_t = typename vector_type<f4x2_pk_t, 8>::type;
using f4x32_t = typename vector_type<f4x2_pk_t, 16>::type;
using f4x64_t = typename vector_type<f4x2_pk_t, 32>::type;

// f6
using f6x16_t   = typename vector_type<f6x16_pk_t, 1>::type;
using f6x16x2_t = typename vector_type<f6x16_pk_t, 2>::type;
using f6x32_t   = typename vector_type<f6x32_pk_t, 1>::type;

// bf6
using bf6x16_t   = typename vector_type<bf6x16_pk_t, 1>::type;
using bf6x16x2_t = typename vector_type<bf6x16_pk_t, 2>::type;
using bf6x32_t   = typename vector_type<bf6x32_pk_t, 1>::type;

#ifndef CK_CODE_GEN_RTC
// e8m0
using e8m0x4_bexp_t = typename vector_type<e8m0_bexp_t, 4>::type;
#endif

// pack int4
using pk_i4x2_t = typename vector_type<pk_i4_t, 2>::type;
using pk_i4x4_t = typename vector_type<pk_i4_t, 4>::type;
using pk_i4x8_t = typename vector_type<pk_i4_t, 8>::type;

} // namespace ck
