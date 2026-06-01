// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <stdint.h>
#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/e8m0.hpp"
#include "ck/utility/statically_indexed_array.hpp"

/// Definitions from <cstdint>, <cmath> conflict with
/// /opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h.

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
#define CHAR_BIT 8
using int8_t   = signed char;
using uint8_t  = unsigned char;
using int16_t  = signed short;
using uint16_t = unsigned short;
using float_t  = float;
#endif // __HIPCC_RTC__

namespace ck {
#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
using byte = unsigned char;
#else
using std::byte;
#endif

using tf32_t  = _BitInt(19); // 1 sign bit, 8 exponent bits, 10 mantissa bits
using bhalf_t = ushort;
using half_t  = _Float16;
using int4_t  = _BitInt(4);
using f4_t    = unsigned _BitInt(4);
using f6_t    = _BitInt(6);          // e2m3 format
using bf6_t   = unsigned _BitInt(6); // e3m2 format

// scalar_type
template <typename TV>
struct scalar_type;

struct f4x2_pk_t
{
    static constexpr int packed_size = 2;

    using type = uint8_t;
    type data;
    __host__ __device__ constexpr f4x2_pk_t() : data{type{}} {}
    __host__ __device__ constexpr f4x2_pk_t(const type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline type unpack(Number<I>) const
    {
        static_assert(I < 2, "Index is out of range.");
        if constexpr(I == 1)
            return (data >> 4);
        else
            return data & 0b00001111;
    }

    __host__ __device__ inline type pack(const type x0, const type x1)
    {
        return (x1 << 4) | (x0 & 0b00001111);
    }

    // Compare operator
    __host__ __device__ friend bool operator==(const f4x2_pk_t& lhs, const f4x2_pk_t& rhs)
    {
        return lhs.data == rhs.data;
    }

    __host__ __device__ friend bool operator!=(const f4x2_pk_t& lhs, const f4x2_pk_t& rhs)
    {
        return !(lhs == rhs);
    }
};

template <typename BitType, index_t pk_size>
struct f6_pk_t
{
    using element_type = uint32_t; // element storage fundamental type

    static constexpr index_t packed_size   = pk_size; // 16 or 32 for now
    static constexpr index_t num_bits_elem = 6;       // specialized for 6-bit data
    // XXX: CHAR_BIT is not defined in HIPRTC, so we must use 8
    static constexpr index_t num_bits_vec_elem =
        sizeof(element_type) * 8; // 32-bit uint for storage
    static_assert((packed_size * num_bits_elem) % num_bits_vec_elem == 0,
                  "Packed elements must fit exactly into the element storage.");
    static constexpr index_t vector_size =
        (packed_size * num_bits_elem) / num_bits_vec_elem; // 3 or 6 element_type units

    using storage_type = element_type __attribute__((ext_vector_type(vector_size)));
    storage_type data_{storage_type(0)}; // packed data

    using type = f6_pk_t<BitType, packed_size>;

    __host__ __device__ constexpr f6_pk_t() {}
    __host__ __device__ constexpr f6_pk_t(const storage_type& init) : data_{init}
    {
        // TODO: consider removing initialization similar to vector_type<T, 256>
    }

    // Initialize from a vector type with the same size as packed_size
    template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == packed_size>>
    __host__ __device__ f6_pk_t(const T& v)
    {
        static_for<0, packed_size, 1>{}(
            [&](auto i) { pack(v[static_cast<index_t>(i)], static_cast<index_t>(i)); });
    }

    // Broadcast single initialization value to all packed elements
    __host__ __device__ f6_pk_t(const int8_t v)
        : f6_pk_t(static_cast<int8_t __attribute__((ext_vector_type(packed_size)))>(v))
    {
        // TODO: consider removing initialization similar to vector_type<T, 256>
    }

    template <typename T>
    __host__ __device__ void pack(const T x, const index_t i)
    {
        static_assert(is_integral<T>::value || is_same_v<T, BitType>,
                      "T must be an integral type.");

        uint32_t bits        = static_cast<uint32_t>(x) & 0x3F;
        const int bit_pos    = i * num_bits_elem;
        const int arr_index  = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;
        uint32_t old_value   = data_[arr_index];

        // insert bits into the current 32-bit block
        old_value |= (bits << bit_offset);
        data_[arr_index] = old_value;

        // if it crosses into the next block, shift the remainder
        if(overhang > 0 && (arr_index + 1) < vector_size)
        {
            uint32_t next_value = data_[arr_index + 1];
            next_value |= (bits >> (num_bits_elem - overhang));
            data_[arr_index + 1] = next_value;
        }
    }

    __host__ __device__ static inline BitType unpack(const type& pk, const index_t i)
    {
        const int bit_pos    = i * num_bits_elem;
        const int arr_idx    = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;

        uint32_t bits = pk.data_[arr_idx] >> bit_offset;
        if(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (pk.data_[arr_idx + 1] & ((1u << overhang) - 1)) << (num_bits_elem - overhang);
        }

        return static_cast<BitType>(bits & 0x3F);
    }

    __host__ __device__ inline BitType unpack(const index_t i) const { return unpack(*this, i); }

    // Compare operator
    __host__ __device__ friend bool operator==(const f6_pk_t& lhs, const f6_pk_t& rhs)
    {
#pragma unroll
        for(index_t i = 0; i < vector_size; ++i)
        {
            if(lhs.data_[i] != rhs.data_[i])
                return false;
        }
        return true;
    }

    __host__ __device__ friend bool operator!=(const f6_pk_t& lhs, const f6_pk_t& rhs)
    {
        return !(lhs == rhs);
    }
};

using f6x16_pk_t  = f6_pk_t<f6_t, 16>;
using f6x32_pk_t  = f6_pk_t<f6_t, 32>;
using bf6x16_pk_t = f6_pk_t<bf6_t, 16>;
using bf6x32_pk_t = f6_pk_t<bf6_t, 32>;

// custom data type - pack int4 data
struct pk_i4_t
{
    using type = int8_t;
    type data;
    __host__ __device__ constexpr pk_i4_t() : data{type{}} {}
    __host__ __device__ constexpr pk_i4_t(type init) : data{init} {}
};

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}

// native types: double, float, _Float16, ushort, int32_t, int8_t, uint8_t, f8_fnuz_t, bf8_fnuz_t,
// native types: bool
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value ||
           is_same<T, uint32_t>::value || is_same<T, int8_t>::value || is_same<T, uint8_t>::value ||
           is_same_v<T, _BitInt(8)> || is_same_v<T, unsigned _BitInt(8)> || is_same<T, bool>::value;
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint8_t>
{
    using type                           = uint8_t;
    static constexpr index_t vector_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<pk_i4_t>
{
    using type                           = pk_i4_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_fnuz_t>
{
    using type                           = f8_fnuz_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_fnuz_t>
{
    using type                           = bf8_fnuz_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_ocp_t>
{
    using type                           = f8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_ocp_t>
{
    using type                           = bf8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

#ifndef CK_CODE_GEN_RTC
template <>
struct scalar_type<e8m0_bexp_t>
{
    using type                           = e8m0_bexp_t::type;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<f4x2_pk_t>
{
    using type                           = f4x2_pk_t::type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f6x32_pk_t>
{
    using type                           = f6x32_pk_t::storage_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf6x32_pk_t>
{
    using type                           = bf6x32_pk_t::storage_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f6x16_pk_t>
{
    using type                           = f6x16_pk_t::storage_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf6x16_pk_t>
{
    using type                           = bf6x16_pk_t::storage_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

template <typename T>
struct packed_type_info
{
    private:
    static constexpr auto get_packed_type_info()
    {
        using U = remove_cvref_t<T>;
        if constexpr(is_same_v<U, pk_i4_t>)
            return ck::Tuple<ck::Number<2>, int4_t>{};
        else if constexpr(is_same_v<U, f4x2_pk_t>)
            return ck::Tuple<ck::Number<2>, f4_t>{};
        else if constexpr(is_same_v<U, f6x16_pk_t>)
            return ck::Tuple<ck::Number<16>, f6_t>{};
        else if constexpr(is_same_v<U, bf6x16_pk_t>)
            return ck::Tuple<ck::Number<16>, bf6_t>{};
        else if constexpr(is_same_v<U, f6x32_pk_t>)
            return ck::Tuple<ck::Number<32>, f6_t>{};
        else if constexpr(is_same_v<U, bf6x32_pk_t>)
            return ck::Tuple<ck::Number<32>, bf6_t>{};
        else
            return ck::Tuple<ck::Number<1>, T>{};
    }

    public:
    using element_type = remove_cvref_t<decltype(get_packed_type_info().At(ck::Number<1>{}))>;
    static constexpr auto packed_size =
        static_cast<index_t>(get_packed_type_info().At(ck::Number<0>{}));
};
template <typename T>
using element_type_t = typename packed_type_info<T>::element_type;

template <typename T>
inline constexpr index_t packed_size_v = packed_type_info<T>::packed_size;

template <typename T>
inline constexpr bool is_packed_type_v = packed_size_v<T> > 1;

template <typename T, index_t N = 0>
struct packed_type_maker
{
    private:
    static constexpr auto get_packed_type()
    {
        using U = remove_cvref_t<T>;
        if constexpr(is_same_v<U, int4_t>)
        {
            static_assert(N == 0 || N == 2, "Packed size N for int4_t must be 2.");
            return pk_i4_t{};
        }
        else if constexpr(is_same_v<U, f4_t>)
        {
            static_assert(N == 0 || N == 2, "Packed size N for f4_t must be 2.");
            return f4x2_pk_t{};
        }
        else if constexpr(is_same_v<U, f6_t>)
        {
            static_assert(N == 0 || N == 16 || N == 32, "Packed size N for f6_t must be 16 or 32.");
            if constexpr(N == 16)
                return f6x16_pk_t{};
            else if constexpr(N == 0 || N == 32)
                return f6x32_pk_t{};
        }
        else if constexpr(is_same_v<U, bf6_t>)
        {
            static_assert(N == 0 || N == 16 || N == 32,
                          "Packed size N for bf6_t must be 16 or 32.");
            if constexpr(N == 16)
                return bf6x16_pk_t{};
            else if constexpr(N == 0 || N == 32)
                return bf6x32_pk_t{};
        }
        else
            return T{};
    }

    public:
    using packed_type = remove_cvref_t<decltype(get_packed_type())>;
};

template <typename T, index_t N = 0>
using packed_type_t = typename packed_type_maker<T, N>::packed_type;

#if defined(_WIN32)
using int64_t = long long;
#else
using int64_t = long;
#endif

template <typename T>
inline const char* get_type_name()
{
    if constexpr(is_same_v<T, half_t>)
        return "fp16";
    else if constexpr(is_same_v<T, bhalf_t>)
        return "bf16";
    else if constexpr(is_same_v<T, tf32_t>)
        return "tf32";
    else if constexpr(is_same_v<T, int4_t>)
        return "int4";
    else if constexpr(is_same_v<T, f4_t>)
        return "f4";
    else if constexpr(is_same_v<T, f6_t>)
        return "f6";
    else if constexpr(is_same_v<T, bf6_t>)
        return "bf6";
    else if constexpr(is_same_v<T, f8_t>)
        return "f8";
    else if constexpr(is_same_v<T, bf8_t>)
        return "bf8";
#ifndef CK_CODE_GEN_RTC
    else if constexpr(is_same_v<T, e8m0_bexp_t>)
        return "e8m0";
#endif
    else if constexpr(is_same_v<T, float>)
        return "fp32";
#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
    else
        return "unknown";
#else
    else
        return typeid(T).name();
#endif
}

} // namespace ck
