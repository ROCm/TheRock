// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

#if defined(__gfx950__)
#define CK_TILE_FP4_CVT_DEVICE 1
#else
#define CK_TILE_FP4_CVT_DEVICE 0
#endif

#define TEST_convert_with_table 0

namespace ck_tile {

using fp32_t   = float;
using fp32x2_t = float __attribute__((ext_vector_type(2)));
using fp16x2_t = _Float16 __attribute__((ext_vector_type(2)));
using bf16x2_t = bfloat16_t __attribute__((ext_vector_type(2)));

// Helpers: constexpr-safe access to elements of ext_vector_type(2)
// Some compilers don't allow operator[] in constant expressions for vector types.
// We use bit_cast to a trivially copyable representation to extract lanes.
namespace detail {
struct fp16x2_repr
{
    _Float16 e[2];
};
struct bf16x2_repr
{
    bfloat16_t e[2];
};
struct fp32x2_repr
{
    float e[2];
};

CK_TILE_HOST_DEVICE constexpr _Float16 lane0(const fp16x2_t& v)
{
    return ck_tile::bit_cast<fp16x2_repr>(v).e[0];
}
CK_TILE_HOST_DEVICE constexpr _Float16 lane1(const fp16x2_t& v)
{
    return ck_tile::bit_cast<fp16x2_repr>(v).e[1];
}

CK_TILE_HOST_DEVICE constexpr bfloat16_t lane0(const bf16x2_t& v)
{
    return ck_tile::bit_cast<bf16x2_repr>(v).e[0];
}
CK_TILE_HOST_DEVICE constexpr bfloat16_t lane1(const bf16x2_t& v)
{
    return ck_tile::bit_cast<bf16x2_repr>(v).e[1];
}

CK_TILE_HOST_DEVICE constexpr float lane0(const fp32x2_t& v)
{
    return ck_tile::bit_cast<fp32x2_repr>(v).e[0];
}
CK_TILE_HOST_DEVICE constexpr float lane1(const fp32x2_t& v)
{
    return ck_tile::bit_cast<fp32x2_repr>(v).e[1];
}
} // namespace detail

struct pk_float4_e2m1_t;
CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t float_to_pk_fp4(const float& x, float scale = 1.f);

// TODO: Add stochastic method
struct pk_float4_e2m1_t
{
    // TODO: Can we merge raw_type and type?
    using raw_type = uint8_t;
    using type     = raw_type;
    type data;

    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t() : data{type{}} {}
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t(T init) : data{static_cast<type>(init)}
    {
    }
    CK_TILE_HOST_DEVICE explicit constexpr pk_float4_e2m1_t(float init, float scale = 1.f)
        : data{float_to_pk_fp4(init, scale)}
    {
    }
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr type& get() { return data; }
    CK_TILE_HOST_DEVICE constexpr type get() const { return data; }

    CK_TILE_HOST_DEVICE constexpr float to_float(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp32x2_t to_fp32x2(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16_t to_fp16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr fp16x2_t to_fp16x2(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16_t to_bf16(float scale = 1.f) const;
    CK_TILE_HOST_DEVICE constexpr bf16x2_t to_bf16x2(float scale = 1.f) const;

    CK_TILE_HOST_DEVICE constexpr operator float() const { return to_float(); }
    CK_TILE_HOST_DEVICE constexpr operator fp32x2_t() const { return to_fp32x2(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16_t() const { return to_fp16(); }
    CK_TILE_HOST_DEVICE constexpr operator fp16x2_t() const { return to_fp16x2(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16_t() const { return to_bf16(); }
    CK_TILE_HOST_DEVICE constexpr operator bf16x2_t() const { return to_bf16x2(); }

    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t unpack(number<I>) const
    {
        return _unpack(number<I>{});
    }
    CK_TILE_HOST_DEVICE constexpr static pk_float4_e2m1_t pack(const pk_float4_e2m1_t& x0,
                                                               const pk_float4_e2m1_t& x1)
    {
        return _pack(x0.get(), x1.get());
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr type _unpack(number<I>) const;
    CK_TILE_HOST_DEVICE constexpr static type _pack(const type x0, const type x1)
    {
        return (x1 << 4) | (x0 & 0b00001111);
    }

#if TEST_convert_with_table
    static constexpr float e2m1_to_fp32_table[16] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6};
    static constexpr fp16_t e2m1_to_fp16_table[16] = {
        bit_cast<fp16_t>(static_cast<uint16_t>(0x0000)), //  0
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3800)), //  0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3C00)), //  1
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3E00)), //  1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4000)), //  2
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4200)), //  3
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4400)), //  4
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4600)), //  6
        bit_cast<fp16_t>(static_cast<uint16_t>(0x8000)), // -0
        bit_cast<fp16_t>(static_cast<uint16_t>(0xB800)), // -0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBC00)), // -1
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBE00)), // -1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC000)), // -2
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC200)), // -3
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC400)), // -4
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC600))  // -6
    };
#endif
};

using pk_fp4_t     = pk_float4_e2m1_t;
using pk_fp4_raw_t = typename pk_fp4_t::type;

template <>
struct numeric_traits<pk_fp4_t>
{
    using bitwise_type = pk_fp4_raw_t;

    static constexpr int exp        = 2;
    static constexpr int mant       = 1;
    static constexpr int bias       = 1;
    static constexpr int PackedSize = 2;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<pk_fp4_t>
{
    static constexpr pk_fp4_raw_t binary_min_normal    = 0b00100010; // 1
    static constexpr pk_fp4_raw_t binary_max_normal    = 0b01110111; // 6
    static constexpr pk_fp4_raw_t binary_lowest_normal = 0b11111111; // -6
    static constexpr pk_fp4_raw_t binary_min_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_max_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_zero          = 0b00000000; // 0
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t min() { return binary_min_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t max() { return binary_max_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t lowest() { return binary_lowest_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t epsilon() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t round_error() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t zero() { return binary_zero; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t denorm_min() { return binary_min_subnorm; }

    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t infinity() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t quiet_NaN() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t signaling_NaN() { return max(); }
};

template <index_t I>
CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t pk_fp4_t::_unpack(number<I>) const
{
    static_assert(I < 2, "Index is out of range.");
    if constexpr(I == 1)
        return (data >> 4);
    else
        return data & 0b00001111;
}
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, pk_fp4_t)
// TODO: consider replace this macro to improve performance

#if CK_TILE_FP4_CVT_DEVICE
namespace impl {

template <typename T>
CK_TILE_DEVICE T _from_f4(pk_fp4_raw_t src, float scale = 1.0f)
{
    if constexpr(std::is_same_v<T, fp32_t>)
    {
        fp32x2_t tmp = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0);
        return detail::lane0(tmp);
    }
    else if constexpr(std::is_same_v<T, fp32x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0);
    else if constexpr(std::is_same_v<T, fp16_t>)
    {
        fp16x2_t tmp = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, 0);
        return detail::lane0(tmp);
    }
    else if constexpr(std::is_same_v<T, fp16x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, 0);
    else if constexpr(std::is_same_v<T, bf16_t>)
    {
        bf16x2_t tmp = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, 0);
        return detail::lane0(tmp);
    }
    else if constexpr(std::is_same_v<T, bf16x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, 0);
    else
        static_assert(std::false_type::value, "Unsupported type.");
    return T{};
}
template <typename T>
CK_TILE_DEVICE pk_fp4_raw_t _to_f4(T src, float scale = 1.0f)
{
    union
    {
        uint32_t u32;
        pk_fp4_raw_t pf4[4];
    } cvt{0};
    if constexpr(std::is_same_v<T, fp32_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(cvt.u32, src, src, scale, 0);
    else if constexpr(std::is_same_v<T, fp32x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            cvt.u32, detail::lane0(src), detail::lane1(src), scale, 0);
    else if constexpr(std::is_same_v<T, fp16_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(cvt.u32, fp16x2_t{src, src}, scale, 0);
    else if constexpr(std::is_same_v<T, fp16x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(cvt.u32, src, scale, 0);
    else if constexpr(std::is_same_v<T, bf16_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, bf16x2_t{src, src}, scale, 0);
    else if constexpr(std::is_same_v<T, bf16x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, src, scale, 0);
    else
        static_assert(std::false_type::value, "Unsupported type.");
    return cvt.pf4[0];
}

} // namespace impl
#endif

CK_TILE_HOST_DEVICE constexpr bf16_t pk_fp4_t::to_bf16(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<bf16_t>(data, scale);
#else
    return bf16_t{type_convert<bf16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale))};
#endif
}

CK_TILE_HOST_DEVICE constexpr bf16x2_t pk_fp4_t::to_bf16x2(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<bf16x2_t>(data, scale);
#else
    return bf16x2_t{type_convert<bf16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale)),
                    type_convert<bf16_t>(convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale))};
#endif
}

// TODO: make it generic so that we can convert from directrly.
CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t float_to_mxfp4(float x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return convert_to_type<pk_fp4_t>(x, scale);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t float_to_pk_fp4(const float& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    auto res = convert_to_type<pk_fp4_t>(x, scale);
    return pk_fp4_t::_pack(res, res);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp16_to_pk_fp4(const fp16_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    auto res = float_to_mxfp4(type_convert<float>(x), scale);
    return pk_fp4_t::_pack(res, res);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t bf16_to_pk_fp4(const bf16_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    auto res = float_to_mxfp4(type_convert<float>(x), scale);
    return pk_fp4_t::_pack(res, res);
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp16x2_to_pk_fp4(const fp16x2_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return pk_fp4_t::_pack(float_to_mxfp4(detail::lane0(x), scale),
                           float_to_mxfp4(detail::lane1(x), scale));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t bf16x2_to_pk_fp4(const bf16x2_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return pk_fp4_t::_pack(float_to_mxfp4(detail::lane0(x), scale),
                           float_to_mxfp4(detail::lane1(x), scale));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp32x2_to_pk_fp4(const fp32x2_t& x, float scale)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x, scale);
#else
    return pk_fp4_t::_pack(float_to_mxfp4(detail::lane0(x), scale),
                           float_to_mxfp4(detail::lane1(x), scale));
#endif
}

CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_to_fp32x2(const pk_fp4_t& x, float scale)
{
    return x.to_fp32x2(scale);
}
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_to_fp16x2(const pk_fp4_t& x, float scale)
{
    return x.to_fp16x2(scale);
}
CK_TILE_HOST_DEVICE constexpr bf16x2_t pk_fp4_to_bf16x2(const pk_fp4_t& x, float scale)
{
    return x.to_bf16x2(scale);
}
CK_TILE_HOST_DEVICE constexpr float pk_fp4_to_float(const pk_fp4_t& x, float scale)
{
    return x.to_float(scale);
}
CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp4_to_fp16(const pk_fp4_t& x, float scale)
{
    return x.to_fp16(scale);
}
CK_TILE_HOST_DEVICE constexpr bf16_t pk_fp4_to_bf16(const pk_fp4_t& x, float scale)
{
    return x.to_bf16(scale);
}

#if TEST_convert_with_table == 0
CK_TILE_HOST_DEVICE constexpr float pk_fp4_t::to_float(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp32_t>(data, scale);
#else
    return convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale);
#endif
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_t::to_fp32x2(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp32x2_t>(data, scale);
#else
    return fp32x2_t{convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale),
                    convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale)};
#endif
}

CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp4_t::to_fp16(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp16_t>(data, scale);
#else
    return fp16_t{type_convert<fp16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale))};
#endif
}
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_t::to_fp16x2(float scale) const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp16x2_t>(data, scale);
#else
    return fp16x2_t{type_convert<fp16_t>(convert_to_float<pk_fp4_t>(_unpack(number<0>{}), scale)),
                    type_convert<fp16_t>(convert_to_float<pk_fp4_t>(_unpack(number<1>{}), scale))};
#endif
}
#else
CK_TILE_HOST_DEVICE constexpr float pk_fp4_t::to_float(float scale) const
{
    return e2m1_to_fp32_table[_unpack(number<0>{})] * scale;
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_t::to_fp32x2(float scale) const
{
    return fp32x2_t{e2m1_to_fp32_table[_unpack(number<0>{})] * scale, e2m1_to_fp32_table[_unpack(number<1>{}] * scale};
}
CK_TILE_HOST_DEVICE constexpr fp16_t pk_fp4_t::to_fp16(float scale) const
{
    return type_convert<float>(e2m1_to_fp16_table[_unpack(number<0>{})]) * scale;
}
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_t::to_fp16x2(float scale) const
{
    return fp16x2_t{
        type_convert<fp16_t>(type_convert<float>(e2m1_to_fp16_table[_unpack(number<0>{})]) * scale),
        type_convert<fp16_t>(type_convert<float>(e2m1_to_fp16_table[_unpack(number<1>{})]) *
                             scale)};
}
#endif

} // namespace ck_tile
