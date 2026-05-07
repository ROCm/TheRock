// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"

template <typename T>
class ConvertTest : public ::testing::Test
{
};

using TestTypes = ::testing::Types<float, ck_tile::fp16_t>;

TYPED_TEST_SUITE(ConvertTest, TestTypes);

TYPED_TEST(ConvertTest, ToFp8)
{
    using SrcT = TypeParam;
    using DstT = ck_tile::fp8_t;

    auto c = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, true>(f)));
    };

    auto c_nosat = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, false>(f)));
    };

#if CK_TILE_USE_OCP_FP8
    EXPECT_EQ(c(+1.0f), 0b0'0111'000);
    EXPECT_EQ(c(-1.0f), 0b1'0111'000);
    // max f8 normal
    EXPECT_EQ(c(+448.0f), 0b0'1111'110);
    EXPECT_EQ(c(-448.0f), 0b1'1111'110);
    // min f8 normal
    EXPECT_EQ(c(+0.015625f), 0b0'0001'000);
    EXPECT_EQ(c(-0.015625f), 0b1'0001'000);
    // max f8 subnormal
    EXPECT_EQ(c(+0.013671875f), 0b0'0000'111);
    EXPECT_EQ(c(-0.013671875f), 0b1'0000'111);
    // min f8 subnormal
    EXPECT_EQ(c(+0.001953125f), 0b0'0000'001);
    EXPECT_EQ(c(-0.001953125f), 0b1'0000'001);
    // arbitrary values (exact)
    EXPECT_EQ(c(+0.203125f), 0b0'0100'101);
    EXPECT_EQ(c(-88.0f), 0b1'1101'011);
    // arbitrary values (rounded)
    EXPECT_EQ(c(+432.919f), 0b0'1111'110);
    EXPECT_EQ(c(-431.111f), 0b1'1111'101);
    EXPECT_EQ(c(-0.76123f), 0b1'0110'100);
    EXPECT_EQ(c(+0.81234f), 0b0'0110'101);
    // midpoint values (rounded to nearest even)
    EXPECT_EQ(c(+58.0f), 0b0'1100'110);
    EXPECT_EQ(c(+62.0f), 0b0'1101'000);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'1111'110);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'1111'110);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'1111'110);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'1111'110);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'1111'110);
    EXPECT_EQ(c(-1.23e9f), 0b1'1111'110);

    constexpr unsigned int nan_mask = 0b0'1111'111;

    // non-saturating mode -> f8 NaN (because OCP e4m3 has no infinity)
    // max f32/f16 normal -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()) & nan_mask, nan_mask);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()) & nan_mask, nan_mask);
    // f32/f16 infinity -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()) & nan_mask, nan_mask);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()) & nan_mask, nan_mask);
    // large f32/f16 -> f8 NaN
    EXPECT_EQ(c_nosat(+1.23e9f) & nan_mask, nan_mask);
    EXPECT_EQ(c_nosat(-1.23e9f) & nan_mask, nan_mask);

    // f32/f16 NaN -> f8 NaN
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::quiet_NaN()) & nan_mask, nan_mask);
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::signaling_NaN()) & nan_mask, nan_mask);

    // f32/f16 zero -> f8 zero
    EXPECT_EQ(c(+0.0f), 0b0'0000'000);
    EXPECT_EQ(c(-0.0f), 0b1'0000'000);
    // min f32/f16 normal -> f8 zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'0000'000);
    // min f32/f16 subnormal -> f8 zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b1'0000'000);

    // All values <= min f8 subnormal/2 must be converted to f8 zero
    EXPECT_EQ(c(+0.001953125f * 0.6f), 0b0'0000'001);
    EXPECT_EQ(c(-0.001953125f * 0.6f), 0b1'0000'001);
    constexpr int src_min_subnorm_exp =
        -(ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1);
    constexpr int dst_min_subnorm_exp =
        -(ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1);
    for(int exp = src_min_subnorm_exp; exp <= 0; ++exp)
    {
        const float f = std::ldexp(1.0, exp);
        if(exp < dst_min_subnorm_exp)
        {
            EXPECT_EQ(c(+f), 0b0'0000'000) << "+f = 2^" << exp << " = " << +f;
            EXPECT_EQ(c(-f), 0b1'0000'000) << "-f = 2^" << exp << " = " << -f;
        }
        else
        {
            EXPECT_GT(c(+f), 0b0'0000'000) << "+f = 2^" << exp << " = " << +f;
            EXPECT_GT(c(-f), 0b1'0000'000) << "-f = 2^" << exp << " = " << -f;
        }
    }
#else // FNUZ
    EXPECT_EQ(c(+1.0f), 0b0'1000'000);
    EXPECT_EQ(c(-1.0f), 0b1'1000'000);
    // max f8 normal
    EXPECT_EQ(c(+240.0f), 0b0'1111'111);
    EXPECT_EQ(c(-240.0f), 0b1'1111'111);
    // min f8 normal
    EXPECT_EQ(c(+0.0078125f), 0b0'0001'000);
    EXPECT_EQ(c(-0.0078125f), 0b1'0001'000);
    // max f8 subnormal
    EXPECT_EQ(c(+0.0068359375f), 0b0'0000'111);
    EXPECT_EQ(c(-0.0068359375f), 0b1'0000'111);
    // min f8 subnormal
    EXPECT_EQ(c(+0.0009765625f), 0b0'0000'001);
    EXPECT_EQ(c(-0.0009765625f), 0b1'0000'001);
    // arbitrary values (exact)
    EXPECT_EQ(c(+0.1015625f), 0b0'0100'101);
    EXPECT_EQ(c(-44.0f), 0b1'1101'011);
    // arbitrary values (rounded)
    EXPECT_EQ(c(+219.91f), 0b0'1111'110);
    EXPECT_EQ(c(-203.11f), 0b1'1111'101);
    EXPECT_EQ(c(-0.3639f), 0b1'0110'100);
    EXPECT_EQ(c(+0.4139f), 0b0'0110'101);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'1111'111);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'1111'111);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'1111'111);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'1111'111);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'1111'111);
    EXPECT_EQ(c(-1.23e9f), 0b1'1111'111);

    constexpr unsigned int nan_value = 0b1'0000'000;

    // non-saturating mode -> f8 NaN (FN means "finite", so no infinity)
    // max f32/f16 normal -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()), nan_value);
    // f32/f16 infinity -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()), nan_value);
    // large f32/f16 -> f8 NaN
    EXPECT_EQ(c_nosat(+1.23e9f), nan_value);
    EXPECT_EQ(c_nosat(-1.23e9f), nan_value);

    // f32/f16 NaN -> f8 NaN
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::quiet_NaN()), nan_value);
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::signaling_NaN()), nan_value);

    // UZ means "unsigned zero" (0b1'0000'000 is NaN)
    // f32/f16 +-zero -> f8 +zero
    EXPECT_EQ(c(+0.0f), 0b0'0000'000);
    EXPECT_EQ(c(-0.0f), 0b0'0000'000);
    // min f32/f16 normal -> f8 +zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b0'0000'000);
    // min f32/f16 subnormal -> f8 +zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'0000'000);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b0'0000'000);

    // All values <= min f8 subnormal/2 must be converted to f8 zero
    EXPECT_EQ(c(+0.0009765625f * 0.6f), 0b0'0000'001);
    EXPECT_EQ(c(-0.0009765625f * 0.6f), 0b1'0000'001);
    constexpr int src_min_subnorm_exp =
        -(ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1);
    constexpr int dst_min_subnorm_exp =
        -(ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1);
    for(int exp = src_min_subnorm_exp; exp <= 0; ++exp)
    {
        const float f = std::ldexp(1.0, exp);
        if(exp < dst_min_subnorm_exp)
        {
            EXPECT_EQ(c(+f), 0b0'0000'000) << "+f = 2^" << exp << " = " << +f;
            EXPECT_EQ(c(-f), 0b0'0000'000) << "-f = 2^" << exp << " = " << -f;
        }
        else
        {
            EXPECT_GT(c(+f), 0b0'0000'000) << "+f = 2^" << exp << " = " << +f;
            EXPECT_GT(c(-f), 0b0'0000'000) << "-f = 2^" << exp << " = " << -f;
        }
    }
#endif
}

TYPED_TEST(ConvertTest, ToBf8)
{
    using SrcT = TypeParam;
    using DstT = ck_tile::bf8_t;

    auto c = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, true>(f)));
    };

    auto c_nosat = [](SrcT f) {
        return static_cast<unsigned int>(
            ck_tile::bit_cast<uint8_t>(ck_tile::impl::run_cast_to_f8<SrcT, DstT, false>(f)));
    };

#if CK_TILE_USE_OCP_FP8
    EXPECT_EQ(c(+1.0f), 0b0'01111'00);
    EXPECT_EQ(c(-1.0f), 0b1'01111'00);
    // max f8 normal
    EXPECT_EQ(c(+57344.0f), 0b0'11110'11);
    EXPECT_EQ(c(-57344.0f), 0b1'11110'11);
    // min f8 normal
    EXPECT_EQ(c(+6.103515625e-05f), 0b0'00001'00);
    EXPECT_EQ(c(-6.103515625e-05f), 0b1'00001'00);
    // max f8 subnormal
    EXPECT_EQ(c(+4.57763671875e-05f), 0b0'00000'11);
    EXPECT_EQ(c(-4.57763671875e-05f), 0b1'00000'11);
    // min f8 subnormal
    EXPECT_EQ(c(+1.52587890625e-05f), 0b0'00000'01);
    EXPECT_EQ(c(-1.52587890625e-05f), 0b1'00000'01);
    // arbitrary values (exact)
    EXPECT_EQ(c(+0.01953125f), 0b0'01001'01);
    EXPECT_EQ(c(-3584.0f), 0b1'11010'11);
    // arbitrary values (rounded)
    EXPECT_EQ(c(+2030.56f), 0b0'11010'00);
    EXPECT_EQ(c(-1801.33f), 0b1'11001'11);
    EXPECT_EQ(c(-0.27891f), 0b1'0110'100);
    EXPECT_EQ(c(+0.33333f), 0b0'0110'101);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'11110'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'11110'11);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'11110'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'11110'11);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'11110'11);
    EXPECT_EQ(c(-1.23e9f), 0b1'11110'11);

    // non-saturating mode -> f8 infinity
    // max f32/f16 normal -> f8 infinity
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()), 0b0'11111'00);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()), 0b1'11111'00);
    // f32/f16 infinity -> f8 infinity
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()), 0b0'11111'00);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()), 0b1'11111'00);
    // large f32/f16 -> f8 infinity
    EXPECT_EQ(c_nosat(+1.23e9f), 0b0'11111'00);
    EXPECT_EQ(c_nosat(-1.23e9f), 0b1'11111'00);

    // f32/f16 NaN -> f8 NaN
    EXPECT_TRUE((c(ck_tile::numeric<SrcT>::quiet_NaN()) & 0b0'11111'11) != 0b0'11111'00);
    EXPECT_TRUE((c(ck_tile::numeric<SrcT>::signaling_NaN()) & 0b0'11111'11) != 0b0'11111'00);

    // f32/f16 zero -> f8 zero
    EXPECT_EQ(c(+0.0f), 0b0'00000'00);
    EXPECT_EQ(c(-0.0f), 0b1'00000'00);
    if constexpr(std::is_same_v<SrcT, float>)
    {
        // min f32 normal -> f8 zero
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00000'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'00000'00);
    }
    else
    {
        // min f16 normal -> min f8 normal (they are equal)
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00001'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'00001'00);
    }
    // min f32/f16 subnormal -> f8 zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'00000'00);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b1'00000'00);

    // All values <= min f8 subnormal/2 must be converted to f8 zero
    EXPECT_EQ(c(+1.52587890625e-05f * 0.6f), 0b0'0000'001);
    EXPECT_EQ(c(-1.52587890625e-05f * 0.6f), 0b1'0000'001);
    constexpr int src_min_subnorm_exp =
        -(ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1);
    constexpr int dst_min_subnorm_exp =
        -(ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1);
    for(int exp = src_min_subnorm_exp; exp <= 0; ++exp)
    {
        const float f = std::ldexp(1.0, exp);
        if(exp < dst_min_subnorm_exp)
        {
            EXPECT_EQ(c(+f), 0b0'00000'00) << "+f = 2^" << exp << " = " << +f;
            EXPECT_EQ(c(-f), 0b1'00000'00) << "-f = 2^" << exp << " = " << -f;
        }
        else
        {
            EXPECT_GT(c(+f), 0b0'00000'00) << "+f = 2^" << exp << " = " << +f;
            EXPECT_GT(c(-f), 0b1'00000'00) << "-f = 2^" << exp << " = " << -f;
        }
    }
#else // FNUZ
    EXPECT_EQ(c(+1.0f), 0b0'10000'00);
    EXPECT_EQ(c(-1.0f), 0b1'10000'00);
    // max f8 normal
    EXPECT_EQ(c(+57344.0f), 0b0'11111'11);
    EXPECT_EQ(c(-57344.0f), 0b1'11111'11);
    // min f8 normal
    EXPECT_EQ(c(+3.0517578125e-05f), 0b0'00001'00);
    EXPECT_EQ(c(-3.0517578125e-05f), 0b1'00001'00);
    // max f8 subnormal
    EXPECT_EQ(c(+2.288818359375e-05f), 0b0'00000'11);
    EXPECT_EQ(c(-2.288818359375e-05f), 0b1'00000'11);
    // min f8 subnormal
    EXPECT_EQ(c(+7.62939453125e-06f), 0b0'00000'01);
    EXPECT_EQ(c(-7.62939453125e-06f), 0b1'00000'01);
    // arbitrary values (exact)
    EXPECT_EQ(c(+0.009765625f), 0b0'01001'01);
    EXPECT_EQ(c(-1792.0f), 0b1'11010'11);
    // arbitrary values (rounded)
    EXPECT_EQ(c(+840.100f), 0b0'11001'11);
    EXPECT_EQ(c(-999.999f), 0b1'11010'00);
    EXPECT_EQ(c(-0.12789f), 0b1'0110'100);
    EXPECT_EQ(c(+0.14444f), 0b0'0110'101);

    // saturating mode -> max f8 normal
    // max f32/f16 normal -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::max()), 0b0'11111'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::max()), 0b1'1111'111);
    // f32/f16 infinity -> max f8 normal
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::infinity()), 0b0'11111'11);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::infinity()), 0b1'1111'111);
    // large f32/f16 -> max f8 normal
    EXPECT_EQ(c(+1.23e9f), 0b0'11111'11);
    EXPECT_EQ(c(-1.23e9f), 0b1'1111'111);

    constexpr unsigned int nan_value = 0b1'00000'00;

    // non-saturating mode -> f8 NaN (FN means "finite", so no infinity)
    // max f32/f16 normal -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::max()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::max()), nan_value);
    // f32/f16 infinity -> f8 NaN
    EXPECT_EQ(c_nosat(+ck_tile::numeric<SrcT>::infinity()), nan_value);
    EXPECT_EQ(c_nosat(-ck_tile::numeric<SrcT>::infinity()), nan_value);
    // large f32/f16 -> f8 NaN
    EXPECT_EQ(c_nosat(+1.23e9f), nan_value);
    EXPECT_EQ(c_nosat(-1.23e9f), nan_value);

    // f32/f16 NaN -> f8 NaN
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::quiet_NaN()), nan_value);
    EXPECT_EQ(c(ck_tile::numeric<SrcT>::signaling_NaN()), nan_value);

    // UZ means "unsigned zero" (0b1'00000'00 is NaN)
    // f32/f16 +-zero -> f8 +zero
    EXPECT_EQ(c(+0.0f), 0b0'00000'00);
    EXPECT_EQ(c(-0.0f), 0b0'00000'00);
    if constexpr(std::is_same_v<SrcT, float>)
    {
        // min f32 normal -> f8 +zero
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00000'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b0'00000'00);
    }
    else
    {
        // min f16 normal -> f8 normal
        EXPECT_EQ(c(+ck_tile::numeric<SrcT>::min()), 0b0'00010'00);
        EXPECT_EQ(c(-ck_tile::numeric<SrcT>::min()), 0b1'00010'00);
    }
    // min f32/f16 subnormal -> f8 +zero
    EXPECT_EQ(c(+ck_tile::numeric<SrcT>::denorm_min()), 0b0'00000'00);
    EXPECT_EQ(c(-ck_tile::numeric<SrcT>::denorm_min()), 0b0'00000'00);

    // All values <= min f8 subnormal/2 must be converted to f8 zero
    EXPECT_EQ(c(+7.62939453125e-06f * 0.6f), 0b0'0000'001);
    EXPECT_EQ(c(-7.62939453125e-06f * 0.6f), 0b1'0000'001);
    constexpr int src_min_subnorm_exp =
        -(ck_tile::numeric_traits<SrcT>::bias + ck_tile::numeric_traits<SrcT>::mant - 1);
    constexpr int dst_min_subnorm_exp =
        -(ck_tile::numeric_traits<DstT>::bias + ck_tile::numeric_traits<DstT>::mant - 1);
    for(int exp = src_min_subnorm_exp; exp <= 0; ++exp)
    {
        const float f = std::ldexp(1.0, exp);
        if(exp < dst_min_subnorm_exp)
        {
            EXPECT_EQ(c(+f), 0b0'00000'00) << "+f = 2^" << exp << " = " << +f;
            EXPECT_EQ(c(-f), 0b0'00000'00) << "-f = 2^" << exp << " = " << -f;
        }
        else
        {
            EXPECT_GT(c(+f), 0b0'00000'00) << "+f = 2^" << exp << " = " << +f;
            EXPECT_GT(c(-f), 0b0'00000'00) << "-f = 2^" << exp << " = " << -f;
        }
    }
#endif
}

TYPED_TEST(ConvertTest, FromFp8)
{
    using SrcT = ck_tile::fp8_t;
    using DstT = TypeParam;

    auto c = [](uint8_t u) {
        return ck_tile::type_convert<float>(
            ck_tile::impl::run_cast_from_f8<SrcT, DstT, true>(ck_tile::bit_cast<SrcT>(u)));
    };

#if CK_TILE_USE_OCP_FP8
    EXPECT_EQ(c(0b0'0111'000), +1.0f);
    EXPECT_EQ(c(0b1'0111'000), -1.0f);
    // max f8 normal
    EXPECT_EQ(c(0b0'1111'110), +448.0f);
    EXPECT_EQ(c(0b1'1111'110), -448.0f);
    // min f8 normal
    EXPECT_EQ(c(0b0'0001'000), +0.015625f);
    EXPECT_EQ(c(0b1'0001'000), -0.015625f);
    // max f8 subnormal
    EXPECT_EQ(c(0b0'0000'111), +0.013671875f);
    EXPECT_EQ(c(0b1'0000'111), -0.013671875f);
    // min f8 subnormal
    EXPECT_EQ(c(0b0'0000'001), +0.001953125f);
    EXPECT_EQ(c(0b1'0000'001), -0.001953125f);
    // arbitrary values
    EXPECT_EQ(c(0b0'0100'101), +0.203125f);
    EXPECT_EQ(c(0b1'1101'011), -88.0f);

    // f8 NaN -> f32/f16 NaN
    EXPECT_TRUE(ck_tile::isnan(c(0b0'1111'111)));
    EXPECT_TRUE(ck_tile::isnan(c(0b1'1111'111)));

    // f8 zero -> f32/f16 zero (sign is preserved)
    EXPECT_EQ(c(0b0'0000'000),
              ck_tile::bit_cast<DstT>(typename ck_tile::numeric_traits<DstT>::bitwise_type{0}));
    EXPECT_EQ(c(0b1'0000'000), ck_tile::bit_cast<DstT>(ck_tile::numeric_traits<DstT>::Neg0));
#else // FNUZ
    EXPECT_EQ(c(0b0'1000'000), +1.0f);
    EXPECT_EQ(c(0b1'1000'000), -1.0f);
    // max f8 normal
    EXPECT_EQ(c(0b0'1111'111), +240.0f);
    EXPECT_EQ(c(0b1'1111'111), -240.0f);
    // min f8 normal
    EXPECT_EQ(c(0b0'0001'000), +0.0078125f);
    EXPECT_EQ(c(0b1'0001'000), -0.0078125f);
    // max f8 subnormal
    EXPECT_EQ(c(0b0'0000'111), +0.0068359375f);
    EXPECT_EQ(c(0b1'0000'111), -0.0068359375f);
    // min f8 subnormal
    EXPECT_EQ(c(0b0'0000'001), +0.0009765625f);
    EXPECT_EQ(c(0b1'0000'001), -0.0009765625f);
    // arbitrary values
    EXPECT_EQ(c(0b0'0100'101), +0.1015625f);
    EXPECT_EQ(c(0b1'1101'011), -44.0f);

    // f8 NaN -> f32/f16 NaN
    EXPECT_TRUE(ck_tile::isnan(c(0b1'0000'000)));

    // UZ means "unsigned zero" (0b1'0000'000 is NaN)
    // f8 +zero -> f32/f16 +zero
    EXPECT_EQ(c(0b0'0000'000),
              ck_tile::bit_cast<DstT>(typename ck_tile::numeric_traits<DstT>::bitwise_type{0}));
#endif
}

TYPED_TEST(ConvertTest, FromBf8)
{
    using SrcT = ck_tile::bf8_t;
    using DstT = TypeParam;

    using DstT = TypeParam;

    auto c = [](uint8_t u) {
        return ck_tile::type_convert<float>(
            ck_tile::impl::run_cast_from_f8<SrcT, DstT, true>(ck_tile::bit_cast<SrcT>(u)));
    };

#if CK_TILE_USE_OCP_FP8
    auto c_nosat = [](uint8_t u) {
        return ck_tile::type_convert<float>(
            ck_tile::impl::run_cast_from_f8<SrcT, DstT, false>(ck_tile::bit_cast<SrcT>(u)));
    };

    EXPECT_EQ(c(0b0'01111'00), +1.0f);
    EXPECT_EQ(c(0b1'01111'00), -1.0f);
    // max f8 normal
    EXPECT_EQ(c(0b0'11110'11), +57344.0f);
    EXPECT_EQ(c(0b1'11110'11), -57344.0f);
    // min f8 normal
    EXPECT_EQ(c(0b0'00001'00), +6.103515625e-05f);
    EXPECT_EQ(c(0b1'00001'00), -6.103515625e-05f);
    // max f8 subnormal
    EXPECT_EQ(c(0b0'00000'11), +4.57763671875e-05f);
    EXPECT_EQ(c(0b1'00000'11), -4.57763671875e-05f);
    // min f8 subnormal
    EXPECT_EQ(c(0b0'00000'01), +1.52587890625e-05f);
    EXPECT_EQ(c(0b1'00000'01), -1.52587890625e-05f);
    // arbitrary values
    EXPECT_EQ(c(0b0'01001'01), +0.01953125f);
    EXPECT_EQ(c(0b1'11010'11), -3584.0f);

    // saturating mode
    // f8 infinity -> max f8 normal as f32/f16
    EXPECT_EQ(c(0b0'11111'00), +57344.0f);
    EXPECT_EQ(c(0b1'11111'00), -57344.0f);

    // non-saturating mode
    // f8 infinity -> f32/f16 infinity
    EXPECT_EQ(c_nosat(0b0'11111'00), +ck_tile::numeric<DstT>::infinity());
    EXPECT_EQ(c_nosat(0b1'11111'00), -ck_tile::numeric<DstT>::infinity());

    // f8 NaN -> f32/f16 NaN
    EXPECT_TRUE(ck_tile::isnan(c(0b0'11111'01)));
    EXPECT_TRUE(ck_tile::isnan(c(0b0'11111'10)));
    EXPECT_TRUE(ck_tile::isnan(c(0b0'11111'11)));
    EXPECT_TRUE(ck_tile::isnan(c(0b1'11111'01)));
    EXPECT_TRUE(ck_tile::isnan(c(0b1'11111'10)));
    EXPECT_TRUE(ck_tile::isnan(c(0b1'11111'11)));

    // f8 zero -> f32/f16 zero (sign is preserved)
    EXPECT_EQ(c(0b0'00000'00),
              ck_tile::bit_cast<DstT>(typename ck_tile::numeric_traits<DstT>::bitwise_type{0}));
    EXPECT_EQ(c(0b1'00000'00), ck_tile::bit_cast<DstT>(ck_tile::numeric_traits<DstT>::Neg0));
    if constexpr(std::is_same_v<DstT, ck_tile::fp16_t>)
    {
        // min f8 normal -> min f16 normal (they are equal)
        EXPECT_EQ(c(0b0'00001'00), +ck_tile::numeric<DstT>::min());
        EXPECT_EQ(c(0b1'00001'00), -ck_tile::numeric<DstT>::min());
    }
#else // FNUZ
    EXPECT_EQ(c(0b0'10000'00), +1.0f);
    EXPECT_EQ(c(0b1'10000'00), -1.0f);
    // max f8 normal
    EXPECT_EQ(c(0b0'11111'11), +57344.0f);
    EXPECT_EQ(c(0b1'11111'11), -57344.0f);
    // min f8 normal
    EXPECT_EQ(c(0b0'00001'00), +3.0517578125e-05f);
    EXPECT_EQ(c(0b1'00001'00), -3.0517578125e-05f);
    // max f8 subnormal
    EXPECT_EQ(c(0b0'00000'11), +2.288818359375e-05f);
    EXPECT_EQ(c(0b1'00000'11), -2.288818359375e-05f);
    // min f8 subnormal
    EXPECT_EQ(c(0b0'00000'01), +7.62939453125e-06f);
    EXPECT_EQ(c(0b1'00000'01), -7.62939453125e-06f);
    // arbitrary values
    EXPECT_EQ(c(0b0'01001'01), +0.009765625f);
    EXPECT_EQ(c(0b1'11010'11), -1792.0f);

    // f8 NaN -> f32/f16 NaN
    EXPECT_TRUE(ck_tile::isnan(c(0b1'00000'00)));

    // UZ means "unsigned zero" (0b1'00000'00 is NaN)
    // f8 +zero -> f32/f16 +zero
    EXPECT_EQ(c(0b0'00000'00),
              ck_tile::bit_cast<DstT>(typename ck_tile::numeric_traits<DstT>::bitwise_type{0}));
    if constexpr(std::is_same_v<DstT, ck_tile::fp16_t>)
    {
        // one of f8 normals -> min f16 normal
        EXPECT_EQ(c(0b0'00010'00), +ck_tile::numeric<DstT>::min());
        EXPECT_EQ(c(0b1'00010'00), -ck_tile::numeric<DstT>::min());
    }
#endif
}

// Convert f8 -> f32/f16 -> f8 to check if all values are covered
// OCP types multiple NaN representations (e4m3 - 2, e5m2 - 6), they are ignored for simplicity.

TYPED_TEST(ConvertTest, FromFp8AndToFp8)
{
    using SrcT = ck_tile::fp8_t;
    using DstT = TypeParam;

    for(int i = 0; i < 256; ++i)
    {
#if CK_TILE_USE_OCP_FP8
        if((i & 0b0'1111'111) == 0b0'1111'111)
        {
            continue;
        }
#endif
        const uint8_t u = static_cast<uint8_t>(i);
        const SrcT from = ck_tile::bit_cast<SrcT>(u);
        const DstT f    = ck_tile::impl::run_cast_from_f8<SrcT, DstT, false>(from);
        const SrcT to   = ck_tile::impl::run_cast_to_f8<DstT, SrcT, false>(f);
        EXPECT_EQ(from, to) << "u8: " << i << " f32/f16: " << ck_tile::type_convert<float>(f);
    }
}

TYPED_TEST(ConvertTest, FromBf8AndToBf8)
{
    using SrcT = ck_tile::bf8_t;
    using DstT = TypeParam;

    for(int i = 0; i < 256; ++i)
    {
#if CK_TILE_USE_OCP_FP8
        if((i & 0b0'11111'11) > 0b0'11111'00)
        {
            continue;
        }
#endif
        const uint8_t u = static_cast<uint8_t>(i);
        const SrcT from = ck_tile::bit_cast<SrcT>(u);
        const DstT f    = ck_tile::impl::run_cast_from_f8<SrcT, DstT, false>(from);
        const SrcT to   = ck_tile::impl::run_cast_to_f8<DstT, SrcT, false>(f);
        EXPECT_EQ(from, to) << "u8: " << i << " f32/f16: " << ck_tile::type_convert<float>(f);
    }
}
