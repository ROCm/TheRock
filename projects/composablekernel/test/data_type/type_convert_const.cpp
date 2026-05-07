// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bhalf_t;
using ck::type_convert;

TEST(TypeConvertConst, ConvertToConst)
{
    constexpr float bf16_epsilon = 0.0078125;
    constexpr float rel_tol      = 2 * bf16_epsilon;

    const std::vector<float> cases = {0.0, -123.f, 3.981323f, 0.2429f};

    for(float x : cases)
    {
        const float abs_tol = std::abs(rel_tol * x);
        {
            bhalf_t y = type_convert<bhalf_t>(x);
            // Test non-const bhalf to const float.
            const float y_float = type_convert<const float>(y);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
        {
            // Test non-const float to const bhalf.
            const bhalf_t y = type_convert<const bhalf_t>(x);
            // Remove the constness manually to not rely on const casts anymore since the
            // possible issue could hide after two casts.
            bhalf_t& y_nonconst = const_cast<bhalf_t&>(y);
            float y_float       = type_convert<float>(y_nonconst);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
    }
}

TEST(TypeConvertConst, ConvertFromConst)
{
    constexpr float bf16_epsilon = 0.0078125;
    constexpr float rel_tol      = 2 * bf16_epsilon;

    const std::vector<float> cases = {0.0, -123.f, 3.981323f, 0.2429f};

    for(const float x : cases)
    {
        const float abs_tol = std::abs(rel_tol * x);
        {
            // Test const float to const bhalf_t.
            const bhalf_t y = type_convert<const bhalf_t>(x);
            // Remove the constness manually to not rely on const casts anymore since the
            // possible issue could hide after two casts.
            bhalf_t& y_nonconst = const_cast<bhalf_t&>(y);
            float y_float       = type_convert<float>(y_nonconst);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
        {
            // Test const float to non-const bhalf.
            bhalf_t y     = type_convert<bhalf_t>(x);
            float y_float = type_convert<float>(y);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
        {
            const bhalf_t y = type_convert<const bhalf_t>(x);
            // Test const bhalf to non-const float.
            float y_float = type_convert<float>(y);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
        // Tests with full type specializations for X.
        {
            // Test const float to const bhalf_t.
            const bhalf_t y = type_convert<const bhalf_t, const float>(x);
            // Remove the constness manually to not rely on const casts anymore since the
            // possible issue could hide after two casts.
            bhalf_t& y_nonconst = const_cast<bhalf_t&>(y);
            float y_float       = type_convert<float>(y_nonconst);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
        {
            // Test const float to non-const bhalf.
            bhalf_t y     = type_convert<bhalf_t, const float>(x);
            float y_float = type_convert<float>(y);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
        {
            const bhalf_t y = type_convert<const bhalf_t, const float>(x);
            // Test const bhalf to non-const float.
            float y_float = type_convert<float, const bhalf_t>(y);
            ASSERT_NEAR(y_float, x, abs_tol);
        }
    }
}
