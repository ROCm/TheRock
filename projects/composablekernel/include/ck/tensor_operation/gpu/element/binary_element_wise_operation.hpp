// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct Add
{
    static constexpr const char* name = "Add";

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const half_t& x1) const
    {
        y = x0 + type_convert<half_t>(x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const float& x1) const
    {
        y = type_convert<half_t>(x0 + x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = x0 + type_convert<float>(x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x1);
        y                  = x0 + x1_tmp;
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp + x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
    {
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x0 + x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 + x1;
    };
};

struct Max
{
    static constexpr const char* name = "Max";

    template <typename Y, typename X0, typename X1>
    __host__ __device__ void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        const Y x0_converted = type_convert<Y>(x0);
        const Y x1_converted = type_convert<Y>(x1);
        y                    = ck::math::max(x0_converted, x1_converted);
    }
};

struct Min
{
    static constexpr const char* name = "Min";

    template <typename Y, typename X0, typename X1>
    __host__ __device__ void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        const Y x0_converted = type_convert<Y>(x0);
        const Y x1_converted = type_convert<Y>(x1);
        y                    = ck::math::min(x0_converted, x1_converted);
    }
};

struct Multiply
{
    static constexpr const char* name = "Multiply";

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const half_t& x1) const
    {
        y = x0 * type_convert<half_t>(x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const float& x1) const
    {
        y = type_convert<half_t>(x0 * x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(x0) * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x1);
        y                  = x0 * x1_tmp;
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp * x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const int8_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp * x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
    {
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x0 * x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 * x1;
    };
};

struct ScaleAdd
{
    static constexpr const char* name = "ScaleAdd";

    __host__ __device__ ScaleAdd(float scale = 1.f) : scale_(scale) {}

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = ck::type_convert<Y>(scale_ * ck::type_convert<float>(x0) + ck::type_convert<float>(x1));
    }

    template <>
    __host__ __device__ void
    operator()<float, float, half_t>(float& y, const float& x0, const half_t& x1) const
    {
        y = scale_ * x0 + ck::type_convert<float>(x1);
    };

    template <>
    __host__ __device__ void
    operator()<float, float, bhalf_t>(float& y, const float& x0, const bhalf_t& x1) const
    {
        y = scale_ * x0 + ck::type_convert<float>(x1);
    };

    float scale_;
};

struct Subtract
{
    static constexpr const char* name = "Subtract";

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp - x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 - x1;
    };
};

struct Bilinear
{
    static constexpr const char* name = "Bilinear";

    Bilinear(float alpha = 1.f, float beta = 1.f) : alpha_(alpha), beta_(beta){};

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&) const;

    template <>
    __host__ __device__ constexpr void
    operator()<double, double, double>(double& y, const double& x0, const double& x1) const
    {
        y = alpha_ * x0 + beta_ * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& y, const float& x0, const float& x1) const
    {
        y = alpha_ * x0 + beta_ * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t, int8_t, int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = type_convert<int8_t>(alpha_ * type_convert<float>(x0) +
                                 beta_ * type_convert<float>(x1));
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(alpha_) * x0 + type_convert<half_t>(beta_) * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(alpha_ * x0 + beta_ * ck::type_convert<float>(x1));
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x0_tmp = type_convert<float>(x0);
        const float x1_tmp = type_convert<float>(x1);
        const float y_tmp  = alpha_ * x0_tmp + beta_ * x1_tmp;
        y                  = type_convert<bhalf_t>(y_tmp);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, float, bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
    {
        y = type_convert<bhalf_t>(alpha_ * x0 + beta_ * ck::type_convert<float>(x1));
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t, int32_t, int8_t>(int8_t& y, const int32_t& x0, const int8_t& x1) const
    {
        y = type_convert<int8_t>(alpha_ * type_convert<float>(x0) +
                                 beta_ * type_convert<float>(x1));
    };

    float alpha_;
    float beta_;
};

struct AddClamp
{
    static constexpr const char* name = "AddClamp";

    AddClamp(float floor = 0.f, float ceil = NumericLimits<float>::Max())
        : floor_(floor), ceil_(ceil){};

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double, double, double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t floor = type_convert<half_t>(floor_);
        const half_t ceil  = type_convert<half_t>(ceil_);
        const half_t a     = x0 + x1;
        y                  = a > floor ? (a < ceil ? a : ceil) : floor;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        const float b = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
        y             = type_convert<half_t>(b);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, half_t>(float& y, const float& x0, const half_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        y             = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, float, bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        const float b = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
        y             = type_convert<bhalf_t>(b);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float a = type_convert<float>(x0) + type_convert<float>(x1);
        const float b = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
        y             = type_convert<bhalf_t>(b);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int, int, int8_t>(int& y, const int& x0, const int8_t& x1) const
    {
        const int8_t a = x0 + x1;
        y              = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t, int8_t, int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        const int8_t a = x0 + x1;
        y              = a > floor_ ? (a < ceil_ ? a : ceil_) : floor_;
    };

    const float floor_;
    const float ceil_;
};

struct AddRelu
{
    static constexpr const char* name = "AddRelu";

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = a > 0.0f ? a : 0.0f;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double, double, double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = a > 0.0 ? a : 0.0;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = a > type_convert<half_t>(0.0f) ? a : type_convert<half_t>(0.0f);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        const float b = a > 0.0f ? a : 0.0f;
        y             = type_convert<half_t>(b);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, half_t>(float& y, const float& x0, const half_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        y             = a > 0.0f ? a : 0.0f;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, float, bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        const float b = a > 0.0f ? a : 0.0f;
        y             = type_convert<bhalf_t>(b);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float a = type_convert<float>(x0) + type_convert<float>(x1);
        const float b = a > 0.0f ? a : 0.0f;
        y             = type_convert<bhalf_t>(b);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int, int, int8_t>(int& y, const int& x0, const int8_t& x1) const
    {
        const int8_t a = x0 + x1;
        y              = a > 0 ? a : 0;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t, int8_t, int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        const int8_t a = x0 + x1;
        y              = a > 0 ? a : 0;
    };
};

struct AddHardswish
{
    static constexpr const char* name = "AddHardswish";

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        double a = x0 + x1;
        double b = a + 3.0;
        double c = (b > 0) * (b > 6.0 ? 6.0 : b) * a * 0.166667;
        y        = c;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        float a = x0 + x1;
        float b = a + 3.0f;
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };
};

// E = FastGelu(C + D)
struct AddFastGelu
{
    static constexpr const char* name = "AddFastGelu";

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& e, const float& c, const float& d) const
    {
        const float x = c + d;

        FastGelu{}.template operator()<float, float>(e, x);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& e, const half_t& c, const half_t& d) const
    {
        const half_t x = c + d;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<half_t, half_t>(e, x);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& e, const float& c, const half_t& d) const
    {
        const float x0_f = c + d;

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<half_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, bhalf_t, bhalf_t>(bhalf_t& e, const bhalf_t& c, const bhalf_t& d) const
    {
        const float x0_f = type_convert<float>(c) + type_convert<float>(d);

        float x1_f = 0;

        FastGelu{}.template operator()<float, float>(x1_f, x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, float, bhalf_t>(bhalf_t& e, const float& c, const bhalf_t& d) const
    {
        const float x0_f = c + type_convert<float>(d);

        float x1_f = 0;

        FastGelu{}.template operator()<float, float>(x1_f, x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }
};

// E = MultiplyFastGelu(C + D)
struct MultiplyFastGelu
{
    static constexpr const char* name = "MultiplyFastGelu";

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& e, const float& c, const float& d) const
    {
        const float x = c * d;

        FastGelu{}.template operator()<float, float>(e, x);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& e, const half_t& c, const half_t& d) const
    {
        const half_t x = c * d;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<half_t, half_t>(e, x);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& e, const float& c, const half_t& d) const
    {
        const float x0_f = c * d;

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<half_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, bhalf_t, bhalf_t>(bhalf_t& e, const bhalf_t& c, const bhalf_t& d) const
    {
        const float x0_f = type_convert<float>(c) * type_convert<float>(d);

        float x1_f = 0;

        FastGelu{}.template operator()<float, float>(x1_f, x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, float, bhalf_t>(bhalf_t& e, const float& c, const bhalf_t& d) const
    {
        const float x0_f = c * type_convert<float>(d);

        float x1_f = 0;

        FastGelu{}.template operator()<float, float>(x1_f, x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }
};

// E = Silu(C + D)
struct AddSilu
{
    static constexpr const char* name = "AddSilu";

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& e, const float& c, const float& d) const
    {
        const float x = c + d;

        Silu{}.template operator()<float>(e, x);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& e, const half_t& c, const half_t& d) const
    {
        const half_t x = c + d;

        Silu{}.template operator()<half_t>(e, x);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& e, const float& c, const half_t& d) const
    {
        const float x0_f = c + d;

        float x1_f = 0;

        Silu{}.template operator()<float>(x1_f, x0_f);

        e = type_convert<half_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t, float, bhalf_t>(bhalf_t& e, const float& c, const bhalf_t& d) const
    {
        const float x0_f = c + type_convert<float>(d);

        float x1_f = 0;

        Silu{}.template operator()<float>(x1_f, x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }
};

struct ConvScaleAdd
{
    static constexpr const char* name = "ConvScaleAdd";

    __host__ __device__ ConvScaleAdd(float scale_in  = 1.f,
                                     float scale_wei = 1.f,
                                     float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C, typename D>
    __host__ __device__ void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ void
    operator()<f8_t, float, float>(f8_t& e, const float& c, const float& d) const
    {
        float x;
        Add{}.template operator()<float>(x, c * scale_in_ * scale_wei_, d);
        e = type_convert<f8_t>(x * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
