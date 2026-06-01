// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/quantization_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// Need to ensure compiler will fail if there is no matching candidate, instead of compiler
// siliently do implicit type conversion
//
// Example:
//
// struct ExampleElementwiseOp
// {
//     template<typename Y, typename X>
//     __host__ __device__ constexpr void
//     operator()(Y&, const X) const;
//
//     template<>
//     __host__ __device__ constexpr void
//     operator()<half_t, half_t>(half_t& y, const half_t& x) const
//     {
//     }
// };

struct AddReluAdd
{
    static constexpr const char* name = "AddReluAdd";

    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&, const X2&) const;

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        half_t a = x0 + x1;
        half_t b = a > 0 ? a : 0;
        y        = b + x2;
    }

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& y,
                                                                              const float& x0,
                                                                              const float& x1,
                                                                              const float& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<float, float, half_t, half_t>(
        float& y, const float& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t, half_t>(
        half_t& y, const float& x0, const half_t& x1, const half_t& x2) const
    {
        float y_float = 0.0;
        (*this)(y_float, x0, x1, x2);
        y = y_float;
    }

    template <>
    __host__ __device__ constexpr void operator()<bhalf_t, float, bhalf_t, bhalf_t>(
        bhalf_t& y, const float& x0, const bhalf_t& x1, const bhalf_t& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    template <>
    __host__ __device__ constexpr void operator()<int8_t, int8_t, int8_t, int8_t>(
        int8_t& y, const int8_t& x0, const int8_t& x1, const int8_t& x2) const
    {
        int32_t a = x0 + x1;
        int32_t b = a > 0 ? a : 0;
        int32_t c = b + x2;
        y         = c;
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    __host__ __device__ constexpr void operator()<int4_t, int8_t, int4_t, int4_t>(
        int4_t& y, const int8_t& x0, const int4_t& x1, const int4_t& x2) const
    {
        int32_t a = x0 + x1;
        int32_t b = a > 0 ? a : 0;
        int32_t c = b + x2;
        y         = c;
    }
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
};

struct AddHardswishAdd
{
    static constexpr const char* name = "AddHardswishAdd";

    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&, const X2&) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& y,
                                                                              const float& x0,
                                                                              const float& x1,
                                                                              const float& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }
};

// C = A * B
// E = C + D0 + D1
struct AddAdd
{
    static constexpr const char* name = "AddAdd";

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const
    {
        // Only support floating so far
        static_assert(is_same<E, half_t>::value || is_same<E, float>::value ||
                          is_same<E, double>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<C, half_t>::value || is_same<C, float>::value ||
                          is_same<C, double>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<D0, half_t>::value || is_same<D0, float>::value ||
                          is_same<D0, double>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<D1, half_t>::value || is_same<D1, float>::value ||
                          is_same<D1, double>::value,
                      "Data type is not supported by this operation!");

        const C y = c + type_convert<C>(d0) + type_convert<C>(d1);
        e         = type_convert<E>(y);
    }
};

// C = A * B
// E = (C + D0) x D1
struct AddMultiply
{
    static constexpr const char* name = "AddMultiply";

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ void operator()<half_t, half_t, half_t, half_t>(half_t& e,
                                                                        const half_t& c,
                                                                        const half_t& d0,
                                                                        const half_t& d1) const
    {
        const half_t y = (c + d0) * d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<half_t, float, half_t, half_t>(half_t& e,
                                                                       const float& c,
                                                                       const half_t& d0,
                                                                       const half_t& d1) const
    {
        const half_t y = (type_convert<half_t>(c) + d0) * d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<float, float, half_t, half_t>(float& e,
                                                                      const float& c,
                                                                      const half_t& d0,
                                                                      const half_t& d1) const
    {
        const float y = (c + d0) * d1;
        e             = y;
    }
};

// C = A * B
// E = C x D0 + D1
struct MultiplyAdd
{
    static constexpr const char* name = "MultiplyAdd";

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ void operator()<half_t, half_t, half_t, half_t>(half_t& e,
                                                                        const half_t& c,
                                                                        const half_t& d0,
                                                                        const half_t& d1) const
    {
        const half_t y = (c * d0) + d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<half_t, float, half_t, half_t>(half_t& e,
                                                                       const float& c,
                                                                       const half_t& d0,
                                                                       const half_t& d1) const
    {
        const half_t y = type_convert<half_t>(c) * d0 + d1;
        e              = y;
    }
    template <>
    __host__ __device__ void operator()<bhalf_t, float, bhalf_t, bhalf_t>(bhalf_t& e,
                                                                          const float& c,
                                                                          const bhalf_t& d0,
                                                                          const bhalf_t& d1) const
    {
        const bhalf_t y = type_convert<bhalf_t>(c) * d0 + d1;
        e               = y;
    }
    template <>
    __host__ __device__ void operator()<float, float, half_t, half_t>(float& e,
                                                                      const float& c,
                                                                      const half_t& d0,
                                                                      const half_t& d1) const
    {
        const float y = c * d0 + d1;
        e             = y;
    }
    template <>
    __host__ __device__ void operator()<half_t, float, float, float>(half_t& e,
                                                                     const float& c,
                                                                     const float& d0,
                                                                     const float& d1) const
    {
        const float y = c * d0 + d1;
        e             = y;
    }
};

struct MultiplyMultiply
{
    static constexpr const char* name = "MultiplyMultiply";

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, float, float>(
        ck::half_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, float, float, float>(
        ck::bhalf_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, ck::half_t, ck::half_t>(
        ck::half_t& e, const int& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, float, float>(
        ck::half_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float>(
        ck::bhalf_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }
};

struct MultiplyAddFastGelu
{
    static constexpr const char* name = "MultiplyAddFastGelu";

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, float, ck::bhalf_t, ck::bhalf_t>(
        ck::bhalf_t& e, const float& c, const ck::bhalf_t& d0, const ck::bhalf_t& d1) const
    {
        const float x0_f = c * ck::type_convert<float>(d0) + ck::type_convert<float>(d1);

        float x1_f = 0;

        FastGelu{}.template operator()<float, float>(x1_f, x0_f);

        e = ck::type_convert<ck::bhalf_t>(x1_f);
    }
};

// E = FastGelu(C + D0 + D1)
struct AddAddFastGelu
{
    static constexpr const char* name = "AddAddFastGelu";

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& e,
                                                                              const float& c,
                                                                              const float& d0,
                                                                              const float& d1) const
    {
        const float x = c + d0 + d1;

        FastGelu{}.template operator()<float, float>(e, x);
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& e, const half_t& c, const half_t& d0, const half_t& d1) const
    {
        const half_t x = c + d0 + d1;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<half_t, half_t>(e, x);
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t, half_t>(
        half_t& e, const float& c, const half_t& d0, const half_t& d1) const
    {
        const float x0_f = c + d0 + d1;

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<half_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<bhalf_t, float, bhalf_t, bhalf_t>(
        bhalf_t& e, const float& c, const bhalf_t& d0, const bhalf_t& d1) const
    {
        const float x0_f = c + type_convert<float>(d0) + type_convert<float>(d1);

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<bhalf_t>(x1_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<int8_t, int32_t, int8_t, int8_t>(
        int8_t& e, const int32_t& c, const int8_t& d0, const int8_t& d1) const
    {
        const float x0_f =
            type_convert<float>(c) + type_convert<float>(d0) + type_convert<float>(d1);

        float x1_f = 0;

        ck::tensor_operation::element_wise::FastGelu{}.template operator()<float, float>(x1_f,
                                                                                         x0_f);

        e = type_convert<int8_t>(x1_f);
    }
};

// E = Relu(alpha1 * C + alpha2 * D0 + D1)
struct ScaleAddScaleAddRelu
{
    static constexpr const char* name = "ScaleAddScaleAddRelu";

    ScaleAddScaleAddRelu(const float alpha1 = 1.f, const float alpha2 = 1.f)
        : alpha1_(alpha1), alpha2_(alpha2)
    {
    }

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float>(float& e,
                                                                              const float& c,
                                                                              const float& d0,
                                                                              const float& d1) const
    {
        const float x = c * alpha1_ + alpha2_ * d0 + d1;
        e             = x > 0 ? x : 0;
    }

    template <>
    __host__ __device__ constexpr void operator()<half_t, half_t, half_t, half_t>(
        half_t& e, const half_t& c, const half_t& d0, const half_t& d1) const
    {
        const float x = type_convert<float>(c) * alpha1_ + alpha2_ * type_convert<float>(d0) +
                        type_convert<float>(d1);

        float result = 0;
        result       = x > 0 ? x : 0;

        e = type_convert<half_t>(result);
    }

    template <>
    __host__ __device__ constexpr void operator()<bhalf_t, bhalf_t, bhalf_t, bhalf_t>(
        bhalf_t& e, const bhalf_t& c, const bhalf_t& d0, const bhalf_t& d1) const
    {
        const float x = type_convert<float>(c) * alpha1_ + alpha2_ * type_convert<float>(d0) +
                        type_convert<float>(d1);

        float result = 0;
        result       = x > 0 ? x : 0;

        e = type_convert<bhalf_t>(result);
    }

    template <>
    __host__ __device__ constexpr void operator()<int8_t, int8_t, float, float>(
        int8_t& e, const int8_t& c, const float& d0, const float& d1) const
    {
        const float x = type_convert<float>(c) * alpha1_ + alpha2_ * d0 + d1;

        float result = 0;
        result       = x > 0 ? x : 0;

        e = type_convert<int8_t>(result);
    }

    const float alpha1_;
    const float alpha2_;
};

struct Normalize
{
    static constexpr const char* name = "Normalize";

    // FIXME: is double absolutely necessary?
    Normalize(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2, typename T3>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& mean_square,
                                                  const T3& gamma,
                                                  const T3& beta) const;

    template <>
    __host__ __device__ constexpr void operator()<half_t, float, half_t>(half_t& y,
                                                                         const half_t& x,
                                                                         const float& mean,
                                                                         const float& mean_square,
                                                                         const half_t& gamma,
                                                                         const half_t& beta) const
    {
        using ck::math::sqrt;

        float variance = mean_square - (mean * mean);

        float tmp_x     = type_convert<float>(x);
        float tmp_gamma = type_convert<float>(gamma);
        float tmp_beta  = type_convert<float>(beta);

        float tmp_y =
            ((tmp_x - mean) / sqrt(variance + type_convert<float>(epsilon_))) * tmp_gamma +
            tmp_beta;

        y = type_convert<half_t>(tmp_y);
    };

    template <>
    __host__ __device__ constexpr void operator()<float, float, float>(float& y,
                                                                       const float& x,
                                                                       const float& mean,
                                                                       const float& mean_square,
                                                                       const float& gamma,
                                                                       const float& beta) const
    {
        using ck::math::sqrt;

        float variance = mean_square - (mean * mean);
        y = ((x - mean) / sqrt(variance + type_convert<float>(epsilon_))) * gamma + beta;
    };

    template <>
    __host__ __device__ constexpr void operator()<double, double, double>(double& y,
                                                                          const double& x,
                                                                          const double& mean,
                                                                          const double& mean_square,
                                                                          const double& gamma,
                                                                          const double& beta) const
    {
        using ck::math::sqrt;

        double variance = mean_square - (mean * mean);
        y               = ((x - mean) / sqrt(variance + epsilon_)) * gamma + beta;
    };

    // FIXME: is double absolutely necessary?
    double epsilon_;
};

// used by BatchNorm inference
// y = gamma * (x-mean) / sqrt(epsilon+variance) + beta
// The data type of mean and variance is used as AccDataType
struct NormalizeInInfer
{
    static constexpr const char* name = "NormalizeInInfer";

    NormalizeInInfer(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2, typename T3, typename T4>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& variance,
                                                  const T3& gamma,
                                                  const T4& beta) const
    {
        static_assert(is_same<T2, float>::value || is_same<T2, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T2 tmp_x, tmp_y;

        tmp_x = type_convert<T2>(x);

        tmp_y = ((tmp_x - mean) / sqrt(variance + type_convert<T2>(epsilon_))) *
                    type_convert<T2>(gamma) +
                type_convert<T2>(beta);
        y = type_convert<T1>(tmp_y);
    };

    double epsilon_;
};

// used by Conv+Bias+BatchNorm+Clamp inference
struct BiasNormalizeInInferClamp
{
    static constexpr const char* name = "BiasNormalizeInInferClamp";

    BiasNormalizeInInferClamp(float floor   = 0.f,
                              float ceil    = NumericLimits<float>::Max(),
                              float epsilon = 1e-4)
        : clamp_(floor, ceil), epsilon_(epsilon)
    {
    }

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y,
                                                  const T& x,
                                                  const T& bias,
                                                  const T& mean,
                                                  const T& variance,
                                                  const T& gamma,
                                                  const T& beta) const
    {
        using ck::type_convert;
        using ck::math::sqrt;

        float tmp_x = type_convert<float>(x) + type_convert<float>(bias);

        float tmp_y =
            ((tmp_x - type_convert<float>(mean)) / sqrt(type_convert<float>(variance) + epsilon_)) *
                type_convert<float>(gamma) +
            type_convert<float>(beta);
        clamp_(tmp_y, tmp_y);
        y = type_convert<T>(tmp_y);
    };

    template <>
    __host__ __device__ constexpr void operator()(float& y,
                                                  const float& x,
                                                  const float& bias,
                                                  const float& mean,
                                                  const float& variance,
                                                  const float& gamma,
                                                  const float& beta) const
    {
        using ck::type_convert;
        using ck::math::sqrt;

        float tmp_y = (((x + bias) - mean) / sqrt(variance + epsilon_)) * gamma + beta;
        clamp_(y, tmp_y);
    };

    Clamp clamp_;
    float epsilon_;
};

template <typename Y, typename X>
struct UnaryTypeConvert;

template <>
struct UnaryTypeConvert<float, ck::bhalf_t>
{
    static constexpr const char* name = "UnaryTypeConvert";

    __host__ __device__ void operator()(float& y, ck::bhalf_t& x) const
    {
        y = ck::type_convert<float, ck::bhalf_t>(x);
    }
};

template <>
struct UnaryTypeConvert<ck::bhalf_t, float>
{
    static constexpr const char* name = "UnaryTypeConvert";

    __host__ __device__ void operator()(ck::bhalf_t& y, float& x) const
    {
        y = ck::type_convert<ck::bhalf_t, float>(x);
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
