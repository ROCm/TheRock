// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
// #include "ck/utility/get_id.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// Y = Sy * Qy
// W = Sw * Qw
// X = Sx * Qx
// B = Sb * Qb = Sw * Sx * Qb
// Where X, W, Y are float32, Qx, Qw, Qy are int8
// Sx, Sw, Sy are scale of x, w, y (float32), which is calculated from quantization range
// Qb is int32, scale of B is Sw * Sx for convenient

// Y = W @ X, where @ is convolution or matrix multiplication
// Sy * Qy = Sw * Qw @ Sx * Qx
// Qy = [(Sw*Sx)/Sy] * Qw @ Qx

// For Activation function which is piecewise linear function, such as relu, leaky relu ...etc
// Activation(Sy * Qy) = Sy * Activation(Qy)
template <typename Activation>
struct Activation_Mul_Clamp
{
    static constexpr const char* name = "Activation_Mul_Clamp";

    // Convolution + Activation (piecewise linear function)
    // If an activation is piecewise linear function, then Activation(Sy * Qy) = Sy * Activation(Qy)
    // Z = Activation(Y) = Activation(W @ X)
    // Sz * Qz = Activation(Sy * Qy)
    // Qz = Sy / Sz * Activation(Qy) = (Sw * Sx / Sz) * Activation(Qw @ Qx)

    // requantScale_ = Sw * Sx / Sz
    Activation_Mul_Clamp(float requantScale, Activation activationOp)
        : requantScale_(requantScale), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void operator()(int8_t& y, const int32_t& x) const
    {
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __device__ constexpr void operator()(int32_t& y, const int32_t& x) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    __host__ constexpr void operator()(float& y, const float& x) const
    {
        // CAUSION - We might float in & float out in reference code
        activationOp_(y, x);
        y = math::clamp(requantScale_ * y, -128.f, 127.f);
    }

    float requantScale_;
    Activation activationOp_;
};

// For Activation function which is non piecewise linear function, such as TanH, Sigmoid ...etc
// If an activation is not piecewise linear function
// then Activation(Sy * Qy) != Sy * Activation(Qy)
template <typename Activation>
struct Mul_Activation_Mul_Clamp
{
    static constexpr const char* name = "Mul_Activation_Mul_Clamp";

    // Convolution + Activation (non piecewise linear function)
    // Z = Activation(Y) = Activation(W @ X)
    // Sz * Qz = Activation(Sy * Qy)
    // Qz = S1 * Activation[Sacc * (Qw @ Qx)]
    // Where S1 = 1 / Sz, Sacc = Sw * Sx
    Mul_Activation_Mul_Clamp(float scale_z_inv, float scaleAcc, Activation activationOp)
        : scale_z_inv_(scale_z_inv), scaleAcc_(scaleAcc), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void operator()(int8_t& y, const int32_t& x) const
    {
        float y_fp32 = ck::type_convert<float>(x);
        y_fp32       = scaleAcc_ * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(scale_z_inv_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    float scale_z_inv_;
    float scaleAcc_;
    Activation activationOp_;
};

// Conv Perchannel quantization + Activation function which is piecewise linear function, such as
// relu, leaky relu ...etc
// Activation(Sy * Qy) = Sy * Activation(Qy)
template <typename Activation>
struct Activation_Mul2_Clamp
{
    static constexpr const char* name = "Activation_Mul2_Clamp";

    Activation_Mul2_Clamp(Activation activationOp) : activationOp_(activationOp) {}

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const float& requantScale) const
    {
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const float& requantScale) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    Activation activationOp_;
};

// For Activation function which is piecewise linear function, such as relu, leaky relu ...etc
// Activation(Sy * Qy) = Sy * Activation(Qy)
template <typename Activation>
struct Add_Activation_Mul_Clamp
{
    static constexpr const char* name = "Add_Activation_Mul_Clamp";

    // Convolution + bias
    // Let Bias = B = Sw * Sx * Qb
    // Where Qb is int32
    // Y = W @ X + B
    // Sy * Qy = Sw * Qw @ Sx * Qx + Sw * Sx * Qb
    // Qy = [(Sw*Sx)/Sy] * (Qw @ Qx + Qb)

    // For activation, Z = Activaiton(Y)
    // Sz * Qz = Activation(Sy * Qy)
    // Qz = Sy / Sz * Activation(Qy) = [(Sw*Sx)/Sz] * Activation(Qw @ Qx + Qb)
    Add_Activation_Mul_Clamp(float requantScale, Activation activationOp)
        : requantScale_(requantScale), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const int32_t& bias) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    float requantScale_;
    Activation activationOp_;
};

// Conv Perchannel quantization + Activation function which is piecewise linear function, such as
// relu, leaky relu ...etc
template <typename Activation>
struct Add_Activation_Mul2_Clamp
{
    static constexpr const char* name = "Add_Activation_Mul2_Clamp";

    Add_Activation_Mul2_Clamp(Activation activationOp) : activationOp_(activationOp) {}

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias, const float& requantScale) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const int32_t& bias, const float& requantScale) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    Activation activationOp_;
};

// For Activation function which is non piecewise linear function, such as TanH, Sigmoid ...etc
// If an activation is not piecewise linear function
// then Activation(Sy * Qy) != Sy * Activation(Qy)
template <typename Activation>
struct Add_Mul_Activation_Mul_Clamp
{
    static constexpr const char* name = "Add_Mul_Activation_Mul_Clamp";

    // Convolution + Activation (non piecewise linear function)
    // Z = Activation(Y) = Activation(W @ X + B)
    // Sz * Qz = Activation(Sy * Qy)
    // Qz = S1 * Activation[Sacc * (Qw @ Qx + Qb)]
    // Where S1 = 1 / Sz, Sacc = Sw * Sx
    Add_Mul_Activation_Mul_Clamp(float scale_z_inv, float scaleAcc, Activation activationOp)
        : scale_z_inv_(scale_z_inv), scaleAcc_(scaleAcc), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        y_fp32       = scaleAcc_ * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(scale_z_inv_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const int32_t& bias) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x + bias);
        y_fp32       = scaleAcc_ * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(scale_z_inv_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    float scale_z_inv_;
    float scaleAcc_;
    Activation activationOp_;
};

// Conv Perchannel quantization + Activation function which is non piecewise linear function,
// such as TanH, Sigmoid ...etc
// If an activation is not piecewise linear function
// then Activation(Sy *Qy) != Sy * Activation(Qy)
template <typename Activation>
struct Add_Mul2_Activation_Mul_Clamp
{
    static constexpr const char* name = "Add_Mul2_Activation_Mul_Clamp";

    Add_Mul2_Activation_Mul_Clamp(float scale_z_inv, Activation activationOp)
        : scale_z_inv_(scale_z_inv), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias, const float& scaleAcc) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        y_fp32       = scaleAcc * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(scale_z_inv_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const int32_t& bias, const float& scaleAcc) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x + bias);
        y_fp32       = scaleAcc * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(scale_z_inv_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    float scale_z_inv_;
    Activation activationOp_;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
