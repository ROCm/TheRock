// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "data_type.hpp"
#include "dtype_fp64.hpp"

namespace ck {

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_add explicit for
// each datatype.
template <typename X>
__device__ X atomic_add(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_add<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ uint32_t atomic_add<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float atomic_add<float>(float* p_dst, const float& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ unsigned short atomic_add<unsigned short>(unsigned short* p_dst, const unsigned short& x)
{
    // Use atomicAdd with unsigned int
    return static_cast<unsigned short>(
        atomicAdd(reinterpret_cast<unsigned int*>(p_dst), static_cast<unsigned int>(x)));
}

template <>
__device__ _Float16 atomic_add<_Float16>(_Float16* p_dst, const _Float16& x)
{
    // Use atomicAdd with unsigned int
    return static_cast<_Float16>(
        atomicAdd(reinterpret_cast<unsigned int*>(p_dst), static_cast<unsigned int>(x)));
}

template <>
__device__ double atomic_add<double>(double* p_dst, const double& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float2_t atomic_add<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

template <>
__device__ float4_t atomic_add<float4_t>(float4_t* p_dst, const float4_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const vector_type<float, 4> vx{x};
    vector_type<float, 4> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);
    vy.template AsType<float>()(I2) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 2, vx.template AsType<float>()[I2]);
    vy.template AsType<float>()(I3) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 3, vx.template AsType<float>()[I3]);

    return vy.template AsType<float4_t>()[I0];
}

template <>
__device__ double2_t atomic_add<double2_t>(double2_t* p_dst, const double2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<double, 2> vx{x};
    vector_type<double, 2> vy{0};

    vy.template AsType<double>()(I0) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst), vx.template AsType<double>()[I0]);
    vy.template AsType<double>()(I1) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst) + 1, vx.template AsType<double>()[I1]);

    return vy.template AsType<double2_t>()[I0];
}

#if defined(__gfx11__)
template <>
__device__ float8_t atomic_add<float8_t>(float8_t* p_dst, const float8_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};

    const vector_type<float, 8> vx{x};
    vector_type<float, 8> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);
    vy.template AsType<float>()(I2) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 2, vx.template AsType<float>()[I2]);
    vy.template AsType<float>()(I3) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 3, vx.template AsType<float>()[I3]);
    vy.template AsType<float>()(I4) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 4, vx.template AsType<float>()[I4]);
    vy.template AsType<float>()(I5) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 5, vx.template AsType<float>()[I5]);
    vy.template AsType<float>()(I6) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 6, vx.template AsType<float>()[I6]);
    vy.template AsType<float>()(I7) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 7, vx.template AsType<float>()[I7]);

    return vy.template AsType<float8_t>()[I0];
}

template <>
__device__ half4_t atomic_add<half4_t>(half4_t* p_dst, const half4_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const vector_type<half_t, 4> vx{x};
    vector_type<half_t, 4> vy{0};

    vy.template AsType<half_t>()(I0) =
        atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst), vx.template AsType<half_t>()[I0]);
    vy.template AsType<half_t>()(I1) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 1,
                                                          vx.template AsType<half_t>()[I1]);
    vy.template AsType<half_t>()(I2) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 2,
                                                          vx.template AsType<half_t>()[I2]);
    vy.template AsType<half_t>()(I3) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 3,
                                                          vx.template AsType<half_t>()[I3]);

    return vy.template AsType<half4_t>()[I0];
}

template <>
__device__ half8_t atomic_add<half8_t>(half8_t* p_dst, const half8_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};

    const vector_type<half_t, 8> vx{x};
    vector_type<half_t, 8> vy{0};

    vy.template AsType<half_t>()(I0) =
        atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst), vx.template AsType<half_t>()[I0]);
    vy.template AsType<half_t>()(I1) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 1,
                                                          vx.template AsType<half_t>()[I1]);
    vy.template AsType<half_t>()(I2) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 2,
                                                          vx.template AsType<half_t>()[I2]);
    vy.template AsType<half_t>()(I3) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 3,
                                                          vx.template AsType<half_t>()[I3]);
    vy.template AsType<half_t>()(I4) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 4,
                                                          vx.template AsType<half_t>()[I4]);
    vy.template AsType<half_t>()(I5) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 5,
                                                          vx.template AsType<half_t>()[I5]);
    vy.template AsType<half_t>()(I6) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 6,
                                                          vx.template AsType<half_t>()[I6]);
    vy.template AsType<half_t>()(I7) = atomic_add<half_t>(c_style_pointer_cast<half_t*>(p_dst) + 7,
                                                          vx.template AsType<half_t>()[I7]);

    return vy.template AsType<half8_t>()[I0];
}
#endif // defined(__gfx11__)

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_max explicit for
// each datatype.

template <typename X>
__device__ X atomic_max(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_max<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ uint32_t atomic_max<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float atomic_max<float>(float* p_dst, const float& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ double atomic_max<double>(double* p_dst, const double& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float2_t atomic_max<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicMax(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicMax(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

} // namespace ck
