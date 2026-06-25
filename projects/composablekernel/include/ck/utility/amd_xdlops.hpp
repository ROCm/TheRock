// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/utility/dtype_fp64.hpp"

namespace ck {
// Define the common macro for MI300 models
#if defined(__gfx942__) || defined(__gfx950__)
#define __gfx94__
#endif

// Helper function to convert float vector to bf16 vectors (big and small parts)
// This is used by both tf32 and xf32 implementations
template <index_t VecSize>
__device__ __forceinline__ void
convert_float_to_bf16_pairs(const vector_type<float, VecSize>& reg_f32,
                            vector_type<bhalf_t, VecSize>& reg_bf16_big,
                            vector_type<bhalf_t, VecSize>& reg_bf16_small)
{
    static_for<0, VecSize, 1>{}([&](auto k) {
        using IK = Number<k>;
        reg_bf16_big.template AsType<bhalf_t>()(k) =
            type_convert<bhalf_t, float>(reg_f32.template AsType<float>()[IK{}]);
        reg_bf16_small.template AsType<bhalf_t>()(k) = type_convert<bhalf_t, float>(
            reg_f32.template AsType<float>()[IK{}] -
            type_convert<float, bhalf_t>(reg_bf16_big.template AsType<bhalf_t>()[IK{}]));
    });
}
/* */

// fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x1f32;

template <>
struct intrin_mfma_f32_32x32x1f32<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
    }
};

template <>
struct intrin_mfma_f32_32x32x1f32<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x2f32;

template <>
struct intrin_mfma_f32_32x32x2f32<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x2f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f32;

template <>
struct intrin_mfma_f32_16x16x4f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x4f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x1f32;

template <>
struct intrin_mfma_f32_16x16x1f32<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x1f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x1f32;

template <>
struct intrin_mfma_f32_4x4x1f32<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
    }
};

template <>
struct intrin_mfma_f32_4x4x1f32<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
    }
};

// fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4f16;

template <>
struct intrin_mfma_f32_32x32x4f16<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
    }
};

template <>
struct intrin_mfma_f32_32x32x4f16<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16f16;

template <>
struct intrin_mfma_f32_32x32x16f16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x16_f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32f16;

template <>
struct intrin_mfma_f32_16x16x32f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8f16;

template <>
struct intrin_mfma_f32_32x32x8f16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x8f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16f16;

template <>
struct intrin_mfma_f32_16x16x16f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x16f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f16;

template <>
struct intrin_mfma_f32_16x16x4f16<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x4f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x4f16;

template <>
struct intrin_mfma_f32_4x4x4f16<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
    }
};

template <>
struct intrin_mfma_f32_4x4x4f16<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
    }
};

// bfp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16bf16;

template <>
struct intrin_mfma_f32_32x32x16bf16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32bf16;

template <>
struct intrin_mfma_f32_16x16x32bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8bf16_1k;

template <>
struct intrin_mfma_f32_32x32x8bf16_1k<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16bf16_1k;

template <>
struct intrin_mfma_f32_16x16x16bf16_1k<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4bf16;

template <>
struct intrin_mfma_f32_32x32x4bf16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4bf16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x8bf16;

template <>
struct intrin_mfma_f32_16x16x8bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x8bf16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x8i8;

template <>
struct intrin_mfma_i32_32x32x8i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_a, const int8x4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_32x32x8i8(bit_cast<int32_t>(reg_a),
                                                bit_cast<int32_t>(reg_b),
                                                reg_c.template AsType<int32x16_t>()[Number<0>{}],
                                                0,
                                                0,
                                                0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x16i8;

template <>
struct intrin_mfma_i32_16x16x16i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_a, const int8x4_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_16x16x16i8(bit_cast<int32_t>(reg_a),
                                                 bit_cast<int32_t>(reg_b),
                                                 reg_c.template AsType<int32x4_t>()[Number<0>{}],
                                                 0,
                                                 0,
                                                 0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x32i8;

template <>
struct intrin_mfma_i32_32x32x32i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<int32x16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_i32_32x32x32_i8(
            reg_a, reg_b, reg_c.template AsType<int32x16_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x64i8;

template <>
struct intrin_mfma_i32_16x16x64i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_i32_16x16x64_i8(
            reg_a, reg_b, reg_c.template AsType<int32x4_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x16i8;

template <>
struct intrin_mfma_i32_32x32x16i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_32x32x16_i8(bit_cast<int64_t>(reg_a),
                                                  bit_cast<int64_t>(reg_b),
                                                  reg_c.template AsType<int32x16_t>()[Number<0>{}],
                                                  0,
                                                  0,
                                                  0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x32i8;

template <>
struct intrin_mfma_i32_16x16x32i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_16x16x32_i8(bit_cast<int64_t>(reg_a),
                                                  bit_cast<int64_t>(reg_b),
                                                  reg_c.template AsType<int32x4_t>()[Number<0>{}],
                                                  0,
                                                  0,
                                                  0);
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f64_16x16x4f64;

template <>
struct intrin_mfma_f64_16x16x4f64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const double& reg_a, const double& reg_b, FloatC& reg_c)
    {
#if defined(__gfx90a__) || defined(__gfx94__)
        reg_c.template AsType<double4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f64_16x16x4f64(
            reg_a, reg_b, reg_c.template AsType<double4_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x64f8f6f4;

/// @brief Performs a matrix fused multiply-accumulate operation on 32x32x64 submatrices for f8, f6,
/// and f4 data types.
///
/// @note Calls scaled version of the instruction as the original instruction is not supported in
/// the backend. That is the intended use. There is a backend optimization to select the unscaled
/// operation if the scale is 0.
template <>
struct intrin_mfma_f32_32x32x64f8f6f4<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x32_t& reg_a, const f4x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)

        int32x4_t arg_a = bit_cast<int32x4_t>(reg_a);
        int32x4_t arg_b = bit_cast<int32x4_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
                reg_c.template AsType<float16_t>()[Number<0>{}],
                4, // cbsz
                4, // blgp
                0, // OPSEL
                0,
                0, // OPSEL
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x32_t& reg_a, const f6x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)

        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float16_t>()[Number<0>{}],
                2, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                2, // blgp
                0, // OPSEL
                0,
                0, // OPSEL
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x32_t& reg_a, const bf6x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)

        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float16_t>()[Number<0>{}],
                3, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                3, // blgp
                0, // OPSEL
                0,
                0, // OPSEL
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, index_t OpselA, index_t OpselB>
struct intrin_mfma_scale_f32_32x32x64f8f6f4;

template <index_t OpselA, index_t OpselB>
struct intrin_mfma_scale_f32_32x32x64f8f6f4<32, 32, OpselA, OpselB>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a,
                               const int32_t& scale_a,
                               const f8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
        // XXX: Note on the scale_a and scale_b parameters:
        // If compiler detects that one or both scales are constant values, it will treat that
        // constant as F32 constant. I.e., if scale_a at some point was declared as
        // `e8m0_bexp_t a_scale{1.0f}`, the instruction would only work if scale_a parameter is
        // assigned value `bit_cast<int32_t>(static_cast<float>(a_scale))`.

        // XXX: Note on the OPSEL parameters: Instruction always takes byte0 as a scale value even
        // when OPSEL is set otherwise.
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a,
                               const int32_t& scale_a,
                               const bf8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
        // XXX: Note on the scale_a and scale_b parameters:
        // If compiler detects that one or both scales are constant values, it will treat that
        // constant as F32 constant. I.e., if scale_a at some point was declared as
        // `e8m0_bexp_t a_scale{1.0f}`, the instruction would only work if scale_a parameter is
        // assigned value `bit_cast<int32_t>(static_cast<float>(a_scale))`.

        // XXX: Note on the OPSEL parameters: Instruction always takes byte0 as a scale value even
        // when OPSEL is set otherwise.
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a,
                               const int32_t& scale_a,
                               const f8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float16_t>()[Number<0>{}],
                1,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
        // XXX: Note on the scale_a and scale_b parameters:
        // If compiler detects that one or both scales are constant values, it will treat that
        // constant as F32 constant. I.e., if scale_a at some point was declared as
        // `e8m0_bexp_t a_scale{1.0f}`, the instruction would only work if scale_a parameter is
        // assigned value `bit_cast<int32_t>(static_cast<float>(a_scale))`.

        // XXX: Note on the OPSEL parameters: Instruction always takes byte0 as a scale value even
        // when OPSEL is set otherwise.
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x32_t& reg_a,
                               const int32_t scale_a,
                               const f6x32_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)

        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float16_t>()[Number<0>{}],
                2,      // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                2,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x32_t& reg_a,
                               const int32_t scale_a,
                               const bf6x32_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)

        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float16_t>()[Number<0>{}],
                3,      // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                3,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x32_t& reg_a,
                               const int32_t scale_a,
                               const f4x32_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)

        int32x4_t arg_a = bit_cast<int32x4_t>(reg_a);
        int32x4_t arg_b = bit_cast<int32x4_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
                reg_c.template AsType<float16_t>()[Number<0>{}],
                4,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                4,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave, index_t OpselA, index_t OpselB>
struct intrin_mfma_scale_f32_16x16x128f8f6f4;

template <index_t OpselA, index_t OpselB>
struct intrin_mfma_scale_f32_16x16x128f8f6f4<16, 16, OpselA, OpselB>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a,
                               const int32_t& scale_a,
                               const f8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                0,      // cbsz   {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a,
                               const int32_t& scale_a,
                               const bf8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                1,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a,
                               const int32_t& scale_a,
                               const bf8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                0,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a,
                               const int32_t& scale_a,
                               const f8x32_t& reg_b,
                               const int32_t& scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                1,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x32_t& reg_a,
                               const int32_t scale_a,
                               const f6x32_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float4_t>()[Number<0>{}],
                2,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                2,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x16x2_t& reg_a,
                               const int32_t scale_a,
                               const f6x16x2_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        using arg_type = int32x8_t;
        arg_type arg_a{
            static_cast<int32_t>(reg_a.template AsType<f6x16x2_t::data_t>()[Number<0>{}][0]),
            static_cast<int32_t>(reg_a.template AsType<f6x16x2_t::data_t>()[Number<0>{}][1]),
            static_cast<int32_t>(reg_a.template AsType<f6x16x2_t::data_t>()[Number<0>{}][2]),
            static_cast<int32_t>(reg_a.template AsType<f6x16x2_t::data_t>()[Number<1>{}][0]),
            static_cast<int32_t>(reg_a.template AsType<f6x16x2_t::data_t>()[Number<1>{}][1]),
            static_cast<int32_t>(reg_a.template AsType<f6x16x2_t::data_t>()[Number<1>{}][2]),
            0,
            0};
        arg_type arg_b{
            static_cast<int32_t>(reg_b.template AsType<f6x16x2_t::data_t>()[Number<0>{}][0]),
            static_cast<int32_t>(reg_b.template AsType<f6x16x2_t::data_t>()[Number<0>{}][1]),
            static_cast<int32_t>(reg_b.template AsType<f6x16x2_t::data_t>()[Number<0>{}][2]),
            static_cast<int32_t>(reg_b.template AsType<f6x16x2_t::data_t>()[Number<1>{}][0]),
            static_cast<int32_t>(reg_b.template AsType<f6x16x2_t::data_t>()[Number<1>{}][1]),
            static_cast<int32_t>(reg_b.template AsType<f6x16x2_t::data_t>()[Number<1>{}][2]),
            0,
            0};

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_a,
                arg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                2,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                2,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x32_t& reg_a,
                               const int32_t scale_a,
                               const bf6x32_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float4_t>()[Number<0>{}],
                3,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                3,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x16x2_t& reg_a,
                               const int32_t scale_a,
                               const bf6x16x2_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        using arg_type = int32x8_t;
        arg_type arg_a{
            static_cast<int32_t>(reg_a.template AsType<bf6x16x2_t::data_t>()[Number<0>{}][0]),
            static_cast<int32_t>(reg_a.template AsType<bf6x16x2_t::data_t>()[Number<0>{}][1]),
            static_cast<int32_t>(reg_a.template AsType<bf6x16x2_t::data_t>()[Number<0>{}][2]),
            static_cast<int32_t>(reg_a.template AsType<bf6x16x2_t::data_t>()[Number<1>{}][0]),
            static_cast<int32_t>(reg_a.template AsType<bf6x16x2_t::data_t>()[Number<1>{}][1]),
            static_cast<int32_t>(reg_a.template AsType<bf6x16x2_t::data_t>()[Number<1>{}][2]),
            0,
            0};
        arg_type arg_b{
            static_cast<int32_t>(reg_b.template AsType<bf6x16x2_t::data_t>()[Number<0>{}][0]),
            static_cast<int32_t>(reg_b.template AsType<bf6x16x2_t::data_t>()[Number<0>{}][1]),
            static_cast<int32_t>(reg_b.template AsType<bf6x16x2_t::data_t>()[Number<0>{}][2]),
            static_cast<int32_t>(reg_b.template AsType<bf6x16x2_t::data_t>()[Number<1>{}][0]),
            static_cast<int32_t>(reg_b.template AsType<bf6x16x2_t::data_t>()[Number<1>{}][1]),
            static_cast<int32_t>(reg_b.template AsType<bf6x16x2_t::data_t>()[Number<1>{}][2]),
            0,
            0};

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_a,
                arg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                3,      // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                3,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x32_t& reg_a,
                               const int32_t scale_a,
                               const f4x32_t& reg_b,
                               const int32_t scale_b,
                               FloatC& reg_c)
    {
#if defined(__gfx950__)
        int32x4_t arg_a = bit_cast<int32x4_t>(reg_a);
        int32x4_t arg_b = bit_cast<int32x4_t>(reg_b);
        using arg_type  = int32x8_t;
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
                reg_c.template AsType<float4_t>()[Number<0>{}],
                4,      // cbsz
                4,      // blgp
                OpselA, // OPSEL
                scale_a,
                OpselB, // OPSEL
                scale_b);
#else
        ignore = reg_a;
        ignore = scale_a;
        ignore = reg_b;
        ignore = scale_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x128f8f6f4;

/// @brief Performs a matrix fused multiply-accumulate operation on 16x16x128 submatrices for f8f6f4
/// data types.
///
/// @note Calls scaled version of the instruction as the original instruction is not supported in
/// the backend. That is the intended use. There is a backend optimization to select the unscaled
/// operation if the scale is 0.
template <>
struct intrin_mfma_f32_16x16x128f8f6f4<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                0, // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                1, // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf8x32_t& reg_a, const f8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                1, // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                0, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f8x32_t& reg_a, const bf8x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        // https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/llvm/test/Verifier/AMDGPU/mfma-scale.ll#L10
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                reg_a,
                reg_b,
                reg_c.template AsType<float4_t>()[Number<0>{}],
                0, // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                1, // blgp
                0,
                0,
                0,
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f4x32_t& reg_a, const f4x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        int32x4_t arg_a = bit_cast<int32x4_t>(reg_a);
        int32x4_t arg_b = bit_cast<int32x4_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], 0, 0, 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], 0, 0, 0, 0},
                reg_c.template AsType<float4_t>()[Number<0>{}],
                4, // cbsz
                4, // blgp
                0, // OPSEL
                0,
                0, // OPSEL
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const f6x32_t& reg_a, const f6x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float4_t>()[Number<0>{}],
                2, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                2, // blgp
                0, // OPSEL
                0,
                0, // OPSEL
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }

    template <class FloatC>
    __device__ static void Run(const bf6x32_t& reg_a, const bf6x32_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        int32x6_t arg_a = bit_cast<int32x6_t>(reg_a);
        int32x6_t arg_b = bit_cast<int32x6_t>(reg_b);

        using arg_type = int32x8_t;

        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                arg_type{arg_a[0], arg_a[1], arg_a[2], arg_a[3], arg_a[4], arg_a[5], 0, 0},
                arg_type{arg_b[0], arg_b[1], arg_b[2], arg_b[3], arg_b[4], arg_b[5], 0, 0},
                reg_c.template AsType<float4_t>()[Number<0>{}],
                3, // cbsz  {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
                3, // blgp
                0, // OPSEL
                0,
                0, // OPSEL
                0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16f8f8;

template <>
struct intrin_mfma_f32_32x32x16f8f8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                bit_cast<int64_t>(reg_a),
                bit_cast<int64_t>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32f8f8;

template <>
struct intrin_mfma_f32_16x16x32f8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            bit_cast<int64_t>(reg_a),
            bit_cast<int64_t>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16bf8bf8;

template <>
struct intrin_mfma_f32_32x32x16bf8bf8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                bit_cast<int64_t>(reg_a),
                bit_cast<int64_t>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32bf8bf8;

template <>
struct intrin_mfma_f32_16x16x32bf8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
            bit_cast<int64_t>(reg_a),
            bit_cast<int64_t>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16f8bf8;

template <>
struct intrin_mfma_f32_32x32x16f8bf8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                bit_cast<int64_t>(reg_a),
                bit_cast<int64_t>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32f8bf8;

template <>
struct intrin_mfma_f32_16x16x32f8bf8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
            bit_cast<int64_t>(reg_a),
            bit_cast<int64_t>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<f8_t, 8> reg_a_v(reg_a);
        vector_type<bf8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<f8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<bf8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16bf8f8;

template <>
struct intrin_mfma_f32_32x32x16bf8f8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                bit_cast<int64_t>(reg_a),
                bit_cast<int64_t>(reg_b),
                reg_c.template AsType<float16_t>()[Number<0>{}],
                0,
                0,
                0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_32x32x2f32<32, 32>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32bf8f8;

template <>
struct intrin_mfma_f32_16x16x32bf8f8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx94__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
            bit_cast<int64_t>(reg_a),
            bit_cast<int64_t>(reg_b),
            reg_c.template AsType<float4_t>()[Number<0>{}],
            0,
            0,
            0);
#else
        vector_type<bf8_t, 8> reg_a_v(reg_a);
        vector_type<f8_t, 8> reg_b_v(reg_b);

        static_for<0, 8, 1>{}([&](auto k) {
            float reg_a_f32 = type_convert<float>(reg_a_v.template AsType<bf8_t>()[Number<k>{}]);
            float reg_b_f32 = type_convert<float>(reg_b_v.template AsType<f8_t>()[Number<k>{}]);

            intrin_mfma_f32_16x16x4f32<16, 16>::Run(reg_a_f32, reg_b_f32, reg_c);
        });
#endif
    }
};

/******************* tf32 on gfx942 *************************************/
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x8xf32;

template <>
struct intrin_mfma_f32_16x16x8xf32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float2_t& reg_a, const float2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx942__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x8_xf32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4xf32;

template <>
struct intrin_mfma_f32_32x32x4xf32<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const float2_t& reg_a, const float2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx942__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4_xf32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

/******************* tf32/xf32 on gfx950 ********************************/
/* bf16x3 simulate tf32/xf32: input/output/accumulator are all float;   */
/* step:                                                                */
/*  1. separate one input to 2 bf16 registers:                          */
/*      in_bf16_big = f32_to_bf16(in_f32)                               */
/*      in_bf16_small = in_f32 - in_bf16_big                            */
/*  2. run 3 xdlops gemm: the accumulator of each gemm is the same.     */
/*      out_f32 = A_bf16_big * B_bf16_big                               */
/*      out_f32 += A_bf16_small * B_bf16_big                            */
/*      out_f32 += A_bf16_big * B_bf16_small                            */
/************************************************************************/
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x32xf32;

template <>
struct intrin_mfma_f32_16x16x32xf32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float8_t& reg_a, const float8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        using I0 = Number<0>;
        vector_type<float, 8> reg_a_v(reg_a);
        vector_type<float, 8> reg_b_v(reg_b);

        vector_type<bhalf_t, 8> v_reg_a_bf16_big;
        vector_type<bhalf_t, 8> v_reg_a_bf16_small;
        vector_type<bhalf_t, 8> v_reg_b_bf16_big;
        vector_type<bhalf_t, 8> v_reg_b_bf16_small;

        convert_float_to_bf16_pairs(reg_a_v, v_reg_a_bf16_big, v_reg_a_bf16_small);
        convert_float_to_bf16_pairs(reg_b_v, v_reg_b_bf16_big, v_reg_b_bf16_small);

        // Run 3 times: big*big, small*big, big*small
        intrin_mfma_f32_16x16x32bf16<16, 16>::Run(
            v_reg_a_bf16_small.template AsType<bhalf8_t>()[I0{}],
            v_reg_b_bf16_big.template AsType<bhalf8_t>()[I0{}],
            reg_c);
        intrin_mfma_f32_16x16x32bf16<16, 16>::Run(
            v_reg_a_bf16_big.template AsType<bhalf8_t>()[I0{}],
            v_reg_b_bf16_small.template AsType<bhalf8_t>()[I0{}],
            reg_c);
        intrin_mfma_f32_16x16x32bf16<16, 16>::Run(
            v_reg_a_bf16_big.template AsType<bhalf8_t>()[I0{}],
            v_reg_b_bf16_big.template AsType<bhalf8_t>()[I0{}],
            reg_c);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x16xf32;

template <>
struct intrin_mfma_f32_32x32x16xf32<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const float8_t& reg_a, const float8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx950__)
        using I0 = Number<0>;
        vector_type<float, 8> reg_a_v(reg_a);
        vector_type<float, 8> reg_b_v(reg_b);

        vector_type<bhalf_t, 8> v_reg_a_bf16_big;
        vector_type<bhalf_t, 8> v_reg_a_bf16_small;
        vector_type<bhalf_t, 8> v_reg_b_bf16_big;
        vector_type<bhalf_t, 8> v_reg_b_bf16_small;

        convert_float_to_bf16_pairs(reg_a_v, v_reg_a_bf16_big, v_reg_a_bf16_small);
        convert_float_to_bf16_pairs(reg_b_v, v_reg_b_bf16_big, v_reg_b_bf16_small);

        // Run 3 times: big*big, small*big, big*small
        intrin_mfma_f32_32x32x16bf16<32, 32>::Run(
            v_reg_a_bf16_small.template AsType<bhalf8_t>()[I0{}],
            v_reg_b_bf16_big.template AsType<bhalf8_t>()[I0{}],
            reg_c);
        intrin_mfma_f32_32x32x16bf16<32, 32>::Run(
            v_reg_a_bf16_big.template AsType<bhalf8_t>()[I0{}],
            v_reg_b_bf16_small.template AsType<bhalf8_t>()[I0{}],
            reg_c);
        intrin_mfma_f32_32x32x16bf16<32, 32>::Run(
            v_reg_a_bf16_big.template AsType<bhalf8_t>()[I0{}],
            v_reg_b_bf16_big.template AsType<bhalf8_t>()[I0{}],
            reg_c);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif // defined(__gfx950__)
    }
};

/******************* tf32/xf32 on gfx950 end ************************************/
} // namespace ck
