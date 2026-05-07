// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_AMD_WMMA_HPP
#define CK_AMD_WMMA_HPP

#include "ck/utility/amd_inline_asm.hpp"
#include "data_type.hpp"
// TODO: Add arch limitation
namespace ck {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx11_generic__)
#define __gfx11__
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx12__
#endif

/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half16_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            reg_a, reg_b, reg_c.template AsType<half16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf16_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                reg_a, reg_b, reg_c.template AsType<bhalf16_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

/********************************WAVE64 MODE***********************************************/

template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w64;

template <>
struct intrin_wmma_f32_16x16x16_f16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w64;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<float4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: fp16, dst: fp16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64;

template <index_t Opsel>
struct intrin_wmma_f16_16x16x16_f16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const half16_t& reg_a, const half16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<half8_t>()(Number<0>{}) = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(
            reg_a, reg_b, reg_c.template AsType<half8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: bf16
template <index_t MPerWave, index_t NPerWave, index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64;

template <index_t Opsel>
struct intrin_wmma_bf16_16x16x16_bf16_w64<16, 16, Opsel>
{
    template <class FloatC>
    __device__ static void Run(const bhalf16_t& reg_a, const bhalf16_t& reg_b, FloatC& reg_c)
    {
        // opsel usage
        // false: D0.[0:15] = result
        // true : D0.[16:31]= result
#if defined(__gfx11__)
        reg_c.template AsType<bhalf8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64(
                reg_a, reg_b, reg_c.template AsType<bhalf8_t>()[Number<0>{}], Opsel);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w64<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x16_t& reg_a, const int8x16_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx11__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(
                neg_a,
                bit_cast<int32x4_t>(reg_a),
                neg_b,
                bit_cast<int32x4_t>(reg_b),
                reg_c.template AsType<int32x4_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// gfx12
/********************************WAVE32 MODE***********************************************/

// src: fp16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half8_t& reg_a, const half8_t& reg_b, FloatC& reg_c)
    {
        // * Inline assembly need to elimate the duplicated data load, compiler won't help you
        // delete them.
        // amd_assembly_wmma_f32_16x16x16_f16_w32(
        //     reg_a, reg_b, reg_c.template AsType<float8_t>()(Number<0>{}));
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf16, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf16_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf8_t& reg_a, const bhalf8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                reg_a, reg_b, reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: iu8, dst: i32
template <index_t MPerWave, index_t NPerWave, bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12;

template <bool neg_a, bool neg_b, bool clamp>
struct intrin_wmma_i32_16x16x16_iu8_w32_gfx12<16, 16, neg_a, neg_b, clamp>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<int32x8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                neg_a,
                bit_cast<int32x2_t>(reg_a),
                neg_b,
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<int32x8_t>()[Number<0>{}],
                clamp);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: f8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const f8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, f8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const f8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

// src: bf8, bf8, dst: fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12;

template <>
struct intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bf8x8_t& reg_a, const bf8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx12__)
        reg_c.template AsType<float8_t>()(Number<0>{}) =
            __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(reg_a),
                bit_cast<int32x2_t>(reg_b),
                reg_c.template AsType<float8_t>()[Number<0>{}]);
#else
        ignore = reg_a;
        ignore = reg_b;
        ignore = reg_c;
#endif
    }
};

} // namespace ck
#endif
