// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_xdlops.hpp"
#include "ck/utility/amd_wmma.hpp"

namespace ck {
/**
 * @brief Define matrix data types that have hardware support for MX GEMMs
 */
template <typename T>
static constexpr bool is_scale_mfma_data_type()
{
    using U = element_type_t<T>;
    return is_same_v<U, f8_ocp_t> || is_same_v<U, bf8_ocp_t> || is_same_v<U, f6_t> ||
           is_same_v<U, bf6_t> || is_same_v<U, f4_t>;
}

#ifndef CK_CODE_GEN_RTC
/**
 * @brief Define scale data types that have hardware support for MX GEMMs
 */
template <typename T>
static constexpr bool is_scale_mfma_scale_type()
{
    return is_same_v<T, e8m0_bexp_t>;
}
#endif

/**
 * @brief Combination of data types that have hardware support for MX GEMMs
 */
template <typename ADataType, typename BDataType, typename AScaleDataType, typename BScaleDataType>
static constexpr bool scale_mfma_hw_support()
{
    return is_scale_mfma_data_type<ADataType>() && is_scale_mfma_data_type<BDataType>() &&
           is_scale_mfma_scale_type<AScaleDataType>() && is_scale_mfma_scale_type<BScaleDataType>();
}

enum struct MfmaInstr
{
    mfma_f32_32x32x1f32 = 0,
    mfma_f32_16x16x1f32,
    mfma_f32_4x4x1f32,
    mfma_f32_32x32x2f32,
    mfma_f32_16x16x4f32,
    mfma_f32_32x32x4f16,
    mfma_f32_16x16x4f16,
    mfma_f32_4x4x4f16,
    mfma_f32_32x32x8f16,
    mfma_f32_16x16x16f16,
    mfma_f32_32x32x8bf16_1k,
    mfma_f32_16x16x16bf16_1k,
    mfma_f32_32x32x4bf16,
    mfma_f32_16x16x8bf16,
    mfma_i32_32x32x8i8,
    mfma_i32_16x16x16i8,
    mfma_i32_32x32x16i8,
    mfma_i32_16x16x32i8,
    mfma_f64_16x16x4f64,
    mfma_f32_32x32x16f8f8,
    mfma_f32_16x16x32f8f8,
    mfma_f32_32x32x16bf8bf8,
    mfma_f32_16x16x32bf8bf8,
    mfma_f32_32x32x16f8bf8,
    mfma_f32_16x16x32f8bf8,
    mfma_f32_32x32x16bf8f8,
    mfma_f32_16x16x32bf8f8,
    mfma_f32_32x32x16f16,
    mfma_f32_16x16x32f16,
    mfma_f32_32x32x16bf16,
    mfma_f32_16x16x32bf16,
    mfma_i32_32x32x32i8,
    mfma_i32_16x16x64i8,
    mfma_f32_32x32x64f8f6f4,
    mfma_f32_16x16x128f8f6f4,
    mfma_scale_f32_32x32x64f8f6f4,
    mfma_scale_f32_16x16x128f8f6f4,
    mfma_f32_16x16x8xf32,  // tf32 on gfx942
    mfma_f32_32x32x4xf32,  // tf32 on gfx942
    mfma_f32_16x16x32xf32, // bf16x3 simulate tf32 on gfx950
    mfma_f32_32x32x16xf32, // bf16x3 simulate tf32 on gfx950
    // gfx11
    wmma_f32_16x16x16_f16,
    wmma_f32_16x16x16_bf16,
    wmma_i32_16x16x16_iu8,
    wmma_unsupport_16x16_gfx11,
    // gfx12
    wmma_f32_16x16x16_f16_gfx12,
    wmma_f32_16x16x16_bf16_gfx12,
    wmma_i32_16x16x16_iu8_gfx12,
    wmma_f32_16x16x16_f8f8_gfx12,
    wmma_f32_16x16x16_f8bf8_gfx12,
    wmma_f32_16x16x16_bf8f8_gfx12,
    wmma_f32_16x16x16_bf8bf8_gfx12,
    wmma_unsupport_16x16_gfx12,
};

template <MfmaInstr instr>
struct mfma_type;

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x1f32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x2f32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x2f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x4f32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x1f32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x1f32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x1f32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x8f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x8f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x16f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x16f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_4x4x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x4f16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16bf16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x8bf16_1k>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x8bf16_1k<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32bf16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x16bf16_1k>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x16bf16_1k<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4bf16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x8bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x8bf16<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_32x32x8i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_32x32x8i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_16x16x16i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_16x16x16i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_32x32x16i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_32x32x16i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_16x16x32i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_16x16x32i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_32x32x32i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 16;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_32x32x32i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_i32_16x16x64i8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 16;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_i32_16x16x64i8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f64_16x16x4f64>
{
    static constexpr index_t group_size          = 1;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 4; // group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4; // wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f64_16x16x4f64<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16f8f8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16f8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32f8f8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32f8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16bf8bf8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16bf8bf8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32bf8bf8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32bf8bf8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16f8bf8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16f8bf8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32f8bf8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32f8bf8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16bf8f8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16bf8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32bf8f8>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32bf8f8<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x64f8f6f4>
{
    // clang-format off
    static constexpr index_t group_size          = 4;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_groups_per_blk  = 4;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_regs_per_blk    = 16;   // m_per_blk * n_per_blk / wave_size
    static constexpr index_t num_threads_per_blk = 32;   // n_per_blk
    static constexpr index_t wave_size           = 64;   // fixed
    static constexpr index_t num_input_blks      = 2;    // m_per_blk / num_regs_per_blk
    static constexpr index_t num_output_blks     = 1;    // (is_k_reduction == true) ???
    static constexpr index_t m_per_blk           = 32;   // from the instruction
    static constexpr index_t n_per_blk           = 32;   // from the instruction
    static constexpr index_t k_per_blk           = 32;   // (is_k_reduction == true) ? KPerXdlops / num_input_blks
    static constexpr bool is_k_reduction         = true; // ???
    // clang-format on

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x64f8f6f4<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x128f8f6f4>
{
    // clang-format off
    static constexpr index_t group_size          = 4;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_groups_per_blk  = 1;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_regs_per_blk    = 4;    // m_per_blk * n_per_blk / wave_size
    static constexpr index_t num_threads_per_blk = 16;   // == n_per_blk
    static constexpr index_t wave_size           = 64;   // fixed
    static constexpr index_t num_input_blks      = 4;    // m_per_blk / num_regs_per_blk
    static constexpr index_t num_output_blks     = 1;    // (is_k_reduction == true) ???
    static constexpr index_t m_per_blk           = 16;   // from the instruction
    static constexpr index_t n_per_blk           = 16;   // from the instruction
    static constexpr index_t k_per_blk           = 32;   // (is_k_reduction == true) ? KPerXdlops / num_input_blks
    static constexpr bool is_k_reduction         = true; // ???
    // clang-format on

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x128f8f6f4<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_scale_f32_32x32x64f8f6f4>
{
    // clang-format off
    static constexpr index_t group_size          = 4;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_groups_per_blk  = 4;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_regs_per_blk    = 16;   // m_per_blk * n_per_blk / wave_size
    static constexpr index_t num_threads_per_blk = 32;   // n_per_blk
    static constexpr index_t wave_size           = 64;   // fixed
    static constexpr index_t num_input_blks      = 2;    // m_per_blk / num_regs_per_blk
    static constexpr index_t num_output_blks     = 1;    // (is_k_reduction == true) ???
    static constexpr index_t m_per_blk           = 32;   // from the instruction
    static constexpr index_t n_per_blk           = 32;   // from the instruction
    static constexpr index_t k_per_blk           = 32;   // (is_k_reduction == true) ? KPerXdlops / num_input_blks
    static constexpr bool is_k_reduction         = true; // ???
    // clang-format on

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t OpselA,
              index_t OpselB,
              class FloatA,
              class ScaleA,
              class FloatB,
              class ScaleB,
              class FloatC>
    __device__ void run(const FloatA& a,
                        const ScaleA& scale_a,
                        const FloatB& b,
                        const ScaleB& scale_b,
                        FloatC& reg_c) const
    {
        intrin_mfma_scale_f32_32x32x64f8f6f4<MPerXdlops, NPerXdlops, OpselA, OpselB>::Run(
            a, bit_cast<uint32_t>(scale_a), b, bit_cast<uint32_t>(scale_b), reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_scale_f32_16x16x128f8f6f4>
{
    // clang-format off
    static constexpr index_t group_size          = 4;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_groups_per_blk  = 1;    // ??? group_size * num_groups_per_blk == num_regs_per_blk
    static constexpr index_t num_regs_per_blk    = 4;    // m_per_blk * n_per_blk / wave_size
    static constexpr index_t num_threads_per_blk = 16;   // == n_per_blk
    static constexpr index_t wave_size           = 64;   // fixed
    static constexpr index_t num_input_blks      = 4;    // m_per_blk / num_regs_per_blk
    static constexpr index_t num_output_blks     = 1;    // (is_k_reduction == true) ???
    static constexpr index_t m_per_blk           = 16;   // from the instruction
    static constexpr index_t n_per_blk           = 16;   // from the instruction
    static constexpr index_t k_per_blk           = 32;   // (is_k_reduction == true) ? KPerXdlops / num_input_blks
    static constexpr bool is_k_reduction         = true; // ???
    // clang-format on

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t OpselA,
              index_t OpselB,
              class FloatA,
              class ScaleA,
              class FloatB,
              class ScaleB,
              class FloatC>
    __device__ void run(const FloatA& a,
                        const ScaleA& scale_a,
                        const FloatB& b,
                        const ScaleB& scale_b,
                        FloatC& reg_c) const
    {

        intrin_mfma_scale_f32_16x16x128f8f6f4<MPerXdlops, NPerXdlops, OpselA, OpselB>::Run(
            a, bit_cast<uint32_t>(scale_a), b, bit_cast<uint32_t>(scale_b), reg_c);
    }
};

/**
 * num_threads_per_blk == n_per_blk
 * num_regs_per_blk * num_input_blks == m_per_blk
 * num_regs_per_blk * wave_size == m_per_blk * n_per_blk
 *
 * group_size * num_groups_per_blk == num_regs_per_blk
 *
 * num_regs_per_blk is output(CD) register size which is determined by the instruction.
 * k_per_blk(K1PerXdlops) is input(AB) register size which is determined by the instruction.
 * group_size is corresponding to CD rows mapping. see: GetBeginOfThreadBlk()
 *
 * is_k_reduction = (k_per_blk == KPerXdlops) ? false: true.
 *
 * if (is_k_reduction){
 *      num_output_blks == 1;
 * } else {
 *      num_input_blks == num_output_blks;
 * }
 */
template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x8xf32>
{
    static constexpr index_t wave_size           = 64;        // fixed
    static constexpr index_t m_per_blk           = 16;        // from the instruction
    static constexpr index_t n_per_blk           = 16;        // from the instruction
    static constexpr index_t num_threads_per_blk = n_per_blk; // 16
    static constexpr index_t num_regs_per_blk    = m_per_blk * n_per_blk / wave_size; // 4
    static constexpr index_t num_input_blks      = m_per_blk / num_regs_per_blk;      // 4
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t k_per_blk           = 2; //  k_per_blk(K1PerXdlops) should be 2.
    static constexpr bool is_k_reduction         = true;

    // AB register size : 2,  register size: 4
    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x8xf32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x4xf32>
{
    static constexpr index_t wave_size           = 64;        // fixed
    static constexpr index_t m_per_blk           = 32;        // from the instruction
    static constexpr index_t n_per_blk           = 32;        // from the instruction
    static constexpr index_t num_threads_per_blk = n_per_blk; // 32
    static constexpr index_t num_regs_per_blk    = m_per_blk * n_per_blk / wave_size; // 16
    static constexpr index_t num_input_blks      = m_per_blk / num_regs_per_blk;      // 2
    static constexpr index_t group_size          = 4; // corresponding to CD rows mapping
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;
    // AB register size: 2, CD register size: 16
    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4xf32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::mfma_f32_32x32x16xf32>
{
    // gfx950 specific: use bf16x3 simulate tf32
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = 16;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x16xf32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};
template <>
struct mfma_type<MfmaInstr::mfma_f32_16x16x32xf32>
{
    // gfx950 specific: use bf16x3 simulate tf32
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 4;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 4;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops, index_t NPerXdlops, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x32xf32<MPerXdlops, NPerXdlops>::Run(a, b, reg_c);
    }
};

// gfx11
struct mfma_type_gfx11_base
{
    static constexpr index_t group_size          = 8;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 8;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 32;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 16;
    static constexpr bool is_k_reduction         = true;
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_f16> : public mfma_type_gfx11_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_f16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_bf16> : public mfma_type_gfx11_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_bf16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_i32_16x16x16_iu8> : public mfma_type_gfx11_base
{
    template <index_t MPerWmma,
              index_t NPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = true,
              bool neg_b = true,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_i32_16x16x16_iu8_w32<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_unsupport_16x16_gfx11> : public mfma_type_gfx11_base
{
    static constexpr index_t k_per_blk = 2;
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA&, const FloatB&, FloatC&) const
    {
        // empty for all unsupported types.
    }
};

// gfx12
struct mfma_type_gfx12_base
{
    static constexpr index_t group_size          = 8;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = 8;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 32;
    static constexpr index_t num_input_blks      = 2;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 8;
    static constexpr bool is_k_reduction         = true;
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_f16_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_f16_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_bf16_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_bf16_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_i32_16x16x16_iu8_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma,
              index_t NPerWmma,
              class FloatA,
              class FloatB,
              class FloatC,
              bool neg_a = true,
              bool neg_b = true,
              bool clamp = false>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_i32_16x16x16_iu8_w32_gfx12<MPerWmma, NPerWmma, neg_a, neg_b, clamp>::Run(
            a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_f8f8_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_f8f8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_f8bf8_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_f8bf8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_bf8f8_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_bf8f8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_f32_16x16x16_bf8bf8_gfx12> : public mfma_type_gfx12_base
{
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_bf8bf8_w32_gfx12<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_type<MfmaInstr::wmma_unsupport_16x16_gfx12> : public mfma_type_gfx12_base
{
    static constexpr index_t k_per_blk = 2;
    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA&, const FloatB&, FloatC&) const
    {
        // empty for all unsupported types.
    }
};

/**
 * @class MfmaSelector
 * @brief Selects the appropriate MFMA instruction type and configuration for given data types
 *          and tile sizes on AMD GPUs.
 *
 * @tparam base_type        The base data type for the matrix operation (e.g., float, half_t).
 * @tparam MPerXdlops       The number of rows per XDLops tile.
 * @tparam NPerXdlops       The number of columns per XDLops tile.
 * @tparam additional_type  (Optional) Additional data type for mixed-precision or special cases.
 *                          Defaults to base_type.
 * @tparam is_single_rate_mfma (Optional) Whether to use single-rate MFMA instructions.
 *                          Defaults to false.
 * @tparam is_scale_mfma    (Optional) Whether to use scale MFMA instructions. Defaults to false.
 */
template <typename base_type,
          index_t MPerXdlops,
          index_t NPerXdlops,
          typename additional_type = base_type,
          bool is_single_rate_mfma = false,
          bool is_scale_mfma       = false>
struct MfmaSelector
{
    template <typename base_type_,
              index_t MPerXdlops_,
              index_t NPerXdlops_,
              typename additional_type_ = base_type_,
              bool is_single_rate_mfma_ = false,
              bool is_scale_mfma_       = false>
    static constexpr auto GetMfma();

    template <>
    constexpr auto GetMfma<double, 16, 16>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_f64_16x16x4f64;
#endif
    }

    template <>
    constexpr auto GetMfma<float, 64, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x1f32;
    }

    template <>
    constexpr auto GetMfma<float, 32, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x1f32;
    }

    template <>
    constexpr auto GetMfma<float, 16, 64>()
    {
        return MfmaInstr::mfma_f32_16x16x1f32;
    }

    template <>
    constexpr auto GetMfma<float, 8, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x1f32;
    }

    template <>
    constexpr auto GetMfma<float, 4, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x1f32;
    }

    template <>
    constexpr auto GetMfma<float, 32, 32>()
    {
        return MfmaInstr::mfma_f32_32x32x2f32;
    }

    template <>
    constexpr auto GetMfma<float, 16, 16>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_f32_16x16x4f32;
#endif
    }

    template <>
    constexpr auto GetMfma<tf32_t, 32, 32, tf32_t>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x16xf32;
#elif defined(__gfx942__)
        return MfmaInstr::mfma_f32_32x32x4xf32;
#else
        return MfmaInstr::mfma_f32_32x32x2f32;
#endif
    }

    template <>
    constexpr auto GetMfma<tf32_t, 16, 16, tf32_t>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x32xf32;
#elif defined(__gfx942__)
        return MfmaInstr::mfma_f32_16x16x8xf32;
#else
        return MfmaInstr::mfma_f32_16x16x4f32;
#endif
    }

    template <>
    constexpr auto GetMfma<half_t, 64, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x4f16;
    }

    template <>
    constexpr auto GetMfma<half_t, 32, 64>()
    {
        return MfmaInstr::mfma_f32_32x32x4f16;
    }

    template <>
    constexpr auto GetMfma<half_t, 32, 32, half_t, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x16f16;
#else
        return MfmaInstr::mfma_f32_32x32x8f16;
#endif
    }
    template <>
    constexpr auto GetMfma<half_t, 32, 32, half_t, true>()
    {
        return MfmaInstr::mfma_f32_32x32x8f16;
    }

    template <>
    constexpr auto GetMfma<half_t, 16, 16, half_t, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_f16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_f32_16x16x16_f16;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x32f16;
#else
        return MfmaInstr::mfma_f32_16x16x16f16;
#endif
    }

    template <>
    constexpr auto GetMfma<half_t, 16, 16, half_t, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_f16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_f32_16x16x16_f16;
#else
        return MfmaInstr::mfma_f32_16x16x16f16;
#endif
    }

    template <>
    constexpr auto GetMfma<half_t, 16, 64>()
    {
        return MfmaInstr::mfma_f32_16x16x4f16;
    }

    template <>
    constexpr auto GetMfma<half_t, 8, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x4f16;
    }

    template <>
    constexpr auto GetMfma<half_t, 4, 64>()
    {
        return MfmaInstr::mfma_f32_4x4x4f16;
    }

    template <>
    constexpr auto GetMfma<bhalf_t, 32, 32, bhalf_t, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x16bf16;
#elif defined(CK_USE_AMD_MFMA_BF16_1K_OP)
        return MfmaInstr::mfma_f32_32x32x8bf16_1k;
#else
        return MfmaInstr::mfma_f32_32x32x4bf16;
#endif
    }

    template <>
    constexpr auto GetMfma<bhalf_t, 32, 32, bhalf_t, true>()
    {
#if defined(CK_USE_AMD_MFMA_BF16_1K_OP)
        return MfmaInstr::mfma_f32_32x32x8bf16_1k;
#else
        return MfmaInstr::mfma_f32_32x32x4bf16;
#endif
    }

    template <>
    constexpr auto GetMfma<bhalf_t, 16, 16, bhalf_t, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_bf16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_f32_16x16x16_bf16;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x32bf16;
#elif defined(CK_USE_AMD_MFMA_BF16_1K_OP)
        return MfmaInstr::mfma_f32_16x16x16bf16_1k;
#else
        return MfmaInstr::mfma_f32_16x16x8bf16;
#endif
    }

    template <>
    constexpr auto GetMfma<bhalf_t, 16, 16, bhalf_t, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_bf16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_f32_16x16x16_bf16;
#elif defined(CK_USE_AMD_MFMA_BF16_1K_OP)
        return MfmaInstr::mfma_f32_16x16x16bf16_1k;
#else
        return MfmaInstr::mfma_f32_16x16x8bf16;
#endif
    }

    template <>
    constexpr auto GetMfma<int8_t, 32, 32, int8_t, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_i32_32x32x32i8;
#elif defined(__gfx942__)
        return MfmaInstr::mfma_i32_32x32x16i8;
#else
        return MfmaInstr::mfma_i32_32x32x8i8;
#endif
    }

    template <>
    constexpr auto GetMfma<int8_t, 32, 32, int8_t, true>()
    {
#if defined(__gfx942__) || defined(__gfx950__)
        return MfmaInstr::mfma_i32_32x32x16i8;
#else
        return MfmaInstr::mfma_i32_32x32x8i8;
#endif
    }

    template <>
    constexpr auto GetMfma<int8_t, 16, 16, int8_t, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_i32_16x16x16_iu8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_i32_16x16x16_iu8;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_i32_16x16x64i8;
#elif defined(__gfx942__)
        return MfmaInstr::mfma_i32_16x16x32i8;
#else
        return MfmaInstr::mfma_i32_16x16x16i8;
#endif
    }

    template <>
    constexpr auto GetMfma<int8_t, 16, 16, int8_t, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_i32_16x16x16_iu8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_i32_16x16x16_iu8;
#elif defined(__gfx942__) || defined(__gfx950__)
        return MfmaInstr::mfma_i32_16x16x32i8;
#else
        return MfmaInstr::mfma_i32_16x16x16i8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 32, 32, f8_t, true, false>()
    {
        return MfmaInstr::mfma_f32_32x32x16f8f8;
    }

    template <>
    constexpr auto GetMfma<f8_t, 32, 32, f8_t, false, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x64f8f6f4;
#else
        return MfmaInstr::mfma_f32_32x32x16f8f8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 32, 32, f8_t, is_single_rate_mfma, true>()
    {
        return MfmaInstr::mfma_scale_f32_32x32x64f8f6f4;
    }

    template <>
    constexpr auto GetMfma<bf8_t, 32, 32, f8_t, is_single_rate_mfma, true>()
    {
        return MfmaInstr::mfma_scale_f32_32x32x64f8f6f4;
    }
    template <>
    constexpr auto GetMfma<f4_t, 32, 32, f4_t, is_single_rate_mfma, true>()
    {
        return MfmaInstr::mfma_scale_f32_32x32x64f8f6f4;
    }
    template <>
    constexpr auto GetMfma<f4_t, 16, 16, f4_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 16, 16, f8_t, true, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_f8f8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_f32_16x16x32f8f8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 16, 16, f8_t, false, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_f8f8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x128f8f6f4;
#else
        return MfmaInstr::mfma_f32_16x16x32f8f8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 16, 16, f8_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 16, 16, bf8_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 16, 16, bf8_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 16, 16, f8_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }

    template <>
    constexpr auto GetMfma<f6_t, 32, 32, f6_t, is_single_rate_mfma, true>()
    {
        return MfmaInstr::mfma_scale_f32_32x32x64f8f6f4;
    }
    template <>
    constexpr auto GetMfma<f6_t, 16, 16, f6_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }
    template <>
    constexpr auto GetMfma<bf6_t, 32, 32, bf6_t, is_single_rate_mfma, true>()
    {
        return MfmaInstr::mfma_scale_f32_32x32x64f8f6f4;
    }
    template <>
    constexpr auto GetMfma<bf6_t, 16, 16, bf6_t, is_single_rate_mfma, true>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_unsupport_16x16_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_scale_f32_16x16x128f8f6f4;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 32, 32, bf8_t, true, false>()
    {
        return MfmaInstr::mfma_f32_32x32x16bf8bf8;
    }

    template <>
    constexpr auto GetMfma<bf8_t, 32, 32, bf8_t, false, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x64f8f6f4;
#else
        return MfmaInstr::mfma_f32_32x32x16bf8bf8;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 16, 16, bf8_t, true, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_bf8bf8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_f32_16x16x32bf8bf8;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 16, 16, bf8_t, false, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_bf8bf8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x128f8f6f4;
#else
        return MfmaInstr::mfma_f32_16x16x32bf8bf8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 32, 32, bf8_t, true, false>()
    {
        return MfmaInstr::mfma_f32_32x32x16f8bf8;
    }

    template <>
    constexpr auto GetMfma<f8_t, 32, 32, bf8_t, false, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x64f8f6f4;
#else
        return MfmaInstr::mfma_f32_32x32x16f8bf8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 16, 16, bf8_t, true, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_f8bf8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_f32_16x16x32f8bf8;
#endif
    }

    template <>
    constexpr auto GetMfma<f8_t, 16, 16, bf8_t, false, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_f8bf8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x128f8f6f4;
#else
        return MfmaInstr::mfma_f32_16x16x32f8bf8;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 32, 32, f8_t, true, false>()
    {
        return MfmaInstr::mfma_f32_32x32x16bf8f8;
    }

    template <>
    constexpr auto GetMfma<bf8_t, 32, 32, f8_t, false, false>()
    {
#if defined(__gfx950__)
        return MfmaInstr::mfma_f32_32x32x64f8f6f4;
#else
        return MfmaInstr::mfma_f32_32x32x16bf8f8;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 16, 16, f8_t, true, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_bf8f8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
        return MfmaInstr::mfma_f32_16x16x32bf8f8;
#endif
    }

    template <>
    constexpr auto GetMfma<bf8_t, 16, 16, f8_t, false, false>()
    {
#if defined(__gfx12__)
        return MfmaInstr::wmma_f32_16x16x16_bf8f8_gfx12;
#elif defined(__gfx11__)
        return MfmaInstr::wmma_unsupport_16x16_gfx11;
#elif defined(__gfx950__)
        return MfmaInstr::mfma_f32_16x16x128f8f6f4;
#else
        return MfmaInstr::mfma_f32_16x16x32bf8f8;
#endif
    }

    static constexpr auto selected_mfma = mfma_type<GetMfma<element_type_t<base_type>,
                                                            MPerXdlops,
                                                            NPerXdlops,
                                                            element_type_t<additional_type>,
                                                            is_single_rate_mfma,
                                                            is_scale_mfma>()>{};

    __host__ __device__ constexpr MfmaSelector()
    {
        static_assert(selected_mfma.group_size * selected_mfma.num_groups_per_blk ==
                          selected_mfma.num_regs_per_blk,
                      "wrong! num_regs_per_blk");

        static_assert(selected_mfma.num_threads_per_blk == selected_mfma.n_per_blk,
                      "n_per_blk != num_threads_per_blk");
#if defined(__gfx11__)
        if constexpr(MPerXdlops == 16 && NPerXdlops == 16)
        {
            static_assert(selected_mfma.num_regs_per_blk * selected_mfma.num_input_blks * 2 ==
                              selected_mfma.m_per_blk,
                          "m_per_blk != num_input_blks * num_regs_per_blk");
        }
#else
        static_assert(selected_mfma.num_regs_per_blk * selected_mfma.num_input_blks ==
                          selected_mfma.m_per_blk,
                      "m_per_blk != num_input_blks * num_regs_per_blk");
#endif

        static_assert(selected_mfma.num_output_blks == selected_mfma.num_input_blks ||
                          selected_mfma.num_output_blks == 1,
                      "incorrect num_output_blks");

        static_assert(selected_mfma.num_regs_per_blk * selected_mfma.wave_size ==
                          selected_mfma.m_per_blk * selected_mfma.n_per_blk,
                      "num_regs_per_blk incorrect");

        static_assert(selected_mfma.is_k_reduction ||
                          (selected_mfma.num_input_blks == selected_mfma.num_output_blks),
                      "is_k_reduction wrong!");
    }

    static constexpr bool IsABroadcast()
    {
        static_assert(NPerXdlops >= MPerXdlops, "only support ABroadcast");
        return true;
    }

    static constexpr index_t GetKPerXdlops()
    {
        return (selected_mfma.is_k_reduction ? selected_mfma.num_input_blks : 1) *
               selected_mfma.k_per_blk;
    }

    static constexpr index_t GetK1PerXdlops() { return selected_mfma.k_per_blk; }
};

template <typename base_type,
          index_t MPerXdlops,
          index_t NPerXdlops,
          index_t KPack,
          typename additional_type = base_type,
          bool TransposeC          = false,
          bool is_scale_mfma       = false>
struct XdlopsGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex4D = MultiIndex<4>;

    __device__ static constexpr index_t GetNumBlks() { return mfma_instr.num_output_blks; }

    __device__ static constexpr index_t GetNumXdlops()
    {
        return MPerXdlops * NPerXdlops /
               (mfma_instr.m_per_blk * mfma_instr.n_per_blk * mfma_instr.num_output_blks);
    }

    __host__ __device__ constexpr XdlopsGemm()
    {
        static_assert(NPerXdlops == 4 || NPerXdlops == 8 || NPerXdlops == 16 || NPerXdlops == 32 ||
                          NPerXdlops == 64,
                      "Only support GemmNPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerXdlops == 4 || MPerXdlops == 8 || MPerXdlops == 16 || MPerXdlops == 32 ||
                          MPerXdlops == 64,
                      "Only support GemmMPerXdlops == 4, 8, 16, 32 or 64 for xdlops");
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(KPack % mfma_instr.k_per_blk == 0, "KPack should be a multiple of k_per_blk");
#endif
    }

    // XDL output supporting C = A * B
    // M2_N2 -> M2_M3_M4_N2
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);
        constexpr auto num_blks = mfma_instr.m_per_blk / mfma_instr.num_regs_per_blk;

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(Number<mfma_instr.num_groups_per_blk>{},
                                                         Number<num_blks>{},
                                                         Number<mfma_instr.group_size>{})),
                       make_pass_through_transform(Number<mfma_instr.num_threads_per_blk>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4, 5, 6>{},
                       Sequence<7>{}));
    }

    // XDL output supporting C = A * B
    // M3_N3 -> M3_M4_M5_N3
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto MakeCDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(
        const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);
        const auto M2           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I4);
        const auto N2           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I5);
        constexpr auto num_blks = mfma_instr.m_per_blk / mfma_instr.num_regs_per_blk;

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_pass_through_transform(M2),
                       make_pass_through_transform(N2),
                       make_unmerge_transform(make_tuple(Number<mfma_instr.num_groups_per_blk>{},
                                                         Number<num_blks>{},
                                                         Number<mfma_instr.group_size>{})),
                       make_pass_through_transform(Number<mfma_instr.num_threads_per_blk>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{},
                       Sequence<6>{},
                       Sequence<7>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{},
                       Sequence<6, 7, 8>{},
                       Sequence<9>{}));
    }

    // transposed XDL output supporting C' = B' * A'
    // M2_N2 -> M2_N2_N3_N4
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1           = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);
        constexpr auto num_blks = mfma_instr.m_per_blk / mfma_instr.num_regs_per_blk;

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_pass_through_transform(Number<mfma_instr.num_threads_per_blk>{}),
                       make_unmerge_transform(make_tuple(Number<mfma_instr.num_groups_per_blk>{},
                                                         Number<num_blks>{},
                                                         Number<mfma_instr.group_size>{}))),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{}));
    }

    template <typename CDesc_G_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
        const CDesc_G_M0_N0_M1_N1_M2_N2& c_desc_g_m0_n0_m1_n1_m2_n2)
    {
        const auto G            = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto M0           = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto N0           = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto M1           = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I3);
        const auto N1           = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I4);
        constexpr auto num_blks = mfma_instr.m_per_blk / mfma_instr.num_regs_per_blk;

        return transform_tensor_descriptor(
            c_desc_g_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(G),
                       make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(
                           mfma_instr.num_groups_per_blk, num_blks, mfma_instr.group_size)),
                       make_pass_through_transform(mfma_instr.num_threads_per_blk)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{},
                       Sequence<6>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{},
                       Sequence<8>{}));
    }

    __device__ __host__ static constexpr index_t GetRegSizePerXdlops()
    {
        return mfma_instr.num_regs_per_blk;
    }

    __device__ static constexpr index_t GetWaveSize() { return mfma_instr.wave_size; }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(
            is_same<base_type, double>::value || is_same<base_type, float>::value ||
                is_same<base_type, tf32_t>::value || is_same<base_type, half_t>::value ||
                is_same<base_type, bhalf_t>::value || is_same<base_type, int8_t>::value ||
                is_same<base_type, f8_t>::value || is_same<base_type, bf8_t>::value ||
                (is_same<base_type, f8_t>::value && is_same<additional_type, bf8_t>::value) ||
                (is_same<base_type, bf8_t>::value && is_same<additional_type, f8_t>::value),
            "base_type must be double, float, tf32_t, half, bfloat16, int8_t, f8_t or bf8_t!");

        static_for<0, KPack / mfma_instr.k_per_blk, 1>{}([&](auto k) {
            if constexpr(!TransposeC)
            {
                mfma_instr.template run<MPerXdlops, NPerXdlops>(
                    p_a_wave[k], p_b_wave[k], p_c_thread);
            }
            else
            {
                mfma_instr.template run<MPerXdlops, NPerXdlops>(
                    p_b_wave[k], p_a_wave[k], p_c_thread);
            }
        });
    }

    template <index_t OpselA,
              index_t OpselB,
              class FloatA,
              class ScaleA,
              class FloatB,
              class ScaleB,
              class FloatC>
    __device__ void Run(const FloatA& p_a_wave,
                        const ScaleA& a_scale_thread,
                        const FloatB& p_b_wave,
                        const ScaleB& b_scale_thread,
                        FloatC& p_c_thread) const
    {
        static_for<0, KPack / mfma_instr.k_per_blk, 1>{}([&](auto k) {
            if constexpr(!TransposeC)
            {
                mfma_instr.template run<MPerXdlops, NPerXdlops, OpselA, OpselB>(
                    p_a_wave[k], a_scale_thread[k], p_b_wave[k], b_scale_thread[k], p_c_thread);
            }
            else
            {
                mfma_instr.template run<MPerXdlops, NPerXdlops, OpselB, OpselA>(
                    p_b_wave[k], b_scale_thread[k], p_a_wave[k], a_scale_thread[k], p_c_thread);
            }
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % mfma_instr.wave_size; }

    __device__ static auto GetBlkIdx()
    {
        const auto laneId       = GetLaneId();
        constexpr auto num_blks = mfma_instr.m_per_blk / mfma_instr.num_regs_per_blk;

        constexpr auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(1, num_blks, mfma_instr.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    template <bool SwizzleA>
    __device__ static auto GetGfx11InputBlkIdx()
    {
        auto laneId = GetLaneId() % mfma_instr.num_threads_per_blk;
        if constexpr(SwizzleA)
        {
            laneId = ((laneId & 1) << 3) | (laneId >> 1);
        }
        constexpr auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(
                make_tuple(1, mfma_instr.num_input_blks, mfma_instr.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto laneId = GetLaneId();
#if defined(__gfx11__)
        const auto blk_idx = GetGfx11InputBlkIdx<!TransposeC>();
#else
        const auto blk_idx = GetBlkIdx();
#endif

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto laneId = GetLaneId();
#if defined(__gfx11__)
        const auto blk_idx = GetGfx11InputBlkIdx<TransposeC>();
#else
        const auto blk_idx = GetBlkIdx();
#endif

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __device__ static CIndex GetBeginOfThreadBlk(index_t xdlops_i, index_t blk_i)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        index_t n_offset = blk_i * mfma_instr.n_per_blk + blk_td;
        index_t m_offset = xdlops_i * mfma_instr.m_per_blk + blk_id * mfma_instr.group_size;

        return TransposeC ? CIndex{n_offset, m_offset} : CIndex{m_offset, n_offset};
    }

    __device__ static CIndex4D GetBeginOfThreadBlk4D(index_t /* xdlops_i */, index_t /* blk_i */)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        return TransposeC ? CIndex4D{blk_td, I0, blk_id, I0} : CIndex4D{I0, blk_id, I0, blk_td};
    }

    // Falls back to single rate instruction on gfx950 if KPack is single rate; no change on gfx942-
    // when base_type is either f8_t or bf8_t, additional_type will always be either f8_t or bf8_t,
    // except Use single rate mfma instruction for this special case A (f8_t) * B (pk_i4_t)
    static constexpr bool is_single_rate_mfma =
        (((is_same<base_type, half_t>::value || is_same<base_type, bhalf_t>::value) &&
          KPack <= 4) ||
         (is_same<base_type, int8_t>::value && KPack <= 8) ||
         ((is_same<base_type, f8_t>::value || is_same<base_type, bf8_t>::value) && KPack < 32) ||
         is_same<additional_type, pk_i4_t>::value)
#if defined(__gfx950__)
                // tf32 on gfx950 is implemented as bf16x3, so it should be treated as bf16.
                || (is_same<base_type, tf32_t>::value && KPack <= 4)
#endif
            ? true
            : false;
    static constexpr auto mfma = MfmaSelector<base_type,
                                              MPerXdlops,
                                              NPerXdlops,
                                              additional_type,
                                              is_single_rate_mfma,
                                              is_scale_mfma>{};

    static constexpr auto mfma_instr = mfma.selected_mfma;

    static constexpr auto KPerXdlops  = mfma.GetKPerXdlops();
    static constexpr auto K1PerXdlops = mfma.GetK1PerXdlops();
    static constexpr auto K0PerXdlops = KPerXdlops / K1PerXdlops;

    __host__ __device__ static constexpr auto GetCM0M1M2NThreadBlkLengths()
    {
        return make_tuple(
            Number<mfma_instr.num_groups_per_blk>{}, I1, Number<mfma_instr.group_size>{}, I1);
    }
};

} // namespace ck
