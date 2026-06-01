// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "mx_mfma_op.hpp"

using ck::bf6_t;
using ck::bf8_t;
using ck::e8m0_bexp_t;
using ck::f4_t;
using ck::f4x2_pk_t;
using ck::f6_t;
using ck::f8_t;
using ck::half_t;
using ck::type_convert;

/**
 * @brief Run the test for the given MFMA instruction
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          ck::MFMA_F8F6F4 mfma>
bool run_mfma_test(ck::index_t init)
{
    using AccType    = float; // only MFMA_F32 instructions supported
    using CPUAccType = AccType;

    ck::mfma_type<static_cast<ck::MfmaInstr>(mfma)> mfma_instr;
    constexpr auto BLOCK_M = mfma_instr.m_per_blk;
    constexpr auto BLOCK_N = mfma_instr.n_per_blk;
    constexpr auto BLOCK_K = mfma_instr.num_input_blks * mfma_instr.k_per_blk;

    const auto mfma_kernel = ck::
        matmul<AType, BType, CType, AccType, BLOCK_M, BLOCK_N, BLOCK_K, ALayout, BLayout, CLayout>;

    bool pass = true;

    pass = ck::mfma_test::TestMFMA<decltype(mfma_kernel),
                                   AType,
                                   BType,
                                   CType,
                                   AccType,
                                   CPUAccType,
                                   ALayout,
                                   BLayout,
                                   CLayout,
                                   BLOCK_M,
                                   BLOCK_N,
                                   BLOCK_K>{}(mfma_kernel, init);

    return pass;
}

const ck::index_t common_init = -4; // set to "< 0" for test-specific initializations

TEST(MFMA, FP8MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::ColumnMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::ColumnMajor;
    auto AB_init  = (common_init < 0) ? 5 : common_init;
    auto pass     = run_mfma_test<ALayout,
                                  BLayout,
                                  CLayout,
                                  f8_t,
                                  f8_t,
                                  half_t,
                                  ck::MFMA_F8F6F4::F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, BF8MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::ColumnMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::ColumnMajor;
    auto AB_init  = (common_init < 0) ? 5 : common_init;
    auto pass     = run_mfma_test<ALayout,
                                  BLayout,
                                  CLayout,
                                  bf8_t,
                                  bf8_t,
                                  half_t,
                                  ck::MFMA_F8F6F4::F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, FP4MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass =
        run_mfma_test<ALayout, BLayout, CLayout, f4_t, f4_t, float, ck::MFMA_F8F6F4::F32_16x16x128>(
            AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, FP6MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass =
        run_mfma_test<ALayout, BLayout, CLayout, f6_t, f6_t, float, ck::MFMA_F8F6F4::F32_16x16x128>(
            AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, BF6MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mfma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 ck::MFMA_F8F6F4::F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, FP8MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::ColumnMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::ColumnMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass =
        run_mfma_test<ALayout, BLayout, CLayout, f8_t, f8_t, float, ck::MFMA_F8F6F4::F32_32x32x64>(
            AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, BF8MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::ColumnMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::ColumnMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mfma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 ck::MFMA_F8F6F4::F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, FP4MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass =
        run_mfma_test<ALayout, BLayout, CLayout, f4_t, f4_t, half_t, ck::MFMA_F8F6F4::F32_32x32x64>(
            AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, FP6MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass =
        run_mfma_test<ALayout, BLayout, CLayout, f6_t, f6_t, half_t, ck::MFMA_F8F6F4::F32_32x32x64>(
            AB_init);
    EXPECT_TRUE(pass);
}

TEST(MFMA, BF6MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mfma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 half_t,
                                 ck::MFMA_F8F6F4::F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}

/**
 * @brief Run the test for the given MX MFMA instruction
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          ck::MFMA_F8F6F4 mfma>
bool run_mxmfma_test(ck::index_t init)
{
    static_assert(mfma == ck::MFMA_F8F6F4::SCALE_F32_16x16x128 ||
                      mfma == ck::MFMA_F8F6F4::SCALE_F32_32x32x64,
                  "Only SCALE_F32_16x16x128 and SCALE_F32_32x32x64 are supported");

    using AccType   = float;           // only MFMA_F32 instructions supported
    using ScaleType = ck::e8m0_bexp_t; // biased exponent type

    ck::mfma_type<static_cast<ck::MfmaInstr>(mfma)> mfma_instr;
    constexpr auto BLOCK_M = mfma_instr.m_per_blk;
    constexpr auto BLOCK_N = mfma_instr.n_per_blk;
    constexpr auto BLOCK_K = mfma_instr.num_input_blks * mfma_instr.k_per_blk;
    constexpr auto BLOCK_X = 32; // scaling vector size

    const auto mx_mfma_kernel = ck::matmul<AType,
                                           BType,
                                           ScaleType,
                                           CType,
                                           AccType,
                                           BLOCK_M,
                                           BLOCK_N,
                                           BLOCK_K,
                                           BLOCK_X,
                                           ALayout,
                                           BLayout,
                                           CLayout>;

    bool pass = true;

    pass = ck::mxmfma_test::TestMXMFMA<decltype(mx_mfma_kernel),
                                       AType,
                                       BType,
                                       ScaleType,
                                       CType,
                                       ALayout,
                                       BLayout,
                                       CLayout,
                                       BLOCK_M,
                                       BLOCK_N,
                                       BLOCK_K,
                                       BLOCK_X>{}(mx_mfma_kernel, init);

    return pass;
}

TEST(MXMFMA, MXFP8MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   f8_t,
                                   f8_t,
                                   float,
                                   ck::MFMA_F8F6F4::SCALE_F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXFP8MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   f8_t,
                                   f8_t,
                                   half_t,
                                   ck::MFMA_F8F6F4::SCALE_F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXBF8MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   bf8_t,
                                   bf8_t,
                                   float,
                                   ck::MFMA_F8F6F4::SCALE_F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXBF8MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   bf8_t,
                                   bf8_t,
                                   half_t,
                                   ck::MFMA_F8F6F4::SCALE_F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXFP6MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   f6_t,
                                   f6_t,
                                   float,
                                   ck::MFMA_F8F6F4::SCALE_F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXFP6MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   f6_t,
                                   f6_t,
                                   half_t,
                                   ck::MFMA_F8F6F4::SCALE_F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXBF6MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   bf6_t,
                                   bf6_t,
                                   float,
                                   ck::MFMA_F8F6F4::SCALE_F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXBF6MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   bf6_t,
                                   bf6_t,
                                   half_t,
                                   ck::MFMA_F8F6F4::SCALE_F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXFP4MFMA16x16x128)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   f4_t,
                                   f4_t,
                                   float,
                                   ck::MFMA_F8F6F4::SCALE_F32_16x16x128>(AB_init);
    EXPECT_TRUE(pass);
}

TEST(MXMFMA, MXFP4MFMA32x32x64)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto AB_init = (common_init < 0) ? 5 : common_init;
    auto pass    = run_mxmfma_test<ALayout,
                                   BLayout,
                                   CLayout,
                                   f4_t,
                                   f4_t,
                                   half_t,
                                   ck::MFMA_F8F6F4::SCALE_F32_32x32x64>(AB_init);
    EXPECT_TRUE(pass);
}
