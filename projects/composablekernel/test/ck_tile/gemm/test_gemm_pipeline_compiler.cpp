// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

// ============================================================================
// Comprehensive GEMM Compiler Validation Test Suite
// This file consolidates all GEMM pipeline tests for compiler validation
// Covers essential combinations of data types, layouts, and pipeline types
// ============================================================================

// ----------------------------------------------------------------------------
// Test Class Definitions for Different Pipeline Types
// ----------------------------------------------------------------------------

template <typename T>
class TestGemmMem : public TestCkTileGemmPipeline<T, TestGemmMem<T>>
{
};

#if defined(CK_TILE_USE_WMMA)
template <typename T>
class TestGemmMemWmma : public TestCkTileGemmPipeline<T, TestGemmMemWmma<T>>
{
};
#endif

template <typename T>
class TestGemmCompV3 : public TestCkTileGemmPipeline<T, TestGemmCompV3<T>>
{
};

#if defined(CK_TILE_USE_WMMA)
template <typename T>
class TestGemmCompV3Wmma : public TestCkTileGemmPipeline<T, TestGemmCompV3Wmma<T>>
{
};
#endif

template <typename T>
class TestGemmCompV4 : public TestCkTileGemmPipeline<T, TestGemmCompV4<T>>
{
};

#if defined(CK_TILE_USE_WMMA)
template <typename T>
class TestGemmCompV4Wmma : public TestCkTileGemmPipeline<T, TestGemmCompV4Wmma<T>>
{
};
#endif

template <typename T>
class TestGemmCompV6 : public TestCkTileGemmPipeline<T, TestGemmCompV6<T>>
{
};

template <typename T>
class TestGemmPersistent : public TestCkTileGemmPipeline<T, TestGemmPersistent<T>>
{
};

#if defined(CK_TILE_USE_WMMA)
template <typename T>
class TestGemmPersistentWmma : public TestCkTileGemmPipeline<T, TestGemmPersistentWmma<T>>
{
};
#endif

// ----------------------------------------------------------------------------
// Type Definitions for Each Pipeline Configuration
// ----------------------------------------------------------------------------

// Memory Pipeline Types
using MemTestTypes = ::testing::Types<
    // Parameters: ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType,
    // M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, K_TileSize, Scheduler,
    // PipelineType

    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Interwave, Mem>,
    std::tuple<Row, Row, Row, BF16, BF16, F32, BF16, I64, I64, I32, I16, I16, I16, Interwave, Mem>>;

#if defined(CK_TILE_USE_WMMA)
// Memory Pipeline WMMA Types
using MemWmmaTestTypes = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Interwave, Mem>,
    std::tuple<Row, Row, Row, BF16, BF16, F32, BF16, I64, I64, I32, I16, I16, I16, Interwave, Mem>>;
#endif

// CompV3 Pipeline Types
using CompV3TestTypes = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Intrawave, CompV3>,
    std::tuple<Row,
               Row,
               Row,
               BF16,
               BF16,
               F32,
               F16,
               I64,
               I64,
               I32,
               I16,
               I16,
               I16,
               Intrawave,
               CompV3>>;

#if defined(CK_TILE_USE_WMMA)
// CompV3 Pipeline WMMA Types
using CompV3WmmaTestTypes = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Intrawave, CompV3>,
    std::tuple<Row,
               Row,
               Row,
               BF16,
               BF16,
               F32,
               F16,
               I64,
               I64,
               I32,
               I16,
               I16,
               I16,
               Intrawave,
               CompV3>>;
#endif

// CompV4 Pipeline Types
using CompV4TestTypes = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Intrawave, CompV4>,
    std::tuple<Row,
               Row,
               Row,
               BF16,
               BF16,
               F32,
               F16,
               I64,
               I64,
               I32,
               I16,
               I16,
               I16,
               Intrawave,
               CompV4>>;

#if defined(CK_TILE_USE_WMMA)
// CompV4 Pipeline WMMA Types
using CompV4WmmaTestTypes = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Intrawave, CompV4>,
    std::tuple<Row,
               Row,
               Row,
               BF16,
               BF16,
               F32,
               F16,
               I64,
               I64,
               I32,
               I16,
               I16,
               I16,
               Intrawave,
               CompV4>>;
#endif

// CompV6 Pipeline Types
using CompV6TestTypes = ::testing::Types<
    std::tuple<Row, Row, Row, F16, F16, F32, F16, I64, I64, I32, I16, I16, I16, Intrawave, CompV6>,
    std::tuple<Row,
               Row,
               Row,
               BF16,
               BF16,
               F32,
               F16,
               I64,
               I64,
               I32,
               I16,
               I16,
               I16,
               Intrawave,
               CompV6>>;

// Persistent CompV3 Pipeline Types
using PersistentTestTypes = ::testing::Types<std::tuple<Row,
                                                        Col,
                                                        Row,
                                                        F16,
                                                        F16,
                                                        F32,
                                                        F16,
                                                        I64,
                                                        I64,
                                                        I32,
                                                        I16,
                                                        I16,
                                                        I16,
                                                        Intrawave,
                                                        CompV3,
                                                        Persistent>,
                                             std::tuple<Row,
                                                        Col,
                                                        Row,
                                                        F16,
                                                        F16,
                                                        F32,
                                                        F16,
                                                        I64,
                                                        I64,
                                                        I32,
                                                        I16,
                                                        I16,
                                                        I16,
                                                        Intrawave,
                                                        CompV3,
                                                        NonPersistent>>;

#if defined(CK_TILE_USE_WMMA)
// Persistent CompV3 Pipeline WMMA Types
using PersistentWmmaTestTypes = ::testing::Types<std::tuple<Row,
                                                            Col,
                                                            Row,
                                                            F16,
                                                            F16,
                                                            F32,
                                                            F16,
                                                            I64,
                                                            I64,
                                                            I32,
                                                            I16,
                                                            I16,
                                                            I16,
                                                            Intrawave,
                                                            CompV3,
                                                            Persistent>,
                                                 std::tuple<Row,
                                                            Col,
                                                            Row,
                                                            F16,
                                                            F16,
                                                            F32,
                                                            F16,
                                                            I64,
                                                            I64,
                                                            I32,
                                                            I16,
                                                            I16,
                                                            I16,
                                                            Intrawave,
                                                            CompV3,
                                                            NonPersistent>>;
#endif

// ----------------------------------------------------------------------------
// Test Suite Registrations
// ----------------------------------------------------------------------------

TYPED_TEST_SUITE(TestGemmMem, MemTestTypes);
#if defined(CK_TILE_USE_WMMA)
TYPED_TEST_SUITE(TestGemmMemWmma, MemWmmaTestTypes);
#endif
TYPED_TEST_SUITE(TestGemmCompV3, CompV3TestTypes);
#if defined(CK_TILE_USE_WMMA)
TYPED_TEST_SUITE(TestGemmCompV3Wmma, CompV3WmmaTestTypes);
#endif
TYPED_TEST_SUITE(TestGemmCompV4, CompV4TestTypes);
#if defined(CK_TILE_USE_WMMA)
TYPED_TEST_SUITE(TestGemmCompV4Wmma, CompV4WmmaTestTypes);
#endif
TYPED_TEST_SUITE(TestGemmCompV6, CompV6TestTypes);
TYPED_TEST_SUITE(TestGemmPersistent, PersistentTestTypes);
#if defined(CK_TILE_USE_WMMA)
TYPED_TEST_SUITE(TestGemmPersistentWmma, PersistentWmmaTestTypes);
#endif

// ============================================================================
// Memory Pipeline Tests (Mem)
// ============================================================================

#define TEST_SUITE_NAME TestGemmMem

TYPED_TEST(TEST_SUITE_NAME, SmallM_SingleRow)
{
    std::vector<int> Ms{1};
    constexpr int N = 1024;
    constexpr int K = TestFixture::K_Tile * 2;

    for(int M : Ms)
    {
        if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                    ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
        }
        else
        {
            this->Run(M, N, K);
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, ExactlyTwoTiles_M)
{
    this->Run(TestFixture::M_Tile * 2, TestFixture::N_Tile, TestFixture::K_Tile * 2);
}

TYPED_TEST(TEST_SUITE_NAME, ExactlyTwoTiles_N)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile * 2, TestFixture::K_Tile * 2);
}

TYPED_TEST(TEST_SUITE_NAME, ExactlyTwoTiles_K)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile * 2);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_512x1024x512)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, Square_1024x1024x1024)
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_2048x2048x2048)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, VeryLargeMatrix_4096x4096x4096)
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, TallSkinny_4096x128x1024)
{
    constexpr int M = 4096;
    constexpr int N = 128;
    constexpr int K = 1024;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, ShortWide_128x4096x1024)
{
    constexpr int M = 128;
    constexpr int N = 4096;
    constexpr int K = 1024;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, DeepNarrow_2048x2048x8192)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 8192;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StressTest_ExtremelyTallMatrix)
{
    constexpr int M = 16384;
    constexpr int N = 64;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StressTest_ExtremelyWideMatrix)
{
    constexpr int M = 64;
    constexpr int N = 16384;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StressTest_VeryDeepK)
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 16384;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME

#if defined(CK_TILE_USE_WMMA)
// ============================================================================
// Memory Pipeline Tests with WMMA
// ============================================================================

#define TEST_SUITE_NAME TestGemmMemWmma

TYPED_TEST(TEST_SUITE_NAME, SingleTile_WMMA)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_WMMA)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_WMMA)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME
#endif // CK_TILE_USE_WMMA

// ============================================================================
// Compute V3 Pipeline Tests
// ============================================================================

#define TEST_SUITE_NAME TestGemmCompV3

TYPED_TEST(TEST_SUITE_NAME, SmallM_CompV3)
{
    std::vector<int> Ms{1, 2};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile_CompV3)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, MidLargeM_CompV3)
{
    std::vector<int> Ms{127, 255};
    constexpr int N = 1024;

    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    constexpr int VecLoadSize = (std::is_same_v<typename TestFixture::ADataType, ck_tile::fp8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::bf8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::int8_t>)
                                    ? 16
                                    : 8;

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                if(M % VecLoadSize == 0)
                {
                    this->Run(M, N, K);
                }
                else
                {
                    EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                }
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, Regular_CompV3)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_CompV3)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, BatchedSmall_CompV3)
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME

#if defined(CK_TILE_USE_WMMA)
// ============================================================================
// Compute V3 Pipeline Tests with WMMA
// ============================================================================

#define TEST_SUITE_NAME TestGemmCompV3Wmma

TYPED_TEST(TEST_SUITE_NAME, SmallM_CompV3Wmma)
{
    std::vector<int> Ms{1, 2};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile_CompV3Wmma)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_CompV3Wmma)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_CompV3Wmma)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME
#endif // CK_TILE_USE_WMMA

// ============================================================================
// Compute V4 Pipeline Tests
// ============================================================================

#define TEST_SUITE_NAME TestGemmCompV4

TYPED_TEST(TEST_SUITE_NAME, SmallM_CompV4)
{
    std::vector<int> Ms{1, 2};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile_CompV4)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_CompV4)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_CompV4)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME

#if defined(CK_TILE_USE_WMMA)
// ============================================================================
// Compute V4 Pipeline Tests with WMMA
// ============================================================================

#define TEST_SUITE_NAME TestGemmCompV4Wmma

TYPED_TEST(TEST_SUITE_NAME, SingleTile_CompV4Wmma)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_CompV4Wmma)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_CompV4Wmma)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME
#endif // CK_TILE_USE_WMMA

// ============================================================================
// Compute V6 Pipeline Tests
// ============================================================================

#define TEST_SUITE_NAME TestGemmCompV6

TYPED_TEST(TEST_SUITE_NAME, SmallM_CompV6)
{
    std::vector<int> Ms{1, 2};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile_CompV6)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, MidLargeM_CompV6)
{
    std::vector<int> Ms{127, 255};
    constexpr int N = 1024;

    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    constexpr int VecLoadSize = (std::is_same_v<typename TestFixture::ADataType, ck_tile::fp8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::bf8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::int8_t>)
                                    ? 16
                                    : 8;

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                if(M % VecLoadSize == 0)
                {
                    this->Run(M, N, K);
                }
                else
                {
                    EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                }
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, Regular_CompV6)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_CompV6)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME

// ============================================================================
// Persistent Kernel Tests
// ============================================================================

#define TEST_SUITE_NAME TestGemmPersistent

TYPED_TEST(TEST_SUITE_NAME, SmallM_Persistent)
{
    std::vector<int> Ms{1, 2};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile_Persistent)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_Persistent)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_Persistent)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME

#if defined(CK_TILE_USE_WMMA)
// ============================================================================
// Persistent Kernel Tests with WMMA
// ============================================================================

#define TEST_SUITE_NAME TestGemmPersistentWmma

TYPED_TEST(TEST_SUITE_NAME, SmallM_PersistentWmma)
{
    std::vector<int> Ms{1, 2};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 4})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile_PersistentWmma)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular_PersistentWmma)
{
    constexpr int M = 512;
    constexpr int N = 1024;
    constexpr int K = 512;
    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix_PersistentWmma)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    this->Run(M, N, K);
}

#undef TEST_SUITE_NAME
#endif // CK_TILE_USE_WMMA
