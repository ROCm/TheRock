// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_256x512x256)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_512x256x256)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_512x512x256)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_256x256x256)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_512x768x256)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_512x1280x256)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_256x1280x256)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_768x512x256)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_1280x512x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch1Default_1280x256x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 256;
    constexpr int kBatch = 1;

    EXPECT_EQ(this->Run(M, N, K, kBatch), true);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_512x512x512)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_256x512x256)
{
    constexpr int M      = 256;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_512x256x256)
{
    constexpr int M      = 512;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_512x512x256)
{
    constexpr int M      = 512;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_256x256x256)
{
    constexpr int M      = 256;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_512x768x256)
{
    constexpr int M      = 512;
    constexpr int N      = 768;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_512x1280x256)
{
    constexpr int M      = 512;
    constexpr int N      = 1280;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_256x1280x256)
{
    constexpr int M      = 256;
    constexpr int N      = 1280;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_768x512x256)
{
    constexpr int M      = 768;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_1280x512x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 512;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}

TYPED_TEST(TestCkTileGemmMultiABD, TestCkTileGemmMultiABDKBatch2Default_1280x256x256)
{
    constexpr int M      = 1280;
    constexpr int N      = 256;
    constexpr int K      = 512;
    constexpr int kBatch = 2;

    EXPECT_THROW(this->Run(M, N, K, kBatch), std::runtime_error);
}
