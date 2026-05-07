// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef TEST_PERMUTE_CASES_INC
#define TEST_PERMUTE_CASES_INC
TYPED_TEST(TEST_SUITE_NAME, Permute1)
{
    std::vector<ck_tile::index_t> shape{3, 8};
    std::string perm{"1,0"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute2)
{
    std::vector<ck_tile::index_t> shape{48, 6, 8};
    std::string perm{"2,1,0"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute3)
{
    std::vector<ck_tile::index_t> shape{24, 128, 3};
    std::string perm{"0,2,1"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute4)
{
    std::vector<ck_tile::index_t> shape{4, 10, 7, 6};
    std::string perm{"0,2,3,1"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute5)
{
    std::vector<ck_tile::index_t> shape{8, 24, 36, 10};
    std::string perm{"3,1,2,0"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute6)
{
    std::vector<ck_tile::index_t> shape{8, 1, 36, 4};
    std::string perm{"2,1,0,3"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute7)
{
    std::vector<ck_tile::index_t> shape{5, 10, 16, 2, 36, 4};
    std::string perm{"4,5,2,1,0,3"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute8)
{
    std::vector<ck_tile::index_t> shape{2, 32, 8, 3, 6, 2, 5, 4};
    std::string perm{"5,2,4,7,1,6,3,0"};

    this->Run(shape, perm);
}
TYPED_TEST(TEST_SUITE_NAME, Permute9)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{3, 6, 4, 32, 16, 2, 8};
    std::string perm{"0,1,4,2,5,3,6"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute10)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{5, 10, 4, 32, 8, 2, 8};
    std::string perm{"0,1,4,2,5,3,6"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute11)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{3, 8, 4, 16, 16, 4, 8};
    std::string perm{"0,1,4,2,5,3,6"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute12)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{3, 6, 4, 32, 16, 2, 8};
    std::string perm{"0,1,2,4,5,3,6"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute13)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{5, 10, 4, 32, 8, 2, 8};
    std::string perm{"0,1,2,4,5,3,6"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute14)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{3, 8, 4, 16, 16, 4, 8};
    std::string perm{"0,1,2,4,5,3,6"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute15)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{2, 8, 16, 8, 4, 8};
    std::string perm{"0,1,3,4,2,5"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute16)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{1, 24, 32, 16, 2, 8};
    std::string perm{"0,1,3,4,2,5"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute17)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{3, 8};
    std::string perm{"1,0"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute18)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{48, 6, 8};
    std::string perm{"2,1,0"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute19)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{24, 128, 3};
    std::string perm{"0,2,1"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute20)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{4, 10, 7, 6};
    std::string perm{"0,2,3,1"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute21)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{8, 24, 36, 10};
    std::string perm{"3,1,2,0"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute22)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{8, 1, 36, 4};
    std::string perm{"2,1,0,3"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute23)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{5, 10, 16, 2, 36, 4};
    std::string perm{"4,5,2,1,0,3"};

    this->Run(shape, perm);
}

TYPED_TEST(TEST_SUITE_NAME, Permute24)
{
    if constexpr(!std::is_same_v<TypeParam, F16Types>)
    {
        GTEST_SKIP() << "Skipping this test: Only run with fp16";
    }

    std::vector<ck_tile::index_t> shape{2, 32, 8, 3, 6, 2, 5, 4};
    std::string perm{"5,2,4,7,1,6,3,0"};

    this->Run(shape, perm);
}

#endif
