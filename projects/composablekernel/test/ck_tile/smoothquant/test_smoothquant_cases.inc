// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef TEST_SMOOTHQUANT_CASES_INC
#define TEST_SMOOTHQUANT_CASES_INC
TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m99_n13)
{
    ck_tile::index_t m = 99;
    ck_tile::index_t n = 13;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m17_n16)
{
    ck_tile::index_t m = 17;
    ck_tile::index_t n = 16;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m1_n100)
{
    ck_tile::index_t m = 1;
    ck_tile::index_t n = 100;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m4_n128)
{
    ck_tile::index_t m = 4;
    ck_tile::index_t n = 128;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m80_n127)
{
    ck_tile::index_t m = 80;
    ck_tile::index_t n = 127;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m22_n255)
{
    ck_tile::index_t m        = 22;
    ck_tile::index_t n        = 255;
    ck_tile::index_t x_stride = 256;

    this->Run(m, n, x_stride);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m7_n599)
{
    ck_tile::index_t m = 7;
    ck_tile::index_t n = 599;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m33_n313)
{
    ck_tile::index_t m        = 33;
    ck_tile::index_t n        = 313;
    ck_tile::index_t x_stride = 1000;

    this->Run(m, n, x_stride);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m11_n510)
{
    ck_tile::index_t m = 11;
    ck_tile::index_t n = 510;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m171_n676)
{
    ck_tile::index_t m        = 171;
    ck_tile::index_t n        = 676;
    ck_tile::index_t x_stride = 818;

    this->Run(m, n, x_stride);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m91_n636)
{
    ck_tile::index_t m = 91;
    ck_tile::index_t n = 636;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m12_n768)
{
    ck_tile::index_t m        = 12;
    ck_tile::index_t n        = 768;
    ck_tile::index_t x_stride = 800;

    this->Run(m, n, x_stride);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m100_n766)
{
    ck_tile::index_t m        = 100;
    ck_tile::index_t n        = 766;
    ck_tile::index_t x_stride = 812;

    this->Run(m, n, x_stride);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m31_n1024)
{
    ck_tile::index_t m = 31;
    ck_tile::index_t n = 1024;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m64_n1000)
{
    ck_tile::index_t m        = 64;
    ck_tile::index_t n        = 1000;
    ck_tile::index_t x_stride = 1004;

    this->Run(m, n, x_stride);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m8_n1501)
{
    ck_tile::index_t m = 8;
    ck_tile::index_t n = 1501;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m3_n1826)
{
    ck_tile::index_t m = 3;
    ck_tile::index_t n = 1826;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m5_n2040)
{
    ck_tile::index_t m = 5;
    ck_tile::index_t n = 2040;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m7_n2734)
{
    ck_tile::index_t m = 7;
    ck_tile::index_t n = 2734;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m1_n3182)
{
    ck_tile::index_t m = 1;
    ck_tile::index_t n = 3182;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m9_n4096)
{
    ck_tile::index_t m = 9;
    ck_tile::index_t n = 4096;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m3_n8192)
{
    ck_tile::index_t m = 3;
    ck_tile::index_t n = 8192;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m1_n10547)
{
    ck_tile::index_t m = 1;
    ck_tile::index_t n = 10547;

    this->Run(m, n);
}

TYPED_TEST(TEST_SUITE_NAME, Smoothqauant_m3_n17134)
{
    ck_tile::index_t m = 3;
    ck_tile::index_t n = 17134;

    this->Run(m, n);
}

#endif
