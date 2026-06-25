// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef TEST_MOE_SMOOTHQUANT_CASES_INC
#define TEST_MOE_SMOOTHQUANT_CASES_INC
TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t99_h13)
{
    ck_tile::index_t tokens      = 99;
    ck_tile::index_t hidden_size = 13;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t17_h16)
{
    ck_tile::index_t tokens      = 17;
    ck_tile::index_t hidden_size = 16;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t1_h100)
{
    ck_tile::index_t tokens      = 1;
    ck_tile::index_t hidden_size = 100;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t4_h128)
{
    ck_tile::index_t tokens      = 4;
    ck_tile::index_t hidden_size = 128;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t80_h127)
{
    ck_tile::index_t tokens      = 80;
    ck_tile::index_t hidden_size = 127;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t22_h255)
{
    ck_tile::index_t tokens      = 22;
    ck_tile::index_t hidden_size = 255;
    ck_tile::index_t stride      = 256;

    this->Run(tokens, hidden_size, stride);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t7_h599)
{
    ck_tile::index_t tokens      = 7;
    ck_tile::index_t hidden_size = 599;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t19_h512)
{
    ck_tile::index_t tokens      = 19;
    ck_tile::index_t hidden_size = 512;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t33_h313)
{
    ck_tile::index_t tokens      = 33;
    ck_tile::index_t hidden_size = 313;
    ck_tile::index_t stride      = 1000;

    this->Run(tokens, hidden_size, stride);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t11_h510)
{
    ck_tile::index_t tokens      = 11;
    ck_tile::index_t hidden_size = 510;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t171_h676)
{
    ck_tile::index_t tokens      = 171;
    ck_tile::index_t hidden_size = 676;
    ck_tile::index_t stride      = 818;

    this->Run(tokens, hidden_size, stride);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t12_h768)
{
    ck_tile::index_t tokens      = 12;
    ck_tile::index_t hidden_size = 768;
    ck_tile::index_t stride      = 800;

    this->Run(tokens, hidden_size, stride);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t100_h766)
{
    ck_tile::index_t tokens      = 100;
    ck_tile::index_t hidden_size = 766;
    ck_tile::index_t stride      = 812;

    this->Run(tokens, hidden_size, stride);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t31_h1024)
{
    ck_tile::index_t tokens      = 31;
    ck_tile::index_t hidden_size = 1024;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t64_h1000)
{
    ck_tile::index_t tokens      = 64;
    ck_tile::index_t hidden_size = 1000;
    ck_tile::index_t stride      = 1004;

    this->Run(tokens, hidden_size, stride);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t8_h1501)
{
    ck_tile::index_t tokens      = 8;
    ck_tile::index_t hidden_size = 1501;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t3_h1826)
{
    ck_tile::index_t tokens      = 3;
    ck_tile::index_t hidden_size = 1826;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t5_h2040)
{
    ck_tile::index_t tokens      = 5;
    ck_tile::index_t hidden_size = 2040;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t7_h2734)
{
    ck_tile::index_t tokens      = 7;
    ck_tile::index_t hidden_size = 2734;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t1_h3182)
{
    ck_tile::index_t tokens      = 1;
    ck_tile::index_t hidden_size = 3182;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t9_h4096)
{
    ck_tile::index_t tokens      = 9;
    ck_tile::index_t hidden_size = 4096;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t3_h8192)
{
    ck_tile::index_t tokens      = 3;
    ck_tile::index_t hidden_size = 8192;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t1_h10547)
{
    ck_tile::index_t tokens      = 1;
    ck_tile::index_t hidden_size = 10547;

    this->Run(tokens, hidden_size);
}

TYPED_TEST(TEST_SUITE_NAME, MoeSmoothquant_t3_h17134)
{
    ck_tile::index_t tokens      = 3;
    ck_tile::index_t hidden_size = 17134;

    this->Run(tokens, hidden_size);
}

#endif
