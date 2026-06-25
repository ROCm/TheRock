// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef TEST_GEMM_PIPELINE_UT_CASES_INC
#define TEST_GEMM_PIPELINE_UT_CASES_INC

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle)
{
    constexpr int M           = 2048;
    constexpr int N           = 4096;
    constexpr int K           = 5120;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_128x128x128)
{
    constexpr int M           = 128;
    constexpr int N           = 128;
    constexpr int K           = 128;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_128x128x4096)
{
    constexpr int M           = 128;
    constexpr int N           = 128;
    constexpr int K           = 4096;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_128x2048x128)
{
    constexpr int M           = 128;
    constexpr int N           = 2048;
    constexpr int K           = 128;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_128x2048x4096)
{
    constexpr int M           = 128;
    constexpr int N           = 2048;
    constexpr int K           = 4096;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_1024x128x128)
{
    constexpr int M           = 1024;
    constexpr int N           = 128;
    constexpr int K           = 128;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_1024x128x4096)
{
    constexpr int M           = 1024;
    constexpr int N           = 128;
    constexpr int K           = 4096;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_1024x2048x128)
{
    constexpr int M           = 1024;
    constexpr int N           = 2048;
    constexpr int K           = 128;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, GemmPreshuffle_1024x2048x4096)
{
    constexpr int M           = 1024;
    constexpr int N           = 2048;
    constexpr int K           = 4096;
    constexpr bool PadM       = false;
    constexpr bool PadN       = false;
    constexpr bool PadK       = false;
    constexpr bool Preshuffle = true;
    this->template Run<PadM, PadN, PadK, Preshuffle>(M, N, K);
}

#endif
