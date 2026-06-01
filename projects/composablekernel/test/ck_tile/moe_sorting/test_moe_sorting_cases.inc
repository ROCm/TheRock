// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef TEST_MOE_SORTING_CASES_INC
#define TEST_MOE_SORTING_CASES_INC
TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase1)
{
    int tokens       = 80;
    int local_tokens = -1;
    int num_experts  = 17;
    int topk         = 4;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 16;
    int moe_buf_elem_bytes = 4;
#else
    int64_t moe_buf_size = 16;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase2)
{
    int tokens       = 111;
    int local_tokens = -1;
    int num_experts  = 117;
    int topk         = 4;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 4;
    int moe_buf_elem_bytes = 4;
#else
    int64_t moe_buf_size = 4;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase3)
{
    int tokens       = 1000;
    int local_tokens = -1;
    int num_experts  = 55;
    int topk         = 4;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 1024;
    int moe_buf_elem_bytes = 1;
#else
    int64_t moe_buf_size = 1024;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, DISABLED_MoeSortingCase4)
{
    int tokens       = 99;
    int local_tokens = -1;
    int num_experts  = 120;
    int topk         = 4;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 10244;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 10244;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase5)
{
    int tokens       = 175;
    int local_tokens = -1;
    int num_experts  = 8;
    int topk         = 8;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase6)
{
    int tokens       = 65;
    int local_tokens = -1;
    int num_experts  = 8;
    int topk         = 2;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase7)
{
    int tokens       = 1;
    int local_tokens = -1;
    int num_experts  = 65;
    int topk         = 4;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase8)
{
    int tokens       = 31;
    int local_tokens = -1;
    int num_experts  = 19;
    int topk         = 15;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase9)
{
    int tokens       = 81;
    int local_tokens = -1;
    int num_experts  = 37;
    int topk         = 7;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase10)
{
    int tokens       = 23;
    int local_tokens = -1;
    int num_experts  = 1;
    int topk         = 1;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase11)
{
    int tokens       = 127;
    int local_tokens = -1;
    int num_experts  = 99;
    int topk         = 19;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase12)
{
    int tokens       = 71;
    int local_tokens = -1;
    int num_experts  = 11;
    int topk         = 11;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase13)
{
    int tokens       = 1;
    int local_tokens = -1;
    int num_experts  = 1;
    int topk         = 1;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase14)
{
    int tokens       = 99;
    int local_tokens = -1;
    int num_experts  = 2;
    int topk         = 1;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase15)
{
    int tokens       = 333;
    int local_tokens = -1;
    int num_experts  = 99;
    int topk         = 13;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase16)
{
    int tokens       = 11;
    int local_tokens = -1;
    int num_experts  = 256;
    int topk         = 5;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase17)
{
    int tokens       = 64;
    int local_tokens = -1;
    int num_experts  = 455;
    int topk         = 8;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase18)
{
    int tokens       = 777;
    int local_tokens = -1;
    int num_experts  = 802;
    int topk         = 99;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase19)
{
    int tokens       = 4097;
    int local_tokens = -1;
    int num_experts  = 906;
    int topk         = 51;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase20)
{
    int tokens       = 128;
    int local_tokens = 6;
    int num_experts  = 32;
    int topk         = 5;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 262144;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 262144;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase21)
{
    int tokens       = 13;
    int local_tokens = -1;
    int num_experts  = 64;
    int topk         = 3;
    int unit_size    = 32;
    std::vector<int> local_eid{4, 5, 6, 7, 8, 9, 10, 11};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase22)
{
    int tokens       = 99;
    int local_tokens = -1;
    int num_experts  = 33;
    int topk         = 9;
    int unit_size    = 32;
    std::vector<int> local_eid{6, 10, 11, 15, 19};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase23)
{
    int tokens       = 80;
    int local_tokens = -1;
    int num_experts  = 99;
    int topk         = 10;
    int unit_size    = 32;
    std::vector<int> local_eid{0, 8, 12, 33};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase24)
{
    int tokens       = 11;
    int local_tokens = -1;
    int num_experts  = 256;
    int topk         = 5;
    int unit_size    = 32;
    std::vector<int> local_eid{99, 110, 129};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase25)
{
    int tokens       = 128;
    int local_tokens = -1;
    int num_experts  = 128;
    int topk         = 6;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 163840;
    int moe_buf_elem_bytes = 1;
#else
    int64_t moe_buf_size = 163840;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase26)
{
    int tokens      = 8192;
    int num_experts = 32;
    int topk        = 5;
    int unit_size   = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int local_tokens       = 11;
    int moe_buf_interm_dim = 163840;
    int moe_buf_elem_bytes = 2;
#else
    int local_tokens     = -1;
    int64_t moe_buf_size = 163840;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase27)
{
    int tokens       = 8192;
    int local_tokens = 12;
    int num_experts  = 32;
    int topk         = 8;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 163840;
    int moe_buf_elem_bytes = 1;
#else
    int64_t moe_buf_size = 163840;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase28)
{
    int tokens      = 8192;
    int num_experts = 256;
    int topk        = 5;
    int unit_size   = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int local_tokens       = 12;
    int moe_buf_interm_dim = 163840;
    int moe_buf_elem_bytes = 0;
#else
    int local_tokens     = -1;
    int64_t moe_buf_size = 163840;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase29)
{
    int tokens      = 8192;
    int num_experts = 256;
    int topk        = 8;
    int unit_size   = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int local_tokens       = 8;
    int moe_buf_interm_dim = 163840;
    int moe_buf_elem_bytes = 2;
#else
    int local_tokens     = -1;
    int64_t moe_buf_size = 163840;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase30)
{
    int tokens      = 163840;
    int num_experts = 256;
    int topk        = 8;
    int unit_size   = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int local_tokens       = 4;
    int moe_buf_interm_dim = 163840;
    int moe_buf_elem_bytes = 4;
#else
    int local_tokens     = -1;
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase31)
{
    int tokens       = 12;
    int local_tokens = 3;
    int num_experts  = 256;
    int topk         = 5;
    int unit_size    = 32;
    std::vector<int> local_eid{9, 10, 199, 145};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase32)
{
    int tokens       = 67;
    int local_tokens = 9;
    int num_experts  = 555;
    int topk         = 5;
    int unit_size    = 32;
    std::vector<int> local_eid{19, 23, 24, 25, 26, 99};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase33)
{
    int tokens       = 99;
    int local_tokens = 93;
    int num_experts  = 121;
    int topk         = 4;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 10244;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 10244;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase34)
{
    int tokens       = 536;
    int local_tokens = 345;
    int num_experts  = 802;
    int topk         = 99;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase35)
{
    int tokens       = 331;
    int local_tokens = 39;
    int num_experts  = 83;
    int topk         = 33;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase36)
{
    int tokens       = 765;
    int local_tokens = 654;
    int num_experts  = 783;
    int topk         = 8;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase37)
{
    int tokens       = 23;
    int local_tokens = 9;
    int num_experts  = 1;
    int topk         = 1;
    int unit_size    = 32;
    std::vector<int> local_eid{};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase38)
{
    int tokens       = 7;
    int local_tokens = 0;
    int num_experts  = 89;
    int topk         = 1;
    int unit_size    = 32;
    std::vector<int> local_eid{0, 8, 12, 33};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase39)
{
    int tokens       = 61;
    int local_tokens = 0;
    int num_experts  = 333;
    int topk         = 99;
    int unit_size    = 32;
    std::vector<int> local_eid{0, 8, 12, 33};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 0;
    int moe_buf_elem_bytes = 2;
#else
    int64_t moe_buf_size = 0;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}

TYPED_TEST(TEST_SUITE_NAME, MoeSortingCase40)
{
    int tokens       = 133940;
    int local_tokens = 2;
    int num_experts  = 256;
    int topk         = 17;
    int unit_size    = 32;
    std::vector<int> local_eid{0, 8, 12, 33};
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = 133940;
    int moe_buf_elem_bytes = 1;
#else
    int64_t moe_buf_size = 133940;
#endif

    this->RunSingle(tokens,
                    local_tokens,
                    num_experts,
                    topk,
                    unit_size,
                    local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                    moe_buf_interm_dim,
                    moe_buf_elem_bytes
#else
                    moe_buf_size
#endif
    );
}
#endif
