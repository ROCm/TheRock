// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaFwdFp16 // or FmhaFwdBf16 / FmhaFwdFp8Bf16 / FmhaFwdFp32
#endif

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

template <typename T>
struct TestConfigs
{
    static constexpr auto HDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{64, -1},
        std::tuple{96, 128},
        std::tuple{128, -1},
        std::tuple{192, 128},
        std::tuple{192, -1},
        std::tuple{256, -1},
    };
    static constexpr auto SplitKVHDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{64, -1},
        std::tuple{96, -1},
        std::tuple{128, -1},
        std::tuple{256, -1},
    };
    static constexpr auto AppendKVHDimValues = std::array{
        std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto ModeValues        = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues = std::array{true};
    static constexpr auto qscale_str        = "n";
    static constexpr bool def_lse           = true;
    static constexpr bool def_is_v_rowmajor = true;
    static int adjust_seqlen(int seqlen) { return seqlen; }
};

template <>
struct TestConfigs<FmhaFwdFp8Bf16>
{
    static constexpr auto HDimValues =
        std::array{std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array{std::tuple{64, -1}, std::tuple{128, -1}};
    static constexpr auto AppendKVHDimValues = std::array{std::tuple{64, -1}, std::tuple{128, -1}};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "pt";
    static constexpr bool def_lse            = false;
    static constexpr bool def_is_v_rowmajor  = true;
    // When there are no fp8 instances with padding, pad seqlen to avoid skipping most of the tests:
    // return ck_tile::integer_least_multiple(seqlen, 128);
    static int adjust_seqlen(int seqlen) { return seqlen; }
};

template <>
struct TestConfigs<FmhaFwdFp32>
{
    static constexpr auto HDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{48, -1},
        std::tuple{64, -1},
        std::tuple{96, 128},
        std::tuple{128, -1},
        std::tuple{192, -1},
        std::tuple{256, -1},
    };
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "n";
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = true;
    static int adjust_seqlen(int seqlen) { return seqlen; }
};

static auto HDimValues           = ValuesIn(TestConfigs<DataTypeConfig>::HDimValues);
static auto SplitKVHDimValues    = ValuesIn(TestConfigs<DataTypeConfig>::SplitKVHDimValues);
static auto AppendKVHDimValues   = ValuesIn(TestConfigs<DataTypeConfig>::AppendKVHDimValues);
static auto ModeValues           = ValuesIn(TestConfigs<DataTypeConfig>::ModeValues);
static auto IsVRowmajorValues    = ValuesIn(TestConfigs<DataTypeConfig>::IsVRowmajorValues);
constexpr static auto qscale_str = TestConfigs<DataTypeConfig>::qscale_str;
constexpr bool def_lse           = TestConfigs<DataTypeConfig>::def_lse;
constexpr bool def_is_v_rowmajor = TestConfigs<DataTypeConfig>::def_is_v_rowmajor;
int adjust_seqlen(int seqlen) { return TestConfigs<DataTypeConfig>::adjust_seqlen(seqlen); }
constexpr auto init_method = "uf";

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run long tests (from smoke_test_fwd.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_LONG_TESTS)

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == fwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, fwd_result::success);                   \
    } while(0)

const ck_tile::stream_config stream_config{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

#define COMMON_ARGS                                                                           \
    init_method, static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))), 1, \
        stream_config

auto EnableTestIf(bool condition)
{
    return ValuesIn(condition ? std::vector<bool>{true} : std::vector<bool>{});
}

class AllLong : public TestWithParam<
                    std::tuple<bool,
                               std::tuple<int, int>,
                               bool,
                               bool,
                               mode_enum,
                               bool,
                               std::string,
                               float,
                               std::tuple<int, int, int, int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AllLong);

// Test cases from example/ck_tile/01_fmha/script/smoke_test_fwd.sh

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    AllLong,
    Combine(EnableTestIf(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_LONG_TESTS))),
            HDimValues,
            Bool(),
            IsVRowmajorValues,
            ModeValues,
            Bool(),
            Values("n", "e", "a"),
            Values(0.0f, 0.2f),
            Values(std::tuple{2, 2, 1, 16, -1, 55, 256, -1, "0"},
                   std::tuple{1, 3, -1, -1, -1, 100, 51, -1, "0"},
                   std::tuple{2, 1, -1, 16, -1, 99, 256, -1, "1"},
                   std::tuple{1, 2, 1, -1, -1, 1024, 256, -1, "2"},
                   std::tuple{2, 1, -1, -1, 24, 3, 99, -1, "2"},
                   std::tuple{3, 2, 1, -1, -1, 200, 520, -1, "t:128,30"},
                   std::tuple{2, 1, -1, -1, -1, 99, 32, -1, "b:4,35"},
                   std::tuple{1, 2, 1, -1, -1, 33, 0, -1, "2"},
                   std::tuple{1, 2, 1, -1, -1, 1, 10, 32, "2"})));

TEST_P(AllLong, DataTypeConfig)
{
    auto [_, hdims, perm, is_v_rowmajor, mode, lse, bias_str, p_drop, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                                        = hdims;
    auto [batch, nhead, nhead_k, hdim_q_, hdim_v_, seqlen_q, seqlen_k, seqlen_kpad, mask_str] =
        dims_mask;

    hdim_q = hdim_q_ == -1 ? hdim_q : hdim_q_;
    hdim_v = hdim_v_ == -1 ? hdim_v : hdim_v_;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,             // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {seqlen_kpad}, // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               0,             // rotary_dim
                                               perm,          // i_perm
                                               perm,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               lse,           // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               bias_str,      // bias_str
                                               p_drop,        // p_drop
                                               123,           // drop_seed
                                               1024,          // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

// ---------------------------------------------------------------
// Negative tests: padding not supported with appendkv/splitkv/pagedkv
// ---------------------------------------------------------------

#if CK_TILE_FMHA_FWD_APPENDKV_API
TEST(TestCkTileFmhaFwd, AppendKvWithBatchEffLensShouldFail)
{
    // batch mode effective lengths simulate padding
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        2,          // batch
        4,          // nhead
        -1,         // nhead_k
        {128},      // seqlen_qs
        {128},      // seqlen_ks
        64,         // hdim_q
        64,         // hdim_v
        32,         // seqlen_knew -> triggers appendkv
        {},         // seqlen_qpads
        {},         // seqlen_kpads
        {100, 120}, // q_eff_lens_per_batch
        {90, 110},  // kv_eff_lens_per_batch
        0,          // rotary_dim
        true,       // i_perm
        true,       // o_perm
        0,          // scale_s
        0,          // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask
        qscale_str,
        true, // is_rotary_interleaved
        1,    // num_splits
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        0,
        stream_config);
    ASSERT_EQ(result, fwd_result::invalid_args);
}
#endif

#if CK_TILE_FMHA_FWD_SPLITKV_API
TEST(TestCkTileFmhaFwd, SplitKvWithGroupPaddingShouldFail)
{
    // group mode physical padding
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::group,
        2,          // batch
        4,          // nhead
        -1,         // nhead_k
        {96, 120},  // seqlen_qs logical
        {96, 120},  // seqlen_ks logical
        64,         // hdim_q
        64,         // hdim_v
        0,          // seqlen_knew
        {128, 128}, // seqlen_qpads
        {128, 128}, // seqlen_kpads
        {},         // q_eff
        {},         // kv_eff
        0,          // rotary_dim
        true,       // i_perm
        true,       // o_perm
        0,          // scale_s
        0,          // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,
        0,
        0,
        false,
        "0",
        qscale_str,
        true,
        2, // num_splits (>1 triggers splitkv)
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        0,
        stream_config);
    ASSERT_EQ(result, fwd_result::invalid_args);
}
#endif

#if CK_TILE_FMHA_FWD_PAGEDKV_API
TEST(TestCkTileFmhaFwd, PagedKvWithGroupPaddingShouldFail)
{
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::group,
        2,
        4,
        -1,
        {80, 100},
        {80, 100},
        64,
        64,
        0,         // seqlen_knew
        {96, 128}, // seqlen_qpads
        {96, 128}, // seqlen_kpads
        {},
        {},
        0,
        true,
        true,
        0,
        0,
        def_is_v_rowmajor,
        def_lse,
        128, // page_block_size triggers pagedkv
        false,
        "n",
        0.0f,
        0,
        0,
        false,
        "0",
        qscale_str,
        true,
        1,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        0,
        stream_config);
    ASSERT_EQ(result, fwd_result::invalid_args);
}
#endif

class HDimPadding
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      bool,
                                      mode_enum,
                                      std::tuple<int, int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         HDimPadding,
                         Combine(Values(std::tuple{24, 48},
                                        std::tuple{120, 160},
                                        std::tuple{256, 108},
                                        std::tuple{40, 64}),
                                 Bool(),
                                 IsVRowmajorValues,
                                 ModeValues,
                                 Values(std::tuple{1, 4, 2, 480, -1, -1, "0"},
                                        std::tuple{2, 2, -1, 300, 400, 512, "t:64,64"},
                                        std::tuple{1, 4, 1, 512, 201, 256, "1"},
                                        std::tuple{1, 2, -1, 900, 256, -1, "0"},
                                        std::tuple{2, 1, -1, 256, 256, -1, "1"})));

TEST_P(HDimPadding, DataTypeConfig)
{
    auto [hdims, perm, is_v_rowmajor, mode, dims_mask]                      = GetParam();
    auto [hdim_q, hdim_v]                                                   = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, seqlen_kpad, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,             // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {seqlen_kpad}, // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               0,             // rotary_dim
                                               perm,          // i_perm
                                               perm,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               def_lse,       // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               "n",           // bias_str
                                               0.0f,          // p_drop
                                               0,             // drop_seed
                                               0,             // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class ElementwiseBias
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      mode_enum,
                                      std::string,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         ElementwiseBias,
                         Combine(HDimValues,
                                 Bool(), // layout of bias is controlled by i_perm
                                 ModeValues,
                                 Values("e:0", "e:1", "e:2"),
                                 Values(std::tuple{1, 4, 2, 1024, 100, "0"},
                                        std::tuple{3, 2, -1, 128, 256, "2"},
                                        std::tuple{2, 2, -1, 130, 499, "t:50,64"})));

TEST_P(ElementwiseBias, DataTypeConfig)
{
    auto [hdims, i_perm, mode, bias_str, dims_mask]            = GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               i_perm,            // i_perm
                                               false,             // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               bias_str,          // bias_str
                                               0.0f,              // p_drop
                                               0,                 // drop_seed
                                               0,                 // drop_offset
                                               false,             // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class Alibi : public TestWithParam<std::tuple<std::tuple<int, int>,
                                              mode_enum,
                                              std::string,
                                              std::tuple<int, int, int, int, int>,
                                              std::string>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         Alibi,
                         Combine(HDimValues,
                                 ModeValues,
                                 Values("a:0", "a:1"),
                                 Values(std::tuple{1, 3, 3, 1024, 1000},
                                        std::tuple{3, 5, 5, 128, 256},
                                        std::tuple{2, 8, 2, 300, 355}),
                                 Values("0", "t", "b", "t:50,64", "b:32,40")));

TEST_P(Alibi, DataTypeConfig)
{
    auto [hdims, mode, bias_str, dims, mask_str]     = GetParam();
    auto [hdim_q, hdim_v]                            = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               true,              // i_perm
                                               true,              // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               bias_str,          // bias_str
                                               0.0f,              // p_drop
                                               0,                 // drop_seed
                                               0,                 // drop_offset
                                               false,             // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class Dropout : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                mode_enum,
                                                float,
                                                std::tuple<uint64_t, uint64_t, bool>,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         Dropout,
                         Combine(HDimValues,
                                 ModeValues,
                                 Values(0.123f, 0.5f),
                                 Values(std::tuple{10, 123, false},
                                        std::tuple{34534564645, 7876878876864, true}),
                                 Values(std::tuple{2, 4, 2, 280, 512, "0"},
                                        std::tuple{3, 2, 2, 256, 128, "1"},
                                        std::tuple{4, 3, 1, 100, 768, "2"})));

TEST_P(Dropout, DataTypeConfig)
{
    auto [hdims, mode, p_drop, drop_seed_offset_prefs, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                         = hdims;
    auto [drop_seed, drop_offset, drop_prefs]                     = drop_seed_offset_prefs;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]    = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               false,             // i_perm
                                               false,             // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               "n",               // bias_str
                                               p_drop,            // p_drop
                                               drop_seed,         // drop_seed
                                               drop_offset,       // drop_offset
                                               drop_prefs,        // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#if CK_TILE_FMHA_FWD_PAGEDKV_API

class PagedKV : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                bool,
                                                bool,
                                                mode_enum,
                                                int,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PagedKV);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         PagedKV,
                         Combine(SplitKVHDimValues,
                                 Bool(),            // layouts of k and v are controlled by i_perm
                                 IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
                                 ModeValues,
                                 Values(128, 256),
                                 Values(std::tuple{2, 3, 1, 200, 1024, "0"},
                                        std::tuple{3, 2, -1, 128, 768, "2"},
                                        std::tuple{2, 2, -1, 230, 899, "t:50,64"})));

TEST_P(PagedKV, DataTypeConfig)
{
    auto [hdims, i_perm, is_v_rowmajor, mode, page_block_size, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                                 = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]            = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,               // seqlen_knew
                                               {-1},            // seqlen_qpads
                                               {-1},            // seqlen_kpads
                                               {},              // q_eff_lens_per_batch
                                               {},              // kv_eff_lens_per_batch
                                               0,               // rotary_dim
                                               i_perm,          // i_perm
                                               false,           // o_perm
                                               0,               // scale_s
                                               0,               // logits_soft_cap
                                               is_v_rowmajor,   // is_v_rowmajor
                                               false,           // lse
                                               page_block_size, // page_block_size
                                               false,           // use_cache_batch_idx
                                               "n",             // bias_str
                                               0.0f,            // p_drop
                                               0,               // drop_seed
                                               0,               // drop_offset
                                               false,           // drop_prefs
                                               mask_str,        // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_PAGEDKV_API

#if CK_TILE_FMHA_FWD_SPLITKV_API

class SplitKV : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                bool,
                                                bool,
                                                std::tuple<mode_enum, bool>,
                                                int,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SplitKV);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKV,
                         Combine(SplitKVHDimValues,
                                 Bool(),            // layouts of k and v are controlled by i_perm
                                 IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
                                 Values(std::tuple{mode_enum::batch, false},
                                        std::tuple{mode_enum::batch, true},
                                        std::tuple{mode_enum::group, false}),
                                 Values(3, 4),
                                 Values(std::tuple{4, 3, 1, 200, 1024, "0"},
                                        std::tuple{2, 2, -1, 512, 2000, "0"},
                                        std::tuple{3, 2, -1, 230, 899, "t:128,128"})));

TEST_P(SplitKV, DataTypeConfig)
{
    auto [hdims, i_perm, is_v_rowmajor, mode_use_cache_batch_idx, num_splits, dims_mask] =
        GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [mode, use_cache_batch_idx]                           = mode_use_cache_batch_idx;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                   // seqlen_knew
                                               {-1},                // seqlen_qpads
                                               {-1},                // seqlen_kpads
                                               {},                  // q_eff_lens_per_batch
                                               {},                  // kv_eff_lens_per_batch
                                               0,                   // rotary_dim
                                               i_perm,              // i_perm
                                               false,               // o_perm
                                               0,                   // scale_s
                                               0,                   // logits_soft_cap
                                               is_v_rowmajor,       // is_v_rowmajor
                                               def_lse,             // lse
                                               0,                   // page_block_size
                                               use_cache_batch_idx, // use_cache_batch_idx
                                               "n",                 // bias_str
                                               0.0f,                // p_drop
                                               0,                   // drop_seed
                                               0,                   // drop_offset
                                               false,               // drop_prefs
                                               mask_str,            // mask_str
                                               qscale_str,
                                               true,       // is_rotary_interleaved
                                               num_splits, // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_SPLITKV_API

#if CK_TILE_FMHA_FWD_APPENDKV_API

class AppendKV : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                 bool,
                                                 bool,
                                                 std::tuple<int, bool>,
                                                 int,
                                                 std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    AppendKV,
    Combine(AppendKVHDimValues,
            Bool(),            // layouts of k and v are controlled by i_perm
            IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
            ValuesIn({std::tuple{0, true}, std::tuple{0, false}, std::tuple{128, false}}),
            Values(1, 64, -1),
            Values(std::tuple{3, 3, -1, 60, 129, "t:32,32"},
                   std::tuple{3, 2, 2, 256, 256, "0"},
                   std::tuple{2, 3, 1, 264, 265, "1"},
                   std::tuple{4, 4, 2, 71, 64, "1"})));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AppendKV);

TEST_P(AppendKV, DataTypeConfig)
{
    auto [hdims,
          i_perm,
          is_v_rowmajor,
          page_block_size_use_cache_batch_idx,
          seqlen_knew,
          dims_mask]                            = GetParam();
    auto [hdim_q, hdim_v]                       = hdims;
    auto [page_block_size, use_cache_batch_idx] = page_block_size_use_cache_batch_idx;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    seqlen_knew = seqlen_knew == -1 ? seqlen_k : seqlen_knew;

    auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               seqlen_knew,         // seqlen_knew
                                               {-1},                // seqlen_qpads
                                               {-1},                // seqlen_kpads
                                               {},                  // q_eff_lens_per_batch
                                               {},                  // kv_eff_lens_per_batch
                                               0,                   // rotary_dim
                                               i_perm,              // i_perm
                                               true,                // o_perm
                                               0,                   // scale_s
                                               0,                   // logits_soft_cap
                                               is_v_rowmajor,       // is_v_rowmajor
                                               def_lse,             // lse
                                               page_block_size,     // page_block_size
                                               use_cache_batch_idx, // use_cache_batch_idx
                                               "n",                 // bias_str
                                               0.0f,                // p_drop
                                               0,                   // drop_seed
                                               0,                   // drop_offset
                                               false,               // drop_prefs
                                               mask_str,            // mask_str
                                               qscale_str,
                                               false, // is_rotary_interleaved
                                               1,     // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class AppendKVRoPE
    : public TestWithParam<std::tuple<bool,
                                      std::tuple<int, int>,
                                      bool,
                                      bool,
                                      std::tuple<int, bool>,
                                      int,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AppendKVRoPE);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         AppendKVRoPE,
                         Combine(EnableTestIf(!std::is_same_v<DataTypeConfig, FmhaFwdFp8Bf16>),
                                 AppendKVHDimValues,
                                 Bool(),            // layouts of k and v are controlled by i_perm
                                 IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
                                 Values(std::tuple{0, false},
                                        std::tuple{16, true},
                                        std::tuple{32, false},
                                        std::tuple{-1, true}),
                                 Values(16, 50, -1),
                                 Values(std::tuple{2, 3, -1, 60, 129, "t:32,32"},
                                        std::tuple{1, 2, 1, 128, 55, "0"},
                                        std::tuple{3, 4, 2, 72, 128, "1"})));

TEST_P(AppendKVRoPE, DataTypeConfig)
{
    auto [_, hdims, i_perm, is_v_rowmajor, rotary, seqlen_knew, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                                  = hdims;
    auto [rotary_dim, is_rotary_interleaved]                               = rotary;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]             = dims_mask;

    rotary_dim  = rotary_dim == -1 ? hdim_q : rotary_dim;
    seqlen_knew = seqlen_knew == -1 ? seqlen_k : seqlen_knew;

    auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               seqlen_knew,   // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {-1},          // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               rotary_dim,    // rotary_dim
                                               i_perm,        // i_perm
                                               true,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               true,          // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               "n",           // bias_str
                                               0.0f,          // p_drop
                                               0,             // drop_seed
                                               0,             // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               is_rotary_interleaved, // is_rotary_interleaved
                                               1,                     // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_APPENDKV_API

// ---------------------------------------------------------------
// Parameterized padding tests (batch & group) using Combine+Values
// ---------------------------------------------------------------

using PaddingParam = std::tuple<mode_enum,        // mode
                                int,              // batch
                                int,              // nhead
                                int,              // nhead_k
                                std::vector<int>, // seqlen_qs (logical)
                                std::vector<int>, // seqlen_ks (logical)
                                std::vector<int>, // seqlen_qpads (physical padded lengths)
                                std::vector<int>, // seqlen_kpads (physical padded lengths)
                                std::vector<int>, // q_eff_lens
                                std::vector<int>, // kv_eff_lens
                                bool,             // i_perm
                                bool,             // o_perm
                                std::string>;     // mask_str

// Ensure headers for containers / algorithms used in padding param builder.
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

class PaddingCases : public TestWithParam<PaddingParam>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PaddingCases);

// Build padding test params programmatically to enforce constraints
static std::vector<PaddingParam> BuildPaddingParams()
{
    std::vector<PaddingParam> params;

    // mask variants to cover
    const std::vector<std::string> mask_variants{"0", "t:50,64", "b:32,40"};
    const std::vector<std::string> mask_variants_reduced{"0", "t:50,64"}; // used for trimmed sets

    // Representative ratio pairs (q_ratio, k_ratio) to avoid explosion
    const std::vector<std::pair<double, double>> ratio_pairs_full{
        {1.0, 1.0}, // both full
        {1.0, 0.5}, // q full, k half
        {0.5, 1.0}, // q half, k full
    };
    const std::vector<std::pair<double, double>> ratio_pairs_reduced{{1.0, 1.0}, {0.5, 1.0}};

    // candidate physical seqlens for batch mode (single value) & for group mode (per batch)
    const std::vector<int> physical_lengths_full{64, 128, 256};
    const std::vector<int> physical_lengths_reduced{64};

    // batch sizes to sample
    const std::vector<int> batch_sizes{1, 4};
    // --------------------------------------------------------------------
    // Head configuration space (cover MHA, GQA, MQA)
    //  - Standard MHA: nhead_k == -1 (treated internally as nhead)
    //  - GQA: nhead_k > 0 and nhead % nhead_k == 0, nhead_k < nhead
    //  - MQA: nhead_k == 1
    // We choose (9, -1), (9, 3), (9, 1) so that divisibility holds. Full
    // combinatorics only applied to the first (standard) configuration to
    // avoid test explosion.
    // --------------------------------------------------------------------
    struct HeadCfg
    {
        int nhead;
        int nhead_k; // -1 for standard; else must divide nhead
        bool full;   // whether to use full coverage sets
    };
    const std::vector<HeadCfg> head_cfgs = {
        {9, -1, true}, // MHA full
        {9, 3, false}, // GQA reduced (nhead/nhead_k=3)
        {9, 1, false}  // MQA reduced
    };

    // Helper to clamp and ensure >=1
    auto logical_len = [](int physical, double ratio) {
        int v = static_cast<int>(std::round(physical * ratio));
        v     = std::max(1, std::min(v, physical));
        return v;
    };
    // Iterate over head configurations
    for(const auto& hc : head_cfgs)
    {
        const auto& ratio_pairs        = hc.full ? ratio_pairs_full : ratio_pairs_reduced;
        const auto& phys_lengths_batch = hc.full ? physical_lengths_full : physical_lengths_reduced;
        const auto& phys_lengths_group_q = phys_lengths_batch; // reuse
        const auto& phys_lengths_group_k = phys_lengths_batch; // reuse
        const auto& masks                = hc.full ? mask_variants : mask_variants_reduced;

        // -----------------
        // Batch mode params (effective lengths only)
        // -----------------
        for(int b : batch_sizes)
        {
            for(int phys_qkv : phys_lengths_batch)
            {
                for(const auto& rkpair : ratio_pairs)
                {
                    double rq = rkpair.first;
                    double rk = rkpair.second;
                    std::vector<int> q_eff(b), kv_eff(b);
                    int log_q = logical_len(phys_qkv, rq);
                    int log_k = logical_len(phys_qkv, rk);
                    for(int i = 0; i < b; ++i)
                    {
                        q_eff[i]  = log_q;
                        kv_eff[i] = log_k;
                    }
                    for(const auto& mask : masks)
                    {
                        params.emplace_back(PaddingParam{mode_enum::batch,
                                                         b,
                                                         hc.nhead,
                                                         hc.nhead_k,
                                                         {phys_qkv}, // seqlen_qs
                                                         {phys_qkv}, // seqlen_ks
                                                         {},         // seqlen_qpads
                                                         {},         // seqlen_kpads
                                                         q_eff,
                                                         kv_eff,
                                                         true,
                                                         true,
                                                         mask});
                    }
                }
                // Single-token logical length case (both q & k = 1)
                for(const auto& mask : masks)
                {
                    std::vector<int> q_eff(b, 1), kv_eff(b, 1);
                    params.emplace_back(PaddingParam{mode_enum::batch,
                                                     b,
                                                     hc.nhead,
                                                     hc.nhead_k,
                                                     {phys_qkv},
                                                     {phys_qkv},
                                                     {},
                                                     {},
                                                     q_eff,
                                                     kv_eff,
                                                     true,
                                                     true,
                                                     mask});
                }
            }
        }

        // -----------------
        // Group mode params (physical padding + logical variants)
        // -----------------
        for(int b : batch_sizes)
        {
            for(int phys_q : phys_lengths_group_q)
            {
                for(int phys_k : phys_lengths_group_k)
                {
                    for(const auto& rkpair : ratio_pairs)
                    {
                        double rq = rkpair.first;
                        double rk = rkpair.second;
                        std::vector<int> seqlen_qs(b), seqlen_ks(b), seqlen_qpads(b),
                            seqlen_kpads(b);
                        for(int i = 0; i < b; ++i)
                        {
                            seqlen_qpads[i] = phys_q;
                            seqlen_kpads[i] = phys_k;
                            seqlen_qs[i]    = logical_len(phys_q, rq);
                            seqlen_ks[i]    = logical_len(phys_k, rk);
                        }
                        std::array<std::pair<std::vector<int>, std::vector<int>>, 3> pad_variants{
                            std::pair{seqlen_qpads, seqlen_kpads}, // both
                            std::pair{seqlen_qpads, seqlen_ks},    // only q padding
                            std::pair{seqlen_qs, seqlen_kpads}     // only kv padding
                        };
                        for(const auto& mask : masks)
                        {
                            for(const auto& pv : pad_variants)
                            {
                                params.emplace_back(PaddingParam{mode_enum::group,
                                                                 b,
                                                                 hc.nhead,
                                                                 hc.nhead_k,
                                                                 seqlen_qs,
                                                                 seqlen_ks,
                                                                 pv.first,
                                                                 pv.second,
                                                                 {},
                                                                 {},
                                                                 true,
                                                                 true,
                                                                 mask});
                            }
                        }
                    }
                    // Single-token logical length case
                    for(const auto& mask : masks)
                    {
                        std::vector<int> seqlen_qs(b, 1), seqlen_ks(b, 1);
                        std::vector<int> seqlen_qpads(b, phys_q), seqlen_kpads(b, phys_k);
                        // both padding variant only (others degenerate)
                        params.emplace_back(PaddingParam{mode_enum::group,
                                                         b,
                                                         hc.nhead,
                                                         hc.nhead_k,
                                                         seqlen_qs,
                                                         seqlen_ks,
                                                         seqlen_qpads,
                                                         seqlen_kpads,
                                                         {},
                                                         {},
                                                         true,
                                                         true,
                                                         mask});
                    }
                }
            }
        }
    }

    return params;
}

static const std::vector<PaddingParam> kPaddingParams = BuildPaddingParams();

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd_Padding, PaddingCases, ValuesIn(kPaddingParams));

TEST_P(PaddingCases, DataTypeConfig)
{
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8Bf16>)
    {
        GTEST_SKIP() << "Skip for fp8";
    }

    auto [mode,
          batch,
          nhead,
          nhead_k,
          seqlen_qs,
          seqlen_ks,
          seqlen_qpads,
          seqlen_kpads,
          q_eff_lens,
          kv_eff_lens,
          i_perm,
          o_perm,
          mask_str] = GetParam();

    // For batch mode we wrap single logical lengths with adjust_seqlen.
    std::vector<int> adj_qs =
        (mode == mode_enum::batch) ? std::vector<int>{adjust_seqlen(seqlen_qs.at(0))} : seqlen_qs;
    std::vector<int> adj_ks =
        (mode == mode_enum::batch) ? std::vector<int>{adjust_seqlen(seqlen_ks.at(0))} : seqlen_ks;

    const int hdim_q      = 64;
    const int hdim_v      = 64;
    const int seqlen_knew = 0;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               adj_qs,
                                               adj_ks,
                                               hdim_q,
                                               hdim_v,
                                               seqlen_knew,  // seqlen_knew
                                               seqlen_qpads, // seqlen_qpads
                                               seqlen_kpads, // seqlen_kpads
                                               q_eff_lens,   // q_eff_lens_per_batch
                                               kv_eff_lens,  // kv_eff_lens_per_batch
                                               0,            // rotary_dim
                                               i_perm,       // i_perm
                                               o_perm,       // o_perm
                                               0,            // scale_s
                                               0,            // logits_soft_cap
                                               def_is_v_rowmajor,
                                               def_lse,  // lse
                                               0,        // page_block_size
                                               false,    // use_cache_batch_idx
                                               "n",      // bias_str
                                               0.0f,     // p_drop
                                               0,        // drop_seed
                                               0,        // drop_offset
                                               false,    // drop_prefs
                                               mask_str, // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}
