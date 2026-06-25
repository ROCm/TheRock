// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"
#include "example/ck_tile/01_fmha/fmha_bwd_runner.hpp"

#include "gtest/gtest.h"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaBwdFp16 // or FmhaBwdBf16 / FmhaBwdFp32
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
        std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
};
template <>
struct TestConfigs<FmhaBwdFp32>
{
    static constexpr auto HDimValues =
        std::array{std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}};
};
static auto HDimValues     = ValuesIn(TestConfigs<DataTypeConfig>::HDimValues);
const auto ModeValues      = ValuesIn(std::vector<mode_enum>{mode_enum::batch, mode_enum::group});
constexpr auto init_method = "uf";

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run long tests (from smoke_test_fwd.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_LONG_TESTS)

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

// batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str
using FmhaBwdDimsMaskParam = std::tuple<int, int, int, int, int, std::string>;
using FmhaBwdTestParam     = std::tuple<      //
    mode_enum,                            // mode
    std::tuple<int, int>,                 // hdim_q, hdim_v
    std::tuple<bool, bool>,               // io_perm
    std::string,                          // bias_str
    bool,                                 // use_dbias
    float,                                // p_drop
    std::tuple<uint64_t, uint64_t, bool>, // drop_seed, drop_offset, drop_prefs
    FmhaBwdDimsMaskParam,
    bool // deterministic
    >;
void fmha_bwd_test(const FmhaBwdTestParam& param)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = param;
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {seqlen_q},
        {seqlen_k},
        {-1},
        {-1},
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0, // scale
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det, // deterministic
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for current parameters";
    ASSERT_EQ(result, bwd_result::success);
}

// Test cases from example/ck_tile/01_fmha/script/smoke_test_bwd.sh
class AllLong : public TestWithParam<FmhaBwdTestParam>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AllLong);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         AllLong,
                         Combine(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_LONG_TESTS))
                                     ? ModeValues
                                     : ValuesIn(std::vector<mode_enum>{}),
                                 HDimValues,
                                 Values(std::tuple{true, true}, std::tuple{false, false}), // perm
                                 Values("n", "a"),
                                 Values(false),                       // use_dbias
                                 Values(0.0f, 0.2f),                  // p_drop
                                 Values(std::tuple{123, 1024, true}), // seed/offset/prefs
                                 Values(std::tuple{1, 4, 2, 259, -1, "0"},
                                        std::tuple{2, 2, -1, 516, 253, "0"},
                                        std::tuple{1, 4, 1, 500, 251, "1"},
                                        std::tuple{1, 2, -1, 900, 258, "2"},
                                        std::tuple{2, 1, -1, 987, 219, "t:128,30"},
                                        std::tuple{2, 3, 1, 244, 499, "b:4,35"}),
                                 Values(false) // deterministic
                                 ));
TEST_P(AllLong, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class HDimPadding : public TestWithParam<FmhaBwdTestParam>
{
};
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         HDimPadding,
                         Combine(ModeValues,
                                 Values(std::tuple{24, 48},
                                        std::tuple{48, 48},
                                        std::tuple{72, 72},
                                        std::tuple{40, 88},
                                        std::tuple{96, 96},
                                        std::tuple{120, 160},
                                        std::tuple{256, 108},
                                        std::tuple{40, 64}),
                                 Values(std::tuple{true, true}, std::tuple{false, false}), // perm
                                 Values("n"),                     // bias_str
                                 Values(false),                   // use_dbias
                                 Values(0.0f),                    // p_drop
                                 Values(std::tuple{0, 0, false}), // seed/offset/prefs
                                 Values(std::tuple{1, 4, 2, 480, -1, "0"},
                                        std::tuple{2, 2, -1, 300, 400, "t:64,64"},
                                        std::tuple{1, 4, 1, 512, 201, "1"},
                                        std::tuple{1, 2, -1, 900, 256, "0"},
                                        std::tuple{2, 1, -1, 256, 256, "1"}),
                                 Values(false) // deterministic
                                 ));
TEST_P(HDimPadding, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class ElementwiseBias : public TestWithParam<FmhaBwdTestParam>
{
};
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         ElementwiseBias,
                         Combine(ModeValues,
                                 HDimValues,
                                 // layouts of bias and dbias are controlled by i_perm
                                 Values(std::tuple{true, false}, std::tuple{false, false}),
                                 Values("e:0", "e:1", "e:2"),
                                 Bool(),                          // use_dbias
                                 Values(0.0f),                    // p_drop
                                 Values(std::tuple{0, 0, false}), // seed/offset/prefs
                                 Values(std::tuple{1, 4, 2, 1024, 100, "0"},
                                        std::tuple{3, 2, -1, 128, 256, "2"},
                                        std::tuple{2, 2, -1, 130, 499, "t:50,64"}),
                                 Values(false) // deterministic
                                 ));
TEST_P(ElementwiseBias, DataTypeConfig) { fmha_bwd_test(GetParam()); }
class Alibi : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    Alibi,
    Combine(ModeValues,
            HDimValues,
            Values(std::tuple{true, true}), // perm
            Values("a:0", "a:1"),
            Values(false),                   // use_dbias
            Values(0.0f),                    // p_drop
            Values(std::tuple{0, 0, false}), // seed/offset/prefs
            ValuesIn([]() {
                const std::array dims{
                    std::tuple{1, 3, 3, 1024, 1000},
                    std::tuple{3, 5, 5, 128, 256},
                    std::tuple{2, 8, 4, 130, 320},
                };
                const std::array mask_strs{"0", "t", "b", "t:50,64", "b:32,40"};
                std::vector<FmhaBwdDimsMaskParam> dims_masks;
                std::for_each(dims.begin(), dims.end(), [&](const auto& d) {
                    const auto& [b, h, hk, sq, sk] = d;
                    std::for_each(mask_strs.begin(), mask_strs.end(), [&](const auto& m) {
                        dims_masks.push_back(std::tuple{b, h, hk, sq, sk, m});
                    });
                });
                return dims_masks;
            }()),
            Values(false) // deterministic
            ));
TEST_P(Alibi, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class Dropout : public TestWithParam<FmhaBwdTestParam>
{
};
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Dropout,
                         Combine(ModeValues,
                                 HDimValues,
                                 Values(std::tuple{true, true}),    // perm
                                 Values("n"),                       // bias_str
                                 Values(false),                     // use_dbias
                                 Values(0.123f, 0.5f),              // p_drop
                                 Values(std::tuple{10, 123, false}, // seed/offset/prefs
                                        std::tuple{34534564645, 7876878876864, true}),
                                 Values(std::tuple{2, 6, 2, 180, 512, "0"},
                                        std::tuple{3, 2, 2, 256, 128, "1"},
                                        std::tuple{4, 2, 1, 100, 768, "2"}),
                                 Values(false) // deterministic
                                 ));

TEST_P(Dropout, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class Deterministic : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Deterministic,
                         Combine(ModeValues,
                                 HDimValues,
                                 Values(std::tuple{false, true}, std::tuple{true, true}), // perm
                                 Values("n"),                     // bias_str
                                 Values(false),                   // use_dbias
                                 Values(0.0f),                    // p_drop
                                 Values(std::tuple{0, 0, false}), // seed/offset/prefs
                                 Values(std::tuple{2, 6, 2, 180, 512, "0"},
                                        std::tuple{3, 3, 1, 256, 128, "1"},
                                        std::tuple{4, 2, 2, 768, 100, "2"}),
                                 Values(true) // deterministic
                                 ));
TEST_P(Deterministic, DataTypeConfig) { fmha_bwd_test(GetParam()); }

// ============================================================================
// Q/KV Padding Tests - High Priority
// ============================================================================

// 1. BasicQPadding: Test Q padding only (K/V have no padding)
class BasicQPadding : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    BasicQPadding,
    Combine(Values(mode_enum::group), // Only group mode supports padding
            HDimValues,
            Values(std::tuple{true, true}),  // perm
            Values("n"),                     // no bias for basic test
            Values(false),                   // use_dbias
            Values(0.0f),                    // no dropout
            Values(std::tuple{0, 0, false}), // seed/offset/prefs
            ValuesIn([]() {
                // Define test cases with Q padding: seqlen_q < seqlen_qpad
                // Format: {batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str}
                // Note: Will set seqlen_qpad separately in the test
                std::vector<FmhaBwdDimsMaskParam> test_cases;

                // Small padding: logical length close to physical
                test_cases.push_back(std::tuple{2, 2, 2, 127, 128, "0"}); // Q: 127->128
                test_cases.push_back(std::tuple{3, 4, 2, 250, 256, "0"}); // Q: 250->256

                // Medium padding: ~20-30% padding
                test_cases.push_back(std::tuple{2, 2, 1, 180, 256, "0"}); // Q: 180->256
                test_cases.push_back(std::tuple{3, 3, 3, 350, 512, "1"}); // Q: 350->512, causal

                // Large padding: ~50% padding
                test_cases.push_back(std::tuple{2, 4, 2, 128, 256, "0"}); // Q: 128->256
                test_cases.push_back(std::tuple{2, 2, 2, 200, 512, "2"}); // Q: 200->512, causal

                return test_cases;
            }()),
            Values(false) // deterministic
            ));

TEST_P(BasicQPadding, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    // Set up Q padding: physical length larger than logical
    std::vector<ck_tile::index_t> seqlen_qs(batch, seqlen_q);
    std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k);

    // Calculate physical Q length (padded)
    ck_tile::index_t seqlen_qpad = ((seqlen_q + 63) / 64) * 64; // Round up to multiple of 64
    if(seqlen_q > 256)
        seqlen_qpad = ((seqlen_q + 127) / 128) * 128; // Larger alignment for longer sequences

    std::vector<ck_tile::index_t> seqlen_qpads(batch, seqlen_qpad);
    std::vector<ck_tile::index_t> seqlen_kpads(batch, seqlen_k); // No K padding

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0, // scale
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for Q padding with hdim_q=" << hdim_q;
    ASSERT_EQ(result, bwd_result::success);
}

// 2. BasicKVPadding: Test K/V padding only (Q has no padding)
class BasicKVPadding : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    BasicKVPadding,
    Combine(Values(mode_enum::group),
            HDimValues,
            Values(std::tuple{true, true}),
            Values("n"),
            Values(false),
            Values(0.0f),
            Values(std::tuple{0, 0, false}),
            ValuesIn([]() {
                std::vector<FmhaBwdDimsMaskParam> test_cases;

                // Small K/V padding
                test_cases.push_back(std::tuple{2, 2, 2, 128, 127, "0"}); // K: 127->128
                test_cases.push_back(std::tuple{3, 4, 2, 256, 250, "0"}); // K: 250->256

                // Medium K/V padding
                test_cases.push_back(std::tuple{2, 2, 1, 256, 180, "0"}); // K: 180->256
                test_cases.push_back(std::tuple{3, 3, 3, 512, 350, "1"}); // K: 350->512

                // Large K/V padding
                test_cases.push_back(std::tuple{2, 4, 2, 256, 128, "0"}); // K: 128->256
                test_cases.push_back(std::tuple{2, 2, 2, 512, 200, "2"}); // K: 200->512

                return test_cases;
            }()),
            Values(false)));

TEST_P(BasicKVPadding, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    std::vector<ck_tile::index_t> seqlen_qs(batch, seqlen_q);
    std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k);

    // No Q padding
    std::vector<ck_tile::index_t> seqlen_qpads(batch, seqlen_q);

    // Set up K/V padding
    ck_tile::index_t seqlen_kpad = ((seqlen_k + 63) / 64) * 64;
    if(seqlen_k > 256)
        seqlen_kpad = ((seqlen_k + 127) / 128) * 128;
    std::vector<ck_tile::index_t> seqlen_kpads(batch, seqlen_kpad);

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0,
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for K/V padding with hdim_q=" << hdim_q;
    ASSERT_EQ(result, bwd_result::success);
}

// 3. QKVPadding: Test both Q and K/V padding simultaneously
class QKVPadding : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    QKVPadding,
    Combine(Values(mode_enum::group),
            HDimValues,
            Values(std::tuple{true, true}),
            Values("n"),
            Values(false),
            Values(0.0f),
            Values(std::tuple{0, 0, false}),
            ValuesIn([]() {
                std::vector<FmhaBwdDimsMaskParam> test_cases;

                // Both Q and K have small padding
                test_cases.push_back(std::tuple{2, 2, 2, 120, 125, "0"}); // Q:120->128, K:125->128

                // Both Q and K have medium padding
                test_cases.push_back(std::tuple{2, 4, 2, 180, 200, "0"}); // Q:180->256, K:200->256
                test_cases.push_back(std::tuple{3, 3, 3, 300, 350, "1"}); // Q:300->320, K:350->384

                // Both Q and K have large padding
                test_cases.push_back(std::tuple{2, 2, 1, 150, 180, "0"}); // Q:150->256, K:180->256
                test_cases.push_back(std::tuple{2, 4, 2, 256, 300, "2"}); // Q:256->384, K:300->384

                // Asymmetric padding (Q more padded than K)
                test_cases.push_back(std::tuple{2, 2, 2, 100, 200, "0"}); // Q:100->128, K:200->256

                // Asymmetric padding (K more padded than Q)
                test_cases.push_back(std::tuple{2, 3, 1, 200, 100, "0"}); // Q:200->256, K:100->128

                return test_cases;
            }()),
            Values(false)));

TEST_P(QKVPadding, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    std::vector<ck_tile::index_t> seqlen_qs(batch, seqlen_q);
    std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k);

    // Set up both Q and K/V padding
    ck_tile::index_t seqlen_qpad = ((seqlen_q + 63) / 64) * 64;
    if(seqlen_q > 256)
        seqlen_qpad = ((seqlen_q + 127) / 128) * 128;

    ck_tile::index_t seqlen_kpad = ((seqlen_k + 63) / 64) * 64;
    if(seqlen_k > 256)
        seqlen_kpad = ((seqlen_k + 127) / 128) * 128;

    std::vector<ck_tile::index_t> seqlen_qpads(batch, seqlen_qpad);
    std::vector<ck_tile::index_t> seqlen_kpads(batch, seqlen_kpad);

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0,
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for Q+K/V padding with hdim_q=" << hdim_q;
    ASSERT_EQ(result, bwd_result::success);
}

// 4. ZeroLengthPadding: Test zero-length sequences with padding
class ZeroLengthPadding : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         ZeroLengthPadding,
                         Combine(Values(mode_enum::group),
                                 Values(std::tuple{64, -1},
                                        std::tuple{128, -1}), // Limited hdim for edge cases
                                 Values(std::tuple{true, true}),
                                 Values("n"),
                                 Values(false),
                                 Values(0.0f),
                                 Values(std::tuple{0, 0, false}),
                                 Values(
                                     // Test case 1: First batch has zero Q length
                                     std::tuple{3, 2, 2, 0, 128, "0"},
                                     // Test case 2: Middle batch has zero Q length (multi-batch)
                                     std::tuple{3, 2, 1, 100, 128, "0"},
                                     // Test case 3: Last batch has zero Q length
                                     std::tuple{3, 3, 3, 150, 200, "0"},
                                     // Test case 4: Zero K length (first batch)
                                     std::tuple{3, 2, 2, 128, 0, "0"},
                                     // Test case 5: Mixed zero lengths with padding
                                     std::tuple{4, 2, 2, 80, 100, "0"}),
                                 Values(false)));

TEST_P(ZeroLengthPadding, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    // Create varied sequence lengths with some zero-length sequences
    std::vector<ck_tile::index_t> seqlen_qs;
    std::vector<ck_tile::index_t> seqlen_ks;
    std::vector<ck_tile::index_t> seqlen_qpads;
    std::vector<ck_tile::index_t> seqlen_kpads;

    for(int b = 0; b < batch; ++b)
    {
        // Create pattern with zero-length sequences
        ck_tile::index_t q_len, k_len;

        if(seqlen_q == 0 && b == 1) // Middle batch zero Q
        {
            q_len = (b == 1) ? 0 : ((b == 0) ? 100 : 80);
            k_len = seqlen_k;
        }
        else if(seqlen_k == 0 && b == 0) // First batch zero K
        {
            q_len = seqlen_q;
            k_len = (b == 0) ? 0 : 100;
        }
        else
        {
            // Varied lengths
            q_len = (b == 0 && seqlen_q == 0) ? 0 : (seqlen_q + b * 10);
            k_len = seqlen_k + b * 15;
        }

        seqlen_qs.push_back(q_len);
        seqlen_ks.push_back(k_len);

        // Add padding for non-zero lengths
        ck_tile::index_t qpad = (q_len == 0) ? 0 : ((q_len + 63) / 64) * 64;
        ck_tile::index_t kpad = (k_len == 0) ? 0 : ((k_len + 63) / 64) * 64;

        seqlen_qpads.push_back(qpad);
        seqlen_kpads.push_back(kpad);
    }

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0,
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for zero-length padding";
    ASSERT_EQ(result, bwd_result::success);
}

// ============================================================================
// Q/KV Padding Tests - Medium Priority
// ============================================================================

// 5. VariedPaddingRatios: Test different padding ratios (waste ratios)
class VariedPaddingRatios : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    VariedPaddingRatios,
    Combine(Values(mode_enum::group),
            HDimValues,
            Values(std::tuple{true, true}),
            Values("n"),
            Values(false),
            Values(0.0f),
            Values(std::tuple{0, 0, false}),
            ValuesIn([]() {
                std::vector<FmhaBwdDimsMaskParam> test_cases;

                // Minimal waste: ~1-5% padding (logical ≈ physical - small delta)
                test_cases.push_back(
                    std::tuple{2, 2, 2, 127, 127, "0"}); // Q:127->128 (~0.8%), K:127->128
                test_cases.push_back(
                    std::tuple{2, 4, 2, 252, 250, "0"}); // Q:252->256 (~1.6%), K:250->256
                test_cases.push_back(std::tuple{2, 2, 1, 509, 505, "1"}); // Q:509->512, K:505->512

                // Low waste: ~10-20% padding
                test_cases.push_back(
                    std::tuple{2, 3, 3, 220, 210, "0"}); // Q:220->256 (~16%), K:210->256
                test_cases.push_back(
                    std::tuple{3, 2, 2, 440, 420, "0"}); // Q:440->512 (~16%), K:420->512
                test_cases.push_back(std::tuple{2, 4, 2, 350, 340, "1"}); // Q:350->384, K:340->384

                // Medium waste: ~30-40% padding
                test_cases.push_back(
                    std::tuple{2, 2, 2, 180, 170, "0"}); // Q:180->256 (~42%), K:170->256
                test_cases.push_back(
                    std::tuple{2, 3, 1, 320, 310, "0"}); // Q:320->384 (~20%), K:310->384
                test_cases.push_back(std::tuple{3, 2, 2, 350, 340, "2"}); // Q:350->512, K:340->512

                // High waste: ~50%+ padding
                test_cases.push_back(
                    std::tuple{2, 2, 2, 130, 130, "0"}); // Q:130->256 (~97%), K:130->256
                test_cases.push_back(
                    std::tuple{2, 4, 2, 260, 260, "0"}); // Q:260->512 (~97%), K:260->512
                test_cases.push_back(
                    std::tuple{2, 2, 1, 200, 200, "1"}); // Q:200->256 (~28%), K:200->256

                // Extreme waste: very small logical vs large physical
                test_cases.push_back(std::tuple{2, 2, 2, 65, 70, "0"});  // Q:65->128, K:70->128
                test_cases.push_back(std::tuple{2, 3, 3, 100, 90, "0"}); // Q:100->128, K:90->128

                return test_cases;
            }()),
            Values(false)));

TEST_P(VariedPaddingRatios, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    std::vector<ck_tile::index_t> seqlen_qs(batch, seqlen_q);
    std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k);

    // Calculate padding based on common alignment strategies
    auto calc_pad = [](ck_tile::index_t len) -> ck_tile::index_t {
        if(len <= 64)
            return 64;
        else if(len <= 128)
            return 128;
        else if(len <= 256)
            return 256;
        else if(len <= 384)
            return 384;
        else if(len <= 512)
            return 512;
        else
            return ((len + 127) / 128) * 128;
    };

    std::vector<ck_tile::index_t> seqlen_qpads(batch, calc_pad(seqlen_q));
    std::vector<ck_tile::index_t> seqlen_kpads(batch, calc_pad(seqlen_k));

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0,
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for varied padding ratios";
    ASSERT_EQ(result, bwd_result::success);
}

// 6. PaddingWithMask: Test padding combined with various mask types
class PaddingWithMask : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    PaddingWithMask,
    Combine(Values(mode_enum::group),
            Values(std::tuple{64, -1}, std::tuple{128, -1}), // Focus on common sizes
            Values(std::tuple{true, true}),
            Values("n"),
            Values(false),
            Values(0.0f),
            Values(std::tuple{0, 0, false}),
            ValuesIn([]() {
                std::vector<FmhaBwdDimsMaskParam> test_cases;

                // No mask with padding (baseline)
                test_cases.push_back(std::tuple{2, 2, 2, 200, 180, "0"});

                // Causal mask (top-left) with Q padding
                test_cases.push_back(std::tuple{2, 2, 2, 200, 256, "1"}); // Q padded, K exact
                test_cases.push_back(std::tuple{2, 4, 2, 180, 200, "t"}); // Both padded, causal

                // Causal mask (bottom-right) with K/V padding
                test_cases.push_back(std::tuple{2, 2, 1, 256, 180, "2"}); // K padded, Q exact
                test_cases.push_back(
                    std::tuple{2, 3, 3, 200, 180, "b"}); // Both padded, bottom-right

                // Sliding window attention with padding
                test_cases.push_back(std::tuple{2, 2, 2, 200, 190, "t:64,32"});  // SWA + padding
                test_cases.push_back(std::tuple{2, 4, 2, 180, 170, "b:32,64"});  // SWA + padding
                test_cases.push_back(std::tuple{3, 2, 1, 220, 210, "t:100,50"}); // Larger window

                // Sliding window with asymmetric padding
                test_cases.push_back(std::tuple{2, 2, 2, 150, 250, "t:80,40"}); // Q more padded
                test_cases.push_back(std::tuple{2, 3, 3, 250, 150, "b:50,70"}); // K more padded

                // Mixed scenarios
                test_cases.push_back(std::tuple{2, 4, 2, 190, 185, "t:50,50"}); // Symmetric window
                test_cases.push_back(std::tuple{3, 2, 2, 300, 280, "1"}); // Multi-batch causal

                return test_cases;
            }()),
            Values(false)));

TEST_P(PaddingWithMask, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    std::vector<ck_tile::index_t> seqlen_qs(batch, seqlen_q);
    std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k);

    // Apply padding
    ck_tile::index_t seqlen_qpad = ((seqlen_q + 63) / 64) * 64;
    ck_tile::index_t seqlen_kpad = ((seqlen_k + 63) / 64) * 64;

    if(seqlen_q > 256)
        seqlen_qpad = ((seqlen_q + 127) / 128) * 128;
    if(seqlen_k > 256)
        seqlen_kpad = ((seqlen_k + 127) / 128) * 128;

    std::vector<ck_tile::index_t> seqlen_qpads(batch, seqlen_qpad);
    std::vector<ck_tile::index_t> seqlen_kpads(batch, seqlen_kpad);

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0,
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for padding with mask";
    ASSERT_EQ(result, bwd_result::success);
}

// 7. MultiBatchPadding: Test multiple batches with different padding configurations
class MultiBatchPadding : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         MultiBatchPadding,
                         Combine(Values(mode_enum::group),
                                 Values(std::tuple{64, -1}, std::tuple{128, -1}),
                                 Values(std::tuple{true, true}),
                                 Values("n"),
                                 Values(false),
                                 Values(0.0f),
                                 Values(std::tuple{0, 0, false}),
                                 Values(
                                     // 3 batches with varied Q/K lengths and padding
                                     std::tuple{3, 2, 2, 150, 200, "0"},
                                     // 4 batches with different patterns
                                     std::tuple{4, 3, 3, 180, 220, "0"},
                                     // 5 batches with mixed scenarios
                                     std::tuple{5, 2, 1, 120, 160, "1"},
                                     // 3 batches with causal mask
                                     std::tuple{3, 4, 2, 200, 180, "t"},
                                     // 4 batches with sliding window
                                     std::tuple{4, 2, 2, 160, 140, "t:50,30"}),
                                 Values(false)));

TEST_P(MultiBatchPadding, DataTypeConfig)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = GetParam();
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, base_seqlen_q, base_seqlen_k, mask_str]             = dims_mask;

    // Create varied sequence lengths for each batch
    std::vector<ck_tile::index_t> seqlen_qs;
    std::vector<ck_tile::index_t> seqlen_ks;
    std::vector<ck_tile::index_t> seqlen_qpads;
    std::vector<ck_tile::index_t> seqlen_kpads;

    for(int b = 0; b < batch; ++b)
    {
        // Generate varied lengths across batches
        // Pattern: decreasing, increasing, or random variation
        ck_tile::index_t q_len, k_len;

        switch(b % 3)
        {
        case 0: // Decreasing
            q_len = base_seqlen_q - b * 20;
            k_len = base_seqlen_k - b * 25;
            break;
        case 1: // Increasing
            q_len = base_seqlen_q + b * 15;
            k_len = base_seqlen_k + b * 20;
            break;
        case 2: // Mixed
            q_len = base_seqlen_q + (b % 2 == 0 ? 10 : -10) * b;
            k_len = base_seqlen_k + (b % 2 == 0 ? -15 : 15) * b;
            break;
        }

        // Ensure positive lengths
        q_len = std::max<ck_tile::index_t>(64, q_len);
        k_len = std::max<ck_tile::index_t>(64, k_len);

        seqlen_qs.push_back(q_len);
        seqlen_ks.push_back(k_len);

        // Calculate different padding strategies per batch
        ck_tile::index_t qpad, kpad;

        if(b % 4 == 0)
        {
            // Tight padding (minimal waste)
            qpad = ((q_len + 31) / 32) * 32;
            kpad = ((k_len + 31) / 32) * 32;
        }
        else if(b % 4 == 1)
        {
            // Medium padding
            qpad = ((q_len + 63) / 64) * 64;
            kpad = ((k_len + 63) / 64) * 64;
        }
        else if(b % 4 == 2)
        {
            // Loose padding
            qpad = ((q_len + 127) / 128) * 128;
            kpad = ((k_len + 127) / 128) * 128;
        }
        else
        {
            // Mixed: Q tight, K loose
            qpad = ((q_len + 31) / 32) * 32;
            kpad = ((k_len + 127) / 128) * 128;
        }

        seqlen_qpads.push_back(qpad);
        seqlen_kpads.push_back(kpad);
    }

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        seqlen_qs,
        seqlen_ks,
        seqlen_qpads,
        seqlen_kpads,
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0,
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for multi-batch padding";
    ASSERT_EQ(result, bwd_result::success);
}
