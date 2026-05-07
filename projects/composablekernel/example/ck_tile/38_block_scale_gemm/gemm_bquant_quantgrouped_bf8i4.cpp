// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigQuantPrefill<T>;

#define RUN_GEMM_EXAMPLE_PREC_TYPE                         \
    run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>, \
                               TypeConfig,                 \
                               QuantGroupSize,             \
                               ck_tile::QuantType::BQuantGrouped>(arg_parser);

void bquant_quantgrouped_bf8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t,
                                                    ck_tile::pk_int4_t,
                                                    ck_tile::half_t,
                                                    ck_tile::bf8_t>{});
#ifndef CK_GFX950_SUPPORT
    lut[hash_multiple_strings(
        {"bf8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x64"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
#endif
    lut[hash_multiple_strings(
        {"bf8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"bf8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x8x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"bf8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x32x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"bf8i4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x64x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
}
