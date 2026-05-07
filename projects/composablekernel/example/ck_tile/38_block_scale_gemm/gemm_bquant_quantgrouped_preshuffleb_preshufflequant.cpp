// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

#if CK_TILE_USE_WMMA
template <typename T>
using GemmConfig = GemmConfigPreshuffleB_PreshuffleBQuant_Prefill_Wmma<T>;
#else
template <typename T>
using GemmConfig = GemmConfigPreshuffleB_PreshuffleBQuant_Prefill<T>;
#endif

void bquant_quantgrouped_preshuffleb_preshufflequant_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
    lut[hash_multiple_strings({"fp8", "bquant", "preshuffleb", "preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                            ck_tile::fp8_t,
                                                            ck_tile::half_t,
                                                            float>{});
            return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                              TypeConfig,
                                              QuantGroupSize,
                                              ck_tile::QuantType::BQuantGrouped>(arg_parser);
        };
    lut[hash_multiple_strings({"bf8", "bquant", "preshuffleb", "preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t,
                                                            ck_tile::bf8_t,
                                                            ck_tile::half_t,
                                                            float>{});
            return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                              TypeConfig,
                                              QuantGroupSize,
                                              ck_tile::QuantType::BQuantGrouped>(arg_parser);
        };
    lut[hash_multiple_strings({"fp8i4", "bquant", "preshuffleb", "preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                            ck_tile::pk_int4_t,
                                                            ck_tile::half_t,
                                                            ck_tile::fp8_t>{});
            return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                              TypeConfig,
                                              QuantGroupSize,
                                              ck_tile::QuantType::BQuantGrouped>(arg_parser);
        };
    lut[hash_multiple_strings({"bf8i4", "bquant", "preshuffleb", "preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t,
                                                            ck_tile::pk_int4_t,
                                                            ck_tile::half_t,
                                                            ck_tile::bf8_t>{});
            return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                              TypeConfig,
                                              QuantGroupSize,
                                              ck_tile::QuantType::BQuantGrouped>(arg_parser);
        };
}
