// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigQuantDecode<T>;

void quant_rowcol_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    // NOTE: QuantGroupSize is a place holder. rowcol pipeline does not use QuantGroupSize
    using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 1>>;
    lut[hash_multiple_strings({"fp8", "rowcol"})] = [](const ck_tile::ArgParser& arg_parser) {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          TypeConfig,
                                          QuantGroupSize,
                                          ck_tile::QuantType::RowColQuant>(arg_parser);
    };
    lut[hash_multiple_strings({"bf8", "rowcol"})] = [](const ck_tile::ArgParser& arg_parser) {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          TypeConfig,
                                          QuantGroupSize,
                                          ck_tile::QuantType::RowColQuant>(arg_parser);
    };
}
