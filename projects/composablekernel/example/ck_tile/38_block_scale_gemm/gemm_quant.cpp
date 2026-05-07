// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/core/config.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "gemm_utils.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("h", "false", "Print help message")
        .insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row or Column")
        .insert("b_layout", "C", "B tensor data layout - Row or Column")
        .insert("bq_layout", "C", "Bq tensor data layout - Row or Column")
        .insert("c_layout", "R", "C tensor data layout - Row or Column")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_q", "0", "Tensor AQ stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0: No validation, 1: Validation on CPU, 2: Validation on GPU")
        .insert("prec",
                "fp8",
                "Data type. For AQuant: fp8, bf8, i4fp8, or i4bf8;  for Bquant: fp8, bf8, fp8i4, "
                "or bf8i4;  for ABQuant: fp8, bf8")
        .insert("warmup", "50", "Number of iterations before benchmarking the kernel")
        .insert("repeat", "1000", "Number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "SplitK value")
        .insert("device", "0", "Device id that will be used to run the kernel")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("flush_cache", "true", "Flush cache before running the kernel")
        .insert("rotating_count", "1000", "Rotating count")
        .insert("quant_mode", "bquant", "Choose aquant, bquant, abquant, tensor or rowcol")
        .insert("preshuffleb", "false", "Enable preshuffle of tensor B")
        .insert("preshufflequant", "false", "Enable preshuffle of quant tensor")
        .insert("group_size",
                "1x1x128",
                "Quantization group size as MxNxK, e.g., 1x1x128, 1x32x128, 1x64x128");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

auto gen_lut_key(const ck_tile::ArgParser& arg_parser)
{
    std::string data_type  = arg_parser.get_str("prec");
    std::string quant_mode = arg_parser.get_str("quant_mode");

    std::vector<std::string> params = {data_type, quant_mode};

    if(quant_mode == "aquant")
    {
        std::string preshufflequant =
            arg_parser.get_bool("preshufflequant") ? "preshufflequant" : "non-preshufflequant";
        params.push_back(preshufflequant);
    }
    if(quant_mode == "bquant")
    {
        std::string preshuffleb =
            arg_parser.get_bool("preshuffleb") ? "preshuffleb" : "non-preshuffleb";
        params.push_back(preshuffleb);

        std::string preshufflequant =
            arg_parser.get_bool("preshufflequant") ? "preshufflequant" : "non-preshufflequant";
        params.push_back(preshufflequant);
    }
    if(quant_mode == "abquant")
    {
        std::string preshuffleb =
            arg_parser.get_bool("preshuffleb") ? "preshuffleb" : "non-preshuffleb";
        params.push_back(preshuffleb);

        std::string preshufflequant =
            arg_parser.get_bool("preshufflequant") ? "preshufflequant" : "non-preshufflequant";
        params.push_back(preshufflequant);
    }
    if(quant_mode != "rowcol" && quant_mode != "tensor")
    {
        // NOTE: rowcol and tensor pipeline do not use group size
        std::string group_size_str = arg_parser.get_str("group_size");
        params.push_back(group_size_str);
    }

    return hash_multiple_strings(params);
}

void abquant_quantgrouped_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void aquant_quantgrouped_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void aquant_quantgrouped_preshufflequant_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_fp8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_bf8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_fp8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_bf8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_bf16fp4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshuffleb_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshufflequant_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshuffleb_preshufflequant_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void quant_rowcol_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void quant_tensor_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result || arg_parser.get_bool("h"))
    {
        arg_parser.print();
        return -1;
    }

    auto device_id = arg_parser.get_int("device");
    std::cout << "Device ID: " << device_id << std::endl;
    ck_tile::hip_check_error(hipSetDevice(device_id));

    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>> lut;
    abquant_quantgrouped_instance_factory(lut);
    aquant_quantgrouped_instance_factory(lut);
    aquant_quantgrouped_preshufflequant_instance_factory(lut);
    bquant_quantgrouped_fp8_instance_factory(lut);
    bquant_quantgrouped_bf8_instance_factory(lut);
    bquant_quantgrouped_fp8i4_instance_factory(lut);
    bquant_quantgrouped_bf8i4_instance_factory(lut);
    bquant_quantgrouped_bf16fp4_instance_factory(lut);
    bquant_quantgrouped_preshuffleb_instance_factory(lut);
    bquant_quantgrouped_preshufflequant_instance_factory(lut);
    bquant_quantgrouped_preshuffleb_preshufflequant_instance_factory(lut);
    quant_rowcol_instance_factory(lut);
    quant_tensor_instance_factory(lut);

    auto key = gen_lut_key(arg_parser);

    if(lut.find(key) != lut.end())
    {
        return lut[key](arg_parser);
    }
    else
    {
        std::cerr << "Error: Combination of prec, quant_mode, preshuffleb, preshufflequant, and "
                     "group_size not supported."
                  << std::endl;
        return -1;
    }
}
