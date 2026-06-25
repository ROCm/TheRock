// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstdlib>
#include <time.h>

#include "test_topk_softmax_api.hpp"

// CPU reference
template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
auto reference_topk_softmax(const ck_tile::HostTensor<InputType>& x,
                            ck_tile::index_t k,
                            ck_tile::index_t dim = -1,
                            bool largest         = true,
                            bool sorted          = true)
{
    using namespace ck_tile;

    auto y = reference_softmax<InputType, WeightType, WeightType>(x, dim);

    auto [y_values, y_indices] = reference_topk(y, k, dim, largest, sorted);

    return ck_tile::make_tuple(y_values, y_indices);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
auto reference_topk_softmax(const ck_tile::HostTensor<InputType>& x,
                            ck_tile::HostTensor<WeightType>& y_values,
                            ck_tile::HostTensor<IndexType>& y_indices,
                            ck_tile::index_t k,
                            ck_tile::index_t dim = -1,
                            bool largest         = true,
                            bool sorted          = true)
{
    using namespace ck_tile;

    auto y = reference_softmax<InputType, WeightType, WeightType>(x, dim);
    reference_topk(y, y_values, y_indices, k, dim, largest, sorted);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
auto reference_topk_sigmoid(const ck_tile::HostTensor<InputType>& x,
                            ck_tile::HostTensor<WeightType>& y_values,
                            ck_tile::HostTensor<IndexType>& y_indices,
                            ck_tile::index_t k,
                            ck_tile::index_t dim = -1,
                            bool largest         = true,
                            bool sorted          = true)
{
    using namespace ck_tile;

    // topk only - no need to apply the sigmoid first
    auto x_fp32 = x.template CopyAsType<float>();
    reference_topk(x_fp32, y_values, y_indices, k, dim, largest, sorted);
    // apply sigmoid
    std::transform(y_values.begin(), y_values.end(), y_values.begin(), [](auto value) {
        return WeightType(1) / (WeightType(1) + exp(-value));
    });
}

// different threshold for different dtype
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("pr_i", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("pr_w", "fp32", "output weight data type(currently only fp32 supported now)")
        .insert("t", "32", "number of input tokens")
        .insert("e", "8", "number of experts")
        .insert("k", "2", "topk")
        .insert("st_i", "-1", "row stride of input, -1 means same as experts")
        .insert("st_o", "-1", "row stride of output/indices, -1 means same as topk")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "when set to 1 it will print kernel name")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel")
        .insert("activation", "softmax", "activation function to use: softmax or sigmoid");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
bool test_topk_softmax(ck_tile::ArgParser args)
{
    int validate            = args.get_int("v");
    std::string input_prec  = args.get_str("pr_i");
    std::string weight_prec = args.get_str("pr_w");
    int tokens              = args.get_int("t");
    int experts             = args.get_int("e");
    int topk                = args.get_int("k");
    int seed                = args.get_int("seed");
    int stride_input        = args.get_int("st_i");
    int stride_output       = args.get_int("st_o");
    int kname               = args.get_int("kname");
    int warmup              = args.get_int("warmup");
    int repeat              = args.get_int("repeat");
    std::string activation  = args.get_str("activation");

    if(stride_input < 0)
    {
        stride_input = experts;
    }
    if(stride_output < 0)
    {
        stride_output = topk;
    }
    assert(stride_input >= experts);
    assert(stride_output >= topk);

    if(seed < 0)
    {
        seed = std::time(nullptr);
    }

    if(topk > experts)
    {
        printf("topk:%d value should be smaller than, or equal to number of experts:%d\n",
               topk,
               experts);
        return false;
    }

    // tokens already considered batch size
    ck_tile::HostTensor<InputType> x_host({tokens, experts}, {stride_input, 1});
    ck_tile::HostTensor<WeightType> value_host({tokens, topk}, {stride_output, 1});
    ck_tile::HostTensor<IndexType> index_host({tokens, topk}, {stride_output, 1});

    {
        // random require per-row unique
        auto rand_gen = ck_tile::FillUniformDistribution_Unique<InputType>{
            -5.f, 5.f, static_cast<uint32_t>(seed)};

        for(int i_t = 0; i_t < tokens; i_t++)
        {
            ck_tile::HostTensor<InputType> x_row({experts});
            rand_gen(x_row);
            std::copy(x_row.begin(), x_row.end(), x_host.begin() + i_t * stride_input);
            rand_gen.clear();
        }
    }

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem value_dev(value_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem index_dev(index_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    topk_softmax_trait trait{input_prec, weight_prec, experts, activation};

    topk_softmax_kargs karg{x_dev.GetDeviceBuffer(),
                            value_dev.GetDeviceBuffer(),
                            index_dev.GetDeviceBuffer(),
                            tokens,
                            experts,
                            topk,
                            stride_input,
                            stride_output};

    ck_tile::stream_config sc{nullptr,
                              true,
                              /* log_level = */ (kname ? 1 : 0),
                              warmup,
                              repeat};
    auto ms = topk_softmax(trait, karg, sc);
    printf("[%s|%s]tokens:%d, experts:%d, topk:%d, st_i:%d, st_o:%d, activation:%s, ms:%f, ",
           input_prec.c_str(),
           weight_prec.c_str(),
           tokens,
           experts,
           topk,
           stride_input,
           stride_output,
           activation.c_str(),
           ms);
    if(ms < 0)
        printf("not supported\n");
    fflush(stdout);
    if(ms < 0)
    {
        return false;
    }

    value_dev.FromDevice(value_host.data());
    index_dev.FromDevice(index_host.data());

    bool rtn = true;
    if(validate)
    {
        ck_tile::HostTensor<WeightType> value_ref({tokens, topk}, {stride_output, 1});
        ck_tile::HostTensor<IndexType> index_ref({tokens, topk}, {stride_output, 1});

        if(activation == "softmax")
        {
            reference_topk_softmax<InputType, WeightType, IndexType>(
                x_host, value_ref, index_ref, topk);
        }
        else if(activation == "sigmoid")
        {
            reference_topk_sigmoid<InputType, WeightType, IndexType>(
                x_host, value_ref, index_ref, topk);
        }
        else
        {
            throw std::runtime_error("unsupported activation type: " + activation);
        }

        auto [rtol, atol] = get_elimit<InputType>("");
        for(int i_t = 0; i_t < tokens; i_t++)
        {
            auto s_begin = std::vector<size_t>{static_cast<size_t>(i_t), static_cast<size_t>(0)};
            auto s_end =
                std::vector<size_t>{static_cast<size_t>(i_t + 1), static_cast<size_t>(topk)};
            auto s_value_host = value_host.slice(s_begin, s_end);
            auto s_value_ref  = value_ref.slice(s_begin, s_end);
            rtn &= ck_tile::check_err(s_value_host,
                                      s_value_ref,
                                      std::string("[") + std::to_string(i_t) +
                                          std::string("] Value Error:"),
                                      rtol,
                                      atol);
            auto s_index_host = index_host.slice(s_begin, s_end);
            auto s_index_ref  = index_ref.slice(s_begin, s_end);
            rtn &= ck_tile::check_err(s_index_host,
                                      s_index_ref,
                                      std::string("[") + std::to_string(i_t) +
                                          std::string("] Index Error:"),
                                      rtol,
                                      atol);
        }
    }

    printf("valid:%s\n", rtn ? "y" : "n");
    fflush(stdout);
    return rtn;
}

template <typename T>
int run_gemm_combinations(std::string const& data_type)
{
    char bufs[7][64];
    char* argv[7] = {bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6]};
    std::vector<std::vector<std::string>> params = {
        {"-t=80", "-e=17"},
        {"-t=111", "-e=117"},
        {"-t=1000", "-e=55"},
        {"-t=99", "-e=180"},
        {"-t=175", "-e=64", "-k=8"},
        {"-t=65", "-e=8", "-k=2"},
        {"-t=1", "-e=25"},
        {"-t=31", "-e=19", "-k=15"},
        {"-t=81", "-e=37", "-k=7"},
        {"-t=199", "-e=128", "-k=13"},
        {"-t=23", "-e=1", "-k=1"},
        {"-t=127", "-e=99", "-k=19", "-st_i=233", "-st_o=31"},
        {"-t=71", "-e=11", "-k=11", "-st_i=30", "-st_o=12"},
        {"-t=1", "-e=1", "-k=1"},
        {"-t=99", "-e=2", "-k=1", "-st_i=11", "-st_o=5"},
        {"-t=333", "-e=99", "-k=13", "-st_i=191", "-st_o=17"},
        {"-t=20", "-e=5", "-k=2", "-activation=sigmoid"},
        {"-t=220", "-e=9", "-k=3", "-activation=sigmoid"},
        {"-t=500", "-e=21", "-k=13", "-activation=sigmoid"}};

    bool result      = true;
    std::string pr_i = "-pr_i=" + data_type;
    strncpy(bufs[0], "test_topk_softmax_bf16", 64);
    strncpy(bufs[1], pr_i.c_str(), 64);
    for(size_t i = 0; i < params.size(); i++)
    {
        for(size_t j = 0; j < params[i].size(); j++)
        {
            strncpy(bufs[j + 2], params[i][j].c_str(), 64);
        }
        int argc = params[i].size() + 2;

        auto [good_args, args] = create_args(argc, argv);
        if(!good_args)
        {
            result = false;
        }
        result = test_topk_softmax<T, float, ck_tile::index_t>(args) && result;
    }
    return result ? 0 : -1;
}
