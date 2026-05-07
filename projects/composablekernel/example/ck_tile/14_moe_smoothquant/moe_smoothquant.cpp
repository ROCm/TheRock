// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "moe_smoothquant.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include <cstring>
#include <set>

// different threshold for different dtype
template <typename DataType>
auto get_elimit()
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::int8_t>()
{
    // due to rounding, int8 quantization might have 1 abs error
    double rtol = 1;
    double atol = 1;
    return ck_tile::make_tuple(rtol, atol);
}

template <typename IndexType>
void topid_unique_gen(
    std::vector<IndexType>& host_tensor, int tokens, int topk, int num_expert, int seed)
{
    size_t total_size = topk * tokens;
    std::srand(seed);
    std::set<IndexType> unique_set;
    IndexType current_v;
    for(size_t i = 0; i < total_size; i++)
    {
        if(i % topk == 0)
        {
            unique_set.clear();
        }
        current_v = std::rand() % num_expert;
        while(unique_set.find(current_v) != unique_set.end())
        {
            current_v = std::rand() % num_expert;
        }
        unique_set.insert(current_v);
        host_tensor[i] = current_v;
    }
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("t", "3328", "tokens dimension")
        .insert("h", "4096", "hidden_size dimension")
        .insert("e", "32", "experts")
        .insert("k", "5", "topk")
        .insert("stride", "-1", "stride per row, if -1 then equal to hidden_size")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec_i", "fp16", "input precision, fp16/bf16")
        .insert("prec_o", "int8", "precision, int8/fp8")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "moe_smoothquant.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename OutputType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t tokens      = arg_parser.get_int("t");
    ck_tile::index_t hidden_size = arg_parser.get_int("h");
    ck_tile::index_t stride      = arg_parser.get_int("stride");
    if(stride < 0)
        stride = hidden_size;
    ck_tile::index_t experts = arg_parser.get_int("e");
    ck_tile::index_t topk    = arg_parser.get_int("k");
    std::string prec_i       = arg_parser.get_str("prec_i");
    std::string prec_o       = arg_parser.get_str("prec_o");
    int kname                = arg_parser.get_int("kname");
    int do_validation        = arg_parser.get_int("v");
    int warmup               = arg_parser.get_int("warmup");
    int repeat               = arg_parser.get_int("repeat");

    assert(stride >= hidden_size);

    using TypeConfig = MoeSmoothquantTypeConfig<InputType, OutputType>;

    using XDataType           = typename TypeConfig::XDataType;
    using SmoothScaleDataType = typename TypeConfig::SmoothScaleDataType;
    using YScaleDataType      = typename TypeConfig::YScaleDataType;
    using QYDataType          = typename TypeConfig::QYDataType;
    using ComputeDataType     = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({tokens, hidden_size}, {stride, 1});
    ck_tile::HostTensor<SmoothScaleDataType> smscale_host({experts * hidden_size});
    ck_tile::HostTensor<ck_tile::index_t> topk_ids_host({tokens, topk});

    ck_tile::HostTensor<YScaleDataType> yscale_host_ref({topk * tokens}, {1});
    ck_tile::HostTensor<YScaleDataType> yscale_host_dev({topk * tokens}, {1});

    ck_tile::HostTensor<QYDataType> qy_host_ref({topk * tokens, hidden_size}, {stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_dev({topk * tokens, hidden_size}, {stride, 1});

    topid_unique_gen<ck_tile::index_t>(topk_ids_host.mData, tokens, topk, experts, 11937);
    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<SmoothScaleDataType>{1e-3, .5f}(smscale_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem smscale_buf(smscale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem topk_ids_buf(topk_ids_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_buf(yscale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_buf(qy_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    smscale_buf.ToDevice(smscale_host.data());
    topk_ids_buf.ToDevice(topk_ids_host.data());

    std::cout << "[" << prec_i << "-" << prec_o << "]" << " tokens:" << tokens
              << ", hidden_size:" << hidden_size << ", stride:" << stride << ", experts:" << experts
              << ", topk:" << topk << std::flush;

    moe_smoothquant_traits traits{prec_i, prec_o};

    moe_smoothquant_args args{x_buf.GetDeviceBuffer(),
                              smscale_buf.GetDeviceBuffer(),
                              topk_ids_buf.GetDeviceBuffer(),
                              yscale_buf.GetDeviceBuffer(),
                              qy_buf.GetDeviceBuffer(),
                              tokens,
                              hidden_size,
                              experts,
                              topk,
                              stride,
                              stride};

    float ave_time = moe_smoothquant(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    std::size_t num_byte = sizeof(XDataType) * tokens * hidden_size +
                           sizeof(SmoothScaleDataType) * topk * hidden_size +
                           sizeof(YScaleDataType) * topk * tokens +
                           sizeof(QYDataType) * topk * tokens * hidden_size;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        using YDataType = ComputeDataType;
        ck_tile::HostTensor<ComputeDataType> y_host({topk * tokens, hidden_size}, {stride, 1});
        // smooth outlier
        {
            auto f = [&](auto i_token) {
                for(int i_topk = 0; i_topk < topk; i_topk++)
                {
                    auto i_expert = topk_ids_host(i_token, i_topk);

                    for(int i_h = 0; i_h < hidden_size; ++i_h)
                    {
                        auto v_smscale = ck_tile::type_convert<ComputeDataType>(
                            smscale_host(i_expert * hidden_size + i_h));
                        auto v_x = ck_tile::type_convert<ComputeDataType>(x_host(i_token, i_h));
                        // y_host(i_token * topk + i_topk, i_h) = v_x * v_smscale;
                        y_host(i_topk * tokens + i_token, i_h) = v_x * v_smscale;
                    }
                }
            };

            ck_tile::make_ParallelTensorFunctor(f, tokens)(std::thread::hardware_concurrency());
        }

        // yscale
        {
            ck_tile::HostTensor<YDataType> y_rowwise_amax_host({topk * tokens});

            using ReduceAmax = ck_tile::ReduceOp::AbsMax;
            ck_tile::reference_reduce<ComputeDataType, ComputeDataType, YDataType>(
                y_host, y_rowwise_amax_host, ReduceAmax{});

            auto op = [](const auto& v0) {
                return v0 /
                       ck_tile::type_convert<ComputeDataType>(ck_tile::numeric<QYDataType>::max());
            };
            ck_tile::reference_unary_elementwise<YDataType, YScaleDataType, ComputeDataType>(
                y_rowwise_amax_host, yscale_host_ref, op);

            yscale_buf.FromDevice(yscale_host_dev.mData.data());

            auto [rtol, atol] = get_elimit<YScaleDataType>();
            pass &= ck_tile::check_err(yscale_host_dev,
                                       yscale_host_ref,
                                       std::string("yscale Error: Incorrect results!"),
                                       rtol,
                                       atol);
        }

        // rowwise quantization
        {
            ck_tile::reference_rowwise_quantization2d<YDataType, YScaleDataType, QYDataType>(
                y_host, yscale_host_ref, qy_host_ref);

            qy_buf.FromDevice(qy_host_dev.data());
            auto [rtol, atol] = get_elimit<QYDataType>();

            if(stride == hidden_size)
            {
                pass = ck_tile::check_err(qy_host_dev,
                                          qy_host_ref,
                                          std::string("qy Error: Incorrect results!"),
                                          rtol,
                                          atol);
            }
            else
            {
                for(int i_r = 0; i_r < topk * tokens; i_r++)
                {
                    std::vector<QYDataType> qy_host_dev_row(qy_host_dev.begin() + i_r * stride,
                                                            qy_host_dev.begin() + i_r * stride +
                                                                hidden_size);
                    std::vector<QYDataType> qy_host_ref_row(qy_host_ref.begin() + i_r * stride,
                                                            qy_host_ref.begin() + i_r * stride +
                                                                hidden_size);
                    pass &= ck_tile::check_err(qy_host_dev_row,
                                               qy_host_ref_row,
                                               std::string("qy[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }
            }
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    if(arg_parser.get_int("json"))
    {
        dump_moe_smoothquant_json(arg_parser.get_str("jsonfile"),
                                  prec_i,
                                  prec_o,
                                  tokens,
                                  hidden_size,
                                  stride,
                                  experts,
                                  topk,
                                  pass,
                                  ave_time,
                                  0,
                                  gb_per_sec);
    }
    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string prec_i = arg_parser.get_str("prec_i");
    const std::string prec_o = arg_parser.get_str("prec_o");
    if(prec_i == "fp16" && prec_o == "int8")
    {
        return run<ck_tile::half_t, ck_tile::int8_t>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "fp16" && prec_o == "fp8")
    {
        return run<ck_tile::half_t, ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "int8")
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "fp8")
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
