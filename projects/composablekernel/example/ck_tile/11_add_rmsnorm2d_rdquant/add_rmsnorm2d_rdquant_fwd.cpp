// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "add_rmsnorm2d_rdquant_fwd.hpp"
#include <cstring>
#include "ck_tile/utility/json_dump.hpp"

// different threshold for different dtype
template <typename InputDataType>
auto get_elimit()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-2;
    double atol = 1e-2;
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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("e", "1e-5", "epsilon")
        .insert("save_x", "1", "save rms(invrms) or not. set to 1 in training case")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec", "fp16", "precision")
        .insert("quant", "int8", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "add_rmsnorm2d_rdquant_fwd.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputDataType, typename QuantizedDataType, bool SaveX>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m      = arg_parser.get_int("m");
    ck_tile::index_t n      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = n;
    float epsilon                   = arg_parser.get_float("e");
    std::string input_data_type     = arg_parser.get_str("prec");
    std::string quantized_data_type = arg_parser.get_str("quant");
    int kname                       = arg_parser.get_int("kname");
    int do_validation               = arg_parser.get_int("v");
    int warmup                      = arg_parser.get_int("warmup");
    int repeat                      = arg_parser.get_int("repeat");

    assert(stride >= n);

    using TypeConfig = AddRmsnormRdquantTypeConfig<InputDataType, QuantizedDataType>;

    using ADataType        = typename TypeConfig::ADataType;
    using BDataType        = typename TypeConfig::BDataType;
    using GammaDataType    = typename TypeConfig::GammaDataType;
    using XDataType        = typename TypeConfig::XDataType;
    using YScaleDataType   = typename TypeConfig::YScaleDataType;
    using QYDataType       = typename TypeConfig::QYDataType;
    using ComputeDataType  = float;
    using UnquantYDataType = ck_tile::null_type;

    // host verify
    ck_tile::HostTensor<ADataType> a_host({m, n}, {stride, 1});
    ck_tile::HostTensor<BDataType> b_host({m, n}, {stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});

    ck_tile::HostTensor<XDataType> x_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_dev({m, n}, {stride, 1});

    ck_tile::HostTensor<YScaleDataType> yscale_host_ref({m}, {1});
    ck_tile::HostTensor<YScaleDataType> yscale_host_dev({m}, {1});

    ck_tile::HostTensor<QYDataType> qy_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_dev({m, n}, {stride, 1});

    ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf(x_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_buf(yscale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_buf(qy_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.data());
    b_buf.ToDevice(b_host.data());
    gamma_buf.ToDevice(gamma_host.data());

    std::cout << "[" << input_data_type << ", " << quantized_data_type << "]" << " m:" << m
              << ", n:" << n << ", stride:" << stride << std::flush;

    add_rmsnorm2d_rdquant_fwd_traits traits{input_data_type, quantized_data_type, SaveX};

    add_rmsnorm2d_rdquant_fwd_args args{a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        gamma_buf.GetDeviceBuffer(),
                                        x_buf.GetDeviceBuffer(),
                                        yscale_buf.GetDeviceBuffer(),
                                        qy_buf.GetDeviceBuffer(),
                                        epsilon,
                                        m,
                                        n,
                                        stride};

    float ave_time = add_rmsnorm2d_rdquant_fwd(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    std::size_t num_byte = sizeof(ADataType) * m * n + sizeof(BDataType) * m * n +
                           sizeof(GammaDataType) * n + sizeof(YScaleDataType) * m +
                           sizeof(QYDataType) * m * n;

    if constexpr(SaveX)
        num_byte += sizeof(XDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        using YDataType      = ComputeDataType;
        using InvRmsDataType = InputDataType;

        // Add
        {
            auto op = [](const auto& v0, const auto& v1) { return v0 + v1; };
            ck_tile::reference_binary_elementwise<ADataType, BDataType, XDataType, ComputeDataType>(
                a_host, b_host, x_host_ref, op);

            if constexpr(SaveX)
            {
                x_buf.FromDevice(x_host_dev.data());

                auto [rtol, atol] = get_elimit<XDataType>();
                if(stride == n)
                {
                    pass = ck_tile::check_err(x_host_dev,
                                              x_host_ref,
                                              std::string("x Error: Incorrect results!"),
                                              rtol,
                                              atol);
                }
                else
                {
                    for(int i_r = 0; i_r < m; i_r++)
                    {
                        std::vector<QYDataType> x_host_dev_row(x_host_dev.begin() + i_r * stride,
                                                               x_host_dev.begin() + i_r * stride +
                                                                   n);
                        std::vector<QYDataType> x_host_ref_row(x_host_ref.begin() + i_r * stride,
                                                               x_host_ref.begin() + i_r * stride +
                                                                   n);
                        pass &= ck_tile::check_err(x_host_dev_row,
                                                   x_host_ref_row,
                                                   std::string("x[") + std::to_string(i_r) +
                                                       std::string("] Error: Incorrect results!"),
                                                   rtol,
                                                   atol);
                    }
                }
            }
        }

        ck_tile::HostTensor<YDataType> y_host({m, n});
        // Rmsnorm2d
        {
            ck_tile::HostTensor<InvRmsDataType> invRms_host_ref({m});
            ck_tile::HostTensor<UnquantYDataType> unquant_y_host_ref({m, n});

            // CAUSION: kernel use ComputeDataType version of x, but we use XDataType here for
            // simplicity
            ck_tile::reference_rmsnorm2d_fwd<XDataType,
                                             GammaDataType,
                                             ComputeDataType,
                                             YDataType,
                                             InvRmsDataType,
                                             UnquantYDataType>(
                x_host_ref, gamma_host, y_host, invRms_host_ref, unquant_y_host_ref, epsilon);
        }

        // yscale
        {
            ck_tile::HostTensor<YDataType> y_rowwise_amax_host({m});

            using ReduceAmax = ck_tile::ReduceOp::AbsMax;
            ck_tile::reference_reduce<YDataType, ComputeDataType, YDataType>(
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

            if(stride == n)
            {
                pass = ck_tile::check_err(qy_host_dev,
                                          qy_host_ref,
                                          std::string("qy Error: Incorrect results!"),
                                          rtol,
                                          atol);
            }
            else
            {
                for(int i_r = 0; i_r < m; i_r++)
                {
                    std::vector<QYDataType> qy_host_dev_row(qy_host_dev.begin() + i_r * stride,
                                                            qy_host_dev.begin() + i_r * stride + n);
                    std::vector<QYDataType> qy_host_ref_row(qy_host_ref.begin() + i_r * stride,
                                                            qy_host_ref.begin() + i_r * stride + n);
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

    if(arg_parser.get_int("json") == 1)
    {
        dump_add_rmsnorm2d_rdquant_fwd_json(arg_parser.get_str("jsonfile"),
                                            input_data_type,
                                            quantized_data_type,
                                            m,
                                            n,
                                            stride,
                                            epsilon,
                                            ave_time,
                                            0,
                                            gb_per_sec,
                                            pass);
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string input_data_type     = arg_parser.get_str("prec");
    const std::string quantized_data_type = arg_parser.get_str("quant");
    int save_x                            = arg_parser.get_int("save_x");
    if(input_data_type == "fp16" && quantized_data_type == "int8" && save_x)
    {
        return run<ck_tile::half_t, ck_tile::int8_t, true>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "fp16" && quantized_data_type == "int8" && !save_x)
    {
        return run<ck_tile::half_t, ck_tile::int8_t, false>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "bf16" && quantized_data_type == "int8" && save_x)
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, true>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "bf16" && quantized_data_type == "int8" && !save_x)
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, true>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "fp16" && quantized_data_type == "fp8" && save_x)
    {
        return run<ck_tile::half_t, ck_tile::fp8_t, true>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "fp16" && quantized_data_type == "fp8" && !save_x)
    {
        return run<ck_tile::half_t, ck_tile::fp8_t, false>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "bf16" && quantized_data_type == "fp8" && save_x)
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t, true>(arg_parser) ? 0 : -2;
    }
    else if(input_data_type == "bf16" && quantized_data_type == "fp8" && !save_x)
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t, true>(arg_parser) ? 0 : -2;
    }

    return -3;
}
