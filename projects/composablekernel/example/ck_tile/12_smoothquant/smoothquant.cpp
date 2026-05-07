// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "smoothquant.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include <cstring>

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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("x_stride", "-1", "input stride per row, if -1 then equal to n")
        .insert("y_stride", "-1", "output stride per row, if -1 then equal to n")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "smoothquant.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m        = arg_parser.get_int("m");
    ck_tile::index_t n        = arg_parser.get_int("n");
    ck_tile::index_t x_stride = arg_parser.get_int("x_stride");
    if(x_stride < 0)
        x_stride = n;
    ck_tile::index_t y_stride = arg_parser.get_int("y_stride");
    if(y_stride < 0)
        y_stride = n;
    std::string data_type = arg_parser.get_str("prec");
    int kname             = arg_parser.get_int("kname");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(x_stride >= n);

    using TypeConfig = SmoothquantTypeConfig<DataType>;

    using XDataType           = typename TypeConfig::XDataType;
    using SmoothScaleDataType = typename TypeConfig::SmoothScaleDataType;
    using YScaleDataType      = typename TypeConfig::YScaleDataType;
    using QYDataType          = typename TypeConfig::QYDataType;
    using ComputeDataType     = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {x_stride, 1});
    ck_tile::HostTensor<SmoothScaleDataType> smscale_host({n});

    ck_tile::HostTensor<YScaleDataType> yscale_host_ref({m}, {1});
    ck_tile::HostTensor<YScaleDataType> yscale_host_dev({m}, {1});

    ck_tile::HostTensor<QYDataType> qy_host_ref({m, n}, {y_stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_dev({m, n}, {y_stride, 1});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<SmoothScaleDataType>{1e-3, .5f}(smscale_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem smscale_buf(smscale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_buf(yscale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_buf(qy_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    smscale_buf.ToDevice(smscale_host.data());

    std::cout << "[" << data_type << "]" << " m:" << m << ", n:" << n << ", x_stride:" << x_stride
              << ", y_stride:" << y_stride << std::flush;

    smoothquant_traits traits{data_type};

    smoothquant_args args{x_buf.GetDeviceBuffer(),
                          smscale_buf.GetDeviceBuffer(),
                          yscale_buf.GetDeviceBuffer(),
                          qy_buf.GetDeviceBuffer(),
                          m,
                          n,
                          x_stride,
                          y_stride};

    float ave_time = smoothquant(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(SmoothScaleDataType) * n +
                           sizeof(YScaleDataType) * m + sizeof(QYDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        using YDataType = ComputeDataType;
        ck_tile::HostTensor<ComputeDataType> y_host({m, n}, {y_stride, 1});
        // smooth outlier
        {
            auto f = [&](auto n_) {
                auto v_smscale = ck_tile::type_convert<ComputeDataType>(smscale_host(n_));

                for(int m_ = 0; m_ < m; ++m_)
                {
                    auto v_x       = ck_tile::type_convert<ComputeDataType>(x_host(m_, n_));
                    y_host(m_, n_) = v_x * v_smscale;
                }
            };

            ck_tile::make_ParallelTensorFunctor(f, smscale_host.get_element_space_size())(
                std::thread::hardware_concurrency());
        }

        // yscale
        {
            ck_tile::HostTensor<YDataType> y_rowwise_amax_host({m});

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

            if(y_stride == n)
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
                    std::vector<QYDataType> qy_host_dev_row(qy_host_dev.begin() + i_r * y_stride,
                                                            qy_host_dev.begin() + i_r * y_stride +
                                                                n);
                    std::vector<QYDataType> qy_host_ref_row(qy_host_ref.begin() + i_r * y_stride,
                                                            qy_host_ref.begin() + i_r * y_stride +
                                                                n);
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
        dump_smoothquant_json(arg_parser.get_str("jsonfile"),
                              data_type,
                              m,
                              n,
                              x_stride,
                              y_stride,
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

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
