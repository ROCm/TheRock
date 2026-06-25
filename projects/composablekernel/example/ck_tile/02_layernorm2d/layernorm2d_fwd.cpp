// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "layernorm2d_fwd.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include <algorithm>
#include <cstring>

// different threshold for different dtype
template <typename DataType>
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
    double rtol = 1e-2;
    double atol = 1.0;
    return ck_tile::make_tuple(rtol, atol);
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("x_stride", "-1", "x row_stride, if -1 then equal to n")
        .insert("xr_stride", "-1", "x residule row_stride, if -1 then equal to n")
        .insert("y_stride", "-1", "y row_stride, if -1 then equal to n")
        .insert("yr_stride", "-1", "y residule row_stride, if -1 then equal to n")
        .insert("e", "1e-5", "epsilon")
        .insert("save_mv", "0", "save mean/variance(invstd) or not. set to 1 in training case")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec_i", "fp16", "input precision")
        .insert("prec_o", "auto", "output precision, set auto will be the same as input")
        .insert("prec_sm",
                "auto",
                "output quant scale type, set auto will use fp32. used when fquant=1")
        .insert("prec_sy",
                "auto",
                "output quant scale type, set auto will use fp32. used when fquant=1 or 2")
        .insert("xbias", "0", "add bias, 0:no add, 1:add bias before fadd")
        .insert("fadd", "0", "fused-add, 0:no fused add, 1:preadd+store, 2:preadd only")
        .insert("fquant", "0", "fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "layernorm2d_fwd.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InDataType,
          typename OutDataType,
          typename SmoothScaleDataType,
          typename YScaleDataType,
          bool SaveMeanVar>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m        = arg_parser.get_int("m");
    ck_tile::index_t n        = arg_parser.get_int("n");
    ck_tile::index_t x_stride = arg_parser.get_int("x_stride");
    if(x_stride < 0)
        x_stride = n;
    ck_tile::index_t xr_stride = arg_parser.get_int("xr_stride");
    if(xr_stride < 0)
        xr_stride = n;
    ck_tile::index_t y_stride = arg_parser.get_int("y_stride");
    if(y_stride < 0)
        y_stride = n;
    ck_tile::index_t yr_stride = arg_parser.get_int("yr_stride");
    if(yr_stride < 0)
        yr_stride = n;
    float epsilon       = arg_parser.get_float("e");
    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_sm = arg_parser.get_str("prec_sm");
    std::string prec_sy = arg_parser.get_str("prec_sy");
    if(prec_o == "auto")
    {
        prec_o = prec_i;
    }
    if(prec_sm == "auto")
    {
        prec_sm = "fp32";
    }
    if(prec_sy == "auto")
    {
        prec_sy = "fp32";
    }

    int kname         = arg_parser.get_int("kname");
    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");
    int xbias         = arg_parser.get_int("xbias");
    int fused_add     = arg_parser.get_int("fadd");
    int fused_quant   = arg_parser.get_int("fquant");
    if(fused_quant == 1 && prec_o != "int8" && prec_o != "fp8")
    {
        std::cout
            << "if fused_quant is 1 or 2, only support \"-prec_o=int8\" or \"-prec_o=fp8\" cases."
            << std::endl;
        return false;
    }

    assert(x_stride >= n);

    using TypeConfig =
        LayerNormTypeConfig<InDataType, OutDataType, SmoothScaleDataType, YScaleDataType>;

    using XDataType         = typename TypeConfig::XDataType;
    using YDataType         = typename TypeConfig::YDataType;
    using XBiasDataType     = typename TypeConfig::XBiasDataType;
    using GammaDataType     = typename TypeConfig::GammaDataType;
    using BetaDataType      = typename TypeConfig::BetaDataType;
    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    using MeanDataType =
        std::conditional_t<SaveMeanVar, typename TypeConfig::MeanDataType, ck_tile::null_type>;
    using InvStdDataType =
        std::conditional_t<SaveMeanVar, typename TypeConfig::InvStdDataType, ck_tile::null_type>;

    using ComputeDataType = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {x_stride, 1});
    ck_tile::HostTensor<XBiasDataType> x_bias_host({n});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});
    ck_tile::HostTensor<BetaDataType> beta_host({n});

    ck_tile::HostTensor<XResidualDataType> x_residual_host({m, n}, {xr_stride, 1});
    ck_tile::HostTensor<YResidualDataType> y_residual_host({m, n}, {yr_stride, 1});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}, {y_stride, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {y_stride, 1});

    ck_tile::HostTensor<MeanDataType> mean_host_ref({m});
    ck_tile::HostTensor<InvStdDataType> invStd_host_ref({m});
    ck_tile::HostTensor<YScaleDataType> y_scale_host_ref({m});
    ck_tile::HostTensor<YScaleDataType> y_scale_host_dev({m});

    ck_tile::HostTensor<SmoothScaleDataType> sm_scale_host({n});
    ck_tile::HostTensor<SmoothScaleDataType> sm_scale_host_dev({n});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<XResidualDataType>{-.5f, .5f}(x_residual_host);
    ck_tile::FillUniformDistribution<SmoothScaleDataType>{-1.f, 1.f}(sm_scale_host);
    ck_tile::FillUniformDistribution<XBiasDataType>{-.5f, .5f}(x_bias_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-.5f, .5f}(beta_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_bias_buf(x_bias_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_scale_buf(y_scale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sm_scale_buf(sm_scale_host_dev.get_element_space_size_in_bytes());

    ck_tile::DeviceMem x_residual_buf(x_residual_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_residual_buf(y_residual_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    x_bias_buf.ToDevice(x_bias_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());
    x_residual_buf.ToDevice(x_residual_host.data());
    sm_scale_buf.ToDevice(sm_scale_host.data());

    auto prec_str = [&]() {
        auto base_str = prec_i;
        if(prec_i != prec_o)
        {
            base_str += "|" + prec_o;
        }
        if(fused_quant == 1)
        {
            base_str += std::string("(") + prec_sy + ")";
        }
        return base_str;
    }();

    std::cout << "[" << prec_str << "]" << " m:" << m << ", n:" << n << ", x_stride:" << x_stride
              << ", xr_stride:" << xr_stride << ", y_stride:" << y_stride
              << ", yr_stride:" << yr_stride << std::flush;

    layernorm2d_fwd_traits traits{
        prec_i, prec_o, prec_sm, prec_sy, SaveMeanVar, xbias, fused_add, fused_quant};

    layernorm2d_fwd_args args{x_buf.GetDeviceBuffer(),
                              fused_add != 0 ? x_residual_buf.GetDeviceBuffer() : nullptr,
                              fused_quant == 1 ? sm_scale_buf.GetDeviceBuffer() : nullptr,
                              x_bias_buf.GetDeviceBuffer(),
                              gamma_buf.GetDeviceBuffer(),
                              beta_buf.GetDeviceBuffer(),

                              y_buf.GetDeviceBuffer(),
                              fused_add == 1 ? y_residual_buf.GetDeviceBuffer() : nullptr,
                              fused_quant != 0 ? y_scale_buf.GetDeviceBuffer() : nullptr,
                              nullptr, // p_mean, unsupported yet
                              nullptr, // p_invStd, unsupported yet

                              epsilon,
                              m,
                              n,
                              x_stride,   // x row_stride
                              xr_stride,  // x residule row stride
                              y_stride,   // y row stride
                              yr_stride}; // y residule row stride

    float ave_time = layernorm2d_fwd(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    if(ave_time < 0)
    {
        std::cout << " not supported!" << std::endl << std::flush;
        return false;
    }

    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(XBiasDataType) * n +
                           sizeof(GammaDataType) * n + sizeof(BetaDataType) * n +
                           sizeof(YDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        // reference
        if(xbias != 0)
        {
            // add bias before fadd
            int M = x_host.mDesc.get_lengths()[0];
            int N = x_host.mDesc.get_lengths()[1];
            for(int idx_m = 0; idx_m < M; ++idx_m)
            {
                for(int idx_n = 0; idx_n < N; ++idx_n)
                {
                    x_host(idx_m, idx_n) = ck_tile::type_convert<XDataType>(
                        ck_tile::type_convert<ComputeDataType>(x_host(idx_m, idx_n)) +
                        ck_tile::type_convert<ComputeDataType>(x_bias_host(idx_n)));
                }
            }
        }

        if(fused_add != 0)
        {
            // fused pre_add/pre_add_store
            // TODO we accumulate directly to x_host for simplcity here...

            std::transform(x_host.mData.cbegin(),
                           x_host.mData.cend(),
                           x_residual_host.mData.cbegin(),
                           x_host.mData.begin(),
                           [](auto x_, auto r_) {
                               auto o_ = ck_tile::type_convert<ComputeDataType>(x_) +
                                         ck_tile::type_convert<ComputeDataType>(r_);
                               return ck_tile::type_convert<XDataType>(o_);
                           });
        }
        ck_tile::reference_layernorm2d_fwd<XDataType,
                                           GammaDataType,
                                           BetaDataType,
                                           ComputeDataType,
                                           YDataType,
                                           MeanDataType,
                                           InvStdDataType>(
            x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

        if(fused_quant != 0)
        {
            auto dquant_functor = [&](int m_, auto& o_, auto& acc_) {
                int N_ = acc_.mDesc.get_lengths()[1];
                if(fused_quant == 1)
                {
                    for(int n_ = 0; n_ < N_; n_++)
                    {
                        // input smooth outlier
                        acc_(m_, n_) = acc_(m_, n_) *
                                       ck_tile::type_convert<ComputeDataType>(sm_scale_host(n_));
                    }
                }
                ComputeDataType absmax = static_cast<ComputeDataType>(0);
                for(int n_ = 0; n_ < N_; n_++)
                {
                    const auto a = ck_tile::abs(acc_(m_, n_));
                    absmax       = a > absmax ? a : absmax;
                }
                // printf("cpu:absmax:%f\n", absmax);
                constexpr ComputeDataType kMaxY =
                    std::is_same<YDataType, ck_tile::fp8_t>::value    ? 240.0
                    : std::is_same<YDataType, ck_tile::int8_t>::value ? 127.0
                                                                      : 0.0;
                ComputeDataType y_scale = absmax / kMaxY;
                y_scale_host_ref(m_)    = ck_tile::type_convert<YScaleDataType>(y_scale);
                for(int n_ = 0; n_ < N_; n_++)
                {
                    o_(m_, n_) = ck_tile::type_convert<YDataType>(acc_(m_, n_) / y_scale);
                }
            };

            ck_tile::reference_layernorm2d_fwd<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(x_host,
                                                               gamma_host,
                                                               beta_host,
                                                               y_host_ref,
                                                               mean_host_ref,
                                                               invStd_host_ref,
                                                               epsilon,
                                                               dquant_functor);
        }
        else
        {
            ck_tile::reference_layernorm2d_fwd<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(
                x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);
        }

        y_buf.FromDevice(y_host_dev.data());

        ck_tile::HostTensor<YResidualDataType> y_residual_host_dev({m, n}, {yr_stride, 1});
        if(fused_add == 1)
        {
            y_residual_buf.FromDevice(y_residual_host_dev.data());
        }

        auto [rtol, atol] = get_elimit<OutDataType>();

        if(x_stride == n)
        {
            pass = ck_tile::check_err(
                y_host_dev, y_host_ref, std::string("OUT Error: Incorrect results!"), rtol, atol);
            if(fused_add == 1)
            {
                pass &= ck_tile::check_err(y_residual_host_dev,
                                           x_host,
                                           std::string("ADD Error: Incorrect results!"),
                                           rtol,
                                           atol);
            }
        }
        else
        {
            for(int i_r = 0; i_r < m; i_r++)
            {
                std::vector<YDataType> y_host_dev_row(y_host_dev.begin() + i_r * y_stride,
                                                      y_host_dev.begin() + i_r * y_stride + n);
                std::vector<YDataType> y_host_ref_row(y_host_ref.begin() + i_r * y_stride,
                                                      y_host_ref.begin() + i_r * y_stride + n);
                pass &= ck_tile::check_err(y_host_dev_row,
                                           y_host_ref_row,
                                           std::string("OUT[") + std::to_string(i_r) +
                                               std::string("] Error: Incorrect results!"),
                                           rtol,
                                           atol);
                if(fused_add == 1)
                {
                    std::vector<YResidualDataType> y_residual_host_dev_row(
                        y_residual_host_dev.begin() + i_r * yr_stride,
                        y_residual_host_dev.begin() + i_r * yr_stride + n);
                    std::vector<YResidualDataType> y_residual_host_ref_row(
                        x_host.begin() + i_r * yr_stride, x_host.begin() + i_r * yr_stride + n);
                    pass &= ck_tile::check_err(y_residual_host_dev_row,
                                               y_residual_host_ref_row,
                                               std::string("ADD[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }
            }
        }
        if(fused_quant == 1)
        {
            y_scale_buf.FromDevice(y_scale_host_dev.data());
            pass &= ck_tile::check_err(y_scale_host_dev,
                                       y_scale_host_ref,
                                       std::string("SCALE Error: Incorrect results!"),
                                       rtol,
                                       atol);
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_layernorm2d_fwd_json_results(arg_parser.get_str("jsonfile"),
                                          prec_i,
                                          prec_o,
                                          prec_sm,
                                          prec_sy,
                                          m,
                                          n,
                                          x_stride,
                                          xr_stride,
                                          y_stride,
                                          yr_stride,
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

    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_sm = arg_parser.get_str("prec_sm");
    std::string prec_sy = arg_parser.get_str("prec_sy");

    if(prec_o == "auto")
    {
        prec_o = prec_i;
    }
    if(prec_sm == "auto")
    {
        prec_sm = "fp32";
    }
    if(prec_sy == "auto")
    {
        prec_sy = "fp32";
    }
    int save_mv = arg_parser.get_int("save_mv");

    // no dynamic quant case
    if(prec_i == "fp16" && prec_o == "fp16" && prec_sm == "fp32" && prec_sy == "fp32" && save_mv)
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float, true>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "fp16" && prec_o == "fp16" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_mv)
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_sm == "fp32" && prec_sy == "fp32" &&
            save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float, true>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float, true>(arg_parser) ? 0 : -2;
    }

    // dynamic quant case, only in inference
    else if(prec_i == "fp16" && prec_o == "int8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_mv)
    {
        return run<ck_tile::half_t, ck_tile::int8_t, float, float, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "int8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, float, float, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "fp16" && prec_o == "fp8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_mv)
    {
        return run<ck_tile::half_t, ck_tile::fp8_t, float, float, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "fp8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t, float, float, false>(arg_parser) ? 0 : -2;
    }

    return -3;
}
