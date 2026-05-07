// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "rmsnorm2d_fwd.hpp"
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
    double rtol = 1e-02;
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
        .insert("save_rms", "0", "save rms(invrms) or not. set to 1 in training case")
        .insert("save_unquant", "0", "save result before quant")
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
        .insert("fadd", "0", "fused-add, 0:no fused add, 1:preadd+store, 2:preadd only")
        .insert("fquant", "0", "fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InDataType,
          typename OutDataType,
          typename SmoothScaleDataType,
          typename YScaleDataType,
          bool SaveRms,
          bool SaveUnquant>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    float epsilon      = arg_parser.get_float("e");
    int kname          = arg_parser.get_int("kname");
    int do_validation  = arg_parser.get_int("v");
    int fused_add      = arg_parser.get_int("fadd");
    int fused_quant    = arg_parser.get_int("fquant");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

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
    assert(x_stride >= n);

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

    if((fused_quant == 1 || fused_quant == 2) && prec_o != "int8" && prec_o != "fp8")
    {
        std::cout
            << "if fused_quant is 1 or 2, only support \"-prec_o=int8\" or \"-prec_o=fp8\" cases."
            << std::endl;
        return false;
    }

    if((fused_quant == 0) && SaveUnquant)
    {
        std::cout
            << "save_unquant should be 0 if quant output is not enabled because it is meaningless. "
            << "Output Y is what wanted." << std::endl;
        return false;
    }

    using TypeConfig =
        RmsnormTypeConfig<InDataType, OutDataType, SmoothScaleDataType, YScaleDataType>;

    using XDataType         = typename TypeConfig::XDataType;
    using YDataType         = typename TypeConfig::YDataType;
    using GammaDataType     = typename TypeConfig::GammaDataType;
    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    using InvRmsDataType =
        std::conditional_t<SaveRms, typename TypeConfig::InvRmsDataType, ck_tile::null_type>;
    using UnquantYDataType =
        std::conditional_t<SaveUnquant, typename TypeConfig::UnquantYDataType, ck_tile::null_type>;

    using ComputeDataType = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {x_stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});
    ck_tile::HostTensor<SmoothScaleDataType> sm_scale_host({n});
    ck_tile::HostTensor<SmoothScaleDataType> sm_scale_host_dev({n});

    ck_tile::HostTensor<XResidualDataType> x_residual_host({m, n}, {xr_stride, 1});
    ck_tile::HostTensor<YResidualDataType> y_residual_host({m, n}, {yr_stride, 1});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}, {y_stride, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {y_stride, 1});
    ck_tile::HostTensor<YScaleDataType> y_scale_host_ref({m});
    ck_tile::HostTensor<YScaleDataType> y_scale_host_dev({m});

    ck_tile::HostTensor<InvRmsDataType> invRms_host_ref({m});

    ck_tile::HostTensor<UnquantYDataType> unquant_y_host_ref({m, n}, {y_stride, 1});
    ck_tile::HostTensor<UnquantYDataType> unquant_y_host_dev({m, n}, {y_stride, 1});
    ck_tile::HostTensor<ck_tile::null_type> unquant_y_null({1});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<XResidualDataType>{-.5f, .5f}(x_residual_host);
    ck_tile::FillUniformDistribution<SmoothScaleDataType>{-1.f, 1.f}(sm_scale_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_scale_buf(y_scale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sm_scale_buf(sm_scale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_residual_buf(x_residual_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_residual_buf(y_residual_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem unquant_y_buf(unquant_y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
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

    rmsnorm2d_fwd_traits traits{
        prec_i, prec_o, prec_sm, prec_sy, SaveRms, SaveUnquant, fused_add, fused_quant};

    rmsnorm2d_fwd_args args{x_buf.GetDeviceBuffer(),
                            fused_add != 0 ? x_residual_buf.GetDeviceBuffer() : nullptr,
                            fused_quant == 1 ? sm_scale_buf.GetDeviceBuffer() : nullptr,
                            gamma_buf.GetDeviceBuffer(),
                            y_buf.GetDeviceBuffer(),
                            fused_add == 1 ? y_residual_buf.GetDeviceBuffer() : nullptr,
                            fused_quant != 0 ? y_scale_buf.GetDeviceBuffer() : nullptr,
                            nullptr, // p_invRms, unsupported yet
                            SaveUnquant ? unquant_y_buf.GetDeviceBuffer() : nullptr,
                            epsilon,
                            m,
                            n,
                            x_stride,   // x row_stride
                            xr_stride,  // x residule row stride
                            y_stride,   // y row stride
                            yr_stride}; // y residule row stride

    float ave_time = rmsnorm2d_fwd(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    std::size_t num_byte =
        sizeof(XDataType) * m * n + sizeof(GammaDataType) * n + sizeof(YDataType) * m * n;
    num_byte += SaveRms ? sizeof(InvRmsDataType) * m * n : 0;
    num_byte += SaveUnquant ? sizeof(UnquantYDataType) * m * n : 0;
    num_byte += fused_add ? sizeof(XResidualDataType) * m * n : 0;
    num_byte += ((fused_quant == 1) || (fused_quant == 2)) ? sizeof(YScaleDataType) * m : 0;
    num_byte += (fused_quant == 1) ? sizeof(SmoothScaleDataType) * n : 0;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        // reference
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

            auto default_and_dquant_functor = [&](int m_, auto& o_unquant_, auto& o_, auto& acc_) {
                const int N = acc_.mDesc.get_lengths()[1];
                for(int n_ = 0; n_ < N; ++n_)
                {
                    o_unquant_(m_, n_) = ck_tile::type_convert<OutDataType>(acc_(m_, n_));
                }

                dquant_functor(m_, o_, acc_);
            };

            if constexpr(SaveUnquant)
            {
                ck_tile::reference_rmsnorm2d_fwd<XDataType,
                                                 GammaDataType,
                                                 ComputeDataType,
                                                 YDataType,
                                                 InvRmsDataType,
                                                 UnquantYDataType>(x_host,
                                                                   gamma_host,
                                                                   y_host_ref,
                                                                   invRms_host_ref,
                                                                   unquant_y_host_ref,
                                                                   epsilon,
                                                                   default_and_dquant_functor);
            }
            else
            {
                ck_tile::reference_rmsnorm2d_fwd<XDataType,
                                                 GammaDataType,
                                                 ComputeDataType,
                                                 YDataType,
                                                 InvRmsDataType,
                                                 UnquantYDataType>(x_host,
                                                                   gamma_host,
                                                                   y_host_ref,
                                                                   invRms_host_ref,
                                                                   unquant_y_host_ref,
                                                                   epsilon,
                                                                   dquant_functor);
            }
        }
        else
        {
            assert(SaveUnquant == false);
            ck_tile::reference_rmsnorm2d_fwd<XDataType,
                                             GammaDataType,
                                             ComputeDataType,
                                             YDataType,
                                             InvRmsDataType,
                                             ck_tile::null_type>(
                x_host, gamma_host, y_host_ref, invRms_host_ref, unquant_y_null, epsilon);
        }

        y_buf.FromDevice(y_host_dev.data());

        ck_tile::HostTensor<YResidualDataType> y_residual_host_dev({m, n}, {yr_stride, 1});
        if(fused_add == 1)
        {
            y_residual_buf.FromDevice(y_residual_host_dev.data());
        }

        auto [rtol, atol] = get_elimit<YDataType>();
        if(x_stride == n)
        {
            pass = ck_tile::check_err(
                y_host_dev, y_host_ref, std::string("\nOUT Error: Incorrect results!"), rtol, atol);

            if constexpr(SaveUnquant)
            {
                pass &= ck_tile::check_err(unquant_y_host_dev,
                                           unquant_y_host_ref,
                                           std::string("\n OUT ERROR: Incorrect unquant results!"),
                                           rtol,
                                           atol);
            }

            if(fused_add == 1)
            {
                pass &= ck_tile::check_err(y_residual_host_dev,
                                           x_host,
                                           std::string("\nADD Error: Incorrect results!"),
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
                                           std::string("\nOUT[") + std::to_string(i_r) +
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
                                               std::string("\nADD[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }

                if constexpr(SaveUnquant)
                {
                    std::vector<UnquantYDataType> unquant_y_host_dev_row(
                        unquant_y_host_dev.begin() + i_r * y_stride,
                        unquant_y_host_dev.begin() + i_r * y_stride + n);
                    std::vector<UnquantYDataType> unquant_y_host_ref_row(
                        unquant_y_host_ref.begin() + i_r * y_stride,
                        unquant_y_host_ref.begin() + i_r * y_stride + n);
                    pass &=
                        ck_tile::check_err(unquant_y_host_dev_row,
                                           unquant_y_host_ref_row,
                                           std::string("\nOUT[") + std::to_string(i_r) +
                                               std::string("] Error: Incorrect unquant y results!"),
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
                                       std::string("\nSCALE Error: Incorrect results!"),
                                       rtol,
                                       atol);
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

bool is_quant_data_type(const std::string& prec) { return (prec == "int8") || (prec == "fp8"); }

bool dispatch_by_type(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return false;

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

    int save_rms    = arg_parser.get_int("save_rms");
    int fused_quant = arg_parser.get_int("fquant");
    int save_unquant =
        arg_parser.get_int("save_unquant") && is_quant_data_type(prec_o) && (fused_quant != 0);

    if(prec_i == "fp16" && prec_o == "fp16" && prec_sm == "fp32" && prec_sy == "fp32" && save_rms)
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float, true, false>(arg_parser);
    }
    else if(prec_i == "fp16" && prec_o == "fp16" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms)
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float, false, false>(arg_parser);
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_sm == "fp32" && prec_sy == "fp32" &&
            save_rms)
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float, true, false>(arg_parser);
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms)
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float, false, false>(arg_parser);
    }

    // dynamic quant case, only in inference
    else if(prec_i == "fp16" && prec_o == "int8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && !save_unquant)
    {
        return run<ck_tile::half_t, ck_tile::int8_t, float, float, true, false>(arg_parser);
    }
    else if(prec_i == "bf16" && prec_o == "int8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && !save_unquant)
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, float, float, true, false>(arg_parser);
    }
    else if(prec_i == "fp16" && prec_o == "fp8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && !save_unquant)
    {
        return run<ck_tile::half_t, ck_tile::fp8_t, float, float, false, false>(arg_parser);
    }
    else if(prec_i == "bf16" && prec_o == "fp8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && !save_unquant)
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t, float, float, false, false>(arg_parser);
    }
    else if(prec_i == "fp16" && prec_o == "int8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && save_unquant)
    {
        return run<ck_tile::half_t, ck_tile::int8_t, float, float, true, true>(arg_parser);
    }
    else if(prec_i == "bf16" && prec_o == "int8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && save_unquant)
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, float, float, true, true>(arg_parser);
    }
    else if(prec_i == "fp16" && prec_o == "fp8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && save_unquant)
    {
        return run<ck_tile::half_t, ck_tile::fp8_t, float, float, false, true>(arg_parser);
    }
    else if(prec_i == "bf16" && prec_o == "fp8" && prec_sm == "fp32" && prec_sy == "fp32" &&
            !save_rms && save_unquant)
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t, float, float, false, true>(arg_parser);
    }

    return false;
}

int run_rmsnorm2d_fwd_combinations(std::string const& data_type)
{
    constexpr size_t PARAM_COUNT = 20;
    char bufs[PARAM_COUNT][64];
    char* argv[PARAM_COUNT];

    for(std::size_t i = 0; i < PARAM_COUNT; i++)
    {
        argv[i] = bufs[i];
    }

    std::vector<std::vector<std::string>> fquant = {
        {},
        {"-fquant=1", "-prec_o=int8"},
        {"-fquant=2", "-prec_o=int8"},
        {"-fquant=1", "-prec_o=fp8"},
        {"-fquant=2", "-prec_o=fp8"},
        {"-fquant=1", "-prec_o=int8", "-save_unquant=1"},
        {"-fquant=2", "-prec_o=int8", "-save_unquant=1"},
        {"-fquant=1", "-prec_o=fp8", "-save_unquant=1"},
        {"-fquant=2", "-prec_o=fp8", "-save_unquant=1"}};

    std::vector<std::string> fadd = {"-fadd=0", "-fadd=1"};

    std::vector<std::vector<std::string>> params = {
        {"-m=99", "-n=13"},
        {"-m=17", "-n=16"},
        {"-m=1", "-n=100"},
        {"-m=4", "-n=128"},
        {"-m=80", "-n=127"},
        {"-m=22", "-n=255", "-stride=256"},
        {"-m=7", "-n=599"},
        {"-m=19", "-n=512"},
        {"-m=33", "-n=313", "-stride=1000"},
        {"-m=11", "-n=510"},
        {"-m=171", "-n=676", "-stride=818"},
        {"-m=91", "-n=636"},
        {"-m=12", "-n=768", "-stride=800"},
        {"-m=100", "-n=766", "-stride=812"},
        {"-m=31", "-n=1024"},
        {"-m=64", "-n=1000", "-stride=1004"},
        {"-m=8", "-n=1501"},
        {"-m=3", "-n=1826"},
        {"-m=5", "-n=2040"},
        {"-m=7", "-n=2734"},
        {"-m=1", "-n=3182"},
        {"-m=9", "-n=4096"},
        {"-m=3", "-n=8192"},
    };

    bool result = true;
    int argc    = 0;
    std::vector<int> argc_stack;
    std::string pr_i = "-prec_i=" + data_type;
    strncpy(bufs[argc++], "rmsnorm2d_fwd", 64);
    strncpy(bufs[argc++], pr_i.c_str(), 64);
    argc_stack.push_back(argc);
    for(size_t fquant_idx = 0; fquant_idx < fquant.size(); fquant_idx++)
    {
        argc = argc_stack.back();
        for(size_t j = 0; j < fquant[fquant_idx].size(); j++)
        {
            strncpy(bufs[argc++], fquant[fquant_idx][j].c_str(), 64);
        }
        argc_stack.push_back(argc);
        for(size_t fadd_idx = 0; fadd_idx < fadd.size(); fadd_idx++)
        {
            argc = argc_stack.back();
            strncpy(bufs[argc++], fadd[fadd_idx].c_str(), 64);
            argc_stack.push_back(argc);
            for(size_t param_idx = 0; param_idx < params.size(); param_idx++)
            {
                argc = argc_stack.back();
                for(size_t j = 0; j < params[param_idx].size(); j++)
                {
                    strncpy(bufs[argc++], params[param_idx][j].c_str(), 64);
                }

                result = dispatch_by_type(argc, argv) && result;
            }
            argc_stack.pop_back();
        }
        argc_stack.pop_back();
    }
    argc_stack.pop_back();
    return result ? 0 : -1;
}
