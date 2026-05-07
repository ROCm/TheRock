// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "smoothquant.hpp"
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

template <typename Tuple>
class TestCkTileSmoothquant : public ::testing::Test
{

    protected:
    using DataType = std::tuple_element_t<0, Tuple>;

    void Run(ck_tile::index_t m,
             ck_tile::index_t n,
             ck_tile::index_t x_stride = -1,
             ck_tile::index_t y_stride = -1)
    {
        if(x_stride < 0)
            x_stride = n;
        if(y_stride < 0)
            y_stride = n;

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

        std::cout << "m:" << m << ", n:" << n << ", x_stride:" << x_stride
                  << ", y_stride:" << y_stride << std::flush;

        smoothquant_args args{x_buf.GetDeviceBuffer(),
                              smscale_buf.GetDeviceBuffer(),
                              yscale_buf.GetDeviceBuffer(),
                              qy_buf.GetDeviceBuffer(),
                              m,
                              n,
                              x_stride,
                              y_stride};

        smoothquant<DataType>(args, ck_tile::stream_config{nullptr, false});

        bool pass = true;

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

        EXPECT_TRUE(pass);
    }
};
