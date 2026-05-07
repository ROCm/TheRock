// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "moe_smoothquant.hpp"
#include <cstring>
#include <set>
#include <hip/hip_runtime.h>

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

template <typename Tuple>
class TestCkTileMoeSmoothquant : public ::testing::Test
{
    protected:
    using InputType  = std::tuple_element_t<0, Tuple>;
    using OutputType = std::tuple_element_t<1, Tuple>;

    void Run(ck_tile::index_t tokens,
             ck_tile::index_t hidden_size,
             ck_tile::index_t stride  = -1,
             ck_tile::index_t experts = 32,
             ck_tile::index_t topk    = 5)
    {
        if(stride < 0)
            stride = hidden_size;

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

        std::cout << "tokens:" << tokens << ", hidden_size:" << hidden_size << ", stride:" << stride
                  << ", experts:" << experts << ", topk:" << topk << std::flush;

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

        moe_smoothquant<InputType, OutputType>(args, ck_tile::stream_config{nullptr, false});

        bool pass = true;

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

        EXPECT_TRUE(pass);
    }
};
