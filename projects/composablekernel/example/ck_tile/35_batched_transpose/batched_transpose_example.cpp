// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "batched_transpose_example.hpp"

#include "ck_tile/utility/json_dump.hpp"
#if 0
template <typename T>
void dump_host_tensor_4d(const ck_tile::HostTensor<T>& x)
{
    auto len = x.get_lengths();
    assert(len.size() == 4);
    std::cout << "[";
    for(size_t i = 0; i < len[0]; i++)
    {
        std::cout << "Batch " << i << ":" << std::endl;
        for(size_t j = 0; j < len[1]; j++)
        {
            std::cout << "  Channel " << j << ":" << std::endl;
            for(size_t k = 0; k < len[2]; k++)
            {
                std::cout << "    Row " << k << ": ";
                for(size_t v = 0; v < len[3]; v++)
                {
                    if constexpr(std::is_same_v<T, ck_tile::fp16_t>)
                    {
                        auto m =
                            ck_tile::type_convert<float>(x(std::vector<std::size_t>{i, j, k, v}));

                        std::cout << m;
                        if(v != len[3] - 1)
                            std::cout << ",";
                    }
                    else
                    {
                        std::cout << static_cast<int>(x(std::vector<std::size_t>{i, j, k, v}))
                                  << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "--------------------" << std::endl;
}
#endif

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
    arg_parser.insert("v", "1", "whether do CPU validation or not")
        .insert("pr", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("N", "1", "input batch size. ")
        .insert("C", "64", "input channel size.")
        .insert("H", "18", "input height size.")
        .insert("W", "64", "input width size. ")
        .insert("layout_in", "NCHW", "input tensor data layout - NCHW by default")
        .insert("layout_out", "NHWC", "output tensor data layout - NHWC by default ")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "t to 1 will print kernel name")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "batched_transpose.json", "json file name to dump results")
        .insert("pipeline", "0", "0: no LDS usage, 1: LDS-accelerated (gfx950)");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename Type>
bool run_batched_transpose(ck_tile::ArgParser args)
{
    int validate           = args.get_int("v");
    std::string prec       = args.get_str("pr");
    int N                  = args.get_int("N");
    int C                  = args.get_int("C");
    int H                  = args.get_int("H");
    int W                  = args.get_int("W");
    int n_warmup           = args.get_int("warmup");
    int n_repeat           = args.get_int("repeat");
    std::string layout_in  = args.get_str("layout_in");
    std::string layout_out = args.get_str("layout_out");
    std::string pipeline   = args.get_str("pipeline");
    int seed               = args.get_int("seed");

    int dim_in[4], dim_out[4];
    int stride_dim_in[4], stride_dim_out[4];
    bool nchw2nhwc = layout_in == "NCHW" && layout_out == "NHWC";
    bool nhwc2nchw = layout_in == "NHWC" && layout_out == "NCHW";
    assert(nchw2nhwc != nhwc2nchw);
    (void)nhwc2nchw;

    dim_in[0]         = N;
    dim_in[1]         = nchw2nhwc ? C : H;
    dim_in[2]         = nchw2nhwc ? H : W;
    dim_in[3]         = nchw2nhwc ? W : C;
    dim_out[0]        = N;
    dim_out[1]        = nchw2nhwc ? H : C;
    dim_out[2]        = nchw2nhwc ? W : H;
    dim_out[3]        = nchw2nhwc ? C : W;
    stride_dim_in[0]  = C * H * W;
    stride_dim_in[1]  = nchw2nhwc ? H * W : C * W;
    stride_dim_in[2]  = nchw2nhwc ? W : C;
    stride_dim_in[3]  = 1;
    stride_dim_out[0] = C * H * W;
    stride_dim_out[1] = nchw2nhwc ? C * W : H * W;
    stride_dim_out[2] = nchw2nhwc ? C : W;
    stride_dim_out[3] = 1;

    if(seed < 0)
    {
        seed = std::time(nullptr);
    }

    ck_tile::HostTensor<Type> x_host(
        {dim_in[0], dim_in[1], dim_in[2], dim_in[3]},
        {stride_dim_in[0], stride_dim_in[1], stride_dim_in[2], stride_dim_in[3]});
    ck_tile::HostTensor<Type> y_host(
        {dim_out[0], dim_out[1], dim_out[2], dim_out[3]},
        {stride_dim_out[0], stride_dim_out[1], stride_dim_out[2], stride_dim_out[3]});

    ck_tile::FillUniformDistribution<Type>{-.5f, .5f}(x_host);

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

    auto trait = batched_transpose_trait{prec, layout_in, pipeline};

    uint32_t height = nchw2nhwc ? C : H * W;
    uint32_t width  = nchw2nhwc ? H * W : C;

    batched_transpose_kargs karg = [&]() {
        batched_transpose_kargs a_;
        a_.p_input  = x_dev.GetDeviceBuffer();
        a_.p_output = y_dev.GetDeviceBuffer();
        a_.batch    = N;
        a_.height   = height;
        a_.width    = width;
        return a_;
    }();

    ck_tile::stream_config sc{nullptr, true, n_warmup, n_repeat};

    auto ms = batched_transpose(trait, karg, sc);

    std::size_t num_bytes = N * C * H * W * sizeof(Type) * 2; // read + written

    float gb_per_sec = num_bytes / ms * 1.E-6;

    std::cout << "Run Batched Transpose kernel with N=" << N << ", C=" << C << ", H=" << H
              << ", W=" << W << ", layout_in=" << layout_in << ", layout_out=" << layout_out
              << " : " << std::endl
              << ms << " ms " << std::endl
              << gb_per_sec << " GB/s " << std::endl;

    printf("[%s]N:%d, C:%d, H:%d, W:%d, layout_in:%s, %f\n",
           prec.c_str(),
           N,
           C,
           H,
           W,
           layout_in.c_str(),
           ms);
    if(ms < 0)
        printf("------------------------------------not "
               "supported-------------------------------------\n");
    fflush(stdout);

    if(ms < 0)
    {
        return false;
    }

    y_dev.FromDevice(y_host.data());

    bool rtn = true;
    if(validate)
    {
        // this host buffer will not copy to GPU, so no need use stride
        ck_tile::HostTensor<Type> y_ref(
            {dim_out[0], dim_out[1], dim_out[2], dim_out[3]},
            {stride_dim_out[0], stride_dim_out[1], stride_dim_out[2], stride_dim_out[3]});

        ck_tile::reference_batched_transpose<Type>(x_host, y_ref, layout_in, layout_out);

        auto [rtol, atol] = get_elimit<Type>("");

        rtn &= ck_tile::check_err(
            y_host, y_ref, std::string("y Error: Incorrect results!"), rtol, atol);
    }
    printf("-----------------------------------------------------------------------valid:%s--------"
           "--------------------------------------------------------------------\n",
           rtn ? "y" : "n");
    fflush(stdout);

    if(args.get_int("json") == 1)
    {
        dump_batched_transpose_json(args.get_str("jsonfile"),
                                    N,
                                    C,
                                    H,
                                    W,
                                    layout_in,
                                    layout_out,
                                    prec,
                                    ms,
                                    0,
                                    gb_per_sec,
                                    rtn);
    }

    return rtn;
}

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;
    std::string prec = args.get_str("pr");

    bool r = true;
    if(prec.compare("fp8") == 0)
    {
        r &= run_batched_transpose<ck_tile::fp8_t>(args);
    }
    else if(prec.compare("fp16") == 0)
    {
        r &= run_batched_transpose<ck_tile::fp16_t>(args);
    }
    else if(prec.compare("bf16") == 0)
    {
        r &= run_batched_transpose<ck_tile::bf16_t>(args);
    }

    return r ? 0 : -1;
}
