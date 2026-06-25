// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("n", "16", "n dimension")
        .insert("h", "64", "h dimension")
        .insert("w", "32", "w dimension")
        .insert("c", "960", "c dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "20", "cold iter")
        .insert("repeat", "100", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "reduce.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t H = arg_parser.get_int("h");
    ck_tile::index_t W = arg_parser.get_int("w");
    ck_tile::index_t C = arg_parser.get_int("c");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    std::vector<ck_tile::index_t> problem_shape = {N, H, W, C};
    std::vector<ck_tile::index_t> strides(4);
    strides[0] = H * W * C;
    strides[1] = W * C;
    strides[2] = C;
    strides[3] = 1;

    // Define reduction specification:
    constexpr auto kept_dim    = ck_tile::sequence<1, 2, 3>{}; // Which dimension to keep
    constexpr auto reduce_dims = ck_tile::sequence<0>{};       // Which dimensions to reduce

    ck_tile::HostTensor<XDataType> x_host(problem_shape, strides);
    ck_tile::HostTensor<YDataType> y_host_ref({H, W, C}, {W * C, C, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({H, W, C}, {W * C, C, 1});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    using ReduceOp   = ck_tile::ReduceOp::Add;
    using BlockWarps = ck_tile::sequence<1, 1>;
    using BlockTile  = ck_tile::sequence<256, 1>;
    using WarpTile   = ck_tile::sequence<256, 1>;
    using ThreadTile = ck_tile::sequence<1, 1>;

    // cross warp-reduce
    // using BlockWarps = ck_tile::sequence<2, 2>;
    // using BlockTile  = ck_tile::sequence<2, 1024>;
    // using WarpTile   = ck_tile::sequence<1, 512>;
    // using ThreadTile = ck_tile::sequence<1, 8>;

    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kept_dim_len_prod     = H * W * C;
    ck_tile::index_t kGridSize = (kept_dim_len_prod + BlockTile::at(ck_tile::number<0>{}) - 1) /
                                 BlockTile::at(ck_tile::number<0>{});
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape   = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;
    using Porblem = ck_tile::Reduce2dProblem<XDataType,
                                             ComputeDataType,
                                             YDataType,
                                             Shape,
                                             ReduceOp,
                                             decltype(kept_dim),
                                             decltype(reduce_dims),
                                             4>;

    using Kernel                      = ck_tile::ReduceKernel<Porblem>;
    const ck_tile::index_t kBlockSize = Kernel::BlockSize();
    // Create input tensor shape and strides
    auto input_shape =
        ck_tile::make_tuple(problem_shape[0], problem_shape[1], problem_shape[2], problem_shape[3]);
    auto input_strides = ck_tile::make_tuple(strides[0], strides[1], strides[2], strides[3]);

    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                          static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                          input_shape,
                                          input_strides));

    std::size_t num_btype = sizeof(XDataType) * N * H * W * C + sizeof(YDataType) * H * W * C;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_reduce<XDataType, ComputeDataType, YDataType>(
            x_host, y_host_ref, ReduceOp{}, kept_dim, reduce_dims);
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_reduce_json_results<DataType, ck_tile::DataTypeTraits>(
            arg_parser.get_str("jsonfile"), N, C, H, W, pass, ave_time, 0, gb_per_sec);
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
}
