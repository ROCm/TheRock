// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/reference/reference_elementwise.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include "elementwise_common.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("dim0", "4", "dimension 0")
        .insert("dim1", "16", "dimension 1")
        .insert("dim2", "32", "dimension 2")
        .insert("dim3", "32", "dimension 3")
        .insert("v", "1", "cpu validation or not")
        .insert("x_prec", "fp16", "input precision")
        .insert("y_prec", "fp16", "output precision")
        .insert("warmup", "10", "cold iter")
        .insert("repeat", "50", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "elementwise_add_4d.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename XDataType, typename YDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t D0 = arg_parser.get_int("dim0");
    ck_tile::index_t D1 = arg_parser.get_int("dim1");
    ck_tile::index_t D2 = arg_parser.get_int("dim2");
    ck_tile::index_t D3 = arg_parser.get_int("dim3");

    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    using ComputeDataType =
        float; // Using float for intermediate calculations can improve numerical stability.
    using XElementwiseOperation = ck_tile::element_wise::Add;

    // Initialize the input data on the host (CPU).
    std::vector<ck_tile::index_t> problem_shape = {D0, D1, D2, D3};

    std::vector<ck_tile::index_t> host_strides(4);
    host_strides[3] = 1;
    host_strides[2] = problem_shape[3];
    host_strides[1] = problem_shape[2] * problem_shape[3];
    host_strides[0] = problem_shape[1] * problem_shape[2] * problem_shape[3];

    ck_tile::HostTensor<XDataType> x_host_a(problem_shape, host_strides);
    ck_tile::HostTensor<XDataType> x_host_b(problem_shape, host_strides);
    ck_tile::HostTensor<YDataType> y_host(problem_shape, host_strides);
    ck_tile::HostTensor<YDataType> y_validation(problem_shape, host_strides);

    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{2.f, 10.f}(x_host_b);

    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());

    using BlockTile  = ck_tile::sequence<256>;
    using BlockWarps = ck_tile::sequence<1>;
    using WarpTile   = ck_tile::sequence<256>;

    using Shape = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, XDataType>;

    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    ck_tile::index_t total_elements = 1;
    for(auto d : problem_shape)
        total_elements *= d;

    const ck_tile::index_t kBlockSize = Kernel::BlockSize();

    constexpr ck_tile::index_t kBlockPerCu = 2;

    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;

    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                             static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()));

    auto problem_shape_tuple =
        ck_tile::make_tuple(problem_shape[0], problem_shape[1], problem_shape[2], problem_shape[3]);

    auto strides_tuple =
        ck_tile::make_tuple(host_strides[0], host_strides[1], host_strides[2], host_strides[3]);

    // Check if the kernel configuration is supported
    if(!Kernel::IsSupportedArgument(problem_shape_tuple))
    {
        throw std::runtime_error(
            "The kernel configuration is not supported for the given input size.");
    }

    // Run the kernel
    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(
            Kernel{},
            kGridSize,
            kBlockSize,
            0,
            problem_shape_tuple, // ck_tile::tuple<index_t, index_t, index_t, index_t>
            strides_tuple, // ck_tile::tuple<index_t, index_t, index_t, index_t> for input strides
            strides_tuple, // ck_tile::tuple<index_t, index_t, index_t, index_t> for output strides
            input_tensors,
            static_cast<YDataType*>(y_buf.GetDeviceBuffer())));

    std::cout << "Average time: " << ave_time << " ms" << std::endl;

    // Verify the output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());
        auto op = [](const auto& v0, const auto& v1) { return v0 + v1; };

        ck_tile::reference_binary_elementwise<XDataType, XDataType, YDataType, ComputeDataType>(
            x_host_a, x_host_b, y_host, op);

        pass = ck_tile::check_err(
            y_validation, y_host, "Elementwise Add Error: Incorrect results!", 0.01, 0.01);
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_elementwise_json_results(arg_parser.get_str("jsonfile"),
                                      arg_parser.get_str("prec"),
                                      kGridSize,
                                      kBlockSize,
                                      ave_time,
                                      0,
                                      0,
                                      "elementwise_add_4d");
    }

    return pass;
}

int main(int argc, char* argv[])
{
    bool result = true;
    ck_tile::ArgParser arg_parser;
    std::tie(result, arg_parser) = create_args(argc, argv);
    if(!result)
        return -1;

    try
    {
        const auto x_prec_variant = string_to_datatype(arg_parser.get_str("x_prec"));
        const auto y_prec_variant = string_to_datatype(arg_parser.get_str("y_prec"));
        return std::visit(
            [&](auto&& x_dt, auto&& y_dt) -> int {
                using XDataType = std::decay_t<decltype(x_dt)>;
                using YDataType = std::decay_t<decltype(y_dt)>;
                return run<XDataType, YDataType>(arg_parser);
            },
            x_prec_variant,
            y_prec_variant);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -3;
    }
}
