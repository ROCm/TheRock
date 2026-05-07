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
    arg_parser.insert("m", "1024", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("v", "1", "cpu validation or not")
        .insert("op", "1", "unary operation, 1: square, 2: convert")
        .insert("x_prec", "fp16", "input precision")
        .insert("y_prec", "fp16", "output precision")
        .insert("warmup", "10", "cold iter")
        .insert("repeat", "50", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "elementwise_unary.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename XElementwiseOperation, typename XDataType, typename YDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = N;
    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    assert(stride >= N);

    // 1. Initialize the input data on the host
    ck_tile::HostTensor<XDataType> x_host_a({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});

    std::vector<ck_tile::index_t> shape = {M, N};

    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);

    // 2. Create device memory buffers and copy input data from host to device
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());
    x_buf_a.ToDevice(x_host_a.data());

    // 3. Create the kernel

    // Dividing the problem into blocktile, warptile, and vector
    using BlockTile = ck_tile::sequence<2048>; // Size of the block tile (Entire problem is divided
                                               // into blocks of this size)
    using BlockWarps = ck_tile::sequence<8>; // How many concurrent warps are in a block (Each warp
                                             // will cover some part of blockTile)
    using WarpTile = ck_tile::sequence<64>;  // How many elements are covered by a warp

    using Shape   = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, XDataType>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        XDataType, // ComputeDataType is same as
                                                                   // XDataType in the unary case
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    // Compute flattened size
    ck_tile::index_t total_elements = 1;
    for(auto d : shape)
        total_elements *= d;

    const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;

    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()));
    auto input_size    = ck_tile::make_tuple(M, N);

    // Check if the kernel configuration is supported
    if(!Kernel::IsSupportedArgument(input_size))
    {
        throw std::runtime_error(
            "The kernel configuration is not supported for the given input size.");
    }

    // 4. Run the kernel
    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          input_size,
                                          ck_tile::make_tuple(N, 1), // Input Stride
                                          ck_tile::make_tuple(N, 1), // Output Stride
                                          input_tensors,
                                          static_cast<YDataType*>(y_buf.GetDeviceBuffer())));

    std::cout << "Average time: " << ave_time << " ms" << std::endl;

    // 5. Verify the output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());

        auto op = [](const XDataType& v0) -> YDataType {
            XElementwiseOperation element_op{};
            YDataType result;
            element_op(result, v0);
            return result;
        };

        ck_tile::reference_unary_elementwise<XDataType, YDataType, YDataType>(x_host_a, y_host, op);

        pass = ck_tile::check_err(
            y_validation, y_host, "Elementwise unary op: Incorrect results!", 0.01, 0.01);
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
                                      "elementwise_unary");
    }

    return pass;
}

template <typename XElementwiseOperation, typename XDataType, typename YDataType>
bool filter_then_run(const ck_tile::ArgParser& arg_parser)
{
    auto throw_unsupported = [&]() {
        const auto x_prec = arg_parser.get_str("x_prec");
        const auto op     = arg_parser.get_str("op");
        throw std::runtime_error("Unsupported! x_prec: " + x_prec + ", op: " + op);
    };
    bool pass = true;

    if constexpr(std::is_same_v<XElementwiseOperation, ck_tile::element_wise::UnarySquare> &&
                 (std::is_same_v<XDataType, ck_tile::bf16_t> ||
                  std::is_same_v<YDataType, ck_tile::bf16_t>))
    {
        throw_unsupported();
    }
    else if constexpr(std::is_same_v<XElementwiseOperation, ck_tile::element_wise::UnaryConvert> &&
                      (std::is_same_v<XDataType, ck_tile::bf16_t> ||
                       std::is_same_v<YDataType, ck_tile::bf16_t>))
    {
        throw_unsupported();
    }
    else
    {
        pass = run<XElementwiseOperation, XDataType, YDataType>(arg_parser);
    }

    return pass;
}

auto string_to_op(const std::string& op)
{
    using OpVariant =
        std::variant<ck_tile::element_wise::UnarySquare, ck_tile::element_wise::UnaryConvert>;

    if(op == "1")
        return OpVariant{ck_tile::element_wise::UnarySquare{}};
    else if(op == "2")
        return OpVariant{ck_tile::element_wise::UnaryConvert{}};
    else
    {
        throw std::runtime_error("Unsupported unary operation: " + op);
    }
};

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
        const auto op_variant     = string_to_op(arg_parser.get_str("op"));
        return std::visit(
            [&](auto&& op, auto&& x_dt, auto&& y_dt) -> int {
                using XElementwiseOperation = std::decay_t<decltype(op)>;
                using XDataType             = std::decay_t<decltype(x_dt)>;
                using YDataType             = std::decay_t<decltype(y_dt)>;
                return filter_then_run<XElementwiseOperation, XDataType, YDataType>(arg_parser);
            },
            op_variant,
            x_prec_variant,
            y_prec_variant);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -3;
    }
}
