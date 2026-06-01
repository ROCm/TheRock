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
        .insert("x_prec", "fp16", "input precision, fp16/bf16/fp32")
        .insert("y_prec", "fp16", "output precision, fp16/bf16/fp32")
        .insert("warmup", "10", "cold iter")
        .insert("repeat", "50", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "elementwise.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// XDataType: Data type of the input tensors.
// ComputeDataType: Data type used for intermediate computations (often float for precision).
// YDataType: Data type of the output tensor.
template <typename XDataType, typename YDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");

    // If stride is negative (default -1), set it to N, assuming a dense row-major layout.
    if(stride < 0)
        stride = N;
    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    if(stride < N)
    {
        throw std::runtime_error("stride must be >= N");
    }

    // XElementwiseOperation: The specific elementwise operation to perform (e.g., Add, Mul).
    using ComputeDataType =
        float; // Using float for intermediate calculations can improve numerical stability.
    using XElementwiseOperation = ck_tile::element_wise::Add;

    // 1. Initialize the input data on the host (CPU).
    // HostTensor is a utility to manage tensor data on the CPU.
    // The first argument is the shape (dimensions) of the tensor {M, N}.
    // The second argument is the strides {stride, 1} for row-major layout.
    // 'x_host_a' and 'x_host_b' are the two input tensors for the elementwise operation.
    ck_tile::HostTensor<XDataType> x_host_a({M, N}, {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_b({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});

    std::vector<ck_tile::index_t> shape = {M, N};

    // Fill the host tensors with random data.
    // FillUniformDistribution populates the tensor with values from a uniform distribution,
    // within an interval.
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_b);

    // 2. Create device memory buffers
    // DeviceMem allocates memory on the GPU.
    // The size is determined by the total number of elements and the size of DataType.
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());

    // Copy data from host input tensors to device buffers.
    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());

    // 3. Configure the kernel execution parameters.
    // Dividing the problem into blocktile, blockwarp and warptile
    // The blocktile is the size of the tile processed by a single work group (also called thread
    // block). The warptile is the size of the tile processed by a single wavefront (also called
    // warp). The vector is the size of the tile processed by a single work item (also called
    // thread). The problem is divided into blocks of size BlockTile. Each block is further divided
    // into wavefronts of size WarpTile. Each wavefront is composed of 64 work items (on AMD; 32
    // threads on NVIDIA). Each work item in a wavefront processes one vector's worth of elements.
    // Note that WarpTile/Vector should be 64 for CDNA (because there are 64 work items per
    // wavefront). Vector size is set to be 16 / sizeof(ComputeDataType), to maximize vectorization.
    using BlockTile = ck_tile::sequence<2048>; // How many elements are handled by a block tile (the
                                               // tensor is divided into blocks of this size)
    using BlockWarps = ck_tile::sequence<8>; // How many concurrent wavefronts are in a block (each
                                             // wavefront will cover some part of the block tile)

    // WarpTile: Defines the size of the data sub-tile processed by a single wavefront.
    // This should be consistent with BlockTile and BlockWarps.
    // If BlockTile is 2048 and BlockWarps is 8, then WarpTile could be 2048/8 = 256.
    // However, this example uses 64, meaning each wavefront processes 64 elements, and multiple
    // such wavefront operations might be needed to cover the BlockTile, or the BlockTile is
    // distributed differently.
    // The current configuration (BlockTile=2048, BlockWarps=8, WarpTile=64) implies that
    // each wavefront processes 64 elements, and 8 wavefronts process 8*64 = 512 elements
    // concurrently. Since 512 is not equal to 2048, it means that warptile(s) will need to iterate
    // over multiple times over different set of elements to cover the entire BlockTile.
    using WarpTile = ck_tile::sequence<64>;

    // 4. Create the kernel

    // ElementWiseShape bundles these tiling parameters.
    // It calculates derived properties like threads per wavefront, repeats, vectorization and total
    // block size.
    using Shape = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, XDataType>;

    // ElementWisePipelineProblem encapsulates all necessary information for the elementwise kernel:
    // - Data types (input, compute, output).
    // - Shape traits (tiling configuration).
    // - The specific elementwise operation (e.g., Add).
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    // ElementWiseKernel refers to the GPU kernel class
    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    // Compute flattened size
    ck_tile::index_t total_elements = 1;
    for(auto d : shape)
        total_elements *= d;

    // kBlockSize: The number of work items in a GPU workgroup (thread block).
    // This is often a multiple of the wavefront size, 64 on CDNA.
    // Here, it's explicitly set to 512. This should be consistent with Shape::kBlockSize.
    // Shape::kBlockSize would be BlockWarps * warpSize (e.g., 8 * 64 = 512).
    const ck_tile::index_t kBlockSize = Kernel::BlockSize();

    // kBlockPerCu: Hint for how many workgroups can be scheduled per Compute Unit (CU).
    // This can influence occupancy and performance.
    constexpr ck_tile::index_t kBlockPerCu = 1;

    // kGridSize: Calculates the total number of workgroups required to process all elements.
    // Each workgroup is responsible for 'elements_per_block' elements.
    // To ensure all elements are covered, especially when 'total_elements' is not perfectly
    // divisible by 'elements_per_block', using ceiling division.
    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;

    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                             static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()));

    auto input_size = ck_tile::make_tuple(M, N);

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
                                      "elementwise_add");
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
