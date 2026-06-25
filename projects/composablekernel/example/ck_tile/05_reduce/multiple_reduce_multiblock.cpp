// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include <cstring>

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("n", "32", "n dimension")
        .insert("h", "19", "h dimension")
        .insert("w", "7", "w dimension")
        .insert("c", "512", "c dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "multi_reduce_multiblock.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = float;

    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t H = arg_parser.get_int("h");
    ck_tile::index_t W = arg_parser.get_int("w");
    ck_tile::index_t C = arg_parser.get_int("c");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    // Validate input dimensions
    const ck_tile::index_t kept_dim_len_prod   = N * C;
    const ck_tile::index_t reduce_total_length = H * W;

    if(kept_dim_len_prod == 0)
    {
        std::cerr << "Warning: Product of kept dimensions is zero (N=" << N << ", C=" << C
                  << ", product=" << kept_dim_len_prod << ")." << std::endl;
        std::cerr << "This will result in an empty output tensor." << std::endl;
        return false;
    }

    if(reduce_total_length == 0)
    {
        std::cerr << "Warning: Product of reduce dimensions is zero (H=" << H << ", W=" << W
                  << ", product=" << reduce_total_length << ")." << std::endl;
        std::cerr << "This will result in an empty reduction with no data to process." << std::endl;
        std::cerr << "The kernel will exit early without performing any computation." << std::endl;
        return false;
    }

    std::vector<ck_tile::index_t> problem_shape = {N, H, W, C};
    std::vector<ck_tile::index_t> strides(4);
    strides[0] = H * W * C;
    strides[1] = W * C;
    strides[2] = C;
    strides[3] = 1;

    // Define reduction specification:
    constexpr auto kept_dim    = ck_tile::sequence<0, 3>{}; // Which dimension to keep
    constexpr auto reduce_dims = ck_tile::sequence<1, 2>{}; // Which dimensions to reduce

    ck_tile::HostTensor<XDataType> x_host(problem_shape, strides);
    ck_tile::HostTensor<YDataType> y_host_add_ref({N, C}, {C, 1});
    ck_tile::HostTensor<YDataType> y_host_max_ref({N, C}, {C, 1});
    auto y_host_ref_tuple = ck_tile::make_tuple(y_host_add_ref, y_host_max_ref);

    ck_tile::HostTensor<YDataType> y_host_add_dev({N, C}, {C, 1});
    ck_tile::HostTensor<YDataType> y_host_max_dev({N, C}, {C, 1});
    auto y_host_dev_tuple = ck_tile::make_tuple(y_host_add_dev, y_host_max_dev);

    const auto number_operations = y_host_dev_tuple.size();

    std::vector<YDataType> h(number_operations * N * C);

    auto y_buf_size = number_operations *
                      y_host_dev_tuple.at(ck_tile::number<0>{}).get_element_space_size_in_bytes();
    ck_tile::DeviceMem y_buf(y_buf_size);

    const auto output_tensor_offset = N * C;

    // Operations: one doing a sum reduction, the other computing the mean square
    // In the case of mean square:
    // 1. The element wise operation squares each element before reduction
    // 2. The reduction operation sum the squared element
    // 3. The accumulator element wise operation divides the result by the total number of reduced
    // elements (intra block operation)
    // 4. The partial result is updated across blocks using inter block reduction, a sum.
    auto reduce_ops =
        ck_tile::make_tuple(ck_tile::ReduceOp::Add{}, ck_tile::ReduceOp::Add{}); // reductions
    auto elementwise_ops = ck_tile::make_tuple(ck_tile::element_wise::PassThrough{},
                                               ck_tile::element_wise::UnarySquare{}); // Elementwise
                                                                                      // ops
    auto accumulator_elementwise_ops = ck_tile::make_tuple(
        ck_tile::element_wise::PassThrough{},
        ck_tile::element_wise::UnaryDivide{
            reduce_total_length}); // Accumulator Elementwise ops on reduction, intra block
    auto inter_block_reduce_ops = ck_tile::make_tuple(
        ck_tile::ReduceOp::Add{}, ck_tile::ReduceOp::Add{}); // Inter block reduction

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<128, 128>;
    using WarpTile   = ck_tile::sequence<32, 128>;
    using ThreadTile = ck_tile::sequence<8, 8>;

    constexpr ck_tile::index_t kBlockPerCu = 1;

    using Shape   = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;
    using Problem = ck_tile::Reduce2dProblem<XDataType,
                                             ComputeDataType,
                                             YDataType,
                                             Shape,
                                             decltype(reduce_ops),
                                             decltype(kept_dim),
                                             decltype(reduce_dims),
                                             4>;

    using Kernel = ck_tile::MultiReduceMultiblock<Problem>;

    // Determine block group size for multi-block reduction
    // block_group_size records how many blocks participate to a reduction (input data dependent)
    //  , for efficiency reasons this size if limited to a maximum of 128. If this is not sufficient
    //  to process the whole reduction, each thread will to process multiple thread tile
    //  a num_block_tile_iterations times
    auto [num_block_tile_iterations, block_group_size] =
        typename Kernel::TilePartitioner{reduce_total_length}.GetBlockGroupParams();

    const ck_tile::index_t kBlockSize = Kernel::BlockSize();
    ck_tile::index_t kGridSize =
        ((kept_dim_len_prod + Shape::Block_M - 1) / Shape::Block_M) * block_group_size;

    std::cout << "Block group size: " << block_group_size
              << ", Num block tile iterations: " << num_block_tile_iterations
              << ", Reduce total length: " << reduce_total_length << std::endl;
    std::cout << "grid size " << kGridSize << ", block size " << kBlockSize << std::endl;

    // Create input tensor shape and strides
    auto input_shape =
        ck_tile::make_tuple(problem_shape[0], problem_shape[1], problem_shape[2], problem_shape[3]);
    auto input_strides = ck_tile::make_tuple(strides[0], strides[1], strides[2], strides[3]);

    if(!Kernel::IsSupportedArgument(
           C, input_strides)) // output tensor's continuous dimension and input strides
    {
        throw std::runtime_error("Wrong! Arguments not supported!\n");
    }

    // Init the output data with identity values respective to each reduce op
    ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
        constexpr auto op                 = reduce_ops.at(i);
        const auto identity_val           = op.template GetIdentityValue<YDataType>();
        const auto output_number_elements = N * C;
        std::fill(h.begin() + i * output_number_elements,
                  h.begin() + (i + 1) * output_number_elements,
                  identity_val);
    });

    auto clear_output_buffer = [&]() { y_buf.ToDevice(h.data()); };

    float ave_time = launch_kernel_time_mask(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        clear_output_buffer,
        ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                          static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                          input_shape,
                                          input_strides,
                                          kept_dim,
                                          reduce_dims,
                                          output_tensor_offset,
                                          elementwise_ops,
                                          accumulator_elementwise_ops,
                                          inter_block_reduce_ops)

    );

    std::size_t num_btype = sizeof(XDataType) * N * C * H * W + sizeof(YDataType) * N * C;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_multiple_reduce_multiblock<XDataType, ComputeDataType, YDataType>(
            x_host,
            y_host_ref_tuple,
            reduce_ops,
            kept_dim,
            reduce_dims,
            elementwise_ops,
            accumulator_elementwise_ops,
            inter_block_reduce_ops,
            block_group_size);
        std::cout << "Read " << y_buf_size / 10 << " Bytes from the device" << std::endl;

        // Transfer data from device and check error for each operation
        y_buf.FromDevice(h.data());
        ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
            std::memcpy(y_host_dev_tuple.get(ck_tile::number<i>{}).data(),
                        h.data() + i * output_tensor_offset,
                        output_tensor_offset * sizeof(YDataType));
            std::cout << "Checking operation " << i << ": " << std::endl;

            bool pass_op = ck_tile::check_err(y_host_dev_tuple.get(ck_tile::number<i>{}),
                                              y_host_ref_tuple.get(ck_tile::number<i>{}));

            if(pass_op)
            {
                std::cout << "✅ valid results for this operation" << std::endl;
            }
            pass &= pass_op;
        });

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
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
}
