// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include <cstring>
#include "copy_basic.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "128", "m dimension")
        .insert("n", "8", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision(fp16 or fp32)")
        .insert("warmup", "50", "cold iter")
        .insert("repeat", "100", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType = DataType;
    using YDataType = DataType;

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    // Create host tensors
    ck_tile::HostTensor<XDataType> x_host({m, n});     // input matrix
    ck_tile::HostTensor<YDataType> y_host_ref({m, n}); // reference output matrix
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}); // device output matrix

    // Initialize input data with increasing values
    ck_tile::half_t value = 1;
    for(int i = 0; i < m; i++)
    {
        value = 1;
        for(int j = 0; j < n; j++)
        {
            x_host(i, j) = value++;
        }
    }

    // Allocate device memory
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    // Define tile configuration
    using ThreadTile = ck_tile::sequence<1, 4>;  // per-thread tile size along M and N
    using WaveTile   = ck_tile::sequence<64, 4>; // per-wave tile size along M and N dimension
    using BlockWaves = ck_tile::sequence<4, 1>; // number of waves per block along M and N dimension
    using BlockTile  = ck_tile::sequence<512, 4>; // per-block tile size along M and N dimension

    // Calculate grid size
    ck_tile::index_t kGridSize =
        ck_tile::integer_divide_ceil(m, BlockTile::at(ck_tile::number<0>{}));
    std::cout << "grid size (number of blocks per grid) " << kGridSize << std::endl;

    // Define kernel types
    using Shape   = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, ThreadTile>;
    using Problem = ck_tile::TileCopyProblem<XDataType, Shape>;
    using Policy  = ck_tile::TileCopyPolicy<Problem>;
    using Kernel  = ck_tile::ElementWiseTileCopyKernel<Problem, Policy>; // operates on element by
                                                                         // element basis.

    // We also implement two variations of the copy kernel:
    // 1. TileCopyKernel: This is the basic copy kernel that operates on tile by tile basis.
    // 2. TileCopyKernel_LDS: This is the copy kernel that operates on tile by tile basis and uses
    // the LDS. using Kernel  = ck_tile::TileCopyKernel<Problem, Policy>; using Kernel =
    // ck_tile::TileCopyKernel_LDS<Problem, Policy>;

    auto blockSize = Kernel::BlockSize();

    // Print configuration information
    std::cout << "block size (number of threads per block) " << blockSize << std::endl;
    std::cout << "wave size (number of threads per wave) " << ck_tile::get_warp_size() << std::endl;
    std::cout << "block waves (number of waves per block) " << BlockWaves::at(ck_tile::number<0>{})
              << " " << BlockWaves::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "block tile (number of elements per block) " << BlockTile::at(ck_tile::number<0>{})
              << " " << BlockTile::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "wave tile (number of elements per wave) " << WaveTile::at(ck_tile::number<0>{})
              << " " << WaveTile::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "thread tile (number of elements per thread) "
              << ThreadTile::at(ck_tile::number<0>{}) << " " << ThreadTile::at(ck_tile::number<1>{})
              << std::endl;
    std::cout << "WaveRepetitionPerBlock_M =  " << Shape::WaveRepetitionPerBlock_M << " --> ("
              << Shape::Block_Tile_M << "/" << Shape::Waves_Per_Block_M << "*" << Shape::Wave_Tile_M
              << ")" << std::endl;
    std::cout << "WaveRepetitionPerBlock_N =  " << Shape::WaveRepetitionPerBlock_N << " --> ("
              << Shape::Block_Tile_N << "/" << Shape::Waves_Per_Block_N << "*" << Shape::Wave_Tile_N
              << ")" << std::endl;

    // Launch kernel
    float ave_time =
        launch_kernel(ck_tile::stream_config{nullptr, true, warmup, repeat, 1},
                      ck_tile::make_kernel<1>(Kernel{},
                                              kGridSize,
                                              blockSize,
                                              0,
                                              static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                              static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                              m,
                                              n));

    // Calculate and print performance metrics
    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;
    float gb_per_sec      = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // Copy results back to host
        y_buf.FromDevice(y_host_dev.mData.data());
        // Use exact equality (tolerance = 0) for copy operations since copy should be exact
        pass = ck_tile::check_err(y_host_dev, x_host, "Error: Copy operation failed!", 0.0, 0.0);
        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    // Print results for debugging
    // std::cout << "Input matrix (x_host):" << std::endl;
    // std::cout << x_host << std::endl;
    // std::cout << "Output matrix (y_host_dev):" << std::endl;
    // std::cout << y_host_dev << std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    if(arg_parser.get_str("prec") == "fp16")
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    else
        return run<float>(arg_parser) ? 0 : -2;
}
