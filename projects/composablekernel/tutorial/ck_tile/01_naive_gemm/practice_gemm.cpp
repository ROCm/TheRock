// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include "ck_tile/host.hpp"
#include "practice_gemm.hpp"
#include "reference_gemm.hpp"

int main(int argc, char* argv[])
{
    // TODO: GemmTypeConfig
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = float;
    using AccDataType = float;

    // Setup simple argument parser for M, N, K
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "512", "m dimension")
        .insert("n", "256", "n dimension")
        .insert("k", "64", "k dimension")
        .insert("v", "1", "verification: 0=off, 1=on");

    auto result = arg_parser.parse(argc, argv);
    if(!result)
        return -1;

    // Get problem dimensions from command line
    ck_tile::index_t M            = arg_parser.get_int("m");
    ck_tile::index_t N            = arg_parser.get_int("n");
    ck_tile::index_t K            = arg_parser.get_int("k");
    ck_tile::index_t verification = arg_parser.get_int("v");

    ck_tile::index_t stride_a = K;
    ck_tile::index_t stride_b = K;
    ck_tile::index_t stride_c = N;

    auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    auto b_lengths = std::array<ck_tile::index_t, 2>{N, K};
    auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};

    auto a_strides = std::array<ck_tile::index_t, 2>{stride_a, 1};
    auto b_strides = std::array<ck_tile::index_t, 2>{stride_b, 1};
    auto c_strides = std::array<ck_tile::index_t, 2>{stride_c, 1};

    // tensors on host (cpu)
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    ck_tile::HostTensor<CDataType> c_host(c_lengths, c_strides);

    // initialize tensors
    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_host);
    c_host.SetZero();

    // Print the tensors using the new print_first_n member function
    // std::cout << "Tensor A (first 10 elements): ";
    // a_host.print_first_n(10);
    // std::cout << std::endl;

    // std::cout << "Tensor B (first 10 elements): ";
    // b_host.print_first_n(10);
    // std::cout << std::endl;

    // std::cout << "Tensor C (first 10 elements): ";
    // c_host.print_first_n(10);
    // std::cout << std::endl;

    // Create device tensors of same size as host tensors and copy data
    ck_tile::DeviceMem a_device(a_host);
    ck_tile::DeviceMem b_device(b_host);
    ck_tile::DeviceMem c_device(c_host);

    // TODO: BlockTileConfig
    using BlockTile = ck_tile::sequence<256, 128, 32>;
    using WaveTile  = ck_tile::sequence<16, 16, 16>;

    std::cout << "Creating PracticeGemmShape, PracticeGemmProblem, PracticeGemmPolicy" << std::endl;
    using PracticeGemmShape = ck_tile::PracticeGemmShape<BlockTile, WaveTile>;
    std::cout << "PracticeGemmShape: " << PracticeGemmShape::GetName() << std::endl;
    using PracticeGemmHostProblem = ck_tile::
        PracticeGemmHostProblem<ADataType, BDataType, CDataType, AccDataType, PracticeGemmShape>;
    using PracticeGemmHostPolicy = ck_tile::PracticeGemmHostPolicy;

    ck_tile::index_t kGridSize = ck_tile::integer_divide_ceil(M, PracticeGemmShape::BlockTile_M) *
                                 ck_tile::integer_divide_ceil(N, PracticeGemmShape::BlockTile_N);

    std::cout << "Total number of thread blocks: " << kGridSize << std::endl;
    constexpr ck_tile::index_t kBlockPerCU = 1; // 1 block per CU

    // Block size is now derived from the shape configuration
    constexpr ck_tile::index_t kBlockSize = PracticeGemmShape::kBlockSize;
    std::cout << "Number of threads per block: " << kBlockSize << std::endl;
    std::cout << "Number of blocks per compute unit: " << kBlockPerCU << std::endl;

    using gemm_kernel =
        ck_tile::PracticeGemmKernel<PracticeGemmHostProblem, PracticeGemmHostPolicy>;

    float ave_time = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, 0, 1},
        ck_tile::make_kernel<kBlockPerCU>(gemm_kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<ADataType*>(a_device.GetDeviceBuffer()),
                                          static_cast<BDataType*>(b_device.GetDeviceBuffer()),
                                          static_cast<CDataType*>(c_device.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          stride_a,
                                          stride_b,
                                          stride_c));

    auto pass = true;

    if(verification)
    {
        // reference gemm
        ck_tile::HostTensor<CDataType> c_host_ref(c_lengths, c_strides);
        reference_basic_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_host, b_host, c_host_ref);
        ck_tile::HostTensor<CDataType> c_host_dev(c_lengths, c_strides);
        c_device.FromDevice(c_host_dev.mData.data());
        pass &= ck_tile::check_err(c_host_dev, c_host_ref, "Error: Incorrect results!", 1e-3, 1e-3);
        std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !pass;
}
