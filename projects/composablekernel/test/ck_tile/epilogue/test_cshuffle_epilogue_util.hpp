// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>

namespace ck_tile {

enum class ScaleType
{
    None,
    RowCol,
    Tensor
};

// Simple test kernel to invoke the CShuffleEpilogue
template <typename Problem, index_t M, index_t N, ScaleType Scale>
__global__ void test_cshuffle_epilogue_kernel(typename Problem::ODataType* __restrict__ output_data,
                                              float* m_scale,
                                              float* n_scale)
{
    using Epilogue = CShuffleEpilogue<Problem>;

    static_assert(Problem::kMPerBlock <= M && Problem::kNPerBlock <= N,
                  "Block size must fit in tensor dimensions");

    // Allocate shared memory for epilogue
    __shared__ char smem[Epilogue::GetSmemSize()];

    // Create accumulator tile
    constexpr auto lds_distribution_encode =
        make_static_tile_distribution(Epilogue::MakeLdsDistributionEncode());
    auto acc_tile =
        make_static_distributed_tensor<typename Epilogue::AccDataType>(lds_distribution_encode);

    // Fill acc_tile with a simple pattern
    auto& acc_buffer = acc_tile.get_thread_buffer();
    acc_buffer[0]    = 2.0F;

    // Create output tensor view
    auto output_tensor_view =
        make_naive_tensor_view<address_space_enum::global>(output_data,
                                                           make_tuple(M, N),
                                                           make_tuple(N, 1),
                                                           number<Epilogue::GetVectorSizeC()>{},
                                                           number<1>{});

    // Create output tile window
    auto output_tile_window =
        make_tile_window(output_tensor_view,
                         make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
                         {0, 0});

    // Create empty D tensors tuple (we're ignoring ds_dram_windows for this test)
    auto empty_ds = make_tuple();

    // Call the epilogue
    if constexpr(Scale == ScaleType::RowCol)
    {
        const auto m_scale_window = make_tile_window(
            make_naive_tensor_view<address_space_enum::global>(
                m_scale, make_tuple(M, N), make_tuple(1, 0), number<1>{}, number<1>{}),
            make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
            {0, 0});
        const auto n_scale_window = make_tile_window(
            make_naive_tensor_view<address_space_enum::global>(
                n_scale, make_tuple(M, N), make_tuple(0, 1), number<1>{}, number<1>{}),
            make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
            {0, 0});
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem, m_scale_window, n_scale_window);
    }
    else if constexpr(Scale == ScaleType::Tensor)
    {
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem, *m_scale, *n_scale);
    }
    else
    {
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem);
    }
}

// Test configuration helper
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename ODataType,
          index_t kM,
          index_t kN,
          index_t MWave,
          index_t NWave,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t KPerXdl>
using SimpleCShuffleEpilogueProblem =
    CShuffleEpilogueProblem<ADataType,
                            BDataType,
                            ck_tile::tuple<>, // Empty Ds datatype tuple
                            AccDataType,
                            ODataType,
                            ck_tile::tuple<>,                   // Empty Ds layout
                            tensor_layout::gemm::RowMajor,      // ELayout
                            ck_tile::element_wise::PassThrough, // CDElementwise
                            kM,
                            kN,
                            MWave,
                            NWave,
                            MPerXdl,
                            NPerXdl,
                            KPerXdl,
                            false // isCTransposed
                            >;

template <typename Problem, index_t M, index_t N>
auto run_cshuffle_epilogue_test(ScaleType scale = ScaleType::None)
{
    using ODataType = typename Problem::ODataType;

    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    index_t kBlockSize = ck_tile::is_wave32() ? Problem::kBlockSize / 2 : Problem::kBlockSize;

    std::cout << "Running CShuffleEpilogue test with M=" << M << ", N=" << N
              << ", MPerBlock=" << kMPerBlock << ", NPerBlock=" << kNPerBlock
              << ", BlockSize=" << kBlockSize << std::endl;

    // Allocate host memory
    const size_t output_size = M * N;

    std::vector<ODataType> host_output(output_size, static_cast<ODataType>(0));

    // Allocate device memory
    ODataType* device_output;

    HIP_CHECK_ERROR(hipMalloc(&device_output, output_size * sizeof(ODataType)));

    HIP_CHECK_ERROR(hipMemcpy(
        device_output, host_output.data(), output_size * sizeof(ODataType), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(kBlockSize, 1, 1);

    if(scale == ScaleType::RowCol)
    {
        float* m_scale;
        float* n_scale;
        std::vector<float> h_m_scale(M, 1.0F);
        std::vector<float> h_n_scale(N, 1.0F);
        h_n_scale[1] = 2.0F; // multiply one col only with 2
        HIP_CHECK_ERROR(hipMalloc(&m_scale, M * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&n_scale, N * sizeof(float)));
        HIP_CHECK_ERROR(
            hipMemcpy(m_scale, h_m_scale.data(), M * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(
            hipMemcpy(n_scale, h_n_scale.data(), N * sizeof(float), hipMemcpyHostToDevice));
        test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::RowCol>
            <<<gridSize, blockSize>>>(device_output, m_scale, n_scale);
    }
    else if(scale == ScaleType::Tensor)
    {
        float* m_scale;
        float* n_scale;
        std::vector<float> h_m_scale(1, 2.0F);
        std::vector<float> h_n_scale(1, 1.0F);
        HIP_CHECK_ERROR(hipMalloc(&m_scale, sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&n_scale, sizeof(float)));
        HIP_CHECK_ERROR(hipMemcpy(m_scale, h_m_scale.data(), sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(hipMemcpy(n_scale, h_n_scale.data(), sizeof(float), hipMemcpyHostToDevice));
        test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::Tensor>
            <<<gridSize, blockSize>>>(device_output, m_scale, n_scale);
    }
    else
    {
        test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::None>
            <<<gridSize, blockSize>>>(device_output, nullptr, nullptr);
    }

    // Check for kernel launch errors
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Copy results back
    HIP_CHECK_ERROR(hipMemcpy(
        host_output.data(), device_output, output_size * sizeof(ODataType), hipMemcpyDeviceToHost));

    // Cleanup
    HIP_CHECK_ERROR(hipFree(device_output));

    return host_output;
}

} // namespace ck_tile
