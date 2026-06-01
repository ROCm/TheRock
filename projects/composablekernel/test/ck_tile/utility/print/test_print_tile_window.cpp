// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_print_common.hpp"
#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {

template <typename DataType>
__global__ void KernelPrintTileWindow(DataType* data, int M, int N)
{
    using namespace ck_tile;

    auto tv = make_naive_tensor_view<address_space_enum::global>(
        data, make_tuple(M, N), make_tuple(N, 1));

    constexpr auto window_lengths = make_tuple(number<2>{}, number<3>{});

    // Create tile window with static lengths 2x3 with origin (0,0)
    auto tw = make_tile_window(tv, window_lengths, make_multi_index(0, 0));

    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        tw.template print_tile_window_range<DataType>(0, 2, 0, 3, "TW");
    }
}

class PrintTileWindowTest : public PrintTest
{
    protected:
    void SetUp() override
    {
        // Initialize HIP
        hipError_t err = hipSetDevice(0);
        if(err != hipSuccess)
        {
            GTEST_SKIP() << "No GPU available for tile window test";
        }
    }

    void TearDown() override {}

    template <typename DataType>
    std::string CaptureTileWindowPrintOutput(const std::vector<DataType>& host_data, int M, int N)
    {
        // Allocate device memory
        DataType* device_data = nullptr;
        size_t size_bytes     = host_data.size() * sizeof(DataType);
        hipError_t err        = hipMalloc(&device_data, size_bytes);
        if(err != hipSuccess)
        {
            ADD_FAILURE() << "Failed to allocate device memory: " << hipGetErrorString(err);
            return "";
        }

        // Copy data to device
        err = hipMemcpy(device_data, host_data.data(), size_bytes, hipMemcpyHostToDevice);
        if(err != hipSuccess)
        {
            ADD_FAILURE() << "Failed to copy data to device: " << hipGetErrorString(err);
            (void)hipFree(device_data);
            return "";
        }

        // Capture stdout
        testing::internal::CaptureStdout();

        // Launch kernel
        dim3 grid_dim(1, 1, 1);
        dim3 block_dim(1, 1, 1);
        hipLaunchKernelGGL(
            KernelPrintTileWindow<DataType>, grid_dim, block_dim, 0, 0, device_data, M, N);

        // Synchronize to ensure print output is captured
        err = hipDeviceSynchronize();
        if(err != hipSuccess)
        {
            ADD_FAILURE() << "Failed to synchronize device: " << hipGetErrorString(err);
            testing::internal::GetCapturedStdout(); // Consume captured output
            (void)hipFree(device_data);
            return "";
        }

        // Get captured output
        std::string output = testing::internal::GetCapturedStdout();

        // Cleanup
        err = hipFree(device_data);
        if(err != hipSuccess)
        {
            ADD_FAILURE() << "Failed to free device memory: " << hipGetErrorString(err);
        }

        return output;
    }
};

TEST_F(PrintTileWindowTest, PrintTileWindow2x3)
{
    // Create a 4x4 tensor with values 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    const int M = 4, N = 4;
    std::vector<float> host_data(M * N);
    for(int i = 0; i < M * N; ++i)
    {
        host_data[i] = static_cast<float>(i);
    }

    std::string output = CaptureTileWindowPrintOutput(host_data, M, N);

    // Expected output for a 2x3 window starting at (0,0) from a 4x4 tensor
    // Values should be: [0,1,2] in first row, [4,5,6] in second row
    std::string expected = "TW Window Range [0:1, 0:2] (origin: 0, 0):\n"
                           "  TW[0,0] = 0.000000  TW[0,1] = 1.000000  TW[0,2] = 2.000000\n"
                           "  TW[1,0] = 4.000000  TW[1,1] = 5.000000  TW[1,2] = 6.000000\n"
                           "\n";

    EXPECT_EQ(output, expected);
}

TEST_F(PrintTileWindowTest, PrintTileWindowScaledValues)
{
    // Test with scaled values (multiples of 10)
    const int M = 3, N = 3;
    std::vector<float> host_data(M * N);
    for(int i = 0; i < M * N; ++i)
    {
        host_data[i] = static_cast<float>(i * 10); // 0, 10, 20, 30, 40, 50, 60, 70, 80
    }

    std::string output = CaptureTileWindowPrintOutput(host_data, M, N);

    // For a 2x3 window from this 3x3 tensor, we should get:
    // [0, 10, 20] in first row, [30, 40, 50] in second row
    std::string expected = "TW Window Range [0:1, 0:2] (origin: 0, 0):\n"
                           "  TW[0,0] = 0.000000  TW[0,1] = 10.000000  TW[0,2] = 20.000000\n"
                           "  TW[1,0] = 30.000000  TW[1,1] = 40.000000  TW[1,2] = 50.000000\n"
                           "\n";

    EXPECT_EQ(output, expected);
}

TEST_F(PrintTileWindowTest, PrintTileWindowFp8)
{
    // Test with fp8_t data type
    const int M = 4, N = 4;
    std::vector<ck_tile::fp8_t> host_data(M * N);
    for(int i = 0; i < M * N; ++i)
    {
        host_data[i] = ck_tile::fp8_t(static_cast<float>(i));
    }

    std::string output = CaptureTileWindowPrintOutput<ck_tile::fp8_t>(host_data, M, N);

    // Expected output for a 2x3 window starting at (0,0) from a 4x4 tensor
    // Values should be: [0, 1, 2] in first row, [4, 5, 6] in second row
    // we type convert on host to match the function implementation
    float val_00 = type_convert<float>(ck_tile::fp8_t(0.0f));
    float val_01 = type_convert<float>(ck_tile::fp8_t(1.0f));
    float val_02 = type_convert<float>(ck_tile::fp8_t(2.0f));
    float val_10 = type_convert<float>(ck_tile::fp8_t(4.0f));
    float val_11 = type_convert<float>(ck_tile::fp8_t(5.0f));
    float val_12 = type_convert<float>(ck_tile::fp8_t(6.0f));

    char expected_buf[512];
    snprintf(expected_buf,
             sizeof(expected_buf),
             "TW Window Range [0:1, 0:2] (origin: 0, 0):\n"
             "  TW[0,0] = %f  TW[0,1] = %f  TW[0,2] = %f\n"
             "  TW[1,0] = %f  TW[1,1] = %f  TW[1,2] = %f\n"
             "\n",
             val_00,
             val_01,
             val_02,
             val_10,
             val_11,
             val_12);
    std::string expected(expected_buf);

    EXPECT_EQ(output, expected);
}

TEST_F(PrintTileWindowTest, PrintTileWindowBf8)
{
    // Test with bf8_t data type
    const int M = 3, N = 3;
    std::vector<ck_tile::bf8_t> host_data(M * N);
    for(int i = 0; i < M * N; ++i)
    {
        host_data[i] = ck_tile::bf8_t(static_cast<float>(i * 10));
    }

    std::string output = CaptureTileWindowPrintOutput<ck_tile::bf8_t>(host_data, M, N);

    // Expected output for a 2x3 window starting at (0,0) from a 3x3 tensor
    // Values should be: [0, 10, 20] in first row, [30, 40, 50] in second row
    // we type convert on host to match the function implementation
    float val_00 = type_convert<float>(ck_tile::bf8_t(0.0f));
    float val_01 = type_convert<float>(ck_tile::bf8_t(10.0f));
    float val_02 = type_convert<float>(ck_tile::bf8_t(20.0f));
    float val_10 = type_convert<float>(ck_tile::bf8_t(30.0f));
    float val_11 = type_convert<float>(ck_tile::bf8_t(40.0f));
    float val_12 = type_convert<float>(ck_tile::bf8_t(50.0f));

    char expected_buf[512];
    snprintf(expected_buf,
             sizeof(expected_buf),
             "TW Window Range [0:1, 0:2] (origin: 0, 0):\n"
             "  TW[0,0] = %f  TW[0,1] = %f  TW[0,2] = %f\n"
             "  TW[1,0] = %f  TW[1,1] = %f  TW[1,2] = %f\n"
             "\n",
             val_00,
             val_01,
             val_02,
             val_10,
             val_11,
             val_12);
    std::string expected(expected_buf);

    EXPECT_EQ(output, expected);
}

} // namespace ck_tile
