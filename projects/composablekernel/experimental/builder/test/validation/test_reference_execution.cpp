// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/types.hpp"
#include "impl/conv_algorithm_types.hpp"
#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_data_gpu.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"
#include "utils/ckb_conv_test_configs.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test; // For ConvAlgorithm_Reference
using namespace ck_tile::builder::test_utils;

TEST(ReferenceExecution, Forward_2D_FP16)
{
    // Note: When you don't specify .operation, it defaults to PassThrough
    // Reference implementation only supports PassThrough elementwise operations
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    // Simple dimensions
    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    // Allocate minimal device memory (just to test API)
    const size_t in_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_dev(out_size);

    in_dev.SetZero();
    wei_dev.SetZero();
    out_dev.SetZero();

    // Prepare parameters for Run()
    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;
    ref_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                   reinterpret_cast<ck::half_t*>(out_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    // If we get here, Run() worked!
    std::cout << "✓ Reference Forward kernel executed!" << std::endl;
    EXPECT_TRUE(true);
}

TEST(ReferenceExecution, BackwardData_2D_FP16)
{
    // Note: When you don't specify .operation, it defaults to PassThrough
    // Reference implementation only supports PassThrough elementwise operations
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_DATA,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_grad_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size      = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_grad_dev(in_grad_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_grad_dev.SetZero();
    wei_dev.SetZero();
    out_grad_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;
    ref_kernel.Run(reinterpret_cast<ck::half_t*>(in_grad_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    std::cout << "✓ Reference Backward Data kernel executed!" << std::endl;
    EXPECT_TRUE(true);
}

TEST(ReferenceExecution, BackwardWeight_2D_FP16)
{
    // Note: When you don't specify .operation, it defaults to PassThrough
    // Reference implementation only supports PassThrough elementwise operations
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_WEIGHT,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_size       = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_grad_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_grad_dev(wei_grad_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_dev.SetZero();
    wei_grad_dev.SetZero();
    out_grad_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;
    ref_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                   reinterpret_cast<ck::half_t*>(wei_grad_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    std::cout << "✓ Reference Backward Weight kernel executed!" << std::endl;
    EXPECT_TRUE(true);
}

// Test the old CK interface: MakeArgumentPointer + MakeInvokerPointer
TEST(ReferenceExecution, BackwardData_2D_FP16_InvokerInterface)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_DATA,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_grad_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size      = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_grad_dev(in_grad_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_grad_dev.SetZero();
    wei_dev.SetZero();
    out_grad_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;

    // TEST: Use the old CK interface (MakeArgumentPointer + MakeInvokerPointer)
    auto argument_ptr = ref_kernel.MakeArgumentPointer(
        reinterpret_cast<ck::half_t*>(in_grad_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    auto invoker_ptr = ref_kernel.MakeInvokerPointer();

    // Run using invoker
    float time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    std::cout << "✓ Reference Backward Data kernel executed via Invoker interface!" << std::endl;
    std::cout << "  (time = " << time << " ms)" << std::endl;
    EXPECT_TRUE(true);
}

// Test the old CK interface for Forward convolution
TEST(ReferenceExecution, Forward_2D_FP16_InvokerInterface)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_dev(out_size);

    in_dev.SetZero();
    wei_dev.SetZero();
    out_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;

    // TEST: Use the old CK interface (MakeArgumentPointer + MakeInvokerPointer)
    auto argument_ptr = ref_kernel.MakeArgumentPointer(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(out_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    auto invoker_ptr = ref_kernel.MakeInvokerPointer();

    // Run using invoker
    float time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    std::cout << "✓ Reference Forward kernel executed via Invoker interface!" << std::endl;
    std::cout << "  (time = " << time << " ms)" << std::endl;
    EXPECT_TRUE(true);
}

// Test the old CK interface for Backward Weight convolution
TEST(ReferenceExecution, BackwardWeight_2D_FP16_InvokerInterface)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_WEIGHT,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_size       = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_grad_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_grad_dev(wei_grad_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_dev.SetZero();
    wei_grad_dev.SetZero();
    out_grad_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;

    // TEST: Use the old CK interface (MakeArgumentPointer + MakeInvokerPointer)
    auto argument_ptr = ref_kernel.MakeArgumentPointer(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(wei_grad_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    auto invoker_ptr = ref_kernel.MakeInvokerPointer();

    // Run using invoker
    float time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    std::cout << "✓ Reference Backward Weight kernel executed via Invoker interface!" << std::endl;
    std::cout << "  (time = " << time << " ms)" << std::endl;
    EXPECT_TRUE(true);
}

// Test Builder Reference vs Direct GPU Reference with RANDOM INPUT
TEST(ReferenceExecution, Forward_2D_FP16_Builder_vs_DirectGPUReference_Random)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 16, K = 16, H = 14, W = 14; // Small for fast testing

    const size_t in_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_size = G * N * K * H * W * sizeof(ck::half_t);

    // Create host buffers with random data
    const size_t in_elements  = G * N * C * H * W;
    const size_t wei_elements = G * K * C * 3 * 3;
    const size_t out_elements = G * N * K * H * W;

    std::vector<ck::half_t> in_host(in_elements);
    std::vector<ck::half_t> wei_host(wei_elements);

    // Fill with random values
    std::srand(12345); // Fixed seed for reproducibility
    for(size_t i = 0; i < in_elements; i++)
    {
        in_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(size_t i = 0; i < wei_elements; i++)
    {
        wei_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    // Allocate GPU memory
    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_builder_dev(out_size);
    ck::DeviceMem out_naive_dev(out_size);

    // Transfer random data to GPU
    in_dev.ToDevice(in_host.data());
    wei_dev.ToDevice(wei_host.data());
    out_builder_dev.SetZero();
    out_naive_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel builder_kernel;

    // Run 1: Builder Reference Factory
    builder_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                       reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                       reinterpret_cast<ck::half_t*>(out_builder_dev.GetDeviceBuffer()),
                       G,
                       N,
                       K,
                       C,
                       input_spatial,
                       filter_spatial,
                       output_spatial,
                       strides,
                       dilations,
                       left_pads);

    // Run 2: Direct GPU Reference (same kernel the Builder calls internally!)
    ck_tile::naive_grouped_conv_fwd<2, ck::half_t, ck::half_t, ck::half_t>(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(out_naive_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    // Copy results back
    std::vector<ck::half_t> out_builder_result(out_elements);
    std::vector<ck::half_t> out_naive_result(out_elements);
    out_builder_dev.FromDevice(out_builder_result.data());
    out_naive_dev.FromDevice(out_naive_result.data());

    // Compare - should be IDENTICAL (both call same kernel)
    bool pass = ck::utils::check_err(out_builder_result,
                                     out_naive_result,
                                     "Error: Builder Reference != Direct GPU Reference",
                                     1e-6,
                                     1e-6); // Very tight tolerance!

    std::cout << "✓ Builder Reference vs Direct GPU Reference (RANDOM INPUT)!" << std::endl;
    std::cout << "  Result: " << (pass ? "IDENTICAL ✓" : "MISMATCH ✗") << std::endl;
    std::cout << "  This validates Builder Reference Factory is correct!" << std::endl;

    EXPECT_TRUE(pass);
}

// Test Builder Reference vs Direct GPU Reference with RANDOM INPUT - Backward Data
TEST(ReferenceExecution, BackwardData_2D_FP16_Builder_vs_DirectGPUReference_Random)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_DATA,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 16, K = 16, H = 14, W = 14;

    const size_t in_grad_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size      = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    const size_t in_grad_elements  = G * N * C * H * W;
    const size_t wei_elements      = G * K * C * 3 * 3;
    const size_t out_grad_elements = G * N * K * H * W;

    std::vector<ck::half_t> wei_host(wei_elements);
    std::vector<ck::half_t> out_grad_host(out_grad_elements);

    // Fill with random values
    std::srand(12346);
    for(size_t i = 0; i < wei_elements; i++)
    {
        wei_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(size_t i = 0; i < out_grad_elements; i++)
    {
        out_grad_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    ck::DeviceMem in_grad_builder_dev(in_grad_size);
    ck::DeviceMem in_grad_naive_dev(in_grad_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    wei_dev.ToDevice(wei_host.data());
    out_grad_dev.ToDevice(out_grad_host.data());
    in_grad_builder_dev.SetZero();
    in_grad_naive_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel builder_kernel;

    // Run 1: Builder Reference Factory
    builder_kernel.Run(reinterpret_cast<ck::half_t*>(in_grad_builder_dev.GetDeviceBuffer()),
                       reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                       reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
                       G,
                       N,
                       K,
                       C,
                       input_spatial,
                       filter_spatial,
                       output_spatial,
                       strides,
                       dilations,
                       left_pads);

    // Run 2: Direct GPU Reference
    ck_tile::naive_grouped_conv_bwd_data<2, ck::half_t, ck::half_t, ck::half_t>(
        reinterpret_cast<ck::half_t*>(in_grad_naive_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    // Compare
    std::vector<ck::half_t> in_grad_builder_result(in_grad_elements);
    std::vector<ck::half_t> in_grad_naive_result(in_grad_elements);
    in_grad_builder_dev.FromDevice(in_grad_builder_result.data());
    in_grad_naive_dev.FromDevice(in_grad_naive_result.data());

    bool pass = ck::utils::check_err(in_grad_builder_result,
                                     in_grad_naive_result,
                                     "Error: Builder Backward Data != Direct GPU Reference",
                                     1e-6,
                                     1e-6);

    std::cout << "✓ Builder Reference vs Direct GPU Reference (RANDOM INPUT - Backward Data)!"
              << std::endl;
    std::cout << "  Result: " << (pass ? "IDENTICAL ✓" : "MISMATCH ✗") << std::endl;
    EXPECT_TRUE(pass);
}

// Test Builder Reference vs Direct GPU Reference with RANDOM INPUT - Backward Weight
TEST(ReferenceExecution, BackwardWeight_2D_FP16_Builder_vs_DirectGPUReference_Random)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_WEIGHT,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 16, K = 16, H = 14, W = 14;

    const size_t in_size       = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_grad_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    const size_t in_elements       = G * N * C * H * W;
    const size_t wei_grad_elements = G * K * C * 3 * 3;
    const size_t out_grad_elements = G * N * K * H * W;

    std::vector<ck::half_t> in_host(in_elements);
    std::vector<ck::half_t> out_grad_host(out_grad_elements);

    // Fill with random values
    std::srand(12347);
    for(size_t i = 0; i < in_elements; i++)
    {
        in_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(size_t i = 0; i < out_grad_elements; i++)
    {
        out_grad_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_grad_builder_dev(wei_grad_size);
    ck::DeviceMem wei_grad_naive_dev(wei_grad_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_dev.ToDevice(in_host.data());
    out_grad_dev.ToDevice(out_grad_host.data());
    wei_grad_builder_dev.SetZero();
    wei_grad_naive_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel builder_kernel;

    // Run 1: Builder Reference Factory
    builder_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                       reinterpret_cast<ck::half_t*>(wei_grad_builder_dev.GetDeviceBuffer()),
                       reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
                       G,
                       N,
                       K,
                       C,
                       input_spatial,
                       filter_spatial,
                       output_spatial,
                       strides,
                       dilations,
                       left_pads);

    // Run 2: Direct GPU Reference
    ck_tile::naive_grouped_conv_bwd_weight<2, ck::half_t, ck::half_t, ck::half_t>(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(wei_grad_naive_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    // Compare
    std::vector<ck::half_t> wei_grad_builder_result(wei_grad_elements);
    std::vector<ck::half_t> wei_grad_naive_result(wei_grad_elements);
    wei_grad_builder_dev.FromDevice(wei_grad_builder_result.data());
    wei_grad_naive_dev.FromDevice(wei_grad_naive_result.data());

    bool pass = ck::utils::check_err(wei_grad_builder_result,
                                     wei_grad_naive_result,
                                     "Error: Builder Backward Weight != Direct GPU Reference",
                                     1e-6,
                                     1e-6);

    std::cout << "✓ Builder Reference vs Direct GPU Reference (RANDOM INPUT - Backward Weight)!"
              << std::endl;
    std::cout << "  Result: " << (pass ? "IDENTICAL ✓" : "MISMATCH ✗") << std::endl;
    EXPECT_TRUE(pass);
}

// Test Invoker Interface vs Direct GPU Reference with RANDOM INPUT - Forward
TEST(ReferenceExecution, Forward_2D_FP16_InvokerInterface_vs_DirectGPUReference_Random)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 16, K = 16, H = 14, W = 14;

    const size_t in_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_size = G * N * K * H * W * sizeof(ck::half_t);

    const size_t in_elements  = G * N * C * H * W;
    const size_t wei_elements = G * K * C * 3 * 3;
    const size_t out_elements = G * N * K * H * W;

    std::vector<ck::half_t> in_host(in_elements);
    std::vector<ck::half_t> wei_host(wei_elements);

    std::srand(12348);
    for(size_t i = 0; i < in_elements; i++)
    {
        in_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(size_t i = 0; i < wei_elements; i++)
    {
        wei_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_invoker_dev(out_size);
    ck::DeviceMem out_naive_dev(out_size);

    in_dev.ToDevice(in_host.data());
    wei_dev.ToDevice(wei_host.data());
    out_invoker_dev.SetZero();
    out_naive_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel builder_kernel;

    // Run 1: Builder Invoker Interface
    auto argument_ptr = builder_kernel.MakeArgumentPointer(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(out_invoker_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    auto invoker_ptr = builder_kernel.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    // Run 2: Direct GPU Reference
    ck_tile::naive_grouped_conv_fwd<2, ck::half_t, ck::half_t, ck::half_t>(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(out_naive_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    // Compare
    std::vector<ck::half_t> out_invoker_result(out_elements);
    std::vector<ck::half_t> out_naive_result(out_elements);
    out_invoker_dev.FromDevice(out_invoker_result.data());
    out_naive_dev.FromDevice(out_naive_result.data());

    bool pass = ck::utils::check_err(out_invoker_result,
                                     out_naive_result,
                                     "Error: Invoker Interface != Direct GPU Reference",
                                     1e-6,
                                     1e-6);

    std::cout << "✓ Invoker Interface vs Direct GPU Reference (RANDOM - Forward)!" << std::endl;
    std::cout << "  Result: " << (pass ? "IDENTICAL ✓" : "MISMATCH ✗") << std::endl;
    EXPECT_TRUE(pass);
}

// Test Invoker Interface vs Direct GPU Reference with RANDOM INPUT - Backward Data
TEST(ReferenceExecution, BackwardData_2D_FP16_InvokerInterface_vs_DirectGPUReference_Random)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_DATA,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 16, K = 16, H = 14, W = 14;

    const size_t in_grad_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size      = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    const size_t in_grad_elements  = G * N * C * H * W;
    const size_t wei_elements      = G * K * C * 3 * 3;
    const size_t out_grad_elements = G * N * K * H * W;

    std::vector<ck::half_t> wei_host(wei_elements);
    std::vector<ck::half_t> out_grad_host(out_grad_elements);

    std::srand(12349);
    for(size_t i = 0; i < wei_elements; i++)
    {
        wei_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(size_t i = 0; i < out_grad_elements; i++)
    {
        out_grad_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    ck::DeviceMem in_grad_invoker_dev(in_grad_size);
    ck::DeviceMem in_grad_naive_dev(in_grad_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    wei_dev.ToDevice(wei_host.data());
    out_grad_dev.ToDevice(out_grad_host.data());
    in_grad_invoker_dev.SetZero();
    in_grad_naive_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel builder_kernel;

    // Run 1: Builder Invoker Interface
    auto argument_ptr = builder_kernel.MakeArgumentPointer(
        reinterpret_cast<ck::half_t*>(in_grad_invoker_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    auto invoker_ptr = builder_kernel.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    // Run 2: Direct GPU Reference
    ck_tile::naive_grouped_conv_bwd_data<2, ck::half_t, ck::half_t, ck::half_t>(
        reinterpret_cast<ck::half_t*>(in_grad_naive_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    // Compare
    std::vector<ck::half_t> in_grad_invoker_result(in_grad_elements);
    std::vector<ck::half_t> in_grad_naive_result(in_grad_elements);
    in_grad_invoker_dev.FromDevice(in_grad_invoker_result.data());
    in_grad_naive_dev.FromDevice(in_grad_naive_result.data());

    bool pass =
        ck::utils::check_err(in_grad_invoker_result,
                             in_grad_naive_result,
                             "Error: Invoker Interface != Direct GPU Reference (Backward Data)",
                             1e-6,
                             1e-6);

    std::cout << "✓ Invoker Interface vs Direct GPU Reference (RANDOM - Backward Data)!"
              << std::endl;
    std::cout << "  Result: " << (pass ? "IDENTICAL ✓" : "MISMATCH ✗") << std::endl;
    EXPECT_TRUE(pass);
}

// Test Invoker Interface vs Direct GPU Reference with RANDOM INPUT - Backward Weight
TEST(ReferenceExecution, BackwardWeight_2D_FP16_InvokerInterface_vs_DirectGPUReference_Random)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_WEIGHT,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 16, K = 16, H = 14, W = 14;

    const size_t in_size       = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_grad_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    const size_t in_elements       = G * N * C * H * W;
    const size_t wei_grad_elements = G * K * C * 3 * 3;
    const size_t out_grad_elements = G * N * K * H * W;

    std::vector<ck::half_t> in_host(in_elements);
    std::vector<ck::half_t> out_grad_host(out_grad_elements);

    std::srand(12350);
    for(size_t i = 0; i < in_elements; i++)
    {
        in_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(size_t i = 0; i < out_grad_elements; i++)
    {
        out_grad_host[i] = ck::half_t(static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_grad_invoker_dev(wei_grad_size);
    ck::DeviceMem wei_grad_naive_dev(wei_grad_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_dev.ToDevice(in_host.data());
    out_grad_dev.ToDevice(out_grad_host.data());
    wei_grad_invoker_dev.SetZero();
    wei_grad_naive_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel builder_kernel;

    // Run 1: Builder Invoker Interface
    auto argument_ptr = builder_kernel.MakeArgumentPointer(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(wei_grad_invoker_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    auto invoker_ptr = builder_kernel.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

    // Run 2: Direct GPU Reference
    ck_tile::naive_grouped_conv_bwd_weight<2, ck::half_t, ck::half_t, ck::half_t>(
        reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
        reinterpret_cast<ck::half_t*>(wei_grad_naive_dev.GetDeviceBuffer()),
        reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
        G,
        N,
        K,
        C,
        input_spatial,
        filter_spatial,
        output_spatial,
        strides,
        dilations,
        left_pads);

    // Compare
    std::vector<ck::half_t> wei_grad_invoker_result(wei_grad_elements);
    std::vector<ck::half_t> wei_grad_naive_result(wei_grad_elements);
    wei_grad_invoker_dev.FromDevice(wei_grad_invoker_result.data());
    wei_grad_naive_dev.FromDevice(wei_grad_naive_result.data());

    bool pass =
        ck::utils::check_err(wei_grad_invoker_result,
                             wei_grad_naive_result,
                             "Error: Invoker Interface != Direct GPU Reference (Backward Weight)",
                             1e-6,
                             1e-6);

    std::cout << "✓ Invoker Interface vs Direct GPU Reference (RANDOM - Backward Weight)!"
              << std::endl;
    std::cout << "  Result: " << (pass ? "IDENTICAL ✓" : "MISMATCH ✗") << std::endl;
    EXPECT_TRUE(pass);
}

} // namespace
