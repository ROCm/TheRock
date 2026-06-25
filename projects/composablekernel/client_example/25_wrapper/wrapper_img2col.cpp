// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <numeric>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"
#include "ck/wrapper/operations/copy.hpp"
#include "ck/wrapper/utils/kernel_utils.hpp"

static constexpr ck::index_t NumDimSpatial = 3;
using DataType                             = float;
using InputLayout                          = ck::tensor_layout::convolution::NDHWGC;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

template <typename InputTensor, typename OutputTensor, typename BlockShape, typename ThreadLayout>
__global__ void __CK_WRAPPER_LAUNCH_BOUNDS__
DeviceImageToColumnPad0(InputTensor input_tensor,
                        OutputTensor output_tensor,
                        const BlockShape tile_shape,
                        const ThreadLayout thread_layout)
{
    // grid layout (dim1, dim0)
    const auto block_idxs =
        ck::make_tuple(static_cast<ck::index_t>(blockIdx.y), static_cast<ck::index_t>(blockIdx.x));

    // Get local tiles for global memory
    auto input_local_tile  = ck::wrapper::make_local_tile(input_tensor, tile_shape, block_idxs);
    auto output_local_tile = ck::wrapper::make_local_tile(output_tensor, tile_shape, block_idxs);

    // Get partition per thread
    const auto input_local_partition =
        ck::wrapper::make_local_partition(input_local_tile, thread_layout, threadIdx.x);
    auto output_local_partition =
        ck::wrapper::make_local_partition(output_local_tile, thread_layout, threadIdx.x);

    // Perform copy
    using DimAccessOrder                    = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim        = 1;
    constexpr ck::index_t scalar_per_vector = 4;
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(input_local_partition,
                                                                     output_local_partition);
}

void PerformImageToColumnPad0(const ck::index_t G,
                              const ck::index_t N,
                              const ck::index_t Di,
                              const ck::index_t Hi,
                              const ck::index_t Wi,
                              const ck::index_t Do,
                              const ck::index_t Ho,
                              const ck::index_t Wo,
                              const ck::index_t C,
                              const ck::index_t Z,
                              const ck::index_t Y,
                              const ck::index_t X,
                              std::array<ck::index_t, NumDimSpatial> filter_strides,
                              std::array<ck::index_t, NumDimSpatial> filter_dilations)
{
    const ck::index_t ZYXC = Z * Y * X * C;
    const ck::index_t GC   = G * C;

    // shape: (G, (Wo, Ho, Do, N)), (C, X, Y, Z))
    const auto shape = ck::make_tuple(ck::make_tuple(G, ck::make_tuple(Wo, Ho, Do, N)),
                                      ck::make_tuple(C, X, Y, Z));
    const auto in_strides =
        ck::make_tuple(ck::make_tuple(C,
                                      ck::make_tuple(filter_strides[2] * GC,
                                                     filter_strides[1] * Wi * GC,
                                                     filter_strides[0] * Hi * Wi * GC,
                                                     Di * Hi * Wi * GC)),
                       ck::make_tuple(1,
                                      filter_dilations[2] * GC,
                                      filter_dilations[1] * Wi * GC,
                                      filter_dilations[0] * Hi * Wi * GC));
    const auto in_layout = ck::wrapper::make_layout(shape, in_strides);

    const auto out_strides = ck::make_tuple(
        ck::make_tuple(
            ZYXC,
            ck::make_tuple(ZYXC * G, Wo * ZYXC * G, Ho * Wo * ZYXC * G, Do * Ho * Wo * ZYXC * G)),
        ck::make_tuple(1, C, X * C, Y * X * C));
    const auto out_layout = ck::wrapper::make_layout(shape, out_strides);

    const ck::index_t input_size = N * Di * Hi * Wi * GC;
    // Global memory buffers
    SimpleDeviceMem in_buf(input_size * sizeof(DataType));
    SimpleDeviceMem out_buf(ck::wrapper::size(out_layout) * sizeof(DataType));

    // User can choose appropriate number of threads and sizes per block
    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<8>{}, ck::Number<16>{}),
                                 ck::make_tuple(ck::Number<16>{}, ck::Number<1>{}));
    // This example doesn't support padding, user should select tile sizes
    // which are divisible by the shape.
    const auto tile_shape = ck::make_tuple(ck::Number<32>{}, ck::Number<64>{});

    // Create buffers for global memory
    auto input_tensor_global = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(in_buf.GetDeviceBuffer()), in_layout);
    auto output_tensor_global = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(out_buf.GetDeviceBuffer()), out_layout);

    // grid layout (dim1, dim0)
    const ck::index_t grid_size_x = ck::math::integer_divide_ceil(ck::wrapper::size<1>(in_layout),
                                                                  ck::wrapper::size<1>(tile_shape));
    const ck::index_t grid_size_y = ck::math::integer_divide_ceil(ck::wrapper::size<0>(in_layout),
                                                                  ck::wrapper::size<0>(tile_shape));

    const auto kernel    = DeviceImageToColumnPad0<decltype(input_tensor_global),
                                                   decltype(output_tensor_global),
                                                   decltype(tile_shape),
                                                   decltype(thread_layout)>;
    const float avg_time = launch_and_time_kernel(StreamConfig{nullptr, true},
                                                  kernel,
                                                  dim3(grid_size_x, grid_size_y, 1),
                                                  dim3(ck::wrapper::size(thread_layout)),
                                                  0,
                                                  input_tensor_global,
                                                  output_tensor_global,
                                                  tile_shape,
                                                  thread_layout);

    std::size_t num_btype = G * N * Do * Ho * Wo * ZYXC * 2 * sizeof(DataType);
    float gb_per_sec      = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
              << std::endl;
}

int main(int argc, char* argv[])
{
    constexpr ck::index_t G  = 4;  // number of groups
    constexpr ck::index_t N  = 32; // batch
    constexpr ck::index_t C  = 64; // input channel (per group)
    constexpr ck::index_t Z  = 3;  // filter D
    constexpr ck::index_t Y  = 3;  // filter H
    constexpr ck::index_t X  = 3;  // filter W
    constexpr ck::index_t Di = 9;  // input D
    constexpr ck::index_t Hi = 9;  // input H
    constexpr ck::index_t Wi = 7;  // input W
    constexpr ck::index_t Do = 7;  // output D
    constexpr ck::index_t Ho = 7;  // output H
    constexpr ck::index_t Wo = 5;  // output W
    PerformImageToColumnPad0(G,
                             N,
                             Di,
                             Hi,
                             Wi,
                             Do,
                             Ho,
                             Wo,
                             C,
                             Z,
                             Y,
                             X,
                             {1, 1, 1} /*filter_strides*/,
                             {1, 1, 1} /*filter_dilations*/);
    return 0;
}
