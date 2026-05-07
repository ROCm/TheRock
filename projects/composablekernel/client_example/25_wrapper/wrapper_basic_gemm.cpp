// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>

#include "ck/utility/common_header.hpp"
// __gfx9__ defined in the above header via ck.hpp
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"
#include "ck/wrapper/operations/copy.hpp"
#include "ck/wrapper/operations/gemm.hpp"
#include "ck/wrapper/utils/kernel_utils.hpp"
#include "ck/host_utility/device_prop.hpp"

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

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout>
__global__ void __CK_WRAPPER_LAUNCH_BOUNDS__ DeviceGemm(const void* p_a,
                                                        const void* p_b,
                                                        void* p_c,
                                                        const ck::index_t M,
                                                        const ck::index_t N,
                                                        const ck::index_t K,
                                                        const BlockShape tile_shape,
                                                        const ThreadLayout thread_layout)
{
    constexpr auto MPerBlock = ck::wrapper::size<0>(tile_shape);
    constexpr auto NPerBlock = ck::wrapper::size<1>(tile_shape);
    constexpr auto KPerBlock = ck::wrapper::size<2>(tile_shape);

    // Specify layouts for global memory.
    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));
    // Specify layouts for tiles.
    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(NPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto c_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, NPerBlock), ck::make_tuple(NPerBlock, ck::Number<1>{}));
    // Apply padding for global memory.
    auto a_global_layout_padded = ck::wrapper::pad(a_global_layout, shape(a_tile_layout));
    auto b_global_layout_padded = ck::wrapper::pad(b_global_layout, shape(b_tile_layout));
    auto c_global_layout_padded = ck::wrapper::pad(c_global_layout, shape(c_tile_layout));
    // Make tensors for global memory.
    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_global_layout_padded);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_global_layout_padded);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_global_layout_padded);
    // Allocate lds memory.
    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout)];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout)];
    // Make tensors for lds memory.
    auto a_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_a), a_tile_layout);
    auto b_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_b), b_tile_layout);
    // Specify block index as tuple.
    const auto block_idxs = ck::make_tuple(static_cast<ck::index_t>(blockIdx.x),
                                           static_cast<ck::index_t>(blockIdx.y),
                                           ck::wrapper::slice());
    // Specify access parameters for copy.
    using DimAccessOrder             = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim = 1;
    // Create tile and partition for C. Use specific function for blockwise_gemm to assign the
    // appropriate partitions.
    auto c_global_local_tile = ck::wrapper::make_local_tile(
        c_global_tensor,
        tile_shape,
        block_idxs,
        make_tuple(ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(KPerBlock)));
    auto c_global_local_partition =
        ck::wrapper::make_blockwise_gemm_xdl_c_local_partition<DataType,
                                                               decltype(a_tile_layout),
                                                               decltype(b_tile_layout),
                                                               ck::wrapper::size(thread_layout),
                                                               GemmTraits>(c_global_local_tile);
    // Create C vgpr to accumulate results.
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    // Clear C vgpr.
    ck::wrapper::clear(c_vgpr_reg);

    // Iterate over K with KPerBlock step.
    const ck::index_t num_loop = ck::math::integer_divide_ceil(K, KPerBlock);
    ck::index_t i              = 0;
    do
    {
        // Get KPerBlock slice.
        const auto k_slice           = ck::wrapper::slice(i * KPerBlock, (i + 1) * KPerBlock);
        auto a_global_tensor_k_slice = a_global_tensor(ck::wrapper::slice(), k_slice);
        auto b_global_tensor_k_slice = b_global_tensor(ck::wrapper::slice(), k_slice);
        // Create local tiles for A and B.
        auto a_global_local_tile = ck::wrapper::make_local_tile(
            a_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}));
        auto b_global_local_tile = ck::wrapper::make_local_tile(
            b_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}));
        // Copy from global to lds.
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            a_global_local_tile, a_lds_tensor, thread_layout);
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            b_global_local_tile, b_lds_tensor, thread_layout);
        // Synchronize lds.
        ck::block_sync_lds();
        // Execute blockwise gemm.
        ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
            a_lds_tensor, b_lds_tensor, c_vgpr_reg);

        ++i;
    } while(i < num_loop);
    // Copy vgpr results to C global memory.
    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
}

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout>
void PerformGemm(const ck::index_t M,
                 const ck::index_t N,
                 const ck::index_t K,
                 const BlockShape& tile_shape,
                 const ThreadLayout& thread_layout)
{
    // Global memory buffers
    SimpleDeviceMem a_mem(M * K * sizeof(DataType));
    SimpleDeviceMem b_mem(K * N * sizeof(DataType));
    SimpleDeviceMem c_mem(M * N * sizeof(DataType));

    const ck::index_t grid_size_x =
        ck::math::integer_divide_ceil(M, ck::wrapper::size<0>(tile_shape));
    const ck::index_t grid_size_y =
        ck::math::integer_divide_ceil(N, ck::wrapper::size<1>(tile_shape));

    const auto kernel =
        DeviceGemm<DataType, GemmTraits, scalar_per_vector, BlockShape, ThreadLayout>;
    const float avg_time = launch_and_time_kernel(StreamConfig{nullptr, true},
                                                  kernel,
                                                  dim3(grid_size_x, grid_size_y, 1),
                                                  dim3(ck::wrapper::size(thread_layout)),
                                                  0,
                                                  a_mem.GetDeviceBuffer(),
                                                  b_mem.GetDeviceBuffer(),
                                                  c_mem.GetDeviceBuffer(),
                                                  M,
                                                  N,
                                                  K,
                                                  tile_shape,
                                                  thread_layout);

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(DataType) * M * K + sizeof(DataType) * K * N + sizeof(DataType) * M * N;

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;

    std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << std::endl;
}

int main(int argc, char* argv[])
{
    bool is_supported = ck::is_xdl_supported();
    if(!is_supported)
    {
        std::cout << "WARNING: xdl example not supported on the platform " << ck::get_device_name()
                  << std::endl;
        return 0;
    }

    using DataType = ck::half_t;
    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<64>{}, ck::Number<4>{}),
                                 ck::make_tuple(ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<256>{}, ck::Number<128>{}, ck::Number<32>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_8K1, 8>(
        3840, 4096, 4096, tile_shape, thread_layout);
    return 0;
}
#endif
