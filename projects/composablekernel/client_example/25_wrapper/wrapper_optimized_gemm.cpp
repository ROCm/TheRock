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

template <bool DoPad, typename Layout, typename PaddingDims>
__device__ auto ApplyPadding(const Layout& layout, const PaddingDims& padding_dims)
{
    if constexpr(DoPad)
    {
        return ck::wrapper::pad(layout, padding_dims);
    }
    else
    {
        return layout;
    }
}

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout,
          bool DoPadding>
__global__ void __CK_WRAPPER_LAUNCH_BOUNDS__ DeviceGemm(const void* p_a,
                                                        const void* p_b,
                                                        void* p_c,
                                                        const ck::index_t M,
                                                        const ck::index_t N,
                                                        const ck::index_t K,
                                                        const BlockShape tile_shape,
                                                        const ThreadLayout thread_layout)
{
    constexpr auto MPerBlock  = ck::wrapper::size<0>(tile_shape);
    constexpr auto NPerBlock  = ck::wrapper::size<1>(tile_shape);
    constexpr auto KPerBlock  = ck::wrapper::size<2>(tile_shape);
    constexpr auto K1         = GemmTraits::K1;
    constexpr auto K0PerBlock = KPerBlock / K1;
    const auto K0             = ck::math::integer_divide_ceil(K, K1);

    const auto tile_shape_k0_m_n_k1 = ck::make_tuple(K0PerBlock, MPerBlock, NPerBlock, K1);
    // Create layouts for global memory
    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));
    // Apply padding
    auto a_padded_global_layout =
        ApplyPadding<DoPadding>(a_global_layout, ck::make_tuple(MPerBlock, KPerBlock));
    auto b_padded_global_layout =
        ApplyPadding<DoPadding>(b_global_layout, ck::make_tuple(NPerBlock, KPerBlock));
    auto c_padded_global_layout =
        ApplyPadding<DoPadding>(c_global_layout, ck::make_tuple(MPerBlock, NPerBlock));
    // Reshape from M,K to K0,M,K1
    const auto reshaped_dims_idxs =
        ck::make_tuple(ck::Number<1>{}, ck::make_tuple(ck::Number<0>{}, ck::Number<2>{}));
    auto a_padded_unmerged_global_layout =
        ck::wrapper::unmerge<1>(a_padded_global_layout, ck::make_tuple(K0, K1), reshaped_dims_idxs);
    auto b_padded_unmerged_global_layout =
        ck::wrapper::unmerge<1>(b_padded_global_layout, ck::make_tuple(K0, K1), reshaped_dims_idxs);
    // Create tensors for global memory
    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_padded_unmerged_global_layout);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_padded_unmerged_global_layout);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_padded_global_layout);
    // Create layouts and tensors for lds memory.
    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(K0PerBlock, MPerBlock, K1),
        ck::make_tuple((MPerBlock + ck::Number<1>{}) * K1, K1, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(K0PerBlock, NPerBlock, K1),
        ck::make_tuple((NPerBlock + ck::Number<1>{}) * K1, K1, ck::Number<1>{}));

    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout) + K0PerBlock];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout) + K0PerBlock];

    auto a_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_a), a_tile_layout);
    auto b_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_b), b_tile_layout);

    const auto block_idxs            = ck::make_tuple(ck::wrapper::slice(),
                                           static_cast<ck::index_t>(blockIdx.x),
                                           static_cast<ck::index_t>(blockIdx.y),
                                           ck::wrapper::slice());
    using DimAccessOrder             = ck::Tuple<ck::Number<1>, ck::Number<0>, ck::Number<2>>;
    constexpr ck::index_t vector_dim = 2;

    // Create tile and partition for C global memory. Use specific gemm
    // functions to get appropriate layouts.
    auto c_global_local_tile =
        ck::wrapper::make_local_tile(c_global_tensor,
                                     tile_shape_k0_m_n_k1,
                                     block_idxs,
                                     make_tuple(ck::wrapper::slice(K0PerBlock),
                                                ck::Number<1>{},
                                                ck::Number<1>{},
                                                ck::wrapper::slice(K1)));
    auto c_global_local_partition =
        ck::wrapper::make_blockwise_gemm_xdl_c_local_partition<DataType,
                                                               decltype(a_tile_layout),
                                                               decltype(b_tile_layout),
                                                               ck::wrapper::size(thread_layout),
                                                               GemmTraits>(c_global_local_tile);
    // Define and clear c vgpr register
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    ck::wrapper::clear(c_vgpr_reg);
    // Local partitions for lds memory
    auto a_lds_tensor_local_partition =
        ck::wrapper::make_local_partition(a_lds_tensor, thread_layout, threadIdx.x);
    auto b_lds_tensor_local_partition =
        ck::wrapper::make_local_partition(b_lds_tensor, thread_layout, threadIdx.x);
    // Lamda to slice tensor, then create local tile and partition
    auto make_global_partition = [&](auto tensor, auto projection, ck::index_t i) {
        const auto k_slice =
            ck::make_tuple(ck::wrapper::slice(i * K0PerBlock, (i + 1) * K0PerBlock),
                           ck::wrapper::slice(),
                           ck::wrapper::slice());
        auto local_tile = ck::wrapper::make_local_tile(
            tensor(k_slice), tile_shape_k0_m_n_k1, block_idxs, projection);
        return ck::wrapper::make_local_partition(local_tile, thread_layout, threadIdx.x);
    };

    auto a_global_local_partition = make_global_partition(
        a_global_tensor,
        make_tuple(ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}),
        0);
    auto b_global_local_partition = make_global_partition(
        b_global_tensor,
        make_tuple(ck::Number<1>{}, ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}),
        0);

    // (row-major vgpr layout)
    auto a_vgpr_tensor =
        ck::wrapper::make_register_tensor<ck::wrapper::MemoryTypeEnum::Vgpr, DataType>(
            ck::wrapper::make_layout(
                shape(a_global_local_partition),
                ck::make_tuple(ck::wrapper::size<1>(a_global_local_partition) *
                                   ck::wrapper::size<2>(a_global_local_partition),
                               ck::wrapper::size<2>(a_global_local_partition),
                               ck::Number<1>{})));
    auto b_vgpr_tensor =
        ck::wrapper::make_register_tensor<ck::wrapper::MemoryTypeEnum::Vgpr, DataType>(
            ck::wrapper::make_layout(
                shape(b_global_local_partition),
                ck::make_tuple(ck::wrapper::size<1>(a_global_local_partition) *
                                   ck::wrapper::size<2>(a_global_local_partition),
                               ck::wrapper::size<2>(a_global_local_partition),
                               ck::Number<1>{})));
    // Copy first values to lds
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(a_global_local_partition,
                                                                     a_vgpr_tensor);
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(b_global_local_partition,
                                                                     b_vgpr_tensor);
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(a_vgpr_tensor,
                                                                     a_lds_tensor_local_partition);
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(b_vgpr_tensor,
                                                                     b_lds_tensor_local_partition);
    // Pipeline loop
    const ck::index_t num_loop =
        __builtin_amdgcn_readfirstlane(ck::math::integer_divide_ceil(K, KPerBlock));
    // Skip if only tile should be processed
    if(num_loop > 1)
    {
        ck::index_t i = 0;
        do
        {
            auto a_global_local_partition_i = make_global_partition(
                a_global_tensor,
                make_tuple(
                    ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}),
                i + 1);
            auto b_global_local_partition_i = make_global_partition(
                b_global_tensor,
                make_tuple(
                    ck::Number<1>{}, ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}),
                i + 1);
            // Copy data to A vgpr.
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                a_global_local_partition_i, a_vgpr_tensor);
            // Synchronize.
            ck::block_sync_lds();
            // Copy data to B vgpr.
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                b_global_local_partition_i, b_vgpr_tensor);
            // Perform gemm.
            ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
                a_lds_tensor, b_lds_tensor, c_vgpr_reg);
            // Synchronize
            ck::block_sync_lds();
            // Copy data to A and B lds tiles.
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                a_vgpr_tensor, a_lds_tensor_local_partition);
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                b_vgpr_tensor, b_lds_tensor_local_partition);

            ++i;
        } while(i < (num_loop - 1));
    }
    // Handle tail.
    ck::block_sync_lds();
    ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
        a_lds_tensor, b_lds_tensor, c_vgpr_reg);
    // Store data from C vgpr to C global memory.
    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
}

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          bool DoPadding,
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
        DeviceGemm<DataType, GemmTraits, scalar_per_vector, BlockShape, ThreadLayout, DoPadding>;
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
    std::size_t flop     = std::size_t(2) * M * N * K;
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
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<4>{}, ck::Number<64>{}, ck::Number<1>{}),
                                 ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<256>{}, ck::Number<128>{}, ck::Number<32>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_8K1, 8, false>(
        3840, 4096, 4096, tile_shape, thread_layout);
    return 0;
}
#endif
