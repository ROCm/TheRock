// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/library/utility/host_tensor.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"
#include "ck/wrapper/operations/copy.hpp"
#include "ck/wrapper/operations/gemm.hpp"
#include "ck/wrapper/utils/kernel_utils.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <typename DataType>
void CheckResult(const std::vector<DataType>& a_data,
                 const std::vector<DataType>& b_data,
                 std::vector<DataType>& c_m_n_device_result,
                 const ck::index_t M,
                 const ck::index_t N,
                 const ck::index_t K)
{
    using PassThrough           = ck::tensor_operation::element_wise::PassThrough;
    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<DataType, DataType, DataType, float, PassThrough, PassThrough, PassThrough>;

    Tensor<DataType> a_m_k(HostTensorDescriptor({M, K}));
    Tensor<DataType> b_k_n(HostTensorDescriptor({K, N}, {1, K}));
    Tensor<DataType> c_m_n_host_result(HostTensorDescriptor({M, N}));

    a_m_k.mData = a_data;
    b_k_n.mData = b_data;

    auto ref_op       = ReferenceGemmInstance{};
    auto ref_invoker  = ref_op.MakeInvoker();
    auto ref_argument = ref_op.MakeArgument(
        a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

    ref_invoker.Run(ref_argument);
    EXPECT_TRUE(ck::utils::check_err(c_m_n_device_result, c_m_n_host_result.mData));
}

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
#if defined(__gfx9__)
    constexpr auto MPerBlock  = ck::wrapper::size<0>(tile_shape);
    constexpr auto NPerBlock  = ck::wrapper::size<1>(tile_shape);
    constexpr auto KPerBlock  = ck::wrapper::size<2>(tile_shape);
    constexpr auto K1         = GemmTraits::K1;
    constexpr auto K0PerBlock = KPerBlock / K1;
    const auto K0             = ck::math::integer_divide_ceil(K, K1);

    const auto tile_shape_k0_m_n_k1 = ck::make_tuple(K0PerBlock, MPerBlock, NPerBlock, K1);

    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));

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

    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_padded_unmerged_global_layout);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_padded_unmerged_global_layout);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_padded_global_layout);

    // Add extra M and N
    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(K0PerBlock, MPerBlock, K1),
        ck::make_tuple((MPerBlock + ck::Number<1>{}) * K1, K1, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(K0PerBlock, NPerBlock, K1),
        ck::make_tuple((NPerBlock + ck::Number<1>{}) * K1, K1, ck::Number<1>{}));

    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout) + NPerBlock];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout) + NPerBlock];

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
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    ck::wrapper::clear(c_vgpr_reg);

    auto a_lds_tensor_local_partition =
        ck::wrapper::make_local_partition(a_lds_tensor, thread_layout, threadIdx.x);
    auto b_lds_tensor_local_partition =
        ck::wrapper::make_local_partition(b_lds_tensor, thread_layout, threadIdx.x);

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

    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(a_global_local_partition,
                                                                     a_vgpr_tensor);
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(b_global_local_partition,
                                                                     b_vgpr_tensor);
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(a_vgpr_tensor,
                                                                     a_lds_tensor_local_partition);
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(b_vgpr_tensor,
                                                                     b_lds_tensor_local_partition);

    const ck::index_t num_loop =
        __builtin_amdgcn_readfirstlane(ck::math::integer_divide_ceil(K, KPerBlock));
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

            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                a_global_local_partition_i, a_vgpr_tensor);

            ck::block_sync_lds();
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                b_global_local_partition_i, b_vgpr_tensor);

            ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
                a_lds_tensor, b_lds_tensor, c_vgpr_reg);

            ck::block_sync_lds();
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                a_vgpr_tensor, a_lds_tensor_local_partition);
            ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(
                b_vgpr_tensor, b_lds_tensor_local_partition);

            ++i;
        } while(i < (num_loop - 1));
    }
    ck::block_sync_lds();
    ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
        a_lds_tensor, b_lds_tensor, c_vgpr_reg);

    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
#else
    ck::ignore = p_a;
    ck::ignore = p_b;
    ck::ignore = p_c;
    ck::ignore = M;
    ck::ignore = N;
    ck::ignore = K;
    ck::ignore = tile_shape;
    ck::ignore = thread_layout;
#endif
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
    DeviceMem a_mem(M * K * sizeof(DataType));
    DeviceMem b_mem(K * N * sizeof(DataType));
    DeviceMem c_mem(M * N * sizeof(DataType));

    std::vector<DataType> a_data(M * K);
    std::vector<DataType> b_data(K * N);
    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(a_data);
    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(b_data);

    a_mem.ToDevice(a_data.data());
    b_mem.ToDevice(b_data.data());
    c_mem.SetZero();

    const ck::index_t grid_size_x =
        ck::math::integer_divide_ceil(M, ck::wrapper::size<0>(tile_shape));
    const ck::index_t grid_size_y =
        ck::math::integer_divide_ceil(N, ck::wrapper::size<1>(tile_shape));

    const auto kernel =
        DeviceGemm<DataType, GemmTraits, scalar_per_vector, BlockShape, ThreadLayout, DoPadding>;
    const float avg_time = launch_and_time_kernel(StreamConfig{nullptr, false},
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

    std::vector<DataType> c_data(M * N);
    c_mem.FromDevice(c_data.data());
    CheckResult<DataType>(a_data, b_data, c_data, M, N, K);
}

TEST(TestGemm, Float)
{
    using DataType = float;
    // (dim1, dim2, dim0 thread layout)
    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<4>{}, ck::Number<64>{}, ck::Number<1>{}),
                                 ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<16>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_4K1, 4, false>(
        512, 512, 128, tile_shape, thread_layout);
    // Irregular case
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_4K1, 1, true>(
        129, 129, 67, tile_shape, thread_layout);
}

TEST(TestGemm, Int8)
{
    using DataType = int8_t;
    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<4>{}, ck::Number<64>{}, ck::Number<1>{}),
                                 ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<64>{});
    PerformGemm<DataType,
                ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_16K1,
                16,
                false>(512, 512, 128, tile_shape, thread_layout);
    // Irregular case
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_16K1, 1, true>(
        129, 129, 67, tile_shape, thread_layout);
}

TEST(TestGemm, Half)
{
    using DataType = ck::half_t;
    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<4>{}, ck::Number<64>{}, ck::Number<1>{}),
                                 ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<32>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_8K1, 8, false>(
        512, 512, 128, tile_shape, thread_layout);
    // Irregular case
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_8K1, 1, true>(
        129, 129, 67, tile_shape, thread_layout);
}

TEST(TestGemm, Float_2x4_4x2_XdlPerWave)
{
    using DataType = float;
    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<4>{}, ck::Number<64>{}, ck::Number<1>{}),
                                 ck::make_tuple(ck::Number<1>{}, ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<256>{}, ck::Number<128>{}, ck::Number<16>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_4K1, 4, false>(
        512, 512, 128, tile_shape, thread_layout);
}

int main(int argc, char** argv)
{
    if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
    {
        std::cout << "This test support gfx9 only" << std::endl;
        return 0;
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
