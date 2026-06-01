// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/host_utility/hip_check_error.hpp"

#include <hip/hip_runtime.h>

namespace ck {
namespace s_prefetch_op_util {

// Enable scalar prefetch in hardware (required on gfx12 before using s_prefetch)
__device__ __forceinline__ void enable_scalar_prefetch()
{
#if defined(__gfx12__)
    // SCALAR_PREFETCH_EN is bit 24 in MODE register (hwreg 1)
    // Set 1 bit at offset 24 to value 1
    __builtin_amdgcn_s_setreg(1 | (24 << 6), 1); // Set bit to 1
#endif
}

template <typename T>
struct SPrefetchDataOp
{
    // Prefetch to constant cache using AMD builtin with cachelines to prefetch(1..32)
    __device__ __forceinline__ void operator()(const T CK_CONSTANT_ADDRESS_SPACE* addr,
                                               unsigned int num_cachelines) const
    {
#if defined(__gfx12__)
        assert(num_cachelines > 0 && num_cachelines <= 32);
        __builtin_amdgcn_s_prefetch_data(addr, num_cachelines - 1); // we need to pass 0..31
#else
        // ignore - not supported
        (void)addr;
        (void)num_cachelines;
#endif
    }
};

template <typename T, uint32_t NUM_SCALARS>
struct SBufferPrefetchDataOp
{
    // Prefetch to constant cache using AMD builtin with cachelines to prefetch(1..32)
    __device__ __forceinline__ void operator()(const T CK_CONSTANT_ADDRESS_SPACE* addr,
                                               unsigned int num_cachelines) const
    {
#if defined(__gfx12__)
        __amdgpu_buffer_rsrc_t buf_res = make_wave_buffer_resource_new(addr, NUM_SCALARS);
        assert(num_cachelines > 0 && num_cachelines <= 32);
        __builtin_amdgcn_s_buffer_prefetch_data(buf_res, 0, num_cachelines - 1);
#else
        // ignore - not supported
        (void)addr;
        (void)num_cachelines;
#endif
    }
};

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS, typename PrefetchOp>
__global__ void kernel_with_prefetch(const T* src,
                                     T* dst,
                                     const T CK_CONSTANT_ADDRESS_SPACE* scalar_data,
                                     bool enable_prefetch)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate number of 128B cachelines needed to cover num_scalars elements
    constexpr index_t cachelineSize              = 128;
    constexpr index_t elements_per_cachelineSize = cachelineSize / sizeof(T);
    constexpr unsigned int cachelinesNeeded =
        (NUM_SCALARS + elements_per_cachelineSize - 1) / elements_per_cachelineSize;

    // Prefetch all scalar data at once
    if(threadIdx.x == 0)
    {
        if(enable_prefetch)
        {
            enable_scalar_prefetch();
        }
        PrefetchOp{}(scalar_data, cachelinesNeeded);
    }

    T sum = 0;
    if(tid < NUM_THREADS)
    {
        sum = src[tid]; // load from global mem to give time for prefetch to finish or be close to
                        // finishs
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < NUM_THREADS)
    {
        // Access prefetched scalar data
        for(uint32_t i = 0; i < NUM_SCALARS; i++)
        {
            sum += scalar_data[i]; // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename PrefetchKernel, typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool test_prefetch_impl(bool time_kernels,
                        const PrefetchKernel& prefetch_kernel,
                        const std::string& kernel_name)
{
    // TODO: maybe add more prefetch instructions inside kernel to support more values
    assert(NUM_SCALARS / sizeof(T) < (128 * 32));
    constexpr index_t num_elements = NUM_THREADS;
    constexpr index_t num_scalars  = NUM_SCALARS;
    constexpr index_t block_size   = 256;
    constexpr index_t grid_size    = (num_elements + block_size - 1) / block_size;

    std::cout << "Testing " << kernel_name << " to constant cache for type: " << typeid(T).name()
              << std::endl;
    std::cout << "Elements: " << num_elements << ", Scalars: " << num_scalars << std::endl;

    // Host data
    std::vector<T> h_src(num_elements);
    std::vector<T> h_scalar(num_scalars);
    std::vector<T> h_dst_with_prefetch_chunks(num_elements);
    std::vector<T> h_expected(num_elements);

    // Initialize data
    for(index_t i = 0; i < num_elements; i++)
    {
        h_src[i] = static_cast<T>(i % 100);
    }

    T scalar_sum = 0;
    for(index_t i = 0; i < num_scalars; i++)
    {
        h_scalar[i] = static_cast<T>(i + 1);
        scalar_sum += h_scalar[i];
    }

    // Expected results
    for(index_t i = 0; i < num_elements; i++)
    {
        h_expected[i] = h_src[i] + scalar_sum;
    }

    // Device memory
    DeviceMem d_src(sizeof(T) * num_elements);
    DeviceMem d_scalar(sizeof(T) * num_scalars);
    DeviceMem d_dst_with_prefetch_chunks(sizeof(T) * num_elements);

    d_src.ToDevice(h_src.data());
    d_scalar.ToDevice(h_scalar.data());

    hipStream_t stream;
    hip_check_error(hipStreamCreate(&stream));

    if(time_kernels)
    {
        ck::static_for<0, 2, 1>{}([&](auto static_i) {
            constexpr bool prefetch_enabled = static_i == 0;
            std::cout << "PREFETCH " << (prefetch_enabled ? "ENABLED!" : "DISABLED!") << std::endl;

            constexpr int num_warmup     = 1;
            constexpr int num_iterations = 10;

            // Warmup runs
            for(int i = 0; i < num_warmup; i++)
            {
                prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<const T*>(d_src.GetDeviceBuffer()),
                    static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
                    cast_pointer_to_constant_address_space(
                        static_cast<const T*>(d_scalar.GetDeviceBuffer())),
                    prefetch_enabled);
            }
            hip_check_error(hipStreamSynchronize(stream));

            // Performance measurement
            hipEvent_t start, stop;
            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipEventRecord(start, stream));
            for(int i = 0; i < num_iterations; i++)
            {
                prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<const T*>(d_src.GetDeviceBuffer()),
                    static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
                    cast_pointer_to_constant_address_space(
                        static_cast<const T*>(d_scalar.GetDeviceBuffer())),
                    prefetch_enabled);
            }
            hip_check_error(hipEventRecord(stop, stream));

            hip_check_error(hipStreamSynchronize(stream));

            float elapsed_ms = 0;
            hip_check_error(hipEventElapsedTime(&elapsed_ms, start, stop));

            float avg_time_us       = (elapsed_ms * 1000.0f) / num_iterations;
            float total_bytes       = (num_elements * sizeof(T) + num_scalars * sizeof(T)); // read
            float bandwidth_gb_s    = (total_bytes / (avg_time_us * 1e-6)) / 1e9;
            float ops_per_iteration = num_elements * num_scalars; // adds
            float gflops            = (ops_per_iteration / (avg_time_us * 1e-6)) / 1e9;

            std::cout << "  Performance: " << std::endl;
            std::cout << "    Average kernel time: " << avg_time_us << " us" << std::endl;
            std::cout << "    Effective bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
            std::cout << "    Compute throughput: " << gflops << " GFLOPS" << std::endl;

            hip_check_error(hipEventDestroy(start));
            hip_check_error(hipEventDestroy(stop));
        });
    }
    else
    {
        prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const T*>(d_src.GetDeviceBuffer()),
            static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(
                static_cast<const T*>(d_scalar.GetDeviceBuffer())),
            true);

        hip_check_error(hipStreamSynchronize(stream));
    }

    // Copy results back
    d_dst_with_prefetch_chunks.FromDevice(h_dst_with_prefetch_chunks.data());

    // Verify results
    bool pass = ck::utils::check_err(h_dst_with_prefetch_chunks, h_expected);

    std::cout << "  Correctness: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    hip_check_error(hipStreamDestroy(stream));

    return pass;
}

} // namespace s_prefetch_op_util
} // namespace ck
