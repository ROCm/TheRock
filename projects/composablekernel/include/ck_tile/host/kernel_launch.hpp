// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <numeric>
#include <functional>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/timer.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

namespace ck_tile {

template <int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_TILE_USE_LAUNCH_BOUNDS
__launch_bounds__(Kernel::kBlockSize, MinBlockPerCu)
#endif
    __global__ void kentry(Args... args)
{
#if defined(__HIP_DEVICE_COMPILE__)
    Kernel{}(args...);
#else
    (..., (ignore = args, 0));
#endif
}

template <typename Arch, int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_TILE_USE_LAUNCH_BOUNDS
__launch_bounds__(Kernel::kBlockSize, MinBlockPerCu)
#endif
    __global__ void kentry(Args... args)
{
#if defined(__HIP_DEVICE_COMPILE__)
    Kernel{}(args...);
#else
    (..., (ignore = args, 0));
#endif
}

//
// return a anonymous functor(lambda) to be called later
// the KernelImpl should be a class without non-static data member, or let's say
// can be instantiate with "KernelImpl{}"
//
// the "static __device__ operator()(some_arg)" is the entry point of KernelImpl
//
// Arch can be used to support linking multiple object files that have the same kernel compiled for
// different architectures. In this case each object file has to use a different tag (gfx9_t,
// gfx12_t etc.), so the kernel will have different symbols for each architecture.
//
template <int MinBlockPerCu = CK_TILE_MIN_BLOCK_PER_CU,
          typename Arch     = void,
          typename KernelImpl,
          typename... Args>
CK_TILE_HOST auto
make_kernel(KernelImpl /*f*/, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    const auto kernel = []() {
        if constexpr(std::is_void_v<Arch>)
        {
            return kentry<MinBlockPerCu, KernelImpl, Args...>;
        }
        else
        {
            return kentry<Arch, MinBlockPerCu, KernelImpl, Args...>;
        }
    }();
    return [=](const stream_config& s) {
        kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
    };
}

template <typename... Callables>
CK_TILE_HOST void launch_and_check(const stream_config& sc, Callables&&... callables)
{
    // abort the sequence in case of intermediate error
    if(!((static_cast<void>(callables(sc)), hipPeekAtLastError() == hipSuccess) && ...))
    {
        HIP_CHECK_ERROR(hipGetLastError());
    }
}

// Measure the preprocess time during the cold iterations
template <typename TimerType, typename PreprocessFunc>
CK_TILE_HOST double
preprocess_profiling_impl(TimerType timer, const stream_config& s, PreprocessFunc preprocess)
{
    timer.start(s.stream_id_);
    for(int i = 0; i < s.nrepeat_; i++)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
    }
    timer.stop(s.stream_id_);

    return timer.duration() / s.nrepeat_;
}

template <typename TimerType, typename CallablesFunc, typename PreprocessFunc = std::nullptr_t>
CK_TILE_HOST double timing_loop_impl(TimerType timer,
                                     const stream_config& s,
                                     CallablesFunc&& callables_func,
                                     PreprocessFunc preprocess = nullptr)
{
    for(int i = 0; i < s.cold_niters_; i++)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
    }
    // Only profile preprocess if it's provided
    auto preprocess_time = 0.0;
    if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
    {
        preprocess_time = preprocess_profiling_impl(gpu_timer{}, s, preprocess);
    }

    int i = 0;
    timer.start(s.stream_id_);
    while(i < s.nrepeat_)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }

        callables_func();
        i++;
    }
    timer.stop(s.stream_id_);

    if(!i)
        return 0.;
    return (timer.duration() / s.nrepeat_) - preprocess_time;
}

// clang-format off
/*
 * launch_kernel()
 *
 * this is the function to launch arbitrary number of kernels with optional timer(selected by stream_config)
 * the callables should have signature as "operator()(const stream_config& s){ ... }" to call
 * 
 * the simplest way is pass in a lambda function, with "[=](const stream_config& s){ call_your_kernel_here() }"
 * as signature, for the callable (pay attention to the capture list)
 * 
 * e.g.
 *  ck_tile::launch_kernel(s,
 *                      [=](const stream_config& s){ hipMemset(ptr, 0, size) },
 *                      [=](const stream_config& s){ some_kernel<<<grids, blocks>>>(arg); }
 *                      );
 * 
 * if you use ck_tile kernel, or similiar to this style (structure with "static __device__ operator()(...){}")
 * you can pass your kernel to ck_tile::make_kernel(), which will create a anonymous functor for you,
 * then pass it to ck_tile::launch_kernel()
 * 
 * e.g.
 *  ck_tile::launch_kernel(s,
 *                      ck_tile::make_kernel<T0, B0>(kernel_0{}, grids0, blocks0, 0, kargs0),
 *                      ck_tile::make_kernel<T0, B1>(kernel_1{}, grids1, blocks1, 0, kargs1),
 *                       ...);
 **/
// clang-format on
template <typename... Callables>
CK_TILE_HOST float launch_kernel(const stream_config& s, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_impl(gpu_timer{}, s, callables_func);
    }
    else
    {
        return timing_loop_impl(cpu_timer{}, s, callables_func);
    }
}

template <typename PreprocessFunc, typename... Callables>
CK_TILE_HOST float
launch_kernel_time_mask(const stream_config& s, PreprocessFunc preprocess, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        preprocess();
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_impl(gpu_timer{}, s, callables_func, preprocess);
    }
    else
    {
        return timing_loop_impl(cpu_timer{}, s, callables_func, preprocess);
    }
}
} // namespace ck_tile
