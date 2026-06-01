/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include <hipblaslt/hipblaslt.h>

template <typename T, typename F>
__global__ void fill_kernel(T* A, size_t size, size_t offset, F f)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx + offset] = f(idx + offset);
}

template <typename T, typename F>
void fill_batch(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, const F& f)
{
    size_t           size_64   = stride >= lda ? lda * N + size_t(batch_count - 1) * stride : lda * N;
    constexpr size_t c_i32_max = size_t(std::numeric_limits<int32_t>::max());
    for(size_t offset = 0; offset < size_64; offset += c_i32_max)
    {
        size_t size       = std::min(size_64 - offset, c_i32_max);
        size_t block_size = 256;
        size_t grid_size  = (size + block_size - 1) / block_size;
        fill_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(A, size, offset, f);
    }
    CHECK_HIP_ERROR(hipGetLastError());
}

__device__ uint32_t pseudo_random_device(size_t idx)
{
    // Numerical Recipes ranqd1, Chapter 7.1, ?An Even Quicker Generator, Eq. 7.1.6. parameters from Knuth and H. W. Lewis
    auto s = idx * 1664525 + 1013904223;
    // Run a few extra iterations to make the generators diverge
    // in case the seeds are still poor (consecutive ints)
    // Marsaglia, G. (2003). "Xorshift RNGs". Journal of Statistical Software. 8 (14). doi:10.18637/jss.v008.i14
    s ^= (s << 13) ^ ( s >> 17) ^ (s << 5);
    s ^= (s << 13) ^ ( s >> 17) ^ (s << 5);
    s ^= (s << 13) ^ ( s >> 17) ^ (s << 5);
    return s;
}

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
__device__ T random_int(size_t idx)
{
    return T(pseudo_random_device(idx) % 10 + 1.f);
}

/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
__device__ hipblasLtHalf random_int<hipblasLtHalf>(size_t idx)
{
    return hipblasLtHalf(pseudo_random_device(idx) % 5 - 2.f);
}

/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
__device__ hip_bfloat16 random_int<hip_bfloat16>(size_t idx)
{
    return hip_bfloat16(pseudo_random_device(idx) % 5 - 2.f);
}

/*! \brief  generate a random number in range [1,2,3] */
template <>
__device__ int8_t random_int<int8_t>(size_t idx)
{
    return pseudo_random_device(idx) % 3 + 1;
}

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
__device__ T random_hpl(size_t idx)
{
    auto r = pseudo_random_device(idx);
    return T(double(r) / double(std::numeric_limits<decltype(r)>::max()) - 0.5);
}

/*! \brief  generate a random number in [-1.0,1.0] doubles  */
template <>
__device__ int8_t random_hpl(size_t idx)
{
    auto r = pseudo_random_device(idx);
    auto v = nearbyint(double(r) / double(std::numeric_limits<decltype(r)>::max()) * 2. - 1.);
    return int8_t(v > 127.f ? 127.f : v < -128.f ? -128.f : v);
}

/*! \brief  generate a random number in [0.,1.0]  */
template <typename T>
__device__ T uniform_01(size_t idx)
{
    auto r = pseudo_random_device(idx);
    return T(double(r) / double(std::numeric_limits<decltype(r)>::max()));
}

/*! \brief  generate a random number in [0.,1.0]  */
template <>
__device__ int8_t uniform_01(size_t idx)
{
    auto r = pseudo_random_device(idx);
    auto v = nearbyint(double(r) / double(std::numeric_limits<decltype(r)>::max()));
    return int8_t(v > 127.f ? 127.f : v < -128.f ? -128.f : v);
}


template <typename T>
void hipblaslt_init_device(ABC_dims                 abc,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           T*                       A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           size_t                   stride,
                           size_t                   batch_count)
{
    if(is_nan)
    {
        std::array<T, 100> rand_nans;
        for(auto& r : rand_nans)
            r = T(hipblaslt_nan_rng());
        fill_batch(A, M, N, lda, stride, batch_count, [rand_nans](size_t idx) -> T {
            return rand_nans[pseudo_random_device(idx) % rand_nans.size()];
        });
    }
    else
    {
        switch(init)
        {
        case hipblaslt_initialization::rand_int:
            if(abc == ABC_dims::A || abc == ABC_dims::C)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return random_int<T>(idx);
                });
            else if(abc == ABC_dims::B)
            {
                if(stride >= lda)
                {
                    stride = std::max(lda * N, stride);
                    fill_batch(A, M, N, lda, stride, batch_count, [stride, lda](size_t idx) -> T {
                        auto b     = idx / stride;
                        auto j     = (idx - b * stride) / lda;
                        auto i     = (idx - b * stride) - j * lda;
                        auto value = random_int<T>(idx);
                        return (i ^ j) & 1 ? value : negate(value);
                    });
                }
                else
                {
                    fill_batch(A, M, N, lda, stride, batch_count, [stride, lda](size_t idx) -> T {
                        auto j     = idx / lda;
                        auto b     = (idx - j * lda) / stride;
                        auto i     = (idx - j * lda) - b * stride;
                        auto value = random_int<T>(idx);
                        return (i ^ j) & 1 ? value : negate(value);
                    });
                }
            }
            break;
        case hipblaslt_initialization::trig_float:
            if(stride >= lda)
            {
                stride = std::max(lda * N, stride);
                if(abc == ABC_dims::A || abc == ABC_dims::C)
                    fill_batch(A, M, N, lda, stride, batch_count, [M, N, stride, lda](size_t idx) -> T {
                        auto b = idx / stride;
                        auto j = (idx - b * stride) / lda;
                        auto i = (idx - b * stride) - j * lda;
                        return T(sin(double(i + j * M + b * M * N)));
                    });
                else if(abc == ABC_dims::B)
                    fill_batch(A, M, N, lda, stride, batch_count, [M, N, stride, lda](size_t idx) -> T {
                        auto b = idx / stride;
                        auto j = (idx - b * stride) / lda;
                        auto i = (idx - b * stride) - j * lda;
                        return T(cos(double(i + j * M + b * M * N)));
                    });
            }
            else
            {
                if(abc == ABC_dims::A || abc == ABC_dims::C)
                    fill_batch(A, M, N, lda, stride, batch_count, [M, N, stride, lda](size_t idx) -> T {
                        auto j = idx / lda;
                        auto b = (idx - j * lda) / stride;
                        auto i = (idx - j * lda) - b * stride;
                        return T(sin(double(i + j * M + b * M * N)));
                    });
                else if(abc == ABC_dims::B)
                    fill_batch(A, M, N, lda, stride, batch_count, [M, N, stride, lda](size_t idx) -> T {
                        auto j = idx / lda;
                        auto b = (idx - j * lda) / stride;
                        auto i = (idx - j * lda) - b * stride;
                        return T(cos(double(i + j * M + b * M * N)));
                    });
            }
            break;
        case hipblaslt_initialization::hpl:
            fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                return random_hpl<T>(idx);
            });
            break;
        case hipblaslt_initialization::special:
            if(abc == ABC_dims::A)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(hipblasLtHalf(65280.0));
                });
            else if(abc == ABC_dims::B)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(hipblasLtHalf(0.0000607967376708984375));
                });
            else if(abc == ABC_dims::C)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(pseudo_random_device(idx) % 10 + 1.f);
                });
            break;
        case hipblaslt_initialization::zero:
            fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T { return T(0); });
            break;
        case hipblaslt_initialization::norm_dist:
            {
                std::random_device rd;
                auto base_seed = rd(); // Get a random seed for each run
                fill_batch(A, M, N, lda, stride, batch_count, [base_seed] __device__ (size_t idx) -> T {
                    hipblaslt_norm_dist::XorwowState state;
                    hipblaslt_norm_dist::init_xorwow(&state, base_seed + idx); // Unique seed per thread
                    return T(hipblaslt_norm_dist::box_muller_normal(&state));
                });
                break;
            }
        case hipblaslt_initialization::uniform_01:
            fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                return uniform_01<T>(idx);
            });
            break;
        default:
            hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
            break;
        }
    }
}

void hipblaslt_init_device(ABC_dims                 abc,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           void*                    A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           hipDataType              type,
                           size_t                   stride,
                           size_t                   batch_count)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_device<float>(
            abc, init, is_nan, static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_device<double>(
            abc, init, is_nan, static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_device<hipblasLtHalf>(
            abc, init, is_nan, static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_device<hip_bfloat16>(
            abc, init, is_nan, static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_device<hipblaslt_f8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_device<hipblaslt_bf8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_device<hipblaslt_f8>(
            abc, init, is_nan, static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_device<hipblaslt_bf8>(
            abc, init, is_nan, static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_device<int32_t>(
            abc, init, is_nan, static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_device<hipblasLtInt8>(
            abc, init, is_nan, static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case static_cast<hipDataType>(HIP_R_6F_E2M3_EXT):
        hipblaslt_cerr << "hip device initialization does NOT support FP6 yet" << std::endl;
        break;
    case static_cast<hipDataType>(HIP_R_6F_E3M2_EXT):
        hipblaslt_cerr << "hip device initialization does NOT support BF6 yet" << std::endl;
        break;
    case static_cast<hipDataType>(HIP_R_4F_E2M1_EXT):
        hipblaslt_cerr << "hip device initialization does NOT support FP4 yet" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
        break;
    }
}
