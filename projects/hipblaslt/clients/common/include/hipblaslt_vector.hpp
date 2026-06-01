/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#pragma once

#include "d_vector.hpp"

#include "device_vector.hpp"

#include "host_vector.hpp"

#include "hipblaslt_init.hpp"

//!
//! @brief enum to check for NaN initialization of the Input vector/matrix
//!
typedef enum hipblaslt_check_nan_init_
{
    // Alpha sets NaN
    hipblaslt_client_alpha_sets_nan,

    // Beta sets NaN
    hipblaslt_client_beta_sets_nan,

    //  Never set NaN
    hipblaslt_client_never_set_nan

} hipblaslt_check_nan_init;

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void hipblaslt_init_template(U& that, T rand_gen(), bool seedReset, bool alternating_sign = false)
{
    if(seedReset)
        hipblaslt_seedrand();

    for(int64_t batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto*     batched_data = that[batch_index];
        ptrdiff_t inc          = that.inc();
        auto      n            = that.n();

        if(inc < 0)
            batched_data -= (n - 1) * inc;

        if(alternating_sign)
        {
            for(int64_t i = 0; i < n; ++i)
            {
                auto value            = rand_gen();
                batched_data[i * inc] = (i ^ 0) & 1 ? value : negate(value);
            }
        }
        else
        {
            for(int64_t i = 0; i < n; ++i)
                batched_data[i * inc] = rand_gen();
        }
    }
}

//!
//! @brief Initialize a host_vector with NaNs.
//! @param that The host_vector to be initialized.
//! @param seedReset reset he seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblaslt_init_nan(host_vector<T>& that, bool seedReset = false)
{
    hipblaslt_init_template(that, random_nan_generator<T>, seedReset);
}

//!
//! @brief Initialize a host_vector.
//! @param that The host_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblaslt_init(host_vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init(that, that.size(), 1, 1);
}

//!
//! @brief Initialize a vector.
//! @param that The vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblaslt_init(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_alternating_sign(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_alternating_sign(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_sin(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_sin(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_cos(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_cos<T>(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_hpl(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_hpl(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_alt_impl_big(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_alt_impl_big(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_alt_impl_small(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_alt_impl_small(that.data(), that.size(), 1, 1);
}

template <typename T>
inline void hipblaslt_init_zero(std::vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblaslt_seedrand();
    hipblaslt_init_zero(that.data(), that.size(), 1, 1);
}

//!
//! @brief Initialize a host_vector.
//! @param hx The host_vector.
//! @param arg Specifies the argument class.
//! @param N Length of the host vector.
//! @param incx Increment for the host vector.
//! @param stride_x Incement between the host vector.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize vector with Nan's depending upon the hipblaslt_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblaslt_init_vector(host_vector<T>&          hx,
                                  const Arguments&         arg,
                                  int64_t                  N,
                                  int64_t                  incx,
                                  int64_t                  stride_x,
                                  int64_t                  batch_count,
                                  hipblaslt_check_nan_init nan_init,
                                  bool                     seedReset        = false,
                                  bool                     alternating_sign = false)
{
    if(seedReset)
        hipblaslt_seedrand();

    if(nan_init == hipblaslt_client_alpha_sets_nan && hipblaslt_isnan(arg.alpha))
    {
        hipblaslt_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(nan_init == hipblaslt_client_beta_sets_nan && hipblaslt_isnan(arg.beta))
    {
        hipblaslt_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::hpl)
    {
        if(alternating_sign)
            hipblaslt_init_hpl_alternating_sign(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblaslt_init_hpl(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::rand_int)
    {
        if(alternating_sign)
            hipblaslt_init_alternating_sign(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblaslt_init(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::trig_float)
    {
        if(seedReset)
            hipblaslt_init_cos(hx.data(), 1, N, incx, stride_x, batch_count);
        else
            hipblaslt_init_sin(hx.data(), 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::zero)
    {
        hipblaslt_init_zero(hx, 1, N, incx, stride_x, batch_count);
    }
}

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param M Length of the host matrix.
//! @param N Length of the host matrix.
//! @param lda Leading dimension of the host matrix.
//! @param stride_A Incement between the host matrix.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize matrix with Nan's depending upon the hipblaslt_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblaslt_init_matrix(host_vector<T>&          hA,
                                  const Arguments&         arg,
                                  int64_t                  M,
                                  int64_t                  N,
                                  int64_t                  lda,
                                  int64_t                  stride_A,
                                  int32_t                  batch_count,
                                  hipblaslt_check_nan_init nan_init,
                                  bool                     seedReset        = false,
                                  bool                     alternating_sign = false)
{
    if(seedReset)
        hipblaslt_seedrand();

    if(nan_init == hipblaslt_client_alpha_sets_nan && hipblaslt_isnan(arg.alpha))
    {
        hipblaslt_init_nan(hA, M, N, lda, stride_A, batch_count);
    }
    else if(nan_init == hipblaslt_client_beta_sets_nan && hipblaslt_isnan(arg.beta))
    {
        hipblaslt_init_nan(hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::hpl)
    {
        if(alternating_sign)
            hipblaslt_init_hpl_alternating_sign(hA, M, N, lda, stride_A, batch_count);
        else
            hipblaslt_init_hpl(hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::rand_int)
    {
        if(alternating_sign)
            hipblaslt_init_alternating_sign(hA, M, N, lda, stride_A, batch_count);
        else
            hipblaslt_init(hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::trig_float)
    {
        if(seedReset)
            hipblaslt_init_cos<T>(hA.data(), M, N, lda, stride_A, batch_count);
        else
            hipblaslt_init_sin<T>(hA.data(), M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == hipblaslt_initialization::zero)
    {
        hipblaslt_init_zero(hA, M, N, lda, stride_A, batch_count);
    }
}
