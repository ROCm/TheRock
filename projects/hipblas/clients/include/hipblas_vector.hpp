/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */
#pragma once

#include "hipblas_init.hpp"

//!
//! @brief Initialize a host_strided_batch_vector.
//! @param hx The host_strided_batch_vector.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_strided_batch_vector<T>& hx,
                                const Arguments&              arg,
                                hipblas_client_nan_init       nan_init,
                                bool                          seedReset        = false,
                                bool                          alternating_sign = false)
{
    for(int64_t batch_index = 0; batch_index < hx.batch_count(); ++batch_index)
    {
        auto*   x    = hx[batch_index];
        int64_t incx = hx.inc();
        int64_t N    = hx.n();
        if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
        {
            hipblas_init_vector(random_nan_generator<T>, x, N, incx);
        }
        else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
        {
            hipblas_init_vector(random_nan_generator<T>, x, N, incx);
        }
        else if(arg.initialization == hipblas_initialization::hpl)
        {
            if(alternating_sign)
                hipblas_init_vector_alternating_sign(random_hpl_generator<T>, x, N, incx);
            else
                hipblas_init_vector(random_hpl_generator<T>, x, N, incx);
        }
        else if(arg.initialization == hipblas_initialization::rand_int)
        {
            if(alternating_sign)
                hipblas_init_vector_alternating_sign(random_generator<T>, x, N, incx);
            else
                hipblas_init_vector(random_generator<T>, x, N, incx);
        }
        else if(arg.initialization == hipblas_initialization::trig_float)
        {
            hipblas_init_vector_trig(x, N, incx, seedReset);
        }
    }
}

//!
//! @brief Initialize a host_batch_vector.
//! @param hx The host_batch_vector.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_batch_vector<T>&   hx,
                                const Arguments&        arg,
                                hipblas_client_nan_init nan_init,
                                bool                    seedReset        = false,
                                bool                    alternating_sign = false)
{
    for(int64_t batch_index = 0; batch_index < hx.batch_count(); ++batch_index)
    {
        auto*   x    = hx[batch_index];
        int64_t incx = hx.inc();
        int64_t N    = hx.n();
        if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
        {
            hipblas_init_vector(random_nan_generator<T>, x, N, incx);
        }
        else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
        {
            hipblas_init_vector(random_nan_generator<T>, x, N, incx);
        }
        else if(arg.initialization == hipblas_initialization::hpl)
        {
            if(alternating_sign)
                hipblas_init_vector_alternating_sign(random_hpl_generator<T>, x, N, incx);
            else
                hipblas_init_vector(random_hpl_generator<T>, x, N, incx);
        }
        else if(arg.initialization == hipblas_initialization::rand_int)
        {
            if(alternating_sign)
                hipblas_init_vector_alternating_sign(random_generator<T>, x, N, incx);
            else
                hipblas_init_vector(random_generator<T>, x, N, incx);
        }
        else if(arg.initialization == hipblas_initialization::trig_float)
        {
            hipblas_init_vector_trig(x, N, incx, seedReset);
        }
    }
}

//!
//! @brief Initialize a host_vector.
//! @param hx The host_vector.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_vector<T>&         hx,
                                const Arguments&        arg,
                                hipblas_client_nan_init nan_init,
                                bool                    seedReset        = false,
                                bool                    alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    int64_t N    = hx.n();
    int64_t incx = hx.inc();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_vector(random_nan_generator<T>, (T*)hx, N, incx);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_vector(random_nan_generator<T>, (T*)hx, N, incx);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_vector_alternating_sign(random_hpl_generator<T>, (T*)hx, N, incx);
        else
            hipblas_init_vector(random_hpl_generator<T>, (T*)hx, N, incx);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_vector_alternating_sign(random_generator<T>, (T*)hx, N, incx);
        else
            hipblas_init_vector(random_generator<T>, (T*)hx, N, incx);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_vector_trig((T*)hx, N, incx, seedReset);
    }
}
