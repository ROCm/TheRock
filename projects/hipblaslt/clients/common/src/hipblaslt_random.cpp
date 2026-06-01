/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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

#include "hipblaslt_random.hpp"

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
hipblaslt_rng_t g_hipblaslt_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
std::thread::id g_main_thread_id = std::this_thread::get_id();

// For the main thread, we use g_hipblaslt_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
thread_local hipblaslt_rng_t t_hipblaslt_rng = get_seed();

thread_local int t_hipblaslt_rand_idx;

// length to allow use as bitmask to wraparound
#define RANDLEN 1024
#define RANDWIN 256
#define RANDBUF RANDLEN + RANDWIN
static thread_local int    t_rand_init = 0;
static thread_local float  t_rand_f_array[RANDBUF];
static thread_local double t_rand_d_array[RANDBUF];

/* ============================================================================================ */

float hipblaslt_uniform_int_1_10()
{
    if(!t_rand_init)
    {
        for(int i = 0; i < RANDBUF; i++)
        {
            t_rand_f_array[i]
                = (float)std::uniform_int_distribution<unsigned>(1, 10)(t_hipblaslt_rng);
            t_rand_d_array[i] = (double)t_rand_f_array[i];
        }
        t_rand_init = 1;
    }
    t_hipblaslt_rand_idx = (t_hipblaslt_rand_idx + 1) & (RANDLEN - 1);
    return t_rand_f_array[t_hipblaslt_rand_idx];
}

inline int pseudo_rand_ptr_offset()
{
    t_hipblaslt_rand_idx = (t_hipblaslt_rand_idx + 1) & (RANDWIN - 1);
    return t_hipblaslt_rand_idx;
}

void hipblaslt_uniform_int_1_10_run_float(float* ptr, size_t num)
{
    if(!t_rand_init)
        hipblaslt_uniform_int_1_10();

    for(size_t i = 0; i < num; i += RANDLEN)
    {
        float* rptr = t_rand_f_array + pseudo_rand_ptr_offset();
        size_t n    = i + RANDLEN < num ? RANDLEN : num - i;
        memcpy(ptr, rptr, sizeof(float) * n);
        ptr += RANDLEN;
    }
}

void hipblaslt_uniform_int_1_10_run_double(double* ptr, size_t num)
{
    if(!t_rand_init)
        hipblaslt_uniform_int_1_10();

    for(size_t i = 0; i < num; i += RANDLEN)
    {
        double* rptr = t_rand_d_array + pseudo_rand_ptr_offset();
        size_t  n    = i + RANDLEN < num ? RANDLEN : num - i;
        memcpy(ptr, rptr, sizeof(double) * n);
        ptr += RANDLEN;
    }
}

#undef RANDLEN
#undef RANDWIN
#undef RANDBUF
