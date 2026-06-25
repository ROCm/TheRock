/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipblas_test.hpp"

#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <iostream>

#define MEM_MAX_GUARD_PAD 8192

// global for device memory padding
extern size_t g_DVEC_PAD;
void          d_vector_set_pad_length(size_t pad);

//
// Forward declaration of hipblas_init_nan
//
template <typename T>
inline void hipblas_init_nan(T* A, size_t N);

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T>
class d_vector
{
private:
    size_t m_size;
    size_t m_pad, m_guard_len;
    size_t m_bytes;

    static bool m_init_guard;

public:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    static hipblas_internal_type<T> m_guard[MEM_MAX_GUARD_PAD];

#ifdef GOOGLE_TEST
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(hipblas_internal_type<T>))
        , m_bytes((s + m_pad * 2) * sizeof(hipblas_internal_type<T>))
        , use_HMM(HMM)
    {
        // Initialize m_guard with random data
        if(!m_init_guard)
        {
            hipblas_init_nan(m_guard, MEM_MAX_GUARD_PAD);
            m_init_guard = true;
        }
    }
#else
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * sizeof(hipblas_internal_type<T>))
        , m_bytes(s ? s * sizeof(hipblas_internal_type<T>) : sizeof(hipblas_internal_type<T>))
        , use_HMM(HMM)
    {
    }
#endif

    hipblas_internal_type<T>* device_vector_setup()
    {
        hipblas_internal_type<T>* d = nullptr;
        if(use_HMM ? hipMallocManaged(&d, m_bytes) : (hipMalloc)(&d, m_bytes) != hipSuccess)
        {
            std::cout << "Warning: hip can't allocate " << m_bytes << " bytes (" << (m_bytes >> 30)
                      << " GB)" << std::endl;

            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(m_guard_len > 0)
            {
                // Copy m_guard to device memory before allocated memory
                if(hipMemcpy(d, m_guard, m_guard_len, hipMemcpyDefault) != hipSuccess)
                    std::cout << "Error: hipMemcpy pre-guard copy failure." << std::endl;

                // Point to allocated block
                d += m_pad;

                // Copy m_guard to device memory after allocated memory
                if(hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyDefault) != hipSuccess)
                    std::cout << "Error: hipMemcpy post-guard copy failure." << std::endl;
            }
        }
#endif
        return d;
    }

    void device_vector_check(hipblas_internal_type<T>* d)
    {
#ifdef GOOGLE_TEST
        if(m_pad > 0)
        {
            std::vector<hipblas_internal_type<T>> host(m_pad);

            // Copy device memory after allocated memory to host
            if(hipMemcpy(host.data(), d + m_size, m_guard_len, hipMemcpyDefault) != hipSuccess)
                std::cout << "Error: hipMemcpy post-guard copy failure." << std::endl;

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host.data(), m_guard, m_guard_len), 0);

            // Point to m_guard before allocated memory
            d -= m_pad;

            // Copy device memory after allocated memory to host
            if(hipMemcpy(host.data(), d, m_guard_len, hipMemcpyDefault) != hipSuccess)
                std::cout << "Error: hipMemcpy pre-guard copy failure." << std::endl;

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host.data(), m_guard, m_guard_len), 0);
        }
#endif
    }

    void device_vector_teardown(hipblas_internal_type<T>* d)
    {
        if(d != nullptr)
        {
            device_vector_check(d);

            if(m_pad > 0)
                d -= m_pad; // restore to start of alloc

            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

template <typename T>
hipblas_internal_type<T> d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

template <typename T>
bool d_vector<T>::m_init_guard = false;

#undef MEM_MAX_GUARD_PAD
