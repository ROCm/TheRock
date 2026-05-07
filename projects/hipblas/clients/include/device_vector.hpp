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

#include "d_vector.hpp"

//
// Forward declaration of the host vector.
//
template <typename T>
class host_vector;

//!
//! @brief pseudo-vector subclass which uses device memory
//!
template <typename T>
class device_vector : public d_vector<T>
{

public:
    //!
    //! @brief Disallow copying.
    //!
    device_vector(const device_vector&) = delete;

    //!
    //! @brief Disallow assigning
    //!
    device_vector& operator=(const device_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n The length of the vector.
    //! @param inc Element index increment. If zero treated as one.
    //! @param HMM HipManagedMemory Flag.
    //!
    explicit device_vector(size_t n, int64_t inc = 1, bool HMM = false)
        : d_vector<T>{calculate_nmemb(n, inc), HMM}
        , m_n{n}
        , m_inc{inc ? inc : 1}
        , m_data{this->device_vector_setup()}
    {
        if(!m_data)
            throw std::bad_alloc{};
    }

    //!
    //! @brief Destructor.
    //!
    ~device_vector()
    {
        this->device_vector_teardown(m_data);
        m_data = nullptr;
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    int64_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    int64_t batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    hipblasStride stride() const
    {
        return 0;
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected.
    //!
    operator hipblas_internal_type<T>*()
    {
        return m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const hipblas_internal_type<T>*() const
    {
        return m_data;
    }

    //!
    //! @brief Transfer data from a host vector.
    //! @param that The host vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_vector<T>& that)
    {
        return hipMemcpy(m_data,
                         (const hipblas_internal_type<T>*)that.data(),
                         this->nmemb() * sizeof(hipblas_internal_type<T>),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return !this->nmemb() || m_data ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t                    m_n{};
    int64_t                   m_inc{};
    hipblas_internal_type<T>* m_data{};

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        // alllocate when n is zero
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }
};
