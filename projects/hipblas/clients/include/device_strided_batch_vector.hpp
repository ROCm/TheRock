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
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_strided_batch_vector;

//!
//! @brief Implementation of a strided batched vector on device.
//!
template <typename T>
class device_strided_batch_vector : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_strided_batch_vector(const device_strided_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_strided_batch_vector& operator=(const device_strided_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n   The length of the vector.
    //! @param inc The increment.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_strided_batch_vector(
        size_t n, int64_t inc, hipblasStride stride, int64_t batch_count, bool HMM = false)
        : d_vector<T>(calculate_nmemb(n, inc, stride, batch_count), HMM)
        , m_n(n)
        , m_inc(inc ? inc : 1)
        , m_stride(stride)
        , m_batch_count(batch_count)
    {
        m_data = this->device_vector_setup();
        if(!m_data)
            throw std::bad_alloc{};
    }

    //!
    //! @brief Destructor.
    //!
    ~device_strided_batch_vector()
    {
        if(nullptr != m_data)
        {
            this->device_vector_teardown(m_data);
            m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    hipblas_internal_type<T>* data()
    {
        return m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const hipblas_internal_type<T>* data() const
    {
        return m_data;
    }

    //!
    //! @brief Returns the length.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment.
    //!
    int64_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return m_batch_count;
    }

    //!
    //! @brief Returns the stride value.
    //!
    hipblasStride stride() const
    {
        return m_stride;
    }

    //!
    //! @brief Returns pointer.
    //! @param batch_index The batch index.
    //! @return A mutable pointer to the batch_index'th vector.
    //!
    hipblas_internal_type<T>* operator[](int64_t batch_index)
    {
        return (m_stride >= 0) ? m_data + batch_index * m_stride
                               : m_data + (batch_index + 1 - m_batch_count) * m_stride;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param batch_index The batch index.
    //! @return A non-mutable mutable pointer to the batch_index'th vector.
    //!
    const hipblas_internal_type<T>* operator[](int64_t batch_index) const
    {
        return (m_stride >= 0) ? m_data + batch_index * m_stride
                               : m_data + (batch_index + 1 - m_batch_count) * m_stride;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first vector.
    //!
    operator hipblas_internal_type<T>*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first vector.
    //!
    operator const T*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != m_data;
    }

    //!
    //! @brief Transfer data from a strided batched vector on device.
    //! @param that That strided batched vector on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_vector<T>& that)
    {
        return hipMemcpy(data(),
                         (hipblas_internal_type<T>*)that.data(),
                         sizeof(hipblas_internal_type<T>) * this->nmemb(),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        if(*this)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t                    m_n{};
    int64_t                   m_inc{};
    hipblasStride             m_stride{};
    int64_t                   m_batch_count{};
    hipblas_internal_type<T>* m_data{};

    static size_t calculate_nmemb(size_t n, int64_t inc, hipblasStride stride, int64_t batch_count)
    {
        // allocate even for zero n and batch_count
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1)
               + size_t((batch_count > 0 ? batch_count : 1) - 1) * std::abs(stride);
    }
};
