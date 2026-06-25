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
// Local declaration of the host strided batch matrix.
//
template <typename T>
class host_strided_batch_matrix;

//!
//! @brief Implementation of a strided batched matrix on device.
//!
template <typename T>
class device_strided_batch_matrix : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_strided_batch_matrix(const device_strided_batch_matrix&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_strided_batch_matrix& operator=(const device_strided_batch_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_strided_batch_matrix(
        size_t m, size_t n, size_t lda, hipblasStride stride, int64_t batch_count, bool HMM = false)
        : d_vector<T>(calculate_nmemb(n, lda, stride, batch_count), HMM)
        , m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_stride(stride)
        , m_batch_count(batch_count)
    {
        bool valid_parameters = calculate_nmemb(n, lda, stride, batch_count) > 0;
        if(valid_parameters)
        {
            this->m_data = this->device_vector_setup();
            if(!this->m_data)
                throw std::bad_alloc{};
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~device_strided_batch_matrix()
    {
        if(nullptr != this->m_data)
        {
            this->device_vector_teardown(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    hipblas_internal_type<T>* data()
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const hipblas_internal_type<T>* data() const
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Returns the stride value.
    //!
    hipblasStride stride() const
    {
        return this->m_stride;
    }

    //!
    //! @brief Returns pointer.
    //! @param batch_index The batch index.
    //! @return A mutable pointer to the batch_index'th matrix.
    //!
    hipblas_internal_type<T>* operator[](int64_t batch_index)
    {
        return (this->m_stride >= 0)
                   ? this->m_data + batch_index * this->m_stride
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param batch_index The batch index.
    //! @return A non-mutable mutable pointer to the batch_index'th matrix.
    //!
    const hipblas_internal_type<T>* operator[](int64_t batch_index) const
    {
        return (this->m_stride >= 0)
                   ? this->m_data + batch_index * this->m_stride
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first matrix.
    //!
    operator hipblas_internal_type<T>*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first matrix.
    //!
    operator const hipblas_internal_type<T>*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Transfer data from a strided batched matrix on device.
    //! @param that That strided batched matrix on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_matrix<T>& that)
    {
        return hipMemcpy(this->data(),
                         (hipblas_internal_type<T>*)that.data(),
                         sizeof(hipblas_internal_type<T>) * this->nmemb(),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    //!
    //! @brief Broadcast data from one matrix on host to each batch_count matrices.
    //! @param that That matrix on host.
    //! @return The hip error.
    //!
    hipError_t broadcast_one_matrix_from(const host_matrix<T>& that)
    {
        hipError_t status = hipSuccess;
        for(int64_t batch_index = 0; batch_index < m_batch_count; batch_index++)
        {
            status = hipMemcpy(this->data() + (batch_index * m_stride),
                               (hipblas_internal_type<T>*)that.data(),
                               sizeof(hipblas_internal_type<T>) * this->m_n * this->m_lda,
                               this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
            if(status != hipSuccess)
                break;
        }
        return status;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        bool valid_parameters = calculate_nmemb(m_n, m_lda, m_stride, m_batch_count) > 0;

        if(*this || !valid_parameters)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t                    m_m{};
    size_t                    m_n{};
    size_t                    m_lda{};
    hipblasStride             m_stride{};
    int64_t                   m_batch_count{};
    hipblas_internal_type<T>* m_data{};

    static size_t calculate_nmemb(size_t n, size_t lda, hipblasStride stride, int64_t batch_count)
    {
        return lda * n + size_t(batch_count - 1) * std::abs(stride);
    }
};
