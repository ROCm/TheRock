/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string.h>

#include "host_alloc.hpp"

//
// Local declaration of the device batch Matrix.
//
template <typename T>
class device_batch_matrix;

//!
//! @brief Implementation of the batch Matrix on host.
//!
template <typename T>
class host_batch_matrix
{
public:
    //!
    //! @brief Delete copy constructor.
    //!
    host_batch_matrix(const host_batch_matrix<T>& that) = delete;

    //!
    //! @brief Delete copy assignment.
    //!
    host_batch_matrix& operator=(const host_batch_matrix<T>& that) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_matrix(size_t m, size_t n, size_t lda, int64_t batch_count)
        : m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_nmemb(n * lda)
        , m_batch_count(batch_count)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
            throw std::bad_alloc{};
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_batch_matrix()
    {
        this->free_memory();
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return m_lda;
    }

    //!
    //! @brief Returns nmemb.
    //!
    size_t nmemb() const
    {
        return m_nmemb;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return m_batch_count;
    }

    //!
    //! @brief Random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](int64_t batch_index)
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Constant random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The non-mutable pointer.
    //!
    const T* operator[](int64_t batch_index) const
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Cast to a double pointer.
    //!
    // clang-format off
    operator T**()
    // clang-format on
    {
        return m_data;
    }

    //!
    //! @brief Constant cast to a double pointer.
    //!
    operator const T* const *()
    {
        return m_data;
    }

    //!
    //! @brief Copy from a host batched Matrix.
    //! @param that the Matrix the data is copied from.
    //! @return true if the copy is done successfully, false otherwise.
    //!
    bool copy_from(const host_batch_matrix<T>& that)
    {
        if((this->batch_count() == that.batch_count()) && (this->m() == that.m())
           && (this->n() == that.n()) && (this->lda() == that.lda()))
        {
            size_t num_bytes = m_nmemb * sizeof(T) * m_batch_count;
            if(m_batch_count > 0)
                memcpy((*this)[0], that[0], num_bytes);
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer from a device batched Matrix.
    //! @param that the Matrix the data is copied from.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_batch_matrix<T>& that)
    {
        hipError_t hip_err;
        size_t     num_bytes = m_nmemb * sizeof(T) * m_batch_count;
        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        hipMemcpyKind kind = that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost;

        if(m_batch_count > 0)
        {
            if(hipSuccess != (hip_err = hipMemcpy((*this)[0], that[0], num_bytes, kind)))
            {
                return hip_err;
            }
        }
        return hipSuccess;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return (nullptr != m_data) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t  m_m{};
    size_t  m_n{};
    size_t  m_lda{};
    size_t  m_nmemb{};
    int64_t m_batch_count{};
    T**     m_data{};

    bool try_initialize_memory()
    {
        bool success = (nullptr != (m_data = (T**)host_calloc(m_batch_count, sizeof(T*))));
        if(success)
        {
            for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
            {
                if(batch_index == 0)
                {
                    success = (nullptr
                               != (m_data[batch_index]
                                   = (T*)host_calloc(m_nmemb * m_batch_count, sizeof(T))));
                    if(false == success)
                    {
                        break;
                    }
                }
                else
                {
                    m_data[batch_index] = m_data[0] + batch_index * m_nmemb;
                }
            }
        }
        return success;
    }

    void free_memory()
    {
        if(nullptr != m_data)
        {
            for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
            {
                if(batch_index == 0 && nullptr != m_data[batch_index])
                {
                    free(m_data[batch_index]);
                    m_data[batch_index] = nullptr;
                }
                else
                {
                    m_data[batch_index] = nullptr;
                }
            }

            free(m_data);
            m_data = nullptr;
        }
    }
};
