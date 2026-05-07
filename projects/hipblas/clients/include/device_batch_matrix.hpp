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
#include "host_alloc.hpp"

//
// Local declaration of the host batch matrix.
//
template <typename T>
class host_batch_matrix;

//!
//! @brief  pseudo-matrix subclass which uses a batch of device memory pointers and
//!  - an array of pointers in host memory
//!  - an array of pointers in device memory
//!
template <typename T>
class device_batch_matrix : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_batch_matrix(const device_batch_matrix&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_batch_matrix& operator=(const device_batch_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //! @param offset      The offset to the memory of each Matrix as held by device_data.
    //!
    explicit device_batch_matrix(
        size_t m, size_t n, size_t lda, int64_t batch_count, bool HMM = false, size_t offset = 0)
        : d_vector<T>(n * lda * batch_count, HMM) // d_vector is single block for all batches
        , m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_nmemb(n * lda)
        , m_batch_count(batch_count)
        , m_offset(HMM ? 0 : offset)
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
    ~device_batch_matrix()
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
    //! @brief Returns the value of batch_count.
    //!
    int64_t batch_count() const
    {
        return m_batch_count;
    }

    //!
    //! @brief Returns the value of offset.
    //!
    int64_t offset() const
    {
        return m_offset;
    }

    //!
    //! @brief Access to device data.
    //! @return Pointer to the device data.
    //!
    hipblas_internal_type<T>** ptr_on_device()
    {
        return m_device_data;
    }

    //!
    //! @brief Const access to device data.
    //! @return Const pointer to the device data.
    //!
    const hipblas_internal_type<T>* const* ptr_on_device() const
    {
        return m_device_data;
    }

    //!
    //! @brief access to device data.
    //! @return Const pointer to the device data.
    //!
    hipblas_internal_type<T>* const* const_batch_ptr()
    {
        return m_device_data;
    }

    //!
    //! @brief Random access.
    //! @param batch_index The batch index.
    //! @return Pointer to the array on device.
    //!
    hipblas_internal_type<T>* operator[](int64_t batch_index)
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Constant random access.
    //! @param batch_index The batch index.
    //! @return Constant pointer to the array on device.
    //!
    const hipblas_internal_type<T>* operator[](int64_t batch_index) const
    {

        return m_data[batch_index];
    }

    //!
    //! @brief Const cast of the data on host.
    //!
    operator const hipblas_internal_type<T>* const *() const
    {
        return m_data;
    }

    //!
    //! @brief Cast of the data on host.
    //!
    // clang-format off
    operator hipblas_internal_type<T>**()
    // clang-format on
    {
        return m_data;
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != m_data;
    }

    //!
    //! @brief Copy from a host batched matrix.
    //! @param that The host_batch_matrix to copy.
    //!
    hipError_t transfer_from(const host_batch_matrix<T>& that)
    {
        hipError_t hip_err;
        //
        // Copy each matrix.
        //
        hipMemcpyKind kind = this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice;
        if(m_batch_count > 0)
        {
            if(hipSuccess
               != (hip_err = hipMemcpy((*this)[0],
                                       (hipblas_internal_type<T>*)that[0],
                                       sizeof(hipblas_internal_type<T>) * m_nmemb * m_batch_count,
                                       kind)))
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
        if(*this)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t                     m_m{};
    size_t                     m_n{};
    size_t                     m_lda{};
    size_t                     m_nmemb{};
    int64_t                    m_batch_count{};
    size_t                     m_offset{};
    hipblas_internal_type<T>** m_data{};
    hipblas_internal_type<T>** m_device_data{};

    //!
    //! @brief Try to allocate the resources.
    //! @return true if success false otherwise.
    //!
    bool try_initialize_memory()
    {
        bool success = false;

        success = (hipSuccess
                   == (!this->use_HMM
                           ? (hipMalloc)(&m_device_data,
                                         m_batch_count * sizeof(hipblas_internal_type<T>*))
                           : hipMallocManaged(&m_device_data,
                                              m_batch_count * sizeof(hipblas_internal_type<T>*))));
        if(success)
        {
            success = (nullptr
                       != (m_data = !this->use_HMM ? (hipblas_internal_type<T>**)host_calloc(
                                        m_batch_count, sizeof(hipblas_internal_type<T>*))
                                                   : m_device_data));
            if(success)
            {
                for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
                {
                    if(batch_index == 0)
                    {
                        success = (nullptr != (m_data[batch_index] = this->device_vector_setup()));
                        if(!success)
                        {
                            break;
                        }
                    }
                    else
                    {
                        m_data[batch_index] = m_data[0] + batch_index * m_nmemb;
                    }
                }

                if(success && !this->use_HMM)
                {
                    if(m_offset)
                    {
                        for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
                            m_data[batch_index] += m_offset;
                    }

                    success = (hipSuccess
                               == hipMemcpy(m_device_data,
                                            m_data,
                                            sizeof(hipblas_internal_type<T>*) * m_batch_count,
                                            hipMemcpyHostToDevice));

                    if(m_offset)
                    {
                        // don't want to deal with offset with m_data, just m_device_data.
                        for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
                            m_data[batch_index] -= m_offset;
                    }
                }
            }
        }
        return success;
    }

    //!
    //! @brief Free the resources, as much as we can.
    //!
    void free_memory()
    {
        if(nullptr != m_data)
        {
            for(int64_t batch_index = 0; batch_index < m_batch_count; ++batch_index)
            {
                if(batch_index == 0 && nullptr != m_data[batch_index])
                {
                    this->device_vector_teardown(m_data[batch_index]);
                    m_data[batch_index] = nullptr;
                }
                else
                {
                    m_data[batch_index] = nullptr;
                }
            }

            if(!this->use_HMM)
            {
                host_free(m_data);
            }
            // else this is just a copy of m_device_data

            m_data = nullptr;
        }

        if(nullptr != m_device_data)
        {
            auto tmp_device_data = m_device_data;
            m_device_data        = nullptr;
            CHECK_HIP_ERROR((hipFree)(tmp_device_data));
        }
    }
};
