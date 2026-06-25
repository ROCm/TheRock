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

#include "d_vector.hpp"

//
// Local declaration of the host strided batch vector.
//
template <typename T>
class host_batch_vector;

//!
//! @brief  pseudo-vector subclass which uses a batch of device memory pointers and
//!  - an array of pointers in host memory
//!  - an array of pointers in device memory
//!
template <typename T>
class device_batch_vector : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_batch_vector(const device_batch_vector&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_batch_vector& operator=(const device_batch_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_batch_vector(size_t n, int64_t inc, int64_t batch_count, bool HMM = false)
        : d_vector<T>(calculate_nmemb(n, inc) * batch_count, HMM)
        // d_vector is a single contiguous block for performance
        , m_n(n)
        , m_inc(inc ? inc : 1)
        , m_nmemb(calculate_nmemb(n, inc))
        , m_batch_count(batch_count)
    {
        if(false == try_initialize_memory())
        {
            free_memory();
            throw std::bad_alloc{};
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~device_batch_vector()
    {
        free_memory();
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
    //! @brief Returns the value of batch_count.
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
        return 0;
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
    //! @brief Copy from a host batched vector.
    //! @param that The host_batch_vector to copy.
    //!
    hipError_t transfer_from(const host_batch_vector<T>& that)
    {
        hipError_t hip_err;
        //
        // Copy each vector.
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
    size_t                     m_n{};
    int64_t                    m_inc{};
    size_t                     m_nmemb{}; // in one batch
    int64_t                    m_batch_count{};
    hipblas_internal_type<T>** m_data{};
    hipblas_internal_type<T>** m_device_data{};

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        // allocate even for zero n
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }

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
                       != (m_data = !this->use_HMM ? (hipblas_internal_type<T>**)calloc(
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
                    success = (hipSuccess
                               == hipMemcpy(m_device_data,
                                            m_data,
                                            sizeof(hipblas_internal_type<T>*) * m_batch_count,
                                            hipMemcpyHostToDevice));
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
                free(m_data);
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
