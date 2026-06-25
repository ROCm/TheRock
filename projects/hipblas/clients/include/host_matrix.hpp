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

#include <cmath>
#include <type_traits>
#include <vector>

#include "device_matrix.hpp"
#include "host_alloc.hpp"

//!
//! @brief  Pseudo-matrix subclass which uses host memory.
//!
template <typename T>
struct host_matrix : std::vector<T, host_memory_allocator<T>>
{
    // Inherit constructors
    using std::vector<T, host_memory_allocator<T>>::vector;

    //!
    //! @brief Constructor.
    //!
    host_matrix(size_t m, size_t n, size_t lda)
        : std::vector<T, host_memory_allocator<T>>(n * lda)
        , m_m(m)
        , m_n(n)
        , m_lda(lda)
    {
    }

    //!
    //! @brief Copy constructor from host_matrix of other types convertible to T
    //!
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    host_matrix(const host_matrix<U>& x)
        : std::vector<T, host_memory_allocator<T>>(x.size())
        , m_m(x.size())
        , m_n(1)
        , m_lda(1)
    {
        for(size_t i = 0; i < m_m; ++i)
            (*this)[i] = x[i];
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device matrix.
    //! @param  that That device matrix.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_matrix<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(*this,
                         that,
                         sizeof(T) * this->size(),
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
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
    //! @brief Returns the batch count (always 1).
    //!
    static constexpr int64_t batch_count()
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    static constexpr hipblasStride stride()
    {
        return 0;
    }

    //!
    //! @brief Random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](int64_t batch_index)
    {
        return this->data();
    }

    //!
    //! @brief Constant random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The non-mutable pointer.
    //!
    const T* operator[](int64_t batch_index) const
    {
        return this->data();
    }

    //!
    //! @brief Check if memory exists (out of context, always hipSuccess)
    //!
    static constexpr hipError_t memcheck()
    {
        return hipSuccess;
    }

private:
    size_t m_m   = 0;
    size_t m_n   = 0;
    size_t m_lda = 0;
};
