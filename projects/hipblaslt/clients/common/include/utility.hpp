/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include "hipblaslt_vector.hpp"
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/*!\file
 * \brief provide common utilities
 */

// We use hipblaslt_cout and hipblaslt_cerr instead of std::cout, std::cerr, stdout and stderr,
// for thread-safe IO.
//
// All stdio and std::ostream functions related to stdout and stderr are poisoned, as are
// functions which can create buffer overflows, or which are inherently thread-unsafe.
//
// This must come after the header #includes above, to avoid poisoning system headers.
//
// This is only enabled for hipblaslt-test and hipblaslt-bench.
//
// If you are here because of a poisoned identifier error, here is the rationale for each
// included identifier:
//
// cout, stdout: hipblaslt_cout should be used instead, for thread-safe and atomic line buffering
// cerr, stderr: hipblaslt_cerr should be used instead, for thread-safe and atomic line buffering
// clog: C++ stream which should not be used
// gets: Always unsafe; buffer-overflows; removed from later versions of the language; use fgets
// puts, putchar, fputs, printf, fprintf, vprintf, vfprintf: Use hipblaslt_cout or hipblaslt_cerr
// sprintf, vsprintf: Possible buffer overflows; us snprintf or vsnprintf instead
// strerror: Thread-unsafe; use snprintf / dprintf with %m or strerror_* alternatives
// strsignal: Thread-unsafe; use sys_siglist[signal] instead
// strtok: Thread-unsafe; use strtok_r
// gmtime, ctime, asctime, localtime: Thread-unsafe
// tmpnam: Thread-unsafe; use mkstemp or related functions instead
// putenv: Use setenv instead
// clearenv, fcloseall, ecvt, fcvt: Miscellaneous thread-unsafe functions
// sleep: Might interact with signals by using alarm(); use nanosleep() instead
// abort: Does not abort as cleanly as hipblaslt_abort, and can be caught by a signal handler

#if defined(GOOGLE_TEST) || defined(HIPBLASLT_BENCH)
#undef stdout
#undef stderr
#pragma GCC poison cout cerr clog stdout stderr gets puts putchar fputs fprintf printf sprintf    \
    vfprintf vprintf vsprintf perror strerror strtok gmtime ctime asctime localtime tmpnam putenv \
        clearenv fcloseall ecvt fcvt sleep abort strsignal
#else
// Suppress warnings about hipMalloc(), hipFree() except in hipblaslt-test and hipblaslt-bench
#undef hipMalloc
#undef hipFree
#endif

#define LIMITED_MEMORY_STRING "Error: Attempting to allocate more memory than available."
#define TOO_MANY_DEVICES_STRING "Error: Too many devices requested."
#define HMM_NOT_SUPPORTED "Error: HMM not supported."

// TODO: This is dependent on internal gtest behaviour.
// Compared with result.message() when a test ended. Note that "Succeeded\n" is
// added to the beginning of the message automatically by gtest, so this must be compared.
#define LIMITED_MEMORY_STRING_GTEST "Succeeded\n" LIMITED_MEMORY_STRING
#define TOO_MANY_DEVICES_STRING_GTEST "Succeeded\n" TOO_MANY_DEVICES_STRING
#define HMM_NOT_SUPPORTED_GTEST "Succeeded\n" HMM_NOT_SUPPORTED

inline bool is_bias_enabled(hipblasLtEpilogue_t value_)
{
    switch(value_)
    {
    case HIPBLASLT_EPILOGUE_BIAS:
    case HIPBLASLT_EPILOGUE_GELU_BIAS:
    case HIPBLASLT_EPILOGUE_RELU_BIAS:
    case HIPBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case HIPBLASLT_EPILOGUE_DGELU_BGRAD:
    case HIPBLASLT_EPILOGUE_BGRADA:
    case HIPBLASLT_EPILOGUE_BGRADB:
    case HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT:
    case HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
        return true;
    default:
        return false;
    }
};

enum class hipblaslt_batch_type
{
    none = 0,
    batched
};

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class hipblaslt_local_handle
{
    hipblasLtHandle_t m_handle;
    std::string    m_sol_selec_saved_status = "";
    bool           m_sol_selec_env_set{false};

public:
    hipblaslt_local_handle();

    explicit hipblaslt_local_handle(const Arguments& arg);

    ~hipblaslt_local_handle();

    hipblaslt_local_handle(const hipblaslt_local_handle&)            = delete;
    hipblaslt_local_handle(hipblaslt_local_handle&&)                 = delete;
    hipblaslt_local_handle& operator=(const hipblaslt_local_handle&) = delete;
    hipblaslt_local_handle& operator=(hipblaslt_local_handle&&)      = delete;

    // Allow hipblaslt_local_handle to be used anywhere hipblasLtHandle_t is expected
    operator hipblasLtHandle_t&()
    {
        return m_handle;
    }
    operator const hipblasLtHandle_t&() const
    {
        return m_handle;
    }
    operator hipblasLtHandle_t*()
    {
        return &m_handle;
    }
    operator const hipblasLtHandle_t*() const
    {
        return &m_handle;
    }
};

/* ============================================================================================ */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class hipblaslt_local_matrix_layout
{
    hipblasLtMatrixLayout_t m_descr;
    hipblasStatus_t         m_status  = HIPBLAS_STATUS_NOT_INITIALIZED;
    static constexpr int    alignment = 16;

public:
    hipblaslt_local_matrix_layout(int64_t row, int64_t col, int64_t ld, hipDataType type)
    {
        this->m_status = hipblasLtMatrixLayoutCreate(&this->m_descr, type, row, col, ld);
    }

    ~hipblaslt_local_matrix_layout()
    {
        if(this->m_status == HIPBLAS_STATUS_SUCCESS)
            hipblasLtMatrixLayoutDestroy(this->m_descr);
    }

    hipblaslt_local_matrix_layout(const hipblaslt_local_matrix_layout&)            = delete;
    hipblaslt_local_matrix_layout(hipblaslt_local_matrix_layout&&)                 = delete;
    hipblaslt_local_matrix_layout& operator=(const hipblaslt_local_matrix_layout&) = delete;
    hipblaslt_local_matrix_layout& operator=(hipblaslt_local_matrix_layout&&)      = delete;

    hipblasStatus_t status()
    {
        return m_status;
    }

    // Allow hipblaslt_local_matrix_layout to be used anywhere hipblasLtMatrixLayout_t is expected
    operator hipblasLtMatrixLayout_t&()
    {
        return m_descr;
    }
    operator const hipblasLtMatrixLayout_t&() const
    {
        return m_descr;
    }
    operator hipblasLtMatrixLayout_t*()
    {
        return &m_descr;
    }
    operator const hipblasLtMatrixLayout_t*() const
    {
        return &m_descr;
    }
};

/* ============================================================================================ */
/*! \brief  local matrix multiplication descriptor which is automatically created and destroyed  */
class hipblaslt_local_matmul_descr
{
    hipblasLtMatmulDesc_t m_descr;
    hipblasStatus_t       m_status = HIPBLAS_STATUS_NOT_INITIALIZED;

public:
    hipblaslt_local_matmul_descr(hipblasOperation_t   opA,
                                 hipblasOperation_t   opB,
                                 hipblasComputeType_t compute_type,
                                 hipDataType          scale_type,
                                 hipDataType compute_input_typeA = HIPBLASLT_DATATYPE_INVALID,
                                 hipDataType compute_input_typeB = HIPBLASLT_DATATYPE_INVALID)
    {
        this->m_status = hipblasLtMatmulDescCreate(&this->m_descr, compute_type, scale_type);

        hipblasLtMatmulDescSetAttribute(
            this->m_descr, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(
            this->m_descr, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(int32_t));

        hipblasLtMatmulDescSetAttribute(this->m_descr,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                        &compute_input_typeA,
                                        sizeof(void*));
        hipblasLtMatmulDescSetAttribute(this->m_descr,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                        &compute_input_typeB,
                                        sizeof(void*));
    }

    ~hipblaslt_local_matmul_descr()
    {
        if(this->m_status == HIPBLAS_STATUS_SUCCESS)
            hipblasLtMatmulDescDestroy(this->m_descr);
    }

    hipblaslt_local_matmul_descr(const hipblaslt_local_matmul_descr&)            = delete;
    hipblaslt_local_matmul_descr(hipblaslt_local_matmul_descr&&)                 = delete;
    hipblaslt_local_matmul_descr& operator=(const hipblaslt_local_matmul_descr&) = delete;
    hipblaslt_local_matmul_descr& operator=(hipblaslt_local_matmul_descr&&)      = delete;

    hipblasStatus_t status()
    {
        return m_status;
    }

    // Allow hipblaslt_local_matmul_descr to be used anywhere hipblasLtMatmulDesc_t is expected
    operator hipblasLtMatmulDesc_t&()
    {
        return m_descr;
    }
    operator const hipblasLtMatmulDesc_t&() const
    {
        return m_descr;
    }
    operator hipblasLtMatmulDesc_t*()
    {
        return &m_descr;
    }
    operator const hipblasLtMatmulDesc_t*() const
    {
        return &m_descr;
    }
};

/* ================================================================================================================= */
/*! \brief  local matrix multiplication preference descriptor which is automatically created and destroyed  */
class hipblaslt_local_preference
{
    hipblasLtMatmulPreference_t m_pref;
    hipblasStatus_t             m_status = HIPBLAS_STATUS_NOT_INITIALIZED;

public:
    hipblaslt_local_preference()
    {

        this->m_status = hipblasLtMatmulPreferenceCreate(&this->m_pref);
    }

    ~hipblaslt_local_preference()
    {
        if(this->m_status == HIPBLAS_STATUS_SUCCESS)
            hipblasLtMatmulPreferenceDestroy(this->m_pref);
    }

    hipblaslt_local_preference(const hipblaslt_local_preference&)            = delete;
    hipblaslt_local_preference(hipblaslt_local_preference&&)                 = delete;
    hipblaslt_local_preference& operator=(const hipblaslt_local_preference&) = delete;
    hipblaslt_local_preference& operator=(hipblaslt_local_preference&&)      = delete;

    hipblasStatus_t status()
    {
        return this->m_status;
    }

    operator hipblasLtMatmulPreference_t&()
    {
        return m_pref;
    }
    operator const hipblasLtMatmulPreference_t&() const
    {
        return m_pref;
    }
    operator hipblasLtMatmulPreference_t*()
    {
        return &m_pref;
    }
    operator const hipblasLtMatmulPreference_t*() const
    {
        return &m_pref;
    }
};

/* ============================================================================================ */
/*  device query and print out their ID and name */
int64_t query_device_property(int device_id, hipDeviceProp_t &props);

/*  set current device to device_id */
void set_device(int64_t device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipblaslt sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall time */
double get_time_us_no_sync();

/* ============================================================================================ */
// Return path of this executable
std::string hipblaslt_exepath();

/* ============================================================================================ */
// Temp directory rooted random path
std::string hipblaslt_tempname();

/* ============================================================================================ */
/* Read environment variable */
const char* read_env_var(const char* env_var);

/* ============================================================================================ */
/* Compute strided batched matrix allocation size allowing for strides smaller than full matrix */
size_t strided_batched_matrix_size(int rows, int cols, int lda, int64_t stride, int batch_count);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
inline void hipblaslt_print_matrix(
    std::vector<T> CPU_result, std::vector<T> GPU_result, size_t m, size_t n, size_t lda)
{
    for(size_t i = 0; i < m; i++)
        for(size_t j = 0; j < n; j++)
        {
            hipblaslt_cout << "matrix  col " << i << ", row " << j
                           << ", CPU result=" << CPU_result[j + i * lda]
                           << ", GPU result=" << GPU_result[j + i * lda] << "\n";
        }
}

template <typename T>
void hipblaslt_print_matrix(const char* name, T* A, size_t m, size_t n, size_t lda)
{
    hipblaslt_cout << "---------- " << name << " ----------\n";
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
            hipblaslt_cout << std::setprecision(0) << std::setw(5) << A[i + j * lda] << " ";
        hipblaslt_cout << std::endl;
    }
}

/* ============================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a banded matrix. */
template <typename T>
inline void regular_to_banded(
    bool upper, const T* A, int64_t lda, T* AB, int64_t ldab, int64_t n, int64_t k)
{
    // convert regular hA matrix to banded hAB matrix
    for(int j = 0; j < n; j++)
    {
        int64_t min1 = upper ? std::max(0, static_cast<int>(j - k)) : j;
        int64_t max1 = upper ? j : std::min(static_cast<int>(n - 1), static_cast<int>(j + k));
        int64_t m    = upper ? k - j : -j;

        // Move bands of hA into new banded hAB format.
        for(int i = min1; i <= max1; i++)
            AB[j * ldab + (m + i)] = A[j * lda + i];

        min1 = upper ? k + 1 : std::min(k + 1, n - j);
        max1 = ldab - 1;

        // fill in bottom with random data to ensure we aren't using it.
        // for !upper, fill in bottom right triangle as well.
        for(int i = min1; i <= max1; i++)
            hipblaslt_init<T>(AB + j * ldab + i, 1, 1, 1);

        // for upper, fill in top left triangle with random data to ensure
        // we aren't using it.
        if(upper)
        {
            for(int i = 0; i < m; i++)
                hipblaslt_init<T>(AB + j * ldab + i, 1, 1, 1);
        }
    }
}

/* =============================================================================== */
/*! \brief For testing purposes, zeros out elements not needed in a banded matrix. */
template <typename T>
inline void banded_matrix_setup(bool upper, T* A, int64_t lda, int64_t n, int64_t k)
{
    // Made A a banded matrix with k sub/super-diagonals
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(upper && (j > k + i || i > j))
                A[j * n + i] = T(0);
            else if(!upper && (i > k + j || j > i))
                A[j * n + i] = T(0);
        }
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a packed matrix.                  */
template <typename T>
inline void regular_to_packed(bool upper, const T* A, T* AP, int64_t n)
{
    int index = 0;
    if(upper)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = i; j < n; j++)
            {
                AP[index++] = A[j + i * n];
            }
        }
    }
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    constexpr bool is_int = std::is_same<T, int8_t>();
    using Tp              = std::conditional_t<is_int, int32_t, float>;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    hipblaslt_cout << "---------- " << name << " ----------\n";
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                hipblaslt_cout << static_cast<Tp>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]) << "|";
            }
            hipblaslt_cout << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            hipblaslt_cout << "\n";
    }
    hipblaslt_cout << std::flush;
}

/* ===================================================================== */
/*! \brief For special numerical types, to convert a value to such type. */
template <typename T, typename Accumulator>
typename std::enable_if<std::is_same<int8_t, T>::value, T>::type saturate_cast(Accumulator val)
{
    if constexpr(std::is_same<Accumulator, hipblasLtHalf>::value
                 || std::is_same<Accumulator, hip_bfloat16>::value)
    {
        float tmp = std::nearbyint((float)val); //round to even
        if(tmp > static_cast<float>(127))
            tmp = static_cast<float>(127);
        else if(tmp < static_cast<float>(-128))
            tmp = static_cast<float>(-128);
        return static_cast<T>(tmp);
    }
    else
    {
        if constexpr(std::is_same<Accumulator, float>::value
                     || std::is_same<Accumulator, double>::value)
            val = std::nearbyint(val); //round to even
        if(val > static_cast<Accumulator>(127))
            val = static_cast<Accumulator>(127);
        else if(val < static_cast<Accumulator>(-128))
            val = static_cast<Accumulator>(-128);
        return static_cast<T>(val);
    }
}

/* ==================================================================== */
/*! \brief For common numerical types, to convert a value to such type. */
template <typename T, typename Accumulator>
typename std::enable_if<!std::is_same<int8_t, T>::value, T>::type saturate_cast(Accumulator val)
{
    return static_cast<T>(val);
}

std::vector<void*> benchmark_allocation();
int32_t            hipblaslt_get_arch_major();
void hipblaslt_print_version();

/* ==================================================================== */
/*! \brief write a matrix to file. */
void hipblasltDispatchValuesToFile(hipblasOperation_t transA, hipDataType TiA,
                                   int row, int col, int lda, void *hA,
                                   std::string ADataFile);
