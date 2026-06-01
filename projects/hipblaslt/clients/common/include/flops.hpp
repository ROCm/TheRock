/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 1, 2,
 * 3. Where possible we are using the values of NOP from the legacy BLAS files [sdcz]blas[23]time.f
 * for flop count.
 */

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(int64_t m, int64_t n, int64_t k)
{
    return (2.0 * m * n * k) / 1e9;
}

constexpr double gemm_gflop_count(int64_t m, int64_t n, int64_t k, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return gemm_gflop_count<float>(m, n, k);
    case HIP_R_64F:
        return gemm_gflop_count<double>(m, n, k);
    case HIP_R_16F:
        return gemm_gflop_count<hipblasLtHalf>(m, n, k);
    case HIP_R_32I:
        return gemm_gflop_count<int32_t>(m, n, k);
    default:
        hipblaslt_cerr << "Error type in gemm_gflop_count()" << std::endl;
        return gemm_gflop_count<float>(m, n, k);
    }
}

template <typename T>
constexpr double sigmoid_gflop_count(int64_t m, int64_t n)
{
    return (4 * m * n) / 1e9;
}

constexpr double sigmoid_gflop_count(int64_t m, int64_t n, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return sigmoid_gflop_count<float>(m, n);
    case HIP_R_64F:
        return sigmoid_gflop_count<double>(m, n);
    case HIP_R_16F:
        return sigmoid_gflop_count<hipblasLtHalf>(m, n);
    case HIP_R_32I:
        return sigmoid_gflop_count<int32_t>(m, n);
    default:
        hipblaslt_cerr << "Error type in sigmoid_gflop_count()" << std::endl;
        return sigmoid_gflop_count<float>(m, n);
    }
}

template <typename T>
constexpr double relu_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 1e9;
}

constexpr double relu_gflop_count(int64_t m, int64_t n, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return relu_gflop_count<float>(m, n);
    case HIP_R_64F:
        return relu_gflop_count<double>(m, n);
    case HIP_R_16F:
        return relu_gflop_count<hipblasLtHalf>(m, n);
    case HIP_R_32I:
        return relu_gflop_count<int32_t>(m, n);
    default:
        hipblaslt_cerr << "Error type in relu_gflop_count()" << std::endl;
        return relu_gflop_count<float>(m, n);
    }
}

template <typename T>
constexpr double gelu_gflop_count(int64_t m, int64_t n)
{
    return (9.0 * m * n) / 1e9;
}

constexpr double gelu_gflop_count(int64_t m, int64_t n, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return gelu_gflop_count<float>(m, n);
    case HIP_R_64F:
        return gelu_gflop_count<double>(m, n);
    case HIP_R_16F:
        return gelu_gflop_count<hipblasLtHalf>(m, n);
    case HIP_R_32I:
        return gelu_gflop_count<int32_t>(m, n);
    default:
        hipblaslt_cerr << "Error type in gelu_gflop_count()" << std::endl;
        return gelu_gflop_count<float>(m, n);
    }
}

template <typename T>
constexpr double silu_gflop_count(int64_t m, int64_t n)
{
    return (5.0 * m * n) / 1e9;
}

constexpr double silu_gflop_count(int64_t m, int64_t n, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return silu_gflop_count<float>(m, n);
    case HIP_R_64F:
        return silu_gflop_count<double>(m, n);
    case HIP_R_16F:
        return silu_gflop_count<hipblasLtHalf>(m, n);
    case HIP_R_32I:
        return silu_gflop_count<int32_t>(m, n);
    default:
        hipblaslt_cerr << "Error type in silu_gflop_count()" << std::endl;
        return silu_gflop_count<float>(m, n);
    }
}

template <typename T>
constexpr double clamp_gflop_count(int64_t m, int64_t n)
{
    return (2.0 * m * n) / 1e9;
}

constexpr double clamp_gflop_count(int64_t m, int64_t n, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return clamp_gflop_count<float>(m, n);
    case HIP_R_64F:
        return clamp_gflop_count<double>(m, n);
    case HIP_R_16F:
        return clamp_gflop_count<hipblasLtHalf>(m, n);
    case HIP_R_32I:
        return clamp_gflop_count<int32_t>(m, n);
    default:
        hipblaslt_cerr << "Error type in clamp_gflop_count()" << std::endl;
        return clamp_gflop_count<float>(m, n);
    }
}
