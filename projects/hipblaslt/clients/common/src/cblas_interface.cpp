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
#include "cblas_interface.hpp"
#include "datatype_interface.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"
#include <bitset>
#include <iostream>
#include <omp.h>

CBLAS_TRANSPOSE HIPOperationToCBLASTanspose(hipblasOperation_t trans)
{
    switch(trans)
    {
    case HIPBLAS_OP_N:
        return CblasNoTrans;
    case HIPBLAS_OP_T:
        return CblasTrans;
    case HIPBLAS_OP_C:
        return CblasConjTrans;
    }
}

template <typename T>
class customVector
{
public:
    void initialize(size_t size)
    {
        m_data.resize(size);
        m_pointer = m_data.data();
    }
    void initialize(const void* buffer)
    {
        m_pointer = const_cast<void*>(buffer);
    }

    operator T*()
    {
        return (T*)m_pointer;
    }

    operator const T*() const
    {
        return (const T*)m_pointer;
    }

    T& operator[](std::size_t i)
    {
        return ((T*)m_pointer)[i];
    }

private:
    std::vector<T> m_data;
    void*          m_pointer = nullptr;
};

template <typename TD, typename TcCast, typename Tc>
void sat_cast_mul(TD* dst, customVector<TcCast>& src, Tc scale, size_t size)
{
    if constexpr(std::is_same<TcCast, float>::value
                 || (!std::is_same<TD, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TD, hipblaslt_f8_fnuz>::value))
    {
        if(scale != 1)
        {
            for(size_t i = 0; i < size; i++)
                dst[i] = saturate_cast<TD>(src[i] * scale);
        }
        else
        {
            for(size_t i = 0; i < size; i++)
                dst[i] = saturate_cast<TD>(src[i]);
        }
    }
}

template <typename TcCast, typename Tc>
void sat_cast_mul(void* dst, hipDataType typeD, customVector<TcCast>& src, Tc scale, size_t size)
{
    switch(typeD)
    {
    case HIP_R_32F:
        sat_cast_mul<float, TcCast, Tc>(static_cast<float*>(dst), src, scale, size);
        break;
    case HIP_R_64F:
        sat_cast_mul<double, TcCast, Tc>(static_cast<double*>(dst), src, scale, size);
        break;
    case HIP_R_16F:
        sat_cast_mul<hipblasLtHalf, TcCast, Tc>(static_cast<hipblasLtHalf*>(dst), src, scale, size);
        break;
    case HIP_R_16BF:
        sat_cast_mul<hip_bfloat16, TcCast, Tc>(static_cast<hip_bfloat16*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        sat_cast_mul<hipblaslt_f8_fnuz, TcCast, Tc>(
            static_cast<hipblaslt_f8_fnuz*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        sat_cast_mul<hipblaslt_bf8_fnuz, TcCast, Tc>(
            static_cast<hipblaslt_bf8_fnuz*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E4M3:
        sat_cast_mul<hipblaslt_f8, TcCast, Tc>(static_cast<hipblaslt_f8*>(dst), src, scale, size);
        break;
    case HIP_R_8F_E5M2:
        sat_cast_mul<hipblaslt_bf8, TcCast, Tc>(static_cast<hipblaslt_bf8*>(dst), src, scale, size);
        break;
    case HIP_R_32I:
        sat_cast_mul<int32_t, TcCast, Tc>(static_cast<int32_t*>(dst), src, scale, size);
        break;
    case HIP_R_8I:
        sat_cast_mul<hipblasLtInt8, TcCast, Tc>(static_cast<hipblasLtInt8*>(dst), src, scale, size);
        break;
    default:
        hipblaslt_cerr << "Error type in sat_cast_mul" << std::endl;
        break;
    }
}

template <typename TcCast, typename TiA>
void cast_mul(customVector<TcCast>& dst, const TiA* src, size_t size)
{
    if constexpr(std::is_same<TcCast, float>::value
                 || (!std::is_same<TiA, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TiA, hipblaslt_f8_fnuz>::value))
    {
        if constexpr(std::is_same<TcCast, float>::value
                     || !(std::is_same<TiA, hipblaslt_bf8>::value
                          || std::is_same<TiA, hipblaslt_f8>::value))
            for(size_t i = 0; i < size; i++)
            {
                dst[i] = static_cast<TcCast>(src[i]);
            }
    }
}

template <typename TcCast>
void cast_mul(customVector<TcCast>& dst, const void* src, hipDataType TiA, size_t size)
{
    switch(TiA)
    {
    case HIP_R_32F:
        cast_mul<TcCast, float>(dst, static_cast<const float*>(src), size);
        break;
    case HIP_R_64F:
        cast_mul<TcCast, double>(dst, static_cast<const double*>(src), size);
        break;
    case HIP_R_16F:
        cast_mul<TcCast, hipblasLtHalf>(dst, static_cast<const hipblasLtHalf*>(src), size);
        break;
    case HIP_R_16BF:
        cast_mul<TcCast, hip_bfloat16>(dst, static_cast<const hip_bfloat16*>(src), size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul<TcCast, hipblaslt_f8_fnuz>(dst, static_cast<const hipblaslt_f8_fnuz*>(src), size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul<TcCast, hipblaslt_bf8_fnuz>(
            dst, static_cast<const hipblaslt_bf8_fnuz*>(src), size);
        break;
    case HIP_R_8F_E4M3:
        cast_mul<TcCast, hipblaslt_f8>(dst, static_cast<const hipblaslt_f8*>(src), size);
        break;
    case HIP_R_8F_E5M2:
        cast_mul<TcCast, hipblaslt_bf8>(dst, static_cast<const hipblaslt_bf8*>(src), size);
        break;
    case HIP_R_32I:
        cast_mul<TcCast, int32_t>(dst, static_cast<const int32_t*>(src), size);
        break;
    case HIP_R_8I:
        cast_mul<TcCast, hipblasLtInt8>(dst, static_cast<const hipblasLtInt8*>(src), size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul" << std::endl;
        break;
    }
}

template <typename TcCast, typename Tc, typename TiA>
void cast_mul(customVector<TcCast>& dst,
              const TiA*            A,
              bool                  isScaleAVec,
              const TcCast*         scaleAVec,
              const TcCast*         AlphaVec,
              bool                  transA,
              int64_t               m,
              int64_t               k,
              size_t                size,
              bool                  isMXFormat = false)
{
    if constexpr((std::is_same<TcCast, float>::value)
                 || (!std::is_same<TiA, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TiA, hipblaslt_f8_fnuz>::value))
    {
        if constexpr(std::is_same<TcCast, float>::value
                     || !(std::is_same<TiA, hipblaslt_bf8>::value
                          || std::is_same<TiA, hipblaslt_f8>::value))
        {
            if(AlphaVec != nullptr)
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        if(isMXFormat)
                        {
                            dst[i] = static_cast<TcCast>(A[i]) * AlphaVec[i % m];
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                            dst[i]      = static_cast<TcCast>(A[i]) * scaleA * AlphaVec[i % m];
                        }
                    }
                } // transA
                else
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        if(isMXFormat)
                        {
                            dst[i] = static_cast<TcCast>(A[i]) * AlphaVec[i / k];
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                            dst[i]      = static_cast<TcCast>(A[i]) * scaleA * AlphaVec[i / k];
                        }
                    }
                }
            } // AlphaVec != nullptr
            else
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        if(isMXFormat)
                        {
                            dst[i] = static_cast<TcCast>(A[i]);
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                            dst[i]      = static_cast<TcCast>(A[i] * scaleA);
                        }
                    }
                }
                else // not transA
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        if(isMXFormat)
                        {
                            dst[i] = static_cast<TcCast>(A[i]);
                        }
                        else
                        {
                            auto scaleA = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                            dst[i]      = static_cast<TcCast>(A[i] * scaleA);
                        }
                    }
                }
            }
        }
    }
}

template <typename TcCast, typename Tc>
void cast_mul(customVector<TcCast>& dst,
              const void*           src,
              hipDataType           TiA,
              bool                  isScaleAVec,
              const TcCast*         scaleAVec,
              const TcCast*         AlphaVec,
              bool                  transA,
              int64_t               m,
              int64_t               k,
              size_t                size,
              bool                  isMXFormat = false)
{
    switch(TiA)
    {
    case HIP_R_32F:
        cast_mul<TcCast, Tc, float>(dst,
                                    static_cast<const float*>(src),
                                    isScaleAVec,
                                    scaleAVec,
                                    AlphaVec,
                                    transA,
                                    m,
                                    k,
                                    size,
                                    isMXFormat);
        break;
    case HIP_R_64F:
        cast_mul<TcCast, Tc, double>(dst,
                                     static_cast<const double*>(src),
                                     isScaleAVec,
                                     scaleAVec,
                                     AlphaVec,
                                     transA,
                                     m,
                                     k,
                                     size,
                                     isMXFormat);
        break;
    case HIP_R_16F:
        cast_mul<TcCast, Tc, hipblasLtHalf>(dst,
                                            static_cast<const hipblasLtHalf*>(src),
                                            isScaleAVec,
                                            scaleAVec,
                                            AlphaVec,
                                            transA,
                                            m,
                                            k,
                                            size,
                                            isMXFormat);
        break;
    case HIP_R_16BF:
        cast_mul<TcCast, Tc, hip_bfloat16>(dst,
                                           static_cast<const hip_bfloat16*>(src),
                                           isScaleAVec,
                                           scaleAVec,
                                           AlphaVec,
                                           transA,
                                           m,
                                           k,
                                           size,
                                           isMXFormat);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul<TcCast, Tc, hipblaslt_f8_fnuz>(dst,
                                                static_cast<const hipblaslt_f8_fnuz*>(src),
                                                isScaleAVec,
                                                scaleAVec,
                                                AlphaVec,
                                                transA,
                                                m,
                                                k,
                                                size,
                                                isMXFormat);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul<TcCast, Tc, hipblaslt_bf8_fnuz>(dst,
                                                 static_cast<const hipblaslt_bf8_fnuz*>(src),
                                                 isScaleAVec,
                                                 scaleAVec,
                                                 AlphaVec,
                                                 transA,
                                                 m,
                                                 k,
                                                 size,
                                                 isMXFormat);
        break;
    case HIP_R_8F_E4M3:
        cast_mul<TcCast, Tc, hipblaslt_f8>(dst,
                                           static_cast<const hipblaslt_f8*>(src),
                                           isScaleAVec,
                                           scaleAVec,
                                           AlphaVec,
                                           transA,
                                           m,
                                           k,
                                           size,
                                           isMXFormat);
        break;
    case HIP_R_8F_E5M2:
        cast_mul<TcCast, Tc, hipblaslt_bf8>(dst,
                                            static_cast<const hipblaslt_bf8*>(src),
                                            isScaleAVec,
                                            scaleAVec,
                                            AlphaVec,
                                            transA,
                                            m,
                                            k,
                                            size,
                                            isMXFormat);
        break;
    case HIP_R_32I:
        cast_mul<TcCast, Tc, int32_t>(dst,
                                      static_cast<const int32_t*>(src),
                                      isScaleAVec,
                                      scaleAVec,
                                      AlphaVec,
                                      transA,
                                      m,
                                      k,
                                      size);
        break;
    case HIP_R_8I:
        cast_mul<TcCast, Tc, hipblasLtInt8>(dst,
                                            static_cast<const hipblasLtInt8*>(src),
                                            isScaleAVec,
                                            scaleAVec,
                                            AlphaVec,
                                            transA,
                                            m,
                                            k,
                                            size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul" << std::endl;
        break;
    }
}

template <typename TcCast, typename Tc, typename TciACast, typename TiA>
void cast_mul_with_Tci(customVector<TcCast>& dst,
                       const TiA*            A,
                       bool                  isScaleAVec,
                       const TcCast*         scaleAVec,
                       const TcCast*         AlphaVec,
                       bool                  transA,
                       int64_t               m,
                       int64_t               k,
                       size_t                size)
{
    if constexpr(std::is_same<TcCast, float>::value
                 || (!std::is_same<TciACast, hipblaslt_bf8_fnuz>::value
                     && !std::is_same<TciACast, hipblaslt_f8_fnuz>::value)
                        && (!std::is_same<TiA, hipblaslt_bf8_fnuz>::value
                            && !std::is_same<TiA, hipblaslt_f8_fnuz>::value))
    {
        if constexpr(std::is_same<TcCast, float>::value
                     || (!std::is_same<TciACast, hipblaslt_bf8>::value
                         && !std::is_same<TciACast, hipblaslt_f8>::value)
                            && (!std::is_same<TiA, hipblaslt_bf8>::value
                                && !std::is_same<TiA, hipblaslt_f8>::value))
        {
            if(AlphaVec != nullptr)
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                        dst[i]      = static_cast<TcCast>(static_cast<TciACast>(A[i] * scaleA))
                                 * AlphaVec[i % m];
                    }
                }
                else
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                        dst[i]      = static_cast<TcCast>(static_cast<TciACast>(A[i] * scaleA))
                                 * AlphaVec[i / k];
                    }
                }
            }
            else
            {
                if(transA)
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA = isScaleAVec ? scaleAVec[i % m] : scaleAVec[0];
                        dst[i]      = static_cast<TcCast>(static_cast<TciACast>(A[i] * scaleA));
                    }
                }
                else
                {
#pragma omp for
                    for(size_t i = 0; i < size; i++)
                    {
                        auto scaleA = isScaleAVec ? scaleAVec[i / k] : scaleAVec[0];
                        dst[i]      = static_cast<TcCast>(static_cast<TciACast>(A[i] * scaleA));
                    }
                }
            }
        }
    }
}

template <typename TcCast, typename Tc, typename TciACast>
void cast_mul_with_Tci(customVector<TcCast>& dst,
                       const void*           src,
                       hipDataType           TiA,
                       bool                  isScaleAVec,
                       const TcCast*         scaleAVec,
                       const TcCast*         AlphaVec,
                       bool                  transA,
                       int64_t               m,
                       int64_t               k,
                       size_t                size)
{
    switch(TiA)
    {
    case HIP_R_32F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, float>(dst,
                                                       static_cast<const float*>(src),
                                                       isScaleAVec,
                                                       scaleAVec,
                                                       AlphaVec,
                                                       transA,
                                                       m,
                                                       k,
                                                       size);
        break;
    case HIP_R_64F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, double>(dst,
                                                        static_cast<const double*>(src),
                                                        isScaleAVec,
                                                        scaleAVec,
                                                        AlphaVec,
                                                        transA,
                                                        m,
                                                        k,
                                                        size);
        break;
    case HIP_R_16F:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblasLtHalf>(
            dst,
            static_cast<const hipblasLtHalf*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_16BF:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hip_bfloat16>(dst,
                                                              static_cast<const hip_bfloat16*>(src),
                                                              isScaleAVec,
                                                              scaleAVec,
                                                              AlphaVec,
                                                              transA,
                                                              m,
                                                              k,
                                                              size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_f8_fnuz>(
            dst,
            static_cast<const hipblaslt_f8_fnuz*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_bf8_fnuz>(
            dst,
            static_cast<const hipblaslt_bf8_fnuz*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_8F_E4M3:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_f8>(dst,
                                                              static_cast<const hipblaslt_f8*>(src),
                                                              isScaleAVec,
                                                              scaleAVec,
                                                              AlphaVec,
                                                              transA,
                                                              m,
                                                              k,
                                                              size);
        break;
    case HIP_R_8F_E5M2:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblaslt_bf8>(
            dst,
            static_cast<const hipblaslt_bf8*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    case HIP_R_32I:
        cast_mul_with_Tci<TcCast, Tc, TciACast, int32_t>(dst,
                                                         static_cast<const int32_t*>(src),
                                                         isScaleAVec,
                                                         scaleAVec,
                                                         AlphaVec,
                                                         transA,
                                                         m,
                                                         k,
                                                         size);
        break;
    case HIP_R_8I:
        cast_mul_with_Tci<TcCast, Tc, TciACast, hipblasLtInt8>(
            dst,
            static_cast<const hipblasLtInt8*>(src),
            isScaleAVec,
            scaleAVec,
            AlphaVec,
            transA,
            m,
            k,
            size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul_with_Tci" << std::endl;
        break;
    }
}

template <typename TcCast, typename Tc>
void cast_mul_with_Tci(customVector<TcCast>& dst,
                       const void*           src,
                       hipDataType           TiA,
                       bool                  isScaleAVec,
                       const TcCast*         scaleAVec,
                       const TcCast*         AlphaVec,
                       bool                  transA,
                       int64_t               m,
                       int64_t               k,
                       hipDataType           TciACast,
                       size_t                size)
{
    switch(TciACast)
    {
    case HIP_R_32F:
        cast_mul_with_Tci<TcCast, Tc, float>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_64F:
        cast_mul_with_Tci<TcCast, Tc, double>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_16F:
        cast_mul_with_Tci<TcCast, Tc, hipblasLtHalf>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_16BF:
        cast_mul_with_Tci<TcCast, Tc, hip_bfloat16>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_f8_fnuz>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_bf8_fnuz>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E4M3:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_f8>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8F_E5M2:
        cast_mul_with_Tci<TcCast, Tc, hipblaslt_bf8>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_32I:
        cast_mul_with_Tci<TcCast, Tc, int32_t>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    case HIP_R_8I:
        cast_mul_with_Tci<TcCast, Tc, hipblasLtInt8>(
            dst, src, TiA, isScaleAVec, scaleAVec, AlphaVec, transA, m, k, size);
        break;
    default:
        hipblaslt_cerr << "Error type in cast_mul_with_Tci" << std::endl;
        break;
    }
}

// legacy BLAS implementation
// gemm for dim and leading dims <= 600 so no int64 multiplies
template <typename T>
void small_gemm(hipblasOperation_t transA,
                hipblasOperation_t transB,
                int                m,
                int                n,
                int                k,
                T                  alpha,
                const T*           A,
                int                lda,
                const T*           B,
                int                ldb,
                T                  beta,
                T*                 C,
                int                ldc)
{
    bool notTA = (transA == HIPBLAS_OP_N);
    bool notTB = (transB == HIPBLAS_OP_N);

    if(!m or !n or (alpha == 0.0 or !k) && (beta == 1.0))
        return;

    if(alpha == 0.0)
    {
        if(beta == 0.0)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i < m; ++i)
                {
                    C[j * ldc + i] = 0.0;
                }
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i < m; ++i)
                {
                    C[j * ldc + i] *= beta;
                }
            }
        }
        return;
    }

    if(notTB)
    {
        if(notTA)
        {
            // C = alpha*A*B + beta*C.
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 0; j < n; ++j)
            {
                if(beta == 0.0)
                {
                    for(int i = 0; i < m; ++i)
                    {
                        C[j * ldc + i] = 0.0;
                    }
                }
                else if(beta != 1.0)
                {
                    for(int i = 0; i < m; ++i)
                    {
                        C[j * ldc + i] *= beta;
                    }
                }

                for(int l = 0; l < k; ++l)
                {
                    float temp = alpha * B[j * ldb + l];
                    for(int i = 0; i < m; ++i)
                    {
                        C[j * ldc + i] += temp * A[l * lda + i];
                    }
                }
            }
        }
        else
        {
            // C = alpha*A**T*B + beta*C
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i < m; ++i)
                {
                    float temp = 0.0f;
                    for(int l = 0; l < k; ++l)
                    {
                        temp += A[i * lda + l] * B[j * ldb + l];
                    }
                    if(beta == 0.0f)
                    {
                        C[j * ldc + i] = alpha * temp;
                    }
                    else
                    {
                        C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
                    }
                }
            }
        }
    }
    else // TB
    {
        if(notTA)
        {
            //  C = alpha*A*B**T + beta*C
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 0; j < n; ++j)
            {
                if(beta == 0.0)
                {
                    for(int i = 0; i < m; ++i)
                    {
                        C[j * ldc + i] = 0.0;
                    }
                }
                else if(beta != 1.0)
                {
                    for(int i = 0; i < m; ++i)
                    {
                        C[j * ldc + i] = beta * C[j * ldc + i];
                    }
                }

                for(int l = 0; l < k; ++l)
                {
                    float temp = alpha * B[l * ldb + j];
                    for(int i = 0; i < m; ++i)
                    {
                        C[j * ldc + i] += temp * A[l * lda + i];
                    }
                }
            }
        }
        else
        {
            // C = alpha*A**T*B**T + beta*C
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i < m; ++i)
                {
                    float temp = 0.0;
                    for(int l = 0; l < k; ++l)
                    {
                        temp += A[i * lda + l] * B[l * ldb + j];
                    }

                    if(beta == 0.0)
                    {
                        C[j * ldc + i] = alpha * temp;
                    }
                    else
                    {
                        C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
                    }
                }
            }
        }
    }
}

template <typename Tc>
void cblas_gemm(hipblasOperation_t       transA,
                hipblasOperation_t       transB,
                int64_t                  m,
                int64_t                  n,
                int64_t                  k,
                Tc                       alpha,
                const void*              A,
                int64_t                  lda,
                const void*              B,
                int64_t                  ldb,
                Tc                       beta,
                std::add_pointer_t<void> C,
                int64_t                  ldc,
                const void*              AlphaVec,
                const void*              scaleAVec,
                const void*              scaleBVec,
                Tc                       scaleD,
                bool                     isScaleAVec,
                bool                     isScaleBVec,
                hipDataType              TiA,
                hipDataType              TiB,
                hipDataType              To,
                hipDataType              Tc_enum,
                hipDataType              TciA,
                hipDataType              TciB,
                bool                     alt,
                bool                     isScaleAMXFormat,
                bool                     isScaleBMXFormat)
{
    using IntTcCast = std::conditional_t<std::is_same<Tc, int32_t>::value, double, Tc>;
    // cblas does not support hipblasLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing
    using HalfTcCast = std::conditional_t<std::is_same<Tc, hipblasLtHalf>::value, float, Tc>;
    using TcCast     = std::conditional_t<std::is_same<Tc, int32_t>::value, IntTcCast, HalfTcCast>;

    if(Tc_enum == HIP_R_32I)
    {
        Tc_enum = HIP_R_64F;
    }
    else if(Tc_enum == HIP_R_16F)
    {
        Tc_enum = HIP_R_32F;
    }

    hipDataType TciACast = (TciA == HIP_R_32I) ? HIP_R_64F : TciA;
    hipDataType TciBCast = (TciB == HIP_R_32I) ? HIP_R_64F : TciB;

    size_t sizeA          = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB          = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC          = n * size_t(ldc);
    size_t scaleAVec_size = isScaleAVec ? m : 1;
    size_t scaleBVec_size = isScaleBVec ? n : 1;

    customVector<TcCast> A_Tc, B_Tc, C_Tc, AlphaVec_Tc, scaleA_Tc, scaleB_Tc;

    if(AlphaVec)
    {
        AlphaVec_Tc.initialize(m);
        cast_mul(AlphaVec_Tc, (Tc*)AlphaVec, m);
    }

    if(scaleAVec)
    {
        scaleA_Tc.initialize(scaleAVec_size);
        cast_mul(scaleA_Tc, (Tc*)scaleAVec, scaleAVec_size);
    }

    if(scaleBVec)
    {
        scaleB_Tc.initialize(scaleBVec_size);
        cast_mul(scaleB_Tc, (Tc*)scaleBVec, scaleBVec_size);
    }

    A_Tc.initialize(sizeA);
    if(realDataTypeSize(TiA) > realDataTypeSize(TciACast))
    {
        cast_mul_with_Tci<TcCast, Tc>(A_Tc,
                                      A,
                                      TiA,
                                      isScaleAVec,
                                      scaleA_Tc,
                                      AlphaVec_Tc,
                                      transA == HIPBLAS_OP_N,
                                      m,
                                      k,
                                      TciACast,
                                      sizeA);
    }
    else
    {
        cast_mul<TcCast, Tc>(A_Tc,
                             A,
                             TiA,
                             isScaleAVec,
                             scaleA_Tc,
                             AlphaVec_Tc,
                             transA == HIPBLAS_OP_N,
                             m,
                             k,
                             sizeA,
                             isScaleAMXFormat);
    }

    B_Tc.initialize(sizeB);
    if(realDataTypeSize(TiB) > realDataTypeSize(TciBCast))
    {
        cast_mul_with_Tci<TcCast, Tc>(B_Tc,
                                      B,
                                      TiB,
                                      isScaleBVec,
                                      scaleB_Tc,
                                      nullptr,
                                      transB != HIPBLAS_OP_N,
                                      n,
                                      k,
                                      TciBCast,
                                      sizeB);
    }
    else
    {
        cast_mul<TcCast, Tc>(B_Tc,
                             B,
                             TiB,
                             isScaleBVec,
                             scaleB_Tc,
                             nullptr,
                             transB != HIPBLAS_OP_N,
                             n,
                             k,
                             sizeB,
                             isScaleBMXFormat);
    }

    if(To == Tc_enum)
    {
        C_Tc.initialize(C);
    }
    else
    {
        C_Tc.initialize(sizeC);
        cast_mul<TcCast>(C_Tc, C, To, sizeC);
    }

    TcCast alphaCast = (TcCast)alpha;
    TcCast betaCast  = (TcCast)beta;

    // just directly cast, since transA, transB are integers in the enum
    //printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    if constexpr(std::is_same<TcCast, float>::value)
    {
        static constexpr int64_t small = 600; // seeing random NaNs with blis on some small sizes
        if(m > small || n > small || k > small || lda > small || ldb > small || ldc > small)
        {
            cblas_sgemm(CblasColMajor,
                        HIPOperationToCBLASTanspose(transA),
                        HIPOperationToCBLASTanspose(transB),
                        m,
                        n,
                        k,
                        alphaCast,
                        A_Tc,
                        lda,
                        B_Tc,
                        ldb,
                        betaCast,
                        C_Tc,
                        ldc);
        }
        else
        {
            small_gemm<float>(
                transA, transB, m, n, k, alphaCast, A_Tc, lda, B_Tc, ldb, betaCast, C_Tc, ldc);
        }
    }
    else if constexpr(std::is_same<TcCast, double>::value)
    {
        static constexpr int64_t small = 600; // seeing random NaNs with blis on some small sizes
        if(m > small || n > small || k > small || lda > small || ldb > small || ldc > small)
        {
            cblas_dgemm(CblasColMajor,
                        HIPOperationToCBLASTanspose(transA),
                        HIPOperationToCBLASTanspose(transB),
                        m,
                        n,
                        k,
                        alphaCast,
                        A_Tc,
                        lda,
                        B_Tc,
                        ldb,
                        betaCast,
                        C_Tc,
                        ldc);
        }
        else
        {
            small_gemm<double>(
                transA, transB, m, n, k, alphaCast, A_Tc, lda, B_Tc, ldb, betaCast, C_Tc, ldc);
        }
    }

    if(scaleD != 1)
    {
        sat_cast_mul<TcCast, Tc>(C, To, C_Tc, scaleD, sizeC);
    }
    else
    {
        if(To != Tc_enum)
        {
            sat_cast_mul<TcCast, Tc>(C, To, C_Tc, scaleD, sizeC);
        }
    }
}

#define CREATEFUNCTION(Tc)                                                  \
    template void cblas_gemm<Tc>(hipblasOperation_t       transA,           \
                                 hipblasOperation_t       transB,           \
                                 int64_t                  m,                \
                                 int64_t                  n,                \
                                 int64_t                  k,                \
                                 Tc                       alpha,            \
                                 const void*              A,                \
                                 int64_t                  lda,              \
                                 const void*              B,                \
                                 int64_t                  ldb,              \
                                 Tc                       beta,             \
                                 std::add_pointer_t<void> C,                \
                                 int64_t                  ldc,              \
                                 const void*              AlphaVec,         \
                                 const void*              scaleAVec,        \
                                 const void*              scaleBVec,        \
                                 Tc                       scaleD,           \
                                 bool                     isScaleAVec,      \
                                 bool                     isScaleBVec,      \
                                 hipDataType              TiA,              \
                                 hipDataType              TiB,              \
                                 hipDataType              To,               \
                                 hipDataType              Tc_enum,          \
                                 hipDataType              TciA,             \
                                 hipDataType              TciB,             \
                                 bool                     alt,              \
                                 bool                     isScaleAMXFormat, \
                                 bool                     isScaleBMXFormat);

CREATEFUNCTION(hipblasLtHalf)
CREATEFUNCTION(float)
CREATEFUNCTION(double)
CREATEFUNCTION(int32_t)
