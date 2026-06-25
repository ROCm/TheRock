/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
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

#ifndef _HIPBLASLT_FLOAT8_H_
#define _HIPBLASLT_FLOAT8_H_

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))
/*! \brief Struct to represent a 8 bit floating-point number. */

typedef struct
{
    uint8_t __x;
} hipblaslt_f8_fnuz;

typedef struct
{
    uint8_t __x;
} hipblaslt_bf8_fnuz;

typedef struct
{
    uint8_t __x;
} hipblaslt_f8;

typedef struct
{
    uint8_t __x;
} hipblaslt_bf8;

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// NANOO E4M3
struct HIPBLASLT_EXPORT hipblaslt_f8_fnuz: public __hip_fp8_e4m3_fnuz
{
    using __hip_fp8_e4m3_fnuz:: __hip_fp8_e4m3_fnuz;

#if HIP_FP8_TYPE_FNUZ
    HIP_HOST_DEVICE hipblaslt_f8_fnuz(const _Float16 f)
#else
    HIP_HOST hipblaslt_f8_fnuz(const _Float16 f)
#endif
    : __hip_fp8_e4m3_fnuz(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_FNUZ
    HIP_HOST_DEVICE operator _Float16() const
#else
    HIP_HOST operator _Float16() const
#endif
    {
        return _Float16(float(*this));
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return __x == 0x00;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return __x == 0x80;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return __x == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ hipblaslt_f8_fnuz& operator=(const hipblaslt_f8_fnuz& a)
    {
        __x = a.__x;
        return *this;
    }
};

// OCPFP8 E4M3
struct HIPBLASLT_EXPORT hipblaslt_f8: public __hip_fp8_e4m3
{
    using __hip_fp8_e4m3:: __hip_fp8_e4m3;

#if HIP_FP8_TYPE_OCP
    HIP_HOST_DEVICE hipblaslt_f8(const _Float16 f)
#else
    HIP_HOST hipblaslt_f8(const _Float16 f)
#endif
    : __hip_fp8_e4m3(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_OCP
    HIP_HOST_DEVICE operator _Float16() const
#else
    HIP_HOST operator _Float16() const
#endif
    {
#if HIP_FP8_CVT_FAST_PATH
        return _Float16(float(*this));
#else
        return _Float16(float(*this));
#endif
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return __x == 0x00;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return __x == 0x80;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return __x == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ hipblaslt_f8& operator=(const hipblaslt_f8& a)
    {
        __x = a.__x;
        return *this;
    }
};

// NANOO E5M2
struct HIPBLASLT_EXPORT hipblaslt_bf8_fnuz: public __hip_fp8_e5m2_fnuz
{

    using __hip_fp8_e5m2_fnuz:: __hip_fp8_e5m2_fnuz;

#if HIP_FP8_TYPE_FNUZ
    HIP_HOST_DEVICE hipblaslt_bf8_fnuz(const _Float16 f)
#else
    HIP_HOST hipblaslt_bf8_fnuz(const _Float16 f)
#endif
    : __hip_fp8_e5m2_fnuz(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_OCP
    HIP_HOST_DEVICE operator _Float16() const
#else
    HIP_HOST operator _Float16() const
#endif
    {
        return _Float16(float(*this));
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return (__x == 0x00 || __x == 0x80);
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return __x == 0x80;
    }

    // check for inf: no inf, so checking nan?
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return __x == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ hipblaslt_bf8_fnuz& operator=(const hipblaslt_bf8_fnuz& a)
    {
        __x = a.__x;
        return *this;
    }
};


// OCPFP8 E5M2
struct HIPBLASLT_EXPORT hipblaslt_bf8: public __hip_fp8_e5m2
{
    using __hip_fp8_e5m2:: __hip_fp8_e5m2;

#if HIP_FP8_TYPE_OCP
    HIP_HOST_DEVICE hipblaslt_bf8(const _Float16 f)
#else
    HIP_HOST hipblaslt_bf8(const _Float16 f)
#endif
    : __hip_fp8_e5m2(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_OCP
    HIP_HOST_DEVICE operator _Float16() const
#else
    HIP_HOST operator _Float16() const
#endif
    {
#if HIP_FP8_CVT_FAST_PATH
        //TODO: use single CVT instruction
        return _Float16(float(*this));
#else
        return _Float16(float(*this));
#endif
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return __x == 0x00 || __x == 0x80;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return (__x == 0x7d) || (__x == 0x7e) || (__x == 0x7f) ||
        (__x == 0xfd) || (__x == 0xfe) || (__x == 0xff);
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return (__x == 0x7c) || (__x == 0xfc);
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ hipblaslt_bf8& operator=(const hipblaslt_bf8& a)
    {
        __x = a.__x;
        return *this;
    }
};

namespace std
{
    inline hipblaslt_f8_fnuz sin(hipblaslt_f8_fnuz a)
    {
        return hipblaslt_f8_fnuz(sinf(float(a)));
    }
    inline hipblaslt_f8 sin(hipblaslt_f8 a)
    {
        return hipblaslt_f8(sinf(float(a)));
    }
    inline hipblaslt_f8_fnuz cos(hipblaslt_f8_fnuz a)
    {
        return hipblaslt_f8_fnuz(cosf(float(a)));
    }
    inline hipblaslt_f8 cos(hipblaslt_f8 a)
    {
        return hipblaslt_f8(cosf(float(a)));
    }
    inline hipblaslt_bf8_fnuz sin(hipblaslt_bf8_fnuz a)
    {
        return hipblaslt_bf8_fnuz(sinf(float(a)));
    }
    inline hipblaslt_bf8 sin(hipblaslt_bf8 a)
    {
        return hipblaslt_bf8(sinf(float(a)));
    }
    inline hipblaslt_bf8_fnuz cos(hipblaslt_bf8_fnuz a)
    {
        return hipblaslt_bf8_fnuz(cosf(float(a)));
    }
    inline hipblaslt_bf8 cos(hipblaslt_bf8 a)
    {
        return hipblaslt_bf8(cosf(float(a)));
    }
    __device__ __host__ constexpr hipblaslt_f8_fnuz real(const hipblaslt_f8_fnuz& a)
    {
        return a;
    }
    __device__ __host__ constexpr hipblaslt_f8 real(const hipblaslt_f8& a)
    {
        return a;
    }
    __device__ __host__ constexpr hipblaslt_bf8_fnuz real(const hipblaslt_bf8_fnuz& a)
    {
        return a;
    }
    __device__ __host__ constexpr hipblaslt_bf8 real(const hipblaslt_bf8& a)
    {
        return a;
    }
}

// TODO: remove all operator overloading below after analyzing where those are used!
// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const hipblaslt_f8_fnuz& f8)
{
    return os << float(f8);
}
inline std::ostream& operator<<(std::ostream& os, const hipblaslt_f8& f8)
{
    return os << float(f8);
}
inline std::ostream& operator<<(std::ostream& os, const hipblaslt_bf8_fnuz& bf8)
{
    return os << float(bf8);
}
inline std::ostream& operator<<(std::ostream& os, const hipblaslt_bf8& bf8)
{
    return os << float(bf8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator+(const float fa, hipblaslt_f8_fnuz b)
{
    return (fa + float(b));
}
inline __host__ __device__ float operator+(const float fa, hipblaslt_f8 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(const float fa, hipblaslt_bf8_fnuz b)
{
    return (fa + float(b));
}
inline __host__ __device__ float operator+(const float fa, hipblaslt_bf8 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_f8_fnuz a, const float fb)
{
    return (float(a) + fb);
}
inline __host__ __device__ float operator+(hipblaslt_f8 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_bf8_fnuz a, const float fb)
{
    return (float(a) + fb);
}
inline __host__ __device__ float operator+(hipblaslt_bf8 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_f8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return (float(a) + float(b));
}
inline __host__ __device__ float operator+(hipblaslt_f8 a, hipblaslt_bf8 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_bf8_fnuz a, hipblaslt_f8_fnuz b)
{
    return (float(a) + float(b));
}
inline __host__ __device__ float operator+(hipblaslt_bf8 a, hipblaslt_f8 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ hipblaslt_f8_fnuz operator+(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return hipblaslt_f8_fnuz(float(a) + float(b));
}
inline __host__ __device__ hipblaslt_f8 operator+(hipblaslt_f8 a, hipblaslt_f8 b)
{
    return hipblaslt_f8(float(a) + float(b));
}

inline __host__ __device__ hipblaslt_bf8_fnuz operator+(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return hipblaslt_bf8_fnuz(float(a) + float(b));
}
inline __host__ __device__ hipblaslt_bf8 operator+(hipblaslt_bf8 a, hipblaslt_bf8 b)
{
    return hipblaslt_bf8(float(a) + float(b));
}

inline __host__ __device__ hipblaslt_f8_fnuz& operator+=(hipblaslt_f8_fnuz& a, hipblaslt_f8_fnuz b)
{
    return a = hipblaslt_f8_fnuz(float(a) + float(b));
}
inline __host__ __device__ hipblaslt_f8& operator+=(hipblaslt_f8& a, hipblaslt_f8 b)
{
    return a = hipblaslt_f8(float(a) + float(b));
}

inline __host__ __device__ hipblaslt_bf8_fnuz& operator+=(hipblaslt_bf8_fnuz& a,
                                                          hipblaslt_bf8_fnuz  b)
{
    return a = hipblaslt_bf8_fnuz(float(a) + float(b));
}
inline __host__ __device__ hipblaslt_bf8& operator+=(hipblaslt_bf8& a,
                                                     hipblaslt_bf8  b)
{
    return a = hipblaslt_bf8(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline __host__ __device__ float operator*(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return float(a) * float(b);
}
inline __host__ __device__ float operator*(hipblaslt_f8 a, hipblaslt_f8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_f8_fnuz b)
{
    return (a * float(b));
}
inline __host__ __device__ float operator*(float a, hipblaslt_f8 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f8_fnuz a, float b)
{
    return (float(a) * b);
}
inline __host__ __device__ float operator*(hipblaslt_f8 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, hipblaslt_f8_fnuz b)
{
    return ((float)a * float(b));
}
inline __host__ __device__ float operator*(int32_t a, hipblaslt_f8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, hipblaslt_f8_fnuz b)
{
    return ((float)a * float(b));
}
inline __host__ __device__ float operator*(double a, hipblaslt_f8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return float(a) * float(b);
}
inline __host__ __device__ float operator*(hipblaslt_bf8 a, hipblaslt_bf8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_bf8_fnuz b)
{
    return (a * float(b));
}
inline __host__ __device__ float operator*(float a, hipblaslt_bf8 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf8_fnuz a, float b)
{
    return (float(a) * b);
}
inline __host__ __device__ float operator*(hipblaslt_bf8 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, hipblaslt_bf8_fnuz b)
{
    return ((float)a * float(b));
}
inline __host__ __device__ float operator*(int32_t a, hipblaslt_bf8 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, hipblaslt_bf8_fnuz b)
{
    return ((float)a * float(b));
}
inline __host__ __device__ float operator*(double a, hipblaslt_bf8 b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator*(hipblaslt_f8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return float(a) * float(b);
}
inline __host__ __device__ float operator*(hipblaslt_f8 a, hipblaslt_bf8 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(hipblaslt_bf8_fnuz a, hipblaslt_f8_fnuz b)
{
    return float(a) * float(b);
}
inline __host__ __device__ float operator*(hipblaslt_bf8 a, hipblaslt_f8 b)
{
    return float(a) * float(b);
}

// overloading for compare
inline __host__ __device__ bool operator==(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return (a.__x == b.__x);
}
inline __host__ __device__ bool operator==(hipblaslt_f8 a, hipblaslt_f8 b)
{
    return (a.__x == b.__x);
}
inline __host__ __device__ bool operator==(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return (a.__x == b.__x);
}
inline __host__ __device__ bool operator==(hipblaslt_bf8 a, hipblaslt_bf8 b)
{
    return (a.__x == b.__x);
}

inline __host__ __device__ bool operator!=(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return (a.__x != b.__x);
}
inline __host__ __device__ bool operator!=(hipblaslt_f8 a, hipblaslt_f8 b)
{
    return (a.__x != b.__x);
}
inline __host__ __device__ bool operator!=(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return (a.__x != b.__x);
}
inline __host__ __device__ bool operator!=(hipblaslt_bf8 a, hipblaslt_bf8 b)
{
    return (a.__x != b.__x);
}

inline __host__ __device__ bool operator>=(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline __host__ __device__ bool operator>=(hipblaslt_f8 a, hipblaslt_f8 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline __host__ __device__ bool operator>(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
inline __host__ __device__ bool operator>(hipblaslt_f8 a, hipblaslt_f8 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
// =================================================================================================

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // _HIPBLASLT_FLOAT8_H_
