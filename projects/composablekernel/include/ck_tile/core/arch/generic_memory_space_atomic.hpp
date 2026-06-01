// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"

#define HAS_GLOBAL_ATOMIC_PK_ADD_BUILTIN                        \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_v2f16) && \
        __has_builtin(__builtin_amdgcn_global_atomic_fadd_v2bf16)

namespace ck_tile {

template <typename T, typename ComputeType>
CK_TILE_HOST_DEVICE T add(const T& a, const T& b)
{
    return type_convert<T>(type_convert<ComputeType>(a) + type_convert<ComputeType>(b));
}

CK_TILE_HOST_DEVICE bf16x2_t add_bf16x2_t(const bf16x2_t& a, const bf16x2_t& b)
{
    bf16x2_t rtn;
    rtn[0] = add<bf16_t, float>(a[0], b[0]);
    rtn[1] = add<bf16_t, float>(a[1], b[1]);
    return rtn;
}

CK_TILE_HOST_DEVICE bf16x4_t add_bf16x4_t(const bf16x4_t& a, const bf16x4_t& b)
{
    bf16x4_t rtn;
    rtn[0] = add<bf16_t, float>(a[0], b[0]);
    rtn[1] = add<bf16_t, float>(a[1], b[1]);
    rtn[2] = add<bf16_t, float>(a[2], b[2]);
    rtn[3] = add<bf16_t, float>(a[3], b[3]);
    return rtn;
}

CK_TILE_HOST_DEVICE fp16x2_t add_f16x2_t(const fp16x2_t& a, const fp16x2_t& b)
{
    fp16x2_t rtn;
    rtn[0] = add<fp16_t, float>(a[0], b[0]);
    rtn[1] = add<fp16_t, float>(a[1], b[1]);
    return rtn;
}

CK_TILE_HOST_DEVICE fp8x4_t add_fp8x4_t(const fp8x4_t& a, const fp8x4_t& b)
{
    fp8x4_t rtn;
    rtn[0] = add<fp8_t, float>(a[0], b[0]);
    rtn[1] = add<fp8_t, float>(a[1], b[1]);
    rtn[2] = add<fp8_t, float>(a[2], b[2]);
    rtn[3] = add<fp8_t, float>(a[3], b[3]);
    return rtn;
}

CK_TILE_HOST_DEVICE fp8x8_t add_fp8x8_t(const fp8x8_t& a, const fp8x8_t& b)
{
    fp8x8_t rtn;
    rtn[0] = add<fp8_t, float>(a[0], b[0]);
    rtn[1] = add<fp8_t, float>(a[1], b[1]);
    rtn[2] = add<fp8_t, float>(a[2], b[2]);
    rtn[3] = add<fp8_t, float>(a[3], b[3]);
    rtn[4] = add<fp8_t, float>(a[4], b[4]);
    rtn[5] = add<fp8_t, float>(a[5], b[5]);
    rtn[6] = add<fp8_t, float>(a[6], b[6]);
    rtn[7] = add<fp8_t, float>(a[7], b[7]);
    return rtn;
}

CK_TILE_HOST_DEVICE bf8x4_t add_bf8x4_t(const bf8x4_t& a, const bf8x4_t& b)
{
    bf8x4_t rtn;
    rtn[0] = add<bf8_t, float>(a[0], b[0]);
    rtn[1] = add<bf8_t, float>(a[1], b[1]);
    rtn[2] = add<bf8_t, float>(a[2], b[2]);
    rtn[3] = add<bf8_t, float>(a[3], b[3]);
    return rtn;
}

CK_TILE_HOST_DEVICE bf8x8_t add_bf8x8_t(const bf8x8_t& a, const bf8x8_t& b)
{
    bf8x8_t rtn;
    rtn[0] = add<bf8_t, float>(a[0], b[0]);
    rtn[1] = add<bf8_t, float>(a[1], b[1]);
    rtn[2] = add<bf8_t, float>(a[2], b[2]);
    rtn[3] = add<bf8_t, float>(a[3], b[3]);
    rtn[4] = add<bf8_t, float>(a[4], b[4]);
    rtn[5] = add<bf8_t, float>(a[5], b[5]);
    rtn[6] = add<bf8_t, float>(a[6], b[6]);
    rtn[7] = add<bf8_t, float>(a[7], b[7]);
    return rtn;
}

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_add explicit for
// each datatype.
template <typename X>
CK_TILE_DEVICE void atomic_add(X* p_dst, const X& x);

template <>
CK_TILE_DEVICE void atomic_add<bf16x2_t>(bf16x2_t* p_dst, const bf16x2_t& x)
{
#if HAS_GLOBAL_ATOMIC_PK_ADD_BUILTIN
    __builtin_amdgcn_global_atomic_fadd_v2bf16(c_style_pointer_cast<bf16x2_t*>(p_dst), x);
#else
    union U32BF162_ADDR
    {
        uint32_t* u32_a;
        bf16x2_t* bf162_a;
    };

    union U32BF162
    {
        uint32_t u32;
        bf16x2_t bf162;
    };

    U32BF162_ADDR dword_addr;
    U32BF162 cur_v;
    U32BF162 new_;
    uint32_t old_v, new_v;
    dword_addr.bf162_a = p_dst;
    cur_v.u32          = *dword_addr.u32_a;

    do
    {
        old_v      = cur_v.u32;
        new_.bf162 = add_bf16x2_t(cur_v.bf162, x);
        new_v      = new_.u32;
        cur_v.u32  = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);
#endif
}

template <>
CK_TILE_DEVICE void atomic_add<bf16x4_t>(bf16x4_t* p_dst, bf16x4_t const& x)
{
    // Union to treat the pointer as either bf16x4_t* or uint64_t*:
    union U64BF164_ADDR
    {
        uint64_t* u64_a;
        bf16x4_t* bf164_a;
    };

    // Union to treat the data as either bf16x4_t or 64-bit integer
    union U64BF164
    {
        uint64_t u64;
        bf16x4_t bf164;
    };

    U64BF164_ADDR addr;
    addr.bf164_a = p_dst; // interpret p_dst as a 64-bit location

    // First read (non-atomic) of the old value
    U64BF164 cur_v;
    cur_v.u64 = *addr.u64_a;

    U64BF164 new_v_union;
    uint64_t old_v, new_v;

    do
    {
        // old 64 bits
        old_v = cur_v.u64;

        // Add elementwise in bf16
        new_v_union.bf164 = add_bf16x4_t(cur_v.bf164, x);
        new_v             = new_v_union.u64;

        // Attempt the 64-bit CAS
        cur_v.u64 = atomicCAS(addr.u64_a, old_v, new_v);

    } while(cur_v.u64 != old_v);
}

template <>
CK_TILE_DEVICE void atomic_add<fp8x4_t>(fp8x4_t* p_dst, const fp8x4_t& x)
{
    union U32FP84_ADDR
    {
        uint32_t* u32_a;
        fp8x4_t* fp84_a;
    };

    union U32FP84
    {
        uint32_t u32;
        fp8x4_t fp84;
    };

    U32FP84_ADDR dword_addr;
    U32FP84 cur_v;
    U32FP84 new_;
    uint32_t old_v, new_v;

    dword_addr.fp84_a = p_dst;
    cur_v.u32         = *dword_addr.u32_a;

    do
    {
        old_v     = cur_v.u32;
        new_.fp84 = add_fp8x4_t(cur_v.fp84, x);
        new_v     = new_.u32;
        cur_v.u32 = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);
}

template <>
CK_TILE_DEVICE void atomic_add<bf8x4_t>(bf8x4_t* p_dst, const bf8x4_t& x)
{
    union U32BF84_ADDR
    {
        uint32_t* u32_a;
        bf8x4_t* bf84_a;
    };

    union U32BF84
    {
        uint32_t u32;
        bf8x4_t bf84;
    };

    U32BF84_ADDR dword_addr;
    U32BF84 cur_v;
    U32BF84 new_;
    uint32_t old_v, new_v;

    dword_addr.bf84_a = p_dst;
    cur_v.u32         = *dword_addr.u32_a;

    do
    {
        old_v     = cur_v.u32;
        new_.bf84 = add_bf8x4_t(cur_v.bf84, x);
        new_v     = new_.u32;
        cur_v.u32 = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);
}

//
// Atomic add for fp8x8_t
//
template <>
CK_TILE_DEVICE void atomic_add<fp8x8_t>(fp8x8_t* p_dst, fp8x8_t const& x)
{
    // Union for addressing 64 bits as either "fp8x8_t" or a 64-bit integer.
    union U64FP88_ADDR
    {
        uint64_t* u64_a; // pointer to 64-bit integer
        fp8x8_t* fp88_a; // pointer to fp8x8_t
    };

    union U64FP88
    {
        uint64_t u64;
        fp8x8_t fp88;
    };

    U64FP88_ADDR dword_addr;
    U64FP88 cur_v;
    U64FP88 new_v_union;
    uint64_t old_v, new_v;

    // Point to the destination as both fp8x8_t* and uint64_t*.
    dword_addr.fp88_a = p_dst;
    // Initial read of 64 bits from memory
    cur_v.u64 = *dword_addr.u64_a;

    do
    {
        old_v = cur_v.u64;
        // Add each fp8 element using your add_fp8x8_t(...) routine
        new_v_union.fp88 = add_fp8x8_t(cur_v.fp88, x);
        new_v            = new_v_union.u64;

        // Attempt 64-bit CAS
        cur_v.u64 = atomicCAS(dword_addr.u64_a, old_v, new_v);
    } while(cur_v.u64 != old_v);
}

//
// Atomic add for bf8x8_t
//
template <>
CK_TILE_DEVICE void atomic_add<bf8x8_t>(bf8x8_t* p_dst, bf8x8_t const& x)
{
    union U64BF88_ADDR
    {
        uint64_t* u64_a;
        bf8x8_t* bf88_a;
    };

    union U64BF88
    {
        uint64_t u64;
        bf8x8_t bf88;
    };

    U64BF88_ADDR dword_addr;
    U64BF88 cur_v;
    U64BF88 new_v_union;
    uint64_t old_v, new_v;

    dword_addr.bf88_a = p_dst;
    // Read the original 64 bits
    cur_v.u64 = *dword_addr.u64_a;

    do
    {
        old_v = cur_v.u64;
        // Add each bf8 element using your add_bf8x8_t(...) routine
        new_v_union.bf88 = add_bf8x8_t(cur_v.bf88, x);
        new_v            = new_v_union.u64;

        // 64-bit CAS loop
        cur_v.u64 = atomicCAS(dword_addr.u64_a, old_v, new_v);
    } while(cur_v.u64 != old_v);
}

//
// Atomic add for fp16x2_t
//
template <>
CK_TILE_DEVICE void atomic_add<fp16x2_t>(fp16x2_t* p_dst, fp16x2_t const& x)
{
#if HAS_GLOBAL_ATOMIC_PK_ADD_BUILTIN
    __builtin_amdgcn_global_atomic_fadd_v2f16(c_style_pointer_cast<fp16x2_t*>(p_dst), x);
#else
    union U32F162_ADDR
    {
        uint32_t* u32_a;
        fp16x2_t* f162_a;
    };

    union U32F162
    {
        uint32_t u32;
        fp16x2_t f162;
    };

    U32F162_ADDR dword_addr;
    U32F162 cur_v;
    U32F162 new_;
    uint32_t old_v, new_v;
    dword_addr.f162_a = p_dst;
    cur_v.u32         = *dword_addr.u32_a;

    do
    {
        old_v     = cur_v.u32;
        new_.f162 = add_f16x2_t(cur_v.f162, x);
        new_v     = new_.u32;
        cur_v.u32 = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);
#endif
}

template <typename T, index_t N>
CK_TILE_DEVICE void atomic_add_g(T* p_dst, const thread_buffer<T, N>& x)
{
    static_assert((std::is_same<T, int32_t>::value && (N == 1)) ||
                      (std::is_same<T, uint32_t>::value && (N == 1)) ||
                      (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same<T, double>::value && (N == 1 || N == 2)) ||
                      (std::is_same<T, fp16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, bf16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, fp8_t>::value && (N == 4 || N == 8 || N == 16)) ||
                      (std::is_same<T, bf8_t>::value && (N == 4 || N == 8 || N == 16)),
                  "The granularity of the thread buffer is unsupported on the hardware!");

    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    constexpr auto I2 = number<2>{};
    constexpr auto I3 = number<3>{};

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<float>(x));
        }
        else if constexpr(N == 2)
        {
            atomicAdd(c_style_pointer_cast<float*>(p_dst), x.template get_as<float>()[I0]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, x.template get_as<float>()[I1]);
        }
        else if constexpr(N == 4)
        {
            atomicAdd(c_style_pointer_cast<float*>(p_dst), x.template get_as<float>()[I0]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, x.template get_as<float>()[I1]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 2, x.template get_as<float>()[I2]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 3, x.template get_as<float>()[I3]);
        }
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            return atomicAdd(p_dst, bit_cast<double>(x));
        }
        else if constexpr(N == 2)
        {
            atomicAdd(c_style_pointer_cast<double*>(p_dst), x.template get_as<double>()[I0]);
            atomicAdd(c_style_pointer_cast<double*>(p_dst) + 1, x.template get_as<double>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<int32_t>(x));
        }
    }
    else if constexpr(std::is_same<T, uint32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<uint32_t>(x));
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value)
    {
        if constexpr(N == 2)
        {
            atomic_add(c_style_pointer_cast<bf16x2_t*>(p_dst), x.template get_as<bf16x2_t>()[I0]);
        }
        else if constexpr(N == 4)
        {
            atomic_add(c_style_pointer_cast<bf16x4_t*>(p_dst), x.template get_as<bf16x4_t>()[I0]);
        }
        else if constexpr(N == 8)
        {
            atomic_add(c_style_pointer_cast<bf16x4_t*>(p_dst), x.template get_as<bf16x4_t>()[I0]);
            atomic_add(c_style_pointer_cast<bf16x4_t*>(p_dst) + 1,
                       x.template get_as<bf16x4_t>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, fp8_t>::value)
    {
        if constexpr(N == 4)
        {
            atomic_add(c_style_pointer_cast<fp8x4_t*>(p_dst), x.template get_as<fp8x4_t>()[I0]);
        }
        if constexpr(N == 8)
        {
            atomic_add(c_style_pointer_cast<fp8x8_t*>(p_dst), x.template get_as<fp8x8_t>()[I0]);
        }
        if constexpr(N == 16)
        {
            atomic_add(c_style_pointer_cast<fp8x8_t*>(p_dst), x.template get_as<fp8x8_t>()[I0]);
            atomic_add(c_style_pointer_cast<fp8x8_t*>(p_dst) + 1, x.template get_as<fp8x8_t>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, bf8_t>::value)
    {
        if constexpr(N == 4)
        {
            atomic_add(c_style_pointer_cast<bf8x4_t*>(p_dst), x.template get_as<bf8x4_t>()[I0]);
        }
        if constexpr(N == 8)
        {
            atomic_add(c_style_pointer_cast<bf8x8_t*>(p_dst), x.template get_as<bf8x8_t>()[I0]);
        }
        if constexpr(N == 16)
        {
            atomic_add(c_style_pointer_cast<bf8x8_t*>(p_dst), x.template get_as<bf8x8_t>()[I0]);
            atomic_add(c_style_pointer_cast<bf8x8_t*>(p_dst) + 1, x.template get_as<bf8x8_t>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value)
    {
        static_for<0, N / 2, 1>{}([&](auto i) {
            atomic_add(c_style_pointer_cast<fp16x2_t*>(p_dst) + i,
                       x.template get_as<fp16x2_t>()[i]);
        });
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void atomic_max_g(T* p_dst, const thread_buffer<T, N>& x)
{
    static_assert((std::is_same<T, int32_t>::value && (N == 1)) ||
                      (std::is_same<T, uint32_t>::value && (N == 1)) ||
                      (std::is_same<T, float>::value && (N == 1 || N == 2)) ||
                      (std::is_same<T, double>::value && (N == 1)),
                  "wrong! not implemented");

    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<float>(x));
        }
        else if constexpr(N == 2)
        {
            atomicMax(c_style_pointer_cast<float*>(p_dst), x.template get_as<float>()[I0]);
            atomicMax(c_style_pointer_cast<float*>(p_dst) + 1, x.template get_as<float>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<double>(x));
        }
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<int32_t>(x));
        }
    }
    else if constexpr(std::is_same<T, uint32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<uint32_t>(x));
        }
    }
}

} // namespace ck_tile
