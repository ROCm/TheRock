// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

#if !CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/core/arch/amd_buffer_coherence.hpp"

// This attribute gives a hint to the compiler that a branch is likely to be taken.
// Then, the compiler should remove if possible the associated s_cbranch_execz branch that would
// have been generated.
#if __cplusplus >= 202002L
#define LIKELY(x) (x) [[likely]]
#else
#define LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

namespace ck_tile {

// amd_wave_read_first_lane is the SGPR function from AMD GPU device to load 1 or a series of the
// memory to the SGPR registers.
__device__ inline uint32_t amd_wave_read_first_lane(uint16_t v)
{
    return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
}

__device__ inline uint32_t amd_wave_read_first_lane(uint8_t v)
{
    return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
}

__device__ inline uint32_t amd_wave_read_first_lane(uint32_t value)
{
    return __builtin_amdgcn_readfirstlane(value);
}

__device__ inline int32_t amd_wave_read_first_lane(int32_t value)
{
    return __builtin_amdgcn_readfirstlane(value);
}

template <typename Object, std::enable_if_t<std::is_trivially_copyable_v<Object>, int> = 0>
__device__ inline auto amd_wave_read_first_lane(const Object& obj)
{
    constexpr size_t ObjectSize = sizeof(Object);
    constexpr size_t SGPR_size  = 4;
    constexpr size_t NumFull    = ObjectSize / SGPR_size;
    constexpr size_t Tail       = ObjectSize % SGPR_size;

    const unsigned char* src = reinterpret_cast<const unsigned char*>(&obj);
    alignas(Object) unsigned char dst[ObjectSize];

    static_for<0, NumFull, 1>{}([&](auto Ic) {
        constexpr size_t offset = Ic * SGPR_size;
        uint32_t read_src;
        __builtin_memcpy(&read_src, src + offset, SGPR_size);
        read_src = __builtin_amdgcn_readfirstlane(read_src);
        __builtin_memcpy(dst + offset, &read_src, SGPR_size);
    });

    if constexpr(Tail != 0)
    {
        constexpr size_t offset = NumFull * SGPR_size;
        uint32_t tail_loc       = 0;
        __builtin_memcpy(&tail_loc, src + offset, Tail);
        tail_loc = __builtin_amdgcn_readfirstlane(tail_loc);
        __builtin_memcpy(dst + offset, &tail_loc, Tail);
    }
    Object out;
    __builtin_memcpy(&out, dst, ObjectSize);
    return out;
}

// 128 bit SGPRs to supply buffer resource in buffer instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
struct __attribute__((packed)) buffer_resource
{
    const void* ptr;
    uint32_t range;
    uint32_t config;
};

template <typename ForceSGPR = std::false_type>
CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr,
                                                   uint32_t size = 0xffffffff,
                                                   ForceSGPR     = {})
{
    buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
    int32x4_t r = __builtin_bit_cast(int32x4_t, res);
    if constexpr(std::is_same_v<ForceSGPR, std::true_type>)
    {
        r = amd_wave_read_first_lane(r);
    }
    return r;
}

namespace impl {
// below type indicate the data type used for buffer load inline asm
// clang-format off
template<index_t N, typename T> struct buffer_load_trait;

template<typename T> struct buffer_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct buffer_load_trait<8 , T> { using payload_t = fp32x2_t; };
template<typename T> struct buffer_load_trait<4 , T> { using payload_t = float; };
template<typename T> struct buffer_load_trait<2 , T> { using payload_t = float; };
template<typename T> struct buffer_load_trait<1 , T> { using payload_t = float; };

#if CK_TILE_BUFFER_LOAD_RAW_BF16_WA
template<> struct buffer_load_trait<16, thread_buffer<bf16_t, 8>> { using payload_t = bf16x8_t; };
template<> struct buffer_load_trait<8 , thread_buffer<bf16_t, 4>> { using payload_t = bf16x4_t; };
template<> struct buffer_load_trait<4 , thread_buffer<bf16_t, 2>> { using payload_t = bf16x2_t; };
#endif
// clang-format on
} // namespace impl

// TODO: glc/slc/...
template <index_t bytes, bool pre_nop = false>
struct buffer_load;

template <index_t bytes, bool pre_nop = false>
struct buffer_load_if;

template <index_t bytes>
struct buffer_store;

template <index_t bytes>
struct buffer_store_if;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
// TODO: strict aliasing rule seems fail when reinterpret_cast between vector type
// (exp_vector_type(xxx))

#define HAS_RAW_BUFFER_BUILTINS                             \
    __has_builtin(__builtin_amdgcn_raw_buffer_load_b32) &&  \
        __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && \
        __has_builtin(__builtin_amdgcn_raw_buffer_store_b32)

#if HAS_RAW_BUFFER_BUILTINS
CK_TILE_DEVICE __amdgpu_buffer_rsrc_t cast_to_amdgpu_buffer_rsrc_t(int32x4_t res)
{
    __amdgpu_buffer_rsrc_t as_rsrc;
    static_assert(sizeof(res) == sizeof(as_rsrc) && "Size of buffer resource should match");
    memcpy(&as_rsrc, &res, sizeof(res));
    return as_rsrc;
}
#endif

template <bool pre_nop>
struct buffer_load<16, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::buffer_load_trait<16, T>::payload_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b128(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<8, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = typename impl::buffer_load_trait<8, T>::payload_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b64(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<4, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::buffer_load_trait<4, T>::payload_t;

#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b32(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dword %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dword %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mbuf_t = typename impl::buffer_load_trait<2, T>::payload_t;

#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b16(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_ushort %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_ushort %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

template <bool pre_nop>
struct buffer_load<1, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::buffer_load_trait<1, T>::payload_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset                 = i_offset;
        reinterpret_cast<mbuf_t&>(value) = __builtin_amdgcn_raw_buffer_load_b16(
            cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_ubyte %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
#endif
    }
};

#if HAS_RAW_BUFFER_BUILTINS
template <index_t bytes, bool pre_nop>
struct buffer_load_if
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        if LIKELY(1 <= flag)
        {
            buffer_load<bytes, pre_nop>{}(
                value, res, v_offset, s_offset, i_offset, flag, bool_constant<pre_nop>{});
        }
    }
};
#else
template <bool pre_nop>
struct buffer_load_if<16, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 16);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<16, T>::payload_t;
        static_assert(sizeof(mbuf_t) == sizeof(T));
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<8, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 8);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<8, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<4, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<4, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dword %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dword %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<2, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ushort %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ushort %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<1, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<1, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ubyte %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ubyte %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};
#endif

#pragma clang diagnostic pop // "-Wundefined-reinterpret-cast"

template <>
struct buffer_store<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = uint32x4_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b128(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = uint32x2_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b64(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_dwordx2 %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = uint32_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b32(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_dword %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 2);
        using mbuf_t = uint16_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b16(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_short %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct buffer_store<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 1);
        using mbuf_t = uint8_t;
#if HAS_RAW_BUFFER_BUILTINS
        index_t s_offset = i_offset;
        __builtin_amdgcn_raw_buffer_store_b8(
            bit_cast<mbuf_t>(value), cast_to_amdgpu_buffer_rsrc_t(res), v_offset, s_offset, 0);
#else
        asm volatile("buffer_store_byte %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
#endif
    }
};

#if HAS_RAW_BUFFER_BUILTINS
template <index_t bytes>
struct buffer_store_if
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t s_offset,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        if LIKELY(1 <= flag)
        {
            buffer_store<bytes>{}(value, res, v_offset, s_offset, i_offset);
        }
    }
};
#else
template <>
struct buffer_store_if<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 16);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = fp32x4_t;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_dwordx4 %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 8);
        auto save_exec = __builtin_amdgcn_read_exec();
        // TODO: ugly. rocm-6.0/6.1 seems neet bit_cast to same base type to avoid scratch
        using mbuf_t = ext_vector_t<typename T::value_type, T::size()>;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_dwordx2 %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = float;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_dword %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 2);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = short;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_short %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = float;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_byte %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};
#endif

CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

CK_TILE_DEVICE void lds_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

template <typename scalar_type, index_t N, bool pre_nop = false>
struct buffer_atomic_add_if;

template <bool pre_nop>
struct buffer_atomic_add_if<bf16_t, 2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = float;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "global_atomic_pk_add_bf16 %0, %1, %2 offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(v_offset),
                       "v"(bit_cast<mbuf_t>(value)),
                       "s"(res.xy),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <typename scalar_type, index_t N, bool pre_nop = false>
struct buffer_atomic_add;

template <bool pre_nop>
struct buffer_atomic_add<bf16_t, 2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag = 1*/)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = float;
        asm volatile("global_atomic_pk_add_bf16 %0, %1, %2 offset:%3"
                     :
                     : "v"(v_offset), "v"(bit_cast<mbuf_t>(value)), "s"(res.xy), "n"(i_offset)
                     : "memory");
    }
};

namespace impl {
// below type indicate the data type used for buffer load inline asm
// clang-format off
template<index_t N, typename T> struct smem_load_trait;

template<typename T> struct smem_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct smem_load_trait<8 , T> { using payload_t = fp32x2_t; };
template<typename T> struct smem_load_trait<4 , T> { using payload_t = float; };
template<typename T> struct smem_load_trait<2 , T> { using payload_t = float; };
template<typename T> struct smem_load_trait<1 , T> { using payload_t = float; };

// clang-format on
} // namespace impl

// NOTE: smem load/store no need pre_nop to make sure dependency by sw, happy :)
template <index_t>
struct smem_load;

template <>
struct smem_load<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::smem_load_trait<16, T>::payload_t;
        asm volatile("ds_read_b128 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = typename impl::smem_load_trait<8, T>::payload_t;
        asm volatile("ds_read_b64 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::smem_load_trait<4, T>::payload_t;
        asm volatile("ds_read_b32 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mbuf_t = typename impl::smem_load_trait<1, T>::payload_t;
        asm volatile("ds_read_u16 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct smem_load<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::smem_load_trait<1, T>::payload_t;
        asm volatile("ds_read_u8 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
    }
};

// clang-format off
namespace impl{

// can't use "+v" since there could be potential extra move(read/write)
// use "v" can help remove such duplicated moves
// besides, fake this as "memory" operation to force later valu after this fence
// TODO: may have scratch (because this is memory?)
//       need to reduce extra move inside compiler
template<index_t N>
CK_TILE_DEVICE void insert_dummy_dep_per_dword(array<float, N>& b)
{
    constexpr auto kSize = remove_cvref_t<decltype(b)>::size(); 
    static_for<0, kSize, 1>{}([&](auto i){
        asm volatile(" " : : "v"(b.get(number<i>{})) : "memory");
    });
}
#if 1
// below specialization just merge size() of dwords into single section
template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<2>(array<float, 2>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<3>(array<float, 3>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<4>(array<float, 4>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<8>(array<float, 8>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<16>(array<float, 16>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})),
                         "v"(b.get(number<8>{})), "v"(b.get(number<9>{})), "v"(b.get(number<10>{})), "v"(b.get(number<11>{})),
                         "v"(b.get(number<12>{})), "v"(b.get(number<13>{})), "v"(b.get(number<14>{})), "v"(b.get(number<15>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<32>(array<float, 32>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})),
                         "v"(b.get(number<8>{})), "v"(b.get(number<9>{})), "v"(b.get(number<10>{})), "v"(b.get(number<11>{})),
                         "v"(b.get(number<12>{})), "v"(b.get(number<13>{})), "v"(b.get(number<14>{})), "v"(b.get(number<15>{})),
                         "v"(b.get(number<16>{})), "v"(b.get(number<17>{})), "v"(b.get(number<18>{})), "v"(b.get(number<19>{})),
                         "v"(b.get(number<20>{})), "v"(b.get(number<21>{})), "v"(b.get(number<22>{})), "v"(b.get(number<23>{})),
                         "v"(b.get(number<24>{})), "v"(b.get(number<25>{})), "v"(b.get(number<26>{})), "v"(b.get(number<27>{})),
                         "v"(b.get(number<28>{})), "v"(b.get(number<29>{})), "v"(b.get(number<30>{})), "v"(b.get(number<31>{})) : "memory");
}
#endif
CK_TILE_DEVICE void insert_dummy_dep() {}

template<typename T>
CK_TILE_DEVICE void insert_dummy_dep(T & buffer)
{
    // TODO: indeed we expect T to be multiple of dword. subdword is always buggy
    using da_type = array<float, (sizeof(T) + 3) / 4>;
    auto & dummy = reinterpret_cast<da_type&>(buffer);
    insert_dummy_dep_per_dword(dummy);
}

template<typename Tx, typename... Ty>
CK_TILE_DEVICE void insert_dummy_dep(Tx& bx, Ty&... by)
{
    insert_dummy_dep(bx);
    insert_dummy_dep(by...);
}
}
// clang-format on
template <typename... T>
CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0, T&... o)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
    impl::insert_dummy_dep(o...);
}

CK_TILE_DEVICE void buffer_store_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

CK_TILE_DEVICE auto async_load_fence_raw(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// buffer load i8
CK_TILE_DEVICE_EXTERN int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

CK_TILE_DEVICE_EXTERN int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i8");

CK_TILE_DEVICE_EXTERN int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8");

// buffer load i16
CK_TILE_DEVICE_EXTERN int16_t
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");

CK_TILE_DEVICE_EXTERN int16x2_t
llvm_amdgcn_raw_buffer_load_i16x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i16");

CK_TILE_DEVICE_EXTERN int16x4_t
llvm_amdgcn_raw_buffer_load_i16x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i16");

// buffer load i32
CK_TILE_DEVICE_EXTERN int32_t
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

CK_TILE_DEVICE_EXTERN int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

// dwordx3 - use union to convert between int32x3 and fp16/bf16 types
union dwordx3_union
{
    int32_t as_i32[3];
    fp16_t as_fp16[6];
    bf16_t as_bf16[6];
};

CK_TILE_DEVICE_EXTERN int32x3_t
llvm_amdgcn_raw_buffer_load_i32x3(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v3i32");

CK_TILE_DEVICE_EXTERN int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// buffer load fp16
CK_TILE_DEVICE_EXTERN _Float16
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

CK_TILE_DEVICE_EXTERN fp16x2_t
llvm_amdgcn_raw_buffer_load_fp16x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

CK_TILE_DEVICE_EXTERN fp16x4_t
llvm_amdgcn_raw_buffer_load_fp16x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f16");

// buffer load fp32
CK_TILE_DEVICE_EXTERN float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

CK_TILE_DEVICE_EXTERN fp32x2_t
llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

CK_TILE_DEVICE_EXTERN fp32x4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

// buffer store i8
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                int32x4_t rsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i8");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i8");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i8");

// buffer store i16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16(int16_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16x2(int16x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16x4(int16x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

// buffer store i32
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i32");

// buffer store ui16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16(uint16_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16x2(uint16x2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16x4(uint16x4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32x2(int32x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

// buffer store fp16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16(_Float16 vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16x2(fp16x2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16x4(fp16x4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f16");

// buffer store fp32
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32x2(fp32x2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32x4(fp32x4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

// buffer atomic-add fp16
CK_TILE_DEVICE_EXTERN fp16x2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    fp16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");

// buffer atomic-add bf16
// TODO: Replace with bf16x2_t, but llvm builins only accept cktile_bf16x2_t now.
CK_TILE_DEVICE_EXTERN bf16x2_t llvm_amdgcn_raw_buffer_atomic_add_bf16x2(
    bf16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2bf16");

// buffer atomic-add i32
CK_TILE_DEVICE_EXTERN int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

// buffer atomic-add fp32
CK_TILE_DEVICE_EXTERN float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

// buffer atomic-max fp64
CK_TILE_DEVICE_EXTERN double
llvm_amdgcn_raw_buffer_atomic_max_fp64(double vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64");

// Direct loads from global to LDS.
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

template <unsigned num_dwords, bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dwordxn_v(void* smem,
                                                int32x4_t rsrc,
                                                index_t voffset,
                                                index_t /*soffset*/,
                                                index_t ioffset /*max 0xFFF*/,
                                                index_t /*flag*/       = 0,
                                                bool_constant<pre_nop> = {})
{
#define CK_TILE_ASYNC_LOAD_WITH_INSTR(instr)                            \
    if constexpr(pre_nop)                                               \
        asm volatile("s_nop 4\n" instr " %1, %2, 0 offen offset:%3 lds" \
                     : "=r"(smem) /*dummy dependency for smem*/         \
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)            \
                     : "memory");                                       \
    else                                                                \
        asm volatile(instr " %1, %2, 0 offen offset:%3 lds"             \
                     : "=r"(smem) /*dummy dependency for smem*/         \
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)            \
                     : "memory");

    if constexpr(num_dwords == 1)
    {
        CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dword");
    }
#if defined(__gfx950__)
    else if constexpr(num_dwords == 3)
    {
        CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dwordx3");
    }
    else if constexpr(num_dwords == 4)
    {
        CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dwordx4");
    }
#endif
    else
    {
        static_assert(false, "wrong! not implemented data width");
    }
#undef CK_TILE_ASYNC_LOAD_WITH_INSTR
}

CK_TILE_DEVICE void async_buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE thread_buffer<int8_t, N>
amd_buffer_load_impl_with_bytes(int32x4_t src_wave_buffer_resource,
                                index_t src_thread_addr_offset,
                                index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    using rtn_type = thread_buffer<int8_t, N>;

    if constexpr(N == 1)
    {
        return bit_cast<rtn_type>(llvm_amdgcn_raw_buffer_load_i8(src_wave_buffer_resource,
                                                                 src_thread_addr_offset,
                                                                 src_wave_addr_offset,
                                                                 static_cast<index_t>(coherence)));
    }
    else if constexpr(N == 2)
    {

        int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 32)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        thread_buffer<int32_t, 8> tmp;

        tmp.template get_as<int32x4_t>()(number<0>{}) = tmp0;
        tmp.template get_as<int32x4_t>()(number<1>{}) = tmp1;

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 64)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp2 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 8 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp3 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 12 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));

        thread_buffer<int32_t, 16> tmp;

        tmp.template get_as<int32x4_t>()(number<0>{}) = tmp0;
        tmp.template get_as<int32x4_t>()(number<1>{}) = tmp1;
        tmp.template get_as<int32x4_t>()(number<2>{}) = tmp2;
        tmp.template get_as<int32x4_t>()(number<3>{}) = tmp3;

        return bit_cast<rtn_type>(tmp);
    }
}

#ifndef BUFFER_LOAD_USE_INLINEASM
#define BUFFER_LOAD_USE_INLINEASM 0
#endif

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE thread_buffer<T, N> amd_buffer_load_impl(int32x4_t src_wave_buffer_resource,
                                                        index_t src_thread_addr_offset,
                                                        index_t src_wave_addr_offset)
{
    static_assert(
        (std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, bf16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, int32_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, e8m0_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, pk_int4_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, pk_fp4_raw_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, pk_fp4_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    using rtn_type = thread_buffer<T, N>;

    if constexpr(std::is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset,
                                                 static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32x2(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 8)
        {
            thread_buffer<float, 8> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp;
        }
        else if constexpr(N == 16)
        {
            thread_buffer<float, 16> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp;
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset,
                                                 static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16x2(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 6)
        {
            // N = 6: load as dwordx3 (12 bytes = 6 fp16), using buffer_load_dwordx3 instruction
            int32x3_t tmp_i32x3 =
                llvm_amdgcn_raw_buffer_load_i32x3(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence));

            // Use union to reinterpret int32x3 as fp16x6
            dwordx3_union tmp_union;
            tmp_union.as_i32[0] = tmp_i32x3[0];
            tmp_union.as_i32[1] = tmp_i32x3[1];
            tmp_union.as_i32[2] = tmp_i32x3[2];

            thread_buffer<fp16_t, N> result;
            static_for<0, N, 1>{}([&](auto i) { result[i] = tmp_union.as_fp16[i]; });

            return result;
        }
        else if constexpr(N == 8)
        {
            // use fp32 load to mimic fp16 load
            fp32x4_t tmp = llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
        else if constexpr(N == 16)
        {
            thread_buffer<float, 8> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
        else if constexpr(N == 32)
        {
            thread_buffer<float, 16> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                src_wave_addr_offset,
                                                static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16x2(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16x4(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 6)
        {
            // N = 6: load as dwordx3 (12 bytes = 6 bf16), using buffer_load_dwordx3 instruction
            int32x3_t tmp_i32x3 =
                llvm_amdgcn_raw_buffer_load_i32x3(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence));

            // Use union to reinterpret int32x3 as bf16x6
            dwordx3_union tmp_union;
            tmp_union.as_i32[0] = tmp_i32x3[0];
            tmp_union.as_i32[1] = tmp_i32x3[1];
            tmp_union.as_i32[2] = tmp_i32x3[2];

            thread_buffer<bf16_t, N> result;
            static_for<0, N, 1>{}([&](auto i) { result[i] = tmp_union.as_bf16[i]; });

            return result;
        }
        else if constexpr(N == 8)
        {
            int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
        else if constexpr(N == 16)
        {
            thread_buffer<float, 8> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
        else if constexpr(N == 32)
        {
            thread_buffer<float, 16> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
    }
    else // other datatype
    {
        auto raw_data = amd_buffer_load_impl_with_bytes<sizeof(T) * N, coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);

        return bit_cast<rtn_type>(raw_data);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw_impl(thread_buffer<T, N>& dst,
                                             int32x4_t src_wave_buffer_resource,
                                             index_t src_thread_addr_offset,
                                             index_t src_wave_addr_offset,
                                             index_t src_linear_addr_offset,
                                             index_t flag           = 0,
                                             bool_constant<pre_nop> = {})
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_load instruction");

    using type = thread_buffer<T, N>;
    if constexpr(oob_conditional_check)
    {
        buffer_load_if<sizeof(type), pre_nop>{}(dst,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                src_wave_addr_offset,
                                                src_linear_addr_offset,
                                                flag,
                                                bool_constant<pre_nop>{});
    }
    else
    {
        buffer_load<sizeof(type), pre_nop>{}(dst,
                                             src_wave_buffer_resource,
                                             src_thread_addr_offset,
                                             src_wave_addr_offset,
                                             src_linear_addr_offset,
                                             flag,
                                             bool_constant<pre_nop>{});
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_impl(CK_TILE_LDS_ADDR T* smem,
                                               int32x4_t src_wave_buffer_resource,
                                               index_t src_thread_addr_offset,
                                               index_t src_wave_addr_offset,
                                               index_t src_immediate_addr_offset = 0,
                                               bool_constant<pre_nop>            = {})
{
    constexpr index_t num_bytes = sizeof(T) * N;
    constexpr index_t num_words = num_bytes / 4;
    static_assert(num_bytes % 4 == 0 && (num_words == 1 || num_words == 3 || num_words == 4),
                  "wrong! only support in dword, dwordx3, dwordx4");

    async_buffer_load_dwordxn_v<num_words>(smem,
                                           src_wave_buffer_resource,
                                           src_thread_addr_offset,
                                           src_wave_addr_offset,
                                           src_immediate_addr_offset,
                                           0,
                                           bool_constant<pre_nop>{});
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_async_buffer_load(CK_TILE_LDS_ADDR T* smem,
                                          int32x4_t src_wave_buffer_resource,
                                          index_t src_thread_addr_offset,
                                          index_t src_wave_addr_offset,
                                          index_t src_immediate_addr_offset    = 0,
                                          index_t flag                         = 0,
                                          bool_constant<oob_conditional_check> = {})
{
    constexpr index_t bytes = sizeof(T) * N;

    // Used to catch the cases when src_immediate_addr_offset is NOT 0.
    // Remove this assert once other sizes are implemented.
    assert(src_immediate_addr_offset == 0 &&
           "wrong! not implemented src_immediate_addr_offset size, only 0 supported");
    ignore = src_immediate_addr_offset;

#if defined(__gfx950__)
    static_assert(bytes == 4 || bytes == 12 || bytes == 16,
                  "wrong! only support in dword, dwordx3, dwordx4");
    src_wave_addr_offset = 0;
#else
    static_assert(bytes == 4, "wrong! not implemented vector size");
#endif

    // Set up v_offset:
    index_t v_offset = src_thread_addr_offset;
    if constexpr(oob_conditional_check)
        v_offset = flag ? v_offset : src_wave_buffer_resource[2];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    // Use C-style cast to change address space without dropping llvm noalias attribute
    llvm_amdgcn_raw_buffer_load_lds(src_wave_buffer_resource,
                                    (as3_uint32_ptr)(smem),
                                    bytes,
                                    v_offset,
                                    src_wave_addr_offset,
                                    /*src_immediate_addr_offset*/ 0,
                                    static_cast<index_t>(coherence));
#pragma clang diagnostic pop
}

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl_with_bytes(const thread_buffer<int8_t, N> src_thread_data,
                                                     int32x4_t dst_wave_buffer_resource,
                                                     index_t dst_thread_addr_offset,
                                                     index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        llvm_amdgcn_raw_buffer_store_i8(bit_cast<int8_t>(src_thread_data),
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 4)
    {
        llvm_amdgcn_raw_buffer_store_i32(bit_cast<int32_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 8)
    {
        llvm_amdgcn_raw_buffer_store_i32x2(bit_cast<int32x2_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 16)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(bit_cast<int32x4_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 32)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<0>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<1>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 4,
            static_cast<index_t>(coherence));
    }
    else if constexpr(N == 64)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<0>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<1>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 4,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<2>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 8,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<3>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 12,
            static_cast<index_t>(coherence));
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl(const thread_buffer<T, N> src_thread_data,
                                          int32x4_t dst_wave_buffer_resource,
                                          index_t dst_thread_addr_offset,
                                          index_t dst_wave_addr_offset)
{
    static_assert(
        (std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int32_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, uint16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, uint8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    if constexpr(std::is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<fp32x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(
                src_thread_data.template get_as<fp32x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));
            llvm_amdgcn_raw_buffer_store_fp32x4(
                src_thread_data.template get_as<fp32x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(float),
                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp16(bit_cast<_Float16>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp16x2(bit_cast<fp16x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp16x4(bit_cast<fp16x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
#if 0
            thread_buffer<fp16_t, 8> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.template get_as<fp16x4_t>()[number<0>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.template get_as<fp16x4_t>()[number<1>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset + 4 * sizeof(fp16_t),
                                                static_cast<index_t>(coherence));
#else
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<fp32x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
#endif
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i16x2(bit_cast<int16x2_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(bit_cast<int16x4_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(
                src_thread_data.template get_as<int16x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_i16x4(
                src_thread_data.template get_as<int16x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(bf16_t),
                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, uint16_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_ui16(bit_cast<uint16_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_ui16x2(bit_cast<uint16x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_ui16x4(bit_cast<uint16x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_ui16x4(
                src_thread_data.template get_as<uint16x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_ui16x4(
                src_thread_data.template get_as<uint16x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(uint16_t),
                static_cast<index_t>(coherence));
        }
    }
    else
    {
        using r_t = thread_buffer<int8_t, sizeof(T) * N>;

        amd_buffer_store_impl_with_bytes<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                                   dst_wave_buffer_resource,
                                                                   dst_thread_addr_offset,
                                                                   dst_wave_addr_offset);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store_raw_impl(const thread_buffer<T, N>& dst_thread_data,
                                              int32x4_t dst_wave_buffer_resource,
                                              index_t dst_thread_addr_offset,
                                              index_t dst_wave_addr_offset,
                                              index_t dst_linear_addr_offset,
                                              index_t is_valid_element = 1)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_store instruction");

    using type = thread_buffer<T, N>;
    if constexpr(oob_conditional_check)
    {
        buffer_store_if<sizeof(type)>{}(dst_thread_data,
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        dst_linear_addr_offset,
                                        is_valid_element);
    }
    else
    {
        buffer_store<sizeof(type)>{}(dst_thread_data,
                                     dst_wave_buffer_resource,
                                     dst_thread_addr_offset,
                                     dst_wave_addr_offset,
                                     dst_linear_addr_offset);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_add_impl(const thread_buffer<T, N>& src_thread_data,
                                               int32x4_t dst_wave_buffer_resource,
                                               index_t dst_thread_addr_offset,
                                               index_t dst_wave_addr_offset)
{
    static_assert((std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same<T, fp16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, bf16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(bit_cast<float>(src_thread_data),
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(float),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(float),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(float),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(float),
                0);
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(bit_cast<fp16x2_t>(src_thread_data),
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
                    src_thread_data.template get_as<fp16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(fp16x2_t),
                    0);
            });
        }
        else if constexpr(N == 8)
        {
            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
                    src_thread_data.template get_as<fp16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(fp16x2_t),
                    0);
            });
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_bf16x2(bit_cast<bf16x2_t>(src_thread_data),
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_bf16x2(
                    src_thread_data.template get_as<bf16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(bf16x2_t),
                    0);
            });
        }
        else if constexpr(N == 8)
        {
            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_bf16x2(
                    src_thread_data.template get_as<bf16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(bf16x2_t),
                    0);
            });
        }
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(bit_cast<int32_t>(src_thread_data),
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(int32_t),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(int32_t),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(int32_t),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(int32_t),
                0);
        }
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_max_impl(const thread_buffer<T, N> src_thread_data,
                                               int32x4_t dst_wave_buffer_resource,
                                               index_t dst_thread_addr_offset,
                                               index_t dst_wave_addr_offset)
{
    static_assert((std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(bit_cast<double>(src_thread_data),
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(double),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(double),
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(double),
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(double),
                0);
        }
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
//   oob_conditional_check : dynamic check if out-of-bound
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return src_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    return amd_buffer_load_impl<T, N, coherence>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);
#else
    thread_buffer<T, N> tmp =
        amd_buffer_load_impl<T, N, coherence>(src_wave_buffer_resource, src_thread_addr_offset, 0);
    if constexpr(oob_conditional_check)
        return src_thread_element_valid ? tmp : thread_buffer<T, N>{numeric<T>::zero()};
    else
        return tmp;
#endif
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    thread_buffer<T, N> tmp =
        amd_buffer_load_impl<T, N, coherence>(src_wave_buffer_resource, src_thread_addr_offset, 0);

    if constexpr(oob_conditional_check)
        return src_thread_element_valid ? tmp : thread_buffer<T, N>{customized_value};
    else
        return tmp;
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw(thread_buffer<T, N>& dst,
                                        const T* p_src_wave,
                                        index_t src_thread_element_offset,
                                        index_t src_linear_element_offset,
                                        index_t src_element_space_size,
                                        index_t is_valid_element = 0,
                                        bool_constant<pre_nop>   = {})
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_buffer_load_raw_impl<T, N, coherence, oob_conditional_check, pre_nop>(
        dst,
        src_wave_buffer_resource,
        src_thread_addr_offset,
        0,
        src_linear_addr_offset,
        is_valid_element,
        bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw(thread_buffer<T, N>& dst,
                                        const int32x4_t src_wave_buffer_resource,
                                        index_t src_thread_element_offset,
                                        index_t src_linear_element_offset,
                                        index_t is_valid_element = 0,
                                        bool_constant<pre_nop>   = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_buffer_load_raw_impl<T, N, coherence, oob_conditional_check, pre_nop>(
        dst,
        src_wave_buffer_resource,
        src_thread_addr_offset,
        0,
        src_linear_addr_offset,
        is_valid_element,
        bool_constant<pre_nop>{});
}

// unfortunately async copy can not make sure invalid data is zero inside LDS
// ... unless people manually write zero to LDS at the proper address.
// so not support invalid_element check for now.
// buffer_load OOB still working.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob_raw(T* smem,
                                                       const T* p_src_wave,
                                                       index_t src_thread_element_offset,
                                                       index_t src_linear_element_offset,
                                                       index_t src_element_space_size,
                                                       bool_constant<pre_nop> = {})
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(smem,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                0,
                                                src_linear_addr_offset,
                                                bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob_raw(T* smem,
                                                       const int32x4_t src_wave_buffer_resource,
                                                       index_t src_thread_element_offset,
                                                       index_t src_linear_element_offset,
                                                       bool_constant<pre_nop> = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(smem,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                0,
                                                src_linear_addr_offset,
                                                bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob(CK_TILE_LDS_ADDR T* smem,
                                                   const int32x4_t src_wave_buffer_resource,
                                                   index_t src_thread_element_offset,
                                                   index_t src_linear_element_offset,
                                                   bool is_valid_element,
                                                   bool_constant<oob_conditional_check> = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load<T, N, coherence>(smem,
                                           src_wave_buffer_resource,
                                           src_thread_addr_offset,
                                           0,
                                           src_linear_addr_offset,
                                           is_valid_element,
                                           bool_constant<oob_conditional_check>{});
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store(const thread_buffer<T, N>& src_thread_data,
                                     T* p_dst_wave,
                                     const index_t dst_thread_element_offset,
                                     const bool dst_thread_element_valid,
                                     const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return dst_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    amd_buffer_store_impl<T, N, coherence>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if constexpr(oob_conditional_check)
    {
        if(dst_thread_element_valid)
        {
            amd_buffer_store_impl<T, N, coherence>(
                src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
        }
    }
    else
    {
        amd_buffer_store_impl<T, N, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store_raw(const thread_buffer<T, N>& src_thread_data,
                                         T* p_dst_wave,
                                         const index_t dst_thread_element_offset,
                                         const index_t dst_linear_element_offset,
                                         const bool dst_thread_element_valid,
                                         const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);
    index_t dst_linear_addr_offset = dst_linear_element_offset * sizeof(T);

    amd_buffer_store_raw_impl<T, N, coherence, oob_conditional_check>(src_thread_data,
                                                                      dst_wave_buffer_resource,
                                                                      dst_thread_addr_offset,
                                                                      0,
                                                                      dst_linear_addr_offset,
                                                                      dst_thread_element_valid);
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_add(const thread_buffer<T, N>& src_thread_data,
                                          T* p_dst_wave,
                                          const index_t dst_thread_element_offset,
                                          const bool dst_thread_element_valid,
                                          const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_add_impl<T, N>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_add_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_atomic_add_raw(const thread_buffer<T, N>& src_thread_data,
                                              T* p_dst_wave,
                                              const index_t dst_thread_element_offset,
                                              const index_t dst_linear_element_offset,
                                              const bool dst_thread_element_valid,
                                              const index_t dst_element_space_size,
                                              bool_constant<pre_nop> = {})
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);
    index_t dst_linear_addr_offset = dst_linear_element_offset * sizeof(T);

    if constexpr(oob_conditional_check)
    {
        buffer_atomic_add_if<T, N, pre_nop>{}(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              0,
                                              dst_linear_addr_offset,
                                              dst_thread_element_valid);
    }
    else
    {
        buffer_atomic_add<T, N, pre_nop>{}(src_thread_data,
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           0,
                                           dst_linear_addr_offset,
                                           1);
    }
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_max(const thread_buffer<T, N>& src_thread_data,
                                          T* p_dst_wave,
                                          const index_t dst_thread_element_offset,
                                          const bool dst_thread_element_valid,
                                          const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<T, N>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

#if defined(__gfx950__)
template <typename T, index_t N>
__device__ auto amd_transpose_load_to_vgpr(const T* __restrict__ in_ptr)
{
#define __LDS_ADDR __attribute__((address_space(3)))

    static_assert(__has_builtin(__builtin_amdgcn_raw_buffer_load_b32),
                  "We need to have the compatible compiler version to build this instruction");

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    // Use C-style cast to change address space without dropping llvm noalias attribute
    const auto in_ptr_ = (__LDS_ADDR T*)(const_cast<T*>(in_ptr));
#pragma clang diagnostic pop
    if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::half_t>)
    {
        typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 llvm_fp16x4_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_fp16x4_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr16_b64_v4f16(lds_ptr));
    }
    else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::bf16_t>)
    {
        typedef __attribute__((__vector_size__(4 * sizeof(__bf16)))) __bf16 llvm_bf16x4_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_bf16x4_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr16_b64_v4bf16(lds_ptr));
    }
    else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::fp8_t> ||
                      std::is_same_v<remove_cvref_t<T>, ck_tile::bf8_t> ||
                      std::is_same_v<remove_cvref_t<T>, ck_tile::int8_t>)
    {
        typedef __attribute__((__vector_size__(2 * sizeof(index_t)))) index_t llvm_i32x2_t;
        auto lds_ptr = reinterpret_cast<__LDS_ADDR llvm_i32x2_t*>(in_ptr_);
        return bit_cast<thread_buffer<T, N>>(__builtin_amdgcn_ds_read_tr8_b64_v2i32(lds_ptr));
    }
    else
    {
        static_assert(false, "not implemented");
    }
#undef __LDS_ADDR
}
#endif

} // namespace ck_tile

#endif // !CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
