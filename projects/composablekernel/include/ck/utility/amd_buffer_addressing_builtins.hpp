// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "data_type.hpp"
#include "ck/utility/amd_buffer_coherence.hpp"

namespace ck {

template <typename T>
union BufferResource
{
    __device__ constexpr BufferResource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    StaticallyIndexedArray<T*, 2> address;
    StaticallyIndexedArray<int32_t, 4> range;
    StaticallyIndexedArray<int32_t, 4> config;
};

template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, index_t element_space_size)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

template <typename T>
__device__ int32x4_t make_wave_buffer_resource_with_default_range(T* p_wave)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = 0xffffffff; // max possible range
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}

template <typename T>
__device__ __amdgpu_buffer_rsrc_t make_wave_buffer_resource_new(T* p_wave,
                                                                index_t element_space_size)
{
    // wavewise base address (64 bit)
    auto p         = const_cast<remove_cv_t<T>*>(p_wave);
    int32_t stride = 0;
    int32_t num    = element_space_size * sizeof(T);
    auto flags     = CK_BUFFER_RESOURCE_3RD_DWORD;

    return __builtin_amdgcn_make_buffer_rsrc(p, stride, num, flags);
}

template <typename T>
__device__ __amdgpu_buffer_rsrc_t make_wave_buffer_resource_with_default_range_new(T* p_wave)
{
    // wavewise base address (64 bit)
    auto p         = const_cast<remove_cv_t<T>*>(p_wave);
    int32_t stride = 0;
    int32_t num    = 0xffffffff;
    auto flags     = CK_BUFFER_RESOURCE_3RD_DWORD;

    return __builtin_amdgcn_make_buffer_rsrc(p, stride, num, flags);
}

// buffer atomic-add fp16
__device__ half2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    half2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");

// buffer atomic-add i32
__device__ int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

// buffer atomic-add fp32
__device__ float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

// buffer atomic-add fp32
__device__ double
llvm_amdgcn_raw_buffer_atomic_max_fp64(double vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64");

template <index_t N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type<int8_t, N>::type
amd_buffer_load_impl_raw(__amdgpu_buffer_rsrc_t src_wave_buffer_resource,
                         index_t src_thread_addr_offset,
                         index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        return __builtin_amdgcn_raw_buffer_load_b8(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        int16_t tmp = __builtin_amdgcn_raw_buffer_load_b16(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));

        return bit_cast<int8x2_t>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = __builtin_amdgcn_raw_buffer_load_b32(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));

        return bit_cast<int8x4_t>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = __builtin_amdgcn_raw_buffer_load_b64(src_wave_buffer_resource,
                                                             src_thread_addr_offset,
                                                             src_wave_addr_offset,
                                                             static_cast<index_t>(coherence));

        return bit_cast<int8x8_t>(tmp);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));
        return bit_cast<int8x16_t>(tmp);
    }
    else if constexpr(N == 32)
    {
        int32x4_t tmp0 = __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                               src_thread_addr_offset,
                                                               src_wave_addr_offset,
                                                               static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset + 4 * sizeof(int32_t),
                                                  static_cast<index_t>(coherence));
        vector_type<int32_t, 8> tmp;

        tmp.AsType<int32x4_t>()(Number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(Number<1>{}) = tmp1;

        return bit_cast<int8x32_t>(tmp);
    }
    else if constexpr(N == 64)
    {
        int32x4_t tmp0 = __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                               src_thread_addr_offset,
                                                               src_wave_addr_offset,
                                                               static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset + 4 * sizeof(int32_t),
                                                  static_cast<index_t>(coherence));
        int32x4_t tmp2 =
            __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset + 8 * sizeof(int32_t),
                                                  static_cast<index_t>(coherence));
        int32x4_t tmp3 =
            __builtin_amdgcn_raw_buffer_load_b128(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset + 12 * sizeof(int32_t),
                                                  static_cast<index_t>(coherence));

        vector_type<int32_t, 16> tmp;

        tmp.AsType<int32x4_t>()(Number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(Number<1>{}) = tmp1;
        tmp.AsType<int32x4_t>()(Number<2>{}) = tmp2;
        tmp.AsType<int32x4_t>()(Number<3>{}) = tmp3;

        return bit_cast<int8x64_t>(tmp);
    }
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type<T, N>::type
amd_buffer_load_impl(__amdgpu_buffer_rsrc_t src_wave_buffer_resource,
                     index_t src_thread_addr_offset,
                     index_t src_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, uint8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, pk_i4_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    using r_t     = typename vector_type<T, N>::type;
    auto raw_data = amd_buffer_load_impl_raw<sizeof(T) * N, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
    return bit_cast<r_t>(raw_data);
}

template <index_t N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void
amd_buffer_store_impl_raw(const typename vector_type<int8_t, N>::type src_thread_data,
                          __amdgpu_buffer_rsrc_t dst_wave_buffer_resource,
                          index_t dst_thread_addr_offset,
                          index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        __builtin_amdgcn_raw_buffer_store_b8(src_thread_data,
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        __builtin_amdgcn_raw_buffer_store_b16(bit_cast<int16_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
    }
    else if constexpr(N == 4)
    {
        __builtin_amdgcn_raw_buffer_store_b32(bit_cast<int32_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
    }
    else if constexpr(N == 8)
    {
        __builtin_amdgcn_raw_buffer_store_b64(bit_cast<int32x2_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
    }
    else if constexpr(N == 16)
    {
        __builtin_amdgcn_raw_buffer_store_b128(bit_cast<int32x4_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
    }
    else if constexpr(N == 32)
    {
        vector_type<int32_t, 8> tmp{bit_cast<int32x8_t>(src_thread_data)};

        __builtin_amdgcn_raw_buffer_store_b128(tmp.template AsType<int32x4_t>()[Number<0>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));

        __builtin_amdgcn_raw_buffer_store_b128(tmp.template AsType<int32x4_t>()[Number<1>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 4,
                                               static_cast<index_t>(coherence));
    }
    else if constexpr(N == 64)
    {
        vector_type<int32_t, 16> tmp{bit_cast<int32x16_t>(src_thread_data)};

        __builtin_amdgcn_raw_buffer_store_b128(tmp.template AsType<int32x4_t>()[Number<0>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));

        __builtin_amdgcn_raw_buffer_store_b128(tmp.template AsType<int32x4_t>()[Number<1>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 4,
                                               static_cast<index_t>(coherence));

        __builtin_amdgcn_raw_buffer_store_b128(tmp.template AsType<int32x4_t>()[Number<2>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 8,
                                               static_cast<index_t>(coherence));

        __builtin_amdgcn_raw_buffer_store_b128(tmp.template AsType<int32x4_t>()[Number<3>{}],
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset + sizeof(int32_t) * 12,
                                               static_cast<index_t>(coherence));
    }
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_buffer_store_impl(const typename vector_type<T, N>::type src_thread_data,
                                      __amdgpu_buffer_rsrc_t dst_wave_buffer_resource,
                                      index_t dst_thread_addr_offset,
                                      index_t dst_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f8_fnuz_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_fnuz_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, fp8_storage_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    using r_t = typename vector_type<int8_t, sizeof(T) * N>::type;

    amd_buffer_store_impl_raw<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                        dst_wave_buffer_resource,
                                                        dst_thread_addr_offset,
                                                        dst_wave_addr_offset);
}

template <typename T, index_t N>
__device__ void amd_global_atomic_add_impl(const typename vector_type<T, N>::type src_thread_data,
                                           T* addr)
{
    static_assert((is_same<T, bhalf_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)),
                  "wrong! not implemented");

    if constexpr(is_same<T, half_t>::value)
    {
        vector_type<half_t, N> tmp{src_thread_data};
        static_for<0, N / 2, 1>{}([&](auto i) {
            __builtin_amdgcn_global_atomic_fadd_v2f16(bit_cast<half2_t*>(addr) + i,
                                                      tmp.template AsType<half2_t>()[i]);
        });
    }
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx12__)
    else if constexpr(is_same<T, bhalf_t>::value)
    {
        vector_type<bhalf_t, N> tmp{src_thread_data};
        static_for<0, N / 2, 1>{}([&](auto i) {
            __builtin_amdgcn_global_atomic_fadd_v2bf16(bit_cast<bhalf2_t*>(addr) + i,
                                                       tmp.template AsType<bhalf2_t>()[i]);
        });
    }
#endif
}

template <typename T, index_t N>
__device__ void amd_buffer_atomic_add_impl(const typename vector_type<T, N>::type src_thread_data,
                                           int32x4_t dst_wave_buffer_resource,
                                           index_t dst_thread_addr_offset,
                                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<float, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<float, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(float),
                                                   0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(src_thread_data,
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            vector_type<half_t, 4> tmp{src_thread_data};

            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
        else if constexpr(N == 8)
        {
            vector_type<half_t, 8> tmp{src_thread_data};

            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(src_thread_data,
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else if constexpr(N == 2)
        {
            vector_type<int32_t, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);
        }
        else if constexpr(N == 4)
        {
            vector_type<int32_t, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<2>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 2 * sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<3>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 3 * sizeof(int32_t),
                                                  0);
        }
    }
}

template <typename T, index_t N>
__device__ void amd_buffer_atomic_max_impl(const typename vector_type<T, N>::type src_thread_data,
                                           int32x4_t dst_wave_buffer_resource,
                                           index_t dst_thread_addr_offset,
                                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<double, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<double, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<3>{}],
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
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const __amdgpu_buffer_rsrc_t src_wave_buffer_resource =
        make_wave_buffer_resource_new(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_element_valid ? 0 : 0x80000000;
    return amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);

#else

    vector_t tmp{amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0)};
    return src_thread_element_valid ? tmp : vector_t(0);
#endif
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const __amdgpu_buffer_rsrc_t src_wave_buffer_resource =
        make_wave_buffer_resource_new(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    vector_t tmp{amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0)};

    return src_thread_element_valid ? tmp : vector_t(customized_value);
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_buffer_store(const typename vector_type_maker<T, N>::type::type src_thread_data,
                                 T* p_dst_wave,
                                 const index_t dst_thread_element_offset,
                                 const bool dst_thread_element_valid,
                                 const index_t dst_element_space_size)
{
    const __amdgpu_buffer_rsrc_t dst_wave_buffer_resource =
        make_wave_buffer_resource_new(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;
    amd_buffer_store_impl<scalar_t, vector_size, coherence>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_store_impl<scalar_t, vector_size, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void
amd_buffer_atomic_add(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    if constexpr(is_same<T, bhalf_t>::value)
    {
        if(dst_thread_element_valid)
        {
            amd_global_atomic_add_impl<scalar_t, vector_size>(
                src_thread_data, p_dst_wave + dst_thread_element_offset);
        }
    }
    else
    {
#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
        uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

        amd_buffer_atomic_add_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
        if(dst_thread_element_valid)
        {
            amd_buffer_atomic_add_impl<scalar_t, vector_size>(
                src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
        }
#endif
    }
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void
amd_buffer_atomic_max(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// Direct loads from global to LDS.
__device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

#ifndef __HIPCC_RTC__
template <typename T, index_t NumElemsPerThread>
__device__ void amd_direct_load_global_to_lds(const T* global_base_ptr,
                                              const index_t global_offset,
                                              T* lds_base_ptr,
                                              const index_t lds_offset,
                                              const bool is_valid,
                                              const index_t src_element_space_size)
{
    // Direct loads require that each thread reads and writes a multiple of DWORDs (4 bytes).
    // For gfx950: supports 1, 3, or 4 DWORDs per thread
    // For gfx942: supports exactly 1 DWORD per thread
    constexpr auto bytes_per_thread = sizeof(T) * NumElemsPerThread;
#if defined(__gfx950__)
    constexpr auto dword_bytes = 4;
    static_assert(bytes_per_thread == dword_bytes || bytes_per_thread == dword_bytes * 3 ||
                  bytes_per_thread == dword_bytes * 4);
#elif defined(__gfx942__)
    constexpr auto dword_bytes = 4;
    static_assert(bytes_per_thread == dword_bytes);
#endif

    const int32x4_t src_resource =
        make_wave_buffer_resource(global_base_ptr, src_element_space_size);
    const index_t global_offset_bytes = is_valid ? global_offset * sizeof(T) : 0x80000000;

#if CK_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    T* lds_ptr = lds_base_ptr + lds_offset;
#ifndef CK_CODE_GEN_RTC
    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(lds_ptr)));
#else
    auto const lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<size_t>(lds_ptr)));
#endif
    asm volatile("s_mov_b32 m0, %0; \n\t"
                 "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
                 "v"(global_offset_bytes),
                 "s"(src_resource)
                 : "memory");
#else
    // LDS pointer must be attributed with the LDS address space.
    __attribute__((address_space(3))) uint32_t* lds_ptr =
#ifndef CK_CODE_GEN_RTC
        reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
            reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));
#else
        reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
            reinterpret_cast<size_t>(lds_base_ptr + lds_offset));
#endif

    llvm_amdgcn_raw_buffer_load_lds(
        src_resource, lds_ptr, bytes_per_thread, global_offset_bytes, 0, 0, 0);
#endif
}
#endif

} // namespace ck
