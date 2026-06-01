// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "data_type.hpp"

namespace ck {

#if defined(__gfx12__)
template <typename T>
__device__ auto amd_global_load_transpose_to_vgpr(const T* in_ptr)
{
    using vector_t = typename vector_type<T, 8>::type;
    if constexpr(sizeof(T) == 2)
    {
        typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 llvm_fp16x8_t;
        __attribute__((address_space(1))) llvm_fp16x8_t* glb_ptr =
            reinterpret_cast<__attribute__((address_space(1))) llvm_fp16x8_t*>(
                reinterpret_cast<uintptr_t>(in_ptr));
        return bit_cast<vector_t>(__builtin_amdgcn_global_load_tr_b128_v8f16(glb_ptr));
    }
    else if constexpr(sizeof(T) == 1)
    {
        typedef __attribute__((__vector_size__(2 * sizeof(int)))) int llvm_intx2_t;
        __attribute__((address_space(1))) llvm_intx2_t* glb_ptr =
            reinterpret_cast<__attribute__((address_space(1))) llvm_intx2_t*>(
                reinterpret_cast<uintptr_t>(in_ptr));
        return bit_cast<vector_t>(__builtin_amdgcn_global_load_tr_b64_v2i32(glb_ptr));
    }
    else
    {
        static_assert(false, "not implemented");
    }
}
#endif

} // namespace ck
