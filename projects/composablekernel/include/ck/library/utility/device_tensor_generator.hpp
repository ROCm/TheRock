// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/library/utility/device_tensor_generator.hpp"
#include "ck/utility/data_type.hpp"

// use xorshift for now since it is simple. Should be suitable enough, but feel free to switch in
// the future
struct ran_state_u32
{
    uint32_t s[4];
};

__device__ uint32_t ran_gen_round_u32(ran_state_u32& state)
{
    uint32_t tmp = state.s[3];
    state.s[3]   = state.s[2];
    state.s[2]   = state.s[1];
    state.s[1]   = state.s[0];
    tmp ^= tmp << 11;
    tmp ^= tmp >> 8;
    state.s[0] = tmp ^ state.s[0] ^ (state.s[0] >> 19);
    return state.s[0];
}

__device__ ran_state_u32 ran_init(uint32_t seed = 0)
{
    ran_state_u32 state;
    // use primes for initialization
    state.s[0] = (blockDim.x * blockIdx.x + threadIdx.x) * 8912741 + 2313212 + seed;
    state.s[1] =
        (gridDim.x * blockDim.x - (blockDim.x * blockIdx.x + threadIdx.x)) * 5013829 + 6012697;
    state.s[2] = (blockDim.x * blockIdx.x + threadIdx.x) * 3412309 + 2912479;
    state.s[3] =
        (gridDim.x * blockDim.x - (blockDim.x * blockIdx.x + threadIdx.x)) * 1001447 + 9912307;

    // run 20 rounds
    for(int i = 0; i < 20; i++)
    {
        ran_gen_round_u32(state);
    }
    return state;
}

template <typename T>
__global__ void fill_tensor_uniform_rand_int_values(T* p,
                                                    int min_value,
                                                    int max_value,
                                                    uint64_t buffer_element_size)
{
    // initial values
    ran_state_u32 s = ran_init();
    for(uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < buffer_element_size / ck::packed_size_v<T>;
        i += blockDim.x * gridDim.x)
    {
        if constexpr(ck::is_same_v<T, ck::pk_i4_t>)
        {
            uint8_t hi      = ((ran_gen_round_u32(s)) % (max_value - min_value)) + min_value + 8;
            uint8_t lo      = ((ran_gen_round_u32(s)) % (max_value - min_value)) + min_value + 8;
            ck::pk_i4_t res = ((hi & 0xf) << 4) + (lo & 0xf);
            p[i]            = res;
        }
        else
        {
            p[i] = ck::type_convert<T, int>(
                static_cast<int>((ran_gen_round_u32(s)) % (max_value - min_value)) + min_value);
        }
    }
}

template <typename T>
__global__ void fill_tensor_uniform_rand_fp_values(T* p,
                                                   float min_value,
                                                   float max_value,
                                                   uint64_t buffer_element_size)
{
    // initial values
    ran_state_u32 s = ran_init();
    for(uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < buffer_element_size / ck::packed_size_v<T>;
        i += blockDim.x * gridDim.x)
    {
        if constexpr(ck::is_same_v<T, ck::f4x2_pk_t>)
        {
            float u1 =
                ran_gen_round_u32(s) * (1.0f / 4294967296.0f) * (max_value - min_value) + min_value;
            float u2 =
                ran_gen_round_u32(s) * (1.0f / 4294967296.0f) * (max_value - min_value) + min_value;

            p[i] = ck::type_convert<ck::f4x2_t>(ck::float2_t{u1, u2});
        }
        else
        {
            float ran = ran_gen_round_u32(s) * (1.0f / 4294967296.0f);
            p[i]      = ck::type_convert<T, float>(ran * (max_value - min_value) + min_value);
        }
    }
}

template <typename T>
__global__ void
fill_tensor_norm_rand_fp_values(T* p, float sigma, float mean, uint64_t buffer_element_size)
{
    static constexpr float PI = 3.141592653f;
    // initial values
    ran_state_u32 s = ran_init();
    float norm[2];
    for(uint64_t i = blockIdx.x * blockDim.x + threadIdx.x, j = 0; i < buffer_element_size;
        i += blockDim.x * gridDim.x, j++)
    {
        if(j % (2 / ck::packed_size_v<T>) == 0)
        {
            float u1    = ran_gen_round_u32(s) * (1.0f / 4294967296.0f);
            float u2    = ran_gen_round_u32(s) * (1.0f / 4294967296.0f);
            float scale = sigma * ck::math::sqrt(-2.0f * ck::math::log(u1));
            norm[0]     = scale * ck::math::cos(2.0f * PI * u2) + mean;
            norm[1]     = scale * ck::math::sin(2.0f * PI * u2) + mean;
        }

        if constexpr(ck::is_same_v<T, ck::f4x2_pk_t>)
        {
            p[i] = ck::type_convert<ck::f4x2_t>(ck::float2_t{norm[0], norm[1]});
        }
        else
        {
            p[i] = ck::type_convert<T, float>(norm[j % 2]);
        }
    }
}
