// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <ck/utility/ignore.hpp>
#include <ck/utility/type.hpp>
#include "ck/ck.hpp"

#ifdef CK_CODE_GEN_RTC
using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
#endif
namespace ck {

// Pseudo random number generator
// version for fp32
template <typename T, uint32_t seed_t, ck::enable_if_t<is_same<float, T>{}, bool> = false>
__host__ __device__ uint32_t prand_generator(index_t id, T val, uint32_t seed = seed_t)
{
    uint32_t x         = bit_cast<uint32_t>(val);
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    drop_bits ^= x >> 16;
    drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    // NOTE: If id is in 64 bit, we are only using lower 32 bit.
    //       So, it can have an effect of using same id for multiple elements when the id is very
    //       large!
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (id * 229791) ^ seed);
    return rng;
}

// version for fp16
template <typename T, uint32_t seed_t, ck::enable_if_t<is_same<_Float16, T>{}, bool> = false>
__host__ __device__ uint32_t prand_generator(index_t id, T val, uint32_t seed = seed_t)
{
    uint16_t x         = bit_cast<uint16_t>(val);
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    drop_bits          = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    // NOTE: If id is in 64 bit, we are only using lower 32 bit.
    //       So, it can have an effect of using same id for multiple elements when the id is very
    //       large!
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (id * 229791) ^ seed);
    return rng;
}

// return 0 if data is not fp16 or fp32
template <typename T,
          uint32_t seed_t,
          ck::enable_if_t<!(is_same<float, T>{} || is_same<_Float16, T>{}), bool> = false>
__host__ __device__ uint32_t prand_generator(int id, T val, uint32_t seed = seed_t)
{
    ck::ignore = id;
    ck::ignore = val;
    ck::ignore = seed;

    return 0;
}

} // namespace ck
