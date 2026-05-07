// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/utility/bit_cast.hpp"
namespace ck_tile {

/**
 * @brief Permute packed int4 vectors for device implementation compatibility
 *
 * This function transforms 4 pk_int4_t values from original layout to hardware-optimized layout:
 * - Original layout (4 pk_int4_t): 0x76543210
 * - Transformed layout (4 pk_int4_t): 0x75316420
 *
 * Each pk_int4_t contains two 4-bit values packed in the high and low nibbles of an int8_t
 *
 * Example:
 * - Input:  0x76, 0x54, 0x32, 0x10
 * - Output: 0x75, 0x31, 0x64, 0x20
 *
 * @note Input tensor length must be a multiple of 4
 *
 * This transformation is required before transferring B matrix data (of type pk_int4_t) to device.
 * The device conversion functions (i4_to_half4, i4_to_bhalf4, amd_assembly_i4_to_fp8x8,
 * amd_assembly_i4_to_bf8x8) require data in 0x75316420 order to correctly convert pk_int4_t to
 * other numeric types.
 */
template <typename Tensor>
void permute_vectors_i4x4_b(Tensor& tensor)
{
    auto tensor_row_buf = tensor.data();
    for(size_t idx = 0; idx < tensor.size(); idx += 4)
    {
        int8_t input[8];

        for(int k = 0; k < 4; k++)
        {
            int8_t i4x2      = bit_cast<int8_t>(tensor_row_buf[idx + k]);
            input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
            input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
        }

        // permute 0x76543210 => 0x75316420
        {
            int8_t hi   = input[2];
            int8_t lo   = input[0];
            int8_t i4x2 = (hi << 4) | lo;

            tensor_row_buf[idx + 0] = bit_cast<pk_int4_t>(i4x2);
        }

        {
            int8_t hi   = input[6];
            int8_t lo   = input[4];
            int8_t i4x2 = (hi << 4) | lo;

            tensor_row_buf[idx + 1] = bit_cast<pk_int4_t>(i4x2);
        }

        {
            int8_t hi   = input[3];
            int8_t lo   = input[1];
            int8_t i4x2 = (hi << 4) | lo;

            tensor_row_buf[idx + 2] = bit_cast<pk_int4_t>(i4x2);
        }

        {
            int8_t hi   = input[7];
            int8_t lo   = input[5];
            int8_t i4x2 = (hi << 4) | lo;

            tensor_row_buf[idx + 3] = bit_cast<pk_int4_t>(i4x2);
        }
    }
}

} // namespace ck_tile
