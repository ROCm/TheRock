// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/amd_address_space.hpp"
#include "ck/utility/dynamic_buffer.hpp"
#include "ck/utility/math.hpp"

namespace ck {

namespace lds_utils {

/** \brief Allocate a given number of buffers in LDS and return them as a tuple.
 *
 * \tparam DataType Data type of elements to be stored in LDS.
 * \tparam NumBuffers Number of buffers to be allocated.
 * \param lds_ptr Address of the beginning of LDS space.
 * \param num_elems_per_buffer Number of elements to allocate per single buffer.
 * \param start_offset_elems Number of elements to move from the start of LDS for the allocation of
 * the first buffer. \param lds_alignment Alignment of every buffer allocation given as a number of
 * elements. \return Tuple of dynamic buffers representing memory allocated in LDS.
 */
template <typename DataType, index_t NumBuffers>
__device__ static auto AllocateLdsBuffers(void* lds_ptr,
                                          int32_t num_elems_per_buffer,
                                          int32_t start_offset_elems,
                                          int32_t lds_alignment)
{
    const DataType* lds_start = static_cast<DataType*>(lds_ptr) + start_offset_elems;
    const int32_t single_buffer_offset =
        math::integer_least_multiple(num_elems_per_buffer, lds_alignment);
    return generate_tuple(
        [&](auto i) {
            const int32_t local_offset = i * single_buffer_offset;
            return make_dynamic_buffer<AddressSpaceEnum::Lds>(lds_start + local_offset,
                                                              num_elems_per_buffer);
        },
        Number<NumBuffers>{});
}

} // namespace lds_utils
} // namespace ck
