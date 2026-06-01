// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <memory>
#include <numeric>
#include <span>
#include <concepts>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/device_tensor_generator.hpp"

namespace ck_tile::builder::test {

/// @brief Initialize tensor data with a uniform int distribution
///
/// This function initializes a tensor's device memory with random integer data,
/// drawn from a uniform distribution. The initialization is done directly on the
/// GPU. Note that the entire buffer is filled with the specified distribution
/// regardless of whether the layout is packed.
///
/// @tparam DT The data type of the tensor memory to initialize
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param buf The device memory to initialize
/// @param descriptor A tensor descriptor describing the precise layout of the
/// tensor memory.
/// @param min_value The minimum value of the distribution (inclusive).
/// @param max_value The maximum value of the distribution (exclusive).
template <DataType DT, size_t RANK>
void init_tensor_buffer_uniform_int(void* buf,
                                    const TensorDescriptor<DT, RANK>& descriptor,
                                    int min_value,
                                    int max_value)
{
    size_t size = descriptor.get_element_space_size_in_bytes();

    if(max_value - min_value <= 1)
    {
        throw std::runtime_error("Error while filling device tensor with random integer data: max "
                                 "value must be at least 2 greater than min value, otherwise "
                                 "tensor will be filled by a constant value (end is exclusive)");
    }

    using ck_type = factory::internal::DataTypeToCK<DT>::type;

    // we might be asked to generate int values on fp data types that don't have the required
    // precision
    if(static_cast<ck_type>(max_value - 1) == static_cast<ck_type>(min_value))
    {
        throw std::runtime_error("Error while filling device tensor with random integer data: "
                                 "insufficient precision in specified range");
    }
    size_t packed_size = ck::packed_size_v<ck_type>;
    fill_tensor_uniform_rand_int_values<<<256, 256>>>(
        static_cast<ck_type>(buf), min_value, max_value, (size * packed_size) / sizeof(ck_type));
}

/// @brief Initialize tensor data with a uniform float distribution
///
/// This function initializes a tensor's device memory with random floating data,
/// drawn from a uniform distribution. The initialization is done directly on the
/// GPU. Note that the entire buffer is filled with the specified distribution
/// regardless of whether the layout is packed.
///
/// @tparam DT The data type of the tensor memory to initialize
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param buf The device memory to initialize
/// @param descriptor A tensor descriptor describing the precise layout of the
/// tensor memory.
/// @param min_value The minimum value of the distribution (inclusive).
/// @param max_value The maximum value of the distribution (exclusive).
template <DataType DT, size_t RANK>
void init_tensor_buffer_uniform_fp(void* buf,
                                   const TensorDescriptor<DT, RANK>& descriptor,
                                   float min_value,
                                   float max_value)
{
    size_t size = descriptor.get_element_space_size_in_bytes();

    using ck_type = factory::internal::DataTypeToCK<DT>::type;

    size_t packed_size = ck::packed_size_v<ck_type>;
    fill_tensor_uniform_rand_fp_values<<<256, 256>>>(reinterpret_cast<ck_type*>(buf),
                                                     min_value,
                                                     max_value,
                                                     (size * packed_size) / sizeof(ck_type));
}

/// @brief Initialize tensor data with a normal float distribution
///
/// This function initializes a tensor's device memory with random floating data,
/// drawn from a normal distribution. The initialization is done directly on the
/// GPU. Note that the entire buffer is filled with the specified distribution
/// regardless of whether the layout is packed.
///
/// @tparam DT The data type of the tensor memory to initialize
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param buf The device memory to initialize
/// @param descriptor A tensor descriptor describing the precise layout of the
/// tensor memory.
/// @param sigma The standard deviation of the distribution.
/// @param mean The mean of the distribution.
template <DataType DT, size_t RANK>
void init_tensor_buffer_normal_fp(void* buf,
                                  const TensorDescriptor<DT, RANK>& descriptor,
                                  float sigma,
                                  float mean)
{
    size_t size = descriptor.get_element_space_size_in_bytes();

    using ck_type      = factory::internal::DataTypeToCK<DT>::type;
    size_t packed_size = ck::packed_size_v<ck_type>;
    fill_tensor_norm_rand_fp_values<<<256, 256>>>(
        static_cast<ck_type*>(buf), sigma, mean, (size * packed_size) / sizeof(ck_type));
}

} // namespace ck_tile::builder::test
