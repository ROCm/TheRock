// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType, typename ComputeDataType, typename YDataType, typename ReduceOp>
CK_TILE_HOST void
reference_reduce(const HostTensor<XDataType>& x_m_n, HostTensor<YDataType>& y_m, ReduceOp reduce_op)
{
    auto f = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        for(int n = 0; n < N; ++n)
        {
            const ComputeDataType v_a = type_convert<ComputeDataType>(x_m_n(m, n));

            v_acc = reduce_op(v_acc, v_a);
        }

        y_m(m) = ck_tile::type_convert<YDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, y_m.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}

// Generic reference reduce for arbitrary dimensions
template <
    typename XDataType,
    typename ComputeDataType,
    typename YDataType,
    typename ReduceOp,
    typename KeptDim, // Expected type: ck_tile::sequence<...> containing dimension indices to keep
    typename ReduceDims> // Expected type: ck_tile::sequence<...> containing dimension indices to
                         // reduce
CK_TILE_HOST void reference_reduce(const HostTensor<XDataType>& x_tensor,
                                   HostTensor<YDataType>& y_tensor,
                                   ReduceOp reduce_op,
                                   KeptDim kept_dim,
                                   ReduceDims reduce_dims)
{
    const auto& x_lengths = x_tensor.mDesc.get_lengths();

    // Calculate total kept elements (product of all kept dimension lengths)
    index_t total_kept_elements = 1;
    static_for<0, kept_dim.size(), 1>{}(
        [&](auto i) { total_kept_elements *= x_lengths[kept_dim.at(i)]; });

    // Calculate total reduce elements (product of all reduce dimension lengths)
    index_t total_reduce_elements = 1;
    static_for<0, reduce_dims.size(), 1>{}(
        [&](auto i) { total_reduce_elements *= x_lengths[reduce_dims.at(i)]; });

    auto f = [&](auto linear_kept_idx) {
        ComputeDataType v_acc = reduce_op.template GetIdentityValue<ComputeDataType>();

        // Convert linear kept index to multi-dimensional kept indices
        std::vector<index_t> kept_indices(kept_dim.size());
        index_t temp_kept = linear_kept_idx;
        static_for<0, kept_dim.size(), 1>{}([&](auto i) {
            constexpr auto dim_idx = kept_dim.size() - 1 - i;
            constexpr auto dim     = kept_dim.at(dim_idx);
            const auto len         = x_lengths[dim];
            kept_indices[dim_idx]  = temp_kept % len;
            temp_kept /= len;
        });

        for(index_t reduce_idx = 0; reduce_idx < total_reduce_elements; ++reduce_idx)
        {
            // Convert linear reduce index to multi-dimensional reduce indices
            std::vector<index_t> reduce_indices(reduce_dims.size());
            index_t temp_reduce = reduce_idx;
            static_for<0, reduce_dims.size(), 1>{}([&](auto i) {
                constexpr auto dim_idx  = reduce_dims.size() - 1 - i;
                constexpr auto dim      = reduce_dims.at(dim_idx);
                const auto len          = x_lengths[dim];
                reduce_indices[dim_idx] = temp_reduce % len;
                temp_reduce /= len;
            });

            // Build full input tensor indices by combining kept and reduce indices
            std::vector<std::size_t> full_indices(x_lengths.size(), 0);
            static_for<0, kept_dim.size(), 1>{}(
                [&](auto i) { full_indices[kept_dim.at(i)] = kept_indices[i]; });
            static_for<0, reduce_dims.size(), 1>{}(
                [&](auto i) { full_indices[reduce_dims.at(i)] = reduce_indices[i]; });

            // Access input tensor element
            const auto v_a = type_convert<ComputeDataType>(x_tensor(full_indices));

            v_acc = reduce_op(v_acc, v_a);
        }

        // Calculate output tensor index using kept indices
        // The output tensor has the same structure as the kept dimensions
        std::vector<std::size_t> y_indices(kept_dim.size());
        static_for<0, kept_dim.size(), 1>{}([&](auto i) { y_indices[i] = kept_indices[i]; });

        y_tensor(y_indices) = type_convert<YDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f, total_kept_elements)(std::thread::hardware_concurrency());
}

template <typename XDataType,
          typename ComputeDataType,
          typename YDataType,
          typename YRefTuple,
          typename ReduceOps, // Expected type: ck_tile::tuple<...> containing reduce operations
          typename KeptDim, // Expected type: ck_tile::sequence<...> containing dimension indices to
                            // keep
          typename ReduceDims, // Expected type: ck_tile::sequence<...> containing dimension indices
                               // to reduce
          typename ElementWiseOps,
          typename AccElementWiseOps>
CK_TILE_HOST void reference_multiple_reduce(const HostTensor<XDataType>& x_tensor,
                                            YRefTuple& y_tensor_tuple,
                                            ReduceOps reduce_ops,
                                            KeptDim kept_dim,
                                            ReduceDims reduce_dims,
                                            ElementWiseOps elementwise_ops,
                                            AccElementWiseOps accumulator_ops)
{
    const auto& x_lengths = x_tensor.mDesc.get_lengths();

    // Calculate total kept elements (product of all kept dimension lengths)
    index_t total_kept_elements = 1;
    static_for<0, kept_dim.size(), 1>{}(
        [&](auto i) { total_kept_elements *= x_lengths[kept_dim.at(i)]; });

    // Calculate total reduce elements (product of all reduce dimension lengths)
    index_t total_reduce_elements = 1;
    static_for<0, reduce_dims.size(), 1>{}(
        [&](auto i) { total_reduce_elements *= x_lengths[reduce_dims.at(i)]; });

    auto f = [&](auto linear_kept_idx) {
        // Initialize accumulators for each reduction operation
        auto v_acc_tuple = ck_tile::generate_tuple(
            [&](auto i) {
                return reduce_ops.template at<i>().template GetIdentityValue<ComputeDataType>();
            },
            number<reduce_ops.size()>{});

        // Convert linear kept index to multi-dimensional kept indices
        std::vector<index_t> kept_indices(kept_dim.size());
        index_t temp_kept = linear_kept_idx;
        static_for<0, kept_dim.size(), 1>{}([&](auto i) {
            constexpr auto dim_idx = kept_dim.size() - 1 - i;
            constexpr auto dim     = kept_dim.at(dim_idx);
            const auto len         = x_lengths[dim];
            kept_indices[dim_idx]  = temp_kept % len;
            temp_kept /= len;
        });

        for(index_t reduce_idx = 0; reduce_idx < total_reduce_elements; ++reduce_idx)
        {
            // Convert linear reduce index to multi-dimensional reduce indices
            std::vector<index_t> reduce_indices(reduce_dims.size());
            index_t temp_reduce = reduce_idx;
            static_for<0, reduce_dims.size(), 1>{}([&](auto i) {
                constexpr auto dim_idx  = reduce_dims.size() - 1 - i;
                constexpr auto dim      = reduce_dims.at(dim_idx);
                const auto len          = x_lengths[dim];
                reduce_indices[dim_idx] = temp_reduce % len;
                temp_reduce /= len;
            });

            // Build full input tensor indices by combining kept and reduce indices
            std::vector<std::size_t> full_indices(x_lengths.size(), 0);
            static_for<0, kept_dim.size(), 1>{}(
                [&](auto i) { full_indices[kept_dim.at(i)] = kept_indices[i]; });
            static_for<0, reduce_dims.size(), 1>{}(
                [&](auto i) { full_indices[reduce_dims.at(i)] = reduce_indices[i]; });

            // Access input tensor element
            auto v_a = type_convert<ComputeDataType>(x_tensor(full_indices));

            // Apply each reduction operation
            static_for<0, reduce_ops.size(), 1>{}([&](auto i) {
                // Apply element-wise operation before reduction
                elementwise_ops.at(i)(v_a, v_a);

                v_acc_tuple.template at<i>() =
                    reduce_ops.template at<i>()(v_acc_tuple.template at<i>(), v_a);
            });
        }

        static_for<0, reduce_ops.size(), 1>{}([&](auto i) {
            // Apply accumulator element-wise operation after reduction
            accumulator_ops.at(i)(v_acc_tuple.template at<i>(), v_acc_tuple.template at<i>());
        });

        // Calculate output tensor index using kept indices
        // The output tensor has the same structure as the kept dimensions
        std::vector<std::size_t> y_indices(kept_dim.size());
        static_for<0, kept_dim.size(), 1>{}([&](auto i) { y_indices[i] = kept_indices[i]; });

        // Store results for each reduction operation in the output tensor
        static_for<0, reduce_ops.size(), 1>{}([&](auto i) {
            y_tensor_tuple.template at<i>()(y_indices) =
                type_convert<YDataType>(v_acc_tuple.template at<i>());
        });
    };

    make_ParallelTensorFunctor(f, total_kept_elements)(std::thread::hardware_concurrency());
}

template <typename XDataType,
          typename ComputeDataType,
          typename YDataType,
          typename YRefTuple,
          typename ReduceOps, // Expected type: ck_tile::tuple<...> containing reduce operations
          typename KeptDim, // Expected type: ck_tile::sequence<...> containing dimension indices to
                            // keep
          typename ReduceDims, // Expected type: ck_tile::sequence<...> containing dimension indices
                               // to reduce
          typename ElementWiseOps,
          typename AccElementWiseOps,
          typename InterBlockReduceOps>
CK_TILE_HOST void reference_multiple_reduce_multiblock(const HostTensor<XDataType>& x_tensor,
                                                       YRefTuple& y_tensor_tuple,
                                                       ReduceOps reduce_ops,
                                                       KeptDim kept_dim,
                                                       ReduceDims reduce_dims,
                                                       ElementWiseOps elementwise_ops,
                                                       AccElementWiseOps accumulator_ops,
                                                       InterBlockReduceOps inter_block_reduce_ops,
                                                       ck_tile::index_t num_blocks)
{
    const auto& x_lengths = x_tensor.mDesc.get_lengths();

    // Calculate total kept elements (product of all kept dimension lengths)
    index_t total_kept_elements = 1;
    static_for<0, kept_dim.size(), 1>{}(
        [&](auto i) { total_kept_elements *= x_lengths[kept_dim.at(i)]; });

    // Calculate total reduce elements (product of all reduce dimension lengths)
    index_t total_reduce_elements = 1;
    static_for<0, reduce_dims.size(), 1>{}(
        [&](auto i) { total_reduce_elements *= x_lengths[reduce_dims.at(i)]; });

    // Initialize output tensors
    static_for<0, reduce_ops.size(), 1>{}([&](auto i) {
        auto& y_tensor = y_tensor_tuple.template at<i>();
        for(auto& val : y_tensor.mData)
        {
            val = inter_block_reduce_ops.template at<i>().template GetIdentityValue<YDataType>();
        }
    });

    auto f = [&](auto linear_kept_idx) {
        // Convert linear kept index to multi-dimensional kept indices
        std::vector<index_t> kept_indices(kept_dim.size());
        index_t temp_kept = linear_kept_idx;
        static_for<0, kept_dim.size(), 1>{}([&](auto i) {
            constexpr auto dim_idx = kept_dim.size() - 1 - i;
            constexpr auto dim     = kept_dim.at(dim_idx);
            const auto len         = x_lengths[dim];
            kept_indices[dim_idx]  = temp_kept % len;
            temp_kept /= len;
        });

        // Calculate output tensor index using kept indices
        std::vector<std::size_t> y_indices(kept_dim.size());
        static_for<0, kept_dim.size(), 1>{}([&](auto i) { y_indices[i] = kept_indices[i]; });

        const auto max_element_per_block = (total_reduce_elements + num_blocks - 1) / num_blocks;

        for(index_t block_id = 0; block_id < num_blocks; ++block_id)
        {
            // Initialize accumulators for each reduction operation for the current block
            auto v_acc_tuple = ck_tile::generate_tuple(
                [&](auto i) {
                    return reduce_ops.template at<i>().template GetIdentityValue<ComputeDataType>();
                },
                number<reduce_ops.size()>{});

            const index_t element_offset = block_id * max_element_per_block;
            const index_t element_end =
                std::min(element_offset + max_element_per_block, total_reduce_elements);

            for(index_t linear_reduce_idx = element_offset; linear_reduce_idx < element_end;
                ++linear_reduce_idx)
            {
                // Convert linear reduce index to multi-dimensional reduce indices
                std::vector<index_t> reduce_indices(reduce_dims.size());
                index_t temp_reduce = linear_reduce_idx;
                static_for<0, reduce_dims.size(), 1>{}([&](auto i) {
                    constexpr auto dim_idx  = reduce_dims.size() - 1 - i;
                    constexpr auto dim      = reduce_dims.at(dim_idx);
                    const auto len          = x_lengths[dim];
                    reduce_indices[dim_idx] = temp_reduce % len;
                    temp_reduce /= len;
                });

                // Build full input tensor indices by combining kept and reduce indices
                std::vector<std::size_t> full_indices(x_lengths.size(), 0);
                static_for<0, kept_dim.size(), 1>{}(
                    [&](auto i) { full_indices[kept_dim.at(i)] = kept_indices[i]; });
                static_for<0, reduce_dims.size(), 1>{}(
                    [&](auto i) { full_indices[reduce_dims.at(i)] = reduce_indices[i]; });

                // Access input tensor element
                const auto v_a_in = type_convert<ComputeDataType>(x_tensor(full_indices));

                // Apply each reduction operation
                static_for<0, reduce_ops.size(), 1>{}([&](auto i) {
                    auto v_a = v_a_in;
                    // Apply element-wise operation before reduction
                    elementwise_ops.at(i)(v_a, v_a);

                    v_acc_tuple.template at<i>() =
                        reduce_ops.template at<i>()(v_acc_tuple.template at<i>(), v_a);
                });
            }

            static_for<0, reduce_ops.size(), 1>{}([&](auto i) {
                // Apply accumulator element-wise operation after reduction
                accumulator_ops.at(i)(v_acc_tuple.template at<i>(), v_acc_tuple.template at<i>());

                // Update the output tensor with the partial result from this block
                auto& y_tensor = y_tensor_tuple.template at<i>();
                auto& y_val    = y_tensor(y_indices);
                y_val          = inter_block_reduce_ops.template at<i>()(
                    y_val, type_convert<YDataType>(v_acc_tuple.template at<i>()));
            });
        }
    };

    make_ParallelTensorFunctor(f, total_kept_elements)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
