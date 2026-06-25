// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

/**
 * @file tensor_descriptor_utils.hpp
 * @brief Utility functions for creating tensor descriptors in batched contraction operations
 *
 * @details This file contains utility functions for creating tensor descriptors with flattened
 * dimensions for GEMM operations. These functions transform multi-dimensional tensors into
 * 2D matrix descriptors by removing batch dimensions and flattening the remaining dimensions.
 *
 * These utilities are used by BatchedContractionKernel to create stride-aware descriptors
 * that support arbitrary multi-dimensional non-contiguous tensor layouts.
 */

namespace ck_tile {

/**
 * @brief Utility class for creating tensor descriptors in batched contraction operations
 *
 * @tparam NumDimG Number of batch dimensions
 * @tparam NumDimM Number of M (output row) dimensions
 * @tparam NumDimN Number of N (output column) dimensions
 * @tparam NumDimK Number of K (contraction) dimensions
 */
template <ck_tile::index_t NumDimG,
          ck_tile::index_t NumDimM,
          ck_tile::index_t NumDimN,
          ck_tile::index_t NumDimK,
          ck_tile::index_t VectorSizeA,
          ck_tile::index_t VectorSizeB,
          ck_tile::index_t VectorSizeE>
struct TensorDescriptorUtils
{
    /// @brief Creates a tensor descriptor for input tensor A with batch dimensions removed.
    /// @param A_dims Dimension vector for tensor A: [G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
    /// @param A_strides Stride vector for tensor A: [G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
    /// @return Flattened tensor descriptor: [M_total, K_total] for GEMM computation
    /// @details Removes batch dimensions and flattens M and K dimensions for efficient GEMM
    /// execution
    CK_TILE_HOST static constexpr auto
    Make_A_GridDescriptor_M_K(const std::vector<ck_tile::index_t>& A_dims    = {},
                              const std::vector<ck_tile::index_t>& A_strides = {})
    {
        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, number<end - start>{});
        };

        // Remove G Dimensions
        const auto A_dims_M_K =
            to_tuple(A_dims, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimK>{});
        const auto A_strides_M_K =
            to_tuple(A_strides, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimK>{});

        // dimension Ids for M and K
        constexpr auto A_dims_M_ids = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};
        constexpr auto A_dims_K_ids =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // Dimensions for M [M0, M1, ...] and K [K0, K1, ...]
        const auto dims_M = get_container_subset(A_dims_M_K, A_dims_M_ids);
        const auto dims_K = get_container_subset(A_dims_M_K, A_dims_K_ids);

        // naive tensor A[M0, M1, M2, ..., K0, K1, K2...] Descriptor with vector size
        const auto A_grid_desc_Ms_Ks = ck_tile::make_naive_tensor_descriptor(
            A_dims_M_K, A_strides_M_K, number<VectorSizeA>{}, number<1>{});

        // transformed tensor to flatten M and K dimensions  [M_total = M0 * M1 * M2 * ... , K_total
        // = K0 * K1 * K2 * ...]
        const auto A_grid_desc_Mflat_Kflat = ck_tile::transform_tensor_descriptor(
            A_grid_desc_Ms_Ks,
            make_tuple(make_merge_transform(dims_M), make_merge_transform(dims_K)),
            make_tuple(A_dims_M_ids, A_dims_K_ids),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return A_grid_desc_Mflat_Kflat;
    }

    /// @brief Creates a tensor descriptor for input tensor B with batch dimensions removed.
    /// @param B_dims Dimension vector for tensor B: [G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
    /// @param B_strides Stride vector for tensor B: [G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
    /// @return Flattened tensor descriptor: [N_total, K_total] for GEMM computation
    /// @details Removes batch dimensions and flattens N and K dimensions for efficient GEMM
    /// execution
    CK_TILE_HOST static constexpr auto
    Make_B_GridDescriptor_N_K(const std::vector<ck_tile::index_t>& B_dims    = {},
                              const std::vector<ck_tile::index_t>& B_strides = {})
    {
        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, number<end - start>{});
        };

        // Remove G Dimensions
        const auto B_dims_N_K =
            to_tuple(B_dims, number<NumDimG>{}, number<NumDimG + NumDimN + NumDimK>{});
        const auto B_strides_N_K =
            to_tuple(B_strides, number<NumDimG>{}, number<NumDimG + NumDimN + NumDimK>{});

        // dimension Ids for N and K
        constexpr auto B_dims_N_ids = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};
        constexpr auto B_dims_K_ids =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // Dimensions for N [N0, N1, ...] and K [K0, K1, ...]
        const auto dims_N = get_container_subset(B_dims_N_K, B_dims_N_ids);
        const auto dims_K = get_container_subset(B_dims_N_K, B_dims_K_ids);

        // naive tensor B[N0, N1, N2, ..., K0, K1, K2...] Descriptor with vector size
        const auto B_grid_desc_Ns_Ks = ck_tile::make_naive_tensor_descriptor(
            B_dims_N_K, B_strides_N_K, number<VectorSizeB>{}, number<1>{});

        // transformed tensor to flatten N and K dimensions  [N_total = N0 * N1 * N2 * ... , K_total
        // = K0 * K1 * K2 * ...]
        const auto B_grid_desc_Nflat_Kflat = ck_tile::transform_tensor_descriptor(
            B_grid_desc_Ns_Ks,
            make_tuple(make_merge_transform(dims_N), make_merge_transform(dims_K)),
            make_tuple(B_dims_N_ids, B_dims_K_ids),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return B_grid_desc_Nflat_Kflat;
    }

    /// @brief Creates a tensor descriptor for output tensor E with batch dimensions removed.
    /// @param E_dims Dimension vector for tensor E: [G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
    /// @param E_strides Stride vector for tensor E: [G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
    /// @return Flattened tensor descriptor: [M_total, N_total] for GEMM computation
    /// @details Removes batch dimensions and flattens M and N dimensions for efficient GEMM
    /// execution
    CK_TILE_HOST static constexpr auto
    Make_E_GridDescriptor_M_N(const std::vector<ck_tile::index_t>& E_dims    = {},
                              const std::vector<ck_tile::index_t>& E_strides = {})
    {
        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, number<end - start>{});
        };

        // Remove G dimensions
        const auto E_dims_M_N =
            to_tuple(E_dims, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimN>{});
        const auto E_strides_M_N =
            to_tuple(E_strides, number<NumDimG>{}, number<NumDimG + NumDimM + NumDimN>{});

        // dimension Ids for M and N
        constexpr auto E_dims_M_ids = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};
        constexpr auto E_dims_N_ids =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // Dimensions for M and N
        const auto dims_M = get_container_subset(E_dims_M_N, E_dims_M_ids);
        const auto dims_N = get_container_subset(E_dims_M_N, E_dims_N_ids);

        // naive tensor E[M0, M1, M2, ..., N0, N1, N2...] Descriptor with vector size
        const auto E_grid_desc_Ms_Ns = ck_tile::make_naive_tensor_descriptor(
            E_dims_M_N, E_strides_M_N, number<VectorSizeE>{}, number<1>{});

        // transformed tensor to flatten M and N dimensions   [M_total = M0 * M1 * M2 * ... ,
        // N_total = N0 * N1 * N2 * ...]
        const auto E_grid_desc_Mflat_Nflat = ck_tile::transform_tensor_descriptor(
            E_grid_desc_Ms_Ns,
            make_tuple(make_merge_transform(dims_M), make_merge_transform(dims_N)),
            make_tuple(E_dims_M_ids, E_dims_N_ids),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return E_grid_desc_Mflat_Nflat;
    }
};

} // namespace ck_tile
