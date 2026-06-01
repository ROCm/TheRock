// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Helper to apply elementwise operation with variable number of D tensors
template <typename EDataType, typename AccDataType, typename CDEElementWise>
struct ApplyCDEElementWise
{
    template <typename... DValues>
    CK_TILE_HOST_DEVICE static void apply(EDataType& result,
                                          AccDataType sum,
                                          const CDEElementWise& cde_elementwise,
                                          DValues... d_vals)
    {
        if constexpr(sizeof...(DValues) == 0)
        {
            result = static_cast<EDataType>(sum);
        }
        else
        {
            cde_elementwise(
                result, ck_tile::type_convert<float>(sum), ck_tile::type_convert<float>(d_vals)...);
        }
    }
};

// Helper to extract D values at a given offset using index sequence
template <typename DDataType,
          ck_tile::index_t NumDTensor,
          typename Indices = std::make_index_sequence<NumDTensor>>
struct ExtractDValues;

template <typename DDataType, ck_tile::index_t NumDTensor, std::size_t... Is>
struct ExtractDValues<DDataType, NumDTensor, std::index_sequence<Is...>>
{
    template <typename EDataType, typename AccDataType, typename CDEElementWise>
    CK_TILE_HOST static void
    apply_at_offsets(EDataType& result,
                     AccDataType sum,
                     const CDEElementWise& cde_elementwise,
                     const std::array<ck_tile::HostTensor<DDataType>, NumDTensor>& ds_tensors,
                     const std::array<std::size_t, NumDTensor>& d_offsets)
    {
        ApplyCDEElementWise<EDataType, AccDataType, CDEElementWise>::apply(
            result, sum, cde_elementwise, ds_tensors[Is].mData[d_offsets[Is]]...);
    }
};

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename CDEElementWise,
          ck_tile::index_t NumDTensor>

void compute_reference_batched_contraction(
    const ck_tile::HostTensor<ADataType>& a_full_dims,
    const ck_tile::HostTensor<BDataType>& b_full_dims,
    const std::array<ck_tile::HostTensor<DDataType>, NumDTensor>& ds_full_dims_host,
    ck_tile::HostTensor<EDataType>& e_full_dims_host_ref,
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    const CDEElementWise& cde_elementwise,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims)
{
    std::cout << "Calculating reference using stride-aware indexing with parallel processing..."
              << std::endl;

    // Extract stride information from tensor descriptors
    const auto a_strides = a_full_dims.get_strides();
    const auto b_strides = b_full_dims.get_strides();
    const auto e_strides = e_full_dims_host_ref.get_strides();

    // Extract D tensor strides
    std::array<std::vector<std::size_t>, NumDTensor> ds_strides;
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        ds_strides[d] = ds_full_dims_host[d].get_strides();
    }

    const ck_tile::index_t num_g_dims = G_dims.size();
    const ck_tile::index_t num_m_dims = M_dims.size();
    const ck_tile::index_t num_n_dims = N_dims.size();
    const ck_tile::index_t num_k_dims = K_dims.size();

    // Helper lambda to compute linear index from flat indices using strides
    auto compute_a_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t k_flat) -> std::size_t {
        std::size_t offset = 0;

        // Decode G dimensions
        ck_tile::index_t temp = g_flat;
        for(int i = num_g_dims - 1; i >= 0; --i)
        {
            offset += (temp % G_dims[i]) * a_strides[i];
            temp /= G_dims[i];
        }

        // Decode M dimensions
        temp = m_flat;
        for(int i = num_m_dims - 1; i >= 0; --i)
        {
            offset += (temp % M_dims[i]) * a_strides[num_g_dims + i];
            temp /= M_dims[i];
        }

        // Decode K dimensions
        temp = k_flat;
        for(int i = num_k_dims - 1; i >= 0; --i)
        {
            offset += (temp % K_dims[i]) * a_strides[num_g_dims + num_m_dims + i];
            temp /= K_dims[i];
        }

        return offset;
    };

    auto compute_b_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t n_flat,
                                ck_tile::index_t k_flat) -> std::size_t {
        std::size_t offset = 0;

        // Decode G dimensions
        ck_tile::index_t temp = g_flat;
        for(int i = num_g_dims - 1; i >= 0; --i)
        {
            offset += (temp % G_dims[i]) * b_strides[i];
            temp /= G_dims[i];
        }

        // Decode N dimensions
        temp = n_flat;
        for(int i = num_n_dims - 1; i >= 0; --i)
        {
            offset += (temp % N_dims[i]) * b_strides[num_g_dims + i];
            temp /= N_dims[i];
        }

        // Decode K dimensions
        temp = k_flat;
        for(int i = num_k_dims - 1; i >= 0; --i)
        {
            offset += (temp % K_dims[i]) * b_strides[num_g_dims + num_n_dims + i];
            temp /= K_dims[i];
        }

        return offset;
    };

    auto compute_e_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t n_flat) -> std::size_t {
        std::size_t offset = 0;

        // Decode G dimensions
        ck_tile::index_t temp = g_flat;
        for(int i = num_g_dims - 1; i >= 0; --i)
        {
            offset += (temp % G_dims[i]) * e_strides[i];
            temp /= G_dims[i];
        }

        // Decode M dimensions
        temp = m_flat;
        for(int i = num_m_dims - 1; i >= 0; --i)
        {
            offset += (temp % M_dims[i]) * e_strides[num_g_dims + i];
            temp /= M_dims[i];
        }

        // Decode N dimensions
        temp = n_flat;
        for(int i = num_n_dims - 1; i >= 0; --i)
        {
            offset += (temp % N_dims[i]) * e_strides[num_g_dims + num_m_dims + i];
            temp /= N_dims[i];
        }

        return offset;
    };

    // Helper to compute D tensor offset (D tensors have same shape as E: [G, M, N])
    auto compute_d_offset = [&](ck_tile::index_t g_flat,
                                ck_tile::index_t m_flat,
                                ck_tile::index_t n_flat,
                                ck_tile::index_t d_idx) -> std::size_t {
        std::size_t offset    = 0;
        const auto& d_strides = ds_strides[d_idx];

        // Decode G dimensions
        ck_tile::index_t temp = g_flat;
        for(int i = num_g_dims - 1; i >= 0; --i)
        {
            offset += (temp % G_dims[i]) * d_strides[i];
            temp /= G_dims[i];
        }

        // Decode M dimensions
        temp = m_flat;
        for(int i = num_m_dims - 1; i >= 0; --i)
        {
            offset += (temp % M_dims[i]) * d_strides[num_g_dims + i];
            temp /= M_dims[i];
        }

        // Decode N dimensions
        temp = n_flat;
        for(int i = num_n_dims - 1; i >= 0; --i)
        {
            offset += (temp % N_dims[i]) * d_strides[num_g_dims + num_m_dims + i];
            temp /= N_dims[i];
        }

        return offset;
    };

    // Parallel computation over G and M dimensions
    auto f_gm = [&](auto g_flat, auto m_flat) {
        for(ck_tile::index_t n_flat = 0; n_flat < N_total; ++n_flat)
        {
            AccDataType sum = 0;

            // Compute dot product over K dimension using stride-aware indexing
            for(ck_tile::index_t k_flat = 0; k_flat < K_total; ++k_flat)
            {
                const std::size_t a_offset = compute_a_offset(g_flat, m_flat, k_flat);
                const std::size_t b_offset = compute_b_offset(g_flat, n_flat, k_flat);

                auto a_val = a_full_dims.mData[a_offset];
                auto b_val = b_full_dims.mData[b_offset];
                sum += static_cast<AccDataType>(a_val) * static_cast<AccDataType>(b_val);
            }

            // Compute output offset using strides
            const std::size_t e_offset = compute_e_offset(g_flat, m_flat, n_flat);

            // Compute individual D tensor offsets using their respective strides
            std::array<std::size_t, NumDTensor> d_offsets;
            for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
            {
                d_offsets[d] = compute_d_offset(g_flat, m_flat, n_flat, d);
            }

            // Apply elementwise operation with D tensors using compile-time dispatch
            EDataType result = static_cast<EDataType>(sum);
            ExtractDValues<DDataType, NumDTensor>::apply_at_offsets(
                result, sum, cde_elementwise, ds_full_dims_host, d_offsets);

            // Store result using stride-aware indexing
            e_full_dims_host_ref.mData[e_offset] = static_cast<EDataType>(result);
        }
    };

    // Execute parallel computation using hardware concurrency
    // Parallelize over G_total and M_total dimensions for optimal CPU utilization
    make_ParallelTensorFunctor(f_gm, G_total, M_total)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
