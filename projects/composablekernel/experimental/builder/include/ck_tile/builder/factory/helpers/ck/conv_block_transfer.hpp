// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/array.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {

// Block transfer parameters for A or B tensor.
struct BlockTransfer
{
    ck::Array<size_t, 3> thread_cluster_dims  = {0, 0, 0}; // k0, m, k1
    ck::Array<size_t, 3> thread_cluster_order = {0, 0, 0};
    ck::Array<size_t, 3> src_access_order     = {0, 0, 0};
    size_t src_vector_dim                     = 0;
    size_t src_scalar_per_vector              = 0;
    size_t lds_dst_scalar_per_vector          = 0;
    bool is_direct_load                       = false;
    bool lds_padding                          = false;
};

template <auto TRANSFER>
constexpr BlockTransfer SetFwdConvBlockTransfer()
{
    auto& block_xfer  = TRANSFER.block_transfer;
    auto& block_order = TRANSFER.block_transfer_access_order;
    auto& src_order   = TRANSFER.src_access_order;
    auto& lds_cfg     = TRANSFER.lds_transfer;

    return BlockTransfer{
        .thread_cluster_dims   = {block_xfer.k0, block_xfer.m_n, block_xfer.k1},
        .thread_cluster_order  = {block_order.order[0], block_order.order[1], block_order.order[2]},
        .src_access_order      = {src_order.order[0], src_order.order[1], src_order.order[2]},
        .src_vector_dim        = lds_cfg.src_vector_dim,
        .src_scalar_per_vector = lds_cfg.src_scalar_per_vector,
        .lds_dst_scalar_per_vector = lds_cfg.lds_dst_scalar_per_vector,
        .is_direct_load            = lds_cfg.is_direct_load,
        .lds_padding               = lds_cfg.lds_padding,
    };
}

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    size_t m_xdl_per_wave_per_shuffle        = 0;
    size_t n_xdl_per_wave_per_shuffle        = 0;
    ck::Array<size_t, 4> thread_cluster_dims = {0, 0, 0, 0};
    size_t scalar_per_vector                 = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    auto& thread_cluster_dims = ALGORITHM.transfer.c.thread_cluster_dims;
    auto& epilogue_config     = ALGORITHM.transfer.c.epilogue;
    return CBlockTransfer{
        .m_xdl_per_wave_per_shuffle = epilogue_config.m_xdl_per_wave_per_shuffle,
        .n_xdl_per_wave_per_shuffle = epilogue_config.n_per_wave_per_shuffle,
        .thread_cluster_dims =
            {
                thread_cluster_dims.m_block,
                thread_cluster_dims.m_wave_per_xdl,
                thread_cluster_dims.n_block,
                thread_cluster_dims.n_wave_per_xdl,
            },
        .scalar_per_vector = epilogue_config.scalar_per_vector,
    };
}

} // namespace ck_tile::builder::factory::internal
