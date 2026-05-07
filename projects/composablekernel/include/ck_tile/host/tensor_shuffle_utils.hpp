// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <stdexcept>

namespace ck_tile {
template <typename T>
auto shuffle_aq(const ck_tile::HostTensor<T>* t, int block_aq_k)
{
    if(t->get_lengths().size() != 2)
    {
        throw std::runtime_error("Host tensor is not rank 2 tensor.");
    }
    int m_   = t->get_lengths()[0];
    int aqk_ = t->get_lengths()[1];

    if(aqk_ % block_aq_k != 0)
    {
        throw std::runtime_error("shuffle_aq needs a aqk of multiple times of block_aq_k.");
    }
    ck_tile::HostTensor<T> t_view({m_, aqk_ / block_aq_k, block_aq_k});
    std::copy(t->begin(), t->end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {1, 0, 2});
}

template <typename T>
auto shuffle_bq(const ck_tile::HostTensor<T>* t, int block_bq_k)
{
    const auto& lengths = t->get_lengths();
    const size_t rank   = lengths.size();

    // Validate block_bq_k divisibility based on rank
    int bqk_dim = (rank == 5) ? lengths[4] : (rank == 2) ? lengths[0] : -1;

    if(bqk_dim < 0)
    {
        throw std::runtime_error("shuffle_bq expects either rank-2 or rank-5 tensor, got rank " +
                                 std::to_string(rank));
    }

    if(bqk_dim % block_bq_k != 0)
    {
        throw std::runtime_error("shuffle_bq needs bqk dimension to be a multiple of block_bq_k.");
    }

    // For TilePermuteN
    if(rank == 5)
    {
        // Handle 5D tensor: [n, nrepeat, nwarp, n_warp_tile, bqk]
        ck_tile::HostTensor<T> t_view({static_cast<int>(lengths[0]),
                                       static_cast<int>(lengths[1]),
                                       static_cast<int>(lengths[2]),
                                       static_cast<int>(lengths[3]),
                                       bqk_dim / block_bq_k,
                                       block_bq_k});
        std::copy(t->begin(), t->end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {4, 0, 1, 2, 3, 5});
    }
    else // rank == 2
    {
        // Handle 2D tensor: [bqk, n]
        int n_ = lengths[1];
        ck_tile::HostTensor<T> t_view({n_, bqk_dim / block_bq_k, block_bq_k});
        std::copy(t->begin(), t->end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {1, 0, 2});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t, const GemmConfig& gemmConfig)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        int kABK0PerLane           = gemmConfig.K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Warp_Tile,
                                       gemmConfig.N_Warp_Tile,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 4, 1, 3, 5});
    }
    else
    {
        int divisor = 1;
        if(ck_tile::is_gfx11_supported())
        {
            divisor = 1;
        }
        else
        {
            assert(is_wave32() == false);
            divisor = gemmConfig.N_Warp_Tile == 32 ? 2 : 4;
        }
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Warp_Tile,
                                       gemmConfig.N_Warp_Tile,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       divisor,
                                       gemmConfig.K_Warp_Tile / divisor});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
{
    return shuffle_b(t, GemmConfig{});
}

template <typename GemmConfig, typename T>
auto bq_permuteN(const ck_tile::HostTensor<T>& t, index_t group_n)
{
    assert(t.get_lengths().size() == 2);

    int n_                = t.get_lengths()[1];
    int bqk_              = t.get_lengths()[0];
    constexpr int NRepeat = GemmConfig::N_Tile / GemmConfig::N_Warp_Tile / GemmConfig::N_Warp;

    ck_tile::HostTensor<T> t_view({n_ / (GemmConfig::N_Tile / group_n),
                                   GemmConfig::N_Warp,
                                   GemmConfig::N_Warp_Tile / group_n,
                                   NRepeat,
                                   bqk_});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 3, 1, 2, 4});
}

template <typename GemmConfig, typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t, const GemmConfig& gemmConfig)
{
    assert(t.get_lengths().size() == 2);
    int n_      = t.get_lengths()[1];
    int k_      = t.get_lengths()[0];
    int NRepeat = gemmConfig.N_Tile / gemmConfig.N_Warp_Tile / gemmConfig.N_Warp;
    if(ck_tile::is_gfx12_supported())
    {
        constexpr int divisor      = 2;
        constexpr int kABK1PerLane = 8;
        int kABK0PerLane           = gemmConfig.K_Warp_Tile / divisor / kABK1PerLane;
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Tile,
                                       gemmConfig.N_Warp,
                                       gemmConfig.N_Warp_Tile,
                                       NRepeat,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       kABK0PerLane,
                                       divisor,
                                       kABK1PerLane});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 6, 5, 2, 7});
    }
    else
    {
        int divisor = 1;
        if(ck_tile::is_gfx11_supported())
        {
            divisor = 1;
        }
        else
        {
            assert(is_wave32() == false);
            divisor = gemmConfig.N_Warp_Tile == 32 ? 2 : 4;
        }
        ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Tile,
                                       gemmConfig.N_Warp,
                                       gemmConfig.N_Warp_Tile,
                                       NRepeat,
                                       k_ / gemmConfig.K_Warp_Tile,
                                       divisor,
                                       gemmConfig.K_Warp_Tile / divisor});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 5, 2, 6});
    }
}

template <typename GemmConfig, typename T>
auto shuffle_b_permuteN(const ck_tile::HostTensor<T>& t)
{
    return shuffle_b_permuteN(t, GemmConfig{});
}
} // namespace ck_tile
