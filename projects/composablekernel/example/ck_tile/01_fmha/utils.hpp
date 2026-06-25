// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ck_tile/core/container/span.hpp"

enum class mode_enum
{
    batch = 0,
    group
};

std::ostream& operator<<(std::ostream& stream, mode_enum mode)
{
    return stream << (mode == mode_enum::batch ? "batch" : "group");
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

std::vector<int32_t> to_seqstarts(ck_tile::span<const int32_t> seqlens)
{
    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    assert(seqstarts.size() == seqlens.size() + 1);
    return seqstarts;
}

template <typename RandomEngine>
std::vector<int32_t> generate_seqlens(mode_enum mode,
                                      unsigned count,
                                      int32_t seqlen_avg,
                                      int32_t seqlen_min, // if not negative, clamp min
                                      int32_t seqlen_max, // if not negative, clamp max
                                      RandomEngine& random_engine)
{
    assert(0 < count);

    seqlen_min = (0 < seqlen_min ? seqlen_min : 1);
    seqlen_max = (0 < seqlen_max ? seqlen_max : std::numeric_limits<int32_t>::max());
    assert(seqlen_min <= seqlen_max);

    std::vector<int32_t> seqlens(count, std::clamp(seqlen_avg, seqlen_min, seqlen_max));

    if(mode == mode_enum::group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlen_avg * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens is in range [seqlen_min, seqlen_max]
            if(seqlens[to_decrease] == seqlen_min)
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            if(seqlens[to_increase] >= seqlen_max)
            {
                continue;
            }

            --seqlens[to_decrease];
            ++seqlens[to_increase];
        }
    }

    return seqlens;
}

// return random integer generated uniformly in range [low, high]
template <typename Int = int, typename RandomEngine>
auto randint(Int low,
             Int high,
             RandomEngine& random_engine) -> std::enable_if_t<std::is_integral_v<Int>, Int>
{
    std::uniform_int_distribution<Int> dist(low, high);
    return dist(random_engine);
}

// return random integers generated uniformly in range [low, high]
template <typename Int, typename ForwardIterator, typename RandomEngine>
auto randints(ForwardIterator first,
              ForwardIterator last,
              Int low,
              Int high,
              RandomEngine& random_engine) -> std::enable_if_t<std::is_integral_v<Int>>
{
    std::uniform_int_distribution<Int> dist(low, high);

    std::generate(first, last, [&] { return dist(random_engine); });
}

/*
 * generate missing values in *_val randomly when the number of values is smaller than batch
 * example (assume batch=3)
 *   q_val=1,2,3 k_val=4,5,6 -> OK
 *   q_val=1,2,3             -> OK, k same as q
 *   q_val=1,2               -> OK, q will rand remaining 1 element, k same as q
 *   q_val=1,2   k_val=4,5   -> OK, q/k will rand remaining 1 element
 *   q_val=1,2,3,4           -> OK, but ignore exceed one
 *
 *   q_val=1,2   k_val=4,5,6 -> not OK, k must have same splits with q
 *   q_val=1,2   k_val=4     -> not OK, k must have same splits with q
 */
template <typename RandomEngine>
std::tuple<std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>>
generate_missing_seqlens(mode_enum mode,
                         ck_tile::index_t batch,
                         const std::vector<ck_tile::index_t>& q_val,
                         const std::vector<ck_tile::index_t>& k_val,
                         const std::vector<ck_tile::index_t>& q_pad_val,
                         const std::vector<ck_tile::index_t>& k_pad_val,
                         ck_tile::index_t seqlen_k_min,
                         bool need_append_kvcache,
                         RandomEngine& random_engine)
{
    if(mode == mode_enum::batch)
    {
        ck_tile::index_t q = q_val[0];
        ck_tile::index_t k = k_val[0];

        auto s_q = std::vector<ck_tile::index_t>(batch, q);
        auto s_k = [&] {
            const ck_tile::index_t seqlen_k_max = (k < 0 ? q : k);
            std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k_max);

            if(1 < batch && need_append_kvcache)
            {
                // to keep the original s_k value, we always use seqlen_k_max in first batch
                randints(std::next(seqlen_ks.begin()),
                         seqlen_ks.end(),
                         seqlen_k_min,
                         seqlen_k_max,
                         random_engine);
                return seqlen_ks;
            }

            return seqlen_ks;
        }();
        auto s_kpad = std::vector<ck_tile::index_t>(batch, -1); // TODO: batch not support k_padding
        auto s_qpad = std::vector<ck_tile::index_t>(batch, -1);
        // s_k should be greater than or equal to seqlen_k_min if provided
        if(s_k.back() < seqlen_k_min)
        {
            std::ostringstream msg;
            msg << __FILE__ << ":" << __LINE__ << ": seqlen_k (=" << s_k.back()
                << ") is less than minimum seqlen_k (=" << seqlen_k_min << ")";
            throw std::runtime_error(msg.str());
        }

        return std::make_tuple(s_q, s_k, s_qpad, s_kpad);
    }
    else
    {
        std::vector<ck_tile::index_t> s_q;
        std::vector<ck_tile::index_t> s_k;
        std::vector<ck_tile::index_t> s_kpad;
        std::vector<ck_tile::index_t> s_qpad;
        ck_tile::index_t idx = 0;
        for(; idx < std::min(static_cast<ck_tile::index_t>(q_val.size()), batch); ++idx)
        {
            ck_tile::index_t q = q_val[idx];
            ck_tile::index_t k =
                k_val[std::min(idx, static_cast<ck_tile::index_t>(k_val.size()) - 1)];
            ck_tile::index_t kp =
                k_pad_val.empty()
                    ? -1
                    : k_pad_val[std::min(idx, static_cast<ck_tile::index_t>(k_pad_val.size()) - 1)];

            ck_tile::index_t qp =
                q_pad_val.empty()
                    ? -1
                    : q_pad_val[std::min(idx, static_cast<ck_tile::index_t>(q_pad_val.size()) - 1)];

            s_q.push_back(q);
            s_k.push_back(k < 0 ? q : k);
            s_kpad.push_back(kp);
            s_qpad.push_back(qp);

            // s_k should be greater than or equal to seqlen_k_min
            if(s_k.back() < seqlen_k_min)
            {
                std::ostringstream msg;
                msg << __FILE__ << ":" << __LINE__ << ": seqlen_k (=" << s_k.back()
                    << ") is less than minimum seqlen_k (=" << seqlen_k_min << ")";
                throw std::runtime_error(msg.str());
            }
        }
        if(idx < batch)
        {
            auto rem_q =
                generate_seqlens(mode, batch - idx, s_q.back(), 1, s_q.back(), random_engine);
            auto rem_k = generate_seqlens(
                mode, batch - idx, s_k.back(), seqlen_k_min, s_kpad.back(), random_engine);

            s_q.insert(s_q.end(), rem_q.begin(), rem_q.end());
            s_k.insert(s_k.end(), rem_k.begin(), rem_k.end());
            s_kpad.insert(s_kpad.end(), batch - idx, s_kpad.back());
            s_qpad.insert(s_qpad.end(), batch - idx, s_qpad.back());
        }
        return std::make_tuple(s_q, s_k, s_qpad, s_kpad);
    }
}

template <typename RandomAccessIterator, typename Int, typename RandomEngine>
std::enable_if_t<std::is_integral_v<Int>> iota_shuffle(RandomAccessIterator first,
                                                       RandomAccessIterator last,
                                                       Int value,
                                                       RandomEngine& random_engine)
{
    std::iota(first, last, value);
    std::shuffle(first, last, random_engine);
}
