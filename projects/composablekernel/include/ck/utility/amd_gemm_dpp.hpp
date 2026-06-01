// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/inner_product_dpp8.hpp"

namespace ck {

namespace dpp8 {

template <class ABDataType>
struct dpp_datatypes;

template <>
struct dpp_datatypes<half_t>
{
    // Dot product of `half2_t` and `half2_t` to get `float`. Reducing 2 elements from K in a
    // single instruction.
    using a_dtype                        = half_t;
    using b_dtype                        = half_t;
    using c_dtype                        = float;
    static constexpr index_t k_per_instr = 2;
};

template <index_t MPerThread,
          index_t NPerThread,
          index_t KPerThread,
          class BaseInputType,
          class AVecDataType,
          class BVecDataType,
          class CVecDataType,
          bool ShareA>
struct DppLanegroupGemm
{
    using datatypes_conf = dpp_datatypes<BaseInputType>;
    using ADataType      = typename datatypes_conf::a_dtype;
    using BDataType      = typename datatypes_conf::b_dtype;
    using CDataType      = typename datatypes_conf::c_dtype;

    __device__ void Run(const AVecDataType& a_vec, const BVecDataType& b_vec, CVecDataType& c_vec)
    {
        constexpr index_t num_c_elems_per_thread = ShareA ? MPerThread : NPerThread;

        const vector_type<ADataType, KPerThread> a_vector{a_vec};
        const vector_type<BDataType, KPerThread> b_vector{b_vec};

        static_for<0, num_c_elems_per_thread, 1>{}([&](auto c_idx) {
            float c = c_vec.template AsType<CDataType>()(c_idx);
            // Next `c_idx` implies that we need to pull data from the next lane.
            constexpr index_t source_lane = c_idx;
            static_for<0, KPerThread / datatypes_conf::k_per_instr, 1>{}([&](auto k_chunk) {
                const auto a_k_vec = a_vector.template AsType<AVecDataType>()[k_chunk];
                const auto b_k_vec = b_vector.template AsType<BVecDataType>()[k_chunk];
                ck::dpp8::
                    inner_product_dpp<AVecDataType, BVecDataType, CDataType, source_lane, ShareA>(
                        a_k_vec, b_k_vec, c);
            });
            c_vec.template AsType<CDataType>()(c_idx) = c;
        });
    }
};

} // namespace dpp8

} // namespace ck
