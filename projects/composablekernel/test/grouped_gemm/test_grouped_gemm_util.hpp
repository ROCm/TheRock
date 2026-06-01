// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "profiler/profile_grouped_gemm_impl.hpp"

extern ck::index_t param_mask;
extern ck::index_t instance_index;

namespace ck {
namespace test {

template <typename Range>
std::string serialize_range(const Range& range)
{
    std::stringstream ss;
    for(auto& r : range)
    {
        ss << r << ", ";
    }
    std::string str = ss.str();
    return std::string(str.begin(), str.end() - 2);
}

// Helper primary template (will be specialized on the boolean)
template <std::size_t N,
          typename Tuple,
          typename Default,
          bool InRange = (N < std::tuple_size_v<std::remove_reference_t<Tuple>>)>
struct tuple_element_or_impl;

// Specialization for the in-range case: use std::tuple_element_t
template <std::size_t N, typename Tuple, typename Default>
struct tuple_element_or_impl<N, Tuple, Default, true>
{
    using type = std::tuple_element_t<N, std::remove_reference_t<Tuple>>;
};

// Specialization for the out-of-range case: use Default
template <std::size_t N, typename Tuple, typename Default>
struct tuple_element_or_impl<N, Tuple, Default, false>
{
    using type = Default;
};

// User-facing alias
template <std::size_t N, typename Tuple, typename Default>
using tuple_element_or_t = typename tuple_element_or_impl<N, Tuple, Default>::type;

template <typename Tuple, bool FailIfNoSupportedInstances = false>
class TestGroupedGemm : public testing::Test
{
    protected:
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ALayout      = std::tuple_element_t<0, Tuple>;
    using BLayout      = std::tuple_element_t<1, Tuple>;
    using ELayout      = std::tuple_element_t<2, Tuple>;
    using ADataType    = std::tuple_element_t<3, Tuple>;
    using BDataType    = std::tuple_element_t<4, Tuple>;
    using EDataType    = std::tuple_element_t<5, Tuple>;
    using AElementOp   = tuple_element_or_t<6, Tuple, PassThrough>;
    using BElementOp   = tuple_element_or_t<7, Tuple, PassThrough>;
    using CDEElementOp = tuple_element_or_t<8, Tuple, PassThrough>;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // integer value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    static constexpr int n_warmup_    = 0;
    static constexpr int n_iter_      = 1;

    bool fail_if_no_supported_instances_ = FailIfNoSupportedInstances;
    std::vector<int> k_batches_;

    bool IsSplitKSupported()
    {
        // gfx11 does not support split-K due to missing atomic add for fp16/bf16
        // Technically, we could still use split-K for fp32, but we currently don't have
        // instances for it so we disable it entirely
        constexpr bool require_16bit_atomic_add =
            std::is_same_v<EDataType, ck::half_t> || std::is_same_v<EDataType, ck::bhalf_t>;
        bool missing_atomic_add = require_16bit_atomic_add && ck::is_gfx11_supported();

        // CDE element operators are not supported in combination with split K
        constexpr bool has_cde_element_operator = !std::is_same_v<CDEElementOp, PassThrough>;

        return !missing_atomic_add && !has_cde_element_operator;
    }

    void SetUp() override
    {
        if(!IsSplitKSupported())
        {
            k_batches_ = {1};
        }
        else
        {
            k_batches_ = {1, 2, 3, 5, 8};
        }
    }

    private:
    template <typename Layout>
    void SetStrides(std::vector<int>& strides,
                    const std::vector<int>& rows,
                    const std::vector<int>& cols) const
    {
        if(std::is_same_v<Layout, Row>)
        {
            for(const auto c : cols)
            {
                strides.emplace_back(c);
            }
        }
        else if(std::is_same_v<Layout, Col>)
        {
            for(const auto r : rows)
            {
                strides.emplace_back(r);
            }
        }
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const std::vector<int>& StrideAs = {},
             const std::vector<int>& StrideBs = {},
             const std::vector<int>& StrideCs = {})
    {
        std::vector<int> stride_as = StrideAs;
        std::vector<int> stride_bs = StrideBs;
        std::vector<int> stride_cs = StrideCs;

        if(stride_as.empty())
        {
            SetStrides<ALayout>(stride_as, Ms, Ks);
        }
        if(stride_bs.empty())
        {
            SetStrides<BLayout>(stride_bs, Ks, Ns);
        }
        if(stride_cs.empty())
        {
            SetStrides<ELayout>(stride_cs, Ms, Ns);
        }
        std::vector<int> k_batches;
        for(size_t i = 0; i < k_batches_.size(); i++)
        {
            if(param_mask & (1 << i))
            {
                k_batches.push_back(k_batches_[i]);
            }
        }

        RunSingle(Ms, Ns, Ks, stride_as, stride_bs, stride_cs, k_batches);
    }

    void RunSingle(const std::vector<int>& Ms,
                   const std::vector<int>& Ns,
                   const std::vector<int>& Ks,
                   const std::vector<int>& StrideAs,
                   const std::vector<int>& StrideBs,
                   const std::vector<int>& StrideCs,
                   const std::vector<int>& kbatches)
    {
        bool pass =
            ck::profiler::profile_grouped_gemm_impl<ADataType,
                                                    BDataType,
                                                    EDataType,
                                                    float,
                                                    ALayout,
                                                    BLayout,
                                                    ELayout,
                                                    AElementOp,
                                                    BElementOp,
                                                    CDEElementOp>(verify_,
                                                                  init_method_,
                                                                  log_,
                                                                  bench_,
                                                                  Ms,
                                                                  Ns,
                                                                  Ks,
                                                                  StrideAs,
                                                                  StrideBs,
                                                                  StrideCs,
                                                                  kbatches,
                                                                  n_warmup_,
                                                                  n_iter_,
                                                                  instance_index,
                                                                  fail_if_no_supported_instances_);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
