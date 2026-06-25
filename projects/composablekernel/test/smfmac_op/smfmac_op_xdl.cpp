// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "test/smfmac_op/smfmac_op_util.hpp"
#include "ck/host_utility/device_prop.hpp"

using BF16        = ck::bhalf_t;
using F16         = ck::half_t;
using F32         = float;
using Row         = ck::tensor_layout::gemm::RowMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename Tuple>
class TestSmfmac : public ::testing::Test
{
    protected:
    using Src1Type                           = std::tuple_element_t<0, Tuple>;
    static constexpr ck::index_t Src1VecSize = std::tuple_element_t<1, Tuple>{}.value;
    using Src2Type                           = std::tuple_element_t<2, Tuple>;
    static constexpr ck::index_t Src2VecSize = std::tuple_element_t<3, Tuple>{}.value;
    using DstType                            = std::tuple_element_t<4, Tuple>;
    static constexpr ck::index_t AccVecSize  = std::tuple_element_t<5, Tuple>{}.value;
    using GPUAccType                         = std::tuple_element_t<6, Tuple>;
    using CPUAccType                         = std::tuple_element_t<7, Tuple>;
    static constexpr ck::index_t M           = std::tuple_element_t<8, Tuple>{}.value;
    static constexpr ck::index_t N           = std::tuple_element_t<9, Tuple>{}.value;
    static constexpr ck::index_t K           = std::tuple_element_t<10, Tuple>{}.value;

    void Run()
    {
        bool pass = true;
        if(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950")
        {
            constexpr auto matmul_default = ck::smfmac_op_util::matmul<Src1Type,
                                                                       Src1VecSize,
                                                                       Src2Type,
                                                                       Src2VecSize,
                                                                       GPUAccType,
                                                                       AccVecSize,
                                                                       DstType,
                                                                       M,
                                                                       N,
                                                                       K>;

            constexpr auto smfmac_kernel_container = std::make_tuple(matmul_default);

            ck::static_for<0, std::tuple_size_v<decltype(smfmac_kernel_container)>, 1>{}(
                [&](auto i) {
                    pass &= ck::smfmac_op_util::TestSmfmac<
                        std::tuple_element_t<i.value, decltype(smfmac_kernel_container)>,
                        Src1Type,
                        Src2Type,
                        DstType,
                        GPUAccType,
                        CPUAccType,
                        decltype(Row{}),
                        decltype(Row{}),
                        decltype(Row{}),
                        PassThrough,
                        PassThrough,
                        PassThrough,
                        AccVecSize,
                        M,
                        N,
                        K>{}(std::get<ck::Number<i>{}>(smfmac_kernel_container));
                });
        }
        EXPECT_TRUE(pass);
    }
};

template <ck::index_t N>
using I = ck::Number<N>;

using KernelTypes =
    ::testing::Types<std::tuple<F16, I<4>, F16, I<8>, F32, I<4>, F32, F32, I<16>, I<16>, I<32>>,
                     std::tuple<BF16, I<4>, BF16, I<8>, F32, I<4>, F32, F32, I<16>, I<16>, I<32>>,
                     std::tuple<F16, I<4>, F16, I<8>, F32, I<16>, F32, F32, I<32>, I<32>, I<16>>,
                     std::tuple<BF16, I<4>, BF16, I<8>, F32, I<16>, F32, F32, I<32>, I<32>, I<16>>>;

TYPED_TEST_SUITE(TestSmfmac, KernelTypes);
TYPED_TEST(TestSmfmac, TestSmfmacFP16BF16) { this->Run(); }
