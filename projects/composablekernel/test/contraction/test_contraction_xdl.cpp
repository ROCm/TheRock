// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <memory>
#include <initializer_list>
#include <vector>
#include <tuple>
#include <gtest/gtest.h>

#include "profiler/profile_contraction_impl.hpp"
#include "profiler/profile_contraction_utils.hpp"

static ck::index_t instance_index = -1;
using F16                         = ck::half_t;
using BF16                        = ck::bhalf_t;
using F32                         = float;
using F64                         = double;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

template <ck::index_t NDims>
struct Dimensions
{
    constexpr static ck::index_t NumDimMNK = NDims;

    std::vector<ck::index_t> M;
    std::vector<ck::index_t> N;
    std::vector<ck::index_t> K;
};

template <typename Tuple>
class TestContraction : public ::testing::Test
{
    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using CDLayout        = std::tuple_element_t<2, Tuple>;
    using DataType        = std::tuple_element_t<3, Tuple>;
    using DTupleDataType  = std::tuple_element_t<4, Tuple>;
    using ComputeDataType = std::tuple_element_t<5, Tuple>;
    using CDElementOp     = std::tuple_element_t<6, Tuple>;

    std::vector<ck::index_t> init_methods = {1, 2};
    std::unique_ptr<CDElementOp> p_cd_element_op;

    template <ck::index_t NumDim>
    void Run(Dimensions<NumDim> dimension_params)
    {
        constexpr ck::index_t NumDimMNK = ck::remove_cvref_t<decltype(dimension_params)>::NumDimMNK;

        std::vector<ck::index_t> StridesA(2 * NumDim);
        std::vector<ck::index_t> StridesB(2 * NumDim);
        std::vector<ck::index_t> StridesC(2 * NumDim);
        std::vector<ck::index_t> StridesD(2 * NumDim);

        const auto& M = dimension_params.M;
        const auto& N = dimension_params.N;
        const auto& K = dimension_params.K;

        auto merge_dims = [](const std::vector<ck::index_t>& dims01,
                             const std::vector<ck::index_t>& dims23) {
            std::vector<ck::index_t> dims_szt(dims01.begin(), dims01.end());
            dims_szt.insert(dims_szt.end(), dims23.begin(), dims23.end());
            return dims_szt;
        };

        assign_default_strides(ALayout{}, StridesA, merge_dims(M, K));
        assign_default_strides(BLayout{}, StridesB, merge_dims(N, K));
        assign_default_strides(CDLayout{}, StridesC, merge_dims(M, N));
        assign_default_strides(CDLayout{}, StridesD, merge_dims(M, N));

        for(const ck::index_t init_method : init_methods)
        {
            bool pass =
                ck::profiler::profile_contraction_impl<NumDimMNK,
                                                       ALayout,
                                                       BLayout,
                                                       CDLayout,
                                                       DataType,
                                                       ComputeDataType,
                                                       DTupleDataType,
                                                       CDElementOp>(true /*do_verification*/,
                                                                    init_method,
                                                                    false /*do_logs*/,
                                                                    false /*time_kernel*/,
                                                                    *p_cd_element_op,
                                                                    dimension_params.M,
                                                                    dimension_params.N,
                                                                    dimension_params.K,
                                                                    StridesA,
                                                                    StridesB,
                                                                    StridesC,
                                                                    StridesD,
                                                                    instance_index);
            EXPECT_TRUE(pass);
        }
    }
};

template <typename Tuple>
class TestContractionScale : public TestContraction<Tuple>
{
};

template <typename Tuple>
class TestContractionBilinear : public TestContraction<Tuple>
{
};

#define ALL_LAYOUT_COMBINATIONS(dt, tuple_dt, compute_dt, op)    \
    std::tuple<Row, Row, Row, dt, tuple_dt, compute_dt, op>,     \
        std::tuple<Row, Col, Row, dt, tuple_dt, compute_dt, op>, \
        std::tuple<Col, Row, Row, dt, tuple_dt, compute_dt, op>, \
        std::tuple<Col, Col, Row, dt, tuple_dt, compute_dt, op>

using BilinearKernelTypes =
    ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<F32>, F32, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<F64>, F64, Bilinear)>;

using ScaleKernelTypes = ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<>, F32, Scale),
                                          ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<>, F64, Scale)>;

TYPED_TEST_SUITE(TestContractionBilinear, BilinearKernelTypes);
TYPED_TEST_SUITE(TestContractionScale, ScaleKernelTypes);

TYPED_TEST(TestContractionBilinear, bilinear)
{
    this->p_cd_element_op = std::make_unique<Bilinear>(1.f, 1.f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});

    this->p_cd_element_op = std::make_unique<Bilinear>(-0.5f, 0.5f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});
}

TYPED_TEST(TestContractionScale, scale)
{
    this->p_cd_element_op = std::make_unique<Scale>(1.f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});

    this->p_cd_element_op = std::make_unique<Scale>(0.5f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});
}

template <typename Tuple>
class TestContractionScaleMixedPrecision : public TestContraction<Tuple>
{
};

template <typename Tuple>
class TestContractionBilinearMixedPrecision : public TestContraction<Tuple>
{
};

using BilinearKernelTypesMixedPrecision =
    ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<F32>, F16, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<F32>, BF16, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<F64>, F32, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(F16, ck::Tuple<F16>, F32, Bilinear),
                     ALL_LAYOUT_COMBINATIONS(BF16, ck::Tuple<BF16>, F32, Bilinear)>;

using ScaleKernelTypesMixedPrecision =
    ::testing::Types<ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<>, F16, Scale),
                     ALL_LAYOUT_COMBINATIONS(F32, ck::Tuple<>, BF16, Scale),
                     ALL_LAYOUT_COMBINATIONS(F64, ck::Tuple<>, F32, Scale),
                     ALL_LAYOUT_COMBINATIONS(F16, ck::Tuple<>, F32, Scale),
                     ALL_LAYOUT_COMBINATIONS(BF16, ck::Tuple<>, F32, Scale)>;

TYPED_TEST_SUITE(TestContractionBilinearMixedPrecision, BilinearKernelTypesMixedPrecision);
TYPED_TEST_SUITE(TestContractionScaleMixedPrecision, ScaleKernelTypesMixedPrecision);

TYPED_TEST(TestContractionBilinearMixedPrecision, bilinear)
{
    this->p_cd_element_op = std::make_unique<Bilinear>(1.f, 1.f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});

    this->p_cd_element_op = std::make_unique<Bilinear>(-0.5f, 0.5f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});
}

TYPED_TEST(TestContractionScaleMixedPrecision, scale)
{
    this->p_cd_element_op = std::make_unique<Scale>(1.f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});

    this->p_cd_element_op = std::make_unique<Scale>(0.5f);
    this->template Run<6>({{2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3}, {2, 2, 2, 2, 2, 4}});
    this->template Run<6>({{1, 1, 1, 3, 2, 3}, {1, 1, 1, 3, 2, 3}, {1, 1, 1, 2, 2, 4}});
    this->template Run<2>({{16, 8}, {16, 8}, {16, 8}});
    this->template Run<2>({{8, 16}, {16, 8}, {8, 16}});

    // special cases
    this->template Run<2>({{1, 1}, {16, 8}, {8, 16}});
    this->template Run<2>({{8, 16}, {16, 8}, {1, 1}});
    this->template Run<2>({{8, 16}, {1, 1}, {8, 16}});
    this->template Run<2>({{1, 1}, {1, 1}, {1, 1}});
}
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 2)
    {
        instance_index = atoi(argv[1]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1: instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
