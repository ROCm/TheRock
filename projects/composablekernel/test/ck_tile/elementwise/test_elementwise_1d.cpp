// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath> // For std::abs
#include <tuple>
#include <type_traits> // For std::is_same_v, std::is_floating_point_v
#include <utility>     // For std::index_sequence, std::forward

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise/kernel/elementwise_kernel.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_shape.hpp"
#include "ck_tile/ops/elementwise/binary_elementwise_operation.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

// Traits to get number of inputs for an elementwise operation
template <typename Op>
struct elementwise_op_traits;

template <>
struct elementwise_op_traits<ck_tile::element_wise::Add>
{
    static constexpr int num_inputs = 2;
};
template <>
struct elementwise_op_traits<ck_tile::element_wise::Relu>
{
    static constexpr int num_inputs = 1;
};

using NegRelu =
    ck_tile::element_wise::Compose<ck_tile::element_wise::Relu, ck_tile::element_wise::Neg>;
template <>
struct elementwise_op_traits<NegRelu>
{
    static constexpr int num_inputs = 1;
};

template <std::size_t D, typename F>
auto make_uniform_array_with_factory(F&& factory)
{
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array<std::invoke_result_t<F, std::size_t>, D>{factory(Is)...};
    }(std::make_index_sequence<D>{});
}

template <typename Tuple>
class TestCkTileElementwise : public ::testing::Test
{
    protected:
    using XDataType         = std::tuple_element_t<0, Tuple>;
    using YDataType         = std::tuple_element_t<1, Tuple>;
    using ComputeDataType   = std::tuple_element_t<2, Tuple>;
    using ElementwiseOpType = std::tuple_element_t<3, Tuple>;
    using BlockWarps_       = std::tuple_element_t<4, Tuple>;
    using BlockTile_        = std::tuple_element_t<5, Tuple>;
    using WarpTile_         = std::tuple_element_t<6, Tuple>;
    using TestElementWiseShape =
        ck_tile::ElementWiseShape<BlockWarps_, BlockTile_, WarpTile_, XDataType>;
    static constexpr int NumInputs = elementwise_op_traits<ElementwiseOpType>::num_inputs;

    void RunTest(ck_tile::index_t total_m_elements)
    {
        // Dims and Strides (1D example)
        auto lens    = ck_tile::make_tuple(total_m_elements);
        auto strides = ck_tile::make_tuple(
            static_cast<ck_tile::index_t>(1)); // Strides for the single dimension

        // Host Tensors
        auto h_xs = make_uniform_array_with_factory<NumInputs>([&](std::size_t) {
            auto ret = ck_tile::HostTensor<XDataType>({total_m_elements});
            ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(ret);
            return ret;
        });
        ck_tile::HostTensor<YDataType> h_y({total_m_elements});
        h_y.SetZero();
        ck_tile::HostTensor<YDataType> h_y_ref({total_m_elements});
        h_y_ref.SetZero();

        // Device Buffers
        auto d_xs_mems_owner = make_uniform_array_with_factory<NumInputs>(
            [&](std::size_t i) { return ck_tile::DeviceMem(h_xs[i]); });
        for(int i = 0; i < NumInputs; ++i)
        {
            d_xs_mems_owner[i].ToDevice(h_xs[i].data());
        }

        ck_tile::DeviceMem d_y_mem(h_y);
        d_y_mem.SetZero();

        auto d_x_ptrs_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return ck_tile::make_tuple(
                static_cast<const XDataType*>(d_xs_mems_owner[Is].GetDeviceBuffer())...);
        }(std::make_index_sequence<NumInputs>{});

        YDataType* p_y_device = static_cast<YDataType*>(d_y_mem.GetDeviceBuffer());

        auto run_elementwise_kernel = [&](auto has_remainder) {
            constexpr bool kPad = decltype(has_remainder)::value;
            using Problem       = ck_tile::ElementWisePipelineProblem<XDataType,
                                                                      ComputeDataType,
                                                                      YDataType,
                                                                      TestElementWiseShape,
                                                                      ElementwiseOpType,
                                                                      kPad>;
            using Policy        = ck_tile::ElementWiseDefaultPolicy;
            ck_tile::ElementWiseKernel<Problem, Policy> ew_kernel;

            ck_tile::index_t grid_size = (total_m_elements + TestElementWiseShape::kBlockM - 1) /
                                         TestElementWiseShape::kBlockM;
            dim3 grid(grid_size, 1, 1);
            dim3 block                             = dim3(ew_kernel.BlockSize());
            constexpr ck_tile::index_t kBlockPerCu = 1;

            ck_tile::stream_config s{nullptr, false, 0}; // Default stream, no timing, no log

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<kBlockPerCu> // MinBlockPerCu
                                   (ew_kernel,
                                    grid,
                                    block,
                                    0, // actual shared memory
                                    lens,
                                    strides, // input strides
                                    strides, // output strides
                                    d_x_ptrs_tuple,
                                    p_y_device));
        };

        // Problem and Policy
        using BaseProblem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                                ComputeDataType,
                                                                YDataType,
                                                                TestElementWiseShape,
                                                                ElementwiseOpType>;
        if(total_m_elements % BaseProblem::BlockShape::kVectorM)
        {
            run_elementwise_kernel(std::true_type{});
        }
        else
        {
            run_elementwise_kernel(std::false_type{});
        }

        d_y_mem.FromDevice(h_y.data());

        // Reference computation on host
        ElementwiseOpType op_host;
        for(ck_tile::index_t i = 0; i < total_m_elements; ++i)
        {
            auto get_host_op_args = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return ck_tile::make_tuple(static_cast<ComputeDataType>(h_xs[Is](i))...);
            }(std::make_index_sequence<NumInputs>{});

            YDataType temp_y_val;
            ck_tile::apply(
                [&](auto&&... host_input_args) {
                    op_host(temp_y_val,
                            std::forward<decltype(host_input_args)>(host_input_args)...);
                },
                get_host_op_args);
            h_y_ref(i) = temp_y_val;
        }

        // Check results
        check_err(h_y, h_y_ref, "Error: Incorrect results!", 1e-5, 1e-5);
    }
};

// Shape parameters (can be shared or varied per test type)
using Shape1_BlockWarps = ck_tile::sequence<1>;   // 1D warp arrangement in M
using Shape1_BlockTile  = ck_tile::sequence<256>; // M-dimension of block tile
using Shape1_WarpTile   = ck_tile::sequence<64>;  // M-dimension of warp tile

// Test configurations
using TestConfig_F32_Add = std::tuple<float,
                                      float,
                                      float,
                                      ck_tile::element_wise::Add,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile>;

using TestConfig_F32_Relu = std::tuple<float,
                                       float,
                                       float,
                                       ck_tile::element_wise::Relu,
                                       Shape1_BlockWarps,
                                       Shape1_BlockTile,
                                       Shape1_WarpTile>;

using TestConfig_F16_Add = std::tuple<ck_tile::half_t,
                                      ck_tile::half_t,
                                      float, // Compute in float for half
                                      ck_tile::element_wise::Add,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile>;

using TestConfig_F32_Neg_Relu =
    std::tuple<float, float, float, NegRelu, Shape1_BlockWarps, Shape1_BlockTile, Shape1_WarpTile>;

using TestTypes = ::testing::
    Types<TestConfig_F32_Add, TestConfig_F32_Relu, TestConfig_F16_Add, TestConfig_F32_Neg_Relu>;

TYPED_TEST_SUITE(TestCkTileElementwise, TestTypes);

TYPED_TEST(TestCkTileElementwise, RunElementwise_1024) { this->RunTest(1024); }

TYPED_TEST(TestCkTileElementwise, RunElementwise_513)
{
    this->RunTest(513); // Test with an input size that's not a multiple of kVectorM
}

TYPED_TEST(TestCkTileElementwise, RunElementwise_516)
{
    this->RunTest(516); // Test with an input size that's not a multiple of blockM
}

TYPED_TEST(TestCkTileElementwise, RunElementwise_Small_32)
{
    this->RunTest(32); // Test with a very small size
}
