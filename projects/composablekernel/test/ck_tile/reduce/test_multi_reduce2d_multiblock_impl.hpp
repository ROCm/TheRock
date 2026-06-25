// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include "test_multi_reduce2d_common.hpp"

template <typename Tuple>
class TestCkTileMultiReduceMultiblock : public ::testing::Test
{
    protected:
    using XDataType               = std::tuple_element_t<0, Tuple>;
    using ComputeDataType         = std::tuple_element_t<1, Tuple>;
    using YDataType               = std::tuple_element_t<2, Tuple>;
    using ReduceOpsType           = std::tuple_element_t<3, Tuple>;
    using ElementwiseOpsType      = std::tuple_element_t<4, Tuple>;
    using AccumulatorOpsType      = std::tuple_element_t<5, Tuple>;
    using InterBlockReduceOpsType = std::tuple_element_t<6, Tuple>;
    using BlockWarps_             = std::tuple_element_t<7, Tuple>;
    using BlockTile_              = std::tuple_element_t<8, Tuple>;
    using WarpTile_               = std::tuple_element_t<9, Tuple>;
    using ThreadTile_             = std::tuple_element_t<10, Tuple>;

    using TestReduce2dShape =
        ck_tile::Reduce2dShape<BlockWarps_, BlockTile_, WarpTile_, ThreadTile_>;

    template <std::size_t InputDim, typename KeptDimSeq, typename ReduceDimSeq>
    void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape,
                        const std::vector<ck_tile::index_t>& input_strides,
                        const std::vector<ck_tile::index_t>& output_shape,
                        const std::vector<ck_tile::index_t>& output_strides,
                        ck_tile::index_t kept_dim_len_prod,
                        ck_tile::index_t total_reduce_elements,
                        KeptDimSeq kept_dims,
                        ReduceDimSeq reduce_dims)
    {
        static_assert(
            ReduceOpsType::size() == ElementwiseOpsType::size() &&
                ReduceOpsType::size() == AccumulatorOpsType::size() &&
                ReduceOpsType::size() == InterBlockReduceOpsType::size(),
            "Error: All operations tuple size must match the number of reduction operations");

        const auto number_operations = ReduceOpsType::size();

        ck_tile::HostTensor<XDataType> h_x(input_shape, input_strides);

        auto h_ys = ck_tile::generate_tuple(
            [&output_shape, &output_strides](auto /*i*/) {
                return ck_tile::HostTensor<YDataType>(output_shape, output_strides);
            },
            ck_tile::number<number_operations>{});

        auto h_ys_ref = ck_tile::generate_tuple(
            [&output_shape, &output_strides](auto /*i*/) {
                return ck_tile::HostTensor<YDataType>(output_shape, output_strides);
            },
            ck_tile::number<number_operations>{});

        ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);

        ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
            h_ys.template at<i>().SetZero();
            h_ys_ref.template at<i>().SetZero();
        });

        auto output_number_elements = [&output_shape]() {
            ck_tile::index_t prod = 1;
            for(auto len : output_shape)
                prod *= len;
            return prod;
        }();

        auto output_buffer_size =
            number_operations * h_ys.get(ck_tile::number<0>{}).get_element_space_size_in_bytes();
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_y_mem(output_buffer_size);

        std::vector<YDataType> h(number_operations * output_number_elements);

        // Init the output data with identity values respective to each reduce op
        ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
            constexpr auto op       = ReduceOpsType{}.at(i);
            const auto identity_val = op.template GetIdentityValue<YDataType>();
            std::fill(h.begin() + i * output_number_elements,
                      h.begin() + (i + 1) * output_number_elements,
                      identity_val);
        });

        d_x_mem.ToDevice(h_x.data());
        d_y_mem.ToDevice(h.data());

        using Problem = ck_tile::Reduce2dProblem<XDataType,
                                                 ComputeDataType,
                                                 YDataType,
                                                 TestReduce2dShape,
                                                 ReduceOpsType,
                                                 KeptDimSeq,
                                                 ReduceDimSeq,
                                                 InputDim>;

        using Kernel = ck_tile::MultiReduceMultiblock<Problem>;

        // Launch configuration
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        auto elementwise_ops =
            make_elementwise_ops_tuple(total_reduce_elements, ElementwiseOpsType{});
        auto accumulator_ops =
            make_elementwise_ops_tuple(total_reduce_elements, AccumulatorOpsType{});

        auto [num_block_tile_iterations, block_group_size] =
            typename Kernel::TilePartitioner{total_reduce_elements}.GetBlockGroupParams();

        std::cout << "Block group size: " << block_group_size
                  << ", Num block tile iterations: " << num_block_tile_iterations
                  << ", Reduce total length: " << total_reduce_elements << std::endl;

        ck_tile::index_t kGridSize =
            ((kept_dim_len_prod + TestReduce2dShape::Block_M - 1) / TestReduce2dShape::Block_M) *
            block_group_size;

        // Generic helper to create tuple from vector based on compile-time size
        auto make_shape_tuple = []<std::size_t N>(const std::vector<ck_tile::index_t>& vec) {
            return [&vec]<std::size_t... I>(std::index_sequence<I...>) {
                return ck_tile::make_tuple(vec[I]...);
            }(std::make_index_sequence<N>{});
        };

        auto input_shape_tuple   = make_shape_tuple.template operator()<InputDim>(input_shape);
        auto input_strides_tuple = make_shape_tuple.template operator()<InputDim>(input_strides);

        if(!Kernel::IsSupportedArgument(
               total_reduce_elements,
               input_strides_tuple)) // output tensor's continuous dimension
        {
            throw std::runtime_error("Wrong! Arguments not supported!\n");
        }

        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
                                              static_cast<YDataType*>(d_y_mem.GetDeviceBuffer()),
                                              input_shape_tuple,
                                              input_strides_tuple,
                                              kept_dims,
                                              reduce_dims,
                                              output_number_elements,
                                              elementwise_ops,
                                              accumulator_ops,
                                              InterBlockReduceOpsType{}));

        // Reference computation
        ck_tile::reference_multiple_reduce_multiblock<XDataType, ComputeDataType, YDataType>(
            h_x,
            h_ys_ref,
            ReduceOpsType{},
            kept_dims,
            reduce_dims,
            elementwise_ops,
            accumulator_ops,
            InterBlockReduceOpsType{},
            block_group_size);

        // Calculate proper error thresholds based on data types and number of accumulations
        // const auto rtol = ck_tile::get_relative_threshold<XDataType, YDataType, ComputeDataType>(
        //     total_reduce_elements);
        // const auto atol = ck_tile::get_absolute_threshold<YDataType, YDataType, ComputeDataType>(
        //     5.0f, total_reduce_elements);

        // Unfortunately due to the non-sequenciality, down-casting on the output buffer
        // and further operations on this buffer, the error is compounding at a faster
        // rate than what the host reference can support. A large tolerance is then required
        const auto rtol = 1e-2;
        const auto atol = 1e-1;

        // Transfer data from device and check error for each operation
        std::vector<YDataType> h_y_tmp(output_number_elements * number_operations);
        d_y_mem.FromDevice(h_y_tmp.data());
        bool result = true;
        ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
            std::memcpy(h_ys.get(ck_tile::number<i>{}).data(),
                        h_y_tmp.data() + i * output_number_elements,
                        output_number_elements * sizeof(YDataType));
            std::cout << "Checking errors for operation: " << i << std::endl;
            result &= ck_tile::check_err(h_ys.get(ck_tile::number<i>{}),
                                         h_ys_ref.get(ck_tile::number<i>{}),
                                         "Error: Incorrect reduce results!",
                                         rtol,
                                         atol);
        });

        EXPECT_TRUE(result);
    }

    // Convenience functions for specific dimensional patterns
    void RunTest2D_KeepDim0_ReduceDim1(ck_tile::index_t dim0, ck_tile::index_t dim1)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0>{};
        constexpr auto reduce_dims = ck_tile::sequence<1>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {dim0, dim1};
        std::vector<ck_tile::index_t> input_strides = {dim1, 1};

        // Output shape and strides (keep dim0)
        std::vector<ck_tile::index_t> output_shape   = {dim0};
        std::vector<ck_tile::index_t> output_strides = {1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = dim0;
        ck_tile::index_t total_reduce_elements = dim1;

        RunGenericTest<2>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest3D_KeepDim0_ReduceDim12(ck_tile::index_t dim0,
                                        ck_tile::index_t dim1,
                                        ck_tile::index_t dim2)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0>{};
        constexpr auto reduce_dims = ck_tile::sequence<1, 2>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {dim0, dim1, dim2};
        std::vector<ck_tile::index_t> input_strides = {dim1 * dim2, dim2, 1};

        // Output shape and strides (keep dim0)
        std::vector<ck_tile::index_t> output_shape   = {dim0};
        std::vector<ck_tile::index_t> output_strides = {1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = dim0;        // product of kept dimensions
        ck_tile::index_t total_reduce_elements = dim1 * dim2; // product of reduced dimensions

        RunGenericTest<3>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest3D_KeepDim01_ReduceDim2(ck_tile::index_t dim0,
                                        ck_tile::index_t dim1,
                                        ck_tile::index_t dim2)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0, 1>{};
        constexpr auto reduce_dims = ck_tile::sequence<2>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {dim0, dim1, dim2};
        std::vector<ck_tile::index_t> input_strides = {dim1 * dim2, dim2, 1};

        // Output shape and strides (keep dim0, dim1)
        std::vector<ck_tile::index_t> output_shape   = {dim0, dim1};
        std::vector<ck_tile::index_t> output_strides = {dim1, 1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = dim0 * dim1; // product of kept dimensions
        ck_tile::index_t total_reduce_elements = dim2;        // product of reduced dimensions

        RunGenericTest<3>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest4D_KeepDim01_ReduceDim23(ck_tile::index_t N,
                                         ck_tile::index_t C,
                                         ck_tile::index_t H,
                                         ck_tile::index_t W)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0, 1>{};
        constexpr auto reduce_dims = ck_tile::sequence<2, 3>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {N, C, H, W};
        std::vector<ck_tile::index_t> input_strides = {C * H * W, H * W, W, 1};

        // Output shape and strides (keep dim0, dim1)
        std::vector<ck_tile::index_t> output_shape   = {N, C};
        std::vector<ck_tile::index_t> output_strides = {C, 1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = N * C; // product of kept dimensions
        ck_tile::index_t total_reduce_elements = H * W; // product of reduced dimensions

        RunGenericTest<4>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }

    void RunTest4D_KeepDim03_ReduceDim12(ck_tile::index_t N,
                                         ck_tile::index_t H,
                                         ck_tile::index_t W,
                                         ck_tile::index_t C)
    {
        constexpr auto kept_dims   = ck_tile::sequence<0, 3>{};
        constexpr auto reduce_dims = ck_tile::sequence<1, 2>{};

        // Input shape and strides
        std::vector<ck_tile::index_t> input_shape   = {N, H, W, C};
        std::vector<ck_tile::index_t> input_strides = {H * W * C, W * C, C, 1};

        // Output shape and strides (keep dim0, dim3)
        std::vector<ck_tile::index_t> output_shape   = {N, C};
        std::vector<ck_tile::index_t> output_strides = {C, 1};

        // Calculate products
        ck_tile::index_t kept_dim_len_prod     = N * C; // product of kept dimensions
        ck_tile::index_t total_reduce_elements = H * W; // product of reduced dimensions

        RunGenericTest<4>(input_shape,
                          input_strides,
                          output_shape,
                          output_strides,
                          kept_dim_len_prod,
                          total_reduce_elements,
                          kept_dims,
                          reduce_dims);
    }
};
