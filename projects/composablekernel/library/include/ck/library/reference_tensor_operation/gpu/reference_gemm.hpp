// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputeTypeA,
          typename ComputeTypeB>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    naive_gemm_kernel(const ADataType* __restrict__ p_a_grid,
                      const BDataType* __restrict__ p_b_grid,
                      CDataType* __restrict__ p_c_grid,
                      index_t m,
                      index_t n,
                      index_t k,
                      const AElementwiseOperation a_element_op,
                      const BElementwiseOperation b_element_op,
                      const CDEElementwiseOperation c_element_op)
{
    using RowMajor = ck::tensor_layout::gemm::RowMajor;
    using ElementDataTypeA =
        ck::conditional_t<is_same_v<ComputeTypeA, ck::tf32_t>, float, ComputeTypeA>;
    using ElementDataTypeB =
        ck::conditional_t<is_same_v<ComputeTypeB, ck::tf32_t>, float, ComputeTypeB>;

    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(row_idx < m && col_idx < n)
    {

        AccDataType v_acc{0};
        ElementDataTypeA v_a{0};
        ElementDataTypeB v_b{0};
        CDataType v_c{0};

        for(int k_idx = 0; k_idx < k; ++k_idx)
        {
            // check input matrices layout
            int element_idx_a = 0;
            int element_idx_b = 0;
            if constexpr(std::is_same_v<ALayout, RowMajor>)
            {
                element_idx_a = row_idx * k + k_idx;
            }
            else
            {
                element_idx_a = row_idx + m * k_idx;
            }
            if constexpr(std::is_same_v<BLayout, RowMajor>)
            {
                element_idx_b = k_idx * n + col_idx;
            }
            else
            {
                element_idx_b = k_idx + k * col_idx;
            }
            // apply a_element_op
            a_element_op(v_a, p_a_grid[element_idx_a]);
            // apply b_element_op
            b_element_op(v_b, p_b_grid[element_idx_b]);
            // multiply and accumulate
            if constexpr(is_same_v<ComputeTypeA, ComputeTypeB> &&
                         is_same_v<ComputeTypeA, ck::tf32_t>)
            {
#if defined(__gfx942__)
                v_acc += ck::type_convert<AccDataType>(ck::type_convert<ck::tf32_t>(v_a)) *
                         ck::type_convert<AccDataType>(ck::type_convert<ck::tf32_t>(v_b));
#elif defined(__gfx950__)
                ck::bhalf_t v_a_bf16_big = ck::type_convert<ck::bhalf_t>(v_a);
                ck::bhalf_t v_a_bf16_small =
                    ck::type_convert<ck::bhalf_t>(v_a - type_convert<float>(v_a_bf16_big));
                ck::bhalf_t v_b_bf16_big = ck::type_convert<ck::bhalf_t>(v_b);
                ck::bhalf_t v_b_bf16_small =
                    ck::type_convert<ck::bhalf_t>(v_b - type_convert<float>(v_b_bf16_big));

                v_acc += ck::type_convert<AccDataType>(v_a_bf16_big) *
                             ck::type_convert<AccDataType>(v_b_bf16_small) +
                         ck::type_convert<AccDataType>(v_a_bf16_small) *
                             ck::type_convert<AccDataType>(v_b_bf16_big) +
                         ck::type_convert<AccDataType>(v_a_bf16_big) *
                             ck::type_convert<AccDataType>(v_b_bf16_big);
#else
                v_acc += type_convert<AccDataType>(v_a) * type_convert<AccDataType>(v_b);
#endif
            }
            else
            {
                v_acc += type_convert<AccDataType>(v_a) * type_convert<AccDataType>(v_b);
            }
        }
        // apply c_element_op
        c_element_op(v_c, v_acc);
        // check output matrix layout
        int element_idx_c = 0;
        if constexpr(std::is_same_v<CLayout, RowMajor>)
        {
            element_idx_c = row_idx * n + col_idx;
        }
        else
        {
            element_idx_c = row_idx + m * col_idx;
        }
        // prepare output
        p_c_grid[element_idx_c] = v_c;
    }
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputeTypeA = CDataType,
          typename ComputeTypeB = ComputeTypeA>
struct ReferenceGemm : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 void* p_c_grid,
                 index_t m,
                 index_t n,
                 index_t k,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_c_grid_{static_cast<CDataType*>(p_c_grid)},
              m_{m},
              n_{n},
              k_{k},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;

        index_t m_;
        index_t n_;
        index_t k_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceGemm::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            int block_size = 16;
            dim3 block_dim(block_size, block_size, 1);
            dim3 grid_dim(
                (arg.m_ + block_size - 1) / block_size, (arg.n_ + block_size - 1) / block_size, 1);

            auto launch_kernel = [&]() {
                const auto kernel = naive_gemm_kernel<ALayout,
                                                      BLayout,
                                                      CLayout,
                                                      ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      AccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation,
                                                      ComputeTypeA,
                                                      ComputeTypeB>;

                return ck::launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  grid_dim,
                                                  block_dim,
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.m_,
                                                  arg.n_,
                                                  arg.k_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_);
            };

            return launch_kernel();
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const void* p_a_grid,
                             const void* p_b_grid,
                             void* p_c_grid,
                             index_t m,
                             index_t n,
                             index_t k,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{
            p_a_grid, p_b_grid, p_c_grid, m, n, k, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "Device Reference Gemm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
