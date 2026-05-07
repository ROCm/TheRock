// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// y = UnaryOp0(UnaryOp1(...(x)))
template <typename... UnaryOpsSet>
struct UnaryCombinedOp
{
    static constexpr const char* name = "UnaryCombinedOp";

    __host__ __device__ UnaryCombinedOp() : unary_ops_() {}

    __host__ __device__ UnaryCombinedOp(UnaryOpsSet... unary_ops) : unary_ops_(unary_ops...) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // Execute first unary op to copy data to y
        unary_ops_.At(Number<0>{})(y, x);

        static_for<1, Tuple<UnaryOpsSet...>::Size(), 1>{}([&](auto i) { unary_ops_.At(i)(y, y); });
    };

    Tuple<UnaryOpsSet...> unary_ops_;
};

// y = BinaryOp(UnaryOp0(x0), UnaryOp1(x1))
template <typename BinaryOp, typename UnaryOp0, typename UnaryOp1>
struct BinaryWithUnaryCombinedOp
{
    static constexpr const char* name = "BinaryWithUnaryCombinedOp";

    __host__ __device__ BinaryWithUnaryCombinedOp() : binary_op_(), unary_op0_(), unary_op1_() {}

    __host__ __device__ BinaryWithUnaryCombinedOp(BinaryOp binary_op,
                                                  UnaryOp0 unary_op0,
                                                  UnaryOp1 unary_op1)
        : binary_op_(binary_op), unary_op0_(unary_op0), unary_op1_(unary_op1)
    {
    }

    template <typename Y, typename X0, typename X1>
    __host__ __device__ void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        Y unary_x0_tmp_result;
        Y unary_x1_tmp_result;
        unary_op0_(unary_x0_tmp_result, x0);
        unary_op1_(unary_x1_tmp_result, x1);
        binary_op_(y, unary_x0_tmp_result, unary_x1_tmp_result);
    };

    private:
    BinaryOp binary_op_;
    UnaryOp0 unary_op0_;
    UnaryOp1 unary_op1_;
};

// y = BinaryOp0(BinaryOp1(UnaryOp0(x0), UnaryOp1(x1)), UnaryOp2(x2))
template <typename BinaryOp0,
          typename BinaryOp1,
          typename UnaryOp0,
          typename UnaryOp1,
          typename UnaryOp2>
struct TrinaryWithUnaryCombinedOp
{
    static constexpr const char* name = "TrinaryWithUnaryCombinedOp";

    __host__ __device__ TrinaryWithUnaryCombinedOp()
        : binary_op0_(), binary_op1_(), unary_op0_(), unary_op1_(), unary_op2_()
    {
    }

    __host__ __device__ TrinaryWithUnaryCombinedOp(BinaryOp0 binary_op0,
                                                   BinaryOp0 binary_op1,
                                                   UnaryOp0 unary_op0,
                                                   UnaryOp1 unary_op1,
                                                   UnaryOp2 unary_op2)
        : binary_op0_(binary_op0),
          binary_op1_(binary_op1),
          unary_op0_(unary_op0),
          unary_op1_(unary_op1),
          unary_op2_(unary_op2)
    {
    }

    template <typename Y, typename X0, typename X1, typename X2>
    __host__ __device__ void operator()(Y& y, const X0& x0, const X1& x1, const X2& x2) const
    {

        Y unary_x0_tmp_result;
        Y unary_x1_tmp_result;
        Y unary_x2_tmp_result;
        unary_op0_(unary_x0_tmp_result, x0);
        unary_op1_(unary_x1_tmp_result, x1);
        unary_op2_(unary_x2_tmp_result, x2);
        binary_op0_(unary_x0_tmp_result, unary_x0_tmp_result, unary_x1_tmp_result);
        binary_op1_(y, unary_x0_tmp_result, unary_x2_tmp_result);
    };

    private:
    BinaryOp0 binary_op0_{};
    BinaryOp1 binary_op1_{};
    UnaryOp0 unary_op0_{};
    UnaryOp1 unary_op1_{};
    UnaryOp2 unary_op2_{};
};

using ScaleScalePass = UnaryCombinedOp<Scale, Scale, PassThrough>;
using ScaleScaleRelu = UnaryCombinedOp<Scale, Scale, Relu>;

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
