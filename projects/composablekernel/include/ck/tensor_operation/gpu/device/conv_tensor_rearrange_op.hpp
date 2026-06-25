// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace conv_tensor_rearrange_op {

struct BaseConvTensorRearrangeOp
{
};

struct ImageToColumn : public BaseConvTensorRearrangeOp
{
    static constexpr const char* name = "Image to Column";
};

struct ColumnToImage : public BaseConvTensorRearrangeOp
{
    static constexpr const char* name = "Column to Image";
};

template <typename Op,
          typename std::enable_if<std::is_base_of<BaseConvTensorRearrangeOp, Op>::value,
                                  bool>::type = false>
std::ostream& operator<<(std::ostream& os, const BaseConvTensorRearrangeOp&)
{
    os << Op::name;
    return os;
}

} // namespace conv_tensor_rearrange_op
} // namespace ck
