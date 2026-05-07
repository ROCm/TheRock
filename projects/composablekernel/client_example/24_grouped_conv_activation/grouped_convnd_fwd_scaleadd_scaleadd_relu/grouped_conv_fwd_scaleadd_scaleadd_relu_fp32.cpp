// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;
// Use std tuple instead of ck tuple to avoid clang
// implicit instantiation of undefined template error.
using DDataTypes = std::tuple<float, float>;

#include "grouped_conv_fwd_scaleadd_scaleadd_relu.inc"

int main() { return execute_conv_fwd_scaleadd_scaleadd_relu(); }
