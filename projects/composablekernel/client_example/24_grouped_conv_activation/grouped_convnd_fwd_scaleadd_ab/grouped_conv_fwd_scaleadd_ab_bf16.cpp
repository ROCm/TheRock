// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"

using InDataType  = ck::Tuple<ck::bhalf_t, ck::bhalf_t>;
using WeiDataType = ck::Tuple<ck::bhalf_t, ck::bhalf_t>;
using OutDataType = ck::bhalf_t;

#include "grouped_conv_fwd_scaleadd_ab.inc"

int main() { return execute_conv_fwd_scaleadd_ab(); }
