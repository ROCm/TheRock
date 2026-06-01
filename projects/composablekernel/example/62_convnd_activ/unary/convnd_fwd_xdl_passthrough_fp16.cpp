// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_activ_unary_common.hpp"

using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using DeviceGroupedConvNDActivInstance = DeviceGroupedConvNDFwdInstance<OutElementOp>;
#include "../run_convnd_activ_example.inc"

int main(int argc, char* argv[]) { return !run_convnd_example(argc, argv); }
