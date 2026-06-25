// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

#include "common.hpp"

// kernel data types
using InKernelDataType       = I8;
using WeiKernelDataType      = I8;
using AccDataType            = I32;
using CShuffleDataType       = I8;
using BiasKernelDataType     = I8;
using ResidualKernelDataType = I8;
using OutKernelDataType      = I8;

// tensor data types
using InUserDataType  = I4;
using WeiUserDataType = I4;
using OutUserDataType = I4;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

#define BUILD_INT4_EXAMPLE
#include "run_grouped_conv_fwd_bias_relu_add_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_fwd_bias_relu_add_example(argc, argv); }
#endif
