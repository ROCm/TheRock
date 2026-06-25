// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_activ_dynamic_unary_common.hpp"

#include "../run_convnd_activ_dynamic_example.inc"

int main(int argc, char* argv[])
{

    ck::tensor_operation::element_wise::TanH out_element_op;
    return !run_convnd_example(argc, argv, out_element_op);
}
