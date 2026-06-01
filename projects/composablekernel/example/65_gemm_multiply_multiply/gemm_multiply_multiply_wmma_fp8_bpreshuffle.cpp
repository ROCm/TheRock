// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

#include "common.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F8   = ck::f8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType           = F8;
using B0DataType           = F8;
static constexpr int KPack = 16;
using ComputeType          = F8;

using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using D1DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
using EDataType        = F16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = MultiplyMultiply;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultiD_Wmma_CShuffle_V3_BPreshuffle<
        Row, Col, DsLayout, ELayout,
        A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
        AElementOp, BElementOp, CDEElementOp, GemmSpec,
        256,
        32, 128, 256,
        16, 16,
        16, 16,
        2, 1,
        S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
        S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
        1, 1, S<1, 16, 1, 16>, S<8, 8, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        ComputeType>;
// clang-format on

#include "run_gemm_multiply_multiply_wp_example.inc"

int main(int argc, char* argv[])
{
    // disable on gfx11 (fp8 not supported)
    if(ck::is_gfx11_supported())
    {
        return 0;
    }

    return run_gemm_example(argc, argv);
}
