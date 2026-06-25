// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_fpAintB_b_scale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = F32;
using CShuffleDataType = F16;
using CDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto PermuteA    = false;
static constexpr bool PermuteB    = false;

static constexpr ck::index_t Scale_Block_N = 1;
static constexpr ck::index_t Scale_Block_K = 128;

static constexpr ck::index_t KPerBlock = 256;

// clang-format off
using DeviceBatchedGemmV2Instance = 
    ck::tensor_operation::device::DeviceBatchedGemm_Xdl_CShuffleV3_BScale<  
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, BScaleDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
        256, Scale_Block_N, Scale_Block_K,
        16, 64,
        KPerBlock, 8, 32,
        16,   16,
        1,    1,
        S<32, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 32, 32, 0,
        1, 1, S<1, 16, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, CDataType, CDataType, PermuteA, PermuteB>;
// clang-format on

using ReferenceBatchedGemmInstance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                      AccDataType,
                                                                                      CDataType,
                                                                                      AccDataType,
                                                                                      AElementOp,
                                                                                      BElementOp,
                                                                                      CElementOp>;
#include "run_batched_gemm_example_fp16int4_b_scale.inc"

int main(int argc, char* argv[]) { return !run_batched_gemm_fp16_int4_b_scale_example(argc, argv); }
