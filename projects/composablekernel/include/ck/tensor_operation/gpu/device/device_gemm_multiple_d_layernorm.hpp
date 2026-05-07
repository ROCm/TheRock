// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   output : H[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
//   H = layernorm(E)
// Assume:
//   D0, D1, ... and E have the same layout
//   Calculate mean & variance along N dimension in layernorm(E)
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename HLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename HDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename HElementwiseOperation>
struct DeviceGemmMultipleDLayernorm : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_h,
                        index_t MRaw,
                        index_t NRaw,
                        index_t KRaw,
                        index_t StrideA,
                        index_t StrideB,
                        std::array<index_t, NumDTensor> StrideDs,
                        index_t StrideH,
                        double epsilon,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        HElementwiseOperation h_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
