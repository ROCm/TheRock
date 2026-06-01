// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A0[M0, M1, ... K0, K1, ...], ...
//   input : B0[N0, N1, ... K0, K1, ...], ...
//   input : D0[M0, M1, ... N0, N1, ...], D1[M0, M1, ... N0, N1, ...], ...
//   output : E[M0, M1, ... N0, N1, ...]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceContractionMultipleABD : public BaseOperator
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                        std::array<const void*, NumBTensor> p_bs,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::array<std::vector<index_t>, NumATensor>& a_ms_ks_lengths,
                        const std::array<std::vector<index_t>, NumATensor>& a_ms_ks_strides,
                        const std::array<std::vector<index_t>, NumBTensor>& b_ns_ks_lengths,
                        const std::array<std::vector<index_t>, NumBTensor>& b_ns_ks_strides,
                        const std::array<std::vector<index_t>, NumDTensor>& d_ms_ns_lengths,
                        const std::array<std::vector<index_t>, NumDTensor>& d_ms_ns_strides,
                        const std::vector<index_t>& e_ms_ns_length,
                        const std::vector<index_t>& e_ms_ns_stride,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
