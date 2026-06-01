// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A0[M, K], B0[K, N],
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleABD : public BaseOperator
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                        std::array<const void*, NumBTensor> p_bs,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        std::array<ck::index_t, NumATensor> StrideAs,
                        std::array<ck::index_t, NumBTensor> StrideBs,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

// GEMM:
//   input : A0[M, K], B0[K, N],
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleABDSplitK : public BaseOperator
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                        std::array<const void*, NumBTensor> p_bs,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        std::array<ck::index_t, NumATensor> StrideAs,
                        std::array<ck::index_t, NumBTensor> StrideBs,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        ck::index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

/// @brief Wrapper for backward compatibility that allows to use instances of
///        DeviceGemmMultipleABDSplitK in contexts where DeviceGemmMultipleABD is expected.
///
/// @note  The main area where it can be used is DeviceOperationInstanceFactory::GetInstances().
///        The only difference between API of DeviceGemmMultipleABD and DeviceGemmMultipleABDSplitK
///        is that DeviceGemmMultipleABDSplitK::MakeArgumentPointer requires an additional parameter
///        KBatch which is explicitly passed as 1 by this wrapper.
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleABDSplitKWrapper : public DeviceGemmMultipleABD<AsLayout,
                                                                         BsLayout,
                                                                         DsLayout,
                                                                         ELayout,
                                                                         AsDataType,
                                                                         BsDataType,
                                                                         DsDataType,
                                                                         EDataType,
                                                                         AElementwiseOperation,
                                                                         BElementwiseOperation,
                                                                         CDEElementwiseOperation>
{

    using DeviceOp = DeviceGemmMultipleABDSplitK<AsLayout,
                                                 BsLayout,
                                                 DsLayout,
                                                 ELayout,
                                                 AsDataType,
                                                 BsDataType,
                                                 DsDataType,
                                                 EDataType,
                                                 AElementwiseOperation,
                                                 BElementwiseOperation,
                                                 CDEElementwiseOperation>;

    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef __HIPCC_RTC__

    explicit DeviceGemmMultipleABDSplitKWrapper(std::unique_ptr<DeviceOp> p_op)
        : p_op_(std::move(p_op))
    {
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return p_op_->IsSupportedArgument(p_arg);
    }
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                        std::array<const void*, NumBTensor> p_bs,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        std::array<ck::index_t, NumATensor> StrideAs,
                        std::array<ck::index_t, NumBTensor> StrideBs,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return p_op_->MakeArgumentPointer(p_as,
                                          p_bs,
                                          p_ds,
                                          p_e,
                                          M,
                                          N,
                                          K,
                                          StrideAs,
                                          StrideBs,
                                          StrideDs,
                                          StrideE,
                                          1, // KBatch
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return p_op_->MakeInvokerPointer();
    }

    std::string GetTypeString() const override { return p_op_->GetTypeString(); }

    private:
    std::unique_ptr<DeviceOp> p_op_;

#endif // __HIPCC_RTC__
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
