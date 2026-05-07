// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K], B[K, N],
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_ABScale : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual void SetKBatch(BaseArgument* arg, int KBatch) const = 0;
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_BlockScale_BPreshuffle : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual int GetPreShuffleParameters() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        index_t KBatch) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual int GetPreShuffleParameters() = 0;
};

/// @brief Wrapper for backward compatibility that allows to use instances of
///        DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK in contexts where
//         DeviceGemmMultipleD_BlockScale_BPreshuffle is expected.
///
/// @note  The main area where it can be used is DeviceOperationInstanceFactory::GetInstances().
///        The only difference between API of DeviceGemmMultipleD_BlockScale_BPreshuffle and
//         DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK is
///        that DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK::MakeArgumentPointer requires
//         an additional parameter KBatch which is explicitly passed as 1 by this wrapper.
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_BlockScale_BPreshuffleWrapper
    : public DeviceGemmMultipleD_BlockScale_BPreshuffle<ALayout,
                                                        BLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        ADataType,
                                                        AScaleType,
                                                        BDataType,
                                                        BScaleType,
                                                        DsDataType,
                                                        EDataType,
                                                        ScaleBlockM,
                                                        ScaleBlockN,
                                                        ScaleBlockK,
                                                        AElementwiseOperation,
                                                        BElementwiseOperation,
                                                        CDEElementwiseOperation>
{
    using DeviceOp = DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK<ALayout,
                                                                      BLayout,
                                                                      DsLayout,
                                                                      ELayout,
                                                                      ADataType,
                                                                      AScaleType,
                                                                      BDataType,
                                                                      BScaleType,
                                                                      DsDataType,
                                                                      EDataType,
                                                                      ScaleBlockM,
                                                                      ScaleBlockN,
                                                                      ScaleBlockK,
                                                                      AElementwiseOperation,
                                                                      BElementwiseOperation,
                                                                      CDEElementwiseOperation>;

    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef __HIPCC_RTC__

    explicit DeviceGemmMultipleD_BlockScale_BPreshuffleWrapper(std::unique_ptr<DeviceOp> p_op)
        : p_op_(std::move(p_op))
    {
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return p_op_->IsSupportedArgument(p_arg);
    }
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return p_op_->MakeArgumentPointer(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          p_a_scale,
                                          p_b_scale,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op,
                                          1); // KBatch
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return p_op_->MakeInvokerPointer();
    }

    int GetPreShuffleParameters() override { return p_op_->GetPreShuffleParameters(); }

    std::string GetTypeString() const override { return p_op_->GetTypeString(); }

    private:
    std::unique_ptr<DeviceOp> p_op_;

#endif // __HIPCC_RTC__
};

// GEMM:
//   input : A[M, K], B[K, N],
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_ABScaleSplitK : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        index_t KBatch) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual void SetKBatch(BaseArgument* arg, int KBatch) const = 0;
};

/// @brief Wrapper for backward compatibility that allows to use instances of
///        DeviceGemmMultipleD_ABScaleSplitK in contexts where DeviceGemmMultipleD_ABScale is
///        expected.
///
/// @note  The main area where it can be used is DeviceOperationInstanceFactory::GetInstances().
///        The only difference between API of DeviceGemmMultipleD_ABScale and
///        DeviceGemmMultipleD_ABScaleSplitK is that
///        DeviceGemmMultipleD_ABScaleSplitK::MakeArgumentPointer requires a additional parameter
///        KBatch which is explicitly passed as 1 by this wrapper.
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleType,
          typename BDataType,
          typename BScaleType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD_ABScaleSplitKWrapper
    : public DeviceGemmMultipleD_ABScale<ALayout,
                                         BLayout,
                                         DsLayout,
                                         ELayout,
                                         ADataType,
                                         AScaleType,
                                         BDataType,
                                         BScaleType,
                                         DsDataType,
                                         EDataType,
                                         ScaleBlockM,
                                         ScaleBlockN,
                                         ScaleBlockK,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CDEElementwiseOperation>
{

    using DeviceOp = DeviceGemmMultipleD_ABScaleSplitK<ALayout,
                                                       BLayout,
                                                       DsLayout,
                                                       ELayout,
                                                       ADataType,
                                                       AScaleType,
                                                       BDataType,
                                                       BScaleType,
                                                       DsDataType,
                                                       EDataType,
                                                       ScaleBlockM,
                                                       ScaleBlockN,
                                                       ScaleBlockK,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CDEElementwiseOperation>;

    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef __HIPCC_RTC__

    explicit DeviceGemmMultipleD_ABScaleSplitKWrapper(std::unique_ptr<DeviceOp> p_op)
        : p_op_(std::move(p_op))
    {
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return p_op_->IsSupportedArgument(p_arg);
    }
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const ck::index_t M,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t StrideA,
                        const ck::index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        const ck::index_t StrideE,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return p_op_->MakeArgumentPointer(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          p_a_scale,
                                          p_b_scale,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op,
                                          1); // KBatch
    }

    void SetKBatch(BaseArgument* arg, int KBatch) const override { p_op_->SetKBatch(arg, KBatch); }

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
