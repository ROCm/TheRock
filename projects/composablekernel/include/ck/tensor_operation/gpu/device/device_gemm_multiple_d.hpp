// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#ifndef __HIPCC_RTC__
#include <array>
#endif

#include "ck/utility/array.hpp"
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
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef __HIPCC_RTC__
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
#endif
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
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleDSplitK : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef __HIPCC_RTC__
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        ck::index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
#endif
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
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleDSplitKBPreShuffle : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef CK_CODE_GEN_RTC
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        ck::index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual int GetPreShuffleParameters() = 0;
#endif
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename DsDataType,
          typename EDataType,
          index_t ScaleBlockSize,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceMoEGemmMXBPreShuffle : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef CK_CODE_GEN_RTC
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_a_scale,
                        const void* p_b,
                        const void* p_b_scale,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideAScale,
                        ck::index_t StrideB,
                        ck::index_t StrideBScale,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        ck::index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual int GetPreShuffleParameters() = 0;
#endif
};

/// @brief Wrapper for backward compatibility that allows to use instances of
///        DeviceGemmMultipleDSplitK in contexts where DeviceGemmMultipleD is expected.
///
/// @note  The main area where it can be used is DeviceOperationInstanceFactory::GetInstances().
///        The only difference between API of DeviceGemmMultipleD and DeviceGemmMultipleDSplitK is
///        that DeviceGemmMultipleDSplitK::MakeArgumentPointer requires an additional parameter
///        KBatch which is explicitly passed as 1 by this wrapper.
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmMultipleDSplitKWrapper : public DeviceGemmMultipleD<ALayout,
                                                                     BLayout,
                                                                     DsLayout,
                                                                     ELayout,
                                                                     ADataType,
                                                                     BDataType,
                                                                     DsDataType,
                                                                     EDataType,
                                                                     AElementwiseOperation,
                                                                     BElementwiseOperation,
                                                                     CDEElementwiseOperation>
{
    using DeviceOp = DeviceGemmMultipleDSplitK<ALayout,
                                               BLayout,
                                               DsLayout,
                                               ELayout,
                                               ADataType,
                                               BDataType,
                                               DsDataType,
                                               EDataType,
                                               AElementwiseOperation,
                                               BElementwiseOperation,
                                               CDEElementwiseOperation>;

    static constexpr index_t NumDTensor = DsDataType::Size();

#ifndef __HIPCC_RTC__

    explicit DeviceGemmMultipleDSplitKWrapper(std::unique_ptr<DeviceOp> p_op)
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
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
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
