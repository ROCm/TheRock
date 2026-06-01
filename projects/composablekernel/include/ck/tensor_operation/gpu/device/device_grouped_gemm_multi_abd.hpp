// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmMultiABDDesc
{
    ck::index_t M_, N_, K_;

    std::vector<ck::index_t> stride_As_;
    std::vector<ck::index_t> stride_Bs_;
    std::vector<ck::index_t> stride_Ds_;

    ck::index_t stride_C_;
};

/*
 * \brief Grouped Gemm Multi ABD
 *
 * C = a_op(A, A1...) * b_op(B, B1...)
 * E = cde_op(C, D0, D1, ...)
 *
 * \tparam AsLayout A layouts (tuple).
 * \tparam BsLayout B layouts (tuple).
 * \tparam DsLayout Ds layouts (tuple).
 * \tparam ELayout Output layout.
 * \tparam AsDataType A data types (tuple).
 * \tparam BsDataType B data types (tuple).
 * \tparam DsDataType D data types (tuple).
 * \tparam EDataType Output data type.
 * \tparam AElementwiseOperation A elementwise operation.
 * \tparam BElementwiseOperation B elementwise operation.
 * \tparam CDEElementwiseOperation C elementwise operation.
 */
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
struct DeviceGroupedGemmMultiABD : public BaseOperator
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(AsLayout::Size() == AsDataType::Size(), "wrong! inconsistent NumATensor");
    static_assert(BsLayout::Size() == BsDataType::Size(), "wrong! inconsistent NumBTensor");
    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsistent NumDTensor");

    /*
     * \brief Make argument pointer for grouped gemm multi abd.
     *
     * \param p_as A pointers to the A.
     * \param p_bs A pointers to the B.
     * \param p_ds A pointers to the Ds.
     * \param p_e A pointers to the E.
     * \param gemm_desc Gemm descriptors for each group.
     * \param a_element_op A elementwise operation object.
     * \param b_element_op B elementwise operation object.
     * \param cde_element_op CDE elementwise operation object.
     * \return Pointer to the argument.
     */
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<std::array<const void*, NumATensor>>& p_as,
                        std::vector<std::array<const void*, NumBTensor>>& p_bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_ds,
                        std::vector<void*>& p_e,
                        std::vector<GemmMultiABDDesc>& gemm_desc,
                        AElementwiseOperation a_element_op   = AElementwiseOperation{},
                        BElementwiseOperation b_element_op   = BElementwiseOperation{},
                        CDEElementwiseOperation c_element_op = CDEElementwiseOperation{}) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual void SetElementwiseOps(BaseArgument* p_arg,
                                   AElementwiseOperation a_element_op,
                                   BElementwiseOperation b_element_op,
                                   CDEElementwiseOperation cde_element_op) const = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
