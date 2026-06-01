// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef CK_CODE_GEN_RTC
#include <array>
#endif

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/utility/is_detected.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

#ifdef CK_CODE_GEN_RTC
template <typename T>
using is_tuple = decltype(ck::declval<T&>().IsTuple());
#else
template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());
#endif

/**
 * \brief Grouped Convolution Forward
 *
 * \details
 * input : input image A[G, N, C, Hi, Wi], A1[G, N, C, Hi, Wi]...
 * input : weight B[G, K, C, Y, X], B1[G, K, C, Y, X]...
 * input : D0[G, N, K, Ho, Wo], D1[G, N, K, Ho, Wo], ...
 * output : output image E[G, N, K, Ho, Wo]
 *
 * C = a_op(A, A1...) * b_op(B, B1...)
 * E = cde_op(C, D0, D1, ...)
 *
 * \tparam NDimSpatial Number of spatial dimensions.
 * \tparam ALayout Input layout (also for a1, a2...).
 * \tparam BLayout Weight layout (also for b1, b2...).
 * \tparam DsLayout Ds layouts.
 * \tparam ELayout Output layout.
 * \tparam ADataType Input data type. Pass tuple if there is multiple A.
 * \tparam BDataType Weight data type. Pass tuple if there is multiple B.
 * \tparam DsDataType D data types.
 * \tparam EDataType Output data type.
 * \tparam AElementwiseOperation A elementwise operation.
 * \tparam BElementwiseOperation B elementwise operation.
 * \tparam CDEElementwiseOperation CDE elementwise operation.
 * \tparam AComputeType Compute data type for A tensor (default: ADataType, first if tuple passed).
 * \tparam BComputeType Compute data type for B tensor (default: AComputeType).
 */
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AComputeType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // AComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeType = AComputeType>
struct DeviceGroupedConvFwdMultipleABD : public BaseOperator
{
    static constexpr bool isMultiA = is_detected<is_tuple, ADataType>::value;
    static constexpr bool isMultiB = is_detected<is_tuple, BDataType>::value;

    static constexpr index_t NumATensor = GetNumABTensors<isMultiA, ADataType>();
    static constexpr index_t NumBTensor = GetNumABTensors<isMultiB, BDataType>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(NumDTensor == DsLayout::Size(), "wrong! Inconsistent NumDTensor");
#ifdef CK_CODE_GEN_RTC
    using APointers = ck::conditional_t<isMultiA, ck::Array<const void*, NumATensor>&, const void*>;
    using BPointers = ck::conditional_t<isMultiB, ck::Array<const void*, NumBTensor>&, const void*>;
#else
    // If DataType is tuple, user has to pass std::array with pointers.
    using APointers =
        ck::conditional_t<isMultiA, std::array<const void*, NumATensor>&, const void*>;
    using BPointers =
        ck::conditional_t<isMultiB, std::array<const void*, NumBTensor>&, const void*>;
#endif

#ifndef CK_CODE_GEN_RTC

    /**
     * \brief Make argument pointer for grouped conv fwd.
     *
     * \param p_a A pointer to the input (std::array<const void*, NumA> with
                  pointers for multiple A).
     * \param p_b A pointer to the weight (std::array<const void*, NumA> with
                  pointers for multiple B).
     * \param p_ds A pointers to the Ds.
     * \param p_e A pointers to the output.
     * \param a_g_n_c_wis_lengths Input lengths [G, N, C, Spatial...] (for 3d).
     * \param a_g_n_c_wis_strides Input strides [G, N, C, Spatial...] (for 3d).
     * \param b_g_k_c_xs_lengths Weight lengths [G, K, C, Spatial...] (for 3d).
     * \param b_g_k_c_xs_strides Weight strides [G, K, C, Spatial...] (for 3d).
     * \param ds_g_n_k_wos_lengths Ds lengths [G, N, K, Spatial...] (for 3d).
     * \param ds_g_n_k_wos_strides Ds strides [G, N, K, Spatial...] (for 3d).
     * \param e_g_n_k_wos_lengths Output lengths [G, N, K, Spatial...] (for 3d).
     * \param e_g_n_k_wos_strides Output strides [G, N, K, Spatial...] (for 3d).
     * \param conv_filter_strides Convolution filter strides.
     * \param conv_filter_dilations Convolution filter dilations.
     * \param input_left_pads Input left paddings.
     * \param input_right_pads Input right paddings.
     * \param a_element_op A elementwise operation object.
     * \param b_element_op B elementwise operation object.
     * \param cde_element_op CDE elementwise operation object.
     * \return Pointer to the argument.
     */
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        APointers p_a,
        BPointers p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) = 0;

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(APointers p_a,
                        BPointers p_b,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_lengths,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<long_index_t, NDimSpatial>& input_left_pads,
                        const std::array<long_index_t, NDimSpatial>& input_right_pads,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CDEElementwiseOperation& cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
#endif
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
