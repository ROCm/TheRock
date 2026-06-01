// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"

namespace ck {
namespace tensor_operation {

template <index_t NDimSpatial,
          device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          bool SplitN              = false,
          typename ADataType       = float,
          typename CDataType       = float,
          index_t NumGroupsToMerge = 1,
          typename IndexType       = index_t,
          bool CTranspose          = false>
struct TransformConvFwdToGemm
{
    private:
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    template <typename ConvDimsType>
    static long_index_t calculate_element_space_size_impl(const ConvDimsType& lengths,
                                                          const ConvDimsType& strides,
                                                          index_t i)
    {
        long_index_t acc = 1;
        for(; i < (NDimSpatial + 3); i++)
        {
            acc +=
                static_cast<long_index_t>(lengths[i] - I1) * static_cast<long_index_t>(strides[i]);
        }

        return acc;
    }

    template <typename ConvDimsType>
    static IndexType GetSplitedNSize(const ConvDimsType& a_g_n_c_wis_lengths,
                                     const ConvDimsType& a_g_n_c_wis_strides,
                                     const ConvDimsType& c_g_n_k_wos_lengths,
                                     const ConvDimsType& c_g_n_k_wos_strides)
    {
        const long_index_t a_element_space_size =
            calculate_element_space_size_impl(a_g_n_c_wis_lengths, a_g_n_c_wis_strides, I1);
        const long_index_t c_element_space_size =
            calculate_element_space_size_impl(c_g_n_k_wos_lengths, c_g_n_k_wos_strides, I1);
        const long_index_t element_space_size = math::max(a_element_space_size * sizeof(ADataType),
                                                          c_element_space_size * sizeof(CDataType));
        constexpr long_index_t TwoGB          = (long_index_t{1} << 31); // 2GB threshold

        const IndexType N = a_g_n_c_wis_lengths[I1];

        if(element_space_size > TwoGB)
        {
            // Minimum divisor of N to not exceed 2GB
            const auto divisor = math::integer_divide_ceil(element_space_size, TwoGB);

            if(divisor <= static_cast<double>(N))
            {
                // Find least divisor of N larger than element_space_size / TwoGB
                // Iterate up to sqrt(N). There are no divisors above this value.
                for(IndexType least_divisor = divisor; least_divisor * least_divisor <= N;
                    least_divisor++)
                {
                    if(N % least_divisor == 0)
                    {
                        return N / least_divisor;
                    }
                }
                // Not found, process one Convolution N per block
                return 1;
            }
            else
            {
                // Split Convolution's N dimension into N workgroups. However
                // this still might not result in sufficiently small tensor,
                // but at least later on we could divide the image as well.
                return 1;
            }
        }
        else
        {
            // Split N is not needed.
            return N;
        }
    }

    public:
    __host__ __device__ constexpr TransformConvFwdToGemm() {}

    template <typename TransformConvFwdToGemmBase>
    __host__ __device__
    TransformConvFwdToGemm(const TransformConvFwdToGemmBase& transform_conv_fwd_to_gemm_base)
        : N_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.N_)},
          Di_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Di_)},
          Hi_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Hi_)},
          Wi_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Wi_)},
          Do_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Do_)},
          Ho_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Ho_)},
          Wo_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Wo_)},
          Z_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Z_)},
          Y_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Y_)},
          X_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.X_)},
          K_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.K_)},
          C_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.C_)},
          DiStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.DiStride_)},
          HiStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.HiStride_)},
          WiStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.WiStride_)},
          DoStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.DoStride_)},
          HoStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.HoStride_)},
          WoStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.WoStride_)},
          XStride_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.XStride_)},
          CStrideTensorA_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.CStrideTensorA_)},
          CStrideTensorB_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.CStrideTensorB_)},
          KStrideTensorB_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.KStrideTensorB_)},
          KStrideTensorC_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.KStrideTensorC_)},
          NStrideTensorA_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.NStrideTensorA_)},
          NStrideTensorC_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.NStrideTensorC_)},
          GStrideTensorA_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.GStrideTensorA_)},
          GStrideTensorB_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.GStrideTensorB_)},
          GStrideTensorC_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.GStrideTensorC_)},
          ConvStrideD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvStrideD_)},
          ConvStrideH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvStrideH_)},
          ConvStrideW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvStrideW_)},
          ConvDilationD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvDilationD_)},
          ConvDilationH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvDilationH_)},
          ConvDilationW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvDilationW_)},
          InLeftPadD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InLeftPadD_)},
          InLeftPadH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InLeftPadH_)},
          InLeftPadW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InLeftPadW_)},
          InRightPadD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadD_)},
          InRightPadH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadH_)},
          InRightPadW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadW_)},
          ZYX_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ZYX_)}
    {
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                  = NDimSpatial,
              typename ck::enable_if<NDim == 1, bool>::type = false>
    __host__ __device__ TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                               const ConvDimsType& a_g_n_c_wis_strides,
                                               const ConvDimsType& b_g_k_c_xs_lengths,
                                               const ConvDimsType& b_g_k_c_xs_strides,
                                               const ConvDimsType& c_g_n_k_wos_lengths,
                                               const ConvDimsType& c_g_n_k_wos_strides,
                                               const ConvSpatialDimsType& conv_filter_strides,
                                               const ConvSpatialDimsType& conv_filter_dilations,
                                               const ConvSpatialDimsType& input_left_pads,
                                               const ConvSpatialDimsType& input_right_pads)
        : Di_{I1},
          Hi_{I1},
          Wi_{a_g_n_c_wis_lengths[I3]},
          Do_{I1},
          Ho_{I1},
          Wo_{c_g_n_k_wos_lengths[I3]},
          Z_{I1},
          Y_{I1},
          X_{b_g_k_c_xs_lengths[I3]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          DiStride_{I1},
          HiStride_{I1},
          WiStride_{a_g_n_c_wis_strides[I3]},
          DoStride_{I1},
          HoStride_{I1},
          WoStride_{c_g_n_k_wos_strides[I3]},
          XStride_{b_g_k_c_xs_strides[I3]},
          CStrideTensorA_{a_g_n_c_wis_strides[I2]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          KStrideTensorC_{c_g_n_k_wos_strides[I2]},
          NStrideTensorA_{a_g_n_c_wis_strides[I1]},
          NStrideTensorC_{c_g_n_k_wos_strides[I1]},
          GStrideTensorA_{a_g_n_c_wis_strides[I0]},
          GStrideTensorB_{b_g_k_c_xs_strides[I0]},
          GStrideTensorC_{c_g_n_k_wos_strides[I0]},
          ConvStrideD_{I1},
          ConvStrideH_{I1},
          ConvStrideW_{conv_filter_strides[I0]},
          ConvDilationD_{I1},
          ConvDilationH_{I1},
          ConvDilationW_{conv_filter_dilations[I0]},
          InLeftPadD_{I0},
          InLeftPadH_{I0},
          InLeftPadW_{input_left_pads[I0]},
          InRightPadD_{I0},
          InRightPadH_{I0},
          InRightPadW_{input_right_pads[I0]},
          ZYX_{X_}
    {
#ifdef CK_CODE_GEN_RTC
        static_assert(is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);
#else
        static_assert(is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);
#endif
        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                  = NDimSpatial,
              typename ck::enable_if<NDim == 2, bool>::type = false>
    __host__ __device__ TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                               const ConvDimsType& a_g_n_c_wis_strides,
                                               const ConvDimsType& b_g_k_c_xs_lengths,
                                               const ConvDimsType& b_g_k_c_xs_strides,
                                               const ConvDimsType& c_g_n_k_wos_lengths,
                                               const ConvDimsType& c_g_n_k_wos_strides,
                                               const ConvSpatialDimsType& conv_filter_strides,
                                               const ConvSpatialDimsType& conv_filter_dilations,
                                               const ConvSpatialDimsType& input_left_pads,
                                               const ConvSpatialDimsType& input_right_pads)
        : Di_{I1},
          Hi_{a_g_n_c_wis_lengths[I3]},
          Wi_{a_g_n_c_wis_lengths[I4]},
          Do_{I1},
          Ho_{c_g_n_k_wos_lengths[I3]},
          Wo_{c_g_n_k_wos_lengths[I4]},
          Z_{I1},
          Y_{b_g_k_c_xs_lengths[I3]},
          X_{b_g_k_c_xs_lengths[I4]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          DiStride_{I1},
          HiStride_{a_g_n_c_wis_strides[I3]},
          WiStride_{a_g_n_c_wis_strides[I4]},
          DoStride_{I1},
          HoStride_{c_g_n_k_wos_strides[I3]},
          WoStride_{c_g_n_k_wos_strides[I4]},
          XStride_{b_g_k_c_xs_strides[I4]},
          CStrideTensorA_{a_g_n_c_wis_strides[I2]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          KStrideTensorC_{c_g_n_k_wos_strides[I2]},
          NStrideTensorA_{a_g_n_c_wis_strides[I1]},
          NStrideTensorC_{c_g_n_k_wos_strides[I1]},
          GStrideTensorA_{a_g_n_c_wis_strides[I0]},
          GStrideTensorB_{b_g_k_c_xs_strides[I0]},
          GStrideTensorC_{c_g_n_k_wos_strides[I0]},
          ConvStrideD_{I1},
          ConvStrideH_{conv_filter_strides[I0]},
          ConvStrideW_{conv_filter_strides[I1]},
          ConvDilationD_{I1},
          ConvDilationH_{conv_filter_dilations[I0]},
          ConvDilationW_{conv_filter_dilations[I1]},
          InLeftPadD_{I0},
          InLeftPadH_{input_left_pads[I0]},
          InLeftPadW_{input_left_pads[I1]},
          InRightPadD_{I0},
          InRightPadH_{input_right_pads[I0]},
          InRightPadW_{input_right_pads[I1]},
          ZYX_{Y_ * X_}
    {
#ifdef CK_CODE_GEN_RTC
        static_assert(is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);
#else
        static_assert(is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);
#endif
        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                  = NDimSpatial,
              typename ck::enable_if<NDim == 3, bool>::type = false>
    __host__ __device__ TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                               const ConvDimsType& a_g_n_c_wis_strides,
                                               const ConvDimsType& b_g_k_c_xs_lengths,
                                               const ConvDimsType& b_g_k_c_xs_strides,
                                               const ConvDimsType& c_g_n_k_wos_lengths,
                                               const ConvDimsType& c_g_n_k_wos_strides,
                                               const ConvSpatialDimsType& conv_filter_strides,
                                               const ConvSpatialDimsType& conv_filter_dilations,
                                               const ConvSpatialDimsType& input_left_pads,
                                               const ConvSpatialDimsType& input_right_pads)
        : Di_{a_g_n_c_wis_lengths[I3]},
          Hi_{a_g_n_c_wis_lengths[I4]},
          Wi_{a_g_n_c_wis_lengths[I5]},
          Do_{c_g_n_k_wos_lengths[I3]},
          Ho_{c_g_n_k_wos_lengths[I4]},
          Wo_{c_g_n_k_wos_lengths[I5]},
          Z_{b_g_k_c_xs_lengths[I3]},
          Y_{b_g_k_c_xs_lengths[I4]},
          X_{b_g_k_c_xs_lengths[I5]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          DiStride_{a_g_n_c_wis_strides[I3]},
          HiStride_{a_g_n_c_wis_strides[I4]},
          WiStride_{a_g_n_c_wis_strides[I5]},
          DoStride_{c_g_n_k_wos_strides[I3]},
          HoStride_{c_g_n_k_wos_strides[I4]},
          WoStride_{c_g_n_k_wos_strides[I5]},
          XStride_{b_g_k_c_xs_strides[I5]},
          CStrideTensorA_{a_g_n_c_wis_strides[I2]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          KStrideTensorC_{c_g_n_k_wos_strides[I2]},
          NStrideTensorA_{a_g_n_c_wis_strides[I1]},
          NStrideTensorC_{c_g_n_k_wos_strides[I1]},
          GStrideTensorA_{a_g_n_c_wis_strides[I0]},
          GStrideTensorB_{b_g_k_c_xs_strides[I0]},
          GStrideTensorC_{c_g_n_k_wos_strides[I0]},
          ConvStrideD_{conv_filter_strides[I0]},
          ConvStrideH_{conv_filter_strides[I1]},
          ConvStrideW_{conv_filter_strides[I2]},
          ConvDilationD_{conv_filter_dilations[I0]},
          ConvDilationH_{conv_filter_dilations[I1]},
          ConvDilationW_{conv_filter_dilations[I2]},
          InLeftPadD_{input_left_pads[I0]},
          InLeftPadH_{input_left_pads[I1]},
          InLeftPadW_{input_left_pads[I2]},
          InRightPadD_{input_right_pads[I0]},
          InRightPadH_{input_right_pads[I1]},
          InRightPadW_{input_right_pads[I2]},
          ZYX_{Z_ * Y_ * X_}
    {
#ifdef CK_CODE_GEN_RTC
        static_assert(is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);
#else
        static_assert(is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);
#endif
        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
    }

    __host__ bool AreDescriptorsSmallerThan2GB() const
    {
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        const long_index_t in_desc_space_size =
            I1 + (N_ - I1) * NStrideTensorA_ + (Di_ - I1) * DiStride_ + (Hi_ - I1) * HiStride_ +
            (Wi_ - I1) * WiStride_ + (C_ - I1) * CStrideTensorA_;
        const long_index_t out_desc_space_size =
            I1 + (N_ - I1) * NStrideTensorC_ + (Do_ - I1) * DoStride_ + (Ho_ - I1) * HoStride_ +
            (Wo_ - I1) * WoStride_ + (K_ - I1) * KStrideTensorC_;

        bool is_a_descriptor_smaller_than_2GB = (in_desc_space_size * sizeof(ADataType)) <= TwoGB;
        bool is_c_descriptor_smaller_than_2GB = (out_desc_space_size * sizeof(CDataType)) <= TwoGB;

        return is_a_descriptor_smaller_than_2GB && is_c_descriptor_smaller_than_2GB;
    }

    template <typename DsPointer>
    __host__ auto SplitConvProblem(const ADataType* a_grid_ptr_base,
                                   DsPointer& ds_grid_ptr_base,
                                   CDataType* c_grid_ptr_base) const
    {
        // Create copies
        auto conv_to_gemm_transformer_left  = *this;
        auto conv_to_gemm_transformer_right = *this;
        IndexType a_right_offset            = 0;
        IndexType c_right_offset            = 0;
        // Calculate real filter size
        const IndexType z_eff = (Z_ - 1) * ConvDilationD_ + 1;
        const IndexType y_eff = (Y_ - 1) * ConvDilationH_ + 1;
        const IndexType x_eff = (X_ - 1) * ConvDilationW_ + 1;
        // Calculate start position in input for right tensor
        const IndexType di_right_transformer_start_idx = (Do_ / 2) * ConvStrideD_;
        const IndexType hi_right_transformer_start_idx = (Ho_ / 2) * ConvStrideH_;
        const IndexType wi_right_transformer_start_idx = (Wo_ / 2) * ConvStrideW_;
        // Calculate last position in input for left tensor
        const IndexType di_left_transformer_end_idx = (Do_ / 2 - 1) * ConvStrideD_ + z_eff;
        const IndexType hi_left_transformer_end_idx = (Ho_ / 2 - 1) * ConvStrideH_ + y_eff;
        const IndexType wi_left_transformer_end_idx = (Wo_ / 2 - 1) * ConvStrideW_ + x_eff;
        // Allow to split if whole left padding will be in left tensor and right padding in right
        // tensor
        const bool is_possible_to_split_d = Do_ != 1 &&
                                            di_right_transformer_start_idx > InLeftPadD_ &&
                                            di_left_transformer_end_idx <= (InLeftPadD_ + Di_);
        const bool is_possible_to_split_h = Ho_ != 1 &&
                                            hi_right_transformer_start_idx > InLeftPadH_ &&
                                            hi_left_transformer_end_idx <= (InLeftPadH_ + Hi_);
        const bool is_possible_to_split_w = Wo_ != 1 &&
                                            wi_right_transformer_start_idx > InLeftPadW_ &&
                                            wi_left_transformer_end_idx <= (InLeftPadW_ + Wi_);

        if(is_possible_to_split_d)
        {
            // Apply new sizes
            // Split output on half
            conv_to_gemm_transformer_left.Do_  = Do_ / 2;
            conv_to_gemm_transformer_right.Do_ = Do_ - Do_ / 2;
            // Assign left padding to left convolution
            conv_to_gemm_transformer_left.InLeftPadD_  = InLeftPadD_;
            conv_to_gemm_transformer_right.InLeftPadD_ = 0;
            // Assign right padding to right convolution
            conv_to_gemm_transformer_left.InRightPadD_  = 0;
            conv_to_gemm_transformer_right.InRightPadD_ = InRightPadD_;
            // Calculate new input size
            conv_to_gemm_transformer_left.Di_ = di_left_transformer_end_idx - InLeftPadD_;
            conv_to_gemm_transformer_right.Di_ =
                math::min(Di_ - (di_right_transformer_start_idx - InLeftPadD_),
                          (conv_to_gemm_transformer_right.Do_ - 1) * ConvStrideD_ + z_eff);
            ;
            // Calcualte offsets
            a_right_offset = ((Do_ / 2) * ConvStrideD_ - InLeftPadD_) * DiStride_;
            c_right_offset = (Do_ / 2) * DoStride_;
        }
        else if(is_possible_to_split_h)
        {
            conv_to_gemm_transformer_left.Ho_  = Ho_ / 2;
            conv_to_gemm_transformer_right.Ho_ = Ho_ - Ho_ / 2;

            conv_to_gemm_transformer_left.InLeftPadH_  = InLeftPadH_;
            conv_to_gemm_transformer_right.InLeftPadH_ = 0;

            conv_to_gemm_transformer_left.InRightPadH_  = 0;
            conv_to_gemm_transformer_right.InRightPadH_ = InRightPadH_;

            conv_to_gemm_transformer_left.Hi_ = hi_left_transformer_end_idx - InLeftPadH_;
            conv_to_gemm_transformer_right.Hi_ =
                math::min(Hi_ - (hi_right_transformer_start_idx - InLeftPadH_),
                          (conv_to_gemm_transformer_right.Ho_ - 1) * ConvStrideH_ + y_eff);
            a_right_offset = ((Ho_ / 2) * ConvStrideH_ - InLeftPadH_) * HiStride_;
            c_right_offset = (Ho_ / 2) * HoStride_;
        }
        else if(is_possible_to_split_w)
        {
            conv_to_gemm_transformer_left.Wo_  = Wo_ / 2;
            conv_to_gemm_transformer_right.Wo_ = Wo_ - Wo_ / 2;

            conv_to_gemm_transformer_left.InLeftPadW_  = InLeftPadW_;
            conv_to_gemm_transformer_right.InLeftPadW_ = 0;

            conv_to_gemm_transformer_left.InRightPadW_  = 0;
            conv_to_gemm_transformer_right.InRightPadW_ = InRightPadW_;

            conv_to_gemm_transformer_left.Wi_ = wi_left_transformer_end_idx - InLeftPadW_;
            conv_to_gemm_transformer_right.Wi_ =
                math::min(Wi_ - (wi_right_transformer_start_idx - InLeftPadW_),
                          (conv_to_gemm_transformer_right.Wo_ - 1) * ConvStrideW_ + x_eff);

            a_right_offset = ((Wo_ / 2) * ConvStrideW_ - InLeftPadW_) * WiStride_;
            c_right_offset = (Wo_ / 2) * WoStride_;
        }

        static constexpr index_t NumDTensor = DsPointer::Size();
        const auto ds_grid_right_ptr        = generate_tuple(
            [&](auto i) { return ds_grid_ptr_base(i) + c_right_offset; }, Number<NumDTensor>{});

        // Return left transform, right transformer, right offset to Input and right offset to
        // Output
        return ck::make_tuple(conv_to_gemm_transformer_left,
                              conv_to_gemm_transformer_right,
                              a_grid_ptr_base + a_right_offset,
                              ds_grid_right_ptr,
                              c_grid_ptr_base + c_right_offset);
    }

    // TODO: implement ck::tensor_layout::convolution that describe packed/strided dimemsion as
    // properties
    template <typename ALayout,
              typename ck::enable_if<NDimSpatial == 1 &&
                                         (is_same_v<ALayout, tensor_layout::convolution::G_NW_C> ||
                                          is_same_v<ALayout, tensor_layout::convolution::NWGC> ||
                                          is_same_v<ALayout, tensor_layout::convolution::GNWC>),
                                     bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const
    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wo_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_));
                return transform_tensor_descriptor(
                    in_gemmm_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {

                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_), make_tuple(NStrideTensorA_, WiStride_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(Number<3>{})),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(Number<3>{})),
                    make_tuple(Sequence<0, 2, 3>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_));

                const auto in_n_wo_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return transform_tensor_descriptor(
                    in_n_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_wo_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_merge_transform(make_tuple(X_, C_))),
                    make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(X_, C_))),
                    make_tuple(Sequence<0, 2, 3>{}, Sequence<1, 4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename ck::enable_if<NDimSpatial == 2 &&
                                         (is_same_v<ALayout, tensor_layout::convolution::G_NHW_C> ||
                                          is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                                          is_same_v<ALayout, tensor_layout::convolution::GNHWC>),
                                     bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const

    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_), make_tuple(NStrideTensorA_, HiStride_, WiStride_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_));

                const auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_y_ho_x_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_ho_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_ho_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_ho_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {

                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_y_ho_x_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5>{},
                               Sequence<6>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3, 6>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename ck::enable_if<
                  NDimSpatial == 3 && (is_same_v<ALayout, tensor_layout::convolution::G_NDHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NDHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNDHWC>),
                  bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const

    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Do_, Ho_, Wo_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_,
                               DiStride_,
                               HiStride_,
                               WiStride_,
                               GStrideTensorA_,
                               CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}, Sequence<5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                        make_merge_transform(make_tuple(Number<3>{}, Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, GStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_merge_transform(make_tuple(Number<3>{}, Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4, 6, 7>{}, Sequence<1, 3, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Do_), make_tuple(ConvStrideD_)),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_do_ho_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_,
                               DiStride_,
                               HiStride_,
                               WiStride_,
                               GStrideTensorA_,
                               CStrideTensorA_));

                const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Do_), make_tuple(ConvStrideD_)),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_do_ho_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}, Sequence<5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Z_, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Z_, Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_,
                               DiStride_,
                               HiStride_,
                               WiStride_,
                               GStrideTensorA_,
                               CStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Z_, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{},
                               Sequence<8>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_merge_transform(make_tuple(Z_, Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4, 6, 7>{}, Sequence<1, 3, 5, 8>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename ck::enable_if<NDimSpatial == 1 &&
                                         is_same_v<ALayout, tensor_layout::convolution::NGCW>,
                                     bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const
    {
        static_assert(NumGroupsToMerge == 1);
        static_assert(ConvForwardSpecialization ==
                      device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0);

        const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Wo_, C_), make_tuple(NStrideTensorA_, I1, CStrideTensorA_));

        return transform_tensor_descriptor(
            in_gemmm_gemmk_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Wo_)), make_pass_through_transform(C_)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <typename ALayout,
              typename ck::enable_if<NDimSpatial == 2 &&
                                         is_same_v<ALayout, tensor_layout::convolution::NGCHW>,
                                     bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const
    {
        static_assert(NumGroupsToMerge == 1);
        static_assert(ConvForwardSpecialization ==
                      device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0);

        const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Ho_ * Wo_, C_), make_tuple(NStrideTensorA_, I1, CStrideTensorA_));

        return transform_tensor_descriptor(
            in_gemmm_gemmk_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Ho_ * Wo_)),
                       make_pass_through_transform(C_)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <typename ALayout,
              typename ck::enable_if<NDimSpatial == 3 &&
                                         is_same_v<ALayout, tensor_layout::convolution::NGCDHW>,
                                     bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const
    {
        static_assert(NumGroupsToMerge == 1);
        static_assert(ConvForwardSpecialization ==
                      device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0);

        const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
            make_tuple(N_, Do_ * Ho_ * Wo_, C_), make_tuple(NStrideTensorA_, I1, CStrideTensorA_));

        return transform_tensor_descriptor(
            in_gemmm_gemmk_desc,
            make_tuple(make_merge_transform(make_tuple(N_, Do_ * Ho_ * Wo_)),
                       make_pass_through_transform(C_)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <typename BLayout,
              typename ck::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKCX> ||
                                         is_same_v<BLayout, tensor_layout::convolution::GKCYX> ||
                                         is_same_v<BLayout, tensor_layout::convolution::GKCZYX>,
                                     bool>::type = false>
    __host__ __device__ auto MakeBDescriptor_N_K() const
    {
        static_assert(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0 ||
                      ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0);
        static_assert(NumGroupsToMerge == 1);
        return make_naive_tensor_descriptor_packed(make_tuple(K_, C_));
    }

    template <typename BLayout,
              typename ck::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                         is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                         is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                     bool>::type = false>
    __host__ __device__ auto MakeBDescriptor_N_K() const
    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter3x3)
        {
            using FilterSizeNumType =
                ck::conditional_t<NDimSpatial == 1,
                                  Number<3>,
                                  ck::conditional_t<NDimSpatial == 2, Number<9>, Number<27>>>;

            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor_packed(make_tuple(K_, FilterSizeNumType{}));
            }
            else
            {

                const auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(K_, NumGroupsToMerge, FilterSizeNumType{}),
                    make_tuple(KStrideTensorB_, GStrideTensorB_, CStrideTensorB_));
                return transform_tensor_descriptor(
                    wei_gemmn_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(K_, NumGroupsToMerge)),
                               make_pass_through_transform(FilterSizeNumType{})),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor_packed(make_tuple(K_, ZYX_ * C_));
            }
            else
            {
                const auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(K_, NumGroupsToMerge, ZYX_ * C_),
                    make_tuple(KStrideTensorB_, GStrideTensorB_, CStrideTensorB_));
                return transform_tensor_descriptor(
                    wei_gemmn_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(K_, NumGroupsToMerge)),
                               make_pass_through_transform(ZYX_ * C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <
        typename BLayout,
        typename ck::enable_if<is_same_v<BLayout, tensor_layout::convolution::G_K_X_C> ||
                                   is_same_v<BLayout, tensor_layout::convolution::G_K_YX_C> ||
                                   is_same_v<BLayout, tensor_layout::convolution::G_K_ZYX_C> ||
                                   is_same_v<BLayout, tensor_layout::convolution::KXGC> ||
                                   is_same_v<BLayout, tensor_layout::convolution::KYXGC> ||
                                   is_same_v<BLayout, tensor_layout::convolution::KZYXGC>,
                               bool>::type = false>
    __host__ __device__ auto MakeBDescriptor_N_K() const
    {
        const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
            make_tuple(K_, ZYX_, C_), make_tuple(KStrideTensorB_, XStride_, CStrideTensorB_));

        const auto wei_gemmn_gemmk_desc = transform_tensor_descriptor(
            wei_k_yx_c_desc,
            make_tuple(make_pass_through_transform(K_), make_merge_transform(make_tuple(ZYX_, C_))),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return wei_gemmn_gemmk_desc;
    }

    template <
        typename CLayout,
        index_t NDimSp = NDimSpatial,

        typename ck::enable_if<NDimSp == 1 && (is_same_v<CLayout, tensor_layout::convolution::G_K>),
                               bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        if constexpr(CTranspose)
        {
            return make_naive_tensor_descriptor(make_tuple(K_, N_ * Wo_),
                                                make_tuple(KStrideTensorC_, I0));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Wo_, K_),
                                                make_tuple(I0, KStrideTensorC_));
        }
    }

    template <
        typename CLayout,
        index_t NDimSp = NDimSpatial,

        typename ck::enable_if<NDimSp == 2 && (is_same_v<CLayout, tensor_layout::convolution::G_K>),
                               bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        if constexpr(CTranspose)
        {
            return make_naive_tensor_descriptor(make_tuple(K_, N_ * Ho_ * Wo_),
                                                make_tuple(KStrideTensorC_, I0));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Ho_ * Wo_, K_),
                                                make_tuple(I0, KStrideTensorC_));
        }
    }

    template <
        typename CLayout,
        index_t NDimSp = NDimSpatial,

        typename ck::enable_if<NDimSp == 3 && (is_same_v<CLayout, tensor_layout::convolution::G_K>),
                               bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        if constexpr(CTranspose)
        {
            return make_naive_tensor_descriptor(make_tuple(K_, N_ * Do_ * Ho_ * Wo_),
                                                make_tuple(KStrideTensorC_, I0));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Do_ * Ho_ * Wo_, K_),
                                                make_tuple(I0, KStrideTensorC_));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,

              typename ck::enable_if<NDimSp == 1 &&
                                         (is_same_v<CLayout, tensor_layout::convolution::G_NW_K> ||
                                          is_same_v<CLayout, tensor_layout::convolution::NWGK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNWK>),
                                     bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(CTranspose == false);
        const IndexType NDoHoWo = N_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_));
        }
        else
        {
            const auto nhwo_groups_k_1_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Wo_, NumGroupsToMerge, K_, 1),
                make_tuple(
                    NStrideTensorC_, WoStride_, GStrideTensorC_, KStrideTensorC_, GStrideTensorC_));
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,

              typename ck::enable_if<NDimSp == 2 &&
                                         (is_same_v<CLayout, tensor_layout::convolution::G_NHW_K> ||
                                          is_same_v<CLayout, tensor_layout::convolution::NHWGK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNHWK>),
                                     bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(CTranspose == false);
        const IndexType NDoHoWo = N_ * Ho_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_));
        }
        else
        {
            const auto nhwo_groups_k_1_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, K_, 1),
                                             make_tuple(NStrideTensorC_,
                                                        HoStride_,
                                                        WoStride_,
                                                        GStrideTensorC_,
                                                        KStrideTensorC_,
                                                        GStrideTensorC_));
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,
              typename ck::enable_if<
                  NDimSp == 3 && (is_same_v<CLayout, tensor_layout::convolution::G_NDHW_K> ||
                                  is_same_v<CLayout, tensor_layout::convolution::NDHWGK> ||
                                  is_same_v<CLayout, tensor_layout::convolution::GNDHWK>),
                  bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(CTranspose == false);
        const IndexType NDoHoWo = N_ * Do_ * Ho_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_));
        }
        else
        {
            const auto nhwo_groups_k_1_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge, K_, 1),
                                             make_tuple(NStrideTensorC_,
                                                        DoStride_,
                                                        HoStride_,
                                                        WoStride_,
                                                        GStrideTensorC_,
                                                        KStrideTensorC_,
                                                        GStrideTensorC_));
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}, Sequence<5>{}, Sequence<6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,

              typename ck::enable_if<NDimSp == 1 &&
                                         (is_same_v<CLayout, tensor_layout::convolution::GNKW> ||
                                          is_same_v<CLayout, tensor_layout::convolution::NGKW>),
                                     bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(NumGroupsToMerge == 1);
        auto n_k_wo_desc = make_naive_tensor_descriptor(
            make_tuple(N_, K_, Wo_), make_tuple(NStrideTensorC_, KStrideTensorC_, I1));
        if constexpr(CTranspose)
        {
            return transform_tensor_descriptor(
                n_k_wo_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(N_, Wo_))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(n_k_wo_desc,
                                               make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                                                          make_pass_through_transform(K_)),
                                               make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,

              typename ck::enable_if<NDimSp == 2 &&
                                         (is_same_v<CLayout, tensor_layout::convolution::GNKHW> ||
                                          is_same_v<CLayout, tensor_layout::convolution::NGKHW>),
                                     bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(NumGroupsToMerge == 1);
        auto n_k_howo_desc = make_naive_tensor_descriptor(
            make_tuple(N_, K_, Ho_ * Wo_), make_tuple(NStrideTensorC_, KStrideTensorC_, I1));
        if constexpr(CTranspose)
        {
            return transform_tensor_descriptor(
                n_k_howo_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(N_, Ho_ * Wo_))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                n_k_howo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_ * Wo_)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,

              typename ck::enable_if<NDimSp == 3 &&
                                         (is_same_v<CLayout, tensor_layout::convolution::GNKDHW> ||
                                          is_same_v<CLayout, tensor_layout::convolution::NGKDHW>),
                                     bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(NumGroupsToMerge == 1);
        auto n_k_dohowo_desc = make_naive_tensor_descriptor(
            make_tuple(N_, K_, Do_ * Ho_ * Wo_), make_tuple(NStrideTensorC_, KStrideTensorC_, I1));

        if constexpr(CTranspose)
        {
            return transform_tensor_descriptor(
                n_k_dohowo_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_merge_transform(make_tuple(N_, Do_ * Ho_ * Wo_))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                n_k_dohowo_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Do_ * Ho_ * Wo_)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }
    IndexType N_;
    IndexType Di_, Hi_, Wi_;
    IndexType Do_, Ho_, Wo_;
    IndexType Z_, Y_, X_;
    IndexType K_, C_;
    IndexType DiStride_, HiStride_, WiStride_;
    IndexType DoStride_, HoStride_, WoStride_;
    IndexType XStride_;
    IndexType CStrideTensorA_, CStrideTensorB_, KStrideTensorB_, KStrideTensorC_;
    IndexType NStrideTensorA_, NStrideTensorC_;
    IndexType GStrideTensorA_, GStrideTensorB_, GStrideTensorC_;
    IndexType ConvStrideD_, ConvStrideH_, ConvStrideW_;
    IndexType ConvDilationD_, ConvDilationH_, ConvDilationW_;
    IndexType InLeftPadD_, InLeftPadH_, InLeftPadW_;
    IndexType InRightPadD_, InRightPadH_, InRightPadW_;
    IndexType ZYX_;
};

// wrapper class to call member functions on TransformConvToGemm struct at runtime
// TODO: figure out aq way to properly pass in layout as an argument
struct TransformConv
{
    TransformConv() {}

    template <index_t NDimSpatial,
              device::ConvolutionForwardSpecialization ConvForwardSpecialization>
    auto
    transform_func(TransformConvFwdToGemm<NDimSpatial, ConvForwardSpecialization> conv_fwd_to_gemm)
    {
        if(NDimSpatial == 2)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<ck::tensor_layout::convolution::NHWGK, 2>();
        }
        else if(NDimSpatial == 3)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<tensor_layout::convolution::NDHWGK, 3>();
        }
        else if(NDimSpatial == 1)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<tensor_layout::convolution::NWGK, 1>();
        }
    }
};

} // namespace tensor_operation
} // namespace ck
