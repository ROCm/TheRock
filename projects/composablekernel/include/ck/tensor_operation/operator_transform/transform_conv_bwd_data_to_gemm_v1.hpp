// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

namespace ck {
namespace tensor_operation {

/**
 * @brief Enable custom tensor transform for convolution backward data output.
 *
 * When set to 1, this macro enables a custom transformation of the output tensor
 * in convolution backward data operations.
 */
#define CK_USE_CUSTOM_TENSOR_TRANSFORM_FOR_BWD_DATA_OUT 1

template <
    index_t NDimSpatial,
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization ConvBwdDataSpecialization,
    index_t AK1,
    index_t BK1,
    index_t GemmMPerBlock,
    index_t GemmNPerBlock,
    index_t GemmKPerBlock,
    bool DoPadGemmM,
    bool DoPadGemmN,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    bool SplitN              = false,
    typename ADataType       = float,
    typename CDataType       = float,
    index_t NumGroupsToMerge = 1,
    typename IndexType       = index_t,
    bool CTranspose          = false>
struct TransformConvBwdDataToGemm_v1
{
    private:
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto NonSpatialDimsNum = Number<3>{};

    static constexpr auto DIdx = NonSpatialDimsNum;
    static constexpr auto HIdx =
        NDimSpatial == 2 ? NonSpatialDimsNum : Number<NonSpatialDimsNum + 1>{};
    static constexpr auto WIdx =
        NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{} : Number<NonSpatialDimsNum + 2>{};

    static constexpr auto ZIdx = NonSpatialDimsNum;
    static constexpr auto YIdx =
        NDimSpatial == 2 ? NonSpatialDimsNum : Number<NonSpatialDimsNum + 1>{};
    static constexpr auto XIdx =
        NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{} : Number<NonSpatialDimsNum + 2>{};

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
    static IndexType GetSplitedNSize(const ConvDimsType& a_g_n_k_wos_lengths,
                                     const ConvDimsType& a_g_n_k_wos_strides,
                                     const ConvDimsType& c_g_n_c_wis_lengths,
                                     const ConvDimsType& c_g_n_c_wis_strides)
    {
        const long_index_t a_element_space_size =
            calculate_element_space_size_impl(a_g_n_k_wos_lengths, a_g_n_k_wos_strides, I1);
        const long_index_t c_element_space_size =
            calculate_element_space_size_impl(c_g_n_c_wis_lengths, c_g_n_c_wis_strides, I1);
        const long_index_t element_space_size = math::max(a_element_space_size * sizeof(ADataType),
                                                          c_element_space_size * sizeof(CDataType));
        constexpr long_index_t TwoGB          = (long_index_t{1} << 31);

        const IndexType N = a_g_n_k_wos_lengths[I1];

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
    __host__ __device__ constexpr TransformConvBwdDataToGemm_v1() {}

    template <typename TransformConvBwdDataToGemm_v1Base>
    __host__ __device__ TransformConvBwdDataToGemm_v1(
        const TransformConvBwdDataToGemm_v1Base& transform_conv_bwd_data_to_gemm_base)
        : N_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.N_)},
          Di_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Di_)},
          Hi_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Hi_)},
          Wi_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Wi_)},
          Do_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Do_)},
          Ho_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Ho_)},
          Wo_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Wo_)},
          Z_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Z_)},
          Y_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.Y_)},
          X_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.X_)},
          K_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.K_)},
          C_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.C_)},
          DiStride_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.DiStride_)},
          HiStride_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.HiStride_)},
          WiStride_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.WiStride_)},
          DoStride_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.DoStride_)},
          HoStride_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.HoStride_)},
          WoStride_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.WoStride_)},
          CStrideTensorB_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.CStrideTensorB_)},
          CStrideTensorC_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.CStrideTensorC_)},
          KStrideTensorA_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.KStrideTensorA_)},
          KStrideTensorB_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.KStrideTensorB_)},
          NStrideTensorA_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.NStrideTensorA_)},
          NStrideTensorC_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.NStrideTensorC_)},
          ConvStrideD_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ConvStrideD_)},
          ConvStrideH_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ConvStrideH_)},
          ConvStrideW_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ConvStrideW_)},
          ConvDilationD_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ConvDilationD_)},
          ConvDilationH_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ConvDilationH_)},
          ConvDilationW_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ConvDilationW_)},
          InLeftPadD_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.InLeftPadD_)},
          InLeftPadH_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.InLeftPadH_)},
          InLeftPadW_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.InLeftPadW_)},
          InRightPadD_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.InRightPadD_)},
          InRightPadH_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.InRightPadH_)},
          InRightPadW_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.InRightPadW_)},
          IdxZTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.IdxZTilde_)},
          IdxYTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.IdxYTilde_)},
          IdxXTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.IdxXTilde_)},
          GcdStrideDilationD_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.GcdStrideDilationD_)},
          GcdStrideDilationH_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.GcdStrideDilationH_)},
          GcdStrideDilationW_{
              static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.GcdStrideDilationW_)},
          ZTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ZTilde_)},
          YTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.YTilde_)},
          XTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.XTilde_)},
          DTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.DTilde_)},
          HTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.HTilde_)},
          WTilde_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.WTilde_)},
          ZDot_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.ZDot_)},
          YDot_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.YDot_)},
          XDot_{static_cast<IndexType>(transform_conv_bwd_data_to_gemm_base.XDot_)},
          batch_k_{transform_conv_bwd_data_to_gemm_base.batch_k_}
    {
    }

    template <typename ConvDimsType, typename ConvSpatialDimsType>
    __host__ __device__
    TransformConvBwdDataToGemm_v1(const ConvDimsType& a_g_n_k_wos_lengths,
                                  const ConvDimsType& a_g_n_k_wos_strides,
                                  const ConvDimsType& b_g_k_c_xs_lengths,
                                  const ConvDimsType& b_g_k_c_xs_strides,
                                  const ConvDimsType& c_g_n_c_wis_lengths,
                                  const ConvDimsType& c_g_n_c_wis_strides,
                                  const ConvSpatialDimsType& conv_filter_strides,
                                  const ConvSpatialDimsType& conv_filter_dilations,
                                  const ConvSpatialDimsType& input_left_pads,
                                  const ConvSpatialDimsType& input_right_pads,
                                  const ConvSpatialDimsType& tildes,
                                  const index_t batch_k = 1)
        : Hi_{c_g_n_c_wis_lengths[HIdx]},
          Wi_{c_g_n_c_wis_lengths[WIdx]},
          Ho_{a_g_n_k_wos_lengths[HIdx]},
          Wo_{a_g_n_k_wos_lengths[WIdx]},
          Y_{b_g_k_c_xs_lengths[YIdx]},
          X_{b_g_k_c_xs_lengths[XIdx]},
          K_{a_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          HiStride_{c_g_n_c_wis_strides[HIdx]},
          WiStride_{c_g_n_c_wis_strides[WIdx]},
          HoStride_{a_g_n_k_wos_strides[HIdx]},
          WoStride_{a_g_n_k_wos_strides[WIdx]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          CStrideTensorC_{c_g_n_c_wis_strides[I2]},
          KStrideTensorA_{a_g_n_k_wos_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          NStrideTensorA_{a_g_n_k_wos_strides[I1]},
          NStrideTensorC_{c_g_n_c_wis_strides[I1]},
          ConvStrideH_{conv_filter_strides[HIdx - NonSpatialDimsNum]},
          ConvStrideW_{conv_filter_strides[WIdx - NonSpatialDimsNum]},
          ConvDilationH_{conv_filter_dilations[HIdx - NonSpatialDimsNum]},
          ConvDilationW_{conv_filter_dilations[WIdx - NonSpatialDimsNum]},
          InLeftPadH_{input_left_pads[HIdx - NonSpatialDimsNum]},
          InLeftPadW_{input_left_pads[WIdx - NonSpatialDimsNum]},
          InRightPadH_{input_right_pads[HIdx - NonSpatialDimsNum]},
          InRightPadW_{input_right_pads[WIdx - NonSpatialDimsNum]},
          IdxYTilde_{tildes[YIdx - NonSpatialDimsNum]},
          IdxXTilde_{tildes[XIdx - NonSpatialDimsNum]},
          batch_k_{batch_k}
    {
        static_assert(is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<IndexType, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<IndexType, NDimSpatial + I3>>);

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_k_wos_lengths, a_g_n_k_wos_strides, c_g_n_c_wis_lengths, c_g_n_c_wis_strides);
        }
        else
        {
            N_ = c_g_n_c_wis_lengths[I1];
        }
        if constexpr(NDimSpatial == 3)
        {
            Di_                 = c_g_n_c_wis_lengths[DIdx];
            Do_                 = a_g_n_k_wos_lengths[DIdx];
            Z_                  = b_g_k_c_xs_lengths[ZIdx];
            DiStride_           = c_g_n_c_wis_strides[DIdx];
            DoStride_           = a_g_n_k_wos_strides[DIdx];
            ConvStrideD_        = conv_filter_strides[DIdx - NonSpatialDimsNum];
            ConvDilationD_      = conv_filter_dilations[DIdx - NonSpatialDimsNum];
            InLeftPadD_         = input_left_pads[DIdx - NonSpatialDimsNum];
            InRightPadD_        = input_right_pads[DIdx - NonSpatialDimsNum];
            IdxZTilde_          = tildes[ZIdx - NonSpatialDimsNum];
            GcdStrideDilationD_ = math::gcd(ConvStrideD_, ConvDilationD_);
            ZTilde_             = ConvStrideD_ / GcdStrideDilationD_;
            DTilde_ = Do_ + math::integer_divide_ceil(ConvDilationD_ * (Z_ - I1), ConvStrideD_);
            ZDot_   = math::integer_divide_ceil(Z_, ZTilde_);
        }
        else
        {
            Di_ = Do_ = Z_ = ZTilde_ = ConvStrideD_ = DTilde_ = ZDot_ = 1;
            InLeftPadD_ = InRightPadD_ = DiStride_ = DoStride_ = IdxZTilde_ = 0;
        }

        GcdStrideDilationH_ = math::gcd(ConvStrideH_, ConvDilationH_);
        GcdStrideDilationW_ = math::gcd(ConvStrideW_, ConvDilationW_);

        YTilde_ = ConvStrideH_ / GcdStrideDilationH_;
        XTilde_ = ConvStrideW_ / GcdStrideDilationW_;

        HTilde_ = Ho_ + math::integer_divide_ceil(ConvDilationH_ * (Y_ - I1), ConvStrideH_);
        WTilde_ = Wo_ + math::integer_divide_ceil(ConvDilationW_ * (X_ - I1), ConvStrideW_);

        YDot_ = math::integer_divide_ceil(Y_, YTilde_);
        XDot_ = math::integer_divide_ceil(X_, XTilde_);
    }

#if 0 // At now not supported to split tensor
    __host__ bool AreDescriptorsSmallerThan2GB() const
    {
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        const long_index_t in_desc_space_size =
            I1 + (N_ - I1) * NStrideTensorC_ + (Di_ - I1) * DiStride_ + (Hi_ - I1) * HiStride_ +
            (Wi_ - I1) * WiStride_ + (C_ - I1) * CStrideTensorC_;
        const long_index_t out_desc_space_size =
            I1 + (N_ - I1) * NStrideTensorA_ + (Do_ - I1) * DoStride_ + (Ho_ - I1) * HoStride_ +
            (Wo_ - I1) * WoStride_ + (K_ - I1) * KStrideTensorA_;

        bool is_a_descriptor_smaller_than_2GB = (out_desc_space_size * sizeof(ADataType)) <= TwoGB;
        bool is_c_descriptor_smaller_than_2GB = (in_desc_space_size * sizeof(CDataType)) <= TwoGB;

        return is_a_descriptor_smaller_than_2GB && is_c_descriptor_smaller_than_2GB;
    }

    __host__ auto SplitConvProblem(const ADataType* a_grid_ptr_base,
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
            a_right_offset = (Do_ / 2) * DoStride_;
            c_right_offset = ((Do_ / 2) * ConvStrideD_ - InLeftPadD_) * DiStride_;
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
            a_right_offset = (Ho_ / 2) * HoStride_;
            c_right_offset = ((Ho_ / 2) * ConvStrideH_ - InLeftPadH_) * HiStride_;
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

            a_right_offset = (Wo_ / 2) * WoStride_;
            c_right_offset = ((Wo_ / 2) * ConvStrideW_ - InLeftPadW_) * WiStride_;
        }
        // Return left transform, right transformer, right offset to Input and right offset to
        // Output
        return ck::make_tuple(conv_to_gemm_transformer_left,
                              conv_to_gemm_transformer_right,
                              a_grid_ptr_base + a_right_offset,
                              c_grid_ptr_base + c_right_offset);
    }

    __host__ auto SplitConvProblem(const ADataType* a_grid_ptr_base,
                                   CDataType* c_grid_ptr_base) const
    {
        // Create copies
        auto conv_to_gemm_transformer_left  = *this;
        auto conv_to_gemm_transformer_right = *this;
        IndexType a_right_offset            = 0;
        IndexType c_right_offset            = 0;

        // Calculate start position in input for right tensor
        const IndexType do_right_transformer_start_idx = math::integer_divide_ceil((Di_ / 2) + InLeftPadD_ - ((Z_ - 1) * ConvDilationD_), ConvStrideD_);
        const IndexType ho_right_transformer_start_idx = math::integer_divide_ceil((Hi_ / 2) + InLeftPadH_ - ((Y_ - 1) * ConvDilationH_), ConvStrideH_);
        const IndexType wo_right_transformer_start_idx = math::integer_divide_ceil((Wi_ / 2) + InLeftPadW_ - ((X_ - 1) * ConvDilationW_), ConvStrideW_);
        // Calculate last position in input for left tensor
        const IndexType do_left_transformer_end_idx = math::integer_divide_ceil((Di_ / 2 - 1) + InLeftPadD_, ConvStrideD_);
        const IndexType ho_left_transformer_end_idx = math::integer_divide_ceil((Hi_ / 2 - 1) + InLeftPadH_, ConvStrideH_);
        const IndexType wo_left_transformer_end_idx = math::integer_divide_ceil((Wi_ / 2 - 1) + InLeftPadW_, ConvStrideW_);


        if(Di_!=1)
        {
            // Apply new sizes
            // Split output on half
            conv_to_gemm_transformer_left.Di_  = Di_ / 2;
            conv_to_gemm_transformer_right.Di_ = Di_ - Di_ / 2;
            // Assign left padding to left convolution
            conv_to_gemm_transformer_left.InLeftPadD_  = InLeftPadD_;
            conv_to_gemm_transformer_right.InLeftPadD_ = 0;
            // // Assign right padding to right convolution
            conv_to_gemm_transformer_left.InRightPadD_  = 0;
            conv_to_gemm_transformer_right.InRightPadD_ = InRightPadD_;
            // Calculate new input size
            conv_to_gemm_transformer_left.Do_ = do_left_transformer_end_idx;
            conv_to_gemm_transformer_right.Do_ = Do_ - do_right_transformer_start_idx;
            ;
            // Calcualte offsets
            a_right_offset = do_right_transformer_start_idx * DoStride_;
            c_right_offset = (Di_ / 2) * DiStride_;
        }
        else if(Hi_!=1)
        {
            // Apply new sizes
            // Split output on half
            conv_to_gemm_transformer_left.Hi_  = Hi_ / 2;
            conv_to_gemm_transformer_right.Hi_ = Hi_ - Hi_ / 2;
            // Assign left padding to left convolution
            conv_to_gemm_transformer_left.InLeftPadH_  = InLeftPadH_;
            conv_to_gemm_transformer_right.InLeftPadH_ = 0;
            // // Assign right padding to right convolution
            conv_to_gemm_transformer_left.InRightPadH_  = 0;
            conv_to_gemm_transformer_right.InRightPadH_ = InRightPadH_;
            // Calculate new input size
            conv_to_gemm_transformer_left.Ho_ = ho_left_transformer_end_idx ;
            conv_to_gemm_transformer_right.Ho_ = Ho_ - ho_right_transformer_start_idx ;
            ;
            // Calcualte offsets
            a_right_offset = ho_right_transformer_start_idx * HoStride_;
            c_right_offset = (Hi_ / 2) * HiStride_;
        }
        else if(Wi_!=1)
        {
            // Apply new sizes
            // Split output on half
            conv_to_gemm_transformer_left.Wi_  = Wi_ / 2;
            conv_to_gemm_transformer_right.Wi_ = Wi_ - Wi_ / 2;
            // Assign left padding to left convolution
            conv_to_gemm_transformer_left.InLeftPadW_  = InLeftPadW_;
            conv_to_gemm_transformer_right.InLeftPadW_ = 0;
            // Assign right padding to right convolution
            conv_to_gemm_transformer_left.InRightPadW_  = 0;
            conv_to_gemm_transformer_right.InRightPadW_ = InRightPadW_;
            // Calculate new input size
            conv_to_gemm_transformer_left.Wo_ = wo_left_transformer_end_idx;
            conv_to_gemm_transformer_right.Wo_ = Wo_ - wo_right_transformer_start_idx;
            ;
            // Calcualte offsets
            a_right_offset = wo_right_transformer_start_idx * WoStride_;
            c_right_offset = (Wi_ / 2) * WiStride_;
        }
        // Return left transform, right transformer, right offset to Input and right offset to
        // Output
        return ck::make_tuple(conv_to_gemm_transformer_left,
                              conv_to_gemm_transformer_right,
                              a_grid_ptr_base + a_right_offset,
                              c_grid_ptr_base + c_right_offset);
    }
#endif

    __host__ __device__ auto MakeOutGridDesc() const
    {
        if constexpr(is_same_v<ALayout, tensor_layout::convolution::NHWGK>)
        {
            if constexpr(ConvBwdDataSpecialization ==
                         ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                             Filter1x1Stride1Pad0)
            {

                return make_naive_tensor_descriptor(make_tuple(N_ * Ho_ * Wo_, K_),
                                                    make_tuple(WoStride_, KStrideTensorA_));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, K_),
                    make_tuple(NStrideTensorA_, HoStride_, WoStride_, KStrideTensorA_));
            }
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::convolution::NDHWGK>)
        {
            if constexpr(ConvBwdDataSpecialization ==
                         ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                             Filter1x1Stride1Pad0)
            {

                return make_naive_tensor_descriptor(make_tuple(N_ * Do_ * Ho_ * Wo_, K_),
                                                    make_tuple(WoStride_, KStrideTensorA_));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Do_, Ho_, Wo_, K_),
                    make_tuple(NStrideTensorA_, DoStride_, HoStride_, WoStride_, KStrideTensorA_));
            }
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNHWK>)
        {
            // assume packed
            if constexpr(ConvBwdDataSpecialization ==
                         ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                             Filter1x1Stride1Pad0)
            {
                return make_naive_tensor_descriptor_packed(make_tuple(N_ * Ho_ * Wo_, K_));
            }
            else
            {
                return make_naive_tensor_descriptor_packed(make_tuple(N_, Ho_, Wo_, K_));
            }
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNDHWK>)
        {
            // assume packed
            if constexpr(ConvBwdDataSpecialization ==
                         ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                             Filter1x1Stride1Pad0)
            {
                return make_naive_tensor_descriptor_packed(make_tuple(N_ * Do_ * Ho_ * Wo_, K_));
            }
            else
            {
                return make_naive_tensor_descriptor_packed(make_tuple(N_, Do_, Ho_, Wo_, K_));
            }
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::convolution::NGKHW>)
        {
            // assume packed
            static_assert(ConvBwdDataSpecialization ==
                          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                              Filter1x1Stride1Pad0);

            const auto out_gemm_raw_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Ho_ * Wo_, K_), make_tuple(NStrideTensorA_, I1, KStrideTensorA_));

            return transform_tensor_descriptor(
                out_gemm_raw_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_ * Wo_)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::convolution::NGKDHW>)
        {
            // assume packed
            static_assert(ConvBwdDataSpecialization ==
                          ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                              Filter1x1Stride1Pad0);

            const auto out_gemm_raw_grid_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Do_ * Ho_ * Wo_, K_),
                                             make_tuple(NStrideTensorA_, I1, KStrideTensorA_));

            return transform_tensor_descriptor(
                out_gemm_raw_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Do_ * Ho_ * Wo_)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            throw std::runtime_error("wrong! unsupported layout: " + ALayout::name());
        }
    }

    __host__ __device__ auto MakeWeiGridDesc() const
    {

        if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKYXC>)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(K_, Y_, X_, C_));
        }
        else if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKZYXC>)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(K_, Z_, Y_, X_, C_));
        }
        else
        {
            throw std::runtime_error("wrong! unsupported layout: " + BLayout::name());
        }
    }

    __host__ __device__ auto MakeInGridDesc() const
    {

        if constexpr(is_same_v<CLayout, tensor_layout::convolution::GNHWC> ||
                     is_same_v<CLayout, tensor_layout::convolution::NHWGC> ||
                     is_same_v<CLayout, tensor_layout::convolution::G_NHW_C>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, C_),
                make_tuple(NStrideTensorC_, HiStride_, WiStride_, CStrideTensorC_));
        }
        else if constexpr(is_same_v<CLayout, tensor_layout::convolution::GNDHWC> ||
                          is_same_v<CLayout, tensor_layout::convolution::NDHWGC>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(N_, Di_, Hi_, Wi_, C_),
                make_tuple(NStrideTensorC_, DiStride_, HiStride_, WiStride_, CStrideTensorC_));
        }
        else
        {
            throw std::runtime_error("wrong! unsupported layout: " + CLayout::name());
        }
    }

    template <
        typename ALayout_                   = ALayout,
        typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                    (is_same_v<ALayout_, tensor_layout::convolution::GNHWK> ||
                                     is_same_v<ALayout_, tensor_layout::convolution::GNDHWK> ||
                                     is_same_v<ALayout_, tensor_layout::convolution::NHWGK> ||
                                     is_same_v<ALayout_, tensor_layout::convolution::NDHWGK> ||
                                     is_same_v<ALayout_, tensor_layout::convolution::NGKHW> ||
                                     is_same_v<ALayout_, tensor_layout::convolution::NGKDHW>),
                                bool>::type = false>
    __host__ __device__ auto MakeADescriptor_AK0_M_AK1() const
    {
        // n_do_ho_wo_k for 3d or n_ho_wo_k for 2d
        const auto out_grid_desc = MakeOutGridDesc();

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            const index_t K0PerBlock = GemmKPerBlock / AK1;
            const index_t AK0 =
                math::integer_divide_ceil(K_, AK1 * K0PerBlock * batch_k_) * K0PerBlock;

            // A: output tensor
            const auto out_gemmak0_gemmmraw_gemmak1_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_pass_through_transform(N_ * Do_ * Ho_ * Wo_),
                           make_unmerge_transform(make_tuple(AK0 * batch_k_, AK1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            const auto out_gemmak0_gemmm_gemmak1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    out_gemmak0_gemmmraw_gemmak1_grid_desc,
                    make_tuple(AK0 * batch_k_, GemmMPerBlock, AK1),
                    Sequence<false, DoPadGemmM, false>{});

            return out_gemmak0_gemmm_gemmak1_grid_desc;
        }
        else
        {
            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IDTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadD_ - ConvDilationD_ * (ZTilde_ - I1)), ConvStrideD_);
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH_ - ConvDilationH_ * (YTilde_ - I1)), ConvStrideH_);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW_ - ConvDilationW_ * (XTilde_ - I1)), ConvStrideW_);

            const auto IDTildeSliceEnd = math::min(
                DTilde_, math::integer_divide_ceil(InLeftPadD_ + Di_ - I1, ConvStrideD_) + I1);
            const auto IHTildeSliceEnd = math::min(
                HTilde_, math::integer_divide_ceil(InLeftPadH_ + Hi_ - I1, ConvStrideH_) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde_, math::integer_divide_ceil(InLeftPadW_ + Wi_ - I1, ConvStrideW_) + I1);

            const auto DTildeSlice = IDTildeSliceEnd - IDTildeSliceBegin;
            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // GemmK is different for each GEMM
            const auto ZDotSlice = math::integer_divide_ceil(Z_ - IdxZTilde_, ZTilde_);
            const auto YDotSlice = math::integer_divide_ceil(Y_ - IdxYTilde_, YTilde_);
            const auto XDotSlice = math::integer_divide_ceil(X_ - IdxXTilde_, XTilde_);

            if constexpr(NDimSpatial == 2)
            {
                const index_t K0PerBlock = GemmKPerBlock / AK1;
                const index_t AK0        = math::integer_divide_ceil(YDotSlice * XDotSlice * K_,
                                                              AK1 * K0PerBlock * batch_k_) *
                                    K0PerBlock;

#if CK_USE_CUSTOM_TENSOR_TRANSFORM_FOR_BWD_DATA_OUT == 0
                // A: output tensor
                const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Ho_, I0, I0),
                               make_pad_transform(Wo_, I0, I0),
                               make_pass_through_transform(K_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto out_n_ydot_htilde_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                    out_n_hop_wop_k_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N_),
                        make_embed_transform(make_tuple(YDot_, HTilde_),
                                             make_tuple(-ConvDilationH_ / GcdStrideDilationH_, I1)),
                        make_embed_transform(make_tuple(XDot_, WTilde_),
                                             make_tuple(-ConvDilationW_ / GcdStrideDilationW_, I1)),
                        make_pass_through_transform(K_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc =
                    transform_tensor_descriptor(
                        out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
                        make_tuple(make_pass_through_transform(N_),
                                   make_slice_transform(YDot_, I0, YDotSlice),
                                   make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                                   make_slice_transform(XDot_, I0, XDotSlice),
                                   make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                                   make_pass_through_transform(K_)),
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

                const auto out_gemmk_gemmmraw_grid_desc = transform_tensor_descriptor(
                    out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K_)),
                               make_merge_transform(make_tuple(N_, HTildeSlice, WTildeSlice))),
                    make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto out_gemmk_gemmm_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        out_gemmk_gemmmraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmMPerBlock),
                        Sequence<true, DoPadGemmM>{});

                const auto out_gemmak0_gemmm_gemmak1_grid_desc = transform_tensor_descriptor(
                    out_gemmk_gemmm_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(AK0 * batch_k_, AK1)),
                               make_pass_through_transform(
                                   out_gemmk_gemmm_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
                return out_gemmak0_gemmm_gemmak1_grid_desc;
#else
                const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Ho_, I0, I0),
                               make_pad_transform(Wo_, I0, I0),
                               make_pass_through_transform(K_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto out_n_hop_wop_k_grid_desc_final = transform_tensor_descriptor(
                    out_n_hop_wop_k_grid_desc,
                    make_tuple(make_conv_bwd_data_out_transform(N_,
                                                                Ho_,
                                                                Wo_,
                                                                K_,
                                                                YDot_,
                                                                XDot_,
                                                                HTilde_,
                                                                WTilde_,
                                                                ConvDilationH_,
                                                                ConvDilationW_,
                                                                HTildeSlice,
                                                                WTildeSlice,
                                                                YDotSlice,
                                                                XDotSlice,
                                                                IHTildeSliceBegin,
                                                                IWTildeSliceBegin,
                                                                GcdStrideDilationH_,
                                                                GcdStrideDilationW_,
                                                                AK0 * batch_k_,
                                                                AK1,
                                                                GemmMPerBlock,
                                                                GemmKPerBlock)),
                    make_tuple(Sequence<0, 1, 2, 3>{}),
                    make_tuple(Sequence<0, 1, 2>{}));

                return out_n_hop_wop_k_grid_desc_final;
#endif
            }
            else if constexpr(NDimSpatial == 3)
            {
                // A: output tensor
                const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Do_, I0, I0),
                               make_pad_transform(Ho_, I0, I0),
                               make_pad_transform(Wo_, I0, I0),
                               make_pass_through_transform(K_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto out_n_zdot_dtilde_ydot_htilde_xdot_wtilde_k_grid_desc =
                    transform_tensor_descriptor(
                        out_n_hop_wop_k_grid_desc,
                        make_tuple(make_pass_through_transform(N_),
                                   make_embed_transform(
                                       make_tuple(ZDot_, DTilde_),
                                       make_tuple(-ConvDilationD_ / GcdStrideDilationD_, I1)),
                                   make_embed_transform(
                                       make_tuple(YDot_, HTilde_),
                                       make_tuple(-ConvDilationH_ / GcdStrideDilationH_, I1)),
                                   make_embed_transform(
                                       make_tuple(XDot_, WTilde_),
                                       make_tuple(-ConvDilationW_ / GcdStrideDilationW_, I1)),
                                   make_pass_through_transform(K_)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1, 2>{},
                                   Sequence<3, 4>{},
                                   Sequence<5, 6>{},
                                   Sequence<7>{}));

                const auto
                    out_n_zdotslice_dtildeslice_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc =
                        transform_tensor_descriptor(
                            out_n_zdot_dtilde_ydot_htilde_xdot_wtilde_k_grid_desc,
                            make_tuple(
                                make_pass_through_transform(N_),
                                make_slice_transform(ZDot_, I0, ZDotSlice),
                                make_slice_transform(DTilde_, IDTildeSliceBegin, DTildeSlice),
                                make_slice_transform(YDot_, I0, YDotSlice),
                                make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                                make_slice_transform(XDot_, I0, XDotSlice),
                                make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                                make_pass_through_transform(K_)),
                            make_tuple(Sequence<0>{},
                                       Sequence<1>{},
                                       Sequence<2>{},
                                       Sequence<3>{},
                                       Sequence<4>{},
                                       Sequence<5>{},
                                       Sequence<6>{},
                                       Sequence<7>{}),
                            make_tuple(Sequence<0>{},
                                       Sequence<1>{},
                                       Sequence<2>{},
                                       Sequence<3>{},
                                       Sequence<4>{},
                                       Sequence<5>{},
                                       Sequence<6>{},
                                       Sequence<7>{}));

                const auto out_gemmk_gemmmraw_grid_desc = transform_tensor_descriptor(
                    out_n_zdotslice_dtildeslice_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K_)),
                        make_merge_transform(
                            make_tuple(N_, DTildeSlice, HTildeSlice, WTildeSlice))),
                    make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto out_gemmk_gemmm_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        out_gemmk_gemmmraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmMPerBlock),
                        Sequence<true, DoPadGemmM>{});

                const index_t K0PerBlock = GemmKPerBlock / AK1;
                const index_t AK0 =
                    math::integer_divide_ceil(out_gemmk_gemmm_padded_grid_desc.GetLength(I0),
                                              AK1 * K0PerBlock * batch_k_) *
                    K0PerBlock;

                const auto out_gemmak0_gemmm_gemmak1_grid_desc = transform_tensor_descriptor(
                    out_gemmk_gemmm_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(AK0 * batch_k_, AK1)),
                               make_pass_through_transform(
                                   out_gemmk_gemmm_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return out_gemmak0_gemmm_gemmak1_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
    }

    template <
        typename BLayout_                   = BLayout,
        typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                    (is_same_v<BLayout_, tensor_layout::convolution::GKYXC> ||
                                     is_same_v<BLayout_, tensor_layout::convolution::GKZYXC> ||
                                     is_same_v<BLayout_, tensor_layout::convolution::GKCYX> ||
                                     is_same_v<BLayout_, tensor_layout::convolution::GKCZYX>),
                                bool>::type = false>
    __host__ __device__ auto MakeBDescriptor_BK0_N_BK1() const
    {

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            const index_t K0PerBlock = GemmKPerBlock / BK1;
            const index_t BK0 =
                math::integer_divide_ceil(K_, BK1 * K0PerBlock * batch_k_) * K0PerBlock;

            // B: weight tensor
            const auto wei_gemmbk0_gemmnraw_gemmbk1_grid_desc = transform_tensor_descriptor(
                make_naive_tensor_descriptor_packed(make_tuple(K_, C_)),
                make_tuple(make_unmerge_transform(make_tuple(BK0 * batch_k_, BK1)),
                           make_pass_through_transform(C_)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
            make_naive_tensor_descriptor(make_tuple(N_ * Do_ * Ho_ * Wo_, C_), make_tuple(I0, I1));

            const auto wei_gemmbk0_gemmn_gemmbk1_grid_desc =
                ck::tensor_operation::device::PadTensorDescriptor(
                    wei_gemmbk0_gemmnraw_gemmbk1_grid_desc,
                    make_tuple(BK0 * batch_k_, GemmNPerBlock, BK1),
                    Sequence<false, DoPadGemmN, false>{});

            return wei_gemmbk0_gemmn_gemmbk1_grid_desc;
        }
        else
        {
            // assume packed
            // k_y_x_c for 2d or k_z_y_x_c for 3d
            static_assert(is_same_v<BLayout_, tensor_layout::convolution::GKYXC> ||
                          is_same_v<BLayout_, tensor_layout::convolution::GKZYXC>);
            const auto wei_grid_desc = MakeWeiGridDesc();

            // GemmK is different for each GEMM
            const auto ZDotSlice = math::integer_divide_ceil(Z_ - IdxZTilde_, ZTilde_);
            const auto YDotSlice = math::integer_divide_ceil(Y_ - IdxYTilde_, YTilde_);
            const auto XDotSlice = math::integer_divide_ceil(X_ - IdxXTilde_, XTilde_);

            // B weight tensor
            if constexpr(NDimSpatial == 2)
            {
                const auto wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
                    wei_grid_desc,
                    make_tuple(
                        make_pass_through_transform(K_),
                        make_embed_transform(make_tuple(YDot_, YTilde_),
                                             make_tuple(ConvStrideH_ / GcdStrideDilationH_, I1)),
                        make_embed_transform(make_tuple(XDot_, XTilde_),
                                             make_tuple(ConvStrideW_ / GcdStrideDilationW_, I1)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto wei_k_ydotslice_xdotslice_c_grid_desc = transform_tensor_descriptor(
                    wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                    make_tuple(make_pass_through_transform(K_),
                               make_slice_transform(YDot_, I0, YDotSlice),
                               make_slice_transform(XDot_, I0, XDotSlice),
                               make_freeze_transform(IdxYTilde_),
                               make_freeze_transform(IdxXTilde_),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<3>{},
                               Sequence<2>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<>{},
                               Sequence<>{},
                               Sequence<3>{}));

                const auto wei_gemmk_gemmnraw_grid_desc = transform_tensor_descriptor(
                    wei_k_ydotslice_xdotslice_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<1, 2, 0>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto wei_gemmk_gemmn_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        wei_gemmk_gemmnraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmNPerBlock),
                        Sequence<true, DoPadGemmN>{});

                const index_t K0PerBlock = GemmKPerBlock / BK1;
                const index_t BK0 =
                    math::integer_divide_ceil(wei_gemmk_gemmn_padded_grid_desc.GetLength(I0),
                                              BK1 * K0PerBlock * batch_k_) *
                    K0PerBlock;

                const auto wei_gemmbk0_gemmn_gemmbk1_grid_desc = transform_tensor_descriptor(
                    wei_gemmk_gemmn_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(BK0 * batch_k_, BK1)),
                               make_pass_through_transform(
                                   wei_gemmk_gemmn_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return wei_gemmbk0_gemmn_gemmbk1_grid_desc;
            }
            else if constexpr(NDimSpatial == 3)
            {
                const auto wei_k_zdot_ztilde_ydot_ytilde_xdot_xtilde_c_grid_desc =
                    transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_pass_through_transform(K_),
                                   make_embed_transform(
                                       make_tuple(ZDot_, ZTilde_),
                                       make_tuple(ConvStrideD_ / GcdStrideDilationD_, I1)),
                                   make_embed_transform(
                                       make_tuple(YDot_, YTilde_),
                                       make_tuple(ConvStrideH_ / GcdStrideDilationH_, I1)),
                                   make_embed_transform(
                                       make_tuple(XDot_, XTilde_),
                                       make_tuple(ConvStrideW_ / GcdStrideDilationW_, I1)),
                                   make_pass_through_transform(C_)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1, 2>{},
                                   Sequence<3, 4>{},
                                   Sequence<5, 6>{},
                                   Sequence<7>{}));

                const auto wei_gemmk_zdotslice_ydotslice_xdotslice_c_grid_desc =
                    transform_tensor_descriptor(
                        wei_k_zdot_ztilde_ydot_ytilde_xdot_xtilde_c_grid_desc,
                        make_tuple(make_pass_through_transform(K_),
                                   make_slice_transform(ZDot_, I0, ZDotSlice),
                                   make_slice_transform(YDot_, I0, YDotSlice),
                                   make_slice_transform(XDot_, I0, XDotSlice),
                                   make_freeze_transform(IdxZTilde_),
                                   make_freeze_transform(IdxYTilde_),
                                   make_freeze_transform(IdxXTilde_),
                                   make_pass_through_transform(C_)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<3>{},
                                   Sequence<5>{},
                                   Sequence<2>{},
                                   Sequence<4>{},
                                   Sequence<6>{},
                                   Sequence<7>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<>{},
                                   Sequence<>{},
                                   Sequence<>{},
                                   Sequence<4>{}));

                const auto wei_gemmk_gemmnraw_grid_desc = transform_tensor_descriptor(
                    wei_gemmk_zdotslice_ydotslice_xdotslice_c_grid_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K_)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<1, 2, 3, 0>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto wei_gemmk_gemmn_padded_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        wei_gemmk_gemmnraw_grid_desc,
                        make_tuple(GemmKPerBlock, GemmNPerBlock),
                        Sequence<true, DoPadGemmN>{});

                const index_t K0PerBlock = GemmKPerBlock / BK1;
                const index_t BK0 =
                    math::integer_divide_ceil(wei_gemmk_gemmn_padded_grid_desc.GetLength(I0),
                                              BK1 * K0PerBlock * batch_k_) *
                    K0PerBlock;

                const auto wei_gemmbk0_gemm_gemmbk1_grid_desc = transform_tensor_descriptor(
                    wei_gemmk_gemmn_padded_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(BK0 * batch_k_, BK1)),
                               make_pass_through_transform(
                                   wei_gemmk_gemmn_padded_grid_desc.GetLength(I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return wei_gemmbk0_gemm_gemmbk1_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
    }

    template <
        typename CLayout_                   = CLayout,
        typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                    (is_same_v<CLayout_, tensor_layout::convolution::GNHWC> ||
                                     is_same_v<CLayout_, tensor_layout::convolution::GNDHWC> ||
                                     is_same_v<CLayout_, tensor_layout::convolution::NHWGC> ||
                                     is_same_v<CLayout_, tensor_layout::convolution::NDHWGC> ||
                                     is_same_v<CLayout_, tensor_layout::convolution::G_NHW_C>),
                                bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        static_assert(CTranspose == false);
        // assume strided
        // n_hi_wi_c for 2d n_di_hi_wi_c for 3d
        const auto in_grid_desc = MakeInGridDesc();

        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            // C: input tensor
            if constexpr(NDimSpatial == 2)
            {
                const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N_),
                        make_embed_transform(make_tuple(I1, Ho_), make_tuple(I1, ConvStrideH_)),
                        make_embed_transform(make_tuple(I1, Wo_), make_tuple(I1, ConvStrideW_)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_grid_desc,
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<1>{}, Sequence<3>{}, Sequence<0, 2, 4>{}, Sequence<5>{}),
                    make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});

                return in_gemmm_gemmn_grid_desc;
            }
            else if constexpr(NDimSpatial == 3)
            {

                // C: input tensor
                const auto in_n_x_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(
                        make_pass_through_transform(N_),
                        make_embed_transform(make_tuple(I1, Do_), make_tuple(I1, ConvStrideD_)),
                        make_embed_transform(make_tuple(I1, Ho_), make_tuple(I1, ConvStrideH_)),
                        make_embed_transform(make_tuple(I1, Wo_), make_tuple(I1, ConvStrideW_)),
                        make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_x_do_y_ho_x_wo_c_grid_desc,
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<1>{},
                               Sequence<3>{},
                               Sequence<5>{},
                               Sequence<0, 2, 4, 6>{},
                               Sequence<7>{}),
                    make_tuple(
                        Sequence<>{}, Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});

                return in_gemmm_gemmn_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
        else
        {
            // only work on DTilde, HTilde and WTilde that contribute to
            // non-padding area of input tensor
            const auto IDTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadD_ - ConvDilationD_ * (ZTilde_ - I1)), ConvStrideD_);
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH_ - ConvDilationH_ * (YTilde_ - I1)), ConvStrideH_);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW_ - ConvDilationW_ * (XTilde_ - I1)), ConvStrideW_);

            const auto IDTildeSliceEnd = math::min(
                DTilde_, math::integer_divide_ceil(InLeftPadD_ + Di_ - I1, ConvStrideD_) + I1);
            const auto IHTildeSliceEnd = math::min(
                HTilde_, math::integer_divide_ceil(InLeftPadH_ + Hi_ - I1, ConvStrideH_) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde_, math::integer_divide_ceil(InLeftPadW_ + Wi_ - I1, ConvStrideW_) + I1);

            const auto DTildeSlice = IDTildeSliceEnd - IDTildeSliceBegin;
            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // C: input tensor
            if constexpr(NDimSpatial == 2)
            {
                const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc =
                    transform_tensor_descriptor(
                        in_n_hip_wip_c_grid_desc,
                        make_tuple(make_pass_through_transform(N_),
                                   make_embed_transform(make_tuple(YTilde_, HTilde_),
                                                        make_tuple(ConvDilationH_, ConvStrideH_)),
                                   make_embed_transform(make_tuple(XTilde_, WTilde_),
                                                        make_tuple(ConvDilationW_, ConvStrideW_)),
                                   make_pass_through_transform(C_)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                        make_tuple(
                            Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                const auto in_n_htildeslice_wtildeslice_c_grid_desc = transform_tensor_descriptor(
                    in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_freeze_transform(IdxYTilde_),
                               make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                               make_freeze_transform(IdxXTilde_),
                               make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<>{},
                               Sequence<1>{},
                               Sequence<>{},
                               Sequence<2>{},
                               Sequence<3>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_htildeslice_wtildeslice_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, HTildeSlice, WTildeSlice)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});

                return in_gemmm_gemmn_grid_desc;
            }
            else if(NDimSpatial == 3)
            {
                const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_ztilde_dtilde_ytilde_htilde_xtilde_wtilde_c_grid_desc =
                    transform_tensor_descriptor(
                        in_n_dip_hip_wip_c_grid_desc,
                        make_tuple(make_pass_through_transform(N_),
                                   make_embed_transform(make_tuple(ZTilde_, DTilde_),
                                                        make_tuple(ConvDilationD_, ConvStrideD_)),
                                   make_embed_transform(make_tuple(YTilde_, HTilde_),
                                                        make_tuple(ConvDilationH_, ConvStrideH_)),
                                   make_embed_transform(make_tuple(XTilde_, WTilde_),
                                                        make_tuple(ConvDilationW_, ConvStrideW_)),
                                   make_pass_through_transform(C_)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<1, 2>{},
                                   Sequence<3, 4>{},
                                   Sequence<5, 6>{},
                                   Sequence<7>{}));

                const auto in_n_dtildeslice_htildeslice_wtildeslice_c_grid_desc =
                    transform_tensor_descriptor(
                        in_n_ztilde_dtilde_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                        make_tuple(make_pass_through_transform(N_),
                                   make_freeze_transform(IdxZTilde_),
                                   make_slice_transform(DTilde_, IDTildeSliceBegin, DTildeSlice),
                                   make_freeze_transform(IdxYTilde_),
                                   make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                                   make_freeze_transform(IdxXTilde_),
                                   make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                                   make_pass_through_transform(C_)),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{},
                                   Sequence<6>{},
                                   Sequence<7>{}),
                        make_tuple(Sequence<0>{},
                                   Sequence<>{},
                                   Sequence<1>{},
                                   Sequence<>{},
                                   Sequence<2>{},
                                   Sequence<>{},
                                   Sequence<3>{},
                                   Sequence<4>{}));

                const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                    in_n_dtildeslice_htildeslice_wtildeslice_c_grid_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, DTildeSlice, HTildeSlice, WTildeSlice)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto in_gemmm_gemmn_grid_desc =
                    ck::tensor_operation::device::PadTensorDescriptor(
                        in_gemmmraw_gemmnraw_grid_desc,
                        make_tuple(GemmMPerBlock, GemmNPerBlock),
                        Sequence<DoPadGemmM, DoPadGemmN>{});
                return in_gemmm_gemmn_grid_desc;
            }
            else
            {
                throw std::runtime_error("wrong! only implemented for 2D and 3D now");
            }
        }
    }

    template <typename CLayout_                   = CLayout,
              typename std::enable_if<(NDimSpatial == 2 || NDimSpatial == 3) &&
                                          (is_same_v<CLayout_, tensor_layout::convolution::NGCHW> ||
                                           is_same_v<CLayout_, tensor_layout::convolution::NGCDHW>),
                                      bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        const auto in_grid_desc = make_naive_tensor_descriptor(
            make_tuple(N_, C_, Di_ * Hi_ * Wi_), make_tuple(NStrideTensorC_, CStrideTensorC_, I1));

        static_assert(ConvBwdDataSpecialization ==
                      ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                          Filter1x1Stride1Pad0);

        if constexpr(CTranspose)
        {
            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(C_),
                           make_merge_transform(make_tuple(N_, Di_ * Hi_ * Wi_))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
            return ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmNPerBlock, GemmMPerBlock),
                Sequence<DoPadGemmN, DoPadGemmM>{});
        }
        else
        {
            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Di_ * Hi_ * Wi_)),
                           make_pass_through_transform(C_)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmMPerBlock, GemmNPerBlock),
                Sequence<DoPadGemmM, DoPadGemmN>{});
        }
    }
    // for input bias
    template <typename CLayout_                   = CLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          (is_same_v<CLayout_, tensor_layout::convolution::GC> ||
                                           is_same_v<CLayout_, tensor_layout::convolution::G_C>),
                                      bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        if constexpr(ConvBwdDataSpecialization ==
                     ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::
                         Filter1x1Stride1Pad0)
        {
            if constexpr(CTranspose)
            {
                const auto in_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                    make_tuple(C_, N_ * Ho_ * Wo_), make_tuple(I1, I0));

                return in_gemmm_gemmn_grid_desc;
            }
            else
            {
                const auto in_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                    make_tuple(N_ * Ho_ * Wo_, C_), make_tuple(I0, I1));

                return in_gemmm_gemmn_grid_desc;
            }
        }
        else
        {
            static_assert(CTranspose == false);
            // only work on HTilde and WTilde that contribute to non-padding area of input
            // tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH_ - ConvDilationH_ * (YTilde_ - I1)), ConvStrideH_);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW_ - ConvDilationW_ * (XTilde_ - I1)), ConvStrideW_);

            const auto IHTildeSliceEnd = math::min(
                HTilde_, math::integer_divide_ceil(InLeftPadH_ + Hi_ - I1, ConvStrideH_) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde_, math::integer_divide_ceil(InLeftPadW_ + Wi_ - I1, ConvStrideW_) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            // bias tensor
            const auto in_gemmmraw_gemmnraw_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N_ * HTildeSlice * WTildeSlice, C_), make_tuple(I0, I1));

            const auto in_gemmm_gemmn_grid_desc = ck::tensor_operation::device::PadTensorDescriptor(
                in_gemmmraw_gemmnraw_grid_desc,
                make_tuple(GemmMPerBlock, GemmNPerBlock),
                Sequence<DoPadGemmM, DoPadGemmN>{});

            return in_gemmm_gemmn_grid_desc;
        }
    }

    IndexType N_;
    IndexType Di_, Hi_, Wi_;
    IndexType Do_, Ho_, Wo_;
    IndexType Z_, Y_, X_;
    IndexType K_, C_;
    IndexType DiStride_, HiStride_, WiStride_;
    IndexType DoStride_, HoStride_, WoStride_;
    IndexType CStrideTensorB_, CStrideTensorC_, KStrideTensorA_, KStrideTensorB_;
    IndexType NStrideTensorA_, NStrideTensorC_;
    IndexType ConvStrideD_, ConvStrideH_, ConvStrideW_;
    IndexType ConvDilationD_, ConvDilationH_, ConvDilationW_;
    IndexType InLeftPadD_, InLeftPadH_, InLeftPadW_;
    IndexType InRightPadD_, InRightPadH_, InRightPadW_;
    IndexType IdxZTilde_, IdxYTilde_, IdxXTilde_;
    IndexType GcdStrideDilationD_, GcdStrideDilationH_, GcdStrideDilationW_;
    IndexType ZTilde_, YTilde_, XTilde_;
    IndexType DTilde_, HTilde_, WTilde_;
    IndexType ZDot_, YDot_, XDot_;
    index_t batch_k_;
};

} // namespace tensor_operation
} // namespace ck
