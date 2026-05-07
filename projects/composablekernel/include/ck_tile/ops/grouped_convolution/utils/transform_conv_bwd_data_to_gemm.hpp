// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"

namespace ck_tile {

template <index_t NDimSpatial,
          ConvolutionSpecialization ConvSpec,
          index_t VectorSizeA,
          index_t VectorSizeB,
          index_t VectorSizeC,
          bool SplitN              = false,
          typename ADataType       = float,
          typename CDataType       = float,
          index_t NumGroupsToMerge = 1,
          typename IndexType       = index_t>
struct TransformConvBwdDataToGemm
{
    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};
    static constexpr auto I5 = number<5>{};

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
    static IndexType GetSplitedNSize(const ConvDimsType& c_g_n_k_wos_lengths,
                                     const ConvDimsType& a_g_n_c_wis_lengths)
    {

        // Calculate strides internally assuming contiguous memory layout
        ConvDimsType c_g_n_k_wos_strides, a_g_n_c_wis_strides;
        const index_t num_dims = c_g_n_k_wos_strides.size();

        // Calculate strides for input tensor (innermost to outermost),
        // Don't include outermost dimension G since it's gemm batch.
        a_g_n_c_wis_strides[num_dims - 1] = 1;
        for(index_t i = num_dims - 2; i >= 1; i--)
        {
            a_g_n_c_wis_strides[i] = a_g_n_c_wis_strides[i + 1] * a_g_n_c_wis_lengths[i + 1];
        }

        // Calculate strides for output tensor,
        // Don't include outermost dimension G since it's gemm batch.
        c_g_n_k_wos_strides[num_dims - 1] = 1;
        for(index_t i = num_dims - 2; i >= 1; i--)
        {
            c_g_n_k_wos_strides[i] = c_g_n_k_wos_strides[i + 1] * c_g_n_k_wos_lengths[i + 1];
        }

        const long_index_t a_element_space_size =
            calculate_element_space_size_impl(a_g_n_c_wis_lengths, a_g_n_c_wis_strides, I1);
        const long_index_t c_element_space_size =
            calculate_element_space_size_impl(c_g_n_k_wos_lengths, c_g_n_k_wos_strides, I1);
        const long_index_t element_space_size = ck_tile::max(
            a_element_space_size * sizeof(ADataType), c_element_space_size * sizeof(CDataType));

        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        const IndexType N = c_g_n_k_wos_lengths[I1];

        if(element_space_size > TwoGB)
        {
            // Minimum divisor of N to not exceed 2GB
            const auto divisor = ck_tile::integer_divide_ceil(element_space_size, TwoGB);

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
    // Public getter methods for Split-N support
    CK_TILE_HOST constexpr IndexType GetN() const { return N_; }
    CK_TILE_HOST constexpr IndexType GetOriginalN() const { return original_N_; }

    CK_TILE_HOST constexpr TransformConvBwdDataToGemm() {}

    template <typename TransformConvBwdDataToGemmBase>
    CK_TILE_HOST
    TransformConvBwdDataToGemm(const TransformConvBwdDataToGemmBase& transform_conv_to_gemm_base)
        : G_{static_cast<IndexType>(transform_conv_to_gemm_base.G_)},
          N_{static_cast<IndexType>(transform_conv_to_gemm_base.N_)},
          original_N_{static_cast<IndexType>(transform_conv_to_gemm_base.original_N_)},
          Di_{static_cast<IndexType>(transform_conv_to_gemm_base.Di_)},
          Hi_{static_cast<IndexType>(transform_conv_to_gemm_base.Hi_)},
          Wi_{static_cast<IndexType>(transform_conv_to_gemm_base.Wi_)},
          Do_{static_cast<IndexType>(transform_conv_to_gemm_base.Do_)},
          Ho_{static_cast<IndexType>(transform_conv_to_gemm_base.Ho_)},
          Wo_{static_cast<IndexType>(transform_conv_to_gemm_base.Wo_)},
          Z_{static_cast<IndexType>(transform_conv_to_gemm_base.Z_)},
          Y_{static_cast<IndexType>(transform_conv_to_gemm_base.Y_)},
          X_{static_cast<IndexType>(transform_conv_to_gemm_base.X_)},
          K_{static_cast<IndexType>(transform_conv_to_gemm_base.K_)},
          C_{static_cast<IndexType>(transform_conv_to_gemm_base.C_)},
          ConvStrideD_{static_cast<IndexType>(transform_conv_to_gemm_base.ConvStrideD_)},
          ConvStrideH_{static_cast<IndexType>(transform_conv_to_gemm_base.ConvStrideH_)},
          ConvStrideW_{static_cast<IndexType>(transform_conv_to_gemm_base.ConvStrideW_)},
          ConvDilationD_{static_cast<IndexType>(transform_conv_to_gemm_base.ConvDilationD_)},
          ConvDilationH_{static_cast<IndexType>(transform_conv_to_gemm_base.ConvDilationH_)},
          ConvDilationW_{static_cast<IndexType>(transform_conv_to_gemm_base.ConvDilationW_)},
          InLeftPadD_{static_cast<IndexType>(transform_conv_to_gemm_base.InLeftPadD_)},
          InLeftPadH_{static_cast<IndexType>(transform_conv_to_gemm_base.InLeftPadH_)},
          InLeftPadW_{static_cast<IndexType>(transform_conv_to_gemm_base.InLeftPadW_)},
          InRightPadD_{static_cast<IndexType>(transform_conv_to_gemm_base.InRightPadD_)},
          InRightPadH_{static_cast<IndexType>(transform_conv_to_gemm_base.InRightPadH_)},
          InRightPadW_{static_cast<IndexType>(transform_conv_to_gemm_base.InRightPadW_)}
    {
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST TransformConvBwdDataToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                            const ConvDimsType& b_g_k_c_xs_lengths,
                                            const ConvDimsType& c_g_n_k_wos_lengths,
                                            const ConvSpatialDimsType& conv_filter_strides,
                                            const ConvSpatialDimsType& conv_filter_dilations,
                                            const ConvSpatialDimsType& input_left_pads,
                                            const ConvSpatialDimsType& input_right_pads,
                                            const ConvSpatialDimsType& tildes)
        : G_{a_g_n_c_wis_lengths[I0]},
          N_{a_g_n_c_wis_lengths[I1]},
          Di_{I1},
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
          IdxZTilde_{I1},
          IdxYTilde_{I1},
          IdxXTilde_{tildes[I0]}
    {

        // Store original N
        original_N_ = a_g_n_c_wis_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(c_g_n_k_wos_lengths, a_g_n_c_wis_lengths);
        }
        else
        {
            N_ = a_g_n_c_wis_lengths[I1];
        }

        GcdStrideDilationW_ = gcd(ConvStrideW_, ConvDilationW_);
        XTilde_             = ConvStrideW_ / GcdStrideDilationW_;
        WTilde_             = Wo_ + integer_divide_ceil(ConvDilationW_ * (X_ - I1), ConvStrideW_);
        XDot_               = integer_divide_ceil(X_, XTilde_);
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST TransformConvBwdDataToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                            const ConvDimsType& b_g_k_c_xs_lengths,
                                            const ConvDimsType& c_g_n_k_wos_lengths,
                                            const ConvSpatialDimsType& conv_filter_strides,
                                            const ConvSpatialDimsType& conv_filter_dilations,
                                            const ConvSpatialDimsType& input_left_pads,
                                            const ConvSpatialDimsType& input_right_pads,
                                            const ConvSpatialDimsType& tildes)
        : G_{a_g_n_c_wis_lengths[I0]},
          N_{a_g_n_c_wis_lengths[I1]},
          Di_{I1},
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
          IdxZTilde_{I1},
          IdxYTilde_{tildes[I0]},
          IdxXTilde_{tildes[I1]}
    {

        // Store original N
        original_N_ = a_g_n_c_wis_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(c_g_n_k_wos_lengths, a_g_n_c_wis_lengths);
        }
        else
        {
            N_ = a_g_n_c_wis_lengths[I1];
        }

        GcdStrideDilationW_ = gcd(ConvStrideW_, ConvDilationW_);
        GcdStrideDilationH_ = gcd(ConvStrideH_, ConvDilationH_);
        XTilde_             = ConvStrideW_ / GcdStrideDilationW_;
        YTilde_             = ConvStrideH_ / GcdStrideDilationH_;
        WTilde_             = Wo_ + integer_divide_ceil(ConvDilationW_ * (X_ - I1), ConvStrideW_);
        HTilde_             = Ho_ + integer_divide_ceil(ConvDilationH_ * (Y_ - I1), ConvStrideH_);
        XDot_               = integer_divide_ceil(X_, XTilde_);
        YDot_               = integer_divide_ceil(Y_, YTilde_);
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST TransformConvBwdDataToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                            const ConvDimsType& b_g_k_c_xs_lengths,
                                            const ConvDimsType& c_g_n_k_wos_lengths,
                                            const ConvSpatialDimsType& conv_filter_strides,
                                            const ConvSpatialDimsType& conv_filter_dilations,
                                            const ConvSpatialDimsType& input_left_pads,
                                            const ConvSpatialDimsType& input_right_pads,
                                            [[maybe_unused]] const ConvSpatialDimsType& tildes)
        : G_{a_g_n_c_wis_lengths[I0]},
          N_{a_g_n_c_wis_lengths[I1]},
          Di_{a_g_n_c_wis_lengths[I3]},
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
          IdxZTilde_{tildes[I0]},
          IdxYTilde_{tildes[I1]},
          IdxXTilde_{tildes[I2]}
    {

        // Store original N
        original_N_ = a_g_n_c_wis_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(c_g_n_k_wos_lengths, a_g_n_c_wis_lengths);
        }
        else
        {
            N_ = a_g_n_c_wis_lengths[I1];
        }

        GcdStrideDilationW_ = gcd(ConvStrideW_, ConvDilationW_);
        GcdStrideDilationH_ = gcd(ConvStrideH_, ConvDilationH_);
        GcdStrideDilationD_ = gcd(ConvStrideD_, ConvDilationD_);
        XTilde_             = ConvStrideW_ / GcdStrideDilationW_;
        YTilde_             = ConvStrideH_ / GcdStrideDilationH_;
        ZTilde_             = ConvStrideD_ / GcdStrideDilationD_;
        WTilde_             = Wo_ + integer_divide_ceil(ConvDilationW_ * (X_ - I1), ConvStrideW_);
        HTilde_             = Ho_ + integer_divide_ceil(ConvDilationH_ * (Y_ - I1), ConvStrideH_);
        DTilde_             = Do_ + integer_divide_ceil(ConvDilationD_ * (Z_ - I1), ConvStrideD_);
        XDot_               = integer_divide_ceil(X_, XTilde_);
        YDot_               = integer_divide_ceil(Y_, YTilde_);
        ZDot_               = integer_divide_ceil(Z_, ZTilde_);
    }

#if 0 // TODO: Enable these functionalities
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
        // Return left transform, right transformer, right offset to Input and right offset to
        // Output
        return ck_tile::make_tuple(conv_to_gemm_transformer_left,
                              conv_to_gemm_transformer_right,
                              a_grid_ptr_base + a_right_offset,
                              c_grid_ptr_base + c_right_offset);
    }
#endif

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto make_out_grid_desc() const
    {
        // NWGK
        const index_t NStride  = Wo_ * G_ * K_;
        const index_t WoStride = G_ * K_;
        constexpr auto KStride = I1;

        // TODO Add support for NumGroupsToMerge > 1

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(
                make_tuple(N_ * Wo_, K_), make_tuple(WoStride, KStride), number<VectorSizeA>{}, I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_, Wo_, K_),
                                                make_tuple(NStride, WoStride, KStride),
                                                number<VectorSizeA>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto make_wei_grid_desc() const
    {
        // GKXC
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_), make_tuple(C_, I1), number<VectorSizeB>{}, I1);
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(K_, X_, C_), make_tuple(X_ * C_, C_, I1), number<VectorSizeB>{}, I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto make_in_grid_desc() const
    {
        // NWGC
        const index_t NStride  = Wi_ * G_ * C_;
        const index_t WiStride = G_ * C_;
        constexpr auto CStride = I1;

        // TODO Add support for NumGroupsToMerge > 1
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(
                make_tuple(N_ * Wi_, C_), make_tuple(WiStride, CStride), number<VectorSizeC>{}, I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_, Wi_, C_),
                                                make_tuple(NStride, WiStride, CStride),
                                                number<VectorSizeC>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto make_out_grid_desc() const
    {
        // NHWGK
        const index_t NStride  = Ho_ * Wo_ * G_ * K_;
        const index_t HoStride = Wo_ * G_ * K_;
        const index_t WoStride = G_ * K_;
        constexpr auto KStride = I1;

        // TODO Add support for NumGroupsToMerge > 1

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Ho_ * Wo_, K_),
                                                make_tuple(WoStride, KStride),
                                                number<VectorSizeA>{},
                                                I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_, Ho_, Wo_, K_),
                                                make_tuple(NStride, HoStride, WoStride, KStride),
                                                number<VectorSizeA>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto make_in_grid_desc() const
    {
        // NHWGC
        const index_t NStride  = Hi_ * Wi_ * G_ * C_;
        const index_t HiStride = Wi_ * G_ * C_;
        const index_t WiStride = G_ * C_;
        constexpr auto CStride = I1;

        // TODO Add support for NumGroupsToMerge > 1
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Hi_ * Wi_, C_),
                                                make_tuple(WiStride, CStride),
                                                number<VectorSizeC>{},
                                                I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_, Hi_, Wi_, C_),
                                                make_tuple(NStride, HiStride, WiStride, CStride),
                                                number<VectorSizeC>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto make_wei_grid_desc() const
    {
        // GKYXC
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(
                make_tuple(K_, C_), make_tuple(C_, I1), number<VectorSizeB>{}, I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(K_, Y_, X_, C_),
                                                make_tuple(C_ * X_ * Y_, C_ * X_, C_, I1),
                                                number<VectorSizeB>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto make_out_grid_desc() const
    {
        // NDHWGK
        const index_t NStride  = Do_ * Ho_ * Wo_ * G_ * K_;
        const index_t DoStride = Ho_ * Wo_ * G_ * K_;
        const index_t HoStride = Wo_ * G_ * K_;
        const index_t WoStride = G_ * K_;
        constexpr auto KStride = I1;

        // TODO Add support for NumGroupsToMerge > 1
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Do_ * Ho_ * Wo_, K_),
                                                make_tuple(WoStride, KStride),
                                                number<VectorSizeA>{},
                                                I1);
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N_, Do_, Ho_, Wo_, K_),
                make_tuple(NStride, DoStride, HoStride, WoStride, KStride),
                number<VectorSizeA>{},
                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto make_in_grid_desc() const
    {
        const index_t NStride  = Di_ * Hi_ * Wi_ * G_ * C_;
        const index_t DiStride = Hi_ * Wi_ * G_ * C_;
        const index_t HiStride = Wi_ * G_ * C_;
        const index_t WiStride = G_ * C_;
        constexpr auto CStride = I1;

        // TODO Add support for NumGroupsToMerge > 1
        return make_naive_tensor_descriptor(
            make_tuple(N_, Di_, Hi_, Wi_, C_),
            make_tuple(NStride, DiStride, HiStride, WiStride, CStride),
            number<VectorSizeC>{},
            I1);
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto make_wei_grid_desc() const
    {
        // GKZYXC
        return make_naive_tensor_descriptor(
            make_tuple(K_, Z_, Y_, X_, C_),
            make_tuple(C_ * X_ * Y_ * Z_, C_ * X_ * Y_, C_ * X_, C_, I1),
            number<VectorSizeB>{},
            I1);
    }
    // TODO: implement ck_tile::tensor_layout::convolution that describe packed/strided dimemsion as
    // properties

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N([[maybe_unused]] const index_t GemmKBatch) const
    {
        // only work on HTilde and WTilde that contribute to non-padding area of input tensor
        const auto IWTildeSliceBegin = integer_divide_floor(
            max(I0, InLeftPadW_ - ConvDilationW_ * (XTilde_ - I1)), ConvStrideW_);

        const auto IWTildeSliceEnd =
            min(WTilde_, integer_divide_ceil(InLeftPadW_ + Wi_ - I1, ConvStrideW_) + I1);

        const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

        // GemmK is different for each GEMM
        const auto XDotSlice = integer_divide_ceil(X_ - IdxXTilde_, XTilde_);

        const auto out_grid_desc = make_out_grid_desc<NDimSpatial>();
        const auto in_grid_desc  = make_in_grid_desc<NDimSpatial>();
        const auto wei_grid_desc = make_wei_grid_desc<NDimSpatial>();

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_tuple(out_grid_desc, wei_grid_desc, in_grid_desc);
        }
        else
        {
            // A: output tensor comes in K_M
            const auto out_n_wop_k_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Wo_, I0, I0),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            const auto out_n_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                out_n_wop_k_grid_desc,
                make_tuple(
                    make_pass_through_transform(N_),
                    make_embed_transform(make_tuple(XDot_, WTilde_),
                                         make_tuple(-ConvDilationW_ / GcdStrideDilationW_, I1)),
                    make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            const auto out_n_xdotslice_wtildeslice_k_grid_desc = transform_tensor_descriptor(
                out_n_xdot_wtilde_k_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_slice_transform(XDot_, I0, XDotSlice),
                           make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            const auto out_gemmm_gemmkraw_grid_desc = transform_tensor_descriptor(
                out_n_xdotslice_wtildeslice_k_grid_desc,
                make_tuple(make_merge_transform(make_tuple(XDotSlice, K_)),
                           make_merge_transform(make_tuple(N_, WTildeSlice))),
                make_tuple(sequence<1, 3>{}, sequence<0, 2>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            // B: weight tensor comes in K_N
            const auto wei_k_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
                wei_grid_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_embed_transform(make_tuple(XDot_, XTilde_),
                                                make_tuple(ConvStrideW_ / GcdStrideDilationW_, I1)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            const auto wei_k_xdotslice_c_grid_desc = transform_tensor_descriptor(
                wei_k_xdot_xtilde_c_grid_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_slice_transform(XDot_, I0, XDotSlice),
                           make_freeze_transform(IdxXTilde_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<>{}, sequence<2>{}));

            const auto wei_gemmn_gemmkraw_grid_desc = transform_tensor_descriptor(
                wei_k_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(XDotSlice, K_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            // c: input
            const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            const auto in_n_xtilde_wtilde_c_grid_desc = transform_tensor_descriptor(
                in_n_wip_c_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(XTilde_, WTilde_),
                                                make_tuple(ConvDilationW_, ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            const auto in_n_wtildeslice_c_grid_desc = transform_tensor_descriptor(
                in_n_xtilde_wtilde_c_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_freeze_transform(IdxXTilde_),
                           make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<>{}, sequence<1>{}, sequence<2>{}));

            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_n_wtildeslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N_, WTildeSlice)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tuple(out_gemmm_gemmkraw_grid_desc,
                              wei_gemmn_gemmkraw_grid_desc,
                              in_gemmmraw_gemmnraw_grid_desc);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N([[maybe_unused]] const index_t GemmKBatch) const
    {
        // only work on HTilde and WTilde that contribute to non-padding area of input tensor
        const auto IHTildeSliceBegin = integer_divide_floor(
            max(I0, InLeftPadH_ - ConvDilationH_ * (YTilde_ - I1)), ConvStrideH_);
        const auto IWTildeSliceBegin = integer_divide_floor(
            max(I0, InLeftPadW_ - ConvDilationW_ * (XTilde_ - I1)), ConvStrideW_);

        const auto IHTildeSliceEnd =
            min(HTilde_, integer_divide_ceil(InLeftPadH_ + Hi_ - I1, ConvStrideH_) + I1);
        const auto IWTildeSliceEnd =
            min(WTilde_, integer_divide_ceil(InLeftPadW_ + Wi_ - I1, ConvStrideW_) + I1);

        const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
        const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

        // GemmK is different for each GEMM
        const auto YDotSlice = integer_divide_ceil(Y_ - IdxYTilde_, YTilde_);
        const auto XDotSlice = integer_divide_ceil(X_ - IdxXTilde_, XTilde_);

        const auto out_grid_desc = make_out_grid_desc<NDimSpatial>();
        const auto wei_grid_desc = make_wei_grid_desc<NDimSpatial>();
        const auto in_grid_desc  = make_in_grid_desc<NDimSpatial>();

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_tuple(out_grid_desc, wei_grid_desc, in_grid_desc);
        }
        else
        {
            // A: output tensor comes in K_M
            const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Ho_, I0, I0),
                           make_pad_transform(Wo_, I0, I0),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            const auto out_n_ydot_htilde_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                out_n_hop_wop_k_grid_desc,
                make_tuple(
                    make_pass_through_transform(N_),
                    make_embed_transform(make_tuple(YDot_, HTilde_),
                                         make_tuple(-ConvDilationH_ / GcdStrideDilationH_, I1)),
                    make_embed_transform(make_tuple(XDot_, WTilde_),
                                         make_tuple(-ConvDilationW_ / GcdStrideDilationW_, I1)),
                    make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

            const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc =
                transform_tensor_descriptor(
                    out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_slice_transform(YDot_, I0, YDotSlice),
                               make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                               make_slice_transform(XDot_, I0, XDotSlice),
                               make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                               make_pass_through_transform(K_)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}));

            const auto out_gemmm_gemmkraw_grid_desc = transform_tensor_descriptor(
                out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K_)),
                           make_merge_transform(make_tuple(N_, HTildeSlice, WTildeSlice))),
                make_tuple(sequence<1, 3, 5>{}, sequence<0, 2, 4>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            // B: weight tensor comes in K_N
            const auto wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
                wei_grid_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_embed_transform(make_tuple(YDot_, YTilde_),
                                                make_tuple(ConvStrideH_ / GcdStrideDilationH_, I1)),
                           make_embed_transform(make_tuple(XDot_, XTilde_),
                                                make_tuple(ConvStrideW_ / GcdStrideDilationW_, I1)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

            const auto wei_k_ydotslice_xdotslice_c_grid_desc =
                transform_tensor_descriptor(wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                                            make_tuple(make_pass_through_transform(K_),
                                                       make_slice_transform(YDot_, I0, YDotSlice),
                                                       make_slice_transform(XDot_, I0, XDotSlice),
                                                       make_freeze_transform(IdxYTilde_),
                                                       make_freeze_transform(IdxXTilde_),
                                                       make_pass_through_transform(C_)),
                                            make_tuple(sequence<0>{},
                                                       sequence<1>{},
                                                       sequence<3>{},
                                                       sequence<2>{},
                                                       sequence<4>{},
                                                       sequence<5>{}),
                                            make_tuple(sequence<0>{},
                                                       sequence<1>{},
                                                       sequence<2>{},
                                                       sequence<>{},
                                                       sequence<>{},
                                                       sequence<3>{}));

            const auto wei_gemmn_gemmkraw_grid_desc = transform_tensor_descriptor(
                wei_k_ydotslice_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<1, 2, 0>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            // c: input
            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

            const auto in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(YTilde_, HTilde_),
                                                make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(XTilde_, WTilde_),
                                                make_tuple(ConvDilationW_, ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

            const auto in_n_htildeslice_wtildeslice_c_grid_desc = transform_tensor_descriptor(
                in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_freeze_transform(IdxYTilde_),
                           make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                           make_freeze_transform(IdxXTilde_),
                           make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{},
                           sequence<1>{},
                           sequence<2>{},
                           sequence<3>{},
                           sequence<4>{},
                           sequence<5>{}),
                make_tuple(sequence<0>{},
                           sequence<>{},
                           sequence<1>{},
                           sequence<>{},
                           sequence<2>{},
                           sequence<3>{}));

            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_n_htildeslice_wtildeslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N_, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tuple(out_gemmm_gemmkraw_grid_desc,
                              wei_gemmn_gemmkraw_grid_desc,
                              in_gemmmraw_gemmnraw_grid_desc);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N([[maybe_unused]] const index_t GemmKBatch) const
    {
        // only work on DTilde, HTilde and WTilde that contribute to non-padding area of input
        // tensor
        const auto IDTildeSliceBegin = integer_divide_floor(
            max(I0, InLeftPadD_ - ConvDilationD_ * (ZTilde_ - I1)), ConvStrideD_);
        const auto IHTildeSliceBegin = integer_divide_floor(
            max(I0, InLeftPadH_ - ConvDilationH_ * (YTilde_ - I1)), ConvStrideH_);
        const auto IWTildeSliceBegin = integer_divide_floor(
            max(I0, InLeftPadW_ - ConvDilationW_ * (XTilde_ - I1)), ConvStrideW_);

        const auto IDTildeSliceEnd =
            min(DTilde_, integer_divide_ceil(InLeftPadD_ + Di_ - I1, ConvStrideD_) + I1);
        const auto IHTildeSliceEnd =
            min(HTilde_, integer_divide_ceil(InLeftPadH_ + Hi_ - I1, ConvStrideH_) + I1);
        const auto IWTildeSliceEnd =
            min(WTilde_, integer_divide_ceil(InLeftPadW_ + Wi_ - I1, ConvStrideW_) + I1);

        const auto DTildeSlice = IDTildeSliceEnd - IDTildeSliceBegin;
        const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
        const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

        // GemmK is different for each GEMM
        const auto ZDotSlice = integer_divide_ceil(Z_ - IdxZTilde_, ZTilde_);
        const auto YDotSlice = integer_divide_ceil(Y_ - IdxYTilde_, YTilde_);
        const auto XDotSlice = integer_divide_ceil(X_ - IdxXTilde_, XTilde_);

        const auto out_grid_desc = make_out_grid_desc<NDimSpatial>();
        const auto in_grid_desc  = make_in_grid_desc<NDimSpatial>();
        const auto wei_grid_desc = make_wei_grid_desc<NDimSpatial>();

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            return make_tuple(out_grid_desc, wei_grid_desc, in_grid_desc);
        }
        else
        {
            // A: output tensor comes in K_M
            const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Do_, I0, I0),
                           make_pad_transform(Ho_, I0, I0),
                           make_pad_transform(Wo_, I0, I0),
                           make_pass_through_transform(K_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

            const auto out_n_ydot_htilde_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                out_n_hop_wop_k_grid_desc,
                make_tuple(
                    make_pass_through_transform(N_),
                    make_embed_transform(make_tuple(ZDot_, DTilde_),
                                         make_tuple(-ConvDilationD_ / GcdStrideDilationD_, I1)),
                    make_embed_transform(make_tuple(YDot_, HTilde_),
                                         make_tuple(-ConvDilationH_ / GcdStrideDilationH_, I1)),
                    make_embed_transform(make_tuple(XDot_, WTilde_),
                                         make_tuple(-ConvDilationW_ / GcdStrideDilationW_, I1)),
                    make_pass_through_transform(K_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(sequence<0>{},
                           sequence<1, 2>{},
                           sequence<3, 4>{},
                           sequence<5, 6>{},
                           sequence<7>{}));

            const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc =
                transform_tensor_descriptor(
                    out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_slice_transform(ZDot_, I0, ZDotSlice),
                               make_slice_transform(DTilde_, IDTildeSliceBegin, DTildeSlice),
                               make_slice_transform(YDot_, I0, YDotSlice),
                               make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                               make_slice_transform(XDot_, I0, XDotSlice),
                               make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                               make_pass_through_transform(K_)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{},
                               sequence<6>{},
                               sequence<7>{}),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{},
                               sequence<6>{},
                               sequence<7>{}));

            const auto out_gemmm_gemmkraw_grid_desc = transform_tensor_descriptor(
                out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k_grid_desc,
                make_tuple(
                    make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K_)),
                    make_merge_transform(make_tuple(N_, DTildeSlice, HTildeSlice, WTildeSlice))),
                make_tuple(sequence<1, 3, 5, 7>{}, sequence<0, 2, 4, 6>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            // B: weight tensor comes in K_N
            const auto wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc = transform_tensor_descriptor(
                wei_grid_desc,
                make_tuple(make_pass_through_transform(K_),
                           make_embed_transform(make_tuple(ZDot_, ZTilde_),
                                                make_tuple(ConvStrideD_ / GcdStrideDilationD_, I1)),
                           make_embed_transform(make_tuple(YDot_, YTilde_),
                                                make_tuple(ConvStrideH_ / GcdStrideDilationH_, I1)),
                           make_embed_transform(make_tuple(XDot_, XTilde_),
                                                make_tuple(ConvStrideW_ / GcdStrideDilationW_, I1)),
                           make_pass_through_transform(C_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(sequence<0>{},
                           sequence<1, 2>{},
                           sequence<3, 4>{},
                           sequence<5, 6>{},
                           sequence<7>{}));

            const auto wei_k_ydotslice_xdotslice_c_grid_desc =
                transform_tensor_descriptor(wei_k_ydot_ytilde_xdot_xtilde_c_grid_desc,
                                            make_tuple(make_pass_through_transform(K_),
                                                       make_slice_transform(ZDot_, I0, ZDotSlice),
                                                       make_slice_transform(YDot_, I0, YDotSlice),
                                                       make_slice_transform(XDot_, I0, XDotSlice),
                                                       make_freeze_transform(IdxZTilde_),
                                                       make_freeze_transform(IdxYTilde_),
                                                       make_freeze_transform(IdxXTilde_),
                                                       make_pass_through_transform(C_)),
                                            make_tuple(sequence<0>{},
                                                       sequence<1>{},
                                                       sequence<3>{},
                                                       sequence<5>{},
                                                       sequence<2>{},
                                                       sequence<4>{},
                                                       sequence<6>{},
                                                       sequence<7>{}),
                                            make_tuple(sequence<0>{},
                                                       sequence<1>{},
                                                       sequence<2>{},
                                                       sequence<3>{},
                                                       sequence<>{},
                                                       sequence<>{},
                                                       sequence<>{},
                                                       sequence<4>{}));

            const auto wei_gemmn_gemmkraw_grid_desc = transform_tensor_descriptor(
                wei_k_ydotslice_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(ZDotSlice, YDotSlice, XDotSlice, K_)),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<1, 2, 3, 0>{}, sequence<4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            // c: input
            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                           make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                           make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                           make_pass_through_transform(C_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

            const auto in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_embed_transform(make_tuple(ZTilde_, DTilde_),
                                                make_tuple(ConvDilationD_, ConvStrideD_)),
                           make_embed_transform(make_tuple(YTilde_, HTilde_),
                                                make_tuple(ConvDilationH_, ConvStrideH_)),
                           make_embed_transform(make_tuple(XTilde_, WTilde_),
                                                make_tuple(ConvDilationW_, ConvStrideW_)),
                           make_pass_through_transform(C_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(sequence<0>{},
                           sequence<1, 2>{},
                           sequence<3, 4>{},
                           sequence<5, 6>{},
                           sequence<7>{}));

            const auto in_n_htildeslice_wtildeslice_c_grid_desc = transform_tensor_descriptor(
                in_n_ytilde_htilde_xtilde_wtilde_c_grid_desc,
                make_tuple(make_pass_through_transform(N_),
                           make_freeze_transform(IdxZTilde_),
                           make_slice_transform(DTilde_, IDTildeSliceBegin, DTildeSlice),
                           make_freeze_transform(IdxYTilde_),
                           make_slice_transform(HTilde_, IHTildeSliceBegin, HTildeSlice),
                           make_freeze_transform(IdxXTilde_),
                           make_slice_transform(WTilde_, IWTildeSliceBegin, WTildeSlice),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0>{},
                           sequence<1>{},
                           sequence<2>{},
                           sequence<3>{},
                           sequence<4>{},
                           sequence<5>{},
                           sequence<6>{},
                           sequence<7>{}),
                make_tuple(sequence<0>{},
                           sequence<>{},
                           sequence<1>{},
                           sequence<>{},
                           sequence<2>{},
                           sequence<>{},
                           sequence<3>{},
                           sequence<4>{}));

            const auto in_gemmmraw_gemmnraw_grid_desc = transform_tensor_descriptor(
                in_n_htildeslice_wtildeslice_c_grid_desc,
                make_tuple(
                    make_merge_transform(make_tuple(N_, DTildeSlice, HTildeSlice, WTildeSlice)),
                    make_pass_through_transform(C_)),
                make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tuple(out_gemmm_gemmkraw_grid_desc,
                              wei_gemmn_gemmkraw_grid_desc,
                              in_gemmmraw_gemmnraw_grid_desc);
        }
    }

    IndexType G_, N_, original_N_;
    IndexType Di_, Hi_, Wi_;
    IndexType Do_, Ho_, Wo_;
    IndexType Z_, Y_, X_;
    IndexType K_, C_;
    IndexType ConvStrideD_, ConvStrideH_, ConvStrideW_;
    IndexType ConvDilationD_, ConvDilationH_, ConvDilationW_;
    IndexType InLeftPadD_, InLeftPadH_, InLeftPadW_;
    IndexType InRightPadD_, InRightPadH_, InRightPadW_;
    IndexType IdxZTilde_, IdxYTilde_, IdxXTilde_;
    IndexType GcdStrideDilationD_, GcdStrideDilationH_, GcdStrideDilationW_;
    IndexType ZTilde_, YTilde_, XTilde_;
    IndexType DTilde_, HTilde_, WTilde_;
    IndexType ZDot_, YDot_, XDot_;
};

} // namespace ck_tile
