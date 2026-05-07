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
          index_t NumGroupsToMerge = 1,
          bool SplitN              = false,
          typename ADataType       = float,
          typename CDataType       = float,
          typename IndexType       = index_t>
struct TransformConvBwdWeightToGemm
{
    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};
    static constexpr auto I5 = number<5>{};
#if 0 // TODO: Enable these functionalities
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
        constexpr long_index_t TwoGB          = (long_index_t{1} << 31);

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
#endif

    public:
    CK_TILE_HOST constexpr TransformConvBwdWeightToGemm() {}

    template <typename TransformConvBwdWeightToGemmBase>
    CK_TILE_HOST TransformConvBwdWeightToGemm(
        const TransformConvBwdWeightToGemmBase& transform_conv_fwd_to_gemm_base)
        : G_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.G_)},
          N_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.N_)},
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
          InRightPadW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadW_)}
    {
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST TransformConvBwdWeightToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                              const ConvDimsType& b_g_k_c_xs_lengths,
                                              const ConvDimsType& c_g_n_k_wos_lengths,
                                              const ConvSpatialDimsType& conv_filter_strides,
                                              const ConvSpatialDimsType& conv_filter_dilations,
                                              const ConvSpatialDimsType& input_left_pads,
                                              const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
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
          InRightPadW_{input_right_pads[I0]}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);
#if 0 // TODO: Enable these functionalities
        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
#endif
        N_ = c_g_n_k_wos_lengths[I1];
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST TransformConvBwdWeightToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                              const ConvDimsType& b_g_k_c_xs_lengths,
                                              const ConvDimsType& c_g_n_k_wos_lengths,
                                              const ConvSpatialDimsType& conv_filter_strides,
                                              const ConvSpatialDimsType& conv_filter_dilations,
                                              const ConvSpatialDimsType& input_left_pads,
                                              const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
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
          InRightPadW_{input_right_pads[I1]}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);
#if 0 // TODO: Enable these functionalities
        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
#endif
        N_ = c_g_n_k_wos_lengths[I1];
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST TransformConvBwdWeightToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                              const ConvDimsType& b_g_k_c_xs_lengths,
                                              const ConvDimsType& c_g_n_k_wos_lengths,
                                              const ConvSpatialDimsType& conv_filter_strides,
                                              const ConvSpatialDimsType& conv_filter_dilations,
                                              const ConvSpatialDimsType& input_left_pads,
                                              const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
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
          InRightPadW_{input_right_pads[I2]}
    {
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);
#if 0 // TODO: Enable these functionalities
        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
#endif
        N_ = c_g_n_k_wos_lengths[I1];
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
        const index_t NDoHoWoStride = G_ * K_;
        constexpr auto KStride      = I1;

        if constexpr(NumGroupsToMerge > 1)
        {
            const index_t BatchStride = K_;
            return make_naive_tensor_descriptor(make_tuple(K_, NumGroupsToMerge, N_ * Wo_),
                                                make_tuple(KStride, BatchStride, NDoHoWoStride),
                                                number<VectorSizeA>{},
                                                I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(K_, N_ * Wo_),
                                                make_tuple(KStride, NDoHoWoStride),
                                                number<VectorSizeA>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto make_in_grid_desc() const
    {
        // NWGC
        const index_t NStride  = Wi_ * G_ * C_;
        const index_t WiStride = G_ * C_;
        constexpr auto CStride = I1;

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const index_t BatchStride = C_;
                return make_naive_tensor_descriptor(make_tuple(N_ * Wi_, NumGroupsToMerge, C_),
                                                    make_tuple(WiStride, BatchStride, CStride),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
            else
            {
                return make_naive_tensor_descriptor(make_tuple(N_ * Wi_, C_),
                                                    make_tuple(WiStride, CStride),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const index_t BatchStride = C_;
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStride, WiStride, BatchStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
            else
            {

                return make_naive_tensor_descriptor(make_tuple(N_, Wi_, C_),
                                                    make_tuple(NStride, WiStride, CStride),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto make_wei_grid_desc() const
    {
        // GKXC
        const index_t KStride   = X_ * C_;
        constexpr auto CXStride = I1;

        if constexpr(NumGroupsToMerge > 1)
        {
            const index_t XStride     = C_;
            const index_t BatchStride = K_ * X_ * C_;
            // Add NumGroupsToMerge for Batch+M dimension and, 1 as a placeholder
            // for Batch+N dimension
            const auto desc = make_naive_tensor_descriptor(
                make_tuple(NumGroupsToMerge, K_, X_, 1, C_),
                make_tuple(BatchStride, KStride, XStride, BatchStride, CXStride),
                number<VectorSizeC>{},
                I1);
            // Pad 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                desc,
                make_tuple(make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pass_through_transform(X_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1),
                           make_pass_through_transform(C_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));
            // We need only matrices from diagonal. Xor returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_),
                           make_pass_through_transform(X_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 3>{}, sequence<1>{}, sequence<2>{}, sequence<4>{}),
                make_tuple(sequence<0, 3>{}, sequence<1>{}, sequence<2>{}, sequence<4>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K_)),
                           make_merge_transform(make_tuple(X_, NumGroupsToMerge, C_))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(K_, X_ * C_), make_tuple(KStride, CXStride), number<VectorSizeC>{}, I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto make_out_grid_desc() const
    {
        // NHWGK
        const index_t NDoHoWoStride = G_ * K_;
        constexpr auto KStride      = I1;

        if constexpr(NumGroupsToMerge > 1)
        {
            const index_t BatchStride = K_;
            return make_naive_tensor_descriptor(
                make_tuple(N_ * Ho_ * Wo_, NumGroupsToMerge, K_), // K_Gm_M
                make_tuple(NDoHoWoStride, BatchStride, KStride),
                number<VectorSizeA>{},
                I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Ho_ * Wo_, K_), // K_M
                                                make_tuple(NDoHoWoStride, KStride),
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

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const index_t BatchStride = C_;
                return make_naive_tensor_descriptor(
                    make_tuple(N_ * Hi_ * Wi_, NumGroupsToMerge, C_), // K_Gm_N
                    make_tuple(WiStride, BatchStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
            else
            {
                return make_naive_tensor_descriptor(make_tuple(N_ * Hi_ * Wi_, C_), // K_N
                                                    make_tuple(WiStride, CStride),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const index_t BatchStride = C_;
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_), // K_Gm_N
                    make_tuple(NStride, HiStride, WiStride, BatchStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_), // K_N
                    make_tuple(NStride, HiStride, WiStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto make_wei_grid_desc() const
    {
        // GKYXC
        const index_t KStride  = Y_ * X_ * C_;
        constexpr auto CStride = I1;

        if constexpr(NumGroupsToMerge > 1)
        {
            const index_t YXStride    = C_;
            const index_t BatchStride = K_ * Y_ * X_ * C_;
            // Add NumGroupsToMerge for Batch+M dimension and, 1 as a placeholder
            // for Batch+N dimension
            const auto desc = make_naive_tensor_descriptor(
                make_tuple(NumGroupsToMerge, K_, Y_ * X_, 1, C_),
                make_tuple(BatchStride, KStride, YXStride, BatchStride, CStride),
                number<VectorSizeC>{},
                I1);
            // Pad 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                desc,
                make_tuple(make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pass_through_transform(Y_ * X_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1),
                           make_pass_through_transform(C_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));
            // We need only matrices from diagonal. Xor returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_),
                           make_pass_through_transform(Y_ * X_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 3>{}, sequence<1>{}, sequence<2>{}, sequence<4>{}),
                make_tuple(sequence<0, 3>{}, sequence<1>{}, sequence<2>{}, sequence<4>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K_)),
                           make_merge_transform(make_tuple(Y_ * X_, NumGroupsToMerge, C_))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(K_, Y_ * X_ * C_),
                                                make_tuple(KStride, CStride),
                                                number<VectorSizeC>{},
                                                I1);
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto make_out_grid_desc() const
    {
        // NDHWGK
        const index_t NDoHoWoStride = G_ * K_;
        constexpr auto KStride      = I1;

        if constexpr(NumGroupsToMerge > 1)
        {
            const auto BatchStride = K_;
            return make_naive_tensor_descriptor(
                make_tuple(N_ * Do_ * Ho_ * Wo_, NumGroupsToMerge, K_),
                make_tuple(NDoHoWoStride, BatchStride, KStride),
                number<VectorSizeA>{},
                I1);
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N_ * Do_ * Ho_ * Wo_, K_),
                                                make_tuple(NDoHoWoStride, KStride),
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

        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const index_t BatchStride = C_;
                return make_naive_tensor_descriptor(
                    make_tuple(N_ * Di_ * Hi_ * Wi_, NumGroupsToMerge, C_),
                    make_tuple(WiStride, BatchStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
            else
            {
                return make_naive_tensor_descriptor(make_tuple(N_ * Di_ * Hi_ * Wi_, C_),
                                                    make_tuple(WiStride, CStride),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const index_t BatchStride = C_;
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStride, DiStride, HiStride, WiStride, BatchStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStride, DiStride, HiStride, WiStride, CStride),
                    number<VectorSizeB>{},
                    I1);
            }
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto make_wei_grid_desc() const
    {
        // GKZYXC
        const index_t KStride  = Z_ * Y_ * X_ * C_;
        constexpr auto CStride = I1;

        if constexpr(NumGroupsToMerge > 1)
        {
            const index_t ZYXStride   = C_;
            const index_t BatchStride = K_ * Z_ * Y_ * X_ * C_;
            // Add NumGroupsToMerge for Batch+M dimension and, 1 as a placeholder
            // for Batch+N dimension
            const auto desc = make_naive_tensor_descriptor(
                make_tuple(NumGroupsToMerge, K_, Z_ * Y_ * X_, 1, C_),
                make_tuple(BatchStride, KStride, ZYXStride, BatchStride, CStride),
                number<VectorSizeC>{},
                I1);
            // Pad 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                desc,
                make_tuple(make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pass_through_transform(Z_ * Y_ * X_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1),
                           make_pass_through_transform(C_)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));
            // We need only matrices from diagonal. Xor returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_),
                           make_pass_through_transform(Z_ * Y_ * X_),
                           make_pass_through_transform(C_)),
                make_tuple(sequence<0, 3>{}, sequence<1>{}, sequence<2>{}, sequence<4>{}),
                make_tuple(sequence<0, 3>{}, sequence<1>{}, sequence<2>{}, sequence<4>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K_)),
                           make_merge_transform(make_tuple(Z_ * Y_ * X_, NumGroupsToMerge, C_))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(K_, Z_ * Y_ * X_ * C_),
                                                make_tuple(KStride, CStride),
                                                number<VectorSizeC>{},
                                                I1);
        }
    }

    // TODO: implement ck_tile::tensor_layout::convolution that describe packed/strided dimemsion as
    // properties

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N() const
    {
        const auto out_grid_desc = make_out_grid_desc<NDimSpatial>();
        const auto in_grid_desc  = make_in_grid_desc<NDimSpatial>();
        const auto wei_grid_desc = make_wei_grid_desc<NDimSpatial>();

        // B: input tensor comes in K_N
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const auto out_grid_merged_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto in_grid_merged_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, C_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return make_tuple(out_grid_merged_desc, in_grid_merged_desc, wei_grid_desc);
            }
            else
            {
                return make_tuple(out_grid_desc, in_grid_desc, wei_grid_desc);
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                // Output tensor transformation
                // [0,       1, 2] -> [0,           1]
                // [(N*Wo), Gm, K] -> [(N*Wo), (Gm*K)]
                const auto out_gemm_k_gemm_m_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                // Input tensor transformation, part 1.
                // [N, Wi, Gm, C] -> [N, (Wi + InLeftPadW + InRightPadW), Gm, C] = [N, Wip, Gm, C]
                const auto in_n_wip_gm_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                // Input tensor transformation, part 2.
                // [N, Wip, Gm, C] -> [N, X, Wo, Gm, C]
                const auto in_n_x_wo_gm_c_grid_desc = transform_tensor_descriptor(
                    in_n_wip_gm_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}, sequence<4>{}));

                // Input tensor transformation, part 3.
                // [0, 1,  2, 3,  4] -> [0,             1]
                // [N, X, Wo, Gm, C] -> [(N*Wo), (Gm*X*C)]
                const auto in_gemm_n_gemm_k_grid_desc = transform_tensor_descriptor(
                    in_n_x_wo_gm_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(X_, NumGroupsToMerge, C_)),
                               make_merge_transform(make_tuple(N_, Wo_))),
                    make_tuple(sequence<1, 3, 4>{}, sequence<0, 2>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return make_tuple(
                    out_gemm_k_gemm_m_grid_desc, in_gemm_n_gemm_k_grid_desc, wei_grid_desc);
            }
            else
            {
                // [N, Wi, C] -> [N, (Wi + InLeftPadW + InRightPadW), C] = [N, Wip, C]
                const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                // [N, Wip, C] -> [N, X, Wo, C]
                const auto in_n_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_n_wip_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                const auto in_gemmn_gemmktotal_grid_desc = transform_tensor_descriptor(
                    in_n_x_wo_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(X_, C_)),
                               make_merge_transform(make_tuple(N_, Wo_))),
                    make_tuple(sequence<1, 3>{}, sequence<0, 2>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return make_tuple(out_grid_desc, in_gemmn_gemmktotal_grid_desc, wei_grid_desc);
            }
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N() const
    {
        const auto out_grid_desc = make_out_grid_desc<NDimSpatial>();
        const auto in_grid_desc  = make_in_grid_desc<NDimSpatial>();
        const auto wei_grid_desc = make_wei_grid_desc<NDimSpatial>();

        // B: input tensor comes in K_N
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const auto out_grid_merged_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Ho_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto in_grid_merged_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Ho_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, C_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return make_tuple(out_grid_merged_desc, in_grid_merged_desc, wei_grid_desc);
            }
            else
            {
                return make_tuple(out_grid_desc, in_grid_desc, wei_grid_desc);
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                // Output tensor transformation
                // [0,          1, 2] -> [0,              1]
                // [(N*Ho*Wo), Gm, K] -> [(N*Ho*Wo), (K*Gm)]
                const auto out_gemm_k_gemm_m_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Ho_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                // Input tensor transformation, part 1.
                // [N, Hi, Wi, Gm, C] -> [N, Hip, Wip, Gm, C]
                const auto in_n_hip_wip_gm_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

                // Input tensor transformation, part 2.
                // [N, Hip, Wip, Gm, C] -> [N, (Y, Wo), (X, Wo), Gm, C]
                const auto in_n_y_ho_x_wo_gm_c_grid_desc = transform_tensor_descriptor(
                    in_n_hip_wip_gm_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5>{},
                               sequence<6>{}));

                // Input tensor transformation, part 3.
                // [0, 1,  2, 3,  4   5  6] -> [0,                  1]
                // [N, Y, Ho, X, Wo, Gm, C] -> [(N*Ho*Wo), (Gm*Y*X*C)]
                const auto in_gemm_n_gemm_k_grid_desc = transform_tensor_descriptor(
                    in_n_y_ho_x_wo_gm_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(Y_, X_, NumGroupsToMerge, C_)),
                               make_merge_transform(make_tuple(N_, Ho_, Wo_))),
                    make_tuple(sequence<1, 3, 5, 6>{}, sequence<0, 2, 4>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return make_tuple(
                    out_gemm_k_gemm_m_grid_desc, in_gemm_n_gemm_k_grid_desc, wei_grid_desc);
            }
            else
            {
                const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

                const auto in_gemmn_gemmktotal_grid_desc = transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(Y_, X_, C_)),
                               make_merge_transform(make_tuple(N_, Ho_, Wo_))),
                    make_tuple(sequence<1, 3, 5>{}, sequence<0, 2, 4>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return make_tuple(out_grid_desc, in_gemmn_gemmktotal_grid_desc, wei_grid_desc);
            }
        }
    }

    template <index_t NDim = NDimSpatial, typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N() const
    {
        const auto out_grid_desc = make_out_grid_desc<NDimSpatial>();
        const auto in_grid_desc  = make_in_grid_desc<NDimSpatial>();
        const auto wei_grid_desc = make_wei_grid_desc<NDimSpatial>();

        // B: input tensor comes in K_N
        if constexpr(ConvSpec == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                const auto out_grid_merged_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Do_ * Ho_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto in_grid_merged_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Do_ * Ho_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, C_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return make_tuple(out_grid_merged_desc, in_grid_merged_desc, wei_grid_desc);
            }
            else
            {
                return make_tuple(out_grid_desc, in_grid_desc, wei_grid_desc);
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge > 1)
            {
                // Output tensor transformation
                // [0,  1,            2] -> [0,                 1]
                // [(N*Do*Ho*Wo), Gm, K] -> [(N*Do*Ho*Wo), (K*Gm)]
                const auto out_gemm_k_gemm_m_grid_desc = transform_tensor_descriptor(
                    out_grid_desc,
                    make_tuple(make_pass_through_transform(N_ * Do_ * Ho_ * Wo_),
                               make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                // Input tensor transformation, part 1.
                // [N, Di, Hi, Wi, Gm, C] -> [N, Dip, Hip, Wip, Gm, C]
                const auto in_n_dip_hip_wip_gm_c_grid_desc = transform_tensor_descriptor(
                    in_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
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

                // Input tensor transformation, part 2.
                // [N, Zip, Hip, Wip, Gm, C] -> [N, (Z, Zo), (Y, Wo), (X, Wo), Gm, C]
                const auto in_n_z_do_y_ho_x_wo_gm_c_grid_desc = transform_tensor_descriptor(
                    in_n_dip_hip_wip_gm_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Z_, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5, 6>{},
                               sequence<7>{},
                               sequence<8>{}));

                // Input tensor transformation, part 3.
                // [0, 1,  2, 3,  4, 5,  6,  7, 8] -> [0,                       1]
                // [N, Z, Do, Y, Ho, X, Wo, Gm, C] -> [(N*Do*Ho*Wo), (Z*Y*X*Gm*C)]
                const auto in_gemm_k_gemm_n_grid_desc = transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_gm_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(Z_, Y_, X_, NumGroupsToMerge, C_)),
                               make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_))),
                    make_tuple(sequence<1, 3, 5, 7, 8>{}, sequence<0, 2, 4, 6>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return make_tuple(
                    out_gemm_k_gemm_m_grid_desc, in_gemm_k_gemm_n_grid_desc, wei_grid_desc);
            }
            else
            {
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

                const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_grid_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Z_, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5, 6>{},
                               sequence<7>{}));

                const auto in_gemmn_gemmktotal_grid_desc = transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_grid_desc,
                    make_tuple(make_merge_transform(make_tuple(Z_, Y_, X_, C_)),
                               make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_))),
                    make_tuple(sequence<1, 3, 5, 7>{}, sequence<0, 2, 4, 6>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return make_tuple(out_grid_desc, in_gemmn_gemmktotal_grid_desc, wei_grid_desc);
            }
        }
    }

    IndexType G_, N_;
    IndexType Di_, Hi_, Wi_;
    IndexType Do_, Ho_, Wo_;
    IndexType Z_, Y_, X_;
    IndexType K_, C_;
    IndexType ConvStrideD_, ConvStrideH_, ConvStrideW_;
    IndexType ConvDilationD_, ConvDilationH_, ConvDilationW_;
    IndexType InLeftPadD_, InLeftPadH_, InLeftPadW_;
    IndexType InRightPadD_, InRightPadH_, InRightPadW_;
};

} // namespace ck_tile
