// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/library/utility/numeric.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_tensor_rearrange.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_tensor_rearrange.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"

#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/conv_tensor_rearrange_op.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Column to Image:
//   input : gemm form [G, N * Do * Ho * Wo, Z * Y * X * C]
//   output : input image [G, N, Di, Hi, Wi, C]
//   input : gemm form [N * Do * Ho * Wo, G, Z * Y * X * C]
//   output : input image [N, Di, Hi, Wi, G, C]
template <index_t NDimSpatial,
          typename ImageLayout,
          typename InputDataType,
          typename OutputDataType,
          index_t BlockSize,
          index_t MPerBlock,
          index_t KPerBlock,
          typename ThreadClusterLengths,
          index_t ScalarPerVector,
          typename std::enable_if<NDimSpatial >= 1 && NDimSpatial <= 3, bool>::type = false>
struct DeviceColumnToImageImpl
    : public DeviceConvTensorRearrange<NDimSpatial,
                                       ImageLayout,
                                       InputDataType,
                                       OutputDataType,
                                       conv_tensor_rearrange_op::ColumnToImage>
{
    static constexpr bool is_NSpatialGC =
        std::is_same_v<ImageLayout, tensor_layout::convolution::NWGC> ||
        std::is_same_v<ImageLayout, tensor_layout::convolution::NHWGC> ||
        std::is_same_v<ImageLayout, tensor_layout::convolution::NDHWGC>;
    static constexpr bool is_GNSpatialC =
        std::is_same_v<ImageLayout, tensor_layout::convolution::GNWC> ||
        std::is_same_v<ImageLayout, tensor_layout::convolution::GNHWC> ||
        std::is_same_v<ImageLayout, tensor_layout::convolution::GNDHWC>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto ZIdx = Number<I0>{};
    static constexpr auto YIdx = NDimSpatial == 1 ? I0 : Number<NDimSpatial - I2>{};
    static constexpr auto XIdx = Number<NDimSpatial - I1>{};

    static constexpr auto spatial_offset = Number<3>{};

    using ConvToGemmFwdTransformer =
        TransformConvFwdToGemm<NDimSpatial, ConvolutionForwardSpecialization::Default>;
    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpecialization::MKPadding, index_t, index_t, index_t>{
            MPerBlock, 0 /* NPerBlock*/, KPerBlock};

    // Calculate number of independent filters for given conv params
    static index_t GetNumberOfIndependentFilters(const index_t input_spatial_len,
                                                 const index_t left_pad,
                                                 const index_t right_pad,
                                                 const index_t filter_len,
                                                 const index_t filter_stride,
                                                 const index_t filter_dilation,
                                                 const index_t image_offset)
    {
        const index_t x_eff = (filter_len - 1) * filter_dilation + 1;
        const index_t next_filter_padded =
            math::integer_divide_ceil(x_eff, filter_stride) * filter_stride;
        // If filter_stride >= x_eff then each filter is independent
        const index_t independent_filter_stride =
            filter_stride >= x_eff ? filter_stride : next_filter_padded;
        const index_t w_eff = input_spatial_len - image_offset + left_pad + right_pad - x_eff;
        // There are no independent filters
        if(w_eff < 0)
            return 0;
        const index_t independent_kernels_num = w_eff / independent_filter_stride + 1;
        return independent_kernels_num;
    }

    // Make column form descriptor
    static auto
    MakeInputDescriptor_M_K(const ck::index_t N,
                            const ck::index_t C,
                            const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                            const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                            const std::array<index_t, NDimSpatial>& conv_filter_strides,
                            const std::array<index_t, 3>& gemm_g_m_k_strides,
                            const std::array<index_t, NDimSpatial>& independent_filters,
                            const std::array<index_t, NDimSpatial>& effs)
    {
        const index_t DoHoWo = ck::accumulate_n<index_t>(
            output_spatial_lengths.begin(), NDimSpatial, 1, std::multiplies<>());
        const index_t CZYX =
            C * ck::accumulate_n<index_t>(
                    filter_spatial_lengths.begin(), NDimSpatial, 1, std::multiplies<>());

        const index_t NStride = DoHoWo * gemm_g_m_k_strides[I1] * gemm_g_m_k_strides[I2];
        // Calculate the appropriate stride for each set of independent filters
        // in each dimension
        const index_t WStride = math::integer_divide_ceil(effs[XIdx], conv_filter_strides[XIdx]) *
                                gemm_g_m_k_strides[I1];
        const index_t HStride = math::integer_divide_ceil(effs[YIdx], conv_filter_strides[YIdx]) *
                                output_spatial_lengths[XIdx] * gemm_g_m_k_strides[I1];
        const index_t DStride = math::integer_divide_ceil(effs[ZIdx], conv_filter_strides[ZIdx]) *
                                output_spatial_lengths[YIdx] * output_spatial_lengths[XIdx] *
                                gemm_g_m_k_strides[I1];
        // Create descriptor for independent filters in each dimension and
        // then merge them into column form
        if constexpr(NDimSpatial == 1)
        {
            const auto desc_gemm_form =
                make_naive_tensor_descriptor(make_tuple(N, independent_filters[XIdx], CZYX),
                                             make_tuple(NStride, WStride, gemm_g_m_k_strides[I2]));
            const auto desc_gemm_form_merged_filters = transform_tensor_descriptor(
                desc_gemm_form,
                make_tuple(make_merge_transform(make_tuple(N, independent_filters[XIdx])),
                           make_pass_through_transform(CZYX)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
            const auto desc_m_k = matrix_padder.PadADescriptor_M_K(desc_gemm_form_merged_filters);
            return desc_m_k;
        }
        else if constexpr(NDimSpatial == 2)
        {
            const auto desc_gemm_form = make_naive_tensor_descriptor(
                make_tuple(N, independent_filters[YIdx], independent_filters[XIdx], CZYX),
                make_tuple(NStride, HStride, WStride, gemm_g_m_k_strides[I2]));
            const auto desc_gemm_form_merged_filters = transform_tensor_descriptor(
                desc_gemm_form,
                make_tuple(make_merge_transform(
                               make_tuple(N, independent_filters[YIdx], independent_filters[XIdx])),
                           make_pass_through_transform(CZYX)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
            const auto desc_m_k = matrix_padder.PadADescriptor_M_K(desc_gemm_form_merged_filters);
            return desc_m_k;
        }
        else if constexpr(NDimSpatial == 3)
        {
            const auto desc_gemm_form = make_naive_tensor_descriptor(
                make_tuple(N,
                           independent_filters[ZIdx],
                           independent_filters[YIdx],
                           independent_filters[XIdx],
                           CZYX),
                make_tuple(NStride, DStride, HStride, WStride, gemm_g_m_k_strides[I2]));
            const auto desc_gemm_form_merged_filters = transform_tensor_descriptor(
                desc_gemm_form,
                make_tuple(make_merge_transform(make_tuple(N,
                                                           independent_filters[ZIdx],
                                                           independent_filters[YIdx],
                                                           independent_filters[XIdx])),
                           make_pass_through_transform(CZYX)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
            const auto desc_m_k = matrix_padder.PadADescriptor_M_K(desc_gemm_form_merged_filters);
            return desc_m_k;
        }
    }

    // Use MakeADescriptor_M_K from grouped convolution forward
    static auto
    MakeOutDescriptor_M_K(const ck::index_t N,
                          const ck::index_t C,
                          const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                          const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                          const std::array<index_t, NDimSpatial + 3>& image_g_n_c_wis_strides,
                          const std::array<index_t, NDimSpatial>& conv_filter_strides,
                          const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                          const std::array<index_t, NDimSpatial>& input_left_pads,
                          const std::array<index_t, NDimSpatial>& input_right_pads,
                          const std::array<index_t, NDimSpatial>& image_offsets,
                          const std::array<index_t, NDimSpatial>& independent_filters,
                          const std::array<index_t, NDimSpatial>& effs)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{1};
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{1};
        std::array<index_t, NDimSpatial + 3> c_g_n_k_wos_lengths{1};

        auto copy = [](const auto& x, auto& y, index_t dst_offset) {
            std::copy(x.begin(), x.end(), y.begin() + dst_offset);
        };

        copy(input_spatial_lengths, a_g_n_c_wis_lengths, spatial_offset);
        copy(filter_spatial_lengths, b_g_k_c_xs_lengths, spatial_offset);
        // Calculate descriptor only for independent filters
        copy(independent_filters, c_g_n_k_wos_lengths, spatial_offset);

        // fill only significant values (C and N)
        a_g_n_c_wis_lengths[I1] = N;
        a_g_n_c_wis_lengths[I2] = C;
        b_g_k_c_xs_lengths[I2]  = C;
        c_g_n_k_wos_lengths[I1] = N;

        // Modify pads to apply offsets
        std::array<index_t, NDimSpatial> input_left_pads_with_offset;
        for(index_t i = 0; i < NDimSpatial; i++)
        {
            input_left_pads_with_offset[i] = math::max(0, input_left_pads[i] - image_offsets[i]);
        }
        // Modify input spatial lengths to apply offsets
        for(index_t i = 0; i < NDimSpatial; i++)
        {
            a_g_n_c_wis_lengths[i + spatial_offset] -=
                math::max(0, image_offsets[i] - input_left_pads[i]);
        }

        // Strides to next independent filters
        std::array<index_t, NDimSpatial> independent_filter_strides;
        for(index_t i = 0; i < NDimSpatial; i++)
        {
            index_t independent_filter_stride =
                math::integer_divide_ceil(effs[i], conv_filter_strides[i]) * conv_filter_strides[i];
            // If conv stride is greater than whole filter size, use conv stride
            independent_filter_strides[i] = conv_filter_strides[i] >= effs[i]
                                                ? conv_filter_strides[i]
                                                : independent_filter_stride;
        }

        ConvToGemmFwdTransformer conv_to_gemm_transformer{a_g_n_c_wis_lengths,
                                                          image_g_n_c_wis_strides,
                                                          b_g_k_c_xs_lengths,
                                                          {}, // not needed for A Descriptor
                                                          c_g_n_k_wos_lengths,
                                                          {}, // not needed for A Descriptor
                                                          // conv_filter_strides,
                                                          independent_filter_strides,
                                                          conv_filter_dilations,
                                                          input_left_pads_with_offset,
                                                          input_right_pads};

        // Calculate image form descriptor for the modified convolution problem
        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<ImageLayout>();

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);
        return in_gemmm_gemmk_desc;
    }

    using InputGridDesc =
        remove_cvref_t<decltype(MakeInputDescriptor_M_K(1, 1, {}, {}, {}, {}, {}, {}))>;
    using OutputGridDesc = remove_cvref_t<decltype(MakeOutDescriptor_M_K(
        1, 1, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}))>;

    using Block2ETileMap = remove_cvref_t<
        decltype(BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, KPerBlock, InputGridDesc>(
            InputGridDesc{}))>;

    using GridwiseTensorRearrangeKernel = GridwiseTensorRearrange<InputGridDesc,
                                                                  InputDataType,
                                                                  OutputGridDesc,
                                                                  OutputDataType,
                                                                  BlockSize,
                                                                  MPerBlock,
                                                                  KPerBlock,
                                                                  ThreadClusterLengths,
                                                                  ScalarPerVector,
                                                                  InMemoryDataOperationEnum::Add,
                                                                  Block2ETileMap,
                                                                  ComputePtrOffsetOfStridedBatch<>>;

    struct Argument : public BaseArgument
    {
        Argument(const void* p_in, // input image
                 void* p_out,      // output image
                 const ck::index_t G,
                 const ck::index_t N,
                 const ck::index_t C,
                 const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                 const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                 const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                 const std::array<index_t, NDimSpatial + 3>& image_g_n_c_wis_strides,
                 const std::array<index_t, 3>& gemm_g_m_k_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads)
            : G_(G),
              C_(C),
              X_(filter_spatial_lengths[NDimSpatial - I1]),
              p_in_{static_cast<const InputDataType*>(p_in)},
              p_out_{static_cast<OutputDataType*>(p_out)},
              image_g_n_c_wis_strides_{image_g_n_c_wis_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            compute_ptr_offset_of_batch_.BatchStrideA_ = gemm_g_m_k_strides[I0];
            compute_ptr_offset_of_batch_.BatchStrideC_ = image_g_n_c_wis_strides[I0];

            const index_t x_eff =
                (filter_spatial_lengths[XIdx] - 1) * conv_filter_dilations[XIdx] + 1;
            const index_t y_eff =
                NDimSpatial < 2
                    ? I1
                    : (filter_spatial_lengths[YIdx] - 1) * conv_filter_dilations[YIdx] + 1;
            const index_t z_eff =
                NDimSpatial < 3
                    ? I1
                    : (filter_spatial_lengths[ZIdx] - 1) * conv_filter_dilations[ZIdx] + 1;

            // Iterate over sets of independent filters
            for(int z_img_offset = 0; z_img_offset < z_eff;
                z_img_offset += conv_filter_strides[ZIdx])
            {
                for(int y_img_offset = 0; y_img_offset < y_eff;
                    y_img_offset += conv_filter_strides[YIdx])
                {
                    for(int x_img_offset = 0; x_img_offset < x_eff;
                        x_img_offset += conv_filter_strides[XIdx])
                    {

                        std::array<index_t, NDimSpatial> image_offsets;
                        std::array<index_t, NDimSpatial> effs;
                        // Calculate the starting offset for a given set of
                        // independent filters
                        if constexpr(NDimSpatial == 1)
                        {
                            image_offsets = {x_img_offset};
                            effs          = {x_eff};
                        }
                        if constexpr(NDimSpatial == 2)
                        {
                            image_offsets = {y_img_offset, x_img_offset};
                            effs          = {y_eff, x_eff};
                        }
                        else if constexpr(NDimSpatial == 3)
                        {
                            image_offsets = {z_img_offset, y_img_offset, x_img_offset};
                            effs          = {z_eff, y_eff, x_eff};
                        }

                        std::array<index_t, NDimSpatial> independent_filters;
                        for(index_t i = 0; i < NDimSpatial; i++)
                        {
                            independent_filters[i] =
                                GetNumberOfIndependentFilters(input_spatial_lengths[i],
                                                              input_left_pads[i],
                                                              input_right_pads[i],
                                                              filter_spatial_lengths[i],
                                                              conv_filter_strides[i],
                                                              conv_filter_dilations[i],
                                                              image_offsets[i]);
                        }
                        const index_t independent_filters_acum = ck::accumulate_n<index_t>(
                            independent_filters.begin(), NDimSpatial, 1, std::multiplies<>());
                        if(independent_filters_acum <= 0)
                            continue;

                        const auto in_grid_desc_m_k =
                            MakeInputDescriptor_M_K(N,
                                                    C,
                                                    filter_spatial_lengths,
                                                    output_spatial_lengths,
                                                    conv_filter_strides,
                                                    gemm_g_m_k_strides,
                                                    independent_filters,
                                                    effs);
                        const auto out_grid_desc_m_k =
                            MakeOutDescriptor_M_K(N,
                                                  C,
                                                  input_spatial_lengths,
                                                  filter_spatial_lengths,
                                                  image_g_n_c_wis_strides,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  image_offsets,
                                                  independent_filters,
                                                  effs);
                        in_grid_desc_m_k_container_.push_back(in_grid_desc_m_k);
                        out_grid_desc_m_k_container_.push_back(out_grid_desc_m_k);

                        const index_t x_idx = x_img_offset / conv_filter_strides[XIdx];
                        const index_t y_idx = y_img_offset / conv_filter_strides[YIdx];
                        const index_t z_idx = z_img_offset / conv_filter_strides[ZIdx];

                        const index_t x_offset_with_pad =
                            math::max(0, x_img_offset - input_left_pads[XIdx]);
                        const index_t y_offset_with_pad =
                            math::max(0, y_img_offset - input_left_pads[YIdx]);
                        const index_t z_offset_with_pad =
                            math::max(0, z_img_offset - input_left_pads[ZIdx]);

                        // Memory offsets to next set of independent filters,
                        // move to independent filters in each dimension
                        const index_t in_offset =
                            (x_idx + y_idx * output_spatial_lengths[XIdx] +
                             z_idx * output_spatial_lengths[YIdx] * output_spatial_lengths[XIdx]) *
                            gemm_g_m_k_strides[I1];
                        // Move to independent filters in appropriate dimensions
                        const index_t out_offset =
                            x_offset_with_pad * image_g_n_c_wis_strides[spatial_offset + XIdx] +
                            y_offset_with_pad * image_g_n_c_wis_strides[spatial_offset + YIdx] +
                            z_offset_with_pad * image_g_n_c_wis_strides[spatial_offset + ZIdx];

                        const InputDataType* p_in_with_offset =
                            static_cast<const InputDataType*>(p_in) + in_offset;
                        OutputDataType* p_out_with_offset =
                            static_cast<OutputDataType*>(p_out) + out_offset;
                        p_in_container_.push_back(p_in_with_offset);
                        p_out_container_.push_back(p_out_with_offset);
                    }
                }
            }
        }

        void Print() const
        {
            for(std::size_t i = 0; i < in_grid_desc_m_k_container_.size(); i++)
            {
                std::cout << in_grid_desc_m_k_container_[i] << std::endl;
                std::cout << out_grid_desc_m_k_container_[i] << std::endl;
            }
        }

        const ck::index_t G_;
        const ck::index_t C_;
        const ck::index_t X_;

        const InputDataType* p_in_;
        OutputDataType* p_out_;

        const std::array<index_t, NDimSpatial + 3>& image_g_n_c_wis_strides_;
        const std::array<index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<index_t, NDimSpatial>& conv_filter_dilations_;
        const std::array<index_t, NDimSpatial>& input_left_pads_;
        const std::array<index_t, NDimSpatial>& input_right_pads_;

        std::vector<InputGridDesc> in_grid_desc_m_k_container_;
        std::vector<OutputGridDesc> out_grid_desc_m_k_container_;

        std::vector<const InputDataType*> p_in_container_;
        std::vector<OutputDataType*> p_out_container_;

        ComputePtrOffsetOfStridedBatch<> compute_ptr_offset_of_batch_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            float elapsed_time = 0.f;
            const auto kernel  = kernel_tensor_rearrange<InputGridDesc,
                                                         InputDataType,
                                                         OutputGridDesc,
                                                         OutputDataType,
                                                         Block2ETileMap,
                                                         ComputePtrOffsetOfStridedBatch<>,
                                                         GridwiseTensorRearrangeKernel>;

            // Execute each set of independent filters
            for(std::size_t i = 0; i < arg.in_grid_desc_m_k_container_.size(); i++)
            {
                const auto block_2_tile_map =
                    BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, KPerBlock, InputGridDesc>(
                        arg.out_grid_desc_m_k_container_[i]);
                const index_t grid_size =
                    block_2_tile_map.CalculateGridSize(arg.in_grid_desc_m_k_container_[i]) * arg.G_;
                elapsed_time += launch_and_time_kernel(stream_config,
                                                       kernel,
                                                       dim3(grid_size),
                                                       dim3(BlockSize),
                                                       0,
                                                       arg.in_grid_desc_m_k_container_[i],
                                                       arg.p_in_container_[i],
                                                       arg.out_grid_desc_m_k_container_[i],
                                                       arg.p_out_container_[i],
                                                       arg.G_,
                                                       block_2_tile_map,
                                                       arg.compute_ptr_offset_of_batch_);
            }
            return elapsed_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const Argument& arg)
    {
        using namespace tensor_layout::convolution;
        if constexpr(!(is_NSpatialGC || is_GNSpatialC))
        {
            return false;
        }

        const auto w_pad_left  = arg.input_left_pads_[NDimSpatial - I1];
        const auto w_pad_right = arg.input_right_pads_[NDimSpatial - I1];
        const auto dilation_x  = arg.conv_filter_dilations_[NDimSpatial - I1];
        const auto stride_x    = arg.conv_filter_strides_[NDimSpatial - I1];
        bool is_w_packed       = arg.image_g_n_c_wis_strides_[NDimSpatial + I2] == arg.C_;
        bool is_c_packed       = arg.image_g_n_c_wis_strides_[I2] == 1;

        // check vector acces with c not packed
        if(!is_c_packed && ScalarPerVector != 1)
            return false;
        // check vector access of filter window row (only C if C is not packed)
        if(!is_w_packed && arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of filter window row (X * C)
        if(arg.X_ * arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of pads (w_pad_left/w_pad_right * C)
        if(w_pad_left * arg.C_ % ScalarPerVector != 0 ||
           w_pad_right * arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of with stride and pad
        if((w_pad_left != 0 || w_pad_right != 0) && stride_x > 1 && arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of with dilation
        if(dilation_x > 1 && arg.C_ % ScalarPerVector != 0)
            return false;

        bool valid = true;
        for(std::size_t i = 0; i < arg.in_grid_desc_m_k_container_.size(); i++)
        {
            valid &= GridwiseTensorRearrangeKernel::CheckValidity(
                arg.in_grid_desc_m_k_container_[i], arg.out_grid_desc_m_k_container_[i]);
        }
        return valid;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_in, // input image
                             void* p_out,      // output image
                             const ck::index_t G,
                             const ck::index_t N,
                             const ck::index_t C,
                             const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                             const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                             const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                             const std::array<index_t, NDimSpatial + 3>& image_g_n_c_wis_strides,
                             const std::array<index_t, 3>& gemm_g_m_k_strides,
                             const std::array<index_t, NDimSpatial>& conv_filter_strides,
                             const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                             const std::array<index_t, NDimSpatial>& input_left_pads,
                             const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        return Argument{static_cast<const InputDataType*>(p_in),
                        static_cast<OutputDataType*>(p_out),
                        G,
                        N,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        image_g_n_c_wis_strides,
                        gemm_g_m_k_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in, // input image
                        void* p_out,      // output image
                        const ck::index_t G,
                        const ck::index_t N,
                        const ck::index_t C,
                        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                        const std::array<index_t, NDimSpatial + 3>& image_g_n_c_wis_strides,
                        const std::array<index_t, 3>& gemm_g_m_k_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads) override
    {
        return std::make_unique<Argument>(static_cast<const InputDataType*>(p_in),
                                          static_cast<OutputDataType*>(p_out),
                                          G,
                                          N,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          image_g_n_c_wis_strides,
                                          gemm_g_m_k_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceColumnToImage"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << KPerBlock << ", "
            << ScalarPerVector
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
