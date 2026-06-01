// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"

#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/impl/split_k_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename DeviceGemmV3Op>
struct DeviceGroupedConvBwdWeight_Explicit
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation>
{
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr bool IsTwoStageNeeded =
        sizeof(WeiDataType) % 4 != 0 &&
        DeviceGemmV3Op::CDEShuffleBlockTransferScalarPerVectors_::At(I0) % 2 != 0;

    using DeviceOp                 = DeviceGroupedConvBwdWeight_Explicit;
    using TwoStageIntermediateType = typename DeviceGemmV3Op::CDataType_;

    static constexpr index_t ElementwiseBlockSize = 256;
    static constexpr index_t ElemsPerBlock        = 256;

    static auto GetElementwiseCGridDesc(index_t merged_filter_dims)
    {
        const auto padd_size = merged_filter_dims % ElemsPerBlock == 0
                                   ? 0
                                   : ElemsPerBlock - merged_filter_dims % ElemsPerBlock;
        const auto desc = make_naive_tensor_descriptor_packed(make_tuple(I1, merged_filter_dims));
        return transform_tensor_descriptor(
            desc,
            make_tuple(make_pass_through_transform(I1),
                       make_right_pad_transform(merged_filter_dims, padd_size)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    using CElementwiseGridDesc     = remove_cvref_t<decltype(GetElementwiseCGridDesc(I1))>;
    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<1, ElemsPerBlock>;
    using GridwiseElementwiseCast  = GridwiseElementwise<Tuple<CElementwiseGridDesc>,
                                                         Tuple<CElementwiseGridDesc>,
                                                         Tuple<const float*>,
                                                         Tuple<WeiDataType*>,
                                                         Block2TileMapElementwise,
                                                         WeiElementwiseOperation,
                                                         ElementwiseBlockSize,
                                                         I1,
                                                         ElemsPerBlock,
                                                         I1,
                                                         ElemsPerBlock / ElementwiseBlockSize,
                                                         Sequence<0, 1>,
                                                         Sequence<1>,
                                                         Sequence<1>,
                                                         I1,
                                                         I1>;

    struct Argument : public BaseArgument, public ArgumentSplitK
    {
        using GemmArgument = typename DeviceGemmV3Op::Argument;

        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>&, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>&,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : filter_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              p_wei_grid_{p_wei_grid}
        {
            constexpr index_t spatial_offset = 3;
            const index_t DoHoWo = std::accumulate(begin(a_g_n_k_wos_lengths) + spatial_offset,
                                                   end(a_g_n_k_wos_lengths),
                                                   index_t{1},
                                                   std::multiplies<>{});
            const index_t M      = e_g_k_c_xs_lengths[I1];
            const index_t N      = e_g_k_c_xs_lengths[I2];
            const index_t K      = a_g_n_k_wos_lengths[I1] * DoHoWo;

            const index_t StrideOut      = a_g_n_k_wos_strides[spatial_offset + NDimSpatial - 1];
            const index_t StrideIn       = b_g_n_c_wis_strides[spatial_offset + NDimSpatial - 1];
            const index_t StrideWei      = e_g_k_c_xs_strides[I1];
            const index_t StrideBatchOut = a_g_n_k_wos_strides[I0];
            const index_t StrideBatchIn  = b_g_n_c_wis_strides[I0];
            const index_t StrideBatchWei = e_g_k_c_xs_strides[I0];

            const index_t BatchSize = a_g_n_k_wos_lengths[I0];

            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));

            if constexpr(IsTwoStageNeeded)
            {
                if(split_k < 0)
                {
                    const auto max_occupancy = DeviceGemmV3Op::GetMaxOccupancy();
                    index_t gdx, gdy, gdz;
                    std::tie(gdx, gdy, gdz) =
                        DeviceGemmV3Op::GridwiseGemm::CalculateGridSize(M, N, BatchSize);
                    const index_t grid_size = gdx * gdy * gdz;
                    k_batch_ = get_best_occupancy_k_batch_value(max_occupancy, grid_size);
                }
                else
                {
                    k_batch_ = split_k;
                }
            }
            else
            {
#if !DISABLE_SPLIT_K_AUTODEDUCE_FOR_ONE_STAGE_KERNELS
                if(split_k < 0)
                {
                    const auto max_occupancy = DeviceGemmV3Op::GetMaxOccupancy();
                    index_t gdx, gdy, gdz;
                    std::tie(gdx, gdy, gdz) =
                        DeviceGemmV3Op::GridwiseGemm::CalculateGridSize(M, N, BatchSize);
                    const index_t grid_size = gdx * gdy * gdz;
                    k_batch_ = get_best_occupancy_k_batch_value(max_occupancy, grid_size);
                }
                else
#endif
                {
                    k_batch_ = split_k;
                }
            }

            if constexpr(IsTwoStageNeeded)
            {
                const index_t merged_filter_dims = std::accumulate(begin(e_g_k_c_xs_lengths),
                                                                   end(e_g_k_c_xs_lengths),
                                                                   index_t{1},
                                                                   std::multiplies<>{});
                elementwise_desc_                = GetElementwiseCGridDesc(merged_filter_dims);
                elementwise_block_2_ctile_map_   = Block2TileMapElementwise{1, merged_filter_dims};
                // Check if stride to last dimension is product of all other dimensions. Then it is
                // packed.
                is_filter_data_packed =
                    e_g_k_c_xs_strides[0] == (merged_filter_dims / e_g_k_c_xs_lengths[0]);

                // Data type is modified during launch. It is checked in IsSupported if user
                // allocated workspace
                explicit_gemm_args = GemmArgument{p_out_grid,
                                                  p_in_grid,
                                                  {},
                                                  static_cast<TwoStageIntermediateType*>(nullptr),
                                                  M,
                                                  N,
                                                  K,
                                                  StrideOut,
                                                  StrideIn,
                                                  {},
                                                  StrideWei,
                                                  StrideBatchOut,
                                                  StrideBatchIn,
                                                  {},
                                                  StrideBatchWei,
                                                  BatchSize,
                                                  out_element_op,
                                                  in_element_op,
                                                  wei_element_op,
                                                  k_batch_};
            }
            else
            {
                explicit_gemm_args = GemmArgument{p_out_grid,
                                                  p_in_grid,
                                                  {},
                                                  p_wei_grid,
                                                  M,
                                                  N,
                                                  K,
                                                  StrideOut,
                                                  StrideIn,
                                                  {},
                                                  StrideWei,
                                                  StrideBatchOut,
                                                  StrideBatchIn,
                                                  {},
                                                  StrideBatchWei,
                                                  BatchSize,
                                                  out_element_op,
                                                  in_element_op,
                                                  wei_element_op,
                                                  k_batch_};
            }
        }

        std::size_t GetWorkspaceETensorSizeBytes() const
        {
            if constexpr(IsTwoStageNeeded)
            {
                return sizeof(TwoStageIntermediateType) * elementwise_desc_.GetElementSpaceSize();
            }
            else
            {
                return 0;
            }
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            if constexpr(IsTwoStageNeeded)
            {
                return GetWorkspaceETensorSizeBytes();
            }
            else
            {
                return 0;
            }
        }

        GemmArgument explicit_gemm_args;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& input_left_pads_;
        const std::array<ck::index_t, NDimSpatial>& input_right_pads_;
        WeiDataType* p_wei_grid_;
        bool is_filter_data_packed;
        CElementwiseGridDesc elementwise_desc_;
        Block2TileMapElementwise elementwise_block_2_ctile_map_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument     = DeviceOp::Argument;
        using GemmArgument = typename DeviceGemmV3Op::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if constexpr(IsTwoStageNeeded)
            {
                // Modify to use workspace as output
                GemmArgument explicit_gemm_args_with_workspace = arg.explicit_gemm_args;
                explicit_gemm_args_with_workspace.template SetEPointer<TwoStageIntermediateType>(
                    arg.p_workspace_);
                float avg_time =
                    explicit_gemm_op.Run(explicit_gemm_args_with_workspace, stream_config);
                const index_t grid_size =
                    arg.elementwise_block_2_ctile_map_.CalculateGridSize(arg.elementwise_desc_);
                const auto kernel = kernel_elementwise<GridwiseElementwiseCast,
                                                       ck::Tuple<CElementwiseGridDesc>,
                                                       ck::Tuple<CElementwiseGridDesc>,
                                                       ck::Tuple<const TwoStageIntermediateType*>,
                                                       ck::Tuple<WeiDataType*>,
                                                       Block2TileMapElementwise,
                                                       WeiElementwiseOperation>;

                avg_time += launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(grid_size),
                    dim3(ElementwiseBlockSize),
                    0,
                    make_tuple(arg.elementwise_desc_),
                    make_tuple(arg.elementwise_desc_),
                    make_tuple(static_cast<const TwoStageIntermediateType*>(arg.p_workspace_)),
                    make_tuple(arg.p_wei_grid_),
                    arg.elementwise_block_2_ctile_map_,
                    element_wise::PassThrough{});
                return avg_time;
            }
            else
            {
                return explicit_gemm_op.Run(arg.explicit_gemm_args, stream_config);
            }
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }

        typename DeviceGemmV3Op::Invoker explicit_gemm_op;
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
#if DISABLE_SPLIT_K_AUTODEDUCE_FOR_ONE_STAGE_KERNELS
        if constexpr(!IsTwoStageNeeded)
        {
            if(arg.k_batch_ < 0)
            {
                return false;
            }
        }
#endif

        if constexpr(NDimSpatial == 2)
        {
            if constexpr(!is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported layout." << std::endl;
                }
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported layout." << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported layout." << std::endl;
            }
            return false;
        }

        // check if it's 1x1, stride=1 pad = 0 conv
        for(int i = 0; i < NDimSpatial; i++)
        {
            if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                 arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported stride / pad." << std::endl;
                }
                return false;
            }
        }
        if constexpr(IsTwoStageNeeded)
        {
            if(!arg.is_filter_data_packed)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported: Filter data must be packed." << std::endl;
                }
                return false;
            }
            // Check this here, it allows to use other instances from factory even
            // if workspace is not allocated
            if(!arg.p_workspace_)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Warning: Workspace for "
                                 "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument is not "
                                 "allocated, use SetWorkSpacePointer."
                              << std::endl;
                }
                return false;
            }
        }
        // Gridwise GEMM size
        return DeviceGemmV3Op::IsSupportedArgument(arg.explicit_gemm_args);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdWeight_Explicit_Xdl"
            << "<" << DeviceGemmV3Op{}.GetTypeString() << ">";
        // clang-format on

        return str.str();
    }
    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
