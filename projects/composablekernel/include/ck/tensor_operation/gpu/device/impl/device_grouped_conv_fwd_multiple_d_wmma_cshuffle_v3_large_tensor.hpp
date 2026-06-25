// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <sstream>
#include <utility>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

template <typename GridwiseGemm,
          index_t MaxGemmsNum,
          typename GemmArgs,
          typename ComputePtrOffset,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          TailNumber TailNum = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_conv_fwd_grouped_gemm_wmma_cshuffle_v3(
        Array<GemmArgs, MaxGemmsNum> gemm_desc_kernel_args,
        const index_t gemms_count,
        const ComputePtrOffset compute_ptr_offset_of_groups,
        const ComputePtrOffset compute_ptr_offset_of_n)
{
#if defined(__gfx11__) || defined(__gfx12__)
    using Epilogue = typename GridwiseGemm::EpilogueCShuffle;
    __shared__ char p_shared[GridwiseGemm::template GetSharedMemoryNumberOfByte<Epilogue>()];

    const index_t block_id_x = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const index_t g_idx      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const index_t n_idx      = __builtin_amdgcn_readfirstlane(blockIdx.z);

    const long_index_t a_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
    const long_index_t b_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
    const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);
    const long_index_t e_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

    const long_index_t a_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
    const long_index_t b_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetBPtrOffset(n_idx));
    const auto& ds_n_offset = compute_ptr_offset_of_n.GetDsPtrOffset(n_idx);
    const long_index_t e_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

    index_t left     = 0;
    index_t right    = gemms_count;
    index_t group_id = index_t((left + right) / 2);
    while((!(block_id_x >= gemm_desc_kernel_args[group_id].BlockStart_ &&
             block_id_x < gemm_desc_kernel_args[group_id].BlockEnd_)) &&
          left <= right)
    {
        if(block_id_x < gemm_desc_kernel_args[group_id].BlockStart_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    const auto& gemm_arg  = gemm_desc_kernel_args[group_id];
    const index_t block_x = block_id_x - gemm_arg.BlockStart_;

    typename GridwiseGemm::AsGridPointer p_as_grid_;
    static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
        using ADataType_ = remove_cvref_t<tuple_element_t<i.value, typename GemmArgs::AsDataType>>;
        p_as_grid_(i) =
            static_cast<const ADataType_*>(gemm_arg.a_ptrs_[i]) + a_group_offset + a_n_offset;
    });

    typename GridwiseGemm::BsGridPointer p_bs_grid_;
    static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
        using BDataType_ = remove_cvref_t<tuple_element_t<i.value, typename GemmArgs::BsDataType>>;
        p_bs_grid_(i) =
            static_cast<const BDataType_*>(gemm_arg.b_ptrs_[i]) + b_group_offset + b_n_offset;
    });

    typename GridwiseGemm::DsGridPointer p_ds_grid_;
    static_for<0, GemmArgs::NumDTensor, 1>{}([&](auto i) {
        using DDataType_ =
            remove_cvref_t<tuple_element_t<i.value, typename GemmArgs::DsDataTypeTuple>>;
        p_ds_grid_(i) = static_cast<const DDataType_*>(gemm_arg.ds_ptrs_[i]) + ds_group_offset[i] +
                        ds_n_offset[i];
    });

    const auto as_grid_desc_ak0_m_ak1 = generate_tuple([&](auto) { return gemm_arg.a_grid_desc_; },
                                                       Number<GridwiseGemm::NumATensor>{});

    const auto bs_grid_desc_bk0_n_bk1 = generate_tuple([&](auto) { return gemm_arg.b_grid_desc_; },
                                                       Number<GridwiseGemm::NumBTensor>{});

    const auto& ds_grid_desc = gemm_arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_;
    const auto& e_grid_desc  = gemm_arg.e_grid_desc_mblock_mperblock_nblock_nperblock_;

    const auto block_2_ctile_map =
        typename GridwiseGemm::Block2CTileMap{gemm_arg.M_, gemm_arg.N_, 4};
    const auto block_work_idx = block_2_ctile_map.CalculateBottomIndex(make_multi_index(block_x));

    if(!block_2_ctile_map.ValidCTileIndex(
           block_work_idx,
           make_tuple(e_grid_desc.GetLength(Number<0>{}), e_grid_desc.GetLength(Number<2>{}))))
    {
        return;
    }

    const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[Number<0>{}]);
    const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[Number<1>{}]);

    using AScale        = typename GridwiseGemm::BlockwiseGemmPipe::Empty;
    auto a_scale_struct = AScale{};

    using BScale        = typename GridwiseGemm::BlockwiseGemmPipe::Empty;
    auto b_scale_struct = BScale{};

    const index_t num_k_block_per_scale = GridwiseGemm::GetKBlockPerScale();

    auto epilogue_args = Epilogue{};

    GridwiseGemm::Base::template Run<decltype(as_grid_desc_ak0_m_ak1),
                                     decltype(bs_grid_desc_bk0_n_bk1),
                                     decltype(ds_grid_desc),
                                     decltype(e_grid_desc),
                                     decltype(a_scale_struct),
                                     decltype(b_scale_struct),
                                     Epilogue,
                                     HasMainKBlockLoop,
                                     EGlobalMemoryDataOperation,
                                     TailNum>(p_as_grid_,
                                              p_bs_grid_,
                                              p_ds_grid_,
                                              gemm_arg.e_ptr_ + e_group_offset + e_n_offset,
                                              p_shared,
                                              as_grid_desc_ak0_m_ak1,
                                              bs_grid_desc_bk0_n_bk1,
                                              ds_grid_desc,
                                              e_grid_desc,
                                              gemm_arg.a_element_op_,
                                              gemm_arg.b_element_op_,
                                              gemm_arg.cde_element_op_,
                                              block_m_id,
                                              block_n_id,
                                              num_k_block_per_scale,
                                              a_scale_struct,
                                              b_scale_struct,
                                              epilogue_args);
#else
    ignore = gemm_desc_kernel_args;
    ignore = gemms_count;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
#endif
}

} // namespace

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename AComputeDataType                   = ADataType,
          typename BComputeDataType                   = AComputeDataType>
struct DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor
    : public DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                             ALayout,
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
    using DeviceOp = DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor;

    static constexpr index_t NumDTensor  = DsDataType::Size();
    static constexpr index_t MaxGemmsNum = 32;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I6 = Number<6>{};
    // K1 = Max Vector Access Pixels
    static constexpr auto K1Number = Number<K1>{};

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);
    static constexpr auto WmmaK  = 16;

    using ConvToGemmFwdTransformerIndexT = TransformConvFwdToGemm<NDimSpatial,
                                                                  ConvForwardSpecialization,
                                                                  true /*SplitN*/,
                                                                  ADataType,
                                                                  EDataType,
                                                                  1,
                                                                  index_t>;

    using ConvToGemmFwdTransformerLongIndexT = TransformConvFwdToGemm<NDimSpatial,
                                                                      ConvForwardSpecialization,
                                                                      true /*SplitN*/,
                                                                      ADataType,
                                                                      EDataType,
                                                                      1,
                                                                      long_index_t>;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    template <typename ALay>
    static auto MakeAGridDescriptor(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<ALay>();

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        const auto M = in_gemmm_gemmk_desc.GetLength(I0);
        const auto K = in_gemmm_gemmk_desc.GetLength(I1);
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        return transform_tensor_descriptor(
            in_gemmm_gemmk_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                       make_pass_through_transform(M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BLay>
    static auto MakeBGridDescriptor(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<BLay>();

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        const auto N = wei_gemmn_gemmk_desc.GetLength(I0);
        const auto K = wei_gemmn_gemmk_desc.GetLength(I1);
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        return transform_tensor_descriptor(
            wei_gemmn_gemmk_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename ELay>
    static auto
    MakeEGridDescriptor_M_N(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<ELay>();

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    static auto
    MakeDsGridDescriptor_M_N(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(conv_to_gemm_transformer);
            },
            Number<NumDTensor>{});
    }

    static auto CastDsPointers(const std::array<const void*, NumDTensor>& p_ds)
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                return static_cast<const DDataType*>(p_ds[i]);
            },
            Number<NumDTensor>{});
    }

    using DsPointer = decltype(CastDsPointers(std::array<const void*, NumDTensor>{}));

    using GemmAsDataType = Tuple<ADataType>;
    using GemmBsDataType = Tuple<BDataType>;
    using GemmDsDataType = DsDataType;

    using CDEBlockTransferScalarPerVectors =
        typename uniform_sequence_gen<NumDTensor + 1,
                                      CDEShuffleBlockTransferScalarPerVector_NPerBlock>::type;
    // desc for problem definition
    constexpr static ConvToGemmFwdTransformerIndexT dummy_conv_to_gemm_transformer;
    using AGridDesc = decltype(MakeAGridDescriptor<ALayout>(dummy_conv_to_gemm_transformer));
    using BGridDesc = decltype(MakeBGridDescriptor<BLayout>(dummy_conv_to_gemm_transformer));
    using DsGridDesc_M_N =
        remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(dummy_conv_to_gemm_transformer))>;
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;

    static auto
    GenerateConvToGemmTransforms(ConvToGemmFwdTransformerLongIndexT conv_to_gemm_transformer_base,
                                 const ADataType* a_grid_ptr_base,
                                 DsPointer ds_grid_ptr_base,
                                 EDataType* c_grid_ptr_base)
    {
        // Max number of splits
        // We need to use it to avoid infinity loop
        constexpr index_t max_split_numbers = MaxGemmsNum / 2;
        // Arrays to store transformers with smaller descs than 2GB
        Array<ConvToGemmFwdTransformerIndexT, MaxGemmsNum> conv_to_gemm_transformers_arr;
        Array<const ADataType*, MaxGemmsNum> a_grid_ptrs_arr;
        Array<DsPointer, MaxGemmsNum> ds_grid_ptrs_arr;
        Array<EDataType*, MaxGemmsNum> c_grid_ptrs_arr;
        // Queue for splitting
        std::queue<ConvToGemmFwdTransformerLongIndexT> conv_to_gemm_transformers_queue(
            {conv_to_gemm_transformer_base});
        std::queue<const ADataType*> a_grid_ptrs_queue({a_grid_ptr_base});
        std::queue<DsPointer> ds_grid_ptrs_queue({ds_grid_ptr_base});
        std::queue<EDataType*> c_grid_ptrs_queue({c_grid_ptr_base});

        index_t gemms_number  = 0;
        index_t split_numbers = 0;
        // Algorithm:
        // While queue is not empty:
        //  1. Get transformer from queue.
        //  2. If descs are smaller than 2GB push to result array.
        //  3. If descs are bigger than 2GB split into left and right transformer.
        while(!conv_to_gemm_transformers_queue.empty() && split_numbers < max_split_numbers &&
              gemms_number < MaxGemmsNum)
        {
            // Get transformer from the queue
            const auto& conv_to_gemm_transformer = conv_to_gemm_transformers_queue.front();
            const ADataType* a_grid_ptr          = a_grid_ptrs_queue.front();
            DsPointer ds_grid_ptr                = ds_grid_ptrs_queue.front();
            EDataType* c_grid_ptr                = c_grid_ptrs_queue.front();

            // Check if convolution not exceed 2GB
            if(conv_to_gemm_transformer.AreDescriptorsSmallerThan2GB())
            {
                // If yes, push into result array
                conv_to_gemm_transformers_arr(gemms_number) =
                    ConvToGemmFwdTransformerIndexT{conv_to_gemm_transformer};
                a_grid_ptrs_arr(gemms_number)  = a_grid_ptr;
                ds_grid_ptrs_arr(gemms_number) = ds_grid_ptr;
                c_grid_ptrs_arr(gemms_number)  = c_grid_ptr;
                gemms_number++;
            }
            else
            {
                // If no, split into left and right convolutions
                ConvToGemmFwdTransformerLongIndexT conv_to_gemm_transformers_left_part,
                    conv_to_gemm_transformers_right_part;
                const ADataType* a_grid_right_ptr;
                DsPointer ds_grid_right_ptr;
                EDataType* c_grid_right_ptr;

                ck::tie(conv_to_gemm_transformers_left_part,
                        conv_to_gemm_transformers_right_part,
                        a_grid_right_ptr,
                        ds_grid_right_ptr,
                        c_grid_right_ptr) =
                    conv_to_gemm_transformer.SplitConvProblem(a_grid_ptr, ds_grid_ptr, c_grid_ptr);

                conv_to_gemm_transformers_queue.push(conv_to_gemm_transformers_left_part);
                conv_to_gemm_transformers_queue.push(conv_to_gemm_transformers_right_part);
                // Left offsets remain the same
                a_grid_ptrs_queue.push(a_grid_ptr);
                a_grid_ptrs_queue.push(a_grid_right_ptr);
                ds_grid_ptrs_queue.push(ds_grid_ptr);
                ds_grid_ptrs_queue.push(ds_grid_right_ptr);
                c_grid_ptrs_queue.push(c_grid_ptr);
                c_grid_ptrs_queue.push(c_grid_right_ptr);
                split_numbers++;
            }
            // Remove from the queue
            conv_to_gemm_transformers_queue.pop();
            a_grid_ptrs_queue.pop();
            ds_grid_ptrs_queue.pop();
            c_grid_ptrs_queue.pop();
        }

        const bool is_split_valid = conv_to_gemm_transformers_queue.empty();

        return ck::make_tuple(conv_to_gemm_transformers_arr,
                              a_grid_ptrs_arr,
                              ds_grid_ptrs_arr,
                              c_grid_ptrs_arr,
                              gemms_number,
                              is_split_valid);
    }

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        DsLayout,
        tensor_layout::gemm::RowMajor,
        GemmAsDataType,
        GemmBsDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        K1,
        K1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        AComputeDataType,
        BComputeDataType,
        false,
        false,
        false,
        true>;

    // desc for blockwise copy
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}, 1, 1))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}, 1, 1))>;

    // Structure for each gemm(conv)
    struct GemmArgs
    {
        using AsDataType      = GemmAsDataType;
        using BsDataType      = GemmBsDataType;
        using DsDataTypeTuple = GemmDsDataType;

        static constexpr index_t NumATensor = GridwiseGemm::NumATensor;
        static constexpr index_t NumBTensor = GridwiseGemm::NumBTensor;
        static constexpr index_t NumDTensor = DeviceOp::NumDTensor;

        std::array<const void*, NumATensor> a_ptrs_{};
        std::array<const void*, NumBTensor> b_ptrs_{};
        std::array<const void*, NumDTensor> ds_ptrs_{};
        EDataType* e_ptr_ = nullptr;

        AElementwiseOperation a_element_op_{};
        BElementwiseOperation b_element_op_{};
        CDEElementwiseOperation cde_element_op_{};

        index_t M_ = 0;
        index_t N_ = 0;

        AGridDesc a_grid_desc_{};
        BGridDesc b_grid_desc_{};
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_{};
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_{};

        ck::index_t BlockStart_ = 0;
        ck::index_t BlockEnd_   = 0;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        template <typename DsGridDesc_M_N_>
        void init_gemm_args(const std::array<const void*, GridwiseGemm::NumATensor>& p_as_grid,
                            const std::array<const void*, GridwiseGemm::NumBTensor>& p_bs_grid,
                            const std::array<const void*, NumDTensor>& p_ds_grid,
                            EDataType* p_e_grid,
                            const AGridDesc& a_grid_desc,
                            const BGridDesc& b_grid_desc,
                            const DsGridDesc_M_N_& ds_grid_desc_m_n,
                            const EGridDesc_M_N& e_grid_desc_m_n,
                            index_t gemm_m,
                            index_t gemm_n,
                            index_t gemm_k,
                            index_t BlockStart,
                            index_t BlockEnd)
        {
            std::array<index_t, GridwiseGemm::NumATensor> stride_as{};
            std::array<index_t, GridwiseGemm::NumBTensor> stride_bs{};
            std::array<index_t, NumDTensor> stride_ds{};

            auto gemm_arg = typename GridwiseGemm::Argument{p_as_grid,
                                                            p_bs_grid,
                                                            p_ds_grid,
                                                            p_e_grid,
                                                            gemm_m,
                                                            gemm_n,
                                                            gemm_k,
                                                            stride_as,
                                                            stride_bs,
                                                            stride_ds,
                                                            index_t{0},
                                                            index_t{1},
                                                            a_element_op_,
                                                            b_element_op_,
                                                            cde_element_op_};

            if(GridwiseGemm::CheckValidity(gemm_arg, true))
            {
                const auto m_block = GridwiseGemm::CalculateMBlock(gemm_m);
                const auto n_block = GridwiseGemm::CalculateNBlock(gemm_n);

                GemmArgs new_args{};
                new_args.a_ptrs_  = p_as_grid;
                new_args.b_ptrs_  = p_bs_grid;
                new_args.ds_ptrs_ = p_ds_grid;
                new_args.e_ptr_   = p_e_grid;

                new_args.a_element_op_   = a_element_op_;
                new_args.b_element_op_   = b_element_op_;
                new_args.cde_element_op_ = cde_element_op_;

                new_args.M_ = gemm_m;
                new_args.N_ = gemm_n;

                new_args.a_grid_desc_ = a_grid_desc;
                new_args.b_grid_desc_ = b_grid_desc;
                new_args.ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n, m_block, n_block);
                new_args.e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeDEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n, m_block, n_block);

                new_args.BlockStart_ = BlockStart;
                new_args.BlockEnd_   = BlockEnd;

                gemm_desc_kernel_args_.At(valid_gemms_count_) = new_args;

                valid_gemms_count_++;
            }
        }
        Argument(const void* p_a,
                 const void* p_b,
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
                 const CDEElementwiseOperation& cde_element_op)
            : num_group_{static_cast<index_t>(a_g_n_c_wis_lengths[0])},
              compute_ptr_offset_of_groups_{},
              compute_ptr_offset_of_n_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // Perform grouped gemm, generate array of tranformer for convolution
            Array<ConvToGemmFwdTransformerIndexT, MaxGemmsNum> conv_to_gemm_transformer_arr;
            Array<const ADataType*, MaxGemmsNum> a_grid_ptrs;
            Array<DsPointer, MaxGemmsNum> ds_grid_ptrs;
            Array<EDataType*, MaxGemmsNum> c_grid_ptrs;

            DsPointer p_ds_casted = CastDsPointers(p_ds);

            ck::tie(conv_to_gemm_transformer_arr,
                    a_grid_ptrs,
                    ds_grid_ptrs,
                    c_grid_ptrs,
                    gemms_count_,
                    is_split_valid_) =
                GenerateConvToGemmTransforms(
                    ConvToGemmFwdTransformerLongIndexT{a_g_n_c_wis_lengths_,
                                                       a_g_n_c_wis_strides_,
                                                       b_g_k_c_xs_lengths_,
                                                       b_g_k_c_xs_strides_,
                                                       e_g_n_k_wos_lengths_,
                                                       e_g_n_k_wos_strides_,
                                                       conv_filter_strides_,
                                                       conv_filter_dilations_,
                                                       input_left_pads_,
                                                       input_right_pads_},
                    static_cast<const ADataType*>(p_a),
                    p_ds_casted,
                    static_cast<EDataType*>(p_e));

            grid_size_         = 0;
            valid_gemms_count_ = 0;

            if(is_split_valid_)
            {
                // Create GemmArg for each gemm(conv)
                for(index_t i = 0; i < gemms_count_; i++)
                {
                    const AGridDesc a_grid_desc{
                        DeviceOp::MakeAGridDescriptor<ALayout>(conv_to_gemm_transformer_arr[i])};
                    const BGridDesc b_grid_desc{
                        DeviceOp::MakeBGridDescriptor<BLayout>(conv_to_gemm_transformer_arr[i])};
                    const EGridDesc_M_N e_grid_desc_m_n{DeviceOp::MakeEGridDescriptor_M_N<ELayout>(
                        conv_to_gemm_transformer_arr[i])};

                    const auto ds_grid_desc_m_n =
                        DeviceOp::MakeDsGridDescriptor_M_N(conv_to_gemm_transformer_arr[i]);

                    const index_t GemmM = e_grid_desc_m_n.GetLength(I0);
                    const index_t GemmN = e_grid_desc_m_n.GetLength(I1);
                    const index_t GemmK = [&]() {
                        return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I2);
                    }();

                    std::array<const void*, GridwiseGemm::NumATensor> p_as_grid{};
                    p_as_grid[0] = static_cast<const void*>(a_grid_ptrs[i]);

                    std::array<const void*, GridwiseGemm::NumBTensor> p_bs_grid{};
                    p_bs_grid[0] = static_cast<const void*>(static_cast<const BDataType*>(p_b));

                    std::array<const void*, NumDTensor> p_ds_grid{};
                    if constexpr(NumDTensor > 0)
                    {
                        static_for<0, NumDTensor, 1>{}([&](auto d) {
                            p_ds_grid[d.value] = static_cast<const void*>(ds_grid_ptrs[i].At(d));
                        });
                    }

                    const index_t grid_size_grp =
                        GridwiseGemm::Block2CTileMap::CalculateGridSize(GemmM, GemmN);

                    const index_t BlockStart = grid_size_;
                    const index_t BlockEnd   = grid_size_ + grid_size_grp;

                    grid_size_ += grid_size_grp;

                    init_gemm_args(p_as_grid,
                                   p_bs_grid,
                                   p_ds_grid,
                                   c_grid_ptrs[i],
                                   a_grid_desc,
                                   b_grid_desc,
                                   ds_grid_desc_m_n,
                                   e_grid_desc_m_n,
                                   GemmM,
                                   GemmN,
                                   GemmK,
                                   BlockStart,
                                   BlockEnd);
                }

                // N is the same for all convs
                conv_N_per_block_ = static_cast<index_t>(conv_to_gemm_transformer_arr[I0].N_);
            }

            // Strides for G and N remain the same
            compute_ptr_offset_of_groups_.BatchStrideA_ = a_g_n_c_wis_strides_[0];
            compute_ptr_offset_of_groups_.BatchStrideB_ = b_g_k_c_xs_strides_[0];
            compute_ptr_offset_of_groups_.BatchStrideE_ = e_g_n_k_wos_strides_[0];

            compute_ptr_offset_of_n_.BatchStrideA_ = a_g_n_c_wis_strides_[1] * conv_N_per_block_;
            compute_ptr_offset_of_n_.BatchStrideE_ = e_g_n_k_wos_strides_[1] * conv_N_per_block_;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                compute_ptr_offset_of_groups_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_[i][0];
                compute_ptr_offset_of_n_.BatchStrideDs_(i) =
                    ds_g_n_k_wos_strides_[i][1] * conv_N_per_block_;
            });
        }

        void Print() const
        {
            std::cout << "===== Convolution summary =====" << std::endl;
            std::cout << "  num_group=" << num_group_
                      << ", conv_N_total=" << a_g_n_c_wis_lengths_[I1]
                      << ", conv_N_per_block=" << conv_N_per_block_ << std::endl;
            std::cout << "  gemms_count=" << gemms_count_
                      << ", valid_gemms_count=" << valid_gemms_count_
                      << ", is_split_valid=" << std::boolalpha << is_split_valid_
                      << std::noboolalpha << ", grid_size=" << grid_size_ << std::endl;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                std::cout << "    Ds[" << i.value
                          << "] group stride=" << compute_ptr_offset_of_groups_.BatchStrideDs_(i)
                          << ", n stride=" << compute_ptr_offset_of_n_.BatchStrideDs_(i)
                          << std::endl;
            });

            std::cout << "===== GEMM splits =====" << std::endl;
            for(index_t i = 0; i < valid_gemms_count_; ++i)
            {
                const auto& gemm = gemm_desc_kernel_args_[i];

                const auto M = gemm.e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I0) *
                               gemm.e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I1);
                const auto N = gemm.e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I2) *
                               gemm.e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I3);

                const auto K = [&]() {
                    return gemm.a_grid_desc_.GetLength(I0) * gemm.a_grid_desc_.GetLength(I2);
                }();

                std::cout << "  gemm[" << i << "] block_range=[" << gemm.BlockStart_ << ", "
                          << gemm.BlockEnd_ << ") (M,N,K)=(" << M << ", " << N << ", " << K
                          << ") grid_span=" << (gemm.BlockEnd_ - gemm.BlockStart_) << std::endl;
                std::cout << "    A descriptor: " << gemm.a_grid_desc_ << std::endl;
                std::cout << "    B descriptor: " << gemm.b_grid_desc_ << std::endl;
                std::cout << "    E[MBlock, MPerBlock, NBlock, NPerBlock]: "
                          << gemm.e_grid_desc_mblock_mperblock_nblock_nperblock_ << std::endl;

                static_for<0, NumDTensor, 1>{}([&](auto d_idx) {
                    std::cout << "    D" << d_idx.value << " descriptor: "
                              << gemm.ds_grid_desc_mblock_mperblock_nblock_nperblock_(d_idx)
                              << std::endl;
                });
            }
        }

        index_t num_group_;
        index_t conv_N_per_block_;

        Array<GemmArgs, MaxGemmsNum> gemm_desc_kernel_args_;

        index_t grid_size_;
        index_t gemms_count_;
        index_t valid_gemms_count_;

        bool is_split_valid_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_groups_;
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_n_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // for checking IsSupportedArgument()
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<long_index_t, NDimSpatial> conv_filter_strides_;
        std::array<long_index_t, NDimSpatial> conv_filter_dilations_;
        std::array<long_index_t, NDimSpatial> input_left_pads_;
        std::array<long_index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;
        template <typename Gridwise>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const index_t num_workgroups_per_Conv_N =
                arg.a_g_n_c_wis_lengths_[I1] / arg.conv_N_per_block_;

            const index_t gdx = arg.grid_size_;
            const index_t gdy = arg.num_group_;
            const index_t gdz = num_workgroups_per_Conv_N;

            const auto K = [&]() {
                return arg.gemm_desc_kernel_args_[I0].a_grid_desc_.GetLength(I0) *
                       arg.gemm_desc_kernel_args_[I0].a_grid_desc_.GetLength(I2);
            }();

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;
                const auto kernel = kernel_grouped_conv_fwd_grouped_gemm_wmma_cshuffle_v3<
                    Gridwise,
                    MaxGemmsNum,
                    GemmArgs,
                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                    has_main_loop,
                    InMemoryDataOperationEnum::Set>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(gdx, gdy, gdz),
                                              dim3(BlockSize),
                                              0,
                                              arg.gemm_desc_kernel_args_,
                                              arg.gemms_count_,
                                              arg.compute_ptr_offset_of_groups_,
                                              arg.compute_ptr_offset_of_n_);
            };

            if(Gridwise::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            return RunImp<GridwiseGemm>(arg, stream_config);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = tensor_layout::convolution;

        const long_index_t K = arg.b_g_k_c_xs_lengths_[I1];
        const long_index_t C = arg.b_g_k_c_xs_lengths_[I2];

        bool ds_valid = true;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            for(int d = 0; d < NDimSpatial + I3; d++)
            {
                if(arg.ds_g_n_k_wos_strides_[i][d] != arg.e_g_n_k_wos_strides_[d])
                {
                    ds_valid = false;
                }
                if(arg.ds_g_n_k_wos_lengths_[i][d] != arg.e_g_n_k_wos_lengths_[d])
                {
                    ds_valid = false;
                }
            }

            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            static_assert(is_same_v<DDataType, EDataType>);
        });

        if(!ds_valid)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Ds tensors must have the same dimensions as E tensor!" << " In "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // Check if all descs are valid
        if(!(arg.is_split_valid_ && arg.gemms_count_ == arg.valid_gemms_count_ &&
             arg.valid_gemms_count_ > 0))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "GEMM splits are not valid!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // check device
        if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Incorrect accumulator data type!" << " In " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "WMMA large tensor not supported on this device!" << " In " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "The input parameters are not valid for "
                                     "Filter1x1Stride1Pad0!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X        = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t LeftPad  = arg.input_left_pads_[i];
                const index_t RightPad = arg.input_right_pads_[i];

                if(!(X == 1 && LeftPad == 0 && RightPad == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "The input parameters are not valid for "
                                     "Filter1x1Pad0!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::G_NW_C> || is_same_v<ALayout, ctc::G_NHW_C> ||
                     is_same_v<ALayout, ctc::G_NDHW_C> || is_same_v<ALayout, ctc::GNWC> ||
                     is_same_v<ALayout, ctc::GNHWC> || is_same_v<ALayout, ctc::GNDHWC> ||
                     is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC>)
        {
            // Check access per C
            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Parameters for A Layout incorrect!" << " In " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported A Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(is_same_v<BLayout, ctc::G_K_X_C> || is_same_v<BLayout, ctc::G_K_YX_C> ||
                     is_same_v<BLayout, ctc::G_K_ZYX_C> || is_same_v<BLayout, ctc::GKXC> ||
                     is_same_v<BLayout, ctc::GKYXC> || is_same_v<BLayout, ctc::GKZYXC> ||
                     is_same_v<BLayout, ctc::KXGC> || is_same_v<BLayout, ctc::KYXGC> ||
                     is_same_v<BLayout, ctc::KZYXGC>)
        {
            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Parameters for B Layout incorrect!" << " In " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported B Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        //  check vector access of Ds
        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            // FIXME: layout
            if constexpr(is_same_v<DLayout, ctc::G_NW_K> || is_same_v<DLayout, ctc::G_NHW_K> ||
                         is_same_v<DLayout, ctc::G_NDHW_K> || is_same_v<DLayout, ctc::GNWK> ||
                         is_same_v<DLayout, ctc::GNHWK> || is_same_v<DLayout, ctc::GNDHWK> ||
                         is_same_v<DLayout, ctc::NWGK> || is_same_v<DLayout, ctc::NHWGK> ||
                         is_same_v<DLayout, ctc::NDHWGK> || is_same_v<DLayout, ctc::G_K>)
            {
                const index_t Kds = arg.ds_g_n_k_wos_lengths_[i][2];

                if(!(Kds % CDEShuffleBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "Parameters for D tensor Layout incorrect!" << " In "
                                  << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                                  << std::endl;
                    }
                    valid = false;
                }
            }
            else
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Unsupported D Layout!" << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                valid = false;
            }
        });

        if(!valid)
        {
            return false;
        }

        // check vector access of E
        if constexpr(!(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                       is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                       is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                       is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                       is_same_v<ELayout, ctc::NDHWGK>))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported E Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        if(!(K % CDEShuffleBlockTransferScalarPerVector_NPerBlock == 0))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Parameters for E Layout incorrect!" << " In " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_a,
        const void* p_b,
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
        const CDEElementwiseOperation& cde_element_op)
    {
        std::array<long_index_t, NDimSpatial + 3> a_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> a_strides_i64;
        std::array<long_index_t, NDimSpatial + 3> b_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> b_strides_i64;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_lengths_i64;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_strides_i64;
        std::array<long_index_t, NDimSpatial + 3> e_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> e_strides_i64;
        std::array<long_index_t, NDimSpatial> conv_strides_i64;
        std::array<long_index_t, NDimSpatial> conv_dilations_i64;
        std::array<long_index_t, NDimSpatial> left_pads_i64;
        std::array<long_index_t, NDimSpatial> right_pads_i64;

        array_convert(a_lengths_i64, a_g_n_c_wis_lengths);
        array_convert(a_strides_i64, a_g_n_c_wis_strides);
        array_convert(b_lengths_i64, b_g_k_c_xs_lengths);
        array_convert(b_strides_i64, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; ++d)
        {
            array_convert(ds_lengths_i64[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_strides_i64[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_lengths_i64, e_g_n_k_wos_lengths);
        array_convert(e_strides_i64, e_g_n_k_wos_strides);
        array_convert(conv_strides_i64, conv_filter_strides);
        array_convert(conv_dilations_i64, conv_filter_dilations);
        array_convert(left_pads_i64, input_left_pads);
        array_convert(right_pads_i64, input_right_pads);

        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_lengths_i64,
                        a_strides_i64,
                        b_lengths_i64,
                        b_strides_i64,
                        ds_lengths_i64,
                        ds_strides_i64,
                        e_lengths_i64,
                        e_strides_i64,
                        conv_strides_i64,
                        conv_dilations_i64,
                        left_pads_i64,
                        right_pads_i64,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto
    MakeArgument(const void* p_a,
                 const void* p_b,
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
                 const CDEElementwiseOperation& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
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
        const CDEElementwiseOperation& cde_element_op) override
    {
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_long{};
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_strides_long{};
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_long{};
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_long{};
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>
            ds_g_n_k_wos_lengths_long{};
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>
            ds_g_n_k_wos_strides_long{};
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_long{};
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_strides_long{};
        std::array<long_index_t, NDimSpatial> conv_filter_strides_long{};
        std::array<long_index_t, NDimSpatial> conv_filter_dilations_long{};
        std::array<long_index_t, NDimSpatial> input_left_pads_long{};
        std::array<long_index_t, NDimSpatial> input_right_pads_long{};

        array_convert(a_g_n_c_wis_lengths_long, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_long, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_long, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_long, b_g_k_c_xs_strides);

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            array_convert(ds_g_n_k_wos_lengths_long[i], ds_g_n_k_wos_lengths[i]);
            array_convert(ds_g_n_k_wos_strides_long[i], ds_g_n_k_wos_strides[i]);
        });

        array_convert(e_g_n_k_wos_lengths_long, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_long, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_long, conv_filter_strides);
        array_convert(conv_filter_dilations_long, conv_filter_dilations);
        array_convert(input_left_pads_long, input_left_pads);
        array_convert(input_right_pads_long, input_right_pads);

        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths_long,
                                          a_g_n_c_wis_strides_long,
                                          b_g_k_c_xs_lengths_long,
                                          b_g_k_c_xs_strides_long,
                                          ds_g_n_k_wos_lengths_long,
                                          ds_g_n_k_wos_strides_long,
                                          e_g_n_k_wos_lengths_long,
                                          e_g_n_k_wos_strides_long,
                                          conv_filter_strides_long,
                                          conv_filter_dilations_long,
                                          input_left_pads_long,
                                          input_right_pads_long,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
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
                        const CDEElementwiseOperation& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        std::stringstream ss;
        ss << "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor" << "<" << BlockSize
           << ", " << MPerBlock << ", " << NPerBlock << ", "
           << getConvForwardSpecializationString(ConvForwardSpecialization) << ", " << MPerWmma
           << ", " << NPerWmma << ", " << MRepeat << ", " << NRepeat << ", "
           << ABlockTransferSrcScalarPerVector << ", " << BBlockTransferSrcScalarPerVector << ", "
           << CDEShuffleBlockTransferScalarPerVector_NPerBlock << ", " << CShuffleMRepeatPerShuffle
           << ", " << CShuffleNRepeatPerShuffle << ">";
        return ss.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
