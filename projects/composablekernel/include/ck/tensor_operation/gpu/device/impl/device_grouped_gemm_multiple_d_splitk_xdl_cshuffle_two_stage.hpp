// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ios>
#include <iostream>
#include <sstream>
#include <tuple>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"
#include <ck/utility/loop_scheduler.hpp>
#include "ck/utility/tuple.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include <ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp>

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
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
          GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_KBatch_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_KBatch_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          typename ComputeDataType    = EDataType,
          // TODO: change gridwise_gemm_v2r4r2 to support AK1 & BK1
          enable_if_t<AK1 == BK1, bool> = false>
struct DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage
    : public DeviceGroupedGemmSplitK<ALayout,
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
    using DeviceOp = DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    // TODO change GridwiseGEMM v2r4r2 to support separate AK1 & BK1
    static constexpr index_t K0PerBlock = KPerBlock / AK1;

    using PassThrough       = ck::tensor_operation::element_wise::PassThrough;
    using WorkspaceDataType = float;

    // First stage GridwiseGEMM kernel.
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        WorkspaceDataType,
        ALayout,
        BLayout,
        ELayout,
        AElementwiseOperation,
        BElementwiseOperation,
        PassThrough, // CElementwiseOperation
        GemmSpec,
        NumGemmKPrefetchStage,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        AK1,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_KBatch_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_KBatch_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        LoopSched,
        PipelineVer,
        ComputeDataType>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    // Use gemm_padder for consistent descriptor creation
    static constexpr auto gemm_padder =
        tensor_operation::device::GemmPadder<GemmSpec, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, KPerBlock};

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(index_t M, index_t N, index_t StrideE)
    {
        const auto e_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideE));
            }
        }();

        // Use gemm_padder for consistent padding (same as C descriptor)
        return gemm_padder.PadCDescriptor_M_N(e_grid_desc_m_n);
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                                         const std::array<index_t, NumDTensor>& NRaws,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    static constexpr auto MakeElementwiseInputSequence()
    {
        return generate_sequence_v2(
            [&]([[maybe_unused]] auto i) constexpr {
                return Number<CDEShuffleBlockTransferScalarPerVector_NPerBlock>{};
            },
            Number<NumDTensor + 1>{});
    }

    using CGridDesc_M_N  = typename GridwiseGemm64::CGridDesc_M_N;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N<ELayout>(0, 0, 0));
    using DsGridDesc_M_N = decltype(MakeDsGridDescriptor_M_N({}, {}, {}));
    using DsGridPointer  = decltype(MakeDsGridPointer());
    using CDGridDesc_M_N = decltype(concat_tuple(ck::Tuple<CGridDesc_M_N>{}, DsGridDesc_M_N{}));
    using CDDataTypes    = decltype(concat_tuple(ck::Tuple<WorkspaceDataType*>{}, DsGridPointer{}));

    using ElementwiseInputSequence = decltype(MakeElementwiseInputSequence());

    static constexpr index_t ClusterLengthMPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    using Block2ETileMapKSplit =
        BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>;
    using Block2TileMap = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;
    using GridwiseElementwise =
        GridwiseElementwise<CDGridDesc_M_N,
                            ck::Tuple<EGridDesc_M_N>,
                            CDDataTypes,
                            ck::Tuple<EDataType*>,
                            Block2TileMap,
                            CDEElementwiseOperation,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<0, 1>,
                            ElementwiseInputSequence,
                            ck::Sequence<CDEShuffleBlockTransferScalarPerVector_NPerBlock>,
                            I1,
                            I1>;

    // Block2CTileMap configuration parameter.
    static constexpr index_t B2E_M01 = 8;
    using GroupedGemmBlock2ETileMap  = OffsettedBlockToCTileMap<Block2ETileMapKSplit>;
    using GemmKernelArgument         = typename GridwiseGemm64::Argument;

    struct GemmTransKernelArg
    {
        GemmKernelArgument karg_;
        GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        GemmTransKernelArg() = default;
        GemmTransKernelArg(GemmKernelArgument&& karg,
                           GroupedGemmBlock2ETileMap&& b2c_map,
                           index_t block_start,
                           index_t block_end)
            : karg_{karg},
              block_2_ctile_map_{b2c_map},
              block_start_{block_start},
              block_end_{block_end}
        {
        }
    };

    static constexpr index_t DefaultKBatch = 1;

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : Argument(p_As,
                       p_Bs,
                       p_Ds,
                       p_Es,
                       gemm_descs,
                       a_element_op,
                       b_element_op,
                       cde_element_op,
                       DefaultKBatch)
        {
        }

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op,
                 index_t kbatch)
            : K_BATCH{kbatch},
              group_count_{0},
              skipped_group_count_{0},
              grid_size_{0},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              p_Ds_{p_Ds}
        {
            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("Error! group_count_ != p_As/Bs/Ds/Es size");
            }

            gemm_kernel_args_.reserve(group_count_);
            elementwise_c_grid_descs_m_n_.reserve(group_count_);
            elementwise_e_grid_descs_m_n_.reserve(group_count_);
            elementwise_d_grid_descs_m_n_.reserve(group_count_);
            ds_grid_pointer_.reserve(group_count_);
            group_grid_size_.reserve(group_count_);
            e_ptrs_.reserve(group_count_);

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                if(M == 0 || N == 0 || K == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A_;
                const index_t stride_b = gemm_descs[i].stride_B_;

                const index_t m_padded  = GridwiseGemm64::CalculateMPadded(M);
                const index_t n_padded  = GridwiseGemm64::CalculateNPadded(N);
                const index_t k_padded  = GridwiseGemm64::CalculateKPadded(K, K_BATCH);
                const index_t k0_padded = GridwiseGemm64::CalculateK0Padded(K, K_BATCH);

                // Two different strides are needed for TwoStage split-K:
                // 1. workspace_stride (below): Stride for intermediate workspace (C grid)
                //    - Used by GEMM kernel to write workspace tiles
                // 2. gemm_descs[i].stride_C_: User-provided stride for final output (E tensor)
                //    - Used by elementwise kernel to write final results
                const index_t workspace_stride =
                    is_same<tensor_layout::gemm::RowMajor, ELayout>::value ? N : M;

                const auto c_grid_desc_m_n =
                    GridwiseGemm64::MakeCGridDescriptor_M_N(M, N, workspace_stride);

                DsGridDesc_M_N ds_grid_desc_m_n;
                DsGridPointer p_ds_grid;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    using DLayout   = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;
                    using DDataType = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;

                    p_ds_grid(j)        = static_cast<const DDataType*>(p_Ds[i][j]);
                    ds_grid_desc_m_n(j) = DeviceOp::MakeEGridDescriptor_M_N<DLayout>(
                        M, N, gemm_descs[i].stride_Ds_[j]);
                });
                const auto local_b2c_tile_map =
                    Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;
                group_grid_size_.push_back(grid_size_grp);
                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                std::array<index_t, NumDTensor> stride_ds;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    if(gemm_descs[i].stride_Ds_.size() != NumDTensor)
                    {
                        throw std::runtime_error(
                            "Error! gemm_descs[i].stride_Ds_.size() does not match NumDTensor");
                    }

                    stride_ds[j] = gemm_descs[i].stride_Ds_[j];
                });
                stride_Ds_.emplace_back(std::move(stride_ds));

                // We first set E pointer to actual operation output, but later on
                // when workspace will be set, this will be updated to workspace memory.
                auto karg = GemmKernelArgument{type_convert<const ADataType*>(p_As[i]),
                                               type_convert<const BDataType*>(p_Bs[i]),
                                               type_convert<WorkspaceDataType*>(p_Es[i]),
                                               M,
                                               N,
                                               K,
                                               stride_a,
                                               stride_b,
                                               workspace_stride,
                                               m_padded,
                                               n_padded,
                                               k_padded,
                                               k0_padded,
                                               K_BATCH};

                gemm_kernel_args_.emplace_back(
                    std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);

                // Create E grid descriptor with user-provided stride (not workspace stride)
                const auto e_grid_desc_m_n =
                    DeviceOp::MakeEGridDescriptor_M_N<ELayout>(M, N, gemm_descs[i].stride_C_);

                elementwise_c_grid_descs_m_n_.push_back(c_grid_desc_m_n);
                elementwise_e_grid_descs_m_n_.push_back(e_grid_desc_m_n);
                elementwise_d_grid_descs_m_n_.push_back(ds_grid_desc_m_n);
                ds_grid_pointer_.push_back(p_ds_grid);
                // Store a copy of E pointers for elementwise kernel destination
                e_ptrs_.push_back(p_Es[i]);
            }
        }

        /**
         * @brief      Set new kbatch value.
         *
         * @param[in]  kbatch  The new splitK parameter value.
         */
        void UpdateKBatch(index_t kbatch)
        {
            K_BATCH    = kbatch;
            grid_size_ = 0;

            for(std::size_t i = 0; i < gemm_kernel_args_.size(); ++i)
            {
                auto& karg = gemm_kernel_args_[i].karg_;

                const index_t k_padded  = GridwiseGemm64::CalculateKPadded(karg.K, K_BATCH);
                const index_t k0_padded = GridwiseGemm64::CalculateK0Padded(karg.K, K_BATCH);

                const auto c_grid_desc_m_n =
                    GridwiseGemm64::MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC);

                const auto local_b2c_tile_map =
                    Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                group_grid_size_[i]                     = grid_size_grp;
                karg.KPadded                            = k_padded;
                karg.K0Padded                           = k0_padded;
                karg.k_batch                            = K_BATCH;
                gemm_kernel_args_[i].block_2_ctile_map_ = grouped_block_2_ctile_map;
                gemm_kernel_args_[i].block_start_       = block_start;
                gemm_kernel_args_[i].block_end_         = block_end;

                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    index_t tiles = (block_end - block_start) / K_BATCH;
                    std::cout << "block_start: " << block_start << "\n"
                              << "block_end: " << block_end << "\n"
                              << "tiles: " << tiles << std::endl
                              << std::endl;

                    std::cout << "KPadded: " << karg.KPadded << std::endl
                              << "K0Padded: " << karg.K0Padded << std::endl
                              << "KBatch: " << karg.k_batch << std::endl
                              << "grid_size_: " << karg.KPadded << std::endl;
                }
            }
        }

        void UpdateEPointers()
        {
            // set-up each group E pointer to it's designated workspace memory.
            WorkspaceDataType* p_workspace = reinterpret_cast<WorkspaceDataType*>(p_workspace_);
            std::size_t offset             = 0;

            for(auto& arg : gemm_kernel_args_)
            {
                arg.karg_.p_c_grid = p_workspace + offset;
                index_t tiles      = (arg.block_end_ - arg.block_start_) / arg.karg_.k_batch;
                offset += tiles * MPerBlock * NPerBlock;
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "block_start: " << arg.block_start_ << "\n"
                              << "block_end: " << arg.block_end_ << "\n"
                              << "tiles: " << tiles << "\n"
                              << "offset: " << offset << std::endl;
                }
            }
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            std::size_t size_bytes{0};

            for(const auto& arg : gemm_kernel_args_)
            {
                index_t tiles = (arg.block_end_ - arg.block_start_) / arg.karg_.k_batch;
                size_bytes += tiles * MPerBlock * NPerBlock * sizeof(WorkspaceDataType);
            }
            return size_bytes;
        }

        std::size_t GetWorkspaceSize(std::size_t group) const
        {
            const auto& arg = gemm_kernel_args_[group];
            index_t tiles   = (arg.block_end_ - arg.block_start_) / arg.karg_.k_batch;
            return tiles * MPerBlock * NPerBlock;
        }

        //  private:
        index_t K_BATCH;
        index_t group_count_;
        index_t skipped_group_count_;
        index_t grid_size_;
        // Pointer to device memory with GEMM kernel arguments.
        void* p_dev_gemm_kargs_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        std::vector<std::array<const void*, NumDTensor>>& p_Ds_;
        std::vector<std::array<index_t, NumDTensor>> stride_Ds_;
        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        std::vector<index_t> group_grid_size_;

        std::vector<CGridDesc_M_N> elementwise_c_grid_descs_m_n_;
        std::vector<EGridDesc_M_N> elementwise_e_grid_descs_m_n_;
        std::vector<DsGridDesc_M_N> elementwise_d_grid_descs_m_n_;
        std::vector<DsGridPointer> ds_grid_pointer_;
        std::vector<void*> e_ptrs_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using user provided device buffer for kernel
        ///             arguments.
        ///
        /// @param[in]  arg                 The structure containing kernel arguments (in host
        ///                                 memory).
        /// @param[in]  dev_gemm_args       The pointer to device memory with kernel arguments.
        /// @param[in]  dev_gemm_workspace  The pointer to device memory for kernel auxiliary
        ///                                 workspace.
        /// @param[in]  stream_config       The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        template <typename GridwiseGemm>
        float Run(const Argument& arg,
                  void* dev_gemm_args,
                  void* dev_gemm_workspace,
                  const StreamConfig& stream_config = StreamConfig{})
        {
            auto [all_have_kbatch_gt_one, all_have_main_k_block_loop] =
                CheckArgument<GridwiseGemm>(arg, stream_config);

            if(dev_gemm_args == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!" << " In " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            if(dev_gemm_workspace == nullptr)
            {
                std::ostringstream err;
                err << "The gemm workspace buffer is not allocated!" << " In " << __FILE__ << ":"
                    << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            float ave_time = 0;

            if(all_have_main_k_block_loop)
            {
                ave_time = DispatchKernel<GridwiseGemm, true>(
                    arg, dev_gemm_args, dev_gemm_workspace, stream_config);
            }
            else
            {
                ave_time = DispatchKernel<GridwiseGemm, false>(
                    arg, dev_gemm_args, dev_gemm_workspace, stream_config);
            }

            return ave_time;
        }

        ///
        /// @brief      Launch Grouped Gemm kernel.
        ///
        /// @note       This function overload is using device buffers (for kernel arguments and
        ///             for kernel auxiliary workspace) provided with an argument. The user should
        ///             call @see GetDeviceKernelArgSize, @see GetWorkSpaceSize and @see
        ///             SetDeviceKernelArgs, @see SetWorkSpacePointer on arg parameter to properly
        ///             allocate those buffers.
        ///
        /// @param[in]  arg            The structure containing kernel arguments (in host memory).
        /// @param[in]  stream_config  The device stream configuration.
        ///
        /// @return     The average kernel execution time (if time measurement is enabled.)
        ///
        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.p_dev_gemm_kargs_ == nullptr)
            {
                std::ostringstream err;
                err << "The gemm arguments device buffer is not allocated!" << " In " << __FILE__
                    << ":" << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            if(arg.p_workspace_ == nullptr)
            {
                std::ostringstream err;
                err << "The gemm workspace buffer is not allocated!" << " In " << __FILE__ << ":"
                    << __LINE__ << ", in function: " << __func__;
                throw std::runtime_error(err.str());
            }

            return Run<GridwiseGemm>(arg, arg.p_dev_gemm_kargs_, arg.p_workspace_, stream_config);
        }

        INVOKER_RUN_IMPL

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }

        private:
        template <typename GridwiseGemm>
        auto CheckArgument(const Argument& arg, const StreamConfig& stream_config) const
        {
            bool all_have_kbatch_gt_one, all_have_main_k_block_loop;

            {
                const auto a_grid_desc_kbatch_ak0_m_ak1 =
                    GridwiseGemm::MakeAGridDescriptor_KBatch_K0_M_K1(
                        arg.gemm_kernel_args_[0].karg_.M,
                        arg.gemm_kernel_args_[0].karg_.MPadded,
                        arg.gemm_kernel_args_[0].karg_.K,
                        arg.gemm_kernel_args_[0].karg_.StrideA,
                        arg.gemm_kernel_args_[0].karg_.k_batch,
                        arg.gemm_kernel_args_[0].karg_.K0Padded,
                        arg.gemm_kernel_args_[0].karg_.KPadded);

                all_have_kbatch_gt_one     = arg.K_BATCH > 1;
                all_have_main_k_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(
                    a_grid_desc_kbatch_ak0_m_ak1.GetLength(I1));
            }

            for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
            {
                const auto& gemm_arg = reinterpret_cast<const typename GridwiseGemm::Argument&>(
                    arg.gemm_kernel_args_[i].karg_);
                if(stream_config.log_level_ > 0)
                {
                    gemm_arg.Print();
                }

                if(!GridwiseGemm::CheckValidity(gemm_arg))
                {
                    std::ostringstream err;
                    err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                const auto a_grid_desc_kbatch_ak0_m_ak1 =
                    GridwiseGemm::MakeAGridDescriptor_KBatch_K0_M_K1(gemm_arg.M,
                                                                     gemm_arg.MPadded,
                                                                     gemm_arg.K,
                                                                     gemm_arg.StrideA,
                                                                     gemm_arg.k_batch,
                                                                     gemm_arg.K0Padded,
                                                                     gemm_arg.KPadded);

                bool not_all_have_main_k_block_loop_same =
                    all_have_main_k_block_loop xor GridwiseGemm::CalculateHasMainK0BlockLoop(
                                                       a_grid_desc_kbatch_ak0_m_ak1.GetLength(I1));
                bool not_all_have_kbatch_value_same =
                    all_have_kbatch_gt_one xor (gemm_arg.k_batch > 1);

                if(not_all_have_main_k_block_loop_same)
                {
                    std::ostringstream err;
                    err << "Not all gemms have same value for main_k0_block_loop! in " << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                if(not_all_have_kbatch_value_same)
                {
                    std::ostringstream err;
                    err << "Not all gemms have same kbatch value (=1 or >1)! " << "group [" << i
                        << "], kbatch: " << gemm_arg.k_batch
                        << ", group [0], kbatch: " << gemm_arg.k_batch << " in " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
            }
            return std::make_tuple(all_have_kbatch_gt_one, all_have_main_k_block_loop);
        }

        template <typename GridwiseGemm, bool HasMainKBlockLoop>
        float DispatchKernel(const Argument& arg,
                             void* dev_gemm_kargs,
                             void* dev_gemm_workspace,
                             const StreamConfig& stream_config) const
        {
            const auto gemm_kernel =
                kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                               GemmTransKernelArg,
                                               HasMainKBlockLoop,
                                               InMemoryDataOperationEnum::AtomicAdd,
                                               AElementwiseOperation,
                                               BElementwiseOperation,
                                               PassThrough>;

            const auto elementwise_kernel = kernel_elementwise<GridwiseElementwise,
                                                               CDGridDesc_M_N,
                                                               ck::Tuple<EGridDesc_M_N>,
                                                               CDDataTypes,
                                                               ck::Tuple<EDataType*>,
                                                               Block2TileMap,
                                                               CDEElementwiseOperation>;
            return LaunchKernel(gemm_kernel,
                                elementwise_kernel,
                                arg,
                                dev_gemm_kargs,
                                dev_gemm_workspace,
                                stream_config);
        }

        template <typename KernelFunction, typename KernelFunction2>
        float LaunchKernel(const KernelFunction& gemm_kernel,
                           const KernelFunction2& elementwise_kernel,
                           const Argument& arg,
                           void* dev_gemm_kargs,
                           [[maybe_unused]] void* dev_gemm_workspace,
                           const StreamConfig& stream_config) const
        {
            float time{0.f};

            hip_check_error(
                hipMemcpyAsync(dev_gemm_kargs,
                               arg.gemm_kernel_args_.data(),
                               arg.gemm_kernel_args_.size() * sizeof(GemmTransKernelArg),
                               hipMemcpyHostToDevice,
                               stream_config.stream_id_));

            auto preprocess = [&]() {
                hip_check_error(hipMemsetAsync(
                    dev_gemm_workspace, 0, arg.GetWorkspaceSizeBytes(), stream_config.stream_id_));
            };

            // GEMM kernel
            time = launch_and_time_kernel_with_preprocess(
                stream_config,
                preprocess,
                gemm_kernel,
                dim3(arg.grid_size_),
                dim3(BlockSize),
                0,
                cast_pointer_to_constant_address_space(dev_gemm_kargs),
                arg.gemm_kernel_args_.size(),
                arg.a_element_op_,
                arg.b_element_op_,
                PassThrough{});

            // Elementwise kernels
            for(size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
            {
                time += launch_and_time_kernel(
                    stream_config,
                    elementwise_kernel,
                    dim3(arg.group_grid_size_[i]),
                    dim3(BlockSize),
                    0,
                    concat_tuple(make_tuple(arg.elementwise_c_grid_descs_m_n_[i]),
                                 arg.elementwise_d_grid_descs_m_n_[i]),
                    make_tuple(arg.elementwise_e_grid_descs_m_n_[i]),
                    concat_tuple(make_tuple(arg.gemm_kernel_args_[i].karg_.p_c_grid),
                                 arg.ds_grid_pointer_[i]),
                    type_convert<EDataType*>(arg.e_ptrs_[i]),
                    Block2TileMap{arg.elementwise_c_grid_descs_m_n_[i].GetLength(I0),
                                  arg.elementwise_c_grid_descs_m_n_[i].GetLength(I1)},
                    arg.cde_element_op_);
            }
            return time;
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_wmma_supported<ADataType, BDataType, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if((ck::type_convert<ck::index_t>(arg.gemm_kernel_args_.size()) +
            arg.skipped_group_count_) != arg.group_count_)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The group count is not equal to sum of skipped groups "
                             "and kernel args size!"
                          << std::endl;
            }
            return false;
        }

        // Check if all groups have compatible HasMainLoop values
        if(!arg.gemm_kernel_args_.empty())
        {
            const auto& first_arg = arg.gemm_kernel_args_[0].karg_;
            const auto first_desc =
                GridwiseGemm64::MakeAGridDescriptor_KBatch_K0_M_K1(first_arg.M,
                                                                   first_arg.MPadded,
                                                                   first_arg.K,
                                                                   first_arg.StrideA,
                                                                   first_arg.k_batch,
                                                                   first_arg.K0Padded,
                                                                   first_arg.KPadded);
            const bool first_has_main_loop =
                GridwiseGemm64::CalculateHasMainK0BlockLoop(first_desc.GetLength(I1));

            for(std::size_t i = 1; i < arg.gemm_kernel_args_.size(); ++i)
            {
                const auto& gemm_arg = arg.gemm_kernel_args_[i].karg_;
                const auto desc =
                    GridwiseGemm64::MakeAGridDescriptor_KBatch_K0_M_K1(gemm_arg.M,
                                                                       gemm_arg.MPadded,
                                                                       gemm_arg.K,
                                                                       gemm_arg.StrideA,
                                                                       gemm_arg.k_batch,
                                                                       gemm_arg.K0Padded,
                                                                       gemm_arg.KPadded);
                const bool has_main_loop =
                    GridwiseGemm64::CalculateHasMainK0BlockLoop(desc.GetLength(I1));

                if(first_has_main_loop != has_main_loop)
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << std::boolalpha
                                  << "Not all groups have compatible HasMainLoop values! "
                                  << "Group 0: " << first_has_main_loop << ", Group " << i << ": "
                                  << has_main_loop << std::endl;
                    }
                    return false;
                }
            }
        }

        bool supported = true;
        bool isWave64  = get_warp_size() == 64;
        for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
        {
            const auto& gemm_arg = arg.gemm_kernel_args_[i].karg_;
            bool group_arg_valid = false;
            if(isWave64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    group_arg_valid = GridwiseGemm64::CheckValidity(gemm_arg);
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    group_arg_valid = GridwiseGemm32::CheckValidity(
                        reinterpret_cast<const typename GridwiseGemm32::Argument&>(gemm_arg));
                }
            }

            if(not group_arg_valid)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[" << __func__ << "] group id: " << i
                              << " has invalid GridwiseGemm settings!" << std::endl;
                    gemm_arg.Print();
                }
            }
            supported = supported && group_arg_valid;
        }
        return supported;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation a_elementwise_op,
                             BElementwiseOperation b_elementwise_op,
                             CDEElementwiseOperation cde_elementwise_op)
    {
        return Argument{p_As,
                        p_Bs,
                        p_Ds,
                        p_Es,
                        gemm_descs,
                        a_elementwise_op,
                        b_elementwise_op,
                        cde_elementwise_op};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_elementwise_op,
                        BElementwiseOperation b_elementwise_op,
                        CDEElementwiseOperation cde_elementwise_op) override
    {
        return std::make_unique<Argument>(p_As,
                                          p_Bs,
                                          p_Ds,
                                          p_Es,
                                          gemm_descs,
                                          a_elementwise_op,
                                          b_elementwise_op,
                                          cde_elementwise_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage"
            << "<"
            << std::string(ALayout::name)[0] << ","
            << std::string(BLayout::name)[0] << ","
            << std::string(ELayout::name)[0] << ","
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << ">";
        // clang-format on

        return str.str();
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->p_dev_gemm_kargs_ = p_dev_kernel_args;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->gemm_kernel_args_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage::Argument structure!");
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
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage::Argument structure!");
    }

    void SetWorkSpacePointer(
        BaseArgument* p_arg,
        void* p_workspace,
        [[maybe_unused]] const StreamConfig& stream_config = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
            p_arg_->UpdateEPointers();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage::Argument structure!");
    }

    [[deprecated]] static void SetKBatchSize(Argument& arg, index_t kbatch)
    {
        arg.UpdateKBatch(kbatch);
    }

    void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->UpdateKBatch(kbatch);
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
