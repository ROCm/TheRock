// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AElementwiseOperation = ck::tensor_operation::element_wise::PassThrough,
          typename BElementwiseOperation = ck::tensor_operation::element_wise::PassThrough,
          typename CElementwiseOperation = ck::tensor_operation::element_wise::PassThrough>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_xdl_splitk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   const index_t group_count,
                                   const AElementwiseOperation a_element_op,
                                   const BElementwiseOperation b_element_op,
                                   const CElementwiseOperation c_element_op)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
        __shared__ uint8_t p_shared[shared_size];

        const index_t block_id   = get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const GemmDesc*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) / 2);
        while((!(block_id >= gemm_desc_ptr[group_id].block_start_ &&
                 block_id < gemm_desc_ptr[group_id].block_end_)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start_)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) / 2);
        }

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
            gemm_desc_ptr[group_id].karg_,
            static_cast<void*>(p_shared),
            gemm_desc_ptr[group_id].block_2_ctile_map_,
            a_element_op,
            b_element_op,
            c_element_op);
    }
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx9__))
}

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
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          // Current implementation does not support multiple D fusions.
          enable_if_t<AK1 == BK1 && is_same_v<DsLayout, ck::Tuple<>> &&
                          is_same_v<DsDataType, ck::Tuple<>>,
                      bool> = false>
struct DeviceGroupedGemmXdlSplitKCShuffle : public DeviceGroupedGemmSplitK<ALayout,
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
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static_assert(KPerBlock % AK1 == 0);
    static constexpr index_t K0PerBlock = KPerBlock / AK1;

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        EDataType,
        ALayout,
        BLayout,
        ELayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
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
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferScalarPerVector_NPerBlock,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        LoopSched,
        PipelineVer>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using CGridDesc_M_N = typename GridwiseGemm64::CGridDesc_M_N;
    using Block2ETileMapKSplit =
        BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>;
    // Block2CTileMap configuration parameter.
    static constexpr index_t B2E_M01 = 8;
    using GroupedGemmBlock2ETileMap  = OffsettedBlockToCTileMap<Block2ETileMapKSplit>;
    using KernelArgument             = typename GridwiseGemm64::Argument;
    using PassThrough                = ck::tensor_operation::element_wise::PassThrough;
    template <typename KernelArgument_>
    struct GemmTransKernelArgBase
    {
        KernelArgument_ karg_;
        GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        GemmTransKernelArgBase() = default;
        GemmTransKernelArgBase(KernelArgument_&& karg,
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
    using GemmTransKernelArg = GemmTransKernelArgBase<KernelArgument>;

    static constexpr index_t DefaultKBatch = 1;

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs)
            : Argument(p_As, p_Bs, p_Es, gemm_descs, DefaultKBatch)
        {
            // TODO: use occupancy api to calculate appropriate batch size.
        }

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 index_t kbatch)
            : K_BATCH{kbatch}, gemm_kernel_host_args_{nullptr}
        {
            grid_size_   = 0;
            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }

            gemm_kernel_args_.reserve(group_count_);

            skipped_group_count_ = 0;

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                if(M == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A_;
                const index_t stride_b = gemm_descs[i].stride_B_;
                const index_t stride_c = gemm_descs[i].stride_C_;

                const index_t m_padded  = GridwiseGemm64::CalculateMPadded(M);
                const index_t n_padded  = GridwiseGemm64::CalculateNPadded(N);
                const index_t k_padded  = GridwiseGemm64::CalculateKPadded(K, K_BATCH);
                const index_t k0_padded = GridwiseGemm64::CalculateK0Padded(K, K_BATCH);

                const auto c_grid_desc_m_n =
                    GridwiseGemm64::MakeCGridDescriptor_M_N(M, N, stride_c);

                const auto local_b2c_tile_map =
                    Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                auto karg = KernelArgument{type_convert<const ADataType*>(p_As[i]),
                                           type_convert<const BDataType*>(p_Bs[i]),
                                           type_convert<EDataType*>(p_Es[i]),
                                           M,
                                           N,
                                           K,
                                           stride_a,
                                           stride_b,
                                           stride_c,
                                           m_padded,
                                           n_padded,
                                           k_padded,
                                           k0_padded,
                                           K_BATCH};

                gemm_kernel_args_.emplace_back(
                    std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);
            }
        }

        /**
         * @brief      Recalculate group grid size for all gemms and update B2C maps.
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

                karg.KPadded                            = k_padded;
                karg.K0Padded                           = k0_padded;
                karg.k_batch                            = K_BATCH;
                gemm_kernel_args_[i].block_2_ctile_map_ = grouped_block_2_ctile_map;
                gemm_kernel_args_[i].block_start_       = block_start;
                gemm_kernel_args_[i].block_end_         = block_end;
            }
        }

        //  private:
        index_t K_BATCH;
        index_t group_count_;
        index_t skipped_group_count_;

        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        void* gemm_kernel_host_args_;
        index_t grid_size_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        template <typename GridwiseGemm>
        float RunImp(const Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{},
                     hipStream_t cpy_stream            = nullptr,
                     hipEvent_t cpy_event              = nullptr)
        {
            using GemmTransKernelArg_ = GemmTransKernelArgBase<typename GridwiseGemm::Argument>;
            static_assert(sizeof(GemmTransKernelArg_) == sizeof(GemmTransKernelArg));
            static_assert(sizeof(typename GridwiseGemm::Argument) ==
                          sizeof(typename GridwiseGemm64::Argument));

            index_t K0                       = arg.gemm_kernel_args_[0].karg_.K0Padded;
            bool all_have_kbatch_gt_one      = arg.gemm_kernel_args_[0].karg_.k_batch > 1;
            bool all_have_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
            {
                const auto& karg = reinterpret_cast<const typename GridwiseGemm::Argument&>(
                    arg.gemm_kernel_args_[i].karg_);
                if(stream_config.log_level_ > 0)
                {
                    karg.Print();
                }

                auto kbatch = karg.k_batch;

                if(!GridwiseGemm::CheckValidity(karg))
                {
                    std::ostringstream err;
                    err << "Group id: " << i << " has invalid GridwiseGemm settings!" << __FILE__
                        << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }

                K0 = karg.K0Padded;
                bool not_all_have_main_k0_block_loop_same =
                    all_have_main_k0_block_loop xor GridwiseGemm::CalculateHasMainK0BlockLoop(K0);
                bool not_all_have_kbatch_value_same = all_have_kbatch_gt_one xor (kbatch > 1);

                if(not_all_have_main_k0_block_loop_same)
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
                        << "], kbatch: " << kbatch
                        << ", group [0], kbatch: " << arg.gemm_kernel_args_[0].karg_.k_batch
                        << " in " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
            }

            // If the user provides copy stream and copy event, we assume that they're also
            // responsible for providing allocated host memory (eg. pinned) which
            // would be used to copy kernel arguments to the device.
            if(cpy_stream && cpy_event)
            {
                if(arg.gemm_kernel_host_args_ == nullptr)
                {
                    std::ostringstream err;
                    err << "No memory has been allocated for gemm kernel host args "
                        << "when providing the copy stream and copy event! In " << __FILE__ << ":"
                        << __LINE__ << ", in function: " << __func__;
                    throw std::runtime_error(err.str());
                }
                hip_check_error(hipMemcpyAsync(arg.p_workspace_,
                                               arg.gemm_kernel_host_args_,
                                               arg.group_count_ * sizeof(GemmTransKernelArg_),
                                               hipMemcpyHostToDevice,
                                               cpy_stream));
                hip_check_error(hipEventRecord(cpy_event, cpy_stream));
                hip_check_error(hipEventSynchronize(cpy_event));
            }
            else // In this case CK owns memory allocated on host.
            {

                hip_check_error(
                    hipMemcpyAsync(arg.p_workspace_,
                                   arg.gemm_kernel_args_.data(),
                                   arg.gemm_kernel_args_.size() * sizeof(GemmTransKernelArg_),
                                   hipMemcpyHostToDevice,
                                   stream_config.stream_id_));
            }

            float ave_time = 0;

            const auto Run = [&](const auto& kernel) {
                if(all_have_kbatch_gt_one)
                {
                    for(const auto& trans_arg : arg.gemm_kernel_args_)
                    {
                        const auto& karg = trans_arg.karg_;
                        hip_check_error(hipMemsetAsync(karg.p_c_grid,
                                                       0,
                                                       karg.M * karg.N * sizeof(EDataType),
                                                       stream_config.stream_id_));
                    }
                }

                ave_time =
                    launch_and_time_kernel(stream_config,
                                           kernel,
                                           dim3(arg.grid_size_),
                                           dim3(BlockSize),
                                           0,
                                           cast_pointer_to_constant_address_space(arg.p_workspace_),
                                           arg.gemm_kernel_args_.size(),
                                           PassThrough{},
                                           PassThrough{},
                                           PassThrough{});
            };

            if(all_have_main_k0_block_loop)
            {
                if(all_have_kbatch_gt_one)
                {
                    const auto kernel =
                        kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                       GemmTransKernelArg_,
                                                       true,
                                                       InMemoryDataOperationEnum::AtomicAdd>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel =
                        kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                       GemmTransKernelArg_,
                                                       true,
                                                       InMemoryDataOperationEnum::Set>;

                    Run(kernel);
                }
            }
            else
            {
                if(all_have_kbatch_gt_one)
                {
                    const auto kernel =
                        kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                       GemmTransKernelArg_,
                                                       false,
                                                       InMemoryDataOperationEnum::AtomicAdd>;

                    Run(kernel);
                }
                else
                {
                    const auto kernel =
                        kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                       GemmTransKernelArg_,
                                                       false,
                                                       InMemoryDataOperationEnum::Set>;

                    Run(kernel);
                }
            }

            return ave_time;
        }

        float Run(const Argument& arg,
                  const StreamConfig& stream_config = StreamConfig{},
                  hipStream_t cpy_stream            = nullptr,
                  hipEvent_t cpy_event              = nullptr)
        {
            if(get_warp_size() == 64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    return RunImp<GridwiseGemm64>(arg, stream_config, cpy_stream, cpy_event);
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    return RunImp<GridwiseGemm32>(arg, stream_config, cpy_stream, cpy_event);
                }
            }
            return 0;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
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
        if(is_gfx11_supported() && arg.K_BATCH > 1)
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

        if(std::is_same_v<EDataType, ck::bhalf_t> && arg.K_BATCH > 1 && !is_bf16_atomic_supported())
        {
            return false;
        }

        bool supported = true;
        bool isWave64  = get_warp_size() == 64;
        for(std::size_t i = 0; i < arg.gemm_kernel_args_.size(); ++i)
        {
            const auto& a = arg.gemm_kernel_args_[i].karg_;

            // Validate stride requirements for SplitK (k_batch > 1)
            // TODO: Enable splitK
            if(a.k_batch > 1)
            {
                if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
                {
                    if(a.StrideC != a.N)
                    {
                        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                        {
                            std::cout << "[" << __func__ << "] group id: " << i
                                      << " SplitK (k_batch=" << a.k_batch
                                      << ") requires contiguous output stride."
                                      << " For RowMajor layout: StrideC must equal N."
                                      << " Got StrideC=" << a.StrideC << ", N=" << a.N << std::endl;
                        }
                        return false;
                    }
                }
                else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
                {
                    if(a.StrideC != a.M)
                    {
                        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                        {
                            std::cout << "[" << __func__ << "] group id: " << i
                                      << " SplitK (k_batch=" << a.k_batch
                                      << ") requires contiguous output stride."
                                      << " For ColumnMajor layout: StrideC must equal M."
                                      << " Got StrideC=" << a.StrideC << ", M=" << a.M << std::endl;
                        }
                        return false;
                    }
                }
            }

            bool group_arg_valid = false;
            if(isWave64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    group_arg_valid = GridwiseGemm64::CheckValidity(a);
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    group_arg_valid = GridwiseGemm32::CheckValidity(
                        reinterpret_cast<const typename GridwiseGemm32::Argument&>(a));
                }
            }

            if(not group_arg_valid)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[" << __func__ << "] group id: " << i
                              << " has invalid GridwiseGemm settings!" << std::endl;
                    a.Print();
                }
            }
            supported = supported && group_arg_valid;
        }
        return supported;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>&,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CDEElementwiseOperation)
    {
        return Argument{p_As, p_Bs, p_Es, gemm_descs};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>&,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation) override
    {
        return std::make_unique<Argument>(p_As, p_Bs, p_Es, gemm_descs);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedGemm_XdlSplitK"
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
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto p_arg_ = dynamic_cast<const Argument*>(p_arg);
        if(p_arg_)
        {
            return p_arg_->gemm_kernel_args_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffle::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return GetWorkSpaceSize(p_arg);
    }

    size_t GetHostKernelArgSize(const BaseArgument* p_arg) const { return GetWorkSpaceSize(p_arg); }

    // TODO: deperecation notice.
    static void SetKBatchSize(Argument& arg, index_t kbatch) { arg.UpdateKBatch(kbatch); }

    // polymorphic
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
                "DeviceGroupedGemmMultipleDSplitKXdlCShuffle::Argument structure!");
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return this->SetWorkSpacePointer(p_arg, p_dev_kernel_args);
    }

    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the host kernel arguments pointer and copies that data on the host side.
    ///             This function can be utilised to use pinned memory for the host args and
    ///             achieve fully async data copy.
    ///
    /// @param      p_arg              The pointer to the Argument we're going to update.
    /// @param[in]  p_host_kernel_args The pointer to the host memory where the kernel
    ///                                arguments will be copied
    ///
    void SetHostKernelArgsPointer(BaseArgument* p_arg, void* p_host_kernel_args) const
    {
        Argument* pArg_ = dynamic_cast<Argument*>(p_arg);
        if(!pArg_)
        {
            throw std::runtime_error("Failed to cast argument pointer!");
        }

        pArg_->gemm_kernel_host_args_ = p_host_kernel_args;
        std::copy(pArg_->gemm_kernel_args_.begin(),
                  pArg_->gemm_kernel_args_.end(),
                  static_cast<GemmTransKernelArg*>(pArg_->gemm_kernel_host_args_));
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
