// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <memory>
#include <array>
#include <stdexcept>

#include "ck/utility/common_header.hpp"
#include "ck/ck.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_threadwise_multi_d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
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
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ReduceDataType                     = CDataType,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct DeviceGemm_Wmma_CShuffleV3R1 : public DeviceGemmV2R1<ALayout,
                                                            BLayout,
                                                            DsLayout,
                                                            CLayout,
                                                            ADataType,
                                                            BDataType,
                                                            DsDataType,
                                                            CDataType,
                                                            AElementwiseOperation,
                                                            BElementwiseOperation,
                                                            CElementwiseOperation>
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        Tuple<>,
        CLayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        GemmAccDataType,
        ReduceDataType,
        Tuple<>,
        ReduceDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        PassThrough,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        false,
        false>;

    struct Argument : public GridwiseGemm::Argument
    {
        Argument(std::array<const void*, 1> p_a_grid_,
                 std::array<const void*, 1> p_b_grid_,
                 const ::std::array<const void*, NumDTensor> p_ds_,
                 CDataType* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 std::array<index_t, 1> StrideA_,
                 std::array<index_t, 1> StrideB_,
                 const ::std::array<index_t, NumDTensor> stride_ds_,
                 index_t StrideC_,
                 index_t KBatch_,
                 AElementwiseOperation a_element_op_,
                 BElementwiseOperation b_element_op_,
                 CElementwiseOperation c_element_op_)
            : GridwiseGemm::Argument(p_a_grid_,
                                     p_b_grid_,
                                     ::std::array<const void*, 0>{},
                                     reinterpret_cast<ReduceDataType*>(p_c_grid_),
                                     M_,
                                     N_,
                                     K_,
                                     StrideA_,
                                     StrideB_,
                                     std::array<index_t, 0>{},
                                     StrideC_,
                                     KBatch_,
                                     a_element_op_,
                                     b_element_op_,
                                     PassThrough{},
                                     true),
              p_c_grid(p_c_grid_),
              c_element_op(c_element_op_),
              p_ds(p_ds_),
              StrideDs(stride_ds_)
        {
        }

        CDataType* p_c_grid;
        CElementwiseOperation c_element_op;
        const ::std::array<const void*, NumDTensor> p_ds;
        ::std::array<index_t, NumDTensor> StrideDs;
    };

    using ReduceAdd               = ck::reduce::Add;
    using OutElementwiseOperation = CElementwiseOperation;

    static constexpr auto DsVectorLengthSequence = generate_sequence_v2(
        [](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
            if constexpr(is_same<CLayout, DLayout>::value)
                return Number<CShuffleBlockTransferScalarPerVector_NPerBlock>{};
            else
                return Number<1>{};
        },
        Number<NumDTensor>{});

    using DeviceReduceInstance = DeviceReduceThreadWiseMultiD<
        ReduceDataType,  // InDataType
        DsDataType,      // DsDatatype
        GemmAccDataType, // AccDataType
        CDataType,       // OutDataType
        3,               // Rank
        1,               // NumReduceDim
        ReduceAdd,
        PassThrough,
        OutElementwiseOperation,
        256,                                            // BlockSize_
        CShuffleBlockTransferScalarPerVector_NPerBlock, // MThreadSliceSize_
        1,                                              // KThreadSliceSize_
        0,                                              // InSrcVectorDim_
        CShuffleBlockTransferScalarPerVector_NPerBlock, // InSrcVectorSize_
        CShuffleBlockTransferScalarPerVector_NPerBlock, // OutDstVectorSize_
        decltype(DsVectorLengthSequence)>;

    struct Invoker : public BaseInvoker
    {
        float RunReduce(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            static constexpr index_t NumInDim  = 3;
            static constexpr index_t NumOutDim = 2;

            ::std::array<index_t, NumInDim> in_lengths   = {arg.KBatch, arg.M, arg.N};
            ::std::array<index_t, NumOutDim> out_lengths = {arg.M, arg.N};

            ::std::array<index_t, NumInDim> in_strides;
            ::std::array<index_t, NumOutDim> out_strides;
            if constexpr(is_same<CLayout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                in_strides  = {arg.M * arg.N, arg.N, 1};
                out_strides = {arg.N, 1};
            }
            else
            {
                in_strides  = {arg.M * arg.N, 1, arg.M};
                out_strides = {1, arg.M};
            }

            ::std::array<int, 1> reduce_dims{0};

            ::std::array<::std::array<index_t, NumOutDim>, NumDTensor> DsLengths;
            ::std::array<::std::array<index_t, NumOutDim>, NumDTensor> DsStrides;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                DsLengths[i] = out_lengths;

                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                if constexpr(is_same<DLayout, ck::tensor_layout::gemm::RowMajor>::value)
                {
                    DsStrides[i] = {arg.StrideDs[i], 1};
                }
                else
                {
                    DsStrides[i] = {1, arg.StrideDs[i]};
                }
            });

            auto reduce = DeviceReduceInstance{};

            auto argument_ptr = reduce.MakeArgumentPointer(in_lengths,
                                                           in_strides,
                                                           DsLengths,
                                                           DsStrides,
                                                           out_lengths,
                                                           out_strides,
                                                           reduce_dims,
                                                           arg.p_workspace_,
                                                           arg.p_ds,
                                                           arg.p_c_grid,
                                                           PassThrough{},
                                                           OutElementwiseOperation{});

            auto invoker_ptr = reduce.MakeInvokerPointer();

            float ave_time = 0;

            if(reduce.IsSupportedArgument(argument_ptr.get()))
            {
                ave_time = invoker_ptr->Run(argument_ptr.get(), stream_config);
            }
            else
            {
                throw ::std::runtime_error(
                    "The runtime parameters are not supported by the device instance.");
            }

            return ave_time;
        }

        float Run(const Argument& arg_, const StreamConfig& stream_config = StreamConfig{})
        {
            auto arg = *dynamic_cast<const typename GridwiseGemm::Argument*>(&arg_);

            // workspace required when doing two-kernel reduce or Ds present
            const bool need_workspace = !(!(arg.IsReduceAdd() || NumDTensor > 0) &&
                                          is_same<CDataType, ReduceDataType>::value);
            if(need_workspace)
            {
                if(arg.p_workspace_ == nullptr)
                {
                    throw ::std::runtime_error("using reduce, but empty workspace!");
                }
                arg.p_e_grid = reinterpret_cast<ReduceDataType*>(arg.p_workspace_);
            }

            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw ::std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            ::std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(has_main_k_block_loop)
            {
                const auto kernel =
                    ::ck::kernel_gemm_wmma_cshuffle_v3<GridwiseGemm,
                                                       true,
                                                       InMemoryDataOperationEnum::Set,
                                                       minimum_occupancy>;
                ave_time = launch_and_time_kernel(
                    stream_config, kernel, ::dim3(gdx, gdy, gdz), ::dim3(BlockSize), 0, arg);
            }
            else
            {
                const auto kernel =
                    ::ck::kernel_gemm_wmma_cshuffle_v3<GridwiseGemm,
                                                       false,
                                                       InMemoryDataOperationEnum::Set,
                                                       minimum_occupancy>;
                ave_time = launch_and_time_kernel(
                    stream_config, kernel, ::dim3(gdx, gdy, gdz), ::dim3(BlockSize), 0, arg);
            }

            if(need_workspace)
            {
                ave_time += RunReduce(arg_, stream_config);
            }

            return ave_time;
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
        // TODO: properly implement this
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_wmma_supported())
        {
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(
            *dynamic_cast<const typename GridwiseGemm::Argument*>(&arg));
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto CalculateGridSize(index_t M, index_t N, index_t KBatch)
    {
        return GridwiseGemm::CalculateGridSize(M, N, KBatch);
    }

    static constexpr index_t GetBlockSize() { return BlockSize; }

    static size_t GetSharedMemoryNumberOfByte()
    {
        return GridwiseGemm::GetSharedMemoryNumberOfByte();
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             const ::std::array<const void*, NumDTensor> p_ds,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             const ::std::array<index_t, NumDTensor> stride_ds,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{std::array<const void*, 1>{p_a},
                        std::array<const void*, 1>{p_b},
                        p_ds,
                        p_c,
                        M,
                        N,
                        K,
                        std::array<index_t, 1>{StrideA},
                        std::array<index_t, 1>{StrideB},
                        stride_ds,
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    ::std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return ::std::make_unique<Invoker>(Invoker{});
    }

    // Polymorphic interfaces
    ::std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                        const void* p_b,
                                                        ::std::array<const void*, NumDTensor> p_ds,
                                                        void* p_c,
                                                        index_t M,
                                                        index_t N,
                                                        index_t K,
                                                        index_t StrideA,
                                                        index_t StrideB,
                                                        ::std::array<index_t, NumDTensor> DsStrides,
                                                        index_t StrideC,
                                                        index_t KSplit,
                                                        AElementwiseOperation a_element_op,
                                                        BElementwiseOperation b_element_op,
                                                        CElementwiseOperation c_element_op) override
    {
        return ::std::make_unique<Argument>(std::array<const void*, 1>{p_a},
                                            std::array<const void*, 1>{p_b},
                                            p_ds,
                                            static_cast<CDataType*>(p_c),
                                            M,
                                            N,
                                            K,
                                            std::array<index_t, 1>{StrideA},
                                            std::array<index_t, 1>{StrideB},
                                            DsStrides,
                                            StrideC,
                                            KSplit,
                                            a_element_op,
                                            b_element_op,
                                            c_element_op);
    }

    ::std::string GetTypeString() const override
    {
        auto str = ::std::stringstream();

        auto BlkGemmPipelineSchedulerToString = [](BlockGemmPipelineScheduler s) {
            switch(s)
            {
            case BlockGemmPipelineScheduler::Intrawave: return ::std::string("Intrawave");
            case BlockGemmPipelineScheduler::Interwave: return ::std::string("Interwave");
            }
            return ::std::string("?");
        };

        auto BlkGemmPipelineVersionToString = [](BlockGemmPipelineVersion v) {
            switch(v)
            {
            case BlockGemmPipelineVersion::v1: return ::std::string("v1");
            case BlockGemmPipelineVersion::v2: return ::std::string("v2");
            case BlockGemmPipelineVersion::v3: return ::std::string("v3");
            case BlockGemmPipelineVersion::v4: return ::std::string("v4");
            case BlockGemmPipelineVersion::v5: return ::std::string("v5");
            }
            return ::std::string("v?");
        };

        // clang-format off
        str << "DeviceGemmWmmaUniversalReduce"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << ::std::string(ALayout::name)[0]
            << ::std::string(BLayout::name)[0]
            << ::std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WmmaTile: "
            << MPerWmma<<"x"<<NPerWmma << ", "
            << "WmmaRepeat: "
            << MRepeat<<"x" << NRepeat<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString(BlkGemmPipeSched) << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString(BlkGemmPipelineVer) << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = *dynamic_cast<const Argument*>(p_arg);

        // Need workspace if using split-K or have D tensors
        if(!(!(arg.IsReduceAdd() || NumDTensor > 0) && is_same<CDataType, ReduceDataType>::value))
        {
            return arg.M * arg.N * arg.KBatch * sizeof(ReduceDataType);
        }

        return 0;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
