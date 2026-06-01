// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_cgemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename GemmAccDataType,
    typename CShuffleDataType,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    GemmSpecialization GemmSpec,
    index_t NumGemmKPrefetchStage,
    index_t BlockSize,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t KPerBlock,
    index_t AK1,
    index_t BK1,
    index_t MPerXDL,
    index_t NPerXDL,
    index_t MXdlPerWave,
    index_t NXdlPerWave,
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
    index_t CShuffleMXdlPerWavePerShuffle,
    index_t CShuffleNXdlPerWavePerShuffle,
    typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
    LoopScheduler LoopSched = make_default_loop_scheduler(),
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct DeviceCGemm_4Gemm_Xdl_CShuffle
    : public DeviceCGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>
{
    using DeviceOp = DeviceCGemm_4Gemm_Xdl_CShuffle;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t MPerThread =
        MPerBlock / CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t NPerThread =
        NPerBlock / CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr auto AScalarPerVector = Number<4>{};
    static constexpr auto BScalarPerVector = Number<4>{};
    static constexpr auto CScalarPerVector = Number<4>{};

    template <typename Desc_M_N>
    static auto PadDescriptor_M_N(Desc_M_N desc)
    {
        const auto M     = desc.GetLength(I0);
        const auto N     = desc.GetLength(I1);
        const auto pad_M = math::integer_divide_ceil(M, MPerThread) * MPerThread - M;
        const auto pad_N = math::integer_divide_ceil(N, NPerThread) * NPerThread - N;

        const auto padded_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_right_pad_transform(M, pad_M), make_right_pad_transform(N, pad_N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return padded_desc;
    }

    static auto MakeDescriptor_M_N(const std::vector<index_t>& lengths,
                                   const std::vector<index_t>& strides)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<2>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return strides[I]; }, Number<2>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
        return PadDescriptor_M_N(desc);
    }

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
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
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using CGridDesc_M_N = decltype(MakeDescriptor_M_N({1, 1}, {1, 1}));

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public GridwiseGemm64::Problem
    {
        using Problem = typename GridwiseGemm64::Problem;

        Argument(const ADataType* p_a_grid_real_,
                 const ADataType* p_a_grid_imag_,
                 const BDataType* p_b_grid_real_,
                 const BDataType* p_b_grid_imag_,
                 CDataType* p_c_grid_real_,
                 CDataType* p_c_grid_imag_,
                 CDataType* p_workspace,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_},
              p_a_grid_real{p_a_grid_real_},
              p_a_grid_imag{p_a_grid_imag_},
              p_b_grid_real{p_b_grid_real_},
              p_b_grid_imag{p_b_grid_imag_},
              p_c_grid_real{p_c_grid_real_},
              p_c_grid_imag{p_c_grid_imag_},
              p_aux_grid{p_workspace}
        {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                c_grid_desc_m_n = DeviceOp::MakeDescriptor_M_N({M_, N_}, {StrideC_, I1});
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                c_grid_desc_m_n = DeviceOp::MakeDescriptor_M_N({M_, N_}, {I1, StrideC_});
            }

            p_aux_2_grid = p_workspace + GetCElementSpaceSize(M_, N_, StrideC_);
        }

        //  private:
        const ADataType* p_a_grid_real;
        const ADataType* p_a_grid_imag;
        const BDataType* p_b_grid_real;
        const BDataType* p_b_grid_imag;
        CDataType* p_c_grid_real;
        CDataType* p_c_grid_imag;
        CDataType* p_aux_grid;
        CDataType* p_aux_2_grid;
        CGridDesc_M_N c_grid_desc_m_n;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            typename GridwiseGemm::Problem problem(
                arg.M, arg.N, arg.K, arg.StrideA, arg.StrideB, arg.StrideC);
            if(!GridwiseGemm::CheckValidity(problem))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N);

            const auto K = GridwiseGemm::CalculateAK0(arg.K) * AK1;

            float ave_time = 0;

            using Add      = ck::tensor_operation::element_wise::Add;
            using Subtract = ck::tensor_operation::element_wise::Subtract;

            using Block2TileMap = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

            using GridwiseBinAdd = GridwiseElementwise<Tuple<CGridDesc_M_N, CGridDesc_M_N>,
                                                       Tuple<CGridDesc_M_N>,
                                                       Tuple<const CDataType*, const CDataType*>,
                                                       Tuple<CDataType*>,
                                                       Block2TileMap,
                                                       Add,
                                                       BlockSize,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       MPerThread,
                                                       NPerThread,
                                                       Sequence<0, 1>,
                                                       Sequence<AScalarPerVector, BScalarPerVector>,
                                                       Sequence<CScalarPerVector>,
                                                       I1,
                                                       I1>;

            using GridwiseBinSubtract =
                GridwiseElementwise<Tuple<CGridDesc_M_N, CGridDesc_M_N>,
                                    Tuple<CGridDesc_M_N>,
                                    Tuple<const CDataType*, const CDataType*>,
                                    Tuple<CDataType*>,
                                    Block2TileMap,
                                    Subtract,
                                    BlockSize,
                                    MPerBlock,
                                    NPerBlock,
                                    MPerThread,
                                    NPerThread,
                                    Sequence<0, 1>,
                                    Sequence<AScalarPerVector, BScalarPerVector>,
                                    Sequence<CScalarPerVector>,
                                    I1,
                                    I1>;

            const index_t M             = arg.c_grid_desc_m_n.GetLength(I0);
            const index_t N             = arg.c_grid_desc_m_n.GetLength(I1);
            const auto block_2_tile_map = Block2TileMap(M, N);

            const auto add_kernel = kernel_elementwise<GridwiseBinAdd,
                                                       Tuple<CGridDesc_M_N, CGridDesc_M_N>,
                                                       Tuple<CGridDesc_M_N>,
                                                       Tuple<const CDataType*, const CDataType*>,
                                                       Tuple<CDataType*>,
                                                       Block2TileMap,
                                                       Add>;

            const auto subtract_kernel =
                kernel_elementwise<GridwiseBinSubtract,
                                   Tuple<CGridDesc_M_N, CGridDesc_M_N>,
                                   Tuple<CGridDesc_M_N>,
                                   Tuple<const CDataType*, const CDataType*>,
                                   Tuple<CDataType*>,
                                   Block2TileMap,
                                   Subtract>;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_gemm_xdl_cshuffle_v1<GridwiseGemm,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                true>;

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_real,
                                                   arg.p_b_grid_real,
                                                   arg.p_aux_grid,
                                                   problem);

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_imag,
                                                   arg.p_b_grid_imag,
                                                   arg.p_aux_2_grid,
                                                   problem);

                // c_real = aux - aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    subtract_kernel,
                    dim3(gdx, gdy, gdz),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_n, arg.c_grid_desc_m_n),
                    make_tuple(arg.c_grid_desc_m_n),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid),
                               const_cast<const CDataType*>(arg.p_aux_2_grid)),
                    make_tuple(arg.p_c_grid_real),
                    block_2_tile_map,
                    Subtract{});

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_real,
                                                   arg.p_b_grid_imag,
                                                   arg.p_aux_grid,
                                                   problem);

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_imag,
                                                   arg.p_b_grid_real,
                                                   arg.p_aux_2_grid,
                                                   problem);

                // c_imag = aux + aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    add_kernel,
                    dim3(gdx, gdy, gdz),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_n, arg.c_grid_desc_m_n),
                    make_tuple(arg.c_grid_desc_m_n),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid),
                               const_cast<const CDataType*>(arg.p_aux_2_grid)),
                    make_tuple(arg.p_c_grid_imag),
                    block_2_tile_map,
                    Add{});
            }
            else
            {
                const auto kernel = kernel_gemm_xdl_cshuffle_v1<GridwiseGemm,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                false>;

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_real,
                                                   arg.p_b_grid_real,
                                                   arg.p_aux_grid,
                                                   problem);

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_imag,
                                                   arg.p_b_grid_imag,
                                                   arg.p_aux_2_grid,
                                                   problem);

                // c_real = aux - aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    subtract_kernel,
                    dim3(gdx, gdy, gdz),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_n, arg.c_grid_desc_m_n),
                    make_tuple(arg.c_grid_desc_m_n),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid),
                               const_cast<const CDataType*>(arg.p_aux_2_grid)),
                    make_tuple(arg.p_c_grid_real),
                    block_2_tile_map,
                    Subtract{});

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_real,
                                                   arg.p_b_grid_imag,
                                                   arg.p_aux_grid,
                                                   problem);

                ave_time += launch_and_time_kernel(stream_config,
                                                   kernel,
                                                   dim3(gdx, gdy, gdz),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.p_a_grid_imag,
                                                   arg.p_b_grid_real,
                                                   arg.p_aux_2_grid,
                                                   problem);

                // c_imag = aux + aux_2
                ave_time += launch_and_time_kernel(
                    stream_config,
                    add_kernel,
                    dim3(gdx, gdy, gdz),
                    dim3(BlockSize),
                    0,
                    make_tuple(arg.c_grid_desc_m_n, arg.c_grid_desc_m_n),
                    make_tuple(arg.c_grid_desc_m_n),
                    make_tuple(const_cast<const CDataType*>(arg.p_aux_grid),
                               const_cast<const CDataType*>(arg.p_aux_2_grid)),
                    make_tuple(arg.p_c_grid_imag),
                    block_2_tile_map,
                    Add{});
            }

            return ave_time;
        }

        INVOKER_RUN_IMPL

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
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(arg);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                typename GridwiseGemm32::Problem problem(
                    arg.M, arg.N, arg.K, arg.StrideA, arg.StrideB, arg.StrideC);
                return GridwiseGemm32::CheckValidity(problem);
            }
        }
        return false;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a_real,
                             const ADataType* p_a_imag,
                             const BDataType* p_b_real,
                             const BDataType* p_b_imag,
                             CDataType* p_c_real,
                             CDataType* p_c_imag,
                             CDataType* p_workspace,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CElementwiseOperation)
    {
        return Argument{p_a_real,
                        p_a_imag,
                        p_b_real,
                        p_b_imag,
                        p_c_real,
                        p_c_imag,
                        p_workspace,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a_real,
                                                      const void* p_a_imag,
                                                      const void* p_b_real,
                                                      const void* p_b_imag,
                                                      void* p_c_real,
                                                      void* p_c_imag,
                                                      void* p_workspace,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation,
                                                      index_t /* KBatch */ = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a_real),
                                          static_cast<const ADataType*>(p_a_imag),
                                          static_cast<const BDataType*>(p_b_real),
                                          static_cast<const BDataType*>(p_b_imag),
                                          static_cast<CDataType*>(p_c_real),
                                          static_cast<CDataType*>(p_c_imag),
                                          static_cast<CDataType*>(p_workspace),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC);
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
        str << "DeviceCGemm_4Gemm_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1
            << ">";
        // clang-format on

        return str.str();
    }

    static std::size_t GetCElementSpaceSize(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n =
            GridwiseGemm64::MakeCGridDescriptor_M_N(M,
                                                    GridwiseGemm64::CalculateMPadded(M),
                                                    N,
                                                    GridwiseGemm64::CalculateNPadded(N),
                                                    StrideC);

        return c_grid_desc_m_n.GetElementSpaceSize();
    }

    std::size_t GetWorkspaceSize(index_t M,
                                 index_t N,
                                 [[maybe_unused]] index_t K,
                                 [[maybe_unused]] index_t StrideA,
                                 [[maybe_unused]] index_t StrideB,
                                 index_t StrideC) const override
    {
        return 2 * sizeof(CDataType) * GetCElementSpaceSize(M, N, StrideC);
    }

    std::size_t GetWorkSpaceSize(const BaseArgument* base_arg) const override
    {
        const auto* parg = dynamic_cast<const Argument*>(base_arg);

        if(!parg)
        {
            std::ostringstream err;
            err << "Provided argument pointer is not of an Argument class!" << " In " << __FILE__
                << ":" << __LINE__ << ", in function: " << __func__;
            throw std::runtime_error(err.str());
        }

        return GetWorkspaceSize(
            parg->M, parg->N, parg->K, parg->StrideA, parg->StrideB, parg->StrideC);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
