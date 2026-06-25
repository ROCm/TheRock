// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_xdl(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                            const index_t group_count,
                            const AElementwiseOperation a_element_op,
                            const BElementwiseOperation b_element_op,
                            const CDEElementwiseOperation c_element_op)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        const index_t block_id = get_block_1d_id();

        const auto gemm_desc_ptr = reinterpret_cast<const GemmDesc*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) / 2);
        while((!(block_id >= gemm_desc_ptr[group_id].BlockStart_ &&
                 block_id < gemm_desc_ptr[group_id].BlockEnd_)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].BlockStart_)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) / 2);
        }

        GridwiseGemm::template Run<HasMainKBlockLoop, InMemoryDataOperationEnum::Set>(
            gemm_desc_ptr[group_id].a_ptr_,
            gemm_desc_ptr[group_id].b_ptr_,
            gemm_desc_ptr[group_id].ds_ptr_,
            gemm_desc_ptr[group_id].e_ptr_,
            p_shared,
            a_element_op,
            b_element_op,
            c_element_op,
            gemm_desc_ptr[group_id].a_grid_desc_ak0_m_ak1_,
            gemm_desc_ptr[group_id].b_grid_desc_bk0_n_bk1_,
            gemm_desc_ptr[group_id].ds_grid_desc_mblock_mperblock_nblock_nperblock_,
            gemm_desc_ptr[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
            gemm_desc_ptr[group_id].block_2_etile_map_);
    }
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif
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
          ck::index_t NumPrefetch,
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
          LoopScheduler LoopSched  = make_default_loop_scheduler(),
          typename ComputeDataType = ADataType>
struct DeviceGroupedGemm_Xdl : public DeviceGroupedGemm<ALayout,
                                                        BLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        ADataType,
                                                        BDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        AElementwiseOperation,
                                                        BElementwiseOperation,
                                                        CDEElementwiseOperation,
                                                        ComputeDataType>
{
    using DeviceOp = DeviceGroupedGemm_Xdl;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static auto MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    static auto MakeBGridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                                         const std::array<index_t, NumDTensor>& NRaws,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K(1, 1, 1));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K(1, 1, 1));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}, {}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N<ELayout>(1, 1, 1));

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        BDataType,
        ComputeDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        NumPrefetch, // NumGemmKPrefetchStage
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
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    using AGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(
            AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(
            BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm64::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm64::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    struct GroupedGemmBlock2ETileMap
    {
        using Block2ETileMap =
            remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

        GroupedGemmBlock2ETileMap()
        {
            block_2_etile_map_ = GridwiseGemm64::MakeDefaultBlock2ETileMap(EGridDesc_M_N{});
            BlockStart_        = -1;
        }

        GroupedGemmBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n, ck::index_t BlockStart)
        {
            block_2_etile_map_ = GridwiseGemm64::MakeDefaultBlock2ETileMap(e_grid_desc_m_n);
            BlockStart_        = BlockStart;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            return block_2_etile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[I0] - BlockStart_));
        }

        // it's actually E-Tile
        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_2_etile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        __host__ bool CheckValidity(const EGridDesc_M_N& e_grid_desc_m_n) const
        {
            return block_2_etile_map_.CheckValidity(e_grid_desc_m_n);
        }

        Block2ETileMap block_2_etile_map_;
        ck::index_t BlockStart_;
    };

    struct GemmBiasTransKernelArg
    {
        // pointers
        const ADataType* a_ptr_;
        const BDataType* b_ptr_;
        typename GridwiseGemm64::DsGridPointer ds_ptr_;
        EDataType* e_ptr_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        GroupedGemmBlock2ETileMap block_2_etile_map_;
        ck::index_t BlockStart_, BlockEnd_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        template <typename GridwiseGemm, typename DsPointer, typename Block2ETileMap>
        void init_gridwise_gemm_desc(const ADataType* a_ptr,
                                     const BDataType* b_ptr,
                                     DsPointer ds_ptr,
                                     EDataType* e_ptr,
                                     const AGridDesc_M_K& a_grid_desc_m_k,
                                     const BGridDesc_N_K& b_grid_desc_n_k,
                                     const DsGridDesc_M_N& ds_grid_desc_m_n,
                                     const EGridDesc_M_N& e_grid_desc_m_n,
                                     const Block2ETileMap& block_2_etile_map,
                                     index_t BlockStart,
                                     index_t BlockEnd)
        {
            // tensor descriptors for block/thread-wise copy
            const auto a_grid_desc_ak0_m_ak1 =
                GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);

            const auto b_grid_desc_bk0_n_bk1 =
                GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

            if(GridwiseGemm::CheckValidity(a_grid_desc_m_k,
                                           b_grid_desc_n_k,
                                           ds_grid_desc_m_n,
                                           e_grid_desc_m_n,
                                           block_2_etile_map))
            {
                // tensor descriptors for block/thread-wise copy
                DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                    ds_grid_desc_mblock_mperblock_nblock_nperblock;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    ds_grid_desc_mblock_mperblock_nblock_nperblock(j) =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            ds_grid_desc_m_n[j]);
                });

                const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n);

                gemm_desc_kernel_arg_.push_back(
                    GemmBiasTransKernelArg{a_ptr,
                                           b_ptr,
                                           ds_ptr,
                                           e_ptr,
                                           a_grid_desc_m_k,
                                           b_grid_desc_n_k,
                                           ds_grid_desc_m_n,
                                           e_grid_desc_m_n,
                                           a_grid_desc_ak0_m_ak1,
                                           b_grid_desc_bk0_n_bk1,
                                           ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                           e_grid_desc_mblock_mperblock_nblock_nperblock,
                                           block_2_etile_map,
                                           BlockStart,
                                           BlockEnd});
            }
        };
        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op)
            : a_element_op_{a_element_op}, b_element_op_{b_element_op}, c_element_op_{c_element_op}
        {
            grid_size_ = 0;

            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            skipped_group_count_ = 0;

            for(std::size_t i = 0; i < gemm_descs.size(); i++)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                a_mtx_mraw_kraw_.emplace_back(M, K);
                b_mtx_nraw_kraw_.emplace_back(N, K);

                if(M == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t StrideA = gemm_descs[i].stride_A_;
                const index_t StrideB = gemm_descs[i].stride_B_;
                const index_t StrideC = gemm_descs[i].stride_C_;

                // pointer
                typename GridwiseGemm64::DsGridPointer p_ds_grid{};

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    using DDataType = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;

                    p_ds_grid(j) = static_cast<const DDataType*>(p_Ds[i][j]);
                });

                // tensor descriptors for problem definiton
                const auto a_grid_desc_m_k = DeviceOp::MakeAGridDescriptor_M_K(M, K, StrideA);
                const auto b_grid_desc_n_k = DeviceOp::MakeBGridDescriptor_N_K(K, N, StrideB);

                DsGridDesc_M_N ds_grid_desc_m_n;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

                    ds_grid_desc_m_n(j) = DeviceOp::MakeEGridDescriptor_M_N<DLayout>(
                        M, N, gemm_descs[i].stride_Ds_[j]);
                });

                const auto e_grid_desc_m_n =
                    DeviceOp::MakeEGridDescriptor_M_N<ELayout>(M, N, StrideC);

                const index_t grid_size_grp =
                    GroupedGemmBlock2ETileMap(e_grid_desc_m_n, 0)
                        .block_2_etile_map_.CalculateGridSize(e_grid_desc_m_n);

                const index_t BlockStart = grid_size_;
                const index_t BlockEnd   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                const auto block_2_etile_map =
                    GroupedGemmBlock2ETileMap(e_grid_desc_m_n, BlockStart);

                if(get_warp_size() == 64)
                {
                    if constexpr(NXdlPerWave64 > 0)
                    {
                        init_gridwise_gemm_desc<GridwiseGemm64>(
                            static_cast<const ADataType*>(p_As[i]),
                            static_cast<const BDataType*>(p_Bs[i]),
                            p_ds_grid,
                            static_cast<EDataType*>(p_Es[i]),
                            a_grid_desc_m_k,
                            b_grid_desc_n_k,
                            ds_grid_desc_m_n,
                            e_grid_desc_m_n,
                            block_2_etile_map,
                            BlockStart,
                            BlockEnd);
                    }
                }
                else
                {
                    if constexpr(NXdlPerWave32 > 0)
                    {
                        init_gridwise_gemm_desc<GridwiseGemm32>(
                            static_cast<const ADataType*>(p_As[i]),
                            static_cast<const BDataType*>(p_Bs[i]),
                            p_ds_grid,
                            static_cast<EDataType*>(p_Es[i]),
                            a_grid_desc_m_k,
                            b_grid_desc_n_k,
                            ds_grid_desc_m_n,
                            e_grid_desc_m_n,
                            block_2_etile_map,
                            BlockStart,
                            BlockEnd);
                    }
                }
            }
        }

        //  private:
        index_t group_count_;
        index_t skipped_group_count_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation c_element_op_;

        std::vector<GemmBiasTransKernelArg> gemm_desc_kernel_arg_;
        std::vector<Tuple<index_t, index_t>> a_mtx_mraw_kraw_;
        std::vector<Tuple<index_t, index_t>> b_mtx_nraw_kraw_;

        index_t grid_size_;
        void* gemm_kernel_host_args_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{},
                     hipStream_t cpy_stream            = nullptr,
                     hipEvent_t cpy_event              = nullptr)
        {
            bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "group: " << i << " arg.a_grid_desc_ak0_m_ak1_{"
                              << arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I1)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2)
                              << "}";

                    std::cout << ", arg.b_grid_desc_bk0_n_bk1_{"
                              << arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_.GetLength(I0)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_.GetLength(I1)
                              << ", "
                              << arg.gemm_desc_kernel_arg_[i].b_grid_desc_bk0_n_bk1_.GetLength(I2)
                              << "}";

                    std::cout << ", arg.e_grid_desc_m_n_{ "
                              << arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_.GetLength(I0) << ", "
                              << arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_.GetLength(I1) << "}"
                              << std::endl;
                }

                if(!GridwiseGemm::CheckValidity(arg.gemm_desc_kernel_arg_[i].a_grid_desc_m_k_,
                                                arg.gemm_desc_kernel_arg_[i].b_grid_desc_n_k_,
                                                arg.gemm_desc_kernel_arg_[i].ds_grid_desc_m_n_,
                                                arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_,
                                                arg.gemm_desc_kernel_arg_[i].block_2_etile_map_))
                {
                    throw std::runtime_error(
                        "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
                }

                const auto K = arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                               arg.gemm_desc_kernel_arg_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2);

                if(GridwiseGemm::CalculateHasMainKBlockLoop(K) != has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
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
                hipGetErrorString(hipMemcpyAsync(arg.p_workspace_,
                                                 arg.gemm_kernel_host_args_,
                                                 arg.group_count_ * sizeof(GemmBiasTransKernelArg),
                                                 hipMemcpyHostToDevice,
                                                 cpy_stream));
                hipGetErrorString(hipEventRecord(cpy_event, cpy_stream));
                hipGetErrorString(hipEventSynchronize(cpy_event));
            }
            else // In this case CK owns memory allocated on host.
            {
                hipGetErrorString(hipMemcpyAsync(arg.p_workspace_,
                                                 arg.gemm_desc_kernel_arg_.data(),
                                                 arg.gemm_desc_kernel_arg_.size() *
                                                     sizeof(GemmBiasTransKernelArg),
                                                 hipMemcpyHostToDevice,
                                                 stream_config.stream_id_));
            }

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel = kernel_grouped_gemm_xdl<GridwiseGemm,
                                                            GemmBiasTransKernelArg,
                                                            AElementwiseOperation,
                                                            BElementwiseOperation,
                                                            CDEElementwiseOperation,
                                                            has_main_k_block_loop_>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.p_workspace_),
                    arg.gemm_desc_kernel_arg_.size(),
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.c_element_op_);
            };

            if(has_main_k_block_loop)
            {
                ave_time = launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                ave_time = launch_kernel(integral_constant<bool, false>{});
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

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_wmma_supported<ADataType, BDataType, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if((ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) +
            arg.skipped_group_count_) != arg.group_count_)
        {
            return false;
        }

        bool supported = true;

        // If we use padding we do not support vector loads for dimensions not divisible by
        // vector load size.
        if constexpr(GemmSpec != GemmSpecialization::Default)
        {
            // [A|B]BlockTransferSrcVectorDim value define dimension in the block {K0,M,K1}
            // layout, thus we have to adapt it to the {M,K} or {N,K} layout.
            const auto a_raw_vector_dim = ABlockTransferSrcVectorDim != 1 ? 1 : 0;
            const auto b_raw_vector_dim = BBlockTransferSrcVectorDim != 1 ? 1 : 0;

            for(index_t i = 0; i < arg.group_count_; ++i)
            {
                const auto a_vector_dim = arg.a_mtx_mraw_kraw_[i].At(Number<a_raw_vector_dim>{});
                const auto b_vector_dim = arg.b_mtx_nraw_kraw_[i].At(Number<b_raw_vector_dim>{});

                supported = supported & (a_vector_dim % ABlockTransferSrcScalarPerVector == 0);
                supported = supported & (b_vector_dim % BBlockTransferSrcScalarPerVector == 0);
            }
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
                             std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation c_element_op)
    {
        return Argument{
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(
            p_As, p_Bs, p_Ds, p_Es, gemm_descs, a_element_op, b_element_op, c_element_op);
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
        str << "DeviceGroupedGemm_Xdl"
            << "<"
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
            return p_arg_->group_count_ * sizeof(GemmBiasTransKernelArg);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemmMultipleDXdlCShuffle::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        return GetWorkSpaceSize(p_arg);
    }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* p_dev_kernel_args) const override
    {
        return this->SetWorkSpacePointer(p_arg, p_dev_kernel_args);
    }

    size_t GetHostKernelArgSize(const BaseArgument* p_arg) const { return GetWorkSpaceSize(p_arg); }

    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the host kernel arguments pointer and copies that data on the host
    /// side.
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
        std::copy(pArg_->gemm_desc_kernel_arg_.begin(),
                  pArg_->gemm_desc_kernel_arg_.end(),
                  static_cast<GemmBiasTransKernelArg*>(pArg_->gemm_kernel_host_args_));
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
