// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_splitk_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          GemmSpecialization GemmSpec,
          bool Zeroing,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DsDataType,
          typename Block2ETileMap,
          typename GroupedGemmBlock2ETileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_gemm_xdl_fixed_nk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                     uint32_t* barrier_count,
                                     const index_t barrier_size_grp,
                                     const index_t group_count,
                                     const index_t grid_size_grp,
                                     const index_t KBatch,
                                     const AElementwiseOperation a_element_op,
                                     const BElementwiseOperation b_element_op,
                                     const CDEElementwiseOperation c_element_op)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<EGlobalMemoryDataOperation>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        const index_t block_id = get_block_1d_id();

        const auto gemm_desc_ptr = reinterpret_cast<const GemmDesc*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = block_id / grid_size_grp;

        if(group_id >= group_count)
            return;

        const index_t M = gemm_desc_ptr[group_id].M;
        const index_t N = gemm_desc_ptr[group_id].N;
        const index_t K = gemm_desc_ptr[group_id].K;

        if(M == 0 || N == 0 || K == 0)
            return;

        const auto StrideA  = gemm_desc_ptr[group_id].StrideA;
        const auto StrideB  = gemm_desc_ptr[group_id].StrideB;
        const auto StrideDs = gemm_desc_ptr[group_id].StrideDs;
        const auto StrideE  = gemm_desc_ptr[group_id].StrideE;

        const auto e_grid_desc_m_n =
            GridwiseGemm::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

        const index_t BlockStart = group_id * grid_size_grp;

        const auto local_b2e_tile_map = Block2ETileMap{e_grid_desc_m_n, KBatch};

        const auto local_grid_size = local_b2e_tile_map.CalculateGridSize(e_grid_desc_m_n);

        constexpr auto NumDTensor = DsDataType::Size();

        using DsGridPointer = decltype(GridwiseGemm::MakeDsGridPointer());

        DsGridPointer p_ds_grid_;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            // D pointer
            p_ds_grid_(i) = static_cast<const DDataType*>(gemm_desc_ptr[group_id].p_ds_grid[i]);
        });

        index_t id_off   = 0;
        index_t id_local = get_block_1d_id() - BlockStart;

        const index_t mn_blocks = local_grid_size / KBatch;

        while(id_local < local_grid_size)
        {
            const auto block_2_etile_map =
                GroupedGemmBlock2ETileMap(local_b2e_tile_map, BlockStart, id_off);

            if constexpr(Zeroing)
            {
                auto barrier_count_finished =
                    barrier_count + group_id * barrier_size_grp + id_local % mn_blocks;
                GridwiseGemm::template RunWithZeroing<HasMainKBlockLoop,
                                                      EGlobalMemoryDataOperation,
                                                      GemmSpec,
                                                      ALayout,
                                                      BLayout,
                                                      DsLayout,
                                                      ELayout>(gemm_desc_ptr[group_id].p_a_grid,
                                                               gemm_desc_ptr[group_id].p_b_grid,
                                                               p_ds_grid_,
                                                               gemm_desc_ptr[group_id].p_e_grid,
                                                               p_shared,
                                                               barrier_count_finished,
                                                               a_element_op,
                                                               b_element_op,
                                                               c_element_op,
                                                               M,
                                                               N,
                                                               K,
                                                               StrideA,
                                                               StrideB,
                                                               StrideDs,
                                                               StrideE,
                                                               KBatch,
                                                               block_2_etile_map);
            }
            else
            {

                GridwiseGemm::template Run<HasMainKBlockLoop,
                                           EGlobalMemoryDataOperation,
                                           GemmSpec,
                                           ALayout,
                                           BLayout,
                                           DsLayout,
                                           ELayout>(gemm_desc_ptr[group_id].p_a_grid,
                                                    gemm_desc_ptr[group_id].p_b_grid,
                                                    p_ds_grid_,
                                                    gemm_desc_ptr[group_id].p_e_grid,
                                                    p_shared,
                                                    nullptr,
                                                    a_element_op,
                                                    b_element_op,
                                                    c_element_op,
                                                    M,
                                                    N,
                                                    K,
                                                    StrideA,
                                                    StrideB,
                                                    StrideDs,
                                                    StrideE,
                                                    KBatch,
                                                    block_2_etile_map);
            }

            id_off += grid_size_grp;
            id_local += grid_size_grp;
        }
    }
#else
    ignore = gemm_descs_const;
    ignore = barrier_count;
    ignore = barrier_size_grp;
    ignore = group_count;
    ignore = grid_size_grp;
    ignore = KBatch;
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
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          typename ComputeType        = ADataType,
          typename ALDSType           = ComputeType,
          typename BLDSType           = ComputeType>
struct DeviceGroupedGemm_Xdl_Fixed_NK : public DeviceGroupedGemmFixedNK<ALayout,
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
    using DeviceOp = DeviceGroupedGemm_Xdl_Fixed_NK;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using AComputeType = ComputeType;
    using BComputeType = ComputeType;

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemmMultipleD_xdl_splitk_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        BDataType,
        AComputeType,
        BComputeType,
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
        LoopSched,
        PipelineVer,
        ALDSType,
        BLDSType>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    template <typename UnderlyingBlockToCTileMap>
    struct OffsettedBlockToCTileMapMLoops
    {
        using underlying_type = UnderlyingBlockToCTileMap;

        __host__ __device__ OffsettedBlockToCTileMapMLoops(
            UnderlyingBlockToCTileMap block_to_ctile_map, index_t block_start, index_t id_off = 0)
        {
            block_to_ctile_map_ = block_to_ctile_map;
            block_start_        = block_start;
            id_off_             = id_off;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto idx_bot = block_to_ctile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[Number<0>{}] - block_start_ + id_off_));

            return make_tuple(idx_bot[Number<0>{}], idx_bot[Number<1>{}], idx_bot[Number<2>{}]);
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        template <typename CGridDesc_M_N>
        __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        template <typename CGridDesc_M_N>
        __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CalculateGridSize(c_grid_desc_m_n);
        }

        UnderlyingBlockToCTileMap block_to_ctile_map_;
        index_t block_start_;
        index_t id_off_;
    };

    template <index_t MPerBlock_, index_t NPerBlock_>
    struct BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops
    {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops() = default;

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            const BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&
        operator=(const BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&
        operator=(BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&&) = default;

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(index_t M,
                                                                          index_t N,
                                                                          index_t KBatch,
                                                                          index_t M01 = 8)
            : M_(M), N_(N), KBatch_(KBatch), M01_(M01)
        {
        }

        template <typename CGridDesc_M_N>
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            const CGridDesc_M_N& c_grid_desc_m_n, index_t KBatch, index_t M01 = 8)
            : BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
                  c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1), KBatch, M01)
        {
        }

        __host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
        {
            const auto M0 = math::integer_divide_ceil(M, MPerBlock);
            const auto N0 = math::integer_divide_ceil(N, NPerBlock);

            return M0 * N0 * KBatch_;
        }

        template <typename CGridDesc_M_N>
        __host__ __device__ constexpr index_t
        CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return CalculateGridSize(c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1));
        }

        template <typename CGridDesc_M_N>
        __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
        {
            return true;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto block_1d_id = idx_top[I0];

            const auto M0 = math::integer_divide_ceil(M_, MPerBlock_);
            const auto N0 = math::integer_divide_ceil(N_, NPerBlock_);

            block_1d_id = block_1d_id % (M0 * N0 * KBatch_); // hide groups

            const index_t idx_ksplit = block_1d_id / (M0 * N0);
            block_1d_id              = block_1d_id % (M0 * N0);

            index_t idx_N0 = block_1d_id % N0;
            index_t idx_M0 = block_1d_id / N0;

            const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

            index_t idx_M00          = idx_M0 / M01_;
            index_t idx_M01          = idx_M0 % M01_;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

            return make_tuple(idx_ksplit,
                              idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                              idx_N0_M01_local / M01_adapt);
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                                 const CTileDim& /* c_tile_dim */) const
        {
            return true; // always valid provided that user gets grid size from CalculateGridSize()
        }

        private:
        index_t M_;
        index_t N_;
        index_t KBatch_;
        index_t M01_;
    };

    using Block2ETileMap = BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops<MPerBlock, NPerBlock>;
    using GroupedGemmBlock2ETileMap = OffsettedBlockToCTileMapMLoops<Block2ETileMap>;

    // TODO: replace with GroupedGemmKernelArgument
    struct GemmBiasTransKernelArg
    {
        // pointers
        const void* a_ptr_;
        const void* b_ptr_;
        std::array<const void*, NumDTensor> ds_ptr_;
        void* e_ptr_;

        index_t M_, N_, K_;
        index_t StrideA_, StrideB_;
        std::array<index_t, NumDTensor> StrideDs_;
        index_t StrideE_;
    };

    // Argument
    struct Argument : public BaseArgument
    {

        void UpdateKBatch(index_t k_batch)
        {
            k_batch_ = k_batch;

            if(k_batch_ < 1)
            {

                throw std::runtime_error("wrong! k_batch must be > 0");
            }

            const index_t AverM = math::integer_divide_ceil(sum_of_m, group_count_);

            const index_t StrideE = gemm_desc_kernel_arg_[0].StrideE_;
            const index_t N       = gemm_desc_kernel_arg_[0].N_;

            const auto e_grid_desc_m_n =
                GridwiseGemm64::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                    AverM, N, StrideE);

            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

            grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

            grid_size_ = grid_size_grp_ * group_count_;
        }

        Argument(std::vector<const void*>&,
                 std::vector<const void*>&,
                 std::vector<std::array<const void*, NumDTensor>>&,
                 std::vector<void*>&,
                 std::vector<GemmDesc>& gemm_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation c_element_op)
            : a_element_op_{a_element_op}, b_element_op_{b_element_op}, c_element_op_{c_element_op}
        {
            grid_size_ = 0;

            k_batch_ = 1;

            grouped_gemm_kernel_args_dev = nullptr;

            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            gemm_desc_kernel_arg_.reserve(group_count_);

            index_t group_id = 0;

            sum_of_m            = gemm_descs[0].M_;
            const index_t AverM = math::integer_divide_ceil(sum_of_m, group_count_);
            const index_t N     = gemm_descs[0].N_;
            const index_t K     = gemm_descs[0].K_;

            for(std::size_t i = 0; i < gemm_descs.size(); i++)
            {
                if(sum_of_m != gemm_descs[i].M_ || N != gemm_descs[i].N_ || K != gemm_descs[i].K_)
                {
                    throw std::runtime_error("wrong! M/N/K is not identical");
                }

                a_mtx_mraw_kraw_.emplace_back(sum_of_m, K);
                b_mtx_nraw_kraw_.emplace_back(N, K);

                const index_t StrideA = gemm_descs[i].stride_A_;
                const index_t StrideB = gemm_descs[i].stride_B_;
                const index_t StrideE = gemm_descs[i].stride_C_;

                // pointer
                std::array<const void*, NumDTensor> p_ds_grid;

                static_for<0, NumDTensor, 1>{}([&](auto j) { p_ds_grid[j] = nullptr; });

                std::array<index_t, NumDTensor> StrideDs;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    // using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

                    if(gemm_descs[i].stride_Ds_.size() != NumDTensor)
                    {
                        throw std::runtime_error(
                            "wrong! gemm_descs[i].stride_Ds_.size() does not match NumDTensor");
                    }

                    StrideDs[j] = gemm_descs[i].stride_Ds_[j];
                });

                const auto e_grid_desc_m_n =
                    GridwiseGemm64::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                        AverM, N, StrideE);

                // block-to-e-tile map
                const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_m_n, k_batch_};

                grid_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_m_n);

                if(group_id * grid_size_grp_ != grid_size_)
                {
                    throw std::runtime_error("wrong! grid_size_grp_ is not identical!");
                }

                grid_size_ += grid_size_grp_;

                // check block-to-E-tile

                if(!local_b2c_tile_map.CheckValidity(e_grid_desc_m_n))
                {
                    throw std::runtime_error("wrong! block_2_etile_map validation failed");
                }

                if(get_warp_size() == 64)
                {
                    if constexpr(NXdlPerWave64 > 0)
                    {
                        if(!GridwiseGemm64::template CheckValidity<ALayout,
                                                                   BLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   GemmSpec>(
                               AverM, N, K, StrideA, StrideB, StrideDs, StrideE, 1))
                        {
                            throw std::runtime_error(
                                "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid "
                                "setting");
                        }
                    }
                    else
                    {
                        throw std::runtime_error(
                            "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
                    }
                }
                else
                {
                    if constexpr(NXdlPerWave32 > 0)
                    {
                        if(!GridwiseGemm32::template CheckValidity<ALayout,
                                                                   BLayout,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   GemmSpec>(
                               AverM, N, K, StrideA, StrideB, StrideDs, StrideE, 1))
                        {
                            throw std::runtime_error(
                                "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid "
                                "setting");
                        }
                    }
                    else
                    {
                        throw std::runtime_error(
                            "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
                    }
                }

                gemm_desc_kernel_arg_.push_back(GemmBiasTransKernelArg{
                    nullptr,
                    nullptr,
                    p_ds_grid,
                    nullptr,
                    AverM,
                    N,
                    K,
                    StrideA,
                    StrideB,
                    StrideDs,
                    StrideE,
                });

                group_id++;
            }

            const auto e_grid_desc_sum_m_n =
                GridwiseGemm64::template MakeEGridDescriptor_M_N<ELayout, GemmSpec>(
                    sum_of_m, gemm_desc_kernel_arg_[0].N_, gemm_desc_kernel_arg_[0].StrideE_);

            const auto local_b2c_tile_map = Block2ETileMap{e_grid_desc_sum_m_n, 1};

            barrier_size_grp_ = local_b2c_tile_map.CalculateGridSize(e_grid_desc_sum_m_n);
        }

        //  private:
        index_t group_count_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation c_element_op_;

        std::vector<GemmBiasTransKernelArg> gemm_desc_kernel_arg_;
        std::vector<Tuple<index_t, index_t>> a_mtx_mraw_kraw_;
        std::vector<Tuple<index_t, index_t>> b_mtx_nraw_kraw_;

        const void* grouped_gemm_kernel_args_dev;

        index_t grid_size_;
        index_t grid_size_grp_;
        index_t barrier_size_grp_;
        index_t sum_of_m;

        index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                const auto KPad =
                    GridwiseGemm::CalculateKPadded(arg.gemm_desc_kernel_arg_[i].K_, arg.k_batch_);

                if(GridwiseGemm::CalculateHasMainKBlockLoop(KPad) != has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
                }
            }

            if(arg.grouped_gemm_kernel_args_dev == nullptr)
            {
                throw std::runtime_error("wrong! grouped_gemm_kernel_args_dev is nullpr");
            }

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_, auto e_global_memory_operation_) {
                if(arg.k_batch_ == 1)
                {
                    const auto kernel =
                        kernel_grouped_gemm_xdl_fixed_nk<GridwiseGemm,
                                                         GroupedGemmKernelArgument<NumDTensor>,
                                                         GemmSpec,
                                                         false,
                                                         ALayout,
                                                         BLayout,
                                                         DsLayout,
                                                         ELayout,
                                                         DsDataType,
                                                         Block2ETileMap,
                                                         GroupedGemmBlock2ETileMap,
                                                         AElementwiseOperation,
                                                         BElementwiseOperation,
                                                         CDEElementwiseOperation,
                                                         e_global_memory_operation_,
                                                         has_main_k_block_loop_>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(arg.grid_size_),
                        dim3(BlockSize),
                        0,
                        cast_pointer_to_constant_address_space(arg.grouped_gemm_kernel_args_dev),
                        nullptr,
                        arg.barrier_size_grp_,
                        arg.gemm_desc_kernel_arg_.size(),
                        arg.grid_size_grp_,
                        arg.k_batch_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.c_element_op_);
                }
                else
                {
                    const auto kernel =
                        kernel_grouped_gemm_xdl_fixed_nk<GridwiseGemm,
                                                         GroupedGemmKernelArgument<NumDTensor>,
                                                         GemmSpec,
                                                         true,
                                                         ALayout,
                                                         BLayout,
                                                         DsLayout,
                                                         ELayout,
                                                         DsDataType,
                                                         Block2ETileMap,
                                                         GroupedGemmBlock2ETileMap,
                                                         AElementwiseOperation,
                                                         BElementwiseOperation,
                                                         CDEElementwiseOperation,
                                                         e_global_memory_operation_,
                                                         has_main_k_block_loop_>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(arg.grid_size_),
                        dim3(BlockSize),
                        0,
                        cast_pointer_to_constant_address_space(arg.grouped_gemm_kernel_args_dev),
                        reinterpret_cast<uint32_t*>(arg.p_workspace_),
                        arg.barrier_size_grp_,
                        arg.gemm_desc_kernel_arg_.size(),
                        arg.grid_size_grp_,
                        arg.k_batch_,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.c_element_op_);
                }
            };

            constexpr auto AtomicAdd = InMemoryDataOperationEnum::AtomicAdd;
            constexpr auto Set       = InMemoryDataOperationEnum::Set;

            // For bf16 datatype only kbatch = 1 scenario is supported. This condition is
            // enforced in IsSupportedArgument function
            if constexpr(std::is_same<ADataType, ck::bhalf_t>::value)
            {
                if(has_main_k_block_loop)
                {
                    ave_time = launch_kernel(integral_constant<bool, true>{},
                                             integral_constant<InMemoryDataOperationEnum, Set>{});
                }
                else
                {
                    ave_time = launch_kernel(integral_constant<bool, false>{},
                                             integral_constant<InMemoryDataOperationEnum, Set>{});
                }
            }
            else
            {
                if(arg.k_batch_ > 1)
                {
                    if(has_main_k_block_loop)
                    {
                        ave_time = launch_kernel(
                            integral_constant<bool, true>{},
                            integral_constant<InMemoryDataOperationEnum, AtomicAdd>{});
                    }
                    else
                    {
                        ave_time = launch_kernel(
                            integral_constant<bool, false>{},
                            integral_constant<InMemoryDataOperationEnum, AtomicAdd>{});
                    }
                }
                else
                {
                    if(has_main_k_block_loop)
                    {
                        ave_time =
                            launch_kernel(integral_constant<bool, true>{},
                                          integral_constant<InMemoryDataOperationEnum, Set>{});
                    }
                    else
                    {
                        ave_time =
                            launch_kernel(integral_constant<bool, false>{},
                                          integral_constant<InMemoryDataOperationEnum, Set>{});
                    }
                }
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

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
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

        // For bf16 datatype only kbatch = 1 is supported since there is no AtomicAdd
        // instruction that supports bf16 and we cannot use splitk because of that
        if constexpr(std::is_same<ADataType, ck::bhalf_t>::value)
        {
            supported = supported & (arg.k_batch_ == 1);
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
        str << "DeviceGroupedGemm_Xdl_Fixed_NK"
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

    // polymorphic
    void SetDeviceKernelArgs(BaseArgument* p_arg, void* kernel_args) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->grouped_gemm_kernel_args_dev = kernel_args;
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Xdl_Fixed_NK::Argument structure!");
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg_ptr = dynamic_cast<const Argument*>(p_arg);
        if(arg_ptr)
        {
            return arg_ptr->group_count_ * arg_ptr->barrier_size_grp_ * sizeof(uint32_t);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Xdl_Fixed_NK::Argument structure!");
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    {
        auto arg_ptr = dynamic_cast<const Argument*>(p_arg);
        if(arg_ptr)
        {
            return arg_ptr->group_count_ * sizeof(GroupedGemmKernelArgument<NumDTensor>);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Xdl_Fixed_NK::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& stream_config = StreamConfig{}) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Xdl_Fixed_NK::Argument structure!");

        hip_check_error(
            hipMemsetAsync(p_workspace, 0, GetWorkSpaceSize(arg_ptr), stream_config.stream_id_));
    }

    static void SetKBatch(Argument& arg, index_t k_batch) { arg.UpdateKBatch(k_batch); }

    // polymorphic
    void SetKBatch(BaseArgument* p_arg, index_t k_batch) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->UpdateKBatch(k_batch);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Xdl_Fixed_NK::Argument structure!");
    }

    void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const override
    {
        auto arg_ptr = dynamic_cast<Argument*>(p_arg);
        if(arg_ptr)
        {
            arg_ptr->UpdateKBatch(kbatch);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "DeviceGroupedGemm_Xdl_Fixed_NK::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
