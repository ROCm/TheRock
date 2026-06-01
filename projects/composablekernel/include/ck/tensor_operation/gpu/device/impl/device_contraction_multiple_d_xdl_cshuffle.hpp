// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_utils.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatDsPointer,
          typename FloatE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_contraction_multiple_d_xdl_cshuffle(
        const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatDsPointer p_ds_grid,
        FloatE* __restrict__ p_e_grid,
        const AElementwiseOperation a_element_op,
        const BElementwiseOperation b_element_op,
        const CDEElementwiseOperation cde_element_op,
        const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
        const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
        const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock,
        const Block2ETileMap block_2_etile_map)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        GridwiseGemm::template Run<HasMainKBlockLoop, InMemoryDataOperationEnum::Set>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_e_grid,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            block_2_etile_map);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_etile_map;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Tensor Contraction:
//   input : A
//   input : B
//   input : D0, D1, ...
//   output : E
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   A[M0, M1, M2, ..., K0, K1, K2, ...]
//   B[N0, N1, N2, ..., K0, K1, K2, ...]
//   D[M0, M1, M2, ..., N0, N1, N2, ...]
//   E[M0, M1, M2, ..., N0, N1, N2, ...]
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
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
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          typename ComputeDataType = ADataType,
          LoopScheduler LoopSched  = make_default_loop_scheduler()>
struct DeviceContractionMultipleD_Xdl_CShuffle
    : public DeviceContractionMultipleD<NumDimM,
                                        NumDimN,
                                        NumDimK,
                                        ADataType,
                                        BDataType,
                                        DsDataType,
                                        EDataType,
                                        AElementwiseOperation,
                                        BElementwiseOperation,
                                        CDEElementwiseOperation,
                                        ComputeDataType>
{
    using DeviceOp = DeviceContractionMultipleD_Xdl_CShuffle;

    static constexpr auto WarpTileConfig64 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               true>();
    static constexpr auto WarpTileConfig32 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               false>();
    static constexpr auto NXdlPerWave64    = WarpTileConfig64.At(3);
    static constexpr auto NXdlPerWave32    = WarpTileConfig32.At(3);
    static constexpr index_t NumDTensor    = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    // Assume: A[M0, M1, M2, ..., K0, K1, K2, ...]
    static auto MakeAGridDescriptor_M_K(const std::vector<index_t>& a_ms_ks_lengths_vec,
                                        const std::vector<index_t>& a_ms_ks_strides_vec)
    {
        assert(a_ms_ks_lengths_vec.size() == NumDimM + NumDimK &&
               a_ms_ks_strides_vec.size() == NumDimM + NumDimK);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto a_ms_ks_lengths = to_tuple(a_ms_ks_lengths_vec, Number<NumDimM + NumDimK>{});
        const auto a_ms_ks_strides = to_tuple(a_ms_ks_strides_vec, Number<NumDimM + NumDimK>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(a_ms_ks_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(a_ms_ks_lengths, kDimIds);

        // naive tensor A[M0, M1, M2, ..., K0, K1, K2...]
        const auto a_grid_desc_ms_ks =
            make_naive_tensor_descriptor(a_ms_ks_lengths, a_ms_ks_strides);

        // transformed tensor A[MRaw = M0 * M1 * M2 * ... , KRaw = K0 * K1 * K2 * ...]
        const auto a_grid_desc_mraw_kraw = transform_tensor_descriptor(
            a_grid_desc_ms_ks,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(kLengths)),
            make_tuple(mDimIds, kDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    // Assume: B[N0, N1, N2, ..., K0, K1, K2, ...]
    static auto MakeBGridDescriptor_N_K(const std::vector<index_t>& b_ns_ks_lengths_vec,
                                        const std::vector<index_t>& b_ns_ks_strides_vec)
    {
        assert(b_ns_ks_lengths_vec.size() == NumDimN + NumDimK &&
               b_ns_ks_strides_vec.size() == NumDimN + NumDimK);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto b_ns_ks_lengths = to_tuple(b_ns_ks_lengths_vec, Number<NumDimN + NumDimK>{});
        const auto b_ns_ks_strides = to_tuple(b_ns_ks_strides_vec, Number<NumDimN + NumDimK>{});

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(b_ns_ks_lengths, kDimIds);

        // lengths for N0, N1, ...
        const auto nLengths = get_container_subset(b_ns_ks_lengths, nDimIds);

        // naive tensor B[N0, N1, N2, ..., K0, K1, K2, ...]
        const auto b_grid_desc_ns_ks =
            make_naive_tensor_descriptor(b_ns_ks_lengths, b_ns_ks_strides);

        // transformed tensor B[NRaw = N0 * N1 * N2 * ..., KRaw = K0 * K1 * K2 * ...]
        const auto b_grid_desc_nraw_kraw = transform_tensor_descriptor(
            b_grid_desc_ns_ks,
            make_tuple(make_merge_transform(nLengths), make_merge_transform(kLengths)),
            make_tuple(nDimIds, kDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    // assume E[M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_M_N(const std::vector<index_t>& e_ms_ns_lengths_vec,
                                        const std::vector<index_t>& e_ms_ns_strides_vec)
    {
        assert(e_ms_ns_lengths_vec.size() == NumDimM + NumDimN &&
               e_ms_ns_strides_vec.size() == NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto e_ms_ns_lengths = to_tuple(e_ms_ns_lengths_vec, Number<NumDimM + NumDimN>{});
        const auto e_ms_ns_strides = to_tuple(e_ms_ns_strides_vec, Number<NumDimM + NumDimN>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_ms_ns_lengths, nDimIds);

        // naive tensor E[M0, M1, M2, ..., N0, N1, N2...]
        const auto e_grid_desc_ms_ns =
            make_naive_tensor_descriptor(e_ms_ns_lengths, e_ms_ns_strides);

        // transformed tensor E[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 * N2 * ...]
        const auto e_grid_desc_mraw_nraw = transform_tensor_descriptor(
            e_grid_desc_ms_ns,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
            make_tuple(mDimIds, nDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths_vec,
        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceOp::MakeEGridDescriptor_M_N(ds_ms_ns_lengths_vec[i],
                                                         ds_ms_ns_strides_vec[i]);
            },
            Number<NumDTensor>{});
    }

    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K({}, {}));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K({}, {}));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({{}}, {{}}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N({}, {}));

    // GridwiseGemm
    template <typename WarpTileConfig>
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
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        WarpTileConfig::At(0),
        WarpTileConfig::At(1),
        WarpTileConfig::At(2),
        WarpTileConfig::At(3),
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
        WarpTileConfig::At(4),
        WarpTileConfig::At(5),
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;
    using GridwiseGemm64 = GridwiseGemmBase<decltype(WarpTileConfig64)>;
    using GridwiseGemm32 = GridwiseGemmBase<decltype(WarpTileConfig32)>;

    // desc for blockwise copy
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

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 const std::vector<index_t>& a_ms_ks_lengths,
                 const std::vector<index_t>& a_ms_ks_strides,
                 const std::vector<index_t>& b_ns_ks_lengths,
                 const std::vector<index_t>& b_ns_ks_strides,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                 const std::vector<index_t>& e_ms_ns_lengths,
                 const std::vector<index_t>& e_ms_ns_strides,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K(a_ms_ks_lengths, a_ms_ks_strides)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K(b_ns_ks_lengths, b_ns_ks_strides)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N(e_ms_ns_lengths, e_ms_ns_strides)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              block_2_etile_map_{GridwiseGemm64::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N(ds_ms_ns_lengths[i], ds_ms_ns_strides[i]);
            });

            // for sanity check of vector memory access
            tie(a_continous_dim_, a_max_read_elems_) =
                CalculateMaxRead<NumDimM, NumDimK>(a_ms_ks_lengths, a_ms_ks_strides);

            tie(b_continous_dim_, b_max_read_elems_) =
                CalculateMaxRead<NumDimN, NumDimK>(b_ns_ks_lengths, b_ns_ks_strides);

            for(index_t i = 0; i < NumDTensor; ++i)
            {
                tie(ds_continous_dim_[i], ds_max_read_elems_[i]) =
                    CalculateMaxRead<NumDimM, NumDimN>(ds_ms_ns_lengths[i], ds_ms_ns_strides[i]);
            }

            tie(e_continous_dim_, e_max_write_elems_) =
                CalculateMaxRead<NumDimM, NumDimN>(e_ms_ns_lengths, e_ms_ns_strides);
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm64::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

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
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // Describe whether the last part of a given dimension of A/B/D/E is continues dim.
        index_t a_continous_dim_;
        index_t b_continous_dim_;
        std::array<index_t, NumDTensor> ds_continous_dim_;
        index_t e_continous_dim_;

        index_t a_max_read_elems_;
        index_t b_max_read_elems_;
        std::array<index_t, NumDTensor> ds_max_read_elems_;
        index_t e_max_write_elems_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                            arg.b_grid_desc_n_k_,
                                            arg.ds_grid_desc_m_n_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemmMultipleD_xdl_cshuffle has invalid setting");
            }
            auto e_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    arg.e_grid_desc_m_n_);

            auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    arg.ds_grid_desc_m_n_);
            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_contraction_multiple_d_xdl_cshuffle<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::Block2ETileMap,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                              e_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.block_2_etile_map_);
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
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
        if(!ck::is_xdl_wmma_supported<ComputeDataType,
                                      ComputeDataType,
                                      MPerXDL,
                                      NPerXDL,
                                      WarpTileConfig32.At(0),
                                      WarpTileConfig32.At(1)>())
        {
            return false;
        }
        if(!ck::is_lds_direct_load_supported() && std::is_same<ADataType, double>::value)
        {
            return false;
        }
        bool valid = false;
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                valid = GridwiseGemm64::CheckValidity(arg.a_grid_desc_m_k_,
                                                      arg.b_grid_desc_n_k_,
                                                      arg.ds_grid_desc_m_n_,
                                                      arg.e_grid_desc_m_n_,
                                                      arg.block_2_etile_map_);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                valid = GridwiseGemm32::CheckValidity(arg.a_grid_desc_m_k_,
                                                      arg.b_grid_desc_n_k_,
                                                      arg.ds_grid_desc_m_n_,
                                                      arg.e_grid_desc_m_n_,
                                                      arg.block_2_etile_map_);
            }
        }

        if(!valid)
        {
            return false;
        }

        // check vector access
        static_assert((ABlockTransferSrcVectorDim == 1 || ABlockTransferSrcVectorDim == 2) &&
                          (BBlockTransferSrcVectorDim == 1 || BBlockTransferSrcVectorDim == 2),
                      "wrong!");

        const bool valid_a_vector_size =
            arg.a_max_read_elems_ % ABlockTransferSrcScalarPerVector == 0;
        const bool valid_a_access_dim_m =
            ABlockTransferSrcVectorDim == 1 && arg.a_continous_dim_ == 0;
        const bool valid_a_access_dim_k =
            ABlockTransferSrcVectorDim == 2 && arg.a_continous_dim_ == 1;
        const bool valid_a_access_dim =
            valid_a_access_dim_m || valid_a_access_dim_k || ABlockTransferSrcScalarPerVector == 1;
        if(!(valid_a_vector_size && valid_a_access_dim))
        {
            return false;
        }

        const bool valid_b_vector_size =
            arg.b_max_read_elems_ % BBlockTransferSrcScalarPerVector == 0;
        const bool valid_b_access_dim_n =
            BBlockTransferSrcVectorDim == 1 && arg.b_continous_dim_ == 0;
        const bool valid_b_access_dim_k =
            BBlockTransferSrcVectorDim == 2 && arg.b_continous_dim_ == 1;
        const bool valid_b_access_dim =
            valid_b_access_dim_n || valid_b_access_dim_k || BBlockTransferSrcScalarPerVector == 1;
        if(!(valid_b_vector_size && valid_b_access_dim))
        {
            return false;
        }

        bool valid_ds_access = true;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            const bool valid_d_vector_size =
                arg.ds_max_read_elems_[i] % CDEBlockTransferScalarPerVector_NPerBlock == 0;
            // Vector read of Ds is always on N dimension.
            const bool valid_d_access_dim =
                arg.ds_continous_dim_[i] == 1 || CDEBlockTransferScalarPerVector_NPerBlock == 1;
            if(!(valid_d_vector_size && valid_d_access_dim))
            {
                valid_ds_access = false;
            }
        });
        if(!valid_ds_access)
        {
            return false;
        }

        const bool valid_e_vector_size =
            arg.e_max_write_elems_ % CDEBlockTransferScalarPerVector_NPerBlock == 0;
        // Vector write of E is always on N dimension.
        const bool valid_e_access_dim =
            arg.e_continous_dim_ == 1 || CDEBlockTransferScalarPerVector_NPerBlock == 1;
        if(!(valid_e_vector_size && valid_e_access_dim))
        {
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             const std::vector<index_t>& a_ms_ks_lengths,
                             const std::vector<index_t>& a_ms_ks_strides,
                             const std::vector<index_t>& b_ns_ks_lengths,
                             const std::vector<index_t>& b_ns_ks_strides,
                             const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                             const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                             const std::vector<index_t>& e_ms_ns_lengths,
                             const std::vector<index_t>& e_ms_ns_strides,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_ms_ks_lengths,
                        a_ms_ks_strides,
                        b_ns_ks_lengths,
                        b_ns_ks_strides,
                        ds_ms_ns_lengths,
                        ds_ms_ns_strides,
                        e_ms_ns_lengths,
                        e_ms_ns_strides,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::vector<index_t>& a_ms_ks_lengths,
                        const std::vector<index_t>& a_ms_ks_strides,
                        const std::vector<index_t>& b_ns_ks_lengths,
                        const std::vector<index_t>& b_ns_ks_strides,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                        const std::vector<index_t>& e_ms_ns_lengths,
                        const std::vector<index_t>& e_ms_ns_strides,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_ms_ks_lengths,
                                          a_ms_ks_strides,
                                          b_ns_ks_lengths,
                                          b_ns_ks_strides,
                                          ds_ms_ns_lengths,
                                          ds_ms_ns_strides,
                                          e_ms_ns_lengths,
                                          e_ms_ns_strides,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
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
        str << "DeviceContractionMultipleD_Xdl_CShuffle"
            << "<"
            << NumDimM << ", "
            << NumDimN << ", "
            << NumDimK << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << MPerXDL   << ", "
            << NPerXDL   << ", "
            << AK1 << ", "
            << BK1 << ", "
            << ABlockTransferSrcVectorDim << ", "
            << BBlockTransferSrcVectorDim
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
