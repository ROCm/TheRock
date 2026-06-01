// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

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
//   A[G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
//   B[G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
//   D[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
//   E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]

// NOTE: TensorSpecialization::Packed specialized tensor is "packed" in a sense that each inner
// dimension in a dimension group (eg [G0, G1] in Gs, [M0, M1, M2] in Ms, etc.) are contiguous and
// ordered. Not in a sense that the tensor [G0, G1, ..., M0, M1, ..., N0, N1...] can be permuted
// while still being a contiguous, unpadded tensor. In other words, it merely degenerates into
// TensorSpecialization::Default with NumDimG/M/N/K = 1
//
// Detail- Packed tensor satisfies
//   stride_0 = 1
//   stride_i = stride_{i - 1} * extent_{i - 1}
// So tensor
//   [G0, G1, G2, M, N]
// transposed into tensor
//   [G0, G2, G1, M, N]
// with strides
//   [G2 * G1 * M * N, G1 * M * N, M * N, N, 1]
// is again a packed tensor. MakeGridDescriptor() currently just merges dimensions and ignores some
// strides from input tensor extents so finer dimension information is lost. Merging dimensions is
// essentially a degenerated case of TensorSpecialization::Default with NumDimG/M/N/K = 1.
//
// Might need to expose dimension order to the interface to fully support
// TensorSpecialization::Packed in a traditional sense of "packed" tensor
template <index_t NumDimG,
          index_t NumDimM,
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
          TensorSpecialization ASpec,
          TensorSpecialization BSpec,
          TensorSpecialization DESpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceBatchedContractionMultipleD_Wmma_CShuffle
    : public DeviceBatchedContractionMultipleD<NumDimG,
                                               NumDimM,
                                               NumDimN,
                                               NumDimK,
                                               ADataType,
                                               BDataType,
                                               DsDataType,
                                               EDataType,
                                               AElementwiseOperation,
                                               BElementwiseOperation,
                                               CDEElementwiseOperation>
{
    using DeviceOp                      = DeviceBatchedContractionMultipleD_Wmma_CShuffle;
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    // K1 = Max Vector Access Pixels
    static constexpr auto K1Number = Number<K1>{};

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);
    static constexpr auto WmmaK  = K1 == 16 ? 32 : 16;

    static constexpr auto MaxVectorLoadA = K1 * sizeof(ADataType) == 16 ? true : false;
    static constexpr auto MaxVectorLoadB = K1 * sizeof(BDataType) == 16 ? true : false;

    static constexpr auto AEnableLds_auto =
        (NWaves == 1 && (MaxVectorLoadA || MRepeat == 1)) ? false : true;
    static constexpr auto BEnableLds_auto =
        (MWaves == 1 && (MaxVectorLoadB || NRepeat == 1)) ? false : true;

    // If true, LDS is used unconditionally
    static constexpr auto AEnableLds_manu = false;
    static constexpr auto BEnableLds_manu = false;

    static constexpr auto AEnableLds = AEnableLds_auto || AEnableLds_manu || (NumPrefetch > 1);
    static constexpr auto BEnableLds = BEnableLds_auto || BEnableLds_manu || (NumPrefetch > 1);

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    // Assume: A[G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
    static auto MakeAGridDescriptor(const std::vector<index_t>& a_gs_ms_ks_lengths_vec,
                                    const std::vector<index_t>& a_gs_ms_ks_strides_vec)
    {
        assert(a_gs_ms_ks_lengths_vec.size() == NumDimG + NumDimM + NumDimK &&
               a_gs_ms_ks_strides_vec.size() == NumDimG + NumDimM + NumDimK);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto a_ms_ks_lengths = to_tuple(
            a_gs_ms_ks_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimK>{});
        const auto a_ms_ks_strides = to_tuple(
            a_gs_ms_ks_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimK>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(a_ms_ks_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(a_ms_ks_lengths, kDimIds);

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(ASpec == TensorSpecialization::Packed)
            {
                auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
                auto K = container_reduce(kLengths, math::multiplies{}, Number<1>{});
                const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                    make_tuple(M, K),
                    make_tuple(a_ms_ks_strides[Number<NumDimM - 1>{}],
                               a_ms_ks_strides[Number<NumDimM + NumDimK - 1>{}]));
                return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
            }
            else
            {
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
        }();

        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);
        assert(K % K1 == 0);

        if constexpr(AEnableLds)
        {
            const index_t K0 = K / K1;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            constexpr auto A_KRow      = 2;
            constexpr auto A_K0PerWmma = WmmaK / A_KRow / K1Number;
            const auto A_KWmma         = K / WmmaK;

            const auto M0 = M / MPerBlock;
            // 0   1     0         1                2        3             4        5          6
            // M - K <-> A_KWmma - MBlock*MRepeat - MWaves - A_K0PerWmma - A_KRow - MPerWmma - A_K1
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(
                               A_KWmma, Number<A_K0PerWmma>{}, Number<A_KRow>{}, K1Number)),
                           make_unmerge_transform(
                               make_tuple(M0 * MRepeat, Number<MWaves>{}, Number<MPerWmma>{}))),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
        }
    }

    // Assume: B[G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
    static auto MakeBGridDescriptor(const std::vector<index_t>& b_gs_ns_ks_lengths_vec,
                                    const std::vector<index_t>& b_gs_ns_ks_strides_vec)
    {
        assert(b_gs_ns_ks_lengths_vec.size() == NumDimG + NumDimN + NumDimK &&
               b_gs_ns_ks_strides_vec.size() == NumDimG + NumDimN + NumDimK);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto b_ns_ks_lengths = to_tuple(
            b_gs_ns_ks_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimN + NumDimK>{});
        const auto b_ns_ks_strides = to_tuple(
            b_gs_ns_ks_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimN + NumDimK>{});

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(b_ns_ks_lengths, kDimIds);

        // lengths for N0, N1, ...
        const auto nLengths = get_container_subset(b_ns_ks_lengths, nDimIds);

        const auto b_grid_desc_n_k = [&]() {
            if constexpr(BSpec == TensorSpecialization::Packed)
            {
                auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
                auto K = container_reduce(kLengths, math::multiplies{}, Number<1>{});
                const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                    make_tuple(N, K),
                    make_tuple(b_ns_ks_strides[Number<NumDimN - 1>{}],
                               b_ns_ks_strides[Number<NumDimN + NumDimK - 1>{}]));
                return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
            }
            else
            {
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
        }();

        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);
        assert(K % K1 == 0);

        if constexpr(BEnableLds)
        {
            const index_t K0 = K / K1;

            return transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            constexpr auto B_KRow      = 2;
            constexpr auto B_K0PerWmma = WmmaK / B_KRow / K1Number;
            const auto B_KWmma         = K / WmmaK;

            const auto N0 = N / NPerBlock;
            // 0   1     0         1                2        3             4        5          6
            // M - K <-> A_KWmma - MBlock*MRepeat - MWaves - A_K0PerWmma - A_KRow - MPerWmma - A_K1
            return transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(
                               B_KWmma, Number<B_K0PerWmma>{}, Number<B_KRow>{}, K1Number)),
                           make_unmerge_transform(
                               make_tuple(N0 * NRepeat, Number<NWaves>{}, Number<NPerWmma>{}))),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
        }
    }

    // assume E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_M_N(const std::vector<index_t>& e_gs_ms_ns_lengths_vec,
                                        const std::vector<index_t>& e_gs_ms_ns_strides_vec)
    {
        assert(e_gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
               e_gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto e_ms_ns_lengths = to_tuple(
            e_gs_ms_ns_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});
        const auto e_ms_ns_strides = to_tuple(
            e_gs_ms_ns_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_ms_ns_lengths, nDimIds);

        if constexpr(DESpec == TensorSpecialization::Packed)
        {
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            const auto e_grid_desc_mraw_nraw = make_naive_tensor_descriptor(
                make_tuple(M, N),
                make_tuple(e_ms_ns_strides[Number<NumDimM - 1>{}],
                           e_ms_ns_strides[Number<NumDimM + NumDimN - 1>{}]));
            return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
        }
        else
        {
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
    }

    // assume E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_G_M_N(const std::vector<index_t>& e_gs_ms_ns_lengths_vec,
                                          const std::vector<index_t>& e_gs_ms_ns_strides_vec)
    {
        assert(e_gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
               e_gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto e_gs_ms_ns_lengths =
            to_tuple(e_gs_ms_ns_lengths_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});
        const auto e_gs_ms_ns_strides =
            to_tuple(e_gs_ms_ns_strides_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});

        // dimension Ids for G0, G1, ...
        constexpr auto gDimIds = typename arithmetic_sequence_gen<0, NumDimG, 1>::type{};

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds =
            typename arithmetic_sequence_gen<NumDimG, NumDimG + NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<NumDimG + NumDimM,
                                                                  NumDimG + NumDimM + NumDimN,
                                                                  1>::type{};

        // lengths for G0, G1, ...
        const auto gLengths = get_container_subset(e_gs_ms_ns_lengths, gDimIds);

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_gs_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_gs_ms_ns_lengths, nDimIds);

        if constexpr(DESpec == TensorSpecialization::Packed)
        {
            auto G = container_reduce(gLengths, math::multiplies{}, Number<1>{});
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            const auto e_grid_desc_g_mraw_nraw = make_naive_tensor_descriptor(
                make_tuple(G, M, N),
                make_tuple(e_gs_ms_ns_strides[Number<NumDimG - 1>{}],
                           e_gs_ms_ns_strides[Number<NumDimG + NumDimM - 1>{}],
                           e_gs_ms_ns_strides[Number<NumDimG + NumDimM + NumDimN - 1>{}]));
            // return matrix_padder.PadCDescriptor_M_N(e_grid_desc_g_mraw_nraw);
            return e_grid_desc_g_mraw_nraw;
        }
        else
        {
            // naive tensor E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
            const auto e_grid_desc_gs_ms_ns =
                make_naive_tensor_descriptor(e_gs_ms_ns_lengths, e_gs_ms_ns_strides);

            // transformed tensor E[G = G0 * G1 * ..., MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
            // N2 * ...]
            const auto e_grid_desc_g_mraw_nraw = transform_tensor_descriptor(
                e_grid_desc_gs_ms_ns,
                make_tuple(make_merge_transform(gLengths),
                           make_merge_transform(mLengths),
                           make_merge_transform(nLengths)),
                make_tuple(gDimIds, mDimIds, nDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // return matrix_padder.PadCDescriptor_M_N(e_grid_desc_g_mraw_nraw);
            return e_grid_desc_g_mraw_nraw;
        }
    }

    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths_vec,
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceOp::MakeEGridDescriptor_M_N(ds_gs_ms_ns_lengths_vec[i],
                                                         ds_gs_ms_ns_strides_vec[i]);
            },
            Number<NumDTensor>{});
    }

    static auto MakeDsGridDescriptor_G_M_N(
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths_vec,
        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceOp::MakeEGridDescriptor_G_M_N(ds_gs_ms_ns_lengths_vec[i],
                                                           ds_gs_ms_ns_strides_vec[i]);
            },
            Number<NumDTensor>{});
    }

    // Gridwise descriptor, mapping to whole given provblem.
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N({}, {}));

    using DsGridDesc_G_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_G_M_N({}, {}))>;
    using EGridDesc_G_M_N  = decltype(MakeEGridDescriptor_G_M_N({}, {}));

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t batch_stride_A,
                                       index_t batch_stride_B,
                                       DsGridDesc_G_M_N ds_grid_desc_g_m_n,
                                       EGridDesc_G_M_N e_grid_desc_g_m_n)
            : batch_stride_A_(batch_stride_A),
              batch_stride_B_(batch_stride_B),
              ds_grid_desc_g_m_n_(ds_grid_desc_g_m_n),
              e_grid_desc_g_m_n_(e_grid_desc_g_m_n)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(g_idx) * batch_stride_A_;
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(g_idx) * batch_stride_B_;
        }

        __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
        {
            std::array<long_index_t, NumDTensor> ds_offset;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                ds_offset[i] = static_cast<long_index_t>(g_idx) *
                               ds_grid_desc_g_m_n_[i].CalculateOffset(make_multi_index(1, 0, 0));
            });

            return ds_offset;
        }

        __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
        {
            return static_cast<long_index_t>(g_idx) *
                   e_grid_desc_g_m_n_.CalculateOffset(make_multi_index(1, 0, 0));
        }

        private:
        index_t batch_stride_A_;
        index_t batch_stride_B_;
        DsGridDesc_G_M_N ds_grid_desc_g_m_n_;
        EGridDesc_G_M_N e_grid_desc_g_m_n_;
    };

    using AGridDesc = decltype(DeviceOp::MakeAGridDescriptor({}, {}));
    using BGridDesc = decltype(DeviceOp::MakeBGridDescriptor({}, {}));

    // GridwiseOp
    using GridwiseOp = GridwiseGemmMultipleD_Wmma<
        // DataType Family
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        // InMemory Data Descriptor
        AGridDesc,
        BGridDesc,
        DsGridDesc_M_N,
        EGridDesc_M_N,
        // ElementwiseOp Family
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // Tiling Family
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWmma,
        NPerWmma,
        K1,
        MRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        AEnableLds,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BEnableLds,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock,
        NumPrefetch,
        LoopSched,
        PipelineVer>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 const std::vector<index_t>& a_gs_ms_ks_lengths,
                 const std::vector<index_t>& b_gs_ns_ks_lengths,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                 const std::vector<index_t>& e_gs_ms_ns_lengths,
                 const std::vector<index_t>& a_gs_ms_ks_strides,
                 const std::vector<index_t>& b_gs_ns_ks_strides,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                 const std::vector<index_t>& e_gs_ms_ns_strides,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              a_grid_desc_{},
              b_grid_desc_{},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{},
              ds_grid_desc_g_m_n_{
                  DeviceOp::MakeDsGridDescriptor_G_M_N(ds_gs_ms_ns_lengths, ds_gs_ms_ns_strides)},
              e_grid_desc_g_m_n_{
                  DeviceOp::MakeEGridDescriptor_G_M_N(e_gs_ms_ns_lengths, e_gs_ms_ns_strides)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock{},
              e_grid_desc_mblock_mperblock_nblock_nperblock{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_mz_stride_{},
              a_kz_stride_{},
              b_nz_stride_{},
              b_kz_stride_{},
              ds_nz_stride_{},
              e_nz_stride_{},
              a_batch_stride_{a_gs_ms_ks_strides[NumDimG - 1]},
              b_batch_stride_{b_gs_ns_ks_strides[NumDimG - 1]},
              compute_ptr_offset_of_batch_{
                  a_batch_stride_, b_batch_stride_, ds_grid_desc_g_m_n_, e_grid_desc_g_m_n_}
        {
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);
            });

            a_grid_desc_ = DeviceOp::MakeAGridDescriptor(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
            b_grid_desc_ = DeviceOp::MakeBGridDescriptor(b_gs_ns_ks_lengths, b_gs_ns_ks_strides);

            ds_grid_desc_m_n_ =
                DeviceOp::MakeDsGridDescriptor_M_N(ds_gs_ms_ns_lengths, ds_gs_ms_ns_strides);

            e_grid_desc_m_n_ =
                DeviceOp::MakeEGridDescriptor_M_N(e_gs_ms_ns_lengths, e_gs_ms_ns_strides);

            block_2_ctile_map_ = GridwiseOp::MakeDefaultBlock2CTileMap(e_grid_desc_m_n_, M01, N01);

            ds_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseOp::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n_);

            e_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseOp::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n_);

            // for sanity check of vector memory access
            a_mz_stride_ = a_gs_ms_ks_strides[NumDimG + NumDimM - 1];
            a_kz_stride_ = a_gs_ms_ks_strides[NumDimG + NumDimM + NumDimK - 1];
            b_nz_stride_ = b_gs_ns_ks_strides[NumDimG + NumDimN - 1];
            b_kz_stride_ = b_gs_ns_ks_strides[NumDimG + NumDimN + NumDimK - 1];

            for(index_t i = 0; i < NumDTensor; ++i)
            {
                ds_nz_stride_[i] = ds_gs_ms_ns_strides[i][NumDimG + NumDimM + NumDimN - 1];
            }

            e_nz_stride_ = e_gs_ms_ns_strides[NumDimG + NumDimM + NumDimN - 1];
        }

        // Pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseOp::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // Tensor Descriptors
        AGridDesc a_grid_desc_;
        BGridDesc b_grid_desc_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;
        DsGridDesc_G_M_N ds_grid_desc_g_m_n_;
        EGridDesc_G_M_N e_grid_desc_g_m_n_;

        typename GridwiseOp::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock;
        typename GridwiseOp::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock;

        // Block to Tile mapping
        typename GridwiseOp::DefaultBlock2CTileMap block_2_ctile_map_;

        // Idle
        index_t M01_;
        index_t N01_;

        // ElementwiseOp
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // Strides for the last M/N/K dimensions of A/B/Ds/E
        //   for sanity check of vector load/store
        index_t a_mz_stride_;
        index_t a_kz_stride_;
        index_t b_nz_stride_;
        index_t b_kz_stride_;
        std::array<index_t, NumDTensor> ds_nz_stride_;
        index_t e_mz_stride_;
        index_t e_nz_stride_;

        index_t a_batch_stride_;
        index_t b_batch_stride_;

        // Batch Offset
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;

        // for checking vector load/store
        // index_t MRaw_;
        // index_t NRaw_;
        // index_t KRaw_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const index_t G = arg.e_grid_desc_g_m_n_.GetLength(I0);

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) * G;

            const auto K = [&]() {
                if constexpr(AEnableLds)
                {
                    return arg.a_grid_desc_.GetLength(I0) * arg.a_grid_desc_.GetLength(I2);
                }
                else
                {
                    return arg.a_grid_desc_.GetLength(I0) * arg.a_grid_desc_.GetLength(I3) *
                           arg.a_grid_desc_.GetLength(I4) * arg.a_grid_desc_.GetLength(I6);
                }
            }();

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_contraction_multiple_d_wmma_cshuffle<
                    GridwiseOp,
                    ADataType,
                    BDataType,
                    typename GridwiseOp::DsGridPointer,
                    EDataType,
                    DeviceOp::AGridDesc,
                    DeviceOp::BGridDesc,
                    typename GridwiseOp::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseOp::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    ComputePtrOffsetOfStridedBatch,
                    typename GridwiseOp::DefaultBlock2CTileMap,
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
                                              G,
                                              arg.a_grid_desc_,
                                              arg.b_grid_desc_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.compute_ptr_offset_of_batch_,
                                              arg.block_2_ctile_map_);
            };

            if(GridwiseOp::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
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
        if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                printf("DeviceOp: Arch check failure\n");
                return false;
            }
        }
        else
        {
            return false;
        }

        if(!GridwiseOp::CheckValidity(arg.a_grid_desc_,
                                      arg.b_grid_desc_,
                                      arg.ds_grid_desc_m_n_,
                                      arg.e_grid_desc_m_n_,
                                      arg.block_2_ctile_map_))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("GridwiseOp: Validity check failure\n");
            }
            return false;
        }

        // check vector access
        static_assert((ABlockTransferSrcVectorDim == 1 || ABlockTransferSrcVectorDim == 2) &&
                          (BBlockTransferSrcVectorDim == 1 || BBlockTransferSrcVectorDim == 2),
                      "wrong!");

        // vector memory access of A: could be on M or AK1 dimension
        if constexpr(ABlockTransferSrcVectorDim == 1)
        {
            if(!(arg.a_mz_stride_ == 1 &&
                 arg.a_grid_desc_.GetLength(I1) % ABlockTransferSrcScalarPerVector == 0))
            {
                printf("DeviceOp: Vector Access A-m check failure\n");
                return false;
            }
        }
        else
        {
            if(!(arg.a_kz_stride_ == 1))
            {
                index_t LastK =
                    AEnableLds ? arg.a_grid_desc_.GetLength(I2) : arg.a_grid_desc_.GetLength(I6);
                if(LastK % ABlockTransferSrcScalarPerVector == 0)
                {
                    printf("DeviceOp: Vector Access A-k check failure\n");
                    return false;
                }
            }
        }

        // vector memory access of B: could be on N or BK1 dimension
        if constexpr(BBlockTransferSrcVectorDim == 1)
        {
            if(!(arg.b_nz_stride_ == 1 &&
                 arg.b_grid_desc_.GetLength(I1) % BBlockTransferSrcScalarPerVector == 0))
            {
                printf("DeviceOp: Vector Access B-n check failure\n");
                return false;
            }
        }
        else
        {
            if(!(arg.b_kz_stride_ == 1 &&
                 arg.b_grid_desc_.GetLength(I2) % BBlockTransferSrcScalarPerVector == 0))
            {
                printf("DeviceOp: Vector Access B-k check failure\n");
                return false;
            }
        }

        // vector memory access of Ds: always on NPerBlock dimension
        bool valid_d_access = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            if(!(arg.ds_nz_stride_[i] == 1 &&
                 arg.ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetLength(I3) %
                         CDEShuffleBlockTransferScalarPerVector_NPerBlock ==
                     0))
            {
                printf("DeviceOp: Vector Access D-n check failure\n");
                valid_d_access = false;
            }
        });

        if(valid_d_access == false)
        {
            return false;
        }

        // vector memory access of E: always on NPerBlock dimension
        if(!((arg.e_nz_stride_ == 1 &&
              arg.e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I3) %
                      CDEShuffleBlockTransferScalarPerVector_NPerBlock ==
                  0) ||
             CDEShuffleBlockTransferScalarPerVector_NPerBlock == 1))
        {
            printf("DeviceOp: Vector Access E-n check failure\n");
            return false;
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_a,
                 const void* p_b,
                 std::array<const void*, NumDTensor> p_ds,
                 void* p_e,
                 const std::vector<index_t>& a_gs_ms_ks_lengths,
                 const std::vector<index_t>& a_gs_ms_ks_strides,
                 const std::vector<index_t>& b_gs_ns_ks_lengths,
                 const std::vector<index_t>& b_gs_ns_ks_strides,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                 const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                 const std::vector<index_t>& e_gs_ms_ns_lengths,
                 const std::vector<index_t>& e_gs_ms_ns_strides,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_gs_ms_ks_lengths,
                        b_gs_ns_ks_lengths,
                        ds_gs_ms_ns_lengths,
                        e_gs_ms_ns_lengths,
                        a_gs_ms_ks_strides,
                        b_gs_ns_ks_strides,
                        ds_gs_ms_ns_strides,
                        e_gs_ms_ns_strides,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::vector<index_t>& a_gs_ms_ks_lengths,
                        const std::vector<index_t>& a_gs_ms_ks_strides,
                        const std::vector<index_t>& b_gs_ns_ks_lengths,
                        const std::vector<index_t>& b_gs_ns_ks_strides,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                        const std::vector<index_t>& e_gs_ms_ns_lengths,
                        const std::vector<index_t>& e_gs_ms_ns_strides,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_gs_ms_ks_lengths,
                                          b_gs_ns_ks_lengths,
                                          ds_gs_ms_ns_lengths,
                                          e_gs_ms_ns_lengths,
                                          a_gs_ms_ks_strides,
                                          b_gs_ns_ks_strides,
                                          ds_gs_ms_ns_strides,
                                          e_gs_ms_ns_strides,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceBatchedContractionMultipleD_Wmma_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << K1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << MRepeat << ", "
            << NRepeat
            << ">"
            << " AEnableLds: "
            << AEnableLds << ", "
            << "BEnableLds: "
            << BEnableLds << ", "
            << "NumPrefetch: "
            << NumPrefetch << ", "
            << "LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
