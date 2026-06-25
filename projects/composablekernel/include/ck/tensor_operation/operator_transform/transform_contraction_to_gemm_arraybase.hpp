// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"

namespace ck {
namespace tensor_operation {

// assume C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          device::TensorSpecialization TensorSpec>
__host__ __device__ static auto
MakeGridDescriptorPair(const std::array<index_t, NumDimG + NumDimM + NumDimN>& gs_ms_ns_lengths_vec,
                       const std::array<index_t, NumDimG + NumDimM + NumDimN>& gs_ms_ns_strides_vec)
{
    // if(!(gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
    //      gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN))
    // {
    //     throw std::runtime_error("wrong! dimension must match input lengths");
    // }

    const auto to_tuple = [&](auto& vec, auto start, auto end) {
        return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
    };

    const auto gs_ms_ns_lengths =
        to_tuple(gs_ms_ns_lengths_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});
    const auto gs_ms_ns_strides =
        to_tuple(gs_ms_ns_strides_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});

    // dimension Ids for G0, G1, ...
    constexpr auto gDimIds = typename arithmetic_sequence_gen<0, NumDimG, 1>::type{};

    // dimension Ids for M0, M1, ...
    constexpr auto mDimIds =
        typename arithmetic_sequence_gen<NumDimG, NumDimG + NumDimM, 1>::type{};

    // dimension Ids for N0, N1, ...
    constexpr auto nDimIds =
        typename arithmetic_sequence_gen<NumDimG + NumDimM, NumDimG + NumDimM + NumDimN, 1>::type{};

    // lengths for G0, G1, ...
    const auto gLengths = get_container_subset(gs_ms_ns_lengths, gDimIds);

    // lengths for M0, M1, ...
    const auto mLengths = get_container_subset(gs_ms_ns_lengths, mDimIds);

    // lengths for N0, N1, ...
    const auto nLengths = get_container_subset(gs_ms_ns_lengths, nDimIds);

    if constexpr(TensorSpec == device::TensorSpecialization::Packed)
    {
        auto G = container_reduce(gLengths, math::multiplies{}, Number<1>{});
        auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
        auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
        const auto grid_desc_g_mraw_nraw = make_naive_tensor_descriptor(
            make_tuple(G, M, N),
            make_tuple(gs_ms_ns_strides[Number<NumDimG - 1>{}],
                       gs_ms_ns_strides[Number<NumDimG + NumDimM - 1>{}],
                       gs_ms_ns_strides[Number<NumDimG + NumDimM + NumDimN - 1>{}]));

        const auto grid_desc_mraw_nraw = make_naive_tensor_descriptor(
            make_tuple(M, N),
            make_tuple(gs_ms_ns_strides[Number<NumDimG + NumDimM - 1>{}],
                       gs_ms_ns_strides[Number<NumDimG + NumDimM + NumDimN - 1>{}]));

        return std::make_pair(grid_desc_g_mraw_nraw, grid_desc_mraw_nraw);
    }
    else
    {
        // naive tensor C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
        const auto grid_desc_gs_ms_ns =
            make_naive_tensor_descriptor(gs_ms_ns_lengths, gs_ms_ns_strides);

        // transformed tensor C[G = G0 * G1 * ..., MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
        // N2 * ...]
        // Note: This does not require padding as it only provides G offset calculation. Technically
        // descriptor for only G is needed. Here we opt for backward compatibility purpose to return
        // G_M_N
        const auto grid_desc_g_mraw_nraw =
            transform_tensor_descriptor(grid_desc_gs_ms_ns,
                                        make_tuple(make_merge_transform(gLengths),
                                                   make_merge_transform(mLengths),
                                                   make_merge_transform(nLengths)),
                                        make_tuple(gDimIds, mDimIds, nDimIds),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto c_ms_ns_lengths = to_tuple(
            gs_ms_ns_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});
        const auto c_ms_ns_strides = to_tuple(
            gs_ms_ns_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});

        // transformed tensor C[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
        // N2 * ...]
        const auto grid_desc_ms_ns = make_naive_tensor_descriptor(c_ms_ns_lengths, c_ms_ns_strides);

        const auto grid_desc_mraw_nraw = transform_tensor_descriptor(
            grid_desc_ms_ns,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
            make_tuple(mDimIds - Number<NumDimG>{}, nDimIds - Number<NumDimG>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return std::make_pair(grid_desc_g_mraw_nraw, grid_desc_mraw_nraw);
    }
}

template <typename NumDims_G_M_N_K_O, // Sequence<>
          typename PerBlock_M_N_K_O,  // Sequence<>
          device::GemmSpecialization GemmSpec,
          device::TensorSpecialization ASpec,
          device::TensorSpecialization B0Spec,
          device::TensorSpecialization B1Spec,
          device::TensorSpecialization CSpec>
struct TransformBatchedContractionContractionToBatchedGemmGemm_Wmma
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    static constexpr index_t NumDimG = NumDims_G_M_N_K_O::At(I0);
    static constexpr index_t NumDimM = NumDims_G_M_N_K_O::At(I1);
    static constexpr index_t NumDimN = NumDims_G_M_N_K_O::At(I2);
    static constexpr index_t NumDimK = NumDims_G_M_N_K_O::At(I3);
    static constexpr index_t NumDimO = NumDims_G_M_N_K_O::At(I4);

    static constexpr index_t MPerBlock = PerBlock_M_N_K_O::At(I0);
    static constexpr index_t NPerBlock = PerBlock_M_N_K_O::At(I1);
    static constexpr index_t KPerBlock = PerBlock_M_N_K_O::At(I2);
    static constexpr index_t OPerBlock = PerBlock_M_N_K_O::At(I3);

    static constexpr auto matrix_padder =
        device::GemmGemmPadder<GemmSpec, index_t, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, KPerBlock, OPerBlock};

    //
    // A
    //
    __host__ __device__ static auto MakeAGridDescriptorPair(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& a_gs_ms_ks_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& a_gs_ms_ks_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimM, NumDimK, ASpec>(a_gs_ms_ks_lengths_vec,
                                                                        a_gs_ms_ks_strides_vec);
    }

    // TODO: rename to G_MRaw_KRaw
    __host__ __device__ static auto MakeAGridDescriptor_G_M_K(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& a_gs_ms_ks_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& a_gs_ms_ks_strides_vec)
    {
        return MakeAGridDescriptorPair(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec).first;
    }
    __host__ __device__ static auto MakeAGridDescriptor_M_K(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& a_gs_ms_ks_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& a_gs_ms_ks_strides_vec)
    {
        return matrix_padder.PadADescriptor_M_K(
            MakeAGridDescriptorPair(a_gs_ms_ks_lengths_vec, a_gs_ms_ks_strides_vec).second);
    }

    template <typename AGridDesc_M_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k, const Number& AK1)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename AGridDesc_M_K,
              typename WmmaK,
              typename MRepeat,
              typename MWaves,
              typename MPerWmma,
              typename AK1>
    __host__ __device__ static constexpr auto
    MakeAGridDescriptor_AKWmma_MBlockRepeat_MWaves_AK0PerWmma_AKRow_MPerWmma_AK1(
        const AGridDesc_M_K& a_grid_desc_m_k,
        const WmmaK&,
        const MRepeat&,
        const MWaves&,
        const MPerWmma&,
        const AK1&)
    {
        const auto M0             = a_grid_desc_m_k.GetLength(I0) / MPerBlock;
        const auto K              = a_grid_desc_m_k.GetLength(I1);
        const auto AKWmma         = K / WmmaK{};
        constexpr auto AKRow      = 2;
        constexpr auto AK0PerWmma = WmmaK{} / AKRow / AK1{};

        return transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_unmerge_transform(
                           make_tuple(AKWmma, Number<AK0PerWmma>{}, Number<AKRow>{}, AK1{})),
                       make_unmerge_transform(make_tuple(M0 * MRepeat{}, MWaves{}, MPerWmma{}))),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
    }

    //
    // B (alias of B0)
    //
    __host__ __device__ static auto MakeB0GridDescriptorPair(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b0_gs_ns_ks_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b0_gs_ns_ks_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimN, NumDimK, B0Spec>(b0_gs_ns_ks_lengths_vec,
                                                                         b0_gs_ns_ks_strides_vec);
    }

    // TODO: rename to G_MRaw_NRaw
    __host__ __device__ static auto MakeB0GridDescriptor_G_N_K(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b0_gs_ns_ks_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b0_gs_ns_ks_strides_vec)
    {
        return MakeB0GridDescriptorPair(b0_gs_ns_ks_lengths_vec, b0_gs_ns_ks_strides_vec).first;
    }
    __host__ __device__ static auto MakeB0GridDescriptor_N_K(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b0_gs_ns_ks_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b0_gs_ns_ks_strides_vec)
    {
        // alias of matrix_padder.PadB0Descriptor_N_K
        return matrix_padder.PadBDescriptor_N_K(
            MakeB0GridDescriptorPair(b0_gs_ns_ks_lengths_vec, b0_gs_ns_ks_strides_vec).second);
    }

    template <typename BGridDesc_N_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeB0GridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k, const Number& BK1)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BGridDesc_L_K,
              typename WmmaK,
              typename LRepeat,
              typename LWaves,
              typename LPerWmma,
              typename BK1>
    __host__ __device__ static constexpr auto
    MakeB0GridDescriptor_BKWmma_LBlockRepeat_LWaves_BK0PerWmma_BKRow_LPerWmma_BK1(
        const BGridDesc_L_K& b_grid_desc_l_k,
        const WmmaK&,
        const LRepeat&,
        const LWaves&,
        const LPerWmma&,
        const BK1&)
    {
        const auto L0             = b_grid_desc_l_k.GetLength(I0) / NPerBlock;
        const auto K              = b_grid_desc_l_k.GetLength(I1);
        const auto BKWmma         = K / WmmaK{};
        constexpr auto BKRow      = 2;
        constexpr auto BK0PerWmma = WmmaK{} / BKRow / BK1{};

        return transform_tensor_descriptor(
            b_grid_desc_l_k,
            make_tuple(make_unmerge_transform(
                           make_tuple(BKWmma, Number<BK0PerWmma>{}, Number<BKRow>{}, BK1{})),
                       make_unmerge_transform(make_tuple(L0 * LRepeat{}, LWaves{}, LPerWmma{}))),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
    }

    //
    // B1
    //
    __host__ __device__ static auto MakeB1GridDescriptorPair(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b1_gs_os_ns_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b1_gs_os_ns_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimO, NumDimN, B1Spec>(b1_gs_os_ns_lengths_vec,
                                                                         b1_gs_os_ns_strides_vec);
    }

    // TODO: rename to G_NRaw_KRaw
    __host__ __device__ static auto MakeB1GridDescriptor_G_N_K(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b1_gs_os_ns_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b1_gs_os_ns_strides_vec)
    {
        return MakeB1GridDescriptorPair(b1_gs_os_ns_lengths_vec, b1_gs_os_ns_strides_vec).first;
    }
    __host__ __device__ static auto MakeB1GridDescriptor_N_K(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b1_gs_os_ns_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& b1_gs_os_ns_strides_vec)
    {
        // alias of matrix_padder.PadB1Descriptor_O_N
        return matrix_padder.PadB1Descriptor_N_K(
            MakeB1GridDescriptorPair(b1_gs_os_ns_lengths_vec, b1_gs_os_ns_strides_vec).second);
    }

    template <typename B1GridDesc_N_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeB1GridDescriptor_BK0_N_BK1(const B1GridDesc_N_K& b1_grid_desc_n_k, const Number& B1K1)
    {
        const auto N = b1_grid_desc_n_k.GetLength(I0);
        const auto K = b1_grid_desc_n_k.GetLength(I1);

        const auto B1K0 = K / B1K1;

        return transform_tensor_descriptor(
            b1_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(B1K0, B1K1)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BGridDesc_N_L,
              typename WmmaL,
              typename NRepeat,
              typename NWaves,
              typename NPerWmma,
              typename BL1>
    __host__ __device__ static constexpr auto
    MakeB1GridDescriptor_BLWmma_NBlockRepeat_NWaves__BL0PerWmma_BLRow_NPerWmma_BL1(
        const BGridDesc_N_L& b_grid_desc_n_l,
        const WmmaL&,
        const NRepeat&,
        const NWaves&,
        const NPerWmma&,
        const BL1&)
    {
        const auto N0             = b_grid_desc_n_l.GetLength(I0) / OPerBlock;
        const auto L              = b_grid_desc_n_l.GetLength(I1);
        const auto BLWmma         = L / WmmaL{};
        constexpr auto BLRow      = 2;
        constexpr auto BL0PerWmma = WmmaL{} / BLRow / BL1{};

        return transform_tensor_descriptor(
            b_grid_desc_n_l,
            make_tuple(make_unmerge_transform(
                           make_tuple(BLWmma, Number<BL0PerWmma>{}, Number<BLRow>{}, BL1{})),
                       make_unmerge_transform(make_tuple(N0 * NRepeat{}, NWaves{}, NPerWmma{}))),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
    }

    //
    // C
    //
    __host__ __device__ static auto MakeCGridDescriptorPair(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& c_gs_ms_os_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& c_gs_ms_os_strides_vec)
    {
        return MakeGridDescriptorPair<NumDimG, NumDimM, NumDimO, CSpec>(c_gs_ms_os_lengths_vec,
                                                                        c_gs_ms_os_strides_vec);
    }

    // TODO: rename to G_MRaw_NRaw
    __host__ __device__ static auto MakeCGridDescriptor_G_M_N(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& c_gs_ms_os_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& c_gs_ms_os_strides_vec)
    {
        return MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).first;
    }
    __host__ __device__ static auto MakeCGridDescriptor_M_N(
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& c_gs_ms_os_lengths_vec,
        const std::array<index_t, NumDimG + NumDimM + NumDimN>& c_gs_ms_os_strides_vec)
    {
        return matrix_padder.PadCDescriptor_M_N(
            MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).second);
    }
};

} // namespace tensor_operation
} // namespace ck
