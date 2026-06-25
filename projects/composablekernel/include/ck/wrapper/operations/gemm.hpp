// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/wrapper/utils/tensor_utils.hpp"
#include "ck/wrapper/traits/blockwise_gemm_xdl_traits.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace ck {
namespace wrapper {
/// @endcond

// Disable from doxygen docs generation
/// @cond INTERNAL
namespace {
namespace detail {
/**
 * \brief Create block descriptor (K0, MPerBlock or NPerBlock, K1).
 *
 *
 * \tparam K1 The number of K-dim elements that are packed together as a separate logical dimension.
 * \tparam TileLayout Tensor data tile layout (M,K) or (N,K).
 *
 * \return Block descriptor (K0, MPerBlock or NPerBlock, K1)
 */
template <index_t K1, typename TileLayout>
__device__ constexpr auto GetBlockDescriptor()
{
    using TileLayoutShape      = typename TileLayout::LayoutShape;
    using TileLayoutDescriptor = typename TileLayout::LayoutUnrolledDescriptorType;

    constexpr auto K0PerBlock = Number<size<1>(TileLayoutShape{})>{} / Number<K1>{};
    // MPerBlock or NPerBlock
    constexpr auto Dim0 = Number<size<0>(TileLayoutShape{})>{};

    constexpr auto a_block_desc_k0_m_k1 = transform_tensor_descriptor(
        TileLayoutDescriptor{},
        make_tuple(make_unmerge_transform(make_tuple(K0PerBlock, Number<K1>{})),
                   make_pass_through_transform(Dim0)),
        make_tuple(Sequence<1>{}, Sequence<0>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    return a_block_desc_k0_m_k1;
}

} // namespace detail
} // namespace
/// @endcond

/**
 * \brief Perform blockwise gemm xdl on tensors stored in lds. Result will be
 * stored in Vgpr register. A data layout must be (MPerBlock, KPerBlock) or
 * (K0PerBlock, MPerBlock, K1) and B data layout must be (NPerBlock, KPerBlock)
 * or (K0PerBlock, NPerBlock, K1).
 *
 * \note C output Vgpr register layout (8D):
 * - MXdlPerWave - The number of MFMA instructions run by single wave in M
 *                 dimension per tile.
 * - NXdlPerWave - The number of MFMA instructions run by single wave in N
 *                 dimension per tile.
 * - MWave - Equals to 1 since this is for single wave.
 * - NWave - Equals to 1 since this is for single wave.
 * - NumGroupsPerBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - NumInputsBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - GroupSize - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - NumThreadsPerBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 *
 * \tparam DataType Input data types.
 * \tparam BlockSize Tensor to pad.
 * \tparam GemmTraits Traits of gemm xdl operation.
 * \param a_local_tile_tensor A tensor in LDS memory for blockwise gemm
 * (MPerBlock, KPerBlock) or (K0PerBlock, MPerBlock, K1) layout.
 * \param b_local_tile_tensor B tensor in LDS memory for blockwise gemm
 * (NPerBlock, KPerBlock) or (K0PerBlock, NPerBlock, K1) layout.
 * \param c_reg_tensor C tensor VGPR memory for blockwise gemm.
 */
template <typename DataType,
          index_t BlockSize,
          typename GemmTraits,
          typename ATensorType,
          typename BTensorType,
          typename CTensorType>
__device__ void blockwise_gemm_xdl(const ATensorType& a_local_tile_tensor,
                                   const BTensorType& b_local_tile_tensor,
                                   CTensorType& c_reg_tensor)
{
    constexpr auto I3 = Number<3>{};

    static_assert(ATensorType::TensorBufferAddressSpace == MemoryTypeEnum::Lds);
    static_assert(BTensorType::TensorBufferAddressSpace == MemoryTypeEnum::Lds);
    static_assert(CTensorType::TensorBufferAddressSpace == MemoryTypeEnum::Vgpr);
    static_assert(is_same_v<DataType, typename ATensorType::TensorElementType>);
    static_assert(is_same_v<DataType, typename BTensorType::TensorElementType>);

    constexpr bool is_integer =
        is_same_v<DataType, int8_t> || is_same_v<DataType, int16_t> || is_same_v<DataType, int32_t>;
    using GemmAccDataType = std::conditional_t<is_integer, int32_t, float>;

    using ATileLayout = remove_cvref_t<decltype(layout(a_local_tile_tensor))>;
    using BTileLayout = remove_cvref_t<decltype(layout(b_local_tile_tensor))>;

    static_assert(typename ATileLayout::LayoutShape{}.Size() ==
                  typename BTileLayout::LayoutShape{}.Size());
    constexpr bool is_3d_desc = typename ATileLayout::LayoutShape{}.Size() == I3;

    using ABlockDesc_K0_M_K1_Type =
        conditional_t<is_3d_desc,
                      typename ATileLayout::LayoutUnrolledDescriptorType,
                      decltype(detail::GetBlockDescriptor<GemmTraits::K1, ATileLayout>())>;
    using BBlockDesc_K0_N_K1_Type =
        conditional_t<is_3d_desc,
                      typename BTileLayout::LayoutUnrolledDescriptorType,
                      decltype(detail::GetBlockDescriptor<GemmTraits::K1, BTileLayout>())>;

    BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                        DataType,
                                                        DataType,
                                                        GemmAccDataType,
                                                        ABlockDesc_K0_M_K1_Type,
                                                        BBlockDesc_K0_N_K1_Type,
                                                        GemmTraits::MPerXDL,
                                                        GemmTraits::NPerXDL,
                                                        GemmTraits::MXdlPerWave,
                                                        GemmTraits::NXdlPerWave,
                                                        GemmTraits::K1>
        blockwise_gemm_xdl_op{};

    blockwise_gemm_xdl_op.Run(
        a_local_tile_tensor.GetBuffer(), b_local_tile_tensor.GetBuffer(), c_reg_tensor.GetBuffer());
}

/**
 * \brief Create local partition per thread for C tensor.
 *
 * \note C output global memory layout (8D):
 * - MXdlPerWave - The number of MFMA instructions run by single wave in M
 *                 dimension.
 * - NXdlPerWave - The number of MFMA instructions run by single wave in N
 *                 dimension.
 * - MWave - The number of waves in single tile M dimension per tile.
 * - NWave - The number of waves in single tile N dimension per tile.
 * - NumGroupsPerBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - NumInputsBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - GroupSize - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - NumThreadsPerBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 *
 * \tparam DataType Input data types.
 * \tparam ATileLayout A tensor layout.
 * \tparam BTileLayout B tensor layout.
 * \tparam BlockSize Number of threads in block.
 * \tparam GemmTraits Traits of gemm xdl operation.
 * \param c_local_tile_tensor C tensor in LDS memory for blockwise gemm
 * (MPerBlock, NPerBlock) layout.
 *
 * \return Partition c tensor for blockwise gemm.
 */
template <typename DataType,
          typename ATileLayout,
          typename BTileLayout,
          index_t BlockSize,
          typename GemmTraits,
          typename CTensorType>
__host__ __device__ constexpr auto
make_blockwise_gemm_xdl_c_local_partition(CTensorType& c_local_tile_tensor)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};

    static_assert(typename ATileLayout::LayoutShape{}.Size() ==
                  typename BTileLayout::LayoutShape{}.Size());

    constexpr bool is_integer =
        is_same_v<DataType, int8_t> || is_same_v<DataType, int16_t> || is_same_v<DataType, int32_t>;
    using GemmAccDataType = std::conditional_t<is_integer, int32_t, float>;

    constexpr bool is_3d_desc = typename ATileLayout::LayoutShape{}.Size() == I3;
    using ABlockDesc_K0_M_K1_Type =
        conditional_t<is_3d_desc,
                      typename ATileLayout::LayoutUnrolledDescriptorType,
                      decltype(detail::GetBlockDescriptor<GemmTraits::K1, ATileLayout>())>;
    using BBlockDesc_K0_N_K1_Type =
        conditional_t<is_3d_desc,
                      typename BTileLayout::LayoutUnrolledDescriptorType,
                      decltype(detail::GetBlockDescriptor<GemmTraits::K1, BTileLayout>())>;

    using BlockwiseGemmXdlops =
        BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                            DataType,
                                                            DataType,
                                                            GemmAccDataType,
                                                            ABlockDesc_K0_M_K1_Type,
                                                            BBlockDesc_K0_N_K1_Type,
                                                            GemmTraits::MPerXDL,
                                                            GemmTraits::NPerXDL,
                                                            GemmTraits::MXdlPerWave,
                                                            GemmTraits::NXdlPerWave,
                                                            GemmTraits::K1>;

    constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
        BlockwiseGemmXdlops::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
    constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
    constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
    constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
    constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
    constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
    constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
    constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
    constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

    // Calculate offset on grid
    const auto c_thread_mtx_on_block =
        BlockwiseGemmXdlops::CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

    const index_t m_thread_data_on_grid =
        c_local_tile_tensor.GetMultiIdxOffsets()[I0] + c_thread_mtx_on_block[I0];

    const index_t n_thread_data_on_grid =
        c_local_tile_tensor.GetMultiIdxOffsets()[I1] + c_thread_mtx_on_block[I1];

    const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
        make_tuple(Sequence<0, 1, 2, 3, 4>{}),
        make_tuple(Sequence<0>{}));

    const auto m_thread_data_on_grid_idx =
        m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
            make_multi_index(m_thread_data_on_grid));

    const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor =
        make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                                         make_tuple(Sequence<0, 1, 2>{}),
                                         make_tuple(Sequence<0>{}));

    const auto n_thread_data_on_grid_idx =
        n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
            make_multi_index(n_thread_data_on_grid));
    // Create partition shape based on descriptor dims.
    const auto partition_shape = make_tuple(M0, N0, I1, I1, M2, I1, M4, I1);

    const auto partition_desc = BlockwiseGemmXdlops::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(
        layout(c_local_tile_tensor).GetUnrolledDescriptor());

    const auto lower_upper_dims =
        generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<8>{});

    auto sliced_desc = transform_tensor_descriptor(
        partition_desc,
        make_tuple(
            make_slice_transform(partition_shape.At(Number<0>{}),
                                 m_thread_data_on_grid_idx[I0],
                                 partition_shape.At(Number<0>{}) + m_thread_data_on_grid_idx[I0]),
            make_slice_transform(partition_shape.At(Number<1>{}),
                                 n_thread_data_on_grid_idx[I0],
                                 partition_shape.At(Number<1>{}) + n_thread_data_on_grid_idx[I0]),
            make_slice_transform(partition_shape.At(Number<2>{}),
                                 m_thread_data_on_grid_idx[I1],
                                 partition_shape.At(Number<2>{}) + m_thread_data_on_grid_idx[I1]),
            make_slice_transform(partition_shape.At(Number<3>{}),
                                 n_thread_data_on_grid_idx[I1],
                                 partition_shape.At(Number<3>{}) + n_thread_data_on_grid_idx[I1]),
            make_slice_transform(partition_shape.At(Number<4>{}),
                                 m_thread_data_on_grid_idx[I2],
                                 partition_shape.At(Number<4>{}) + m_thread_data_on_grid_idx[I2]),
            make_slice_transform(partition_shape.At(Number<5>{}),
                                 m_thread_data_on_grid_idx[I3],
                                 partition_shape.At(Number<5>{}) + m_thread_data_on_grid_idx[I3]),
            make_slice_transform(partition_shape.At(Number<6>{}),
                                 m_thread_data_on_grid_idx[I4],
                                 partition_shape.At(Number<6>{}) + m_thread_data_on_grid_idx[I4]),
            make_slice_transform(partition_shape.At(Number<7>{}),
                                 n_thread_data_on_grid_idx[I2],
                                 partition_shape.At(Number<7>{}) + n_thread_data_on_grid_idx[I2])),
        lower_upper_dims,
        lower_upper_dims);

    const auto partition_layout =
        Layout<remove_reference_t<decltype(partition_shape)>, decltype(sliced_desc)>(
            partition_shape, sliced_desc);
    auto partition_tensor = make_tensor<CTensorType::TensorBufferAddressSpace>(
        c_local_tile_tensor.GetPointer(), partition_layout);
    return partition_tensor;
}

/**
 * \brief Create local partition per thread for C tensor.
 *
 * \note C output Vgpr register layout (8D):
 * - MXdlPerWave - The number of MFMA instructions run by single wave in M
 *                 dimension per tile.
 * - NXdlPerWave - The number of MFMA instructions run by single wave in N
 *                 dimension per tile.
 * - MWave - Equals to 1 since this is for single wave.
 * - NWave - Equals to 1 since this is for single wave.
 * - NumGroupsPerBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - NumInputsBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - GroupSize - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 * - NumThreadsPerBlock - Mfma instruction internal layout (depeneds on the
 *                       instruction size).
 *
 * \tparam DataType Input data types.
 * \tparam ATileLayout A tensor layout.
 * \tparam BTileLayout B tensor layout.
 * \tparam BlockSize Number of threads in block.
 * \tparam GemmTraits Traits of gemm xdl operation.
 *
 * \return Vgpr c tensor for blockwise gemm.
 */
template <typename DataType,
          typename ATileLayout,
          typename BTileLayout,
          index_t BlockSize,
          typename GemmTraits>
__host__ __device__ constexpr auto make_blockwise_gemm_xdl_c_vgpr()
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};

    static_assert(typename ATileLayout::LayoutShape{}.Size() ==
                  typename BTileLayout::LayoutShape{}.Size());

    constexpr bool is_integer =
        is_same_v<DataType, int8_t> || is_same_v<DataType, int16_t> || is_same_v<DataType, int32_t>;
    using GemmAccDataType = std::conditional_t<is_integer, int32_t, float>;

    constexpr bool is_3d_desc = typename ATileLayout::LayoutShape{}.Size() == I3;
    using ABlockDesc_K0_M_K1_Type =
        conditional_t<is_3d_desc,
                      typename ATileLayout::LayoutUnrolledDescriptorType,
                      decltype(detail::GetBlockDescriptor<GemmTraits::K1, ATileLayout>())>;
    using BBlockDesc_K0_N_K1_Type =
        conditional_t<is_3d_desc,
                      typename BTileLayout::LayoutUnrolledDescriptorType,
                      decltype(detail::GetBlockDescriptor<GemmTraits::K1, BTileLayout>())>;

    using BlockwiseGemmXdlops =
        BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                            DataType,
                                                            DataType,
                                                            GemmAccDataType,
                                                            ABlockDesc_K0_M_K1_Type,
                                                            BBlockDesc_K0_N_K1_Type,
                                                            GemmTraits::MPerXDL,
                                                            GemmTraits::NPerXDL,
                                                            GemmTraits::MXdlPerWave,
                                                            GemmTraits::NXdlPerWave,
                                                            GemmTraits::K1>;
    // Calcualte descriptor, shape and layout
    constexpr auto vgpr_desc = BlockwiseGemmXdlops::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
    const auto vgpr_shape    = make_tuple(vgpr_desc.GetLengths()[I0],
                                       vgpr_desc.GetLengths()[I1],
                                       vgpr_desc.GetLengths()[I2],
                                       vgpr_desc.GetLengths()[I3],
                                       vgpr_desc.GetLengths()[I4],
                                       vgpr_desc.GetLengths()[I5],
                                       vgpr_desc.GetLengths()[I6],
                                       vgpr_desc.GetLengths()[I7]);
    const auto vgpr_layout = Layout<remove_reference_t<decltype(vgpr_shape)>, decltype(vgpr_desc)>(
        vgpr_shape, vgpr_desc);
    // Get vector type for Vgpr
    constexpr index_t ScalarPerVector = BlockwiseGemmXdlops::xdlops_gemm.GetRegSizePerXdlops();
    using VgprVectorType = typename vector_type<GemmAccDataType, ScalarPerVector>::type;
    return ck::wrapper::make_register_tensor<ck::wrapper::MemoryTypeEnum::Vgpr, VgprVectorType>(
        vgpr_layout);
}

} // namespace wrapper
} // namespace ck
