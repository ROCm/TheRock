// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/amd_address_space.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"

namespace ck {

template <typename ABLayout,
          typename ABMajorLayout,
          typename LDSTypeAB,
          index_t BlockSize,
          index_t MNPerBlock,
          index_t KPerBlock,
          index_t MNPerWmma,
          index_t ABK1Value,
          index_t KPack,
          index_t KInner,
          index_t KPerWmmaBlk,
          bool UseBlockPaddingAB,
          bool PermuteAB,
          typename ABBlockTransferThreadClusterLengths_ABK0_MN_ABK1,
          typename ABBlockTransferThreadClusterArrangeOrder,
          typename ABBlockTransferSrcAccessOrder,
          index_t ABBlockTransferSrcVectorDim,
          index_t ABBlockTransferSrcScalarPerVector,
          index_t ABBlockTransferDstScalarPerVector_ABK1,
          bool ABThreadTransferSrcResetCoordinateAfterRun>
struct ABTransferThreadTiles
{
    __device__ static constexpr bool IsLDSNeeded() { return true; }

    static constexpr auto ABK0Number = Number<KPerBlock / ABK1Value>{};
    static constexpr auto ABK1Number = Number<ABK1Value>{};

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t ABPackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<LDSTypeAB>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    template <bool PadMN, bool PadK, typename GridDescriptorBase>
    __host__ __device__ static auto MakeGridDescriptor(const GridDescriptorBase& ab_grid_desc,
                                                       index_t MN,
                                                       index_t MNPad,
                                                       index_t K,
                                                       index_t KPad,
                                                       index_t StrideAB,
                                                       index_t ABK0)
    {

        if constexpr(PadMN && PadK)
        {
            // pad both MN and K
            const auto ab_grid_desc_n_k =
                transform_tensor_descriptor(ab_grid_desc,
                                            make_tuple(make_right_pad_transform(MN, MNPad - MN),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto ab_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                ab_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(ABK0, ABK1Value)),
                           make_pass_through_transform(MNPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return ab_grid_desc_bk0_n_bk1;
        }
        else if constexpr(PadMN && !PadK)
        {
            // pad MN, but not K
            const auto ab_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                ab_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(ABK0, ABK1Value)),
                           make_right_pad_transform(MN, MNPad - MN)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return ab_grid_desc_bk0_n_bk1;
        }
        else if constexpr(!PadMN && PadK)
        {
            // pad K, but not MN
            const auto ab_grid_desc_n_k = transform_tensor_descriptor(
                ab_grid_desc,
                make_tuple(make_pass_through_transform(MN), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto ab_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                ab_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(ABK0, ABK1Value)),
                           make_pass_through_transform(MN)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return ab_grid_desc_bk0_n_bk1;
        }
        else
        {
            if constexpr(!PermuteAB)
            {
                // not pad MN or K
                const auto ab_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                    ab_grid_desc,
                    make_tuple(make_unmerge_transform(make_tuple(ABK0, ABK1Value)),
                               make_pass_through_transform(MN)),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return ab_grid_desc_bk0_n_bk1;
            }
            else
            {
                // Pre-shuffled Weight
                // BGlobal[K / KPerBlock, MN, KPerBlock / K1, K1] -> BTile[K / K1, MN, K1]
                constexpr index_t ABK01 = KPerBlock / ABK1Value;
                const index_t ABK0_     = StrideAB / ABK1Value;
                const index_t ABK00     = ABK0_ / ABK01;

                const auto ab_grid_desc_abk00_mn_abk01_abk1_permute =
                    make_naive_tensor_descriptor_packed(make_tuple(ABK00, MN, ABK01, ABK1Value));

                const auto ab_grid_desc_abk0_mn_abk1_permute = transform_tensor_descriptor(
                    ab_grid_desc_abk00_mn_abk01_abk1_permute,
                    make_tuple(make_merge_transform(make_tuple(ABK00, ABK01)),
                               make_pass_through_transform(make_tuple(MN)),
                               make_pass_through_transform(ABK1Value)),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return ab_grid_desc_abk0_mn_abk1_permute;
            }
        }
    }

    __device__ static constexpr auto GetBlockDescriptor()
    {
        // A matrix in LDS memory, dst of blockwise copy
        if constexpr(UseBlockPaddingAB)
        {
            // bank conflict when writting the data into LDS, but don't worry, we have whole entire
            // loop to hide it in v4. it may give you some benefit from less valu in compute address
            return make_naive_tensor_descriptor(
                make_tuple(ABK0Number, Number<MNPerBlock>{}, ABK1Number),
                make_tuple(Number<MNPerBlock + 1>{} * ABK1Number, ABK1Number, I1));
        }
        // xor tensor transformation request more unnecessary vgpr usage, would cause register spill
        // in some cases.
        else if constexpr(is_same<ABMajorLayout, ABLayout>::value)
        {
            constexpr index_t LdsSize = 32 * 4 / KPerBlock / sizeof(LDSTypeAB) / ABPackedSize;
            constexpr auto MNLdsLayer = LdsSize < 1 ? 1 : LdsSize;
            constexpr auto ab_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(ABK0Number * Number<MNLdsLayer>{},
                           Number<MNPerBlock / MNLdsLayer>{},
                           ABK1Number),
                make_tuple(ABK1Number, Number<KPerBlock * MNLdsLayer>{}, I1));

            constexpr auto ab_lds_block_desc_permuted = transform_tensor_descriptor(
                ab_lds_block_desc,
                make_tuple(
                    make_xor_with_modulo_transform(make_tuple(Number<MNPerBlock / MNLdsLayer>{},
                                                              Number<ABK0Number * MNLdsLayer>{})),
                    make_pass_through_transform(ABK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto ab_lds_block_desc_abk0_mnldslayer_mn_abk1 = transform_tensor_descriptor(
                ab_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(ABK0Number, Number<MNLdsLayer>{})),
                           make_pass_through_transform(Number<MNPerBlock / MNLdsLayer>{}),
                           make_pass_through_transform(ABK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto ab_lds_block_desc_abk0_mn_abk1 = transform_tensor_descriptor(
                ab_lds_block_desc_abk0_mnldslayer_mn_abk1,
                make_tuple(make_pass_through_transform(ABK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<MNPerBlock / MNLdsLayer>{}, Number<MNLdsLayer>{})),
                           make_pass_through_transform(ABK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return ab_lds_block_desc_abk0_mn_abk1;
        }
        else
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto MN0 = ABBlockTransferThreadClusterLengths_ABK0_MN_ABK1{}.At(I1);
            constexpr auto MN1 = MNPerBlock / MN0;

            constexpr auto KThreadWrite = ABBlockTransferThreadClusterLengths_ABK0_MN_ABK1{}.At(I0);
            constexpr auto K0PerThreadWrite = ABK0Number / KThreadWrite;
            constexpr auto KThreadRead      = 64 / MNPerWmma;
            constexpr auto K0PerThreadRead  = ABK0Number / KThreadRead;

            constexpr auto kfold = (ABK1Number * MN0 * sizeof(LDSTypeAB) > 128)
                                       ? 1
                                       : 128 / (ABK1Number * MN0 * sizeof(LDSTypeAB));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=n0
            constexpr auto mpair = (ABK1Number * MNPerWmma * sizeof(LDSTypeAB) > 128)
                                       ? 1
                                       : ((128 / (ABK1Number * MNPerWmma * sizeof(LDSTypeAB))) > MN0
                                              ? MN0
                                              : 128 / (ABK1Number * MNPerWmma * sizeof(LDSTypeAB)));

            constexpr auto ab_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * MN1>{},
                           Number<kfold * MN0 / mpair>{},
                           Number<mpair>{},
                           ABK1Number));

            constexpr auto ab_lds_block_desc_permuted = transform_tensor_descriptor(
                ab_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * MN1>{}, Number<kfold * MN0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(ABK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto ab_lds_block_desc_unmerged = transform_tensor_descriptor(
                ab_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<MN1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<MN0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(ABK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto ab_lds_block_desc_abk0_mn_abk1 = transform_tensor_descriptor(
                ab_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<MN0 / mpair>{}, Number<mpair>{}, Number<MN1>{})),
                           make_pass_through_transform(ABK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return ab_lds_block_desc_abk0_mn_abk1;
        }
    }

    template <typename GridDescriptor,
              typename BlockDescriptor,
              typename ABsDataType,
              typename ABElementwiseOperation,
              index_t GlobalBufferNum>
    __device__ static auto GetBlockTransfer(GridDescriptor& grid_descriptor,
                                            BlockDescriptor& block_descriptor,
                                            ABElementwiseOperation& ab_element_op,
                                            const index_t block_mn_id,
                                            const index_t k_id)
    {
        constexpr index_t NumABTensor = ABsDataType::Size();
        const index_t mn_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_mn_id * MNPerBlock);
        // workaround because v7r2 is not as general as v4r1
        if constexpr(NumABTensor > 1)
        {
            const auto idx_as_block_begin = generate_tuple(
                [&](auto) { return make_multi_index(k_id, mn_block_data_idx_on_grid, 0); },
                Number<NumABTensor>{});

            return ThreadGroupTensorSliceTransfer_v7r2<
                ThisThreadBlock,
                ABsDataType,
                Tuple<LDSTypeAB>,
                GridDescriptor,
                decltype(tie(block_descriptor)),
                ABElementwiseOperation,
                Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
                Sequence<ABK0Number, MNPerBlock, ABK1Number>,
                ABBlockTransferThreadClusterLengths_ABK0_MN_ABK1,
                ABBlockTransferThreadClusterArrangeOrder,
                ABBlockTransferSrcAccessOrder,
                Sequence<1, 0, 2>,
                ABBlockTransferSrcVectorDim,
                2,
                ABBlockTransferSrcScalarPerVector,
                ABBlockTransferDstScalarPerVector_ABK1,
                uniform_sequence_gen_t<NumABTensor, ABThreadTransferSrcResetCoordinateAfterRun>,
                Sequence<true>,
                GlobalBufferNum>{grid_descriptor,
                                 idx_as_block_begin,
                                 tie(block_descriptor),
                                 make_tuple(make_multi_index(0, 0, 0)),
                                 ab_element_op};
        }
        else
        {
            return ThreadGroupTensorSliceTransfer_v4r1<
                ThisThreadBlock,
                ABElementwiseOperation,
                ck::tensor_operation::element_wise::PassThrough,
                InMemoryDataOperationEnum::Set,
                Sequence<ABK0Number, MNPerBlock, ABK1Number>,
                ABBlockTransferThreadClusterLengths_ABK0_MN_ABK1,
                ABBlockTransferThreadClusterArrangeOrder,
                remove_cvref_t<tuple_element_t<0, ABsDataType>>,
                remove_cvref_t<tuple_element_t<0, ABsDataType>>,
                decltype(grid_descriptor[I0]),
                decltype(block_descriptor),
                ABBlockTransferSrcAccessOrder,
                Sequence<0, 1, 2>,
                ABBlockTransferSrcVectorDim,
                2,
                ABBlockTransferSrcScalarPerVector,
                ABBlockTransferDstScalarPerVector_ABK1,
                1,
                1,
                ABThreadTransferSrcResetCoordinateAfterRun,
                true,
                GlobalBufferNum>(grid_descriptor[I0],
                                 make_multi_index(k_id, mn_block_data_idx_on_grid, 0),
                                 ab_element_op,
                                 block_descriptor,
                                 make_multi_index(0, 0, 0),
                                 ck::tensor_operation::element_wise::PassThrough{});
        }
    }

    template <index_t MNRepeat, index_t MNWaves>
    __host__ __device__ static constexpr auto MakeWmmaTileDescriptor()
    {
        // This is a block descriptor used to read LDS memory into register
        // It's defined in a way consistent with the existing implementation to
        // avoid changes in the pipelines
        using BlockDesc = decltype(GetBlockDescriptor());
        // ABK0_MN_ABK1 -> ABK0_MNRepeat_MNWaves_KRow_MNPerWmma_ABK1
        constexpr auto ABK0 = BlockDesc{}.GetLength(I0);
        constexpr auto ABK1 = BlockDesc{}.GetLength(I2);
#ifdef __gfx12__
        constexpr auto KRow = I2;
#else
        constexpr auto KRow = I1;
#endif
        if constexpr(KInner > 1)
        {
            // KPack = KInner * KPerWmma
            // K1 = KInner * KPerWmmaBlk
            // Each thread loads multiple tiles with one instruction
            // 1 - MNRepeat - K0 / KRow - MNWaves - KRow - MNPerWmma - K1
            return transform_tensor_descriptor(
                BlockDesc{},
                make_tuple(make_unmerge_transform(make_tuple(
                               Number<ABK0 / KRow>{}, KRow, Number<KPack / KRow / ABK1>{})),
                           make_unmerge_transform(make_tuple(
                               Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                           make_pass_through_transform(Number<ABK1>{})),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 4, 0>{}, Sequence<1, 3, 5>{}, Sequence<6>{}));
        }
        else
        {
            // KPack = KPerWmma (KInner == 1)
            if constexpr(ABK1 <= KPerWmmaBlk)
            {
                // K1 <= single tile (KPerWmmaBlk)
                // Each thread will load KPerWmmaBlk for the WMMA instruction
                // Since K1 <= single tile, K0 is unmerged first over KPack / KRow / K1
                // (rest of the single WMMA tile for single thread) and then over KRow
                // (rest of the single WMMA tile for single wave)
                // KPack / KRow / K1 - MNRepeat - K0 / KRow - MNWaves - KRow - MNPerWmma - K1
                return transform_tensor_descriptor(
                    BlockDesc{},
                    make_tuple(
                        make_unmerge_transform(make_tuple(
                            Number<ABK0 / (KPack / ABK1)>{}, KRow, Number<KPack / KRow / ABK1>{})),
                        make_unmerge_transform(
                            make_tuple(Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                        make_pass_through_transform(Number<ABK1>{})),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 4, 0>{}, Sequence<1, 3, 5>{}, Sequence<6>{}));
            }
            else
            {
                // K1 > single tile (KPerWmmaBlk)
                // Each thread will load KPerWmmaBlk for the WMMA instruction
                // Since K1 > single tile, each thread loads KPerWmmaBlk and the next
                // KPerWmmaBlk chunk is loaded by a different thread in the same wave (WMMA layout).
                // This layout is needed to support for example AK1 > single tile and
                // BK1 <= single tile in the same gemm
                // KPack / KPerWmmaBlk / KRow - MNRepeat - K0 / KRow - MNWaves - KRow - MNPerWmma -
                // K1
                constexpr auto desc1 = transform_tensor_descriptor(
                    BlockDesc{},
                    make_tuple(
                        make_pass_through_transform(Number<ABK0>{}),
                        make_unmerge_transform(
                            make_tuple(Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                        make_unmerge_transform(make_tuple(Number<ABK1 / KPack>{},
                                                          Number<KPack / KPerWmmaBlk / KRow>{},
                                                          Number<KRow>{},
                                                          Number<KPerWmmaBlk>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2>{}, Sequence<1, 4, 6>{}, Sequence<3, 0, 5, 7>{}));

                return transform_tensor_descriptor(
                    desc1,
                    make_tuple(
                        make_pass_through_transform(Number<KPack / KPerWmmaBlk / KRow>{}),
                        make_pass_through_transform(Number<MNRepeat>{}),
                        make_merge_transform(make_tuple(Number<ABK0>{}, Number<ABK1 / KPack>{})),
                        make_pass_through_transform(Number<MNWaves>{}),
                        make_pass_through_transform(Number<KRow>{}),
                        make_pass_through_transform(Number<MNPerWmma>{}),
                        make_pass_through_transform(Number<KPerWmmaBlk>{})),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2, 3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{},
                               Sequence<7>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{}));
            }
        }
    }

    __device__ static constexpr auto GetBlockStep()
    {
        // Grid descriptor step (MoveSrcSliceWindow)
        return make_multi_index(KPerBlock / ABK1Number, 0, 0);
    }

    template <typename GridDescriptor>
    __device__ static constexpr index_t GetKDimension(const GridDescriptor& grid_desc)
    {
        // K dimension size. This should always be called with the A matrix grid descriptor
        // because it doesn't work for B matrix when packed int4 is used
        return grid_desc.GetLength(I0) * grid_desc.GetLength(I2);
    }

    template <typename LDSType, typename IndexType>
    __device__ static auto GetBuffer(LDSType* p_shared_AB, const IndexType& size)
    {
        return make_dynamic_buffer<AddressSpaceEnum::Lds>(p_shared_AB, size);
    }
};

} // namespace ck
