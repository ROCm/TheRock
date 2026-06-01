// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp"

namespace ck {

template <typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename CShuffleDataType,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          typename CDEElementwiseOperation,
          typename ThisThreadBlock,
          typename BlockwiseGemmPipe>
struct EpilogueCShuffle
    : EpilogueCShuffleBase<DsDataType,
                           EDataType,
                           AccDataType,
                           CShuffleDataType,
                           MPerBlock,
                           NPerBlock,
                           MPerWmma,
                           NPerWmma,
                           MRepeat,
                           NRepeat,
                           CShuffleMRepeatPerShuffle,
                           CShuffleNRepeatPerShuffle,
                           CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                           CDEShuffleBlockTransferScalarPerVectors,
                           CDEElementwiseOperation,
                           ThisThreadBlock,
                           BlockwiseGemmPipe>
{
    using Base = EpilogueCShuffleBase<
        DsDataType,
        EDataType,
        AccDataType,
        CShuffleDataType,
        MPerBlock,
        NPerBlock,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        CDEElementwiseOperation,
        ThisThreadBlock,
        BlockwiseGemmPipe>;

    using Base::GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat;
    using Base::GetCShuffleLDSDescriptor;
    using Base::GetVgprToLDSEpilogueDescriptor;
    using Base::I1;
    using Base::NumDTensor;

    template <InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              typename CThreadBuf,
              typename DsGridPointer,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
    __device__ static void Run(CThreadBuf& c_thread_buf,
                               DsGridPointer p_ds_grid,
                               EDataType* p_e_grid,
                               void* p_shared,
                               const DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               CDEElementwiseOperation& cde_element_op,
                               const index_t& block_m_id,
                               const index_t& block_n_id)
    {
        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // C mapping in single thread.
        constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
            BlockwiseGemmPipe::
                GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

        // LDS buffer
        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared),
            c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat
                .GetElementSpaceSize());

        // Thread transfer Vgpr to LDS
        auto c_thread_copy_vgpr_to_lds = GetVgprToLDSEpilogueDescriptor();

        // Space Filling Curve Vgpr
        constexpr auto sfc_c_vgpr = typename Base::SpaceFillingCurveVgpr{};

        // Space Filling Curve Vmem
        constexpr auto sfc_cde_global = typename Base::SpaceFillingCurveVmem{};

        // Block descriptor
        constexpr auto
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
                GetCShuffleLDSDescriptor();

        // tuple of reference to C/Ds tensor descriptors
        const auto c_ds_desc_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                         Number<NumDTensor>{}));

        // Thread transfer LDS to Vmem
        auto cde_shuffle_block_copy_lds_and_global =
            Base::template GetLDSToVmemEpilogueDescriptor<EGlobalMemoryDataOperation, EDataType>(
                c_ds_desc_refs,
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                cde_element_op,
                block_m_id,
                block_n_id);

        // tuple of reference to C/Ds tensor buffers
        const auto c_ds_buf_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_buf),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_buf[i]; },
                         Number<NumDTensor>{}));

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

        static_assert(num_access == sfc_cde_global.GetNumOfAccess(), "wrong!");

        // CShuffle and Store
        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to write to LDS
            block_sync_lds();

            // each thread write its data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(
                c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                c_thread_buf,
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                c_shuffle_block_buf);

            // make sure it's safe to read from LDS
            block_sync_lds();

            // each block loads its C data from LDS, D from global, applies elementwise
            // operation and stores result E to global
            cde_shuffle_block_copy_lds_and_global.Run(
                c_ds_desc_refs,
                c_ds_buf_refs,
                tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                tie(e_grid_buf));

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto cde_global_step = sfc_cde_global.GetForwardStep(access_id);
                // move on Ds
                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    cde_shuffle_block_copy_lds_and_global.MoveSrcSliceWindow(
                        c_ds_desc_refs, i + I1, cde_global_step);
                });

                // move on E
                cde_shuffle_block_copy_lds_and_global.MoveDstSliceWindow(
                    tie(e_grid_desc_mblock_mperblock_nblock_nperblock), cde_global_step);
            }
        });
    }
};

} // namespace ck
