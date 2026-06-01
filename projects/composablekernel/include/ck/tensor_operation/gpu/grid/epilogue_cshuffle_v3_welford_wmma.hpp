// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"

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
          typename BlockwiseGemmPipe,
          index_t BlockSize>
struct EpilogueWelfordCShuffle
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
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using Base::NumDTensor;

    template <typename DoPads, index_t MPerTile, index_t NPerTile>
    __host__ __device__ static auto MakeMeanVarDescriptor_M_N(index_t M, index_t N)
    {
        const auto grid_desc_m_n =
            make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(N, I1));
        return tensor_operation::device::PadTensorDescriptor(
            grid_desc_m_n, make_tuple(MPerTile, NPerTile), DoPads{});
    }

    template <typename DoPads, index_t MPerTile, index_t NPerTile>
    __host__ __device__ static auto MakeCountDescriptor_M_N(index_t M, index_t N)
    {
        // We will broadcast [N] to [M, N] in this descriptor
        // Hence, 1st stride is 0
        const auto grid_desc_m_n =
            make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I0, I1));
        return tensor_operation::device::PadTensorDescriptor(
            grid_desc_m_n, make_tuple(MPerTile, NPerTile), DoPads{});
    }

    template <typename GridDescriptor_M_N>
    __host__ __device__ static constexpr auto
    MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(const GridDescriptor_M_N& grid_desc_m_n)
    {
        const auto M      = grid_desc_m_n.GetLength(I0);
        const auto NBlock = grid_desc_m_n.GetLength(I1);
        const auto MBlock = M / MPerBlock;

        const auto grid_desc_mblock_mperblock_nblock = transform_tensor_descriptor(
            grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_pass_through_transform(NBlock)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        return grid_desc_mblock_mperblock_nblock;
    }

    using GemmMeanVarGridDesc_M_N =
        decltype(MakeMeanVarDescriptor_M_N<Sequence<true, false>, MPerBlock, 1>(1, 1));

    using GemmCountGridDesc_M_N =
        decltype(MakeCountDescriptor_M_N<Sequence<true, false>, MPerBlock, 1>(1, 1));

    __device__ EpilogueWelfordCShuffle(EDataType* p_welford_mean_grid_,
                                       EDataType* p_welford_var_grid_,
                                       int32_t* p_welford_count_grid_,
                                       index_t MRaw_,
                                       index_t NRaw_)
        : p_welford_mean_grid(p_welford_mean_grid_),
          p_welford_var_grid(p_welford_var_grid_),
          p_welford_count_grid(p_welford_count_grid_),
          NRaw(NRaw_)
    {
        index_t gemm_nblock = math::integer_divide_ceil(NRaw_, NPerBlock);

        gemm_mean_var_grid_desc_m_nblock =
            MakeMeanVarDescriptor_M_N<Sequence<true, false>, MPerBlock, 1>(MRaw_, gemm_nblock);

        gemm_count_grid_desc_m_nblock =
            MakeCountDescriptor_M_N<Sequence<true, false>, MPerBlock, 1>(MRaw_, gemm_nblock);
    }

    template <InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              typename CThreadBuf,
              typename DsGridPointer,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
    __device__ void Run(CThreadBuf& c_thread_buf,
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
        // Vmem buffers
        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        auto mean_var_grid_desc_mblock_mperblock_nblock =
            MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
                gemm_mean_var_grid_desc_m_nblock);

        auto mean_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_mean_grid, mean_var_grid_desc_mblock_mperblock_nblock.GetElementSpaceSize());

        auto var_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_var_grid, mean_var_grid_desc_mblock_mperblock_nblock.GetElementSpaceSize());

        auto count_grid_desc_mblock_mperblock_nblock =
            MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(gemm_count_grid_desc_m_nblock);
        auto welford_count_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_count_grid, count_grid_desc_mblock_mperblock_nblock.GetElementSpaceSize());

        // LDS buffer
        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared),
            c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat
                .GetElementSpaceSize());

        // tuple of reference to C/Ds tensor buffers (mix LDS and Vmem)
        const auto c_ds_buf_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_buf),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_buf[i]; },
                         Number<NumDTensor>{}));

        // Thread transfer Vgpr to LDS
        auto c_thread_copy_vgpr_to_lds = GetVgprToLDSEpilogueDescriptor();

        // Space Filling Curve Vgpr
        constexpr auto sfc_c_vgpr = typename Base::SpaceFillingCurveVgpr{};

        // Space Filling Curve Vmem
        constexpr auto sfc_cde_global = typename Base::SpaceFillingCurveVmem{};

        // C thread descriptor
        constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
            BlockwiseGemmPipe::
                GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

        // tuple of reference to C/Ds tensor descriptors
        const auto c_ds_desc_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                         Number<NumDTensor>{}));

        // Thread transfer LDS to Vmem
        auto cde_shuffle_block_copy_lds_and_global =
            Base::template GetLDSToVmemEpilogueDescriptor<EGlobalMemoryDataOperation, AccDataType>(
                c_ds_desc_refs,
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                cde_element_op,
                block_m_id,
                block_n_id);

        // Block descriptor
        constexpr auto
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
                GetCShuffleLDSDescriptor();

        // E Vgpr buffer
        constexpr index_t PostShuffleThreadSliceSize_M =
            (CShuffleMRepeatPerShuffle * BlockwiseGemmPipe::MWaves * MPerWmma) /
            CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I1);

        constexpr index_t PostShuffleThreadSliceSize_N =
            (CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves * NPerWmma) /
            CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I3);

        constexpr auto PostShuffleThreadSliceSize_M_N =
            Sequence<PostShuffleThreadSliceSize_M, PostShuffleThreadSliceSize_N>{};

        // Welford
        constexpr auto post_shuffle_thread_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{},
                                                           Number<PostShuffleThreadSliceSize_M>{},
                                                           Number<1>{},
                                                           Number<PostShuffleThreadSliceSize_N>{}));

        auto e_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            post_shuffle_thread_desc_m_n.GetElementSpaceSize());

        using PostShuffleThreadClusterSize_M_N = Sequence<
            CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I1),
            CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I3)>;

        constexpr auto post_shuffle_thread_cluster_desc =
            make_cluster_descriptor(PostShuffleThreadClusterSize_M_N{}, Sequence<0, 1>{});

        const auto post_shuffle_thread_cluster_idx =
            post_shuffle_thread_cluster_desc.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

        const auto post_shuffle_thread_data_idx_begin =
            post_shuffle_thread_cluster_idx * PostShuffleThreadSliceSize_M_N;

        constexpr auto thread_welford_src_desc_m_k = make_naive_tensor_descriptor_packed(make_tuple(
            Number<PostShuffleThreadSliceSize_M>{}, Number<PostShuffleThreadSliceSize_N>{}));

        constexpr auto thread_welford_dst_desc_m =
            make_naive_tensor_descriptor_packed(make_tuple(Number<PostShuffleThreadSliceSize_M>{}));

        using ThreadwiseWelford = ThreadwiseWelford<AccDataType,
                                                    decltype(thread_welford_src_desc_m_k),
                                                    decltype(thread_welford_dst_desc_m)>;

        using BlockwiseWelford = BlockwiseWelford<AccDataType,
                                                  BlockSize,
                                                  PostShuffleThreadClusterSize_M_N,
                                                  Sequence<0, 1>,
                                                  false>;

        constexpr int num_shuffleM =
            MPerBlock / (CShuffleMRepeatPerShuffle * BlockwiseGemmPipe::MWaves * MPerWmma);

        constexpr int num_shuffleN =
            NPerBlock / (CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves * NPerWmma);

        using mean_var_vgpr_type = decltype(make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            thread_welford_dst_desc_m.GetElementSpaceSize()));

        using welford_count_vgpr_type =
            decltype(make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                thread_welford_dst_desc_m.GetElementSpaceSize()));

        Array<ThreadwiseWelford, num_shuffleM> threadwise_welfords;
        Array<mean_var_vgpr_type, num_shuffleM> mean_thread_bufs;
        Array<mean_var_vgpr_type, num_shuffleM> var_thread_bufs;
        Array<welford_count_vgpr_type, num_shuffleM> welford_count_thread_bufs;

        int max_count     = PostShuffleThreadSliceSize_N * num_shuffleN;
        const auto nblock = mean_var_grid_desc_mblock_mperblock_nblock.GetLength(I2);

        // tail block
        if(block_n_id % nblock == nblock - 1)
        {
            constexpr index_t NPerShuffleBlock =
                CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves * NPerWmma;

            int NPerBlockTail = NRaw - NPerBlock * (nblock - 1);
            int thread_max_len =
                PostShuffleThreadSliceSize_N * (post_shuffle_thread_cluster_idx[I1] + 1);
            int shuffle_step = 0;
            while(thread_max_len <= NPerBlockTail && shuffle_step < num_shuffleN)
            {
                ++shuffle_step;
                thread_max_len += NPerShuffleBlock;
            }

            int delta = 0;
            if(thread_max_len - NPerBlockTail > PostShuffleThreadSliceSize_N)
                delta = 0;
            else if(NPerBlockTail > thread_max_len)
                delta = PostShuffleThreadSliceSize_N;
            else
                delta = PostShuffleThreadSliceSize_N - thread_max_len + NPerBlockTail;

            max_count = shuffle_step * PostShuffleThreadSliceSize_N + delta;
        }

        // Initialize Welford
        static_for<0, num_shuffleM, 1>{}([&](auto i) {
            threadwise_welfords(i).max_count_ = max_count;
            mean_thread_bufs(i) = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                thread_welford_dst_desc_m.GetElementSpaceSize());

            var_thread_bufs(i) = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                thread_welford_dst_desc_m.GetElementSpaceSize());

            welford_count_thread_bufs(i) = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                thread_welford_dst_desc_m.GetElementSpaceSize());

            static_for<0, PostShuffleThreadSliceSize_M, 1>{}([&](auto j) {
                mean_thread_bufs(i)(j)          = type_convert<AccDataType>(0.0f);
                var_thread_bufs(i)(j)           = type_convert<AccDataType>(0.0f);
                welford_count_thread_bufs(i)(j) = 0;
            });
        });

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

        static_assert(num_access == sfc_cde_global.GetNumOfAccess(), "wrong!");

        // Run CShuffle + Store E + Welford threadwise
        int shuffleM_index = __builtin_amdgcn_readfirstlane(0);
        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to read from LDS
            block_sync_lds();

            // each thread shuffle data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(
                c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                c_thread_buf,
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                c_shuffle_block_buf);

            // make sure it's safe to write to LDS
            block_sync_lds();

            // Read LDS / Vmem + CDE elementwise operation
            cde_shuffle_block_copy_lds_and_global.RunRead(c_ds_desc_refs, c_ds_buf_refs);

            // Store to Vmem, but keep data in Vgpr for Welford
            cde_shuffle_block_copy_lds_and_global.RunWriteAndStoreVgpr(
                tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                tie(e_grid_buf),
                tie(post_shuffle_thread_desc_m_n),
                tie(e_thread_buf));

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

            // Threadwise welford
            auto& threadwise_welford = threadwise_welfords(shuffleM_index);
            auto& mean_thread_buf    = mean_thread_bufs(shuffleM_index);
            auto& var_thread_buf     = var_thread_bufs(shuffleM_index);

            threadwise_welford.Run(e_thread_buf, mean_thread_buf, var_thread_buf);

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto de_global_step = sfc_cde_global.GetForwardStep(access_id);
                constexpr int shuffleMInc =
                    de_global_step[I1] /
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetLength(
                        I1);
                shuffleM_index = __builtin_amdgcn_readfirstlane(shuffleM_index + shuffleMInc);
            }
        });

        // Blockwise welford and write out
        static_for<0, num_shuffleM, 1>{}([&](auto i) {
            auto& mean_thread_buf  = mean_thread_bufs(i);
            auto& var_thread_buf   = var_thread_bufs(i);
            auto& count_thread_buf = welford_count_thread_bufs(i);

            static_for<0, PostShuffleThreadSliceSize_M, 1>{}([&](auto j) {
                block_sync_lds();
                count_thread_buf(j) = threadwise_welfords(i).cur_count_;
                BlockwiseWelford::Run(mean_thread_buf(j), var_thread_buf(j), count_thread_buf(j));
            });

            if(post_shuffle_thread_cluster_idx[I1] == 0)
            {
                constexpr auto thread_welford_desc_I_m_I = make_naive_tensor_descriptor_packed(
                    make_tuple(I1, Number<PostShuffleThreadSliceSize_M>{}, I1));

                constexpr int shuffleMPerBlock =
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetLength(
                        I1);

                auto mean_var_count_thread_copy_index = make_multi_index(
                    block_m_id,                                                    // mblock
                    shuffleMPerBlock * i + post_shuffle_thread_data_idx_begin[I0], // mperblock
                    block_n_id);                                                   // nblock

                auto mean_var_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
                    AccDataType,
                    EDataType,
                    decltype(thread_welford_desc_I_m_I),
                    decltype(mean_var_grid_desc_mblock_mperblock_nblock),
                    tensor_operation::element_wise::PassThrough,
                    Sequence<1, PostShuffleThreadSliceSize_M, 1>,
                    Sequence<0, 1, 2>,
                    1,
                    1,
                    InMemoryDataOperationEnum::Set,
                    1,
                    true>{mean_var_grid_desc_mblock_mperblock_nblock,
                          mean_var_count_thread_copy_index,
                          tensor_operation::element_wise::PassThrough{}};

                mean_var_thread_copy_vgpr_to_global.Run(thread_welford_desc_I_m_I,
                                                        make_tuple(I0, I0, I0),
                                                        mean_thread_buf,
                                                        mean_var_grid_desc_mblock_mperblock_nblock,
                                                        mean_grid_buf); // write mean

                mean_var_thread_copy_vgpr_to_global.Run(thread_welford_desc_I_m_I,
                                                        make_tuple(I0, I0, I0),
                                                        var_thread_buf,
                                                        mean_var_grid_desc_mblock_mperblock_nblock,
                                                        var_grid_buf); // write variance

                // Stride of count is [0, 1]. Only the first row in count[0, 0:nblock] need
                // to be written.
                if(i == 0 && block_m_id == 0 && post_shuffle_thread_cluster_idx[I0] == 0)
                {
                    auto count_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
                        int32_t,
                        int32_t,
                        decltype(thread_welford_desc_I_m_I),
                        decltype(count_grid_desc_mblock_mperblock_nblock),
                        tensor_operation::element_wise::PassThrough,
                        Sequence<1, PostShuffleThreadSliceSize_M, 1>,
                        Sequence<0, 1, 2>,
                        1,
                        1,
                        InMemoryDataOperationEnum::Set,
                        1,
                        false>{count_grid_desc_mblock_mperblock_nblock,
                               mean_var_count_thread_copy_index,
                               tensor_operation::element_wise::PassThrough{}};

                    count_thread_copy_vgpr_to_global.Run(thread_welford_desc_I_m_I,
                                                         make_tuple(I0, I0, I0),
                                                         count_thread_buf,
                                                         count_grid_desc_mblock_mperblock_nblock,
                                                         welford_count_grid_buf); // write count
                }
            }
        });
    }

    EDataType* p_welford_mean_grid;
    EDataType* p_welford_var_grid;
    int32_t* p_welford_count_grid;
    index_t NRaw;
    GemmMeanVarGridDesc_M_N gemm_mean_var_grid_desc_m_nblock;
    GemmCountGridDesc_M_N gemm_count_grid_desc_m_nblock;
};

} // namespace ck
