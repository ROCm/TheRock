// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"

namespace ck {

template <typename ReduceAccDataType,
          typename ReducePtrsGlobal,
          typename D0ElementwiseOperation,
          typename ReduceOperations,
          typename ReduceInElementwiseOperations,
          typename ReduceAccElementwiseOperations,
          typename ReduceGlobalMemoryDataOperation,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>
struct ReduceTrait_
{
    using ReduceAccDataType_               = ReduceAccDataType;
    using ReducePtrsGlobal_                = ReducePtrsGlobal;
    using D0ElementwiseOperation_          = D0ElementwiseOperation;
    using ReduceOperations_                = ReduceOperations;
    using ReduceInElementwiseOperations_   = ReduceInElementwiseOperations;
    using ReduceAccElementwiseOperations_  = ReduceAccElementwiseOperations;
    using ReduceGlobalMemoryDataOperation_ = ReduceGlobalMemoryDataOperation;
    using CReduceThreadClusterLengths_MPerBlock_NPerBlock_ =
        CReduceThreadClusterLengths_MPerBlock_NPerBlock;
    static constexpr index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock_ =
        CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock;
    static constexpr index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock_ =
        CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock;
};

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
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
          typename ReduceTrait>
struct EpilogueReduceCShuffle
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
    using Base::I3;
    using Base::NumDTensor;

    // assume Reduce is packed tensor
    __device__ static auto MakeReduceGridDescriptor_M(index_t MRaw)
    {
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        const auto d_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(d_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return d_grid_desc_mraw;
        }
    }

    using ReduceGridDesc_M = decltype(MakeReduceGridDescriptor_M(1));

    __device__ static constexpr auto
    MakeReduceGridDescriptor_MBlock_MPerBlock(const ReduceGridDesc_M& d_grid_desc_m)
    {
        const auto M      = d_grid_desc_m.GetLength(I0);
        const auto MBlock = M / MPerBlock;

        const auto reduce_grid_desc_mblock_mperblock = transform_tensor_descriptor(
            d_grid_desc_m,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1>{}));

        return reduce_grid_desc_mblock_mperblock;
    }

    __device__ EpilogueReduceCShuffle(
        typename ReduceTrait::ReducePtrsGlobal_ p_reduces_grid_,
        const typename ReduceTrait::ReduceInElementwiseOperations_ reduce_in_element_ops_,
        const typename ReduceTrait::ReduceAccElementwiseOperations_ reduce_out_element_ops_,
        const index_t MRaw_,
        const typename ReduceTrait::D0ElementwiseOperation_ d0_element_op_)
        : p_reduces_grid(p_reduces_grid_),
          reduce_in_element_ops(reduce_in_element_ops_),
          reduce_out_element_ops(reduce_out_element_ops_),
          MRaw(MRaw_),
          d0_element_op{d0_element_op_},
          reduce_grid_desc_m{MakeReduceGridDescriptor_M(MRaw)}
    {
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
        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        auto reduce_grid_desc_mblock_mperblock =
            MakeReduceGridDescriptor_MBlock_MPerBlock(reduce_grid_desc_m);

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

        // LDS c_reduce_block_desc_mperblock_nperblock
        constexpr auto c_reduce_block_desc_mperblock_nperblock = transform_tensor_descriptor(
            c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
            make_tuple(
                make_freeze_transform(I0),
                make_pass_through_transform(
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetLength(
                        I1)),
                make_freeze_transform(I0),
                make_pass_through_transform(
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetLength(
                        I3))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<>{}, Sequence<1>{}));

        static_assert(
            ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_::At(I0) *
                    ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_::At(I1) ==
                BlockSize,
            "wrong!");

        static_assert(
            (CShuffleMRepeatPerShuffle * BlockwiseGemmPipe::MWaves * MPerWmma) %
                        ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_::At(I0) ==
                    0 &&
                (CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves * NPerWmma) %
                        ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_::At(I1) ==
                    0,
            "wrong!");

        constexpr index_t mreduce_per_thread =
            (CShuffleMRepeatPerShuffle * BlockwiseGemmPipe::MWaves * MPerWmma) /
            ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_::At(I0);

        constexpr index_t nreduce_per_thread =
            (CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves * NPerWmma) /
            ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_::At(I1);

        static constexpr index_t NumReduce = ReduceTrait::ReducePtrsGlobal_::Size();

        constexpr auto c_reduce_thread_lengths_mperblock_nperblock =
            Sequence<mreduce_per_thread, nreduce_per_thread>{};

        // VGPR c_reduce_thread_desc_mperblock_nperblock
        constexpr auto c_reduce_thread_desc_mperblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(Number<mreduce_per_thread>{}, Number<nreduce_per_thread>{}));

        // VGPR reduce_thread_desc_mperblock
        constexpr auto reduce_thread_desc_mperblock =
            make_naive_tensor_descriptor_packed(make_tuple(Number<mreduce_per_thread>{}));

        // VGPR reduce_thread_desc_mblock_mperblock
        constexpr auto reduce_thread_desc_mblock_mperblock =
            make_naive_tensor_descriptor_packed(make_tuple(I1, Number<mreduce_per_thread>{}));

        auto c_reduce_thread_buf =
            make_static_buffer<AddressSpaceEnum::Vgpr, typename ReduceTrait::ReduceAccDataType_>(
                c_reduce_thread_desc_mperblock_nperblock.GetElementSpaceSize());

        // reduce: threadwise copy from LDS to VGPR
        constexpr auto c_reduce_thread_cluster_desc = make_cluster_descriptor(
            typename ReduceTrait::CReduceThreadClusterLengths_MPerBlock_NPerBlock_{},
            Sequence<1, 0>{});

        const auto c_reduce_thread_cluster_idx = c_reduce_thread_cluster_desc.CalculateBottomIndex(
            make_multi_index(get_thread_local_1d_id()));

        const auto c_reduce_thread_data_idx_begin =
            c_reduce_thread_cluster_idx * c_reduce_thread_lengths_mperblock_nperblock;

        auto c_reduce_thread_copy_lds_to_vgpr = ThreadwiseTensorSliceTransfer_v2<
            CShuffleDataType,
            typename ReduceTrait::ReduceAccDataType_,
            decltype(c_reduce_block_desc_mperblock_nperblock),
            decltype(c_reduce_thread_desc_mperblock_nperblock),
            decltype(c_reduce_thread_lengths_mperblock_nperblock),
            Sequence<0, 1>,
            1,
            ReduceTrait::CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock_,
            1,
            true>{c_reduce_block_desc_mperblock_nperblock, c_reduce_thread_data_idx_begin};

        auto reduce_tuple_thread_copy_vgpr_to_global = generate_tuple(
            [&](auto I) {
                auto p_reduce_grid         = p_reduces_grid[I];
                auto reduce_acc_element_op = reduce_out_element_ops[I];

                return ThreadwiseTensorSliceTransfer_v1r3<
                    typename ReduceTrait::ReduceAccDataType_,
                    remove_pointer_t<decltype(p_reduce_grid)>,
                    decltype(reduce_thread_desc_mblock_mperblock),
                    decltype(reduce_grid_desc_mblock_mperblock),
                    decltype(reduce_acc_element_op),
                    Sequence<1, mreduce_per_thread>,
                    Sequence<0, 1>,
                    1,
                    ReduceTrait::CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock_,
                    ReduceTrait::ReduceGlobalMemoryDataOperation_::At(I),
                    1,
                    false>{reduce_grid_desc_mblock_mperblock,
                           make_multi_index(block_m_id,                          // mblock
                                            c_reduce_thread_data_idx_begin[I0]), // mperblock
                           reduce_acc_element_op};
            },
            Number<NumReduce>{});

        // multiple Ds
        constexpr auto d_reduce_thread_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1, Number<mreduce_per_thread>{}, I1, Number<nreduce_per_thread>{}));

        constexpr auto ds_reduce_thread_desc_mblock_mperblock_nblock_nperblock = generate_tuple(
            [&](auto) { return d_reduce_thread_desc_mblock_mperblock_nblock_nperblock; },
            Number<NumDTensor>{});

        constexpr auto ds_thread_buf_size =
            d_reduce_thread_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        auto c01_thread_buf =
            make_static_buffer<AddressSpaceEnum::Vgpr, typename ReduceTrait::ReduceAccDataType_>(
                Number<ds_thread_buf_size>{});

        auto ds_thread_copy_global_to_vgpr = generate_tuple(
            [&](auto I) {
                return ThreadwiseTensorSliceTransfer_v2<
                    remove_cvref_t<tuple_element_t<I.value, DsDataType>>,
                    typename ReduceTrait::ReduceAccDataType_,
                    decltype(ds_grid_desc_mblock_mperblock_nblock_nperblock[I]),
                    remove_cvref_t<
                        decltype(ds_reduce_thread_desc_mblock_mperblock_nblock_nperblock[I])>,
                    Sequence<I1, mreduce_per_thread, I1, nreduce_per_thread>,
                    Sequence<0, 1, 2, 3>,
                    3,
                    ReduceTrait::CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock_,
                    1,
                    true>(ds_grid_desc_mblock_mperblock_nblock_nperblock[I],
                          make_multi_index(
                              I0,
                              m_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I0],
                              I0,
                              n_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I1]));
            },
            Number<NumDTensor>{});

        constexpr auto c_reduce_thread_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1, Number<mreduce_per_thread>{}, I1, Number<nreduce_per_thread>{}));

        // Write E from Vgpr to Vmem
        auto c_reduce_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            typename ReduceTrait::ReduceAccDataType_,
            EDataType,
            decltype(c_reduce_thread_desc_mblock_mperblock_nblock_nperblock),
            decltype(e_grid_desc_mblock_mperblock_nblock_nperblock),
            tensor_operation::element_wise::PassThrough,
            Sequence<I1, mreduce_per_thread, I1, nreduce_per_thread>, // SliceLengths
            Sequence<0, 1, 2, 3>,                                     // DimAccessOrder
            3,                                                        // DstVectorDim
            ReduceTrait::CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock_,
            EGlobalMemoryDataOperation,
            1,
            true>{e_grid_desc_mblock_mperblock_nblock_nperblock,
                  make_multi_index(I0,
                                   m_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I0],
                                   I0,
                                   n_block_data_idx_on_grid + c_reduce_thread_data_idx_begin[I1]),
                  NumDTensor > 0 ? tensor_operation::element_wise::PassThrough{} : cde_element_op};

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
            {
                c_reduce_thread_copy_lds_to_vgpr.Run(c_reduce_block_desc_mperblock_nperblock,
                                                     c_shuffle_block_buf,
                                                     c_reduce_thread_desc_mperblock_nperblock,
                                                     make_tuple(I0, I0),
                                                     c_reduce_thread_buf);

                // Note: currently multiple Ds supports only Bias + Add.
                // It needs to be generalized for other operations (currently not needed)
                if constexpr(NumDTensor > 0)
                {
                    auto& d0_thread_copy_global_to_vgpr = ds_thread_copy_global_to_vgpr(I0);
                    // d0 / d1 operations
                    d0_thread_copy_global_to_vgpr.Run(
                        ds_grid_desc_mblock_mperblock_nblock_nperblock[I0],
                        ds_grid_buf[I0],
                        ds_reduce_thread_desc_mblock_mperblock_nblock_nperblock[I0],
                        make_tuple(I0, I0, I0, I0),
                        c01_thread_buf);

                    // c = activation(c + bias)
                    static_for<0, c_reduce_thread_desc_mperblock_nperblock.GetElementSize(), 1>{}(
                        [&](auto i) {
                            typename ReduceTrait::ReduceAccDataType_ out;
                            cde_element_op(out, c_reduce_thread_buf(i) + c01_thread_buf(i));
                            c_reduce_thread_buf(i) = out;
                        });

                    auto& d1_thread_copy_global_to_vgpr = ds_thread_copy_global_to_vgpr(I1);

                    d1_thread_copy_global_to_vgpr.Run(
                        ds_grid_desc_mblock_mperblock_nblock_nperblock[I1],
                        ds_grid_buf[I1],
                        ds_reduce_thread_desc_mblock_mperblock_nblock_nperblock[I1],
                        make_tuple(I0, I0, I0, I0),
                        c01_thread_buf);

                    // c = c + c1_function(c1)
                    static_for<0, c_reduce_thread_desc_mperblock_nperblock.GetElementSize(), 1>{}(
                        [&](auto i) {
                            d0_element_op(c01_thread_buf(i), c01_thread_buf(i));
                            c_reduce_thread_buf(i) += c01_thread_buf(i);
                        });
                }

                // Write E
                c_reduce_thread_copy_vgpr_to_global.Run(
                    c_reduce_thread_desc_mblock_mperblock_nblock_nperblock,
                    make_tuple(I0, I0, I0, I0),
                    c_reduce_thread_buf,
                    e_grid_desc_mblock_mperblock_nblock_nperblock,
                    e_grid_buf);

                // Reduction
                static_for<0, NumReduce, 1>{}([&](auto In) {
                    auto& p_reduce_grid = p_reduces_grid[In];

                    auto reduce_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
                        p_reduce_grid, reduce_grid_desc_mblock_mperblock.GetElementSpaceSize());

                    auto reduce_thread_buf =
                        make_static_buffer<AddressSpaceEnum::Vgpr,
                                           typename ReduceTrait::ReduceAccDataType_>(
                            reduce_thread_desc_mperblock.GetElementSpaceSize());

                    auto& reduce_in_element_op = reduce_in_element_ops[In];

                    auto& reduce_thread_copy_vgpr_to_global =
                        reduce_tuple_thread_copy_vgpr_to_global(In);

                    using ReduceOperation =
                        remove_cvref_t<decltype(typename ReduceTrait::ReduceOperations_{}[In])>;
                    using ThreadwiseReduce =
                        ThreadwiseReduction<typename ReduceTrait::ReduceAccDataType_,
                                            decltype(c_reduce_thread_desc_mperblock_nperblock),
                                            decltype(reduce_thread_desc_mperblock),
                                            ReduceOperation,
                                            false>;

                    // Global write Gemm shuffle + reduction
                    const auto reduce_identityVal = ReduceOperation::template GetIdentityValue<
                        typename ReduceTrait::ReduceAccDataType_>();

                    static_for<0, mreduce_per_thread, 1>{}(
                        [&](auto I) { reduce_thread_buf(I) = reduce_identityVal; });

                    // reduce in VGPR
                    static_for<0, mreduce_per_thread, 1>{}([&](auto im) {
                        static_for<0, nreduce_per_thread, 1>{}([&](auto in) {
                            constexpr auto offset =
                                Number<c_reduce_thread_desc_mperblock_nperblock.CalculateOffset(
                                    make_tuple(im, in))>{};

                            reduce_in_element_op(c_reduce_thread_buf(offset),
                                                 c_reduce_thread_buf(offset));
                        });
                    });

                    ThreadwiseReduce::Reduce(c_reduce_thread_buf, reduce_thread_buf);

                    // copy from VGPR to Global
                    reduce_thread_copy_vgpr_to_global.Run(reduce_thread_desc_mblock_mperblock,
                                                          make_tuple(I0, I0),
                                                          reduce_thread_buf,
                                                          reduce_grid_desc_mblock_mperblock,
                                                          reduce_grid_buf);

                    if constexpr(access_id < num_access - 1)
                    {
                        constexpr auto c_global_step = sfc_cde_global.GetForwardStep(access_id);
                        reduce_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                            reduce_grid_desc_mblock_mperblock,
                            make_tuple(c_global_step[I0], c_global_step[I1]));
                    }
                });
            }

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto cde_global_step = sfc_cde_global.GetForwardStep(access_id);
                // move on Ds
                static_for<0, NumDTensor, 1>{}([&](auto I) {
                    auto& d_thread_copy_global_to_vgpr = ds_thread_copy_global_to_vgpr(I);
                    d_thread_copy_global_to_vgpr.MoveSrcSliceWindow(
                        ds_grid_desc_mblock_mperblock_nblock_nperblock[I], cde_global_step);
                });

                // move on E
                c_reduce_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                    e_grid_desc_mblock_mperblock_nblock_nperblock, cde_global_step);
            }
        });
    }

    typename ReduceTrait::ReducePtrsGlobal_ p_reduces_grid;
    typename ReduceTrait::ReduceInElementwiseOperations_ reduce_in_element_ops;
    typename ReduceTrait::ReduceAccElementwiseOperations_ reduce_out_element_ops;
    index_t MRaw;
    typename ReduceTrait::D0ElementwiseOperation_ d0_element_op;
    ReduceGridDesc_M reduce_grid_desc_m;
};

} // namespace ck
