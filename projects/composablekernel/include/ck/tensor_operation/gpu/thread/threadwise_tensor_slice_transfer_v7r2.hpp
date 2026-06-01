// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/utility/is_detected.hpp"
#include "ck/tensor/static_tensor.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

namespace ck {
// Thread-level multi-source, multi-destination tensor slice data movement
// Assume:
//   1. All sources and destinations are DynamicBuffer
//   2. Same VectorDim and ScalerPerVector for all sources and destinations
//   3. DstInMemOps are per destination tensor
//   4. ThreadTransferSrcResetCoordinateAfterRunFlags are per source tensor
//   5. ThreadTransferDstResetCoordinateAfterRunFlags are per destination tensor
//   6. Does not need to know src_descs and dst_descs at compile-time
//   7. Does not need to know src_slice_origins and dst_slice_origins at compile-time,
//
// Does following things to avoid scratch memory issue
//   1. Use StaticallyIndexedArray or vector_type instead of C array for thread buffer
//   2. Pass tensor descritpors by reference (or tuple of references)
//   3. Does not keep reference to tensor descriptor
//   4. Does not construct new tensor coordinate when call Run()
template <typename SrcDatas,
          typename DstDatas,
          typename SrcDescs,
          typename DstDescs,
          typename ElementwiseOperation,
          typename DstInMemOps, // Sequence<InMemoryDataOperationEnum ...>
          typename SliceLengths,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          typename SrcResetCoordinateAfterRunFlags, // Sequence<bool ...>
          typename DstResetCoordinateAfterRunFlags, // Sequence<bool ...>
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v7r2
{
    static constexpr auto I0 = Number<0>{};

    static constexpr index_t nDim = SliceLengths::Size();

    static constexpr index_t nSrc = SrcDescs::Size();
    static constexpr index_t nDst = DstDescs::Size();

    using Index = MultiIndex<nDim>;

    // return a tuple of coordiantes for a tuple of tensor
    template <typename Descs,
              typename Indices,
              enable_if_t<Descs::Size() == Indices::Size(), bool> = false>
    static constexpr auto MakeCoordinates(const Descs& descs, const Indices& indices)
    {
        return generate_tuple([&](auto i) { return make_tensor_coordinate(descs[i], indices[i]); },
                              Number<Descs::Size()>{});
    }

    using SrcCoords = decltype(MakeCoordinates(SrcDescs{}, StaticallyIndexedArray<Index, nSrc>{}));
    using DstCoords = decltype(MakeCoordinates(DstDescs{}, StaticallyIndexedArray<Index, nDst>{}));

    // scalar per access on each dim
    // FIXME: don't use lambda_scalar_per_access
    static constexpr auto src_scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

    static constexpr auto dst_scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

    using SrcSpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                   SrcDimAccessOrder,
                                                   remove_cv_t<decltype(src_scalar_per_access)>,
                                                   false>;

    using DstSpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                   DstDimAccessOrder,
                                                   remove_cv_t<decltype(dst_scalar_per_access)>,
                                                   false>;

    __device__ constexpr ThreadwiseTensorSliceTransfer_v7r2(
        const SrcDescs& src_descs,
        const StaticallyIndexedArray<Index, nSrc>& src_slice_origins,
        const DstDescs& dst_descs,
        const StaticallyIndexedArray<Index, nDst>& dst_slice_origins,
        const ElementwiseOperation& element_op)
        : src_coords_(MakeCoordinates(src_descs, src_slice_origins)),
          dst_coords_(MakeCoordinates(dst_descs, dst_slice_origins)),
          element_op_(element_op)
    {
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! cannot evenly divide");

        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! cannot evenly divide");
    }

    template <typename Indices, enable_if_t<SrcDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetSrcSliceOrigins(const SrcDescs& src_descs,
                                       const Indices& src_slice_origin_idxs)
    {
        static_for<0, nSrc, 1>{}([&](auto i) {
            src_coords_(i) = make_tensor_coordinate(src_descs[i], src_slice_origin_idxs[i]);
        });
    }

    template <typename Indices, enable_if_t<DstDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetDstSliceOrigins(const DstDescs& dst_descs,
                                       const Indices& dst_slice_origin_idxs)
    {
        static_for<0, nDst, 1>{}([&](auto i) {
            dst_coords_(i) = make_tensor_coordinate(dst_descs[i], dst_slice_origin_idxs[i]);
        });
    }

    template <typename DataTypes, index_t ScalarPerVector>
    __device__ static auto generate_vectors()
    {
        auto data_types = DataTypes{};

        constexpr index_t num = data_types.Size();

        return generate_tuple(
            [&](auto i) {
                using DataType = remove_cvref_t<decltype(data_types[i])>;

                return vector_type_maker_t<DataType, ScalarPerVector>{};
            },
            Number<num>{});
    }

    // SrcDescs: Tuple<const SrcDesc0&, const SrcDesc1&, ...>
    // SrcBuffers: Tuple<const SrcBuffer0&, const SrcBuffer1&, ...>
    template <typename SrcBuffers,
              index_t ThreadScratchId                                   = 0,
              enable_if_t<SrcDescs::Size() == SrcBuffers::Size(), bool> = false>
    __device__ void RunRead(const SrcDescs& src_descs,
                            const SrcBuffers& src_bufs,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // loop over space-filling curve
        static_for<0, src_num_access, 1>{}([&](auto iAccess) {
            auto src_vectors = generate_vectors<SrcDatas, SrcScalarPerVector>();
            auto elm_vectors = generate_vectors<DstDatas, SrcScalarPerVector>();

            bool oob_val = true;

            // copy data from src_bufs into src_vectors
            static_for<0, nSrc, 1>{}([&](auto i) {
                using src_vector_t = typename remove_cvref_t<decltype(src_vectors[i])>::type;

                const bool is_src_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(src_descs[i],
                                                                                src_coords_[i]);

                oob_val = oob_val & is_src_valid;

                src_vectors(i).template AsType<src_vector_t>()(I0) =
                    src_bufs[i].template Get<src_vector_t>(src_coords_[i].GetOffset(), true);
            });

            constexpr auto get_elem_op_vec_len = []() {
                if constexpr(is_detected<is_pack8_invocable_t, decltype(element_op_)>::value)
                {
                    if constexpr(decltype(element_op_)::is_pack8_invocable)
                        return math::min(8, SrcScalarPerVector);
                }
                if constexpr(is_detected<is_pack4_invocable_t, decltype(element_op_)>::value)
                {
                    if constexpr(decltype(element_op_)::is_pack4_invocable)
                        return math::min(4, SrcScalarPerVector);
                }
                if constexpr(is_detected<is_pack2_invocable_t, decltype(element_op_)>::value)
                {
                    if constexpr(decltype(element_op_)::is_pack2_invocable)
                        return math::min(2, SrcScalarPerVector);
                }
                return 1;
            };

            constexpr index_t elem_op_vec_len = get_elem_op_vec_len();

            // apply pointwise function
            static_for<0, SrcScalarPerVector / elem_op_vec_len, 1>{}([&](auto i) {
                // get reference to src data
                const auto src_data_refs = generate_tie(
                    // return type should be lvalue
                    [&](auto iSrc) -> const auto& {
                        using SrcData = remove_cvref_t<tuple_element_t<iSrc.value, SrcDatas>>;

                        using elem_op_vec_t = typename vector_type<SrcData, elem_op_vec_len>::type;

                        return src_vectors[iSrc].template AsType<elem_op_vec_t>()[i];
                    },
                    Number<nSrc>{});

                // get reference to dst data
                auto dst_data_refs = generate_tie(
                    // return type should be lvalue
                    [&](auto iDst) -> auto& {
                        using DstData = remove_cvref_t<tuple_element_t<iDst.value, DstDatas>>;

                        using elem_op_vec_t = typename vector_type<DstData, elem_op_vec_len>::type;

                        return elm_vectors(iDst).template AsType<elem_op_vec_t>()(i);
                    },
                    Number<nDst>{});

                // apply pointwise function
                // pointwise function signature:
                // element_op_(dst_data_refs[I0],
                //             dst_data_refs[I1],
                //             ...,
                //             src_data_refs[I0],
                //             src_data_refs[I1],
                //             ...)
                unpack2(element_op_, dst_data_refs, src_data_refs);
            });

            elm_vectors_tuple_(thread_scratch_id)(iAccess) = elm_vectors;
            oob_vectors_tuple_(thread_scratch_id)(iAccess) = oob_val;

            // move coordinate
            if constexpr(iAccess.value != src_num_access - 1)
            {
                constexpr auto forward_step = SrcSpaceFillingCurve::GetForwardStep(iAccess);

                static_for<0, nSrc, 1>{}([&](auto i) {
                    move_tensor_coordinate(src_descs[i],
                                           src_coords_(i),
                                           make_tensor_coordinate_step(src_descs[i], forward_step));
                });
            }
        });

        // move coordinate back to slice origin (or not)
        static_for<0, nSrc, 1>{}([&](auto i) {
            if constexpr(SrcResetCoordinateAfterRunFlags::At(i))
            {
                const auto src_reset_step =
                    make_tensor_coordinate_step(src_descs[i], GetSrcCoordinateResetStep());

                move_tensor_coordinate(src_descs[i], src_coords_(i), src_reset_step);
            }
        });
    }

#if 1
    template <index_t ThreadScratchId = 0>
    __device__ void OOBCheck(Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // loop over space-filling curve
        static_for<0, src_num_access, 1>{}([&](auto iAccess) {
            auto elm_vectors = elm_vectors_tuple_[thread_scratch_id][iAccess];
            auto oob_val     = oob_vectors_tuple_[thread_scratch_id][iAccess];

            static_for<0, nDst, 1>{}([&](auto i) {
                using elm_vector_t = typename remove_cvref_t<decltype(elm_vectors[i])>::type;
                elm_vectors(i).template AsType<elm_vector_t>()(I0) =
                    oob_val ? elm_vectors(i).template AsType<elm_vector_t>()[I0] : elm_vector_t{0};
            });

            elm_vectors_tuple_(thread_scratch_id)(iAccess) = elm_vectors;
        });
    }
#endif

    template <index_t ThreadScratchId = 0>
    __device__ void
    TransposeFromElmToDst(Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        using DstData = remove_cvref_t<decltype(DstDatas{}[I0])>;

        using ElmThreadScratch =
            StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                            DstData,
                                            SrcScalarPerVector,
                                            decltype(GetSrcThreadScratchDescriptor()),
                                            true>;
        using DstThreadScratch =
            StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                            DstData,
                                            DstScalarPerVector,
                                            decltype(GetDstThreadScratchDescriptor()),
                                            true>;

        ElmThreadScratch elm_thread_scratch_;
        DstThreadScratch dst_thread_scratch_;

        elm_thread_scratch_.data_ =
            bit_cast<decltype(elm_thread_scratch_.data_)>(elm_vectors_tuple_[thread_scratch_id]);

        if constexpr(SrcVectorDim != DstVectorDim &&
                     ((is_same<half_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 2 == 0 && DstScalarPerVector % 2 == 0) ||
                      (is_same<f8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0) ||
                      (is_same<int8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0)))
        {
            // each transpose does
            // DstScalarPerVector # of src vectors in src_thread_scratch_
            // SrcScalarPerVector # of dst vectors in dst_thread_scratch_
            constexpr index_t num_src_vector = Number<DstScalarPerVector>{};
            constexpr index_t num_dst_vector = Number<SrcScalarPerVector>{};

            // Assume SrcVectorDim is not the same as DstVectorDim, so we do transpose
            // TODO: make this logic generic for all scenario

            constexpr auto src_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

            constexpr auto dst_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

            constexpr auto scalar_per_access = generate_sequence(
                detail::lambda_scalar_per_access_for_src_and_dst<SrcVectorDim,
                                                                 SrcScalarPerVector,
                                                                 DstVectorDim,
                                                                 DstScalarPerVector>{},
                Number<nDim>{});

            constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

            static_ford<decltype(access_lengths)>{}([&](auto access_idx) {
                constexpr auto data_idx = access_idx * scalar_per_access;

                constexpr auto data_idx_seq = generate_sequence_v2(
                    [&](auto i) { return Number<data_idx[i]>{}; }, Number<nDim>{});

                using src_vector_t = vector_type_maker_t<DstData, SrcScalarPerVector>;
                using dst_vector_t = vector_type_maker_t<DstData, DstScalarPerVector>;

                // get DstScalarPerVector # of read-only references to src vectors from
                // src_thread_scratch_
                const auto src_vector_refs = generate_tie(
                    [&](auto i) -> const src_vector_t& {
                        // i increment corresponds to movement in DstVectorDim
                        return elm_thread_scratch_.GetVectorTypeReference(
                            data_idx_seq + i * dst_scalar_step_in_vector);
                    },
                    Number<num_src_vector>{});

                // get SrcScalarPerVector # of references to dst vectors from dst_thread_scratch_
                auto dst_vector_refs = generate_tie(
                    [&](auto i) -> dst_vector_t& {
                        // i increment corresponds to movement in SrcVectorDim
                        return dst_thread_scratch_.GetVectorTypeReference(
                            data_idx_seq + i * src_scalar_step_in_vector);
                    },
                    Number<num_dst_vector>{});

                // do data transpose
                transpose_vectors<DstData, DstScalarPerVector, SrcScalarPerVector>{}(
                    src_vector_refs, dst_vector_refs);
            });
        }
        else
        {
            static_ford<SliceLengths>{}(
                [&](auto idx) { dst_thread_scratch_(idx) = elm_thread_scratch_[idx]; });
        }

        dst_vectors_tuple_(thread_scratch_id) = bit_cast<DstVectorTuple>(dst_thread_scratch_.data_);
    }

    // DstDescs: Tuple<const DstDesc0&, const DstDesc1&, ...>
    // DstBuffers: Tuple<const DstBuffer0&, const DstBuffer1&, ...>
    template <typename DstBuffers,
              index_t ThreadScratchId                                             = 0,
              enable_if_t<DstDescs::Size() == 1 && DstBuffers::Size() == 1, bool> = false>
    __device__ void RunWrite(const DstDescs& dst_descs,
                             DstBuffers dst_bufs,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        OOBCheck(thread_scratch_id);
        TransposeFromElmToDst(thread_scratch_id);

        // loop over space-filling curve
        static_for<0, dst_num_access, 1>{}([&](auto iAccess) {
            auto dst_vectors = dst_vectors_tuple_[thread_scratch_id][iAccess];

            // copy data from buf_vectors into dst_bufs
            static_for<0, nDst, 1>{}([&](auto i) {
                using dst_vector_t = typename remove_cvref_t<decltype(dst_vectors[i])>::type;

                const bool is_dst_valid =
                    coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_descs[i],
                                                                                dst_coords_[i]);

                constexpr InMemoryDataOperationEnum DstInMemOp =
                    static_cast<InMemoryDataOperationEnum>(DstInMemOps::At(i.value));

                dst_bufs(i).template Update<DstInMemOp, dst_vector_t>(
                    dst_coords_[i].GetOffset(),
                    is_dst_valid,
                    dst_vectors[i].template AsType<dst_vector_t>()[I0]);
            });

            // move coordinate
            if constexpr(iAccess.value != dst_num_access - 1)
            {
                constexpr auto forward_step = DstSpaceFillingCurve::GetForwardStep(iAccess);

                static_for<0, nDst, 1>{}([&](auto i) {
                    move_tensor_coordinate(dst_descs[i],
                                           dst_coords_(i),
                                           make_tensor_coordinate_step(dst_descs[i], forward_step));
                });
            }
        });

        static_for<0, nDst, 1>{}([&](auto i) {
            if constexpr(DstResetCoordinateAfterRunFlags::At(i))
            {
                const auto dst_reset_step =
                    make_tensor_coordinate_step(dst_descs[i], GetDstCoordinateResetStep());

                move_tensor_coordinate(dst_descs[i], dst_coords_(i), dst_reset_step);
            }
        });
    }

    // SrcDescs: Tuple<const SrcDesc0&, const SrcDesc1&, ...>
    // SrcBuffers: Tuple<const SrcBuffer0&, const SrcBuffer1&, ...>
    // DstDescs: Tuple<const DstDesc0&, const DstDesc1&, ...>
    // DstBuffers: Tuple<const DstBuffer0&, const DstBuffer1&, ...>
    template <typename SrcBuffers,
              typename DstBuffers,
              enable_if_t<SrcDescs::Size() == SrcBuffers::Size() &&
                              DstDescs::Size() == DstBuffers::Size(),
                          bool> = false>
    __device__ void Run(const SrcDescs& src_descs,
                        const SrcBuffers& src_bufs,
                        const DstDescs& dst_descs,
                        DstBuffers dst_bufs)
    {
        RunRead(src_descs, src_bufs);
        RunWrite(dst_descs, dst_bufs);
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        if constexpr(src_num_access == 0)
        {
            return typename SrcSpaceFillingCurve::Index{};
        }
        else
        {
            return SrcSpaceFillingCurve::GetStepBetween(Number<src_num_access - 1>{}, Number<0>{});
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        if constexpr(dst_num_access == 0)
        {
            return typename DstSpaceFillingCurve::Index{};
        }
        else
        {
            return DstSpaceFillingCurve::GetStepBetween(Number<dst_num_access - 1>{}, Number<0>{});
        }
    }

    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        // constexpr auto src_scalar_per_access = generate_sequence(
        // detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(src_access_lengths), Number<SrcScalarPerVector>{});

        // 1st stage of transforms
        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(src_access_lengths_and_vector_length);

        // 2nd stage of transforms
        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(src_access_lengths_and_vector_length[i],
                                   src_access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(src_access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        // 1st stage of transforms
        // constexpr auto dst_scalar_per_access = generate_sequence(
        // detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(dst_access_lengths), Number<DstScalarPerVector>{});

        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(dst_access_lengths_and_vector_length);

        // 2nd stage of transforms
        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(dst_access_lengths_and_vector_length[i],
                                   dst_access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(dst_access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <index_t ISrc>
    __device__ void MoveSrcSliceWindow(const SrcDescs& src_descs,
                                       Number<ISrc> iSrc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRunFlags::At(iSrc)
                ? src_slice_origin_step_idx
                : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_descs[iSrc], adjusted_step_idx);

        move_tensor_coordinate(src_descs[iSrc], src_coords_(iSrc), adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <index_t IDst>
    __device__ void MoveDstSliceWindow(const DstDescs& dst_descs,
                                       Number<IDst> iDst,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRunFlags::At(iDst)
                ? dst_slice_origin_step_idx
                : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_descs[iDst], adjusted_step_idx);

        move_tensor_coordinate(dst_descs[iDst], dst_coords_(iDst), adjusted_step);
    }

    private:
    using SrcVectorsType = decltype(generate_vectors<SrcDatas, SrcScalarPerVector>());
    using ElmVectorsType = decltype(generate_vectors<DstDatas, SrcScalarPerVector>());
    using DstVectorsType = decltype(generate_vectors<DstDatas, DstScalarPerVector>());

    static constexpr auto src_num_access = SrcSpaceFillingCurve::GetNumOfAccess();
    static constexpr auto dst_num_access = DstSpaceFillingCurve::GetNumOfAccess();

    using ElmVectorTuple = StaticallyIndexedArray<ElmVectorsType, src_num_access>;
    using DstVectorTuple = StaticallyIndexedArray<DstVectorsType, dst_num_access>;

    StaticallyIndexedArray<ElmVectorTuple, NumThreadScratch> elm_vectors_tuple_;
    StaticallyIndexedArray<DstVectorTuple, NumThreadScratch> dst_vectors_tuple_;

    using OOBVectorTuple = StaticallyIndexedArray<bool, src_num_access>;
    StaticallyIndexedArray<OOBVectorTuple, NumThreadScratch> oob_vectors_tuple_;

    SrcCoords src_coords_;
    DstCoords dst_coords_;
    const ElementwiseOperation element_op_;
};

} // namespace ck
