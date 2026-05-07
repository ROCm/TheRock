// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor/static_tensor.hpp"

namespace ck {

namespace detail {
// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t DstVectorDim,
          index_t DstScalarPerVector>
struct lambda_scalar_per_access_for_src_and_dst_idle
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if(i == SrcVectorDim && i == DstVectorDim)
        {
            return math::lcm(SrcScalarPerVector, DstScalarPerVector);
        }
        else if(i == SrcVectorDim)
        {
            return SrcScalarPerVector;
        }
        else if(i == DstVectorDim)
        {
            return DstScalarPerVector;
        }
        else
        {
            return 1;
        }
    }
};

} // namespace detail

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
//   5. Dequantization happened between read and write.
template <typename SliceLengths,
          typename ScaleSliceLengths,
          typename SrcElementwiseOperation,
          typename ScaleElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcData,
          typename ScaleData,
          typename DstData,
          typename SrcDesc,
          typename ScaleDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t ScaleScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t ScaleScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun, // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v3r1_dequant
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord   = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using ScaleCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord   = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3r1_dequant(
        const SrcDesc& src_desc,
        const Index& src_slice_origin,
        const SrcElementwiseOperation& src_element_op,
        const ScaleDesc& scale_desc,
        const Index& scale_slice_origin,
        const ScaleElementwiseOperation& scale_element_op,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          scale_coord_(make_tensor_coordinate(scale_desc, scale_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          src_element_op_(src_element_op),
          scale_element_op_(scale_element_op),
          dst_element_op_(dst_element_op)
    {
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetScaleSliceOrigin(const ScaleDesc& scale_desc,
                                        const Index& scale_slice_origin_idx)
    {
        scale_coord_ = make_tensor_coordinate(scale_desc, scale_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcBuffer, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcBuffer& src_buf,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value,
            "wrong! SrcBuffer and SrcData data type are inconsistent");

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward steps
        const auto src_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? src_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(src_desc, forward_step_idx);
            },
            Number<nDim>{});

        // make backward steps
        const auto src_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -src_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(src_desc, backward_step_idx);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_src_access_lengths)>{}([&](auto ordered_src_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_src_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate src data index
            constexpr auto src_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_src_access_idx[i]
                                                      : ordered_src_access_lengths[i] - 1 -
                                                            ordered_src_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                       src_scalar_per_access;
            }();

            constexpr auto src_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<src_data_idx[i]>{}; }, Number<src_data_idx.Size()>{});

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            using src_vector_type = vector_type_maker_t<SrcData, SrcScalarPerVector>;
            using src_vector_t    = typename src_vector_type::type;

            // copy data from src_buf into src_vector_container
            auto src_vector_container = src_vector_type{
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid)};

            // copy data from src_vector_container into src_thread_scratch_
            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<src_vector_t>(
                    src_data_idx_seq, src_vector_container.template AsType<src_vector_t>()[I0]);

            constexpr auto move_on_dim = [&]() constexpr {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_src_access_idx[i] < ordered_src_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_src_access_idx[j] == ordered_src_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }();

            // move src coord
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_forward_steps[src_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_backward_steps[src_dim_access_order[i]]);
                    }
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    template <typename ScaleBuffer>
    __device__ void RunScaleRead(const ScaleDesc& scale_desc, const ScaleBuffer& scale_buf)
    {
        static_assert(ScaleBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          ScaleBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename ScaleBuffer::type>, remove_cvref_t<ScaleData>>::value,
            "wrong! ScaleBuffer and ScaleData data type are inconsistent");

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto scale_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, ScaleScalarPerVector>{}, Number<nDim>{});

        constexpr auto scale_access_lengths = SliceLengths{} / scale_scalar_per_access;

        constexpr auto scale_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_scale_access_lengths =
            container_reorder_given_new2old(scale_access_lengths, scale_dim_access_order);

        // make forward steps
        const auto scale_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? scale_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(scale_desc, forward_step_idx);
            },
            Number<nDim>{});

        // make backward steps
        const auto scale_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -scale_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(scale_desc, backward_step_idx);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_scale_access_lengths)>{}([&](auto ordered_scale_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_scale_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_scale_access_lengths[j] + ordered_scale_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate scale data index
            constexpr auto scale_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_scale_access_idx[i]
                                                      : ordered_scale_access_lengths[i] - 1 -
                                                            ordered_scale_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, scale_dim_access_order) *
                       scale_scalar_per_access;
            }();

            constexpr auto scale_data_idx_seq =
                generate_sequence_v2([&](auto i) { return Number<scale_data_idx[i]>{}; },
                                     Number<scale_data_idx.Size()>{});

            const bool is_scale_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                scale_desc, scale_coord_);

            using scale_vector_type = vector_type_maker_t<ScaleData, ScaleScalarPerVector>;
            using scale_vector_t    = typename scale_vector_type::type;

            // copy data from scale_buf into scale_vector_container
            auto scale_vector_container = scale_vector_type{
                scale_buf.template Get<scale_vector_t>(scale_coord_.GetOffset(), is_scale_valid)};

            // copy data from scale_vector_container into scale_thread_scratch_
            scale_thread_scratch_.template SetAsType<scale_vector_t>(
                scale_data_idx_seq, scale_vector_container.template AsType<scale_vector_t>()[I0]);

            constexpr auto move_on_dim = [&]() constexpr {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) =
                        ordered_scale_access_idx[i] < ordered_scale_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_scale_access_idx[j] == ordered_scale_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }();

            // move scale coord
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(scale_desc,
                                               scale_coord_,
                                               scale_forward_steps[scale_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(scale_desc,
                                               scale_coord_,
                                               scale_backward_steps[scale_dim_access_order[i]]);
                    }
                }
            });
        });

        // don't need to move scale coordinate back to slice origin
        /*
            if constexpr(SrcResetCoordinateAfterRun)
            {
                const auto scale_reset_step =
                    make_tensor_coordinate_step(scale_desc, GetScaleCoordinateResetStep());

                move_tensor_coordinate(scale_desc, scale_coord_, scale_reset_step);
            }
        */
    }

    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
#if !CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
        static_ford<SliceLengths>{}([&](auto idx) {
            // convert from SrcData to DstData here
            dst_thread_scratch_(idx) =
                type_convert<DstData>(src_thread_scratch_tuple_[thread_scratch_id][idx]);
        });
#else
        // sub-dword transpose between src_thread_scratch_ and dst_thread_scratch_
        // TODO make this logic more generic for more sub-dword datatype
        if constexpr(SrcVectorDim != DstVectorDim &&
                     ((is_same<half_t, remove_cvref_t<SrcData>>::value &&
                       is_same<half_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 2 == 0 && DstScalarPerVector % 2 == 0) ||
                      (is_same<int8_t, remove_cvref_t<SrcData>>::value &&
                       is_same<int8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0)))
        {
            // each transpose does
            // DstScalarPerVector # of src vectors in src_thread_scratch_
            // SrcScalarPerVector # of dst vectors in dst_thread_scratch_
            constexpr index_t num_src_vector = Number<DstScalarPerVector>{};
            constexpr index_t num_dst_vector = Number<SrcScalarPerVector>{};

            // Assume SrcVectorDim is not the same as DstVectorDim, so we do transpose
            // TODO: make this logic generic for all scenario
            static_assert(SrcVectorDim != DstVectorDim, "wrong");

            constexpr auto src_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

            constexpr auto dst_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

            constexpr auto scalar_per_access = generate_sequence(
                detail::lambda_scalar_per_access_for_src_and_dst_idle<SrcVectorDim,
                                                                      SrcScalarPerVector,
                                                                      DstVectorDim,
                                                                      DstScalarPerVector>{},
                Number<nDim>{});

            constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

            static_ford<decltype(access_lengths)>{}([&](auto access_idx) {
                constexpr auto data_idx = access_idx * scalar_per_access;

                constexpr auto data_idx_seq = generate_sequence_v2(
                    [&](auto i) { return Number<data_idx[i]>{}; }, Number<nDim>{});

                using src_vector_t = vector_type_maker_t<SrcData, SrcScalarPerVector>;
                using dst_vector_t = vector_type_maker_t<DstData, DstScalarPerVector>;

                // get DstScalarPerVector # of read-only references to src vectors from
                // src_thread_scratch_
                const auto src_vector_refs = generate_tie(
                    [&](auto i) -> const src_vector_t& {
                        // i increment corresponds to movement in DstVectorDim
                        return src_thread_scratch_tuple_[thread_scratch_id].GetVectorTypeReference(
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
                transpose_vectors<SrcData, DstScalarPerVector, SrcScalarPerVector>{}(
                    src_vector_refs, dst_vector_refs);
            });
        }

        // Do fast numeric convert
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access_for_src_and_dst_idle<SrcVectorDim,
                                                                  SrcScalarPerVector,
                                                                  DstVectorDim,
                                                                  DstScalarPerVector>{},
            Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        using src_vector_type = vector_type_maker_t<SrcData, SrcScalarPerVector>;
        using src_vector_t    = typename src_vector_type::type;

        using src_converted_vector_type = vector_type_maker_t<DstData, SrcScalarPerVector>;
        using src_converted_vector_t    = typename src_converted_vector_type::type;
        // Vector-wise type convert
        static_ford<decltype(access_lengths)>{}([&](auto access_idx) {
            auto src_vector_container = src_vector_type{
                src_thread_scratch_tuple_[thread_scratch_id].template GetAsType<src_vector_t>(
                    access_idx)};

            auto src_converted_vector_container =
                src_converted_vector_type{fast_numeric_converter(src_vector_container)};

            src_converted_thread_scratch_.template SetAsType<src_converted_vector_t>(
                access_idx,
                src_converted_vector_container.template AsType<src_converted_vector_t>()[I0]);
        });

        // Element-scale operation, expect packed multiplication
        static_ford<SliceLengths>{}([&](auto idx) {
            DstData dst_v;
            constexpr auto scale_idx = Sequence<I0, idx.At(1), I0>{};
            // printf("Tid: %03d, scale: %04x\n", get_thread_local_1d_id(),
            // *(reinterpret_cast<const uint16_t*>(&scale_thread_scratch_[scale_idx])));
            src_element_op_(dst_v,
                            src_converted_thread_scratch_[idx] * scale_thread_scratch_[scale_idx]);
            dst_thread_scratch_(idx) = dst_v;
        });
#endif
    }

    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // if there is transpose, it's done here
        // TODO move this elsewhere
        TransferDataFromSrcThreadScratchToDstThreadScratch(thread_scratch_id);

        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        // src scalar per access on each dim
        // TODO: don't use this
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // make forward steps
        const auto dst_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(dst_desc, forward_step_idx);
            },
            Number<nDim>{});

        // make backward steps
        const auto dst_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(dst_desc, backward_step_idx);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_dst_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_idx[i]
                                                      : ordered_dst_access_lengths[i] - 1 -
                                                            ordered_dst_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                       dst_scalar_per_access;
            }();

            constexpr auto dst_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<dst_data_idx[i]>{}; }, Number<dst_data_idx.Size()>{});

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            using dst_vector_type = vector_type_maker_t<DstData, DstScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            // copy data from dst_thread_scratch_ into dst_vector_container
            auto dst_vector_container = dst_vector_type{
                dst_thread_scratch_.template GetAsType<dst_vector_t>(dst_data_idx_seq)};

            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                DstData dst_v;

                // apply DstElementwiseOperation
                dst_element_op_(dst_v, dst_vector_container.template AsType<DstData>()[i]);

                dst_vector_container.template AsType<DstData>()(i) = dst_v;
            });

            // copy data from dst_vector_container to dst_buf
            dst_buf.template Set<dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector_container.template AsType<dst_vector_t>()[I0]);

            constexpr auto move_on_dim = [&]() constexpr {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_dst_access_idx[i] < ordered_dst_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_dst_access_idx[j] == ordered_dst_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }();

            // move dst coord
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_forward_steps[dst_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_backward_steps[dst_dim_access_order[i]]);
                    }
                }
            });
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_src_access_lengths[I0] - 1;

                static_for<1, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate src data index after last iteration in RunRead(), if it has not being reset by
        // RunRead()
        constexpr auto src_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_src_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                   src_scalar_per_access;
        }();

        //
        constexpr auto reset_src_data_step = [&]() {
            Index reset_src_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_src_data_step_(i) = -src_data_idx[i]; });

            return reset_src_data_step_;
        }();

        return reset_src_data_step;
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_dst_access_lengths[I0] - 1;

                static_for<1, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate dst data index after last iteration in RunWrite(), if it has not being reset by
        // RunWrite()
        constexpr auto dst_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                   dst_scalar_per_access;
        }();

        //
        constexpr auto reset_dst_data_step = [&]() {
            Index reset_dst_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_dst_data_step_(i) = -dst_data_idx[i]; });

            return reset_dst_data_step_;
        }();

        return reset_dst_data_step;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

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

    __device__ static constexpr auto GetScaleThreadScratchDescriptor()
    {

        constexpr auto scale_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, ScaleScalarPerVector>{}, Number<nDim>{});

        constexpr auto scale_access_lengths = SliceLengths{} / scale_scalar_per_access;

        constexpr auto scale_access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(scale_access_lengths), Number<ScaleScalarPerVector>{});

        // 1st stage of transforms
        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(scale_access_lengths_and_vector_length);

        // 2nd stage of transforms
        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(scale_access_lengths_and_vector_length[i],
                                   scale_access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(scale_access_lengths_and_vector_length[i]);
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
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

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

    private:
    static constexpr auto src_thread_scratch_desc_ = decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto scale_thread_scratch_desc_ =
        decltype(GetScaleThreadScratchDescriptor()){};
    static constexpr auto dst_thread_scratch_desc_ = decltype(GetDstThreadScratchDescriptor()){};

    /*
        template <bool kLastDim>
        struct ScaleThreadScratchDesc{};
    */

    // Registers, contain raw data loaded from global buffer
    using SrcThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             SrcData,
                                                             SrcScalarPerVector,
                                                             decltype(src_thread_scratch_desc_),
                                                             true>;

    // Registers, contain fast converted data
    using SrcThreadConvertedScratch =
        StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                        DstData,
                                        SrcScalarPerVector,
                                        decltype(src_thread_scratch_desc_),
                                        true>;

    // Registers, contain scale data
    using ScaleThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                               ScaleData,
                                                               ScaleScalarPerVector,
                                                               decltype(scale_thread_scratch_desc_),
                                                               true>;

    // Registers, contain dequantized data
    using DstThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             DstData,
                                                             DstScalarPerVector,
                                                             decltype(dst_thread_scratch_desc_),
                                                             true>;

    using FastTypeConverter = tensor_operation::element_wise::
        FastNumericArrayConverter<SrcData, DstData, SrcScalarPerVector>;

    StaticallyIndexedArray<SrcThreadScratch, NumThreadScratch> src_thread_scratch_tuple_;
    SrcThreadConvertedScratch src_converted_thread_scratch_;
    ScaleThreadScratch scale_thread_scratch_;

    DstThreadScratch dst_thread_scratch_;
    FastTypeConverter fast_numeric_converter;

    SrcCoord src_coord_;
    ScaleCoord scale_coord_;
    DstCoord dst_coord_;
    const SrcElementwiseOperation src_element_op_;
    const ScaleElementwiseOperation scale_element_op_;
    const DstElementwiseOperation dst_element_op_;
};

} // namespace ck
