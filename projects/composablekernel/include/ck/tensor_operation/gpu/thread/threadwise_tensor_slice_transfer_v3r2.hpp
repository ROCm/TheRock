// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/utility/is_detected.hpp"

namespace ck {

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
template <typename SliceLengths,
          typename ElementwiseOperation,
          typename DstInMemOps, // Sequence
          typename SrcDatas,
          typename DstDatas,
          typename SrcDescs,
          typename DstDescs,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          typename SrcsScalarPerVector,         // Sequence
          typename DstsScalarPerVector,         // Sequence
          typename SrcsScalarStrideInVector,    // Sequence
          typename DstsScalarStrideInVector,    // Sequence
          typename SrcsResetCoordinateAfterRun, // control whether to move back src coordinate after
                                                // each RunRead(),  will be fused with
                                                // MoveSrcSliceWindow to save addr computation
          typename DstsResetCoordinateAfterRun, // control whether to move back dst coordinate after
                                                // each RunWrite(),  will be fused with
                                                // MoveDstSliceWindow to save addr computation
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_v3r2
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    static constexpr index_t nSrc = SrcDescs::Size();
    static constexpr index_t nDst = DstDescs::Size();

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

    static constexpr auto I0 = Number<0>{};

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3r2(
        const SrcDescs& src_descs,
        const StaticallyIndexedArray<Index, nSrc>& src_slice_origins,
        const DstDescs& dst_descs,
        const StaticallyIndexedArray<Index, nDst>& dst_slice_origins,
        const ElementwiseOperation& element_op)
        : src_coords_(MakeCoordinates(src_descs, src_slice_origins)),
          dst_coords_(MakeCoordinates(dst_descs, dst_slice_origins)),
          element_op_(element_op)
    {
    }

    template <typename Indices, enable_if_t<SrcDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetSrcSliceOrigins(const SrcDescs& src_descs,
                                       const Indices& src_slice_origin_idxs)
    {
        static_for<0, nSrc, 1>{}([&](auto src_i) {
            src_coords_(src_i) =
                make_tensor_coordinate(src_descs.At(src_i), src_slice_origin_idxs[src_i]);
        });
    }

    template <typename Indices, enable_if_t<DstDescs::Size() == Indices::Size(), bool> = false>
    __device__ void SetDstSliceOrigins(const DstDescs& dst_descs,
                                       const Indices& dst_slice_origin_idxs)
    {
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            dst_coords_(dst_i) =
                make_tensor_coordinate(dst_descs.At(dst_i), dst_slice_origin_idxs[dst_i]);
        });
    }

    template <typename SrcBuffers, index_t ThreadScratchId = 0>
    __device__ void RunRead(const SrcDescs& src_descs,
                            const SrcBuffers& src_bufs,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access_tuple = generate_tuple(
            [&](auto src_i) {
                return generate_sequence(
                    detail::lambda_scalar_per_access<SrcVectorDim,
                                                     SrcsScalarPerVector::At(src_i)>{},
                    Number<nDim>{});
            },
            Number<nSrc>{});

        constexpr auto src_access_lengths_tuple = generate_tuple(
            [&](auto src_i) {
                return SliceLengths{} / src_scalar_per_access_tuple.At(src_i);
                static_assert(
                    SliceLengths::At(SrcVectorDim) % SrcsScalarPerVector::At(src_i) == 0,
                    "SliceLengths[SrcVectorDim] must be divisible by SrcsScalarPerVector");
            },
            Number<nSrc>{});

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths_tuple = generate_tuple(
            [&](auto src_i) {
                return container_reorder_given_new2old(src_access_lengths_tuple.At(src_i),
                                                       src_dim_access_order);
            },
            Number<nSrc>{});

        // make forward steps
        const auto src_forward_steps_tuple = generate_tuple(
            [&](auto src_i) {
                return generate_tuple(
                    [&](auto i) {
                        Index forward_step_idx;

                        static_for<0, nDim, 1>{}([&](auto j) {
                            forward_step_idx(j) =
                                (i.value == j.value) ? src_scalar_per_access_tuple.At(src_i)[i] : 0;
                        });

                        return make_tensor_coordinate_step(src_descs.At(src_i), forward_step_idx);
                    },
                    Number<nDim>{});
            },
            Number<nSrc>{});

        // make backward steps
        const auto src_backward_steps_tuple = generate_tuple(
            [&](auto src_i) {
                return generate_tuple(
                    [&](auto i) {
                        Index backward_step_idx;

                        static_for<0, nDim, 1>{}([&](auto j) {
                            backward_step_idx(j) = (i.value == j.value)
                                                       ? -src_scalar_per_access_tuple.At(src_i)[i]
                                                       : 0;
                        });

                        return make_tensor_coordinate_step(src_descs.At(src_i), backward_step_idx);
                    },
                    Number<nDim>{});
            },
            Number<nSrc>{});

        // loop over tensor and copy
        static_for<0, nSrc, 1>{}([&](auto src_i) {
            static_ford<remove_cvref_t<decltype(ordered_src_access_lengths_tuple.At(src_i))>>{}(
                [&](auto ordered_src_access_idx) {
                    // judge move forward or move backward
                    constexpr auto forward_sweep = [&]() {
                        StaticallyIndexedArray<bool, nDim> forward_sweep_;

                        forward_sweep_(I0) = true;

                        static_for<1, nDim, 1>{}([&](auto i) {
                            index_t tmp = ordered_src_access_idx[I0];

                            static_for<1, i, 1>{}([&](auto j) {
                                tmp = tmp * ordered_src_access_lengths_tuple[j] +
                                      ordered_src_access_idx[j];
                            });

                            forward_sweep_(i) = tmp % 2 == 0;
                        });

                        return forward_sweep_;
                    }();

                    // calculate src data index
                    constexpr auto src_data_idx = [&]() {
                        Index ordered_idx;

                        static_for<0, nDim, 1>{}([&](auto i) {
                            ordered_idx(i) = forward_sweep[i]
                                                 ? ordered_src_access_idx[i]
                                                 : ordered_src_access_lengths_tuple.At(src_i)[i] -
                                                       1 - ordered_src_access_idx[i];
                        });

                        return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                               src_scalar_per_access_tuple.At(src_i);
                    }();

                    constexpr auto src_data_idx_seq =
                        generate_sequence_v2([&](auto i) { return Number<src_data_idx[i]>{}; },
                                             Number<src_data_idx.Size()>{});

                    const bool is_src_valid =
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            src_descs.At(src_i), src_coords_.At(src_i));

                    using src_vector_type = vector_type_maker_t<tuple_element_t<src_i, SrcDatas>,
                                                                SrcsScalarPerVector::At(src_i)>;
                    using src_vector_t    = typename src_vector_type::type;

                    // copy data from src_buf into src_vector_container
                    auto src_vector_container =
                        src_vector_type{src_bufs.At(src_i).template Get<src_vector_t>(
                            src_coords_.At(src_i).GetOffset(), is_src_valid)};

                    // copy data from src_vector_container into src_thread_scratch_
                    src_thread_scratch_tuple_(thread_scratch_id)
                        .At(src_i)
                        .template SetAsType<src_vector_t>(
                            src_data_idx_seq,
                            src_vector_container.template AsType<src_vector_t>()[I0]);

                    constexpr auto move_on_dim = [&]() constexpr {
                        StaticallyIndexedArray<bool, nDim> move_on_dim_;

                        static_for<0, nDim, 1>{}([&](auto i) {
                            move_on_dim_(i) = ordered_src_access_idx[i] <
                                              ordered_src_access_lengths_tuple.At(src_i)[i] - 1;

                            static_for<i + 1, nDim, 1>{}([&](auto j) {
                                move_on_dim_(i) &=
                                    ordered_src_access_idx[j] ==
                                    ordered_src_access_lengths_tuple.At(src_i)[j] - 1;
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
                                    src_descs.At(src_i),
                                    src_coords_.At(src_i),
                                    src_forward_steps_tuple.At(src_i)[src_dim_access_order[i]]);
                            }
                            else
                            {
                                move_tensor_coordinate(
                                    src_descs.At(src_i),
                                    src_coords_.At(src_i),
                                    src_backward_steps_tuple.At(src_i)[src_dim_access_order[i]]);
                            }
                        }
                    });
                });
        });

        static_for<0, nSrc, 1>{}([&](auto src_i) {
            // move src coordinate back to slice origin (or not)
            if constexpr(SrcsResetCoordinateAfterRun::At(src_i))
            {
                const auto src_reset_step = make_tensor_coordinate_step(
                    src_descs.At(src_i), GetSrcCoordinateResetStep<src_i>());

                move_tensor_coordinate(src_descs.At(src_i), src_coords_.At(src_i), src_reset_step);
            }
        });
    }

    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
        // TODO: Add support for CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
        // (it requires to add Elementwise support in transpose_vectors)
        static_ford<SliceLengths>{}([&](auto idx) {
            const auto src_data_refs = generate_tie(
                [&](auto src_i) -> const auto& {
                    return src_thread_scratch_tuple_[thread_scratch_id].At(src_i)[idx];
                },
                Number<nSrc>{});

            auto dst_data_refs = generate_tie(
                [&](auto dst_i) -> auto& { return dst_thread_scratch_tuple_.At(dst_i)(idx); },
                Number<nDst>{});
            unpack2(element_op_, dst_data_refs, src_data_refs);
        });
    }

    template <typename DstBuffers, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDescs& dst_descs,
                             DstBuffers& dst_bufs,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        // if there is transpose, it's done here
        // TODO move this elsewhere
        TransferDataFromSrcThreadScratchToDstThreadScratch(thread_scratch_id);

        // src scalar per access on each dim
        // TODO: don't use this
        constexpr auto dst_scalar_per_access_tuple = generate_tuple(
            [&](auto dst_i) {
                return generate_sequence(
                    detail::lambda_scalar_per_access<DstVectorDim,
                                                     DstsScalarPerVector::At(dst_i)>{},
                    Number<nDim>{});
            },
            Number<nDst>{});

        constexpr auto dst_access_lengths_tuple = generate_tuple(
            [&](auto dst_i) { return SliceLengths{} / dst_scalar_per_access_tuple.At(dst_i); },
            Number<nDst>{});

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths_tuple = generate_tuple(
            [&](auto dst_i) {
                return container_reorder_given_new2old(dst_access_lengths_tuple.At(dst_i),
                                                       dst_dim_access_order);
            },
            Number<nDst>{});

        // make forward steps
        const auto dst_forward_steps_tuple = generate_tuple(
            [&](auto dst_i) {
                return generate_tuple(
                    [&](auto i) {
                        Index forward_step_idx;

                        static_for<0, nDim, 1>{}([&](auto j) {
                            forward_step_idx(j) =
                                (i.value == j.value) ? dst_scalar_per_access_tuple.At(dst_i)[i] : 0;
                        });

                        return make_tensor_coordinate_step(dst_descs.At(dst_i), forward_step_idx);
                    },
                    Number<nDim>{});
            },
            Number<nDst>{});

        // make backward steps
        const auto dst_backward_steps_tuple = generate_tuple(
            [&](auto dst_i) {
                return generate_tuple(
                    [&](auto i) {
                        Index backward_step_idx;

                        static_for<0, nDim, 1>{}([&](auto j) {
                            backward_step_idx(j) = (i.value == j.value)
                                                       ? -dst_scalar_per_access_tuple.At(dst_i)[i]
                                                       : 0;
                        });

                        return make_tensor_coordinate_step(dst_descs.At(dst_i), backward_step_idx);
                    },
                    Number<nDim>{});
            },
            Number<nDst>{});

        // loop over tensor and copy
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            static_ford<remove_cvref_t<decltype(ordered_dst_access_lengths_tuple.At(dst_i))>>{}(
                [&](auto ordered_dst_access_idx) {
                    // judge move forward or move backward
                    constexpr auto forward_sweep = [&]() {
                        StaticallyIndexedArray<bool, nDim> forward_sweep_;

                        forward_sweep_(I0) = true;

                        static_for<1, nDim, 1>{}([&](auto i) {
                            index_t tmp = ordered_dst_access_idx[I0];

                            static_for<1, i, 1>{}([&](auto j) {
                                tmp = tmp * ordered_dst_access_lengths_tuple.At(dst_i)[j] +
                                      ordered_dst_access_idx[j];
                            });

                            forward_sweep_(i) = tmp % 2 == 0;
                        });

                        return forward_sweep_;
                    }();

                    // calculate dst data index
                    constexpr auto dst_data_idx = [&]() {
                        Index ordered_idx;

                        static_for<0, nDim, 1>{}([&](auto i) {
                            ordered_idx(i) = forward_sweep[i]
                                                 ? ordered_dst_access_idx[i]
                                                 : ordered_dst_access_lengths_tuple.At(dst_i)[i] -
                                                       1 - ordered_dst_access_idx[i];
                        });

                        return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                               dst_scalar_per_access_tuple.At(dst_i);
                    }();

                    constexpr auto dst_data_idx_seq =
                        generate_sequence_v2([&](auto i) { return Number<dst_data_idx[i]>{}; },
                                             Number<dst_data_idx.Size()>{});

                    const bool is_dst_valid =
                        coordinate_has_valid_offset_assuming_visible_index_is_valid(
                            dst_descs.At(dst_i), dst_coords_.At(dst_i));

                    using dst_vector_type = vector_type_maker_t<tuple_element_t<dst_i, DstDatas>,
                                                                DstsScalarPerVector::At(dst_i)>;
                    using dst_vector_t    = typename dst_vector_type::type;

                    // copy data from dst_thread_scratch_ into dst_vector_container
                    auto dst_vector_container = dst_vector_type{
                        dst_thread_scratch_tuple_.At(dst_i).template GetAsType<dst_vector_t>(
                            dst_data_idx_seq)};

                    constexpr InMemoryDataOperationEnum DstInMemOp =
                        static_cast<InMemoryDataOperationEnum>(DstInMemOps::At(dst_i.value));

                    // copy data from dst_vector_container to dst_buf
                    dst_bufs.At(dst_i).template Update<DstInMemOp, dst_vector_t>(
                        dst_coords_.At(dst_i).GetOffset(),
                        is_dst_valid,
                        dst_vector_container.template AsType<dst_vector_t>()[I0]);

                    constexpr auto move_on_dim = [&]() constexpr {
                        StaticallyIndexedArray<bool, nDim> move_on_dim_;

                        static_for<0, nDim, 1>{}([&](auto i) {
                            move_on_dim_(i) = ordered_dst_access_idx[i] <
                                              ordered_dst_access_lengths_tuple.At(dst_i)[i] - 1;

                            static_for<i + 1, nDim, 1>{}([&](auto j) {
                                move_on_dim_(i) &=
                                    ordered_dst_access_idx[j] ==
                                    ordered_dst_access_lengths_tuple.At(dst_i)[j] - 1;
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
                                    dst_descs.At(dst_i),
                                    dst_coords_.At(dst_i),
                                    dst_forward_steps_tuple.At(dst_i)[dst_dim_access_order[i]]);
                            }
                            else
                            {
                                move_tensor_coordinate(
                                    dst_descs.At(dst_i),
                                    dst_coords_.At(dst_i),
                                    dst_backward_steps_tuple.At(dst_i)[dst_dim_access_order[i]]);
                            }
                        }
                    });
                });
        });

        // move dst coordinate back to slice origin (or not)
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            if constexpr(DstsResetCoordinateAfterRun::At(dst_i))
            {
                const auto dst_reset_step = make_tensor_coordinate_step(
                    dst_descs.At(dst_i), GetDstCoordinateResetStep<dst_i>());

                move_tensor_coordinate(dst_descs.At(dst_i), dst_coords_.At(dst_i), dst_reset_step);
            }
        });
    }

    template <index_t src_i>
    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcsScalarPerVector::At(src_i)>{},
            Number<nDim>{});

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

    template <index_t dst_i>
    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstsScalarPerVector::At(dst_i)>{},
            Number<nDim>{});

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
                   dst_scalar_per_access.At(dst_i);
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
    __device__ void MoveSrcSliceWindow(const SrcDescs& src_descs,
                                       const Index& src_slice_origin_step_idx)
    {
        static_for<0, nSrc, 1>{}([&](auto src_i) {
            // if src coord was not reset by RunRead(), then need to adjust the step here
            const auto adjusted_step_idx =
                SrcsResetCoordinateAfterRun::At(src_i)
                    ? src_slice_origin_step_idx
                    : src_slice_origin_step_idx + GetSrcCoordinateResetStep<src_i>();

            // is it OK to construct a new step every time?
            const auto adjusted_step =
                make_tensor_coordinate_step(src_descs.At(src_i), adjusted_step_idx);

            move_tensor_coordinate(src_descs.At(src_i), src_coords_.At(src_i), adjusted_step);
        });
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDescs& dst_descs,
                                       const Index& dst_slice_origin_step_idx)
    {
        static_for<0, nDst, 1>{}([&](auto dst_i) {
            // if dst coord was not reset by RunWrite(), then need to adjust the step here
            const auto adjusted_step_idx =
                DstsResetCoordinateAfterRun::At(dst_i)
                    ? dst_slice_origin_step_idx
                    : dst_slice_origin_step_idx + GetDstCoordinateResetStep<dst_i>();

            // is it OK to construct a new step every time?
            const auto adjusted_step =
                make_tensor_coordinate_step(dst_descs.At(dst_i), adjusted_step_idx);

            move_tensor_coordinate(dst_descs.At(dst_i), dst_coords_.At(dst_i), adjusted_step);
        });
    }

    template <index_t src_i>
    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcsScalarPerVector::At(src_i)>{},
            Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_access_lengths_and_vector_length =
            container_push_back(sequence_to_tuple_of_number(src_access_lengths),
                                Number<SrcsScalarPerVector::At(src_i)>{});

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

    template <index_t dst_i>
    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        // 1st stage of transforms
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstsScalarPerVector::At(dst_i)>{},
            Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_access_lengths_and_vector_length =
            container_push_back(sequence_to_tuple_of_number(dst_access_lengths),
                                Number<DstsScalarPerVector::At(dst_i)>{});

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

    __device__ static constexpr auto MakeSrcThreadScratchTuple()
    {
        return generate_tuple(
            [&](auto src_i) {
                constexpr auto src_thread_scratch_desc =
                    decltype(GetSrcThreadScratchDescriptor<src_i>()){};
                using SrcThreadScratch =
                    StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                    tuple_element_t<src_i, SrcDatas>,
                                                    SrcsScalarPerVector::At(src_i),
                                                    decltype(src_thread_scratch_desc),
                                                    true>;
                return SrcThreadScratch{};
            },
            Number<nSrc>{});
    }

    __device__ static constexpr auto MakeDstThreadScratchTuple()
    {
        return generate_tuple(
            [&](auto dst_i) {
                constexpr auto dst_thread_scratch_desc =
                    decltype(GetDstThreadScratchDescriptor<dst_i>()){};
                using DstThreadScratch =
                    StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                    tuple_element_t<dst_i, DstDatas>,
                                                    DstsScalarPerVector::At(dst_i),
                                                    decltype(dst_thread_scratch_desc),
                                                    true>;
                return DstThreadScratch{};
            },
            Number<nDst>{});
    }

    private:
    using SrcThreadScratchTuple = decltype(MakeSrcThreadScratchTuple());
    using DstThreadScratchTuple = decltype(MakeDstThreadScratchTuple());

    StaticallyIndexedArray<SrcThreadScratchTuple, NumThreadScratch> src_thread_scratch_tuple_;

    DstThreadScratchTuple dst_thread_scratch_tuple_;

    SrcCoords src_coords_;
    DstCoords dst_coords_;
    const ElementwiseOperation element_op_;
};

} // namespace ck
