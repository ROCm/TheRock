// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/meta_data_buffer.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename Distribution>
CK_TILE_HOST_DEVICE auto get_partition_index(Distribution)
{
    return Distribution::get_partition_index();
}

// distributed span
template <index_t... PartialHsLengths>
struct tile_distributed_span
{
    using Impl = sequence<PartialHsLengths...>;

    static constexpr auto impl_ = Impl{};

    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; }
};

// distributed index
template <index_t... PartialHsIndices>
struct tile_distributed_index
{
    using Impl = sequence<PartialHsIndices...>;

    static constexpr auto impl_ = Impl{};

    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; }
};

namespace detail {

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto make_tile_distributed_span(sequence<Is...>)
{
    return tile_distributed_span<Is...>{};
}

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto make_tile_distributed_index(sequence<Is...>)
{
    return tile_distributed_index<Is...>{};
}

} // namespace detail

template <typename PsYs2XsAdaptor_,
          typename Ys2DDescriptor_,
          typename StaticTileDistributionEncoding_,
          typename TileDistributionDetail_> // FIXME: this is for hold ad-hoc but useful info,
                                            // should be more elegnat
struct tile_distribution
{
    using PsYs2XsAdaptor = remove_cvref_t<PsYs2XsAdaptor_>;
    using Ys2DDescriptor = remove_cvref_t<Ys2DDescriptor_>;
    using DstrEncode     = remove_cvref_t<StaticTileDistributionEncoding_>;
    using DstrDetail     = remove_cvref_t<TileDistributionDetail_>;

    static_assert(PsYs2XsAdaptor::is_static() && Ys2DDescriptor::is_static(),
                  "wrong! should be static");

    static constexpr index_t NDimX = PsYs2XsAdaptor::get_num_of_bottom_dimension();
    static constexpr index_t NDimY = Ys2DDescriptor::get_num_of_top_dimension();
    static constexpr index_t NDimP = PsYs2XsAdaptor::get_num_of_top_dimension() - NDimY;
    static constexpr index_t NDimR = StaticTileDistributionEncoding_::NDimR;

    PsYs2XsAdaptor ps_ys_to_xs_;
    Ys2DDescriptor ys_to_d_;

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_dimension_x() { return NDimX; }
    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_dimension_y() { return NDimY; }
    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_dimension_p() { return NDimP; }
    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_dimension_r() { return NDimR; }

    CK_TILE_HOST_DEVICE static auto get_partition_index()
    {
        // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        if constexpr(NDimP == 1)
        {
            return array<index_t, 1>{get_lane_id()};
        }
        else if constexpr(NDimP == 2)
        {
            return array<index_t, 2>{get_warp_id(), get_lane_id()};
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_lengths()
    {
#if 0
        // FIXME: tensor_adaptor::GetBottomDimensionLengths is wrong. re-enable this after it's fixed
        ps_ys_to_xs_.GetBottomDimensionLengths();
#else
        return generate_tuple(
            [&](auto i) {
                constexpr index_t x_length =
                    container_reduce(typename DstrEncode::HsLengthss{}[i], multiplies{}, 1);

                return number<x_length>{};
            },
            number<NDimX>{});
#endif
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_ps_ys_to_xs_adaptor() const
    {
        return ps_ys_to_xs_;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_ys_to_d_descriptor() const { return ys_to_d_; }

    CK_TILE_HOST_DEVICE static constexpr auto get_static_tile_distribution_encoding()
    {
        return DstrEncode{};
    }

#if 1
    // Calculate Replication index [R0, R1, ...] based on Partion index
    // FIXME: very nasty implementation
    template <typename PartitionIndex>
    CK_TILE_HOST_DEVICE auto calculate_rs_index_from_ps_index(const PartitionIndex& ps_idx) const
    {
        static_assert(PartitionIndex::size() == NDimP, "wrong!");

        const auto ps_ys_idx = container_concat(ps_idx, array<index_t, NDimY>{0});

        const auto dummy_adaptor_coord = make_tensor_adaptor_coordinate(ps_ys_to_xs_, ps_ys_idx);

        array<index_t, NDimR> rs_idx;

        static_for<0, NDimP, 1>{}([&](auto idim_p) {
            constexpr index_t ndim_low = DstrEncode::ps_to_rhss_major_[idim_p].size();

            static_for<0, ndim_low, 1>{}([&](auto i) {
                constexpr index_t rh_major = DstrEncode::ps_to_rhss_major_[idim_p][i];
                constexpr index_t rh_minor = DstrEncode::ps_to_rhss_minor_[idim_p][i];

                // 0-th rh_major is the replicate dimension
                if constexpr(rh_major == 0)
                {
                    constexpr index_t adaptor_hidden_id =
                        DstrDetail::rh_major_minor_to_adaptor_hidden_idss_[rh_major][rh_minor];

                    // fill in
                    rs_idx(rh_minor) = dummy_adaptor_coord.get_hidden_index()[adaptor_hidden_id];
                }
            });
        });

        return rs_idx;
    }
#endif

    template <typename PartitionIndex = decltype(get_partition_index())>
    CK_TILE_HOST_DEVICE auto
    calculate_index(const PartitionIndex& ps_idx = get_partition_index()) const
    {
        const auto ps_ys_idx = container_concat(ps_idx, array<index_t, NDimY>{0});
        const auto window_adaptor_thread_coord_tmp =
            make_tensor_adaptor_coordinate(ps_ys_to_xs_, ps_ys_idx);
        return window_adaptor_thread_coord_tmp.get_bottom_index();
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_distributed_spans()
    {
        constexpr auto distributed_spans_impl = DstrEncode::detail::distributed_spans_lengthss_;
        constexpr auto ndims_spans_minor      = DstrEncode::detail::ndims_distributed_spans_minor_;

        return generate_tuple(
            [&](auto i) {
                constexpr auto span_impl          = distributed_spans_impl[i];
                constexpr index_t ndim_span_minor = ndims_spans_minor[i];

                constexpr auto span = TO_SEQUENCE(span_impl, ndim_span_minor);

                return detail::make_tile_distributed_span(span);
            },
            number<NDimX>{});
    }

    // FIXME: it's hacky to get Y index from Distributed-Index
    template <typename DistributedIndices>
    CK_TILE_HOST_DEVICE static constexpr auto
    get_y_indices_from_distributed_indices(DistributedIndices)
    {
        constexpr auto ys_idx_arr = [] {
            array<index_t, NDimY> ys_idx;

            static_for<0, NDimY, 1>{}([&](auto i) {
                constexpr index_t span_major = DstrEncode::detail::ys_to_span_major_[i];
                constexpr index_t span_minor = DstrEncode::detail::ys_to_span_minor_[i];

                constexpr auto dstr_index = DistributedIndices{}[number<span_major>{}];

                ys_idx(i) = dstr_index.impl_[span_minor];
            });

            return ys_idx;
        }();

        constexpr index_t ndim_y = NDimY;

        return TO_SEQUENCE(ys_idx_arr, ndim_y);
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_static()
    {
        return PsYs2XsAdaptor::is_static() && Ys2DDescriptor::is_static();
    }
};

template <typename T>
struct is_tile_distribution : std::false_type
{
};
template <typename PsYs2XsAdaptor,
          typename Ys2DDescriptor,
          typename StaticTileDistributionEncoding,
          typename TileDistributionDetail>
struct is_tile_distribution<tile_distribution<PsYs2XsAdaptor,
                                              Ys2DDescriptor,
                                              StaticTileDistributionEncoding,
                                              TileDistributionDetail>> : std::true_type
{
};
template <typename T>
inline constexpr bool is_tile_distribution_v = is_tile_distribution<T>::value;

namespace detail {

template <index_t NDimMax>
CK_TILE_HOST_DEVICE constexpr auto make_sequential_index(index_t ibegin, index_t iend)
{
    array<index_t, NDimMax> arr{0};

    for(index_t i = 0; i < iend - ibegin; ++i)
    {
        arr(i) = ibegin + i;
    }

    return arr;
}

// this returns a constexpr encoding of tile_distribution
template <typename StaticTileDistributionEncoding_>
CK_TILE_HOST_DEVICE constexpr auto
make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_)
{
    using RsLengths    = typename StaticTileDistributionEncoding_::RsLengths;
    using HsLengthss   = typename StaticTileDistributionEncoding_::HsLengthss;
    using Ps2RHssMajor = typename StaticTileDistributionEncoding_::Ps2RHssMajor;
    using Ps2RHssMinor = typename StaticTileDistributionEncoding_::Ps2RHssMinor;
    using Ys2RHsMajor  = typename StaticTileDistributionEncoding_::Ys2RHsMajor;
    using Ys2RHsMinor  = typename StaticTileDistributionEncoding_::Ys2RHsMinor;

    // FIXME: increase max value if fail
    constexpr index_t kMaxNumTransforms = 20;
    constexpr index_t kMaxMetaDataSize  = 128;
    constexpr index_t kMaxNumDim        = 10;

    using Name     = coord_transform_enum;
    using MetaData = meta_data_buffer<kMaxMetaDataSize>;
    using NumDim   = index_t;
    using Dims     = array<index_t, kMaxNumDim>;
    using Lengths  = array<index_t, kMaxNumDim>;

    // Tile Adaptor
    //   bottom dims [x0, x1, x2, ...]
    //   top dims [p0, p1, ..., y0, y1, ...]
    constexpr index_t ndim_x = HsLengthss::size();

    // Dim Ids: [idim_x_major, idim_x_minor] to [idim_hidden]
    array<array<index_t, kMaxNumDim>, ndim_x + 1> rh_major_minor_to_hidden_ids;
    array<array<index_t, kMaxNumDim>, ndim_x + 1> rh_major_minor_to_hidden_lengths;

    auto trans = array<tuple<Name, MetaData, NumDim, Dims, NumDim, Dims>, kMaxNumTransforms>{};

    index_t num_tran       = 0;
    index_t hidden_dim_cnt = ndim_x;

    // this is replicate transform
    {
        constexpr index_t ndim_r_minor = RsLengths::size();

        constexpr auto r_minor_lengths = RsLengths{};

        trans(num_tran++) = {
            coord_transform_enum::replicate,
            MetaData{to_array<index_t, ndim_r_minor>(r_minor_lengths)},
            NumDim{0},
            Dims{},
            NumDim{ndim_r_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_r_minor)};

        for(index_t i = 0; i < ndim_r_minor; ++i)
        {
            rh_major_minor_to_hidden_ids(0)(i)     = hidden_dim_cnt;
            rh_major_minor_to_hidden_lengths(0)(i) = r_minor_lengths[i];

            hidden_dim_cnt++;
        }
    };

    // these are Unmerge transforms for X dimesions
    static_for<0, ndim_x, 1>{}([&trans,
                                &num_tran,
                                &hidden_dim_cnt,
                                &rh_major_minor_to_hidden_ids,
                                &rh_major_minor_to_hidden_lengths](auto idim_x) {
        // typename HsLengthss::base{}.foo();
        constexpr auto h_minor_lengths =
            HsLengthss{}.get(idim_x); // std::tuple_element_t<idim_x, HsLengthss>{};
        // constexpr auto h_minor_lengths = impl::getv<idim_x>(HsLengthss{});

        constexpr index_t ndim_h_minor = h_minor_lengths.size();

        trans(num_tran++) = {
            coord_transform_enum::unmerge,
            MetaData{to_array<index_t, ndim_h_minor>(h_minor_lengths)},
            NumDim{1},
            Dims{idim_x},
            NumDim{ndim_h_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_h_minor)};

        for(index_t i = 0; i < ndim_h_minor; ++i)
        {
            rh_major_minor_to_hidden_ids(idim_x + 1)(i)     = hidden_dim_cnt;
            rh_major_minor_to_hidden_lengths(idim_x + 1)(i) = h_minor_lengths[i];

            hidden_dim_cnt++;
        }
    });

    // transform: P dimensions
    constexpr index_t ndim_p = Ps2RHssMajor::size();

    Dims hidden_dim_id_ps;

    static_for<0, ndim_p, 1>{}([&](auto iDimP) {
        //
        index_t hidden_dim_id_p = hidden_dim_cnt++;

        hidden_dim_id_ps(iDimP) = hidden_dim_id_p;

        constexpr auto p2RHsMajor = Ps2RHssMajor{}[iDimP];
        constexpr auto p2RHsMinor = Ps2RHssMinor{}[iDimP];

        static_assert(p2RHsMajor.size() == p2RHsMinor.size(), "wrong!");

        constexpr index_t ndim_low = p2RHsMajor.size();

        Dims low_dims;
        Lengths low_lengths;

        for(index_t i = 0; i < ndim_low; ++i)
        {
            index_t rh_major = p2RHsMajor[i];
            index_t rh_minor = p2RHsMinor[i];
            low_dims(i)      = rh_major_minor_to_hidden_ids[rh_major][rh_minor];
            low_lengths(i)   = rh_major_minor_to_hidden_lengths[rh_major][rh_minor];
        }

        trans(num_tran++) = {coord_transform_enum::merge,
                             MetaData{to_array<index_t, ndim_low>(low_lengths)},
                             NumDim{ndim_low},
                             low_dims,
                             NumDim{1},
                             Dims{hidden_dim_id_p}};
    });

    constexpr index_t ndim_bottom = ndim_x;

    constexpr auto bottom_dim_ids = make_sequential_index<kMaxNumDim>(0, ndim_bottom);

    constexpr auto ys_to_rhs_major = Ys2RHsMajor{};
    constexpr auto ys_to_rhs_minor = Ys2RHsMinor{};

    constexpr index_t ndim_y   = Ys2RHsMajor::size();
    constexpr index_t ndim_top = ndim_p + ndim_y;

    auto top_dim_ids = hidden_dim_id_ps;

    {
        for(index_t i = 0; i < ndim_y; ++i)
        {
            index_t rh_major        = ys_to_rhs_major[i];
            index_t rh_minor        = ys_to_rhs_minor[i];
            top_dim_ids(ndim_p + i) = rh_major_minor_to_hidden_ids[rh_major][rh_minor];
        }
    }

    //
    const auto ps_ys_to_xs_adaptor_encoding =
        make_tuple(trans, num_tran, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top);

    // descriptor: [y0, y1, ...] to [d]
    Lengths y_lengths;
    index_t d_length = 1;

    for(index_t i = 0; i < ndim_y; ++i)
    {
        index_t rh_major = ys_to_rhs_major[i];
        index_t rh_minor = ys_to_rhs_minor[i];
        index_t y_length = rh_major_minor_to_hidden_lengths[rh_major][rh_minor];
        y_lengths(i)     = y_length;
        d_length *= y_length;
    }

    auto tran = make_tuple(coord_transform_enum::unmerge,
                           MetaData{to_array<index_t, ndim_y>(y_lengths)},
                           NumDim{1},
                           Dims{0},
                           NumDim{ndim_y},
                           make_sequential_index<kMaxNumDim>(1, ndim_y + 1));

    const auto ys_to_d_adaptor_encoding = make_tuple(
        make_tuple(tran), 1, Dims{0}, 1, make_sequential_index<kMaxNumDim>(1, ndim_y + 1), ndim_y);

    return make_tuple(ps_ys_to_xs_adaptor_encoding,
                      ys_to_d_adaptor_encoding,
                      d_length,
                      rh_major_minor_to_hidden_ids);
}

// FIXME: this is nasty. Move it inside TileDistributionEncoding::detail
template <typename RhMajorMinor2AdaptorHiddenIdss> // tuple<sequence<...>, ...>
struct tile_distribution_detail
{
    static constexpr auto rh_major_minor_to_adaptor_hidden_idss_ =
        to_array_of_array(RhMajorMinor2AdaptorHiddenIdss{});
};

} // namespace detail

#if 0
// this returns a constexpr tile_distribution
template <typename StaticTileDistributionEncoding_>
CK_TILE_HOST_DEVICE constexpr auto make_tile_distribution(StaticTileDistributionEncoding_)
{
    using DstrEncode = remove_cvref_t<StaticTileDistributionEncoding_>;

    constexpr auto adaptor_impl =
        detail::make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_{});

    constexpr auto ps_ys_to_xs_adaptor_impl          = adaptor_impl.template at<0>();
    constexpr auto ys_to_d_adaptor_impl              = adaptor_impl.template at<1>();
    constexpr index_t d_length                       = adaptor_impl.template at<2>();
    constexpr auto rh_major_minor_to_hidden_ids_impl = adaptor_impl.template at<3>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(ps_ys_to_xs_adaptor_impl);

    constexpr auto ys_to_d_adaptor = CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(ys_to_d_adaptor_impl);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, d_length);

    //
    constexpr index_t ndim_rh_major = DstrEncode::detail::ndim_rh_major_;
    constexpr auto ndims_rhs_minor  = DstrEncode::detail::ndims_rhs_minor_;

    constexpr auto rh_major_minor_to_hidden_ids =
        TO_TUPLE_OF_SEQUENCE(rh_major_minor_to_hidden_ids_impl, ndim_rh_major, ndims_rhs_minor);

    return tile_distribution<
        remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
        remove_cvref_t<decltype(ys_to_d_descriptor)>,
        remove_cvref_t<DstrEncode>,
        detail::tile_distribution_detail<remove_cvref_t<decltype(rh_major_minor_to_hidden_ids)>>>{
        ps_ys_to_xs_adaptor, ys_to_d_descriptor};
}
#endif

// this returns a static tile_distribution
template <typename StaticTileDistributionEncoding_>
CK_TILE_HOST_DEVICE constexpr auto make_static_tile_distribution(StaticTileDistributionEncoding_)
{
    using DstrEncode = remove_cvref_t<StaticTileDistributionEncoding_>;

    constexpr auto adaptor_impl =
        detail::make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_{});

    constexpr auto ps_ys_to_xs_adaptor_impl          = adaptor_impl.template at<0>();
    constexpr auto ys_to_d_adaptor_impl              = adaptor_impl.template at<1>();
    constexpr index_t d_length                       = adaptor_impl.template at<2>();
    constexpr auto rh_major_minor_to_hidden_ids_impl = adaptor_impl.template at<3>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(ps_ys_to_xs_adaptor_impl);

    constexpr auto ys_to_d_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(ys_to_d_adaptor_impl);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, number<d_length>{});

    //
    constexpr index_t ndim_rh_major = DstrEncode::detail::ndim_rh_major_;
    constexpr auto ndims_rhs_minor  = DstrEncode::detail::ndims_rhs_minor_;

    constexpr auto rh_major_minor_to_hidden_ids =
        TO_TUPLE_OF_SEQUENCE(rh_major_minor_to_hidden_ids_impl, ndim_rh_major, ndims_rhs_minor);

    return tile_distribution<
        remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
        remove_cvref_t<decltype(ys_to_d_descriptor)>,
        remove_cvref_t<DstrEncode>,
        detail::tile_distribution_detail<remove_cvref_t<decltype(rh_major_minor_to_hidden_ids)>>>{
        ps_ys_to_xs_adaptor, ys_to_d_descriptor};
}

//***********************************************************************************

namespace detail {
//
// slice tensor from x_dim, result in split in y_dim, not p_dim.
// We don't support slice cross p_dim (aka, slice different threads)
// also, sliced along y_dim need be the first dim of current dim.
// Multiply Y dim before sliced dim does not make sense
//
// e.g
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice start:<0, 0>, end:<-1, 32>, (-1 means the last one)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 4, 2, 4> -> OK
//                     |--> slice along this Y dim, is the first dim of X1, totally 4 slices
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice start:<0, 0>, end:<-1, 8>, (-1 means the last one)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 2, 4> -> OK
//                           |--> slice along this Y dim, the P dim is 1 in the left, so is OK
//                                 totally 16 slices
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice start:<0, 0>, end:<-1, 4>, (-1 means the last one)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 1, 4> -> Fail
//                              |--> slice along this P dim, will split threads, not supported
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice start:<0, 0>, end:<-1, 16>, (-1 means the last one)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 2, 2, 4> -> OK
//                           |--> slice along this Y dim, but this Y sim need to split into 2
//                           subdime
//                                the P dim in the left is 1, means actually not crossing P
//
template <typename Distribution, index_t... XSliceBegins, index_t... XSliceEnds>
CK_TILE_HOST_DEVICE constexpr auto slice_distribution_from_x(
    Distribution, sequence<XSliceBegins...> x_slice_begins, sequence<XSliceEnds...> x_slice_ends)
{
    // NOTE: this function need to be called under constexpr context,
    // due to https://wg21.link/p2280r0 we have to use non-reference type for distribution
    using Encoding = decltype(Distribution::get_static_tile_distribution_encoding());

    static_assert(sizeof...(XSliceBegins) == sizeof...(XSliceEnds));
    static_assert(sizeof...(XSliceBegins) == Encoding::NDimX, "only support slice over h, not r");

    constexpr auto p_len_over_h = Encoding::detail::get_uniformed_p_dim_lengths_over_h();

    constexpr auto x_slice_ends_ = generate_sequence_v2(
        [&](auto i) {
            if constexpr(x_slice_ends[i] == -1)
            {
                // -1 means till the end
                constexpr auto x_length_ =
                    container_reduce(typename Encoding::HsLengthss{}[i], multiplies{}, number<1>{});
                return x_length_;
            }
            else
            {
                return x_slice_ends[i];
            }
        },
        number<x_slice_ends.size()>{});

    constexpr auto x_slice_lengths = x_slice_ends_ - x_slice_begins;

    constexpr auto x_slice_lengths_without_p = generate_sequence_v2(
        [&](auto i) constexpr {
            constexpr auto len_ = x_slice_lengths[i];
            static_assert(len_ % p_len_over_h[i] == 0,
                          "slice length must be dividable by p_len_over_h");
            return number<len_ / p_len_over_h[i]>{};
        },
        number<x_slice_lengths.size()>{});

    constexpr auto src_h_prefix_sum = Encoding::detail::get_h_dim_lengths_prefix_sum();
    constexpr auto src_y_info       = Encoding::detail::get_sorted_y_to_h_info();
    constexpr auto src_y_dims       = src_y_info[number<0>{}];
    constexpr auto src_y_maps       = src_y_info[number<1>{}];
    constexpr auto src_y_prefix_sum = src_y_info[number<2>{}];

    constexpr auto sliced_hlen_yidx_ylen = [&]() constexpr {
        auto y_slice_sorted_origins = make_zero_multi_index<Encoding::NDimY>();
        auto y_slice_lengths        = Encoding::detail::ys_lengths_;
        constexpr auto y_to_h_masks = Encoding::detail::get_y_to_h_masks();

        // This lambda will modify some value outside, so c++ will not treat return value as
        // constexpr
        // TODO: ugly
        auto new_h_lengths = transform_tuples(
            [&](auto h_len, auto id) {
                constexpr auto sliced_h = reverse_slice_sequence(
                    h_len, number<x_slice_lengths_without_p[id]>{}, y_to_h_masks[id]);

                constexpr auto sliced_h_lens  = sliced_h[number<0>{}];
                constexpr auto sliced_h_index = sliced_h[number<2>{}];

                // update y_slice_lengths
                constexpr auto uniformed_h_index = sliced_h_index + number<src_h_prefix_sum[id]>{};
                constexpr auto found_y_index     = container_find(src_y_dims, uniformed_h_index);
                constexpr auto y_to_h_dim_end    = src_y_prefix_sum[id + 1];

                static_assert(found_y_index >= 0 && found_y_index < src_y_dims.size(),
                              "not sliced at y dim, please check");

                {
                    constexpr auto sliced_y_to_h_lens =
                        pick_sequence_elements_by_mask(sliced_h_lens, y_to_h_masks[id]);
                    constexpr auto sliced_y_to_h_dims = sliced_y_to_h_lens.size();
                    static_for<0, sliced_y_to_h_dims, 1>{}([&](auto i) {
                        y_slice_lengths(src_y_maps[y_to_h_dim_end - 1 - i]) =
                            sliced_y_to_h_lens[sliced_y_to_h_dims - 1 - i];
                    });
                }
                // TODO: add validations not across p dim

                // NOTE: this y_origin is for all dims, not only current dim
                //       will later use pick to select target dim
                constexpr auto y_origin = [&]() {
                    // can't use Encoding::Ys2RHsMajor/Ys2RHsMinor, these are unordered
                    constexpr auto y_to_h_len =
                        pick_sequence_elements_by_mask(h_len, y_to_h_masks[id]);
                    constexpr auto y_to_h_dims = y_to_h_len.size();

                    constexpr auto h_trans  = make_merge_transform_v3_division_mod(y_to_h_len);
                    auto h_origin_          = make_zero_multi_index<h_trans.NDimLow>();
                    constexpr auto y_begin_ = x_slice_begins[id] / p_len_over_h[id];
                    h_trans.calculate_lower_index(h_origin_, sequence<y_begin_.value>{});

                    auto y_origin_ = make_zero_multi_index<Encoding::NDimY>();

                    static_for<0, y_to_h_dims, 1>{}([&](auto i) {
                        y_origin_(y_to_h_dim_end - 1 - i) = h_origin_[y_to_h_dims - 1 - i];
                    });
                    return y_origin_;
                }();

                constexpr auto y_picks = typename arithmetic_sequence_gen<src_y_prefix_sum[id],
                                                                          src_y_prefix_sum[id + 1],
                                                                          1>::type{};

                set_container_subset(
                    y_slice_sorted_origins, y_picks, get_container_subset(y_origin, y_picks));
                return sliced_h_lens;
            },
            typename Encoding::HsLengthss{},
            typename arithmetic_sequence_gen<0, Encoding::HsLengthss::size(), 1>::type{});

        auto y_slice_origins = container_reorder_given_old2new(y_slice_sorted_origins, src_y_maps);

        return make_tuple(new_h_lengths, y_slice_origins, y_slice_lengths);
    }();

    constexpr auto sliced_h_lengths       = sliced_hlen_yidx_ylen[number<0>{}];
    constexpr auto sliced_y_origins_array = sliced_hlen_yidx_ylen[number<1>{}];
    constexpr auto sliced_y_origins_size  = sliced_y_origins_array.size();
    constexpr auto sliced_y_lengths_array = sliced_hlen_yidx_ylen[number<2>{}];
    constexpr auto sliced_y_lengths_size  = sliced_y_lengths_array.size();

    constexpr auto sliced_y_origins = TO_SEQUENCE(sliced_y_origins_array, sliced_y_origins_size);
    constexpr auto sliced_y_lengths = TO_SEQUENCE(sliced_y_lengths_array, sliced_y_lengths_size);

    return make_tuple(
        make_static_tile_distribution(
            tile_distribution_encoding<typename Encoding::RsLengths,
                                       remove_cvref_t<decltype(sliced_h_lengths)>, // only need to
                                                                                   // change the
                                                                                   // h_lengths type
                                       typename Encoding::Ps2RHssMajor,
                                       typename Encoding::Ps2RHssMinor,
                                       typename Encoding::Ys2RHsMajor,
                                       typename Encoding::Ys2RHsMinor>{}),
        sliced_y_origins,
        sliced_y_lengths);
}

} // namespace detail

// Free print function for tile_distribution
template <typename PsYs2XsAdaptor_,
          typename Ys2DDescriptor_,
          typename StaticTileDistributionEncoding_,
          typename TileDistributionDetail_>
CK_TILE_HOST_DEVICE void print(const tile_distribution<PsYs2XsAdaptor_,
                                                       Ys2DDescriptor_,
                                                       StaticTileDistributionEncoding_,
                                                       TileDistributionDetail_>& distribution)
{
    printf("tile_distribution{");
    printf("tile_distribution_encoding: ");
    print(StaticTileDistributionEncoding_{});
    printf(", ");
    printf("ps_ys_to_xs_: ");
    print(distribution.ps_ys_to_xs_);
    printf(", ");
    printf("ys_to_d_: ");
    print(distribution.ys_to_d_);
    printf("}\n");
}

} // namespace ck_tile
