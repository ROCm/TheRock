// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// 1d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NWGC_GKXC_NWGK()
{
    return is_same_v<InLayout, tensor_layout::convolution::NWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNWC_GKXC_GNWK()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNWK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCW_GKXC_NGKW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKW>;
}

// 2d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NHWGC_GKYXC_NHWGK()
{
    return is_same_v<InLayout, tensor_layout::convolution::NHWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NHWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNHWC_GKYXC_GNHWK()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNHWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNHWK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCHW_GKYXC_NGKHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCHW_GKCYX_NGKHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKCYX> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCHW_NGKHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKHW>;
}

// 3d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NDHWGC_GKZYXC_NDHWGK()
{
    return is_same_v<InLayout, tensor_layout::convolution::NDHWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NDHWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNDHWC_GKZYXC_GNDHWK()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNDHWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNDHWK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCDHW_GKZYXC_NGKDHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCDHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKDHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCDHW_GKCZYX_NGKDHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCDHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKCZYX> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKDHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCDHW_NGKDHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCDHW> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKDHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NSpatialGC_GKSpatial_NSpatialGK()
{
    return is_NWGC_GKXC_NWGK<InLayout, WeiLayout, OutLayout>() ||
           is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
           is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>();
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNSpatialC_GKSpatial_GNSpatialK()
{
    return is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>() ||
           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>() ||
           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>();
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCSpatial_GKSpatial_NGKSpatial()
{
    return is_NGCW_GKXC_NGKW<InLayout, WeiLayout, OutLayout>() ||
           is_NGCHW_GKYXC_NGKHW<InLayout, WeiLayout, OutLayout>() ||
           is_NGCDHW_GKZYXC_NGKDHW<InLayout, WeiLayout, OutLayout>();
}

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0, typename = void>
struct ComputePtrOffsetOfStridedBatch
{
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      enable_if_t<(NumATensor > 1 || NumBTensor > 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(Array<long_index_t, NumATensor>& BatchStrideAs,
                                   Array<long_index_t, NumBTensor>& BatchStrideBs,
                                   Array<long_index_t, NumDTensor>& BatchStrideDs,
                                   long_index_t BatchStrideE)
        : BatchStrideA_(BatchStrideAs),
          BatchStrideB_(BatchStrideBs),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr auto GetAsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumATensor> as_offset;
        static_for<0, NumATensor, 1>{}(
            [&](auto i) { as_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideA_[i]; });
        return as_offset;
    }

    __host__ __device__ constexpr auto GetBsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumBTensor> bs_offset;
        static_for<0, NumBTensor, 1>{}(
            [&](auto i) { bs_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideB_[i]; });
        return bs_offset;
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideDs_[i]; });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    Array<long_index_t, NumATensor> BatchStrideA_{};
    Array<long_index_t, NumBTensor> BatchStrideB_{};
    Array<long_index_t, NumDTensor> BatchStrideDs_{};
    long_index_t BatchStrideE_{};
    long_index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      enable_if_t<(NumATensor == 1 && NumBTensor == 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(long_index_t BatchStrideA,
                                   long_index_t BatchStrideB,
                                   Array<long_index_t, NumDTensor> BatchStrideDs,
                                   long_index_t BatchStrideE)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideA_;
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideB_;
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideDs_[i]; });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    long_index_t BatchStrideA_{};
    long_index_t BatchStrideB_{};
    Array<long_index_t, NumDTensor> BatchStrideDs_{};
    long_index_t BatchStrideE_{};
    long_index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <bool isTuple, typename Tensors>
constexpr static auto GetNumABTensors()
{
    if constexpr(isTuple)
    {
        return Number<Tensors::Size()>{};
    }
    else
    {
        return Number<1>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetAGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::AsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetBGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::BsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename Id, typename Type>
constexpr static auto UnpackDataType()
{
    if constexpr(isTuple)
    {
        // unpack if tuple
        return tuple_element_t<Id{}, Type>{};
    }
    else
    {
        // if no, return Type
        return Type{};
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
