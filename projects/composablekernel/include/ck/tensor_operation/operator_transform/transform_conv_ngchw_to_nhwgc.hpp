// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"

namespace ck {
namespace tensor_operation {

/*
 * Transform Convolution NGCHW to NHWGC. We transform [N, G, C, H, W] tensor
 * descriptor to [N * G * C, H * W] (input or output image). The first
 * dimension is store dimension, the second one is load dimension. For
 * NHWGC to NGCHW load and store are reverted. For weight we transform
 * [G, K, C, Y, X] to [G * K * Y * X, C]. First dim is load dimension,
 * second dim is store dimension.
 */

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          index_t NDimSpatial,
          index_t MPerThread,
          index_t NPerThread>
struct TransformConvNGCHWToNHWGC
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto
    MakeNGCHWTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_strides,
                           const index_t split_n_size = 1)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t N   = g_n_c_wis_lengths[I1] / split_n_size;
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Wi = g_n_c_wis_lengths[I3];

        const index_t& GStride  = g_n_c_wis_strides[I0];
        const index_t& NStride  = g_n_c_wis_strides[I1];
        const index_t& CStride  = g_n_c_wis_strides[I2];
        const index_t& WiStride = g_n_c_wis_strides[I3];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Wi), make_tuple(NStride, GStride, CStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto
    MakeNHWGCTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_strides,
                           const index_t split_n_size = 1)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t N   = g_n_c_wis_lengths[I1] / split_n_size;
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Wi = g_n_c_wis_lengths[I3];

        const index_t& NStride = g_n_c_wis_strides[I1];
        const index_t WiStride = G * C;
        const index_t GStride  = C;
        const index_t CStride  = 1;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Wi), make_tuple(NStride, GStride, CStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto
    MakeNGCHWTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_strides,
                           const index_t split_n_size = 1)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t N   = g_n_c_wis_lengths[I1] / split_n_size;
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Hi = g_n_c_wis_lengths[I3];
        const index_t& Wi = g_n_c_wis_lengths[I4];

        const index_t& GStride  = g_n_c_wis_strides[I0];
        const index_t& NStride  = g_n_c_wis_strides[I1];
        const index_t& CStride  = g_n_c_wis_strides[I2];
        const index_t& HiStride = g_n_c_wis_strides[I3];
        const index_t& WiStride = g_n_c_wis_strides[I4];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Hi, Wi), make_tuple(NStride, GStride, CStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto
    MakeNHWGCTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_strides,
                           const index_t split_n_size = 1)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t N   = g_n_c_wis_lengths[I1] / split_n_size;
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Hi = g_n_c_wis_lengths[I3];
        const index_t& Wi = g_n_c_wis_lengths[I4];

        const index_t& NStride = g_n_c_wis_strides[I1];
        const index_t HiStride = Wi * G * C;
        const index_t WiStride = G * C;
        const index_t GStride  = C;
        const index_t CStride  = 1;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Hi, Wi), make_tuple(NStride, GStride, CStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto
    MakeNGCHWTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_strides,
                           const index_t split_n_size = 1)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t N   = g_n_c_wis_lengths[I1] / split_n_size;
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Di = g_n_c_wis_lengths[I3];
        const index_t& Hi = g_n_c_wis_lengths[I4];
        const index_t& Wi = g_n_c_wis_lengths[I5];

        const index_t& GStride  = g_n_c_wis_strides[I0];
        const index_t& NStride  = g_n_c_wis_strides[I1];
        const index_t& CStride  = g_n_c_wis_strides[I2];
        const index_t& DiStride = g_n_c_wis_strides[I3];
        const index_t& HiStride = g_n_c_wis_strides[I4];
        const index_t& WiStride = g_n_c_wis_strides[I5];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Di, Hi, Wi),
            make_tuple(NStride, GStride, CStride, DiStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Di, Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto
    MakeNHWGCTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_n_c_wis_strides,
                           const index_t split_n_size = 1)
    {
        const index_t& G  = g_n_c_wis_lengths[I0];
        const index_t N   = g_n_c_wis_lengths[I1] / split_n_size;
        const index_t& C  = g_n_c_wis_lengths[I2];
        const index_t& Di = g_n_c_wis_lengths[I3];
        const index_t& Hi = g_n_c_wis_lengths[I4];
        const index_t& Wi = g_n_c_wis_lengths[I5];

        const index_t& NStride = g_n_c_wis_strides[I1];
        const index_t DiStride = Hi * Wi * G * C;
        const index_t HiStride = Wi * G * C;
        const index_t WiStride = G * C;
        const index_t GStride  = C;
        const index_t CStride  = 1;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(N, G, C, Di, Hi, Wi),
            make_tuple(NStride, GStride, CStride, DiStride, HiStride, WiStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(N, G, C)),
                                                   make_merge_transform(make_tuple(Di, Hi, Wi))),
                                        make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto
    MakeGKCYXTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        const index_t& G = g_k_c_wis_lengths[I0];
        const index_t& K = g_k_c_wis_lengths[I1];
        const index_t& C = g_k_c_wis_lengths[I2];
        const index_t& X = g_k_c_wis_lengths[I3];

        const index_t& GStride = g_k_c_wis_strides[I0];
        const index_t& KStride = g_k_c_wis_strides[I1];
        const index_t& CStride = g_k_c_wis_strides[I2];
        const index_t& XStride = g_k_c_wis_strides[I3];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(G, K, C, X), make_tuple(GStride, KStride, CStride, XStride));
        const auto merged_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(make_tuple(G, K, X)), make_pass_through_transform(C)),
            make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto
    MakeGKYXCTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        const index_t& G = g_k_c_wis_lengths[I0];
        const index_t& K = g_k_c_wis_lengths[I1];
        const index_t& C = g_k_c_wis_lengths[I2];
        const index_t& X = g_k_c_wis_lengths[I3];

        const index_t& GStride = g_k_c_wis_strides[I0];
        const index_t KStride  = g_k_c_wis_strides[I1];
        const index_t CStride  = 1;
        const index_t XStride  = C;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(G, K, C, X), make_tuple(GStride, KStride, CStride, XStride));
        const auto merged_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(make_tuple(G, K, X)), make_pass_through_transform(C)),
            make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto
    MakeGKCYXTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        const index_t& G = g_k_c_wis_lengths[I0];
        const index_t& K = g_k_c_wis_lengths[I1];
        const index_t& C = g_k_c_wis_lengths[I2];
        const index_t& Y = g_k_c_wis_lengths[I3];
        const index_t& X = g_k_c_wis_lengths[I4];

        const index_t& GStride = g_k_c_wis_strides[I0];
        const index_t& KStride = g_k_c_wis_strides[I1];
        const index_t& CStride = g_k_c_wis_strides[I2];
        const index_t& YStride = g_k_c_wis_strides[I3];
        const index_t& XStride = g_k_c_wis_strides[I4];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(G, K, C, Y, X), make_tuple(GStride, KStride, CStride, YStride, XStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(G, K, Y, X)),
                                                   make_pass_through_transform(C)),
                                        make_tuple(Sequence<0, 1, 3, 4>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto
    MakeGKYXCTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        const index_t& G = g_k_c_wis_lengths[I0];
        const index_t& K = g_k_c_wis_lengths[I1];
        const index_t& C = g_k_c_wis_lengths[I2];
        const index_t& Y = g_k_c_wis_lengths[I3];
        const index_t& X = g_k_c_wis_lengths[I4];

        const index_t& GStride = g_k_c_wis_strides[I0];
        const index_t KStride  = g_k_c_wis_strides[I1];
        const index_t CStride  = 1;
        const index_t YStride  = X * C;
        const index_t XStride  = C;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(G, K, C, Y, X), make_tuple(GStride, KStride, CStride, YStride, XStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(G, K, Y, X)),
                                                   make_pass_through_transform(C)),
                                        make_tuple(Sequence<0, 1, 3, 4>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto
    MakeGKCYXTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        const index_t& G = g_k_c_wis_lengths[I0];
        const index_t& K = g_k_c_wis_lengths[I1];
        const index_t& C = g_k_c_wis_lengths[I2];
        const index_t& Z = g_k_c_wis_lengths[I3];
        const index_t& Y = g_k_c_wis_lengths[I4];
        const index_t& X = g_k_c_wis_lengths[I5];

        const index_t& GStride = g_k_c_wis_strides[I0];
        const index_t& KStride = g_k_c_wis_strides[I1];
        const index_t& CStride = g_k_c_wis_strides[I2];
        const index_t& ZStride = g_k_c_wis_strides[I3];
        const index_t& YStride = g_k_c_wis_strides[I4];
        const index_t& XStride = g_k_c_wis_strides[I5];

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(G, K, C, Z, Y, X),
            make_tuple(GStride, KStride, CStride, ZStride, YStride, XStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(G, K, Z, Y, X)),
                                                   make_pass_through_transform(C)),
                                        make_tuple(Sequence<0, 1, 3, 4, 5>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto
    MakeGKYXCTransposeDesc(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                           const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        const index_t& G = g_k_c_wis_lengths[I0];
        const index_t& K = g_k_c_wis_lengths[I1];
        const index_t& C = g_k_c_wis_lengths[I2];
        const index_t& Z = g_k_c_wis_lengths[I3];
        const index_t& Y = g_k_c_wis_lengths[I4];
        const index_t& X = g_k_c_wis_lengths[I5];

        const index_t& GStride = g_k_c_wis_strides[I0];
        const index_t KStride  = g_k_c_wis_strides[I1];
        const index_t CStride  = 1;
        const index_t ZStride  = Y * X * C;
        const index_t YStride  = X * C;
        const index_t XStride  = C;

        const auto desc = make_naive_tensor_descriptor(
            make_tuple(G, K, C, Z, Y, X),
            make_tuple(GStride, KStride, CStride, ZStride, YStride, XStride));
        const auto merged_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(make_tuple(G, K, Z, Y, X)),
                                                   make_pass_through_transform(C)),
                                        make_tuple(Sequence<0, 1, 3, 4, 5>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        return device::PadTensorDescriptor(
            merged_desc, make_tuple(MPerThread, NPerThread), Sequence<true, true>{});
    }

    static auto TransposeInOutStrides(const std::array<index_t, NDimSpatial + 3>& g_n_c_wis_lengths,
                                      const std::array<index_t, NDimSpatial + 3>& g_n_c_wis_strides)
    {
        if constexpr(device::is_NGCHW_NGKHW<ALayout, BLayout, ELayout>() ||
                     device::is_NGCDHW_NGKDHW<ALayout, BLayout, ELayout>())
        {
            std::array<index_t, NDimSpatial + 3> g_n_c_wis_strides_transposed;
            const auto G = g_n_c_wis_lengths[I0];
            const auto C = g_n_c_wis_lengths[I2];

            g_n_c_wis_strides_transposed[I0] = C;
            g_n_c_wis_strides_transposed[I1] = g_n_c_wis_strides[I1];
            g_n_c_wis_strides_transposed[I2] = I1;
            if constexpr(NDimSpatial == 2)
            {
                g_n_c_wis_strides_transposed[I3] = g_n_c_wis_lengths[I4] * G * C;
                g_n_c_wis_strides_transposed[I4] = G * C;
            }
            else if constexpr(NDimSpatial == 3)
            {
                g_n_c_wis_strides_transposed[I3] =
                    g_n_c_wis_lengths[I4] * g_n_c_wis_lengths[I5] * G * C;
                g_n_c_wis_strides_transposed[I4] = g_n_c_wis_lengths[I5] * G * C;
                g_n_c_wis_strides_transposed[I5] = G * C;
            }
            return g_n_c_wis_strides_transposed;
        }
        else
        {
            // transpose not needed
            return g_n_c_wis_strides;
        }
    }

    static auto
    TransposeWeiStrides(const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_lengths,
                        const std::array<ck::index_t, NDimSpatial + 3>& g_k_c_wis_strides)
    {
        if constexpr(device::is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>() ||
                     device::is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>())
        {
            std::array<index_t, NDimSpatial + 3> g_k_c_wis_strides_transposed = g_k_c_wis_strides;
            const index_t C = g_k_c_wis_lengths[I2];

            if constexpr(NDimSpatial == 2)
            {
                const index_t X                  = g_k_c_wis_lengths[I4];
                g_k_c_wis_strides_transposed[I2] = 1;
                g_k_c_wis_strides_transposed[I3] = X * C;
                g_k_c_wis_strides_transposed[I4] = C;
            }
            else if constexpr(NDimSpatial == 3)
            {
                const index_t Y                  = g_k_c_wis_lengths[I4];
                const index_t X                  = g_k_c_wis_lengths[I5];
                g_k_c_wis_strides_transposed[I2] = 1;
                g_k_c_wis_strides_transposed[I3] = Y * X * C;
                g_k_c_wis_strides_transposed[I4] = X * C;
                g_k_c_wis_strides_transposed[I5] = C;
            }
            return g_k_c_wis_strides_transposed;
        }
        else
        {
            // transpose not needed
            return g_k_c_wis_strides;
        }
    }
};

} // namespace tensor_operation
} // namespace ck
