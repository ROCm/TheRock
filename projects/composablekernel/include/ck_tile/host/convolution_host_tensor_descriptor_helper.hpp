// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {
namespace conv {
namespace detail {

template <typename OldLayout>
CK_TILE_HOST std::vector<std::size_t> get_layout_transpose_gnchw_to_old()
{
    using namespace ck_tile::tensor_layout::convolution;

    if constexpr(is_any_of<OldLayout, GNCW, GKCX, GNKW>::value)
    {
        return {0, 1, 2, 3};
    }
    else if constexpr(is_any_of<OldLayout, GNCHW, GKCYX, GNKHW>::value)
    {
        return {0, 1, 2, 3, 4};
    }
    else if constexpr(is_any_of<OldLayout, GNCDHW, GKCZYX, GNKDHW>::value)
    {
        return {0, 1, 2, 3, 4, 5};
    }
    if constexpr(is_any_of<OldLayout, GNWC, GKXC, GNWK>::value)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(is_any_of<OldLayout, GNHWC, GKYXC, GNHWK>::value)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(is_any_of<OldLayout, GNDHWC, GKZYXC, GNDHWK>::value)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    else if constexpr(is_any_of<OldLayout, NWGC, KXGC, NWGK>::value)
    {
        return {2, 0, 3, 1};
    }
    else if constexpr(is_any_of<OldLayout, NHWGC, KYXGC, NHWGK>::value)
    {
        return {3, 0, 4, 1, 2};
    }
    else if constexpr(is_any_of<OldLayout, NDHWGC, KZYXGC, NDHWGK>::value)
    {
        return {4, 0, 5, 1, 2, 3};
    }
    else
    {
        printf("%s\n", __func__);
        throw std::runtime_error("wrong! unsupported layout");
    }
}

} // namespace detail

// make tensor descriptor for packed input tensor, and order the dimension in the order of GNCHW
// regardless of physical layout
template <typename InLayout>
CK_TILE_HOST HostTensorDescriptor
make_input_host_tensor_descriptor_g_n_c_wis_packed(const ck_tile::conv::ConvParam& param)
{
    using namespace ck_tile::tensor_layout::convolution;

    std::vector<std::size_t> physical_lengths;

    if constexpr(is_any_of<InLayout, GNCW, GNCHW, GNCDHW>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(is_any_of<InLayout, GNWC, GNHWC, GNDHWC>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(is_any_of<InLayout, NWGC, NHWGC, NDHWGC>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", InLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<InLayout>());
}

// make tensor descriptor for packed weight tensor, and order the dimension in the order of GKCYX
// regardless of physical layout
template <typename WeiLayout>
CK_TILE_HOST HostTensorDescriptor
make_weight_host_tensor_descriptor_g_k_c_xs_packed(const ck_tile::conv::ConvParam& param)
{
    using namespace ck_tile::tensor_layout::convolution;

    std::vector<std::size_t> physical_lengths;

    if constexpr(is_any_of<WeiLayout, KXC, KYXC, KZYXC>::value)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(is_any_of<WeiLayout, GKCX, GKCYX, GKCZYX>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(is_any_of<WeiLayout, GKXC, GKYXC, GKZYXC>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(is_any_of<WeiLayout, KXGC, KYXGC, KZYXGC>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", WeiLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<WeiLayout>());
}

// make tensor descriptor for packed output tensor, and order the dimension in the order of GNKHW
// regardless of physical layout
template <typename OutLayout>
CK_TILE_HOST HostTensorDescriptor
make_output_host_tensor_descriptor_g_n_k_wos_packed(const ck_tile::conv::ConvParam& param)
{
    using namespace ck_tile::tensor_layout::convolution;

    std::vector<std::size_t> physical_lengths;

    if constexpr(is_any_of<OutLayout, GNKW, GNKHW, GNKDHW>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(is_any_of<OutLayout, GNWK, GNHWK, GNDHWK>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(is_any_of<OutLayout, NWGK, NHWGK, NDHWGK>::value)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", OutLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<OutLayout>());
}

} // namespace conv
} // namespace ck_tile
