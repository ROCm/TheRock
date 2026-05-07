// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>

#include "ck_tile/host.hpp"
#include "image_to_column.hpp"

// Host API implementation
template <>
float image_to_column(const image_to_column_traits& traits,
                      const image_to_column_args<2>& args,
                      const ck_tile::stream_config& stream_conf)
{
    if(traits.data_type.compare("fp16") == 0)
    {
        constexpr ck_tile::index_t NDimSpatial = 2;
        constexpr ck_tile::index_t VectorSize  = 8;

        using thread_tile = ck_tile::sequence<8, 8>;
        using warp_tile   = ck_tile::sequence<64, 64>;
        using block_tile  = ck_tile::sequence<128, 128>;

        using Shape = ck_tile::TileImageToColumnShape<thread_tile, warp_tile, block_tile>;

        using InDataType  = ck_tile::half_t;
        using OutDataType = ck_tile::half_t;

        using PipelineProblem = ck_tile::BlockImageToColumnProblem<InDataType,
                                                                   OutDataType,
                                                                   Shape,
                                                                   NDimSpatial,
                                                                   VectorSize,
                                                                   VectorSize>;

        using Kernel = ck_tile::ImageToColumn<PipelineProblem>;

        auto kargs = Kernel::MakeKargs(args.p_in,
                                       args.p_out,
                                       args.G,
                                       args.N,
                                       args.C,
                                       args.input_spatial_lengths,
                                       args.filter_spatial_lengths,
                                       args.output_spatial_lengths,
                                       args.image_g_n_c_wis_strides,
                                       args.gemm_g_m_k_strides,
                                       args.conv_filter_strides,
                                       args.conv_filter_dilations,
                                       args.input_left_pads,
                                       args.input_right_pads);

        const dim3 grids = Kernel::GridSize(
            args.N * args.output_spatial_lengths[0] * args.output_spatial_lengths[1],
            args.filter_spatial_lengths[0] * args.filter_spatial_lengths[1] * args.C,
            args.G);
        const dim3 blocks = Kernel::BlockSize();

        constexpr ck_tile::index_t kBlockPerCu = 2;

        float ave_time = ck_tile::launch_kernel(
            stream_conf, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    constexpr ck_tile::index_t NDimSpatial = 2;

    ExecutionConfig config;
    ck_tile::conv::ConvParam conv_params = DefaultConvParams;

    if(!parse_cmd_args(argc, argv, config, conv_params))
    {
        return EXIT_FAILURE;
    }

    if(conv_params.num_dim_spatial_ != NDimSpatial)
    {
        std::cerr << "unsupported # of spatial dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    using InDataType  = ck_tile::half_t;
    using OutDataType = ck_tile::half_t;
    using ImLayout    = ck_tile::tensor_layout::convolution::NHWGC;

    const auto G = conv_params.G_;
    const auto N = conv_params.N_;
    const auto C = conv_params.C_;

    const ck_tile::long_index_t NHoWo =
        N * std::accumulate(conv_params.output_spatial_lengths_.begin(),
                            std::next(conv_params.output_spatial_lengths_.begin(), NDimSpatial),
                            1,
                            std::multiplies<>());

    const ck_tile::long_index_t CYX =
        C * std::accumulate(conv_params.filter_spatial_lengths_.begin(),
                            std::next(conv_params.filter_spatial_lengths_.begin(), NDimSpatial),
                            1,
                            std::multiplies<>());

    const auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ImLayout>(conv_params);
    const auto out_desc = ck_tile::HostTensorDescriptor({G, NHoWo, CYX});

    // host verify
    ck_tile::HostTensor<InDataType> in(in_desc);
    ck_tile::HostTensor<OutDataType> out_device(out_desc);
    ck_tile::HostTensor<OutDataType> out_host(out_desc);

    switch(config.init_method)
    {
    case 0: break;
    case 1: ck_tile::FillUniformDistributionIntegerValue<InDataType>{-5.f, 5.f}(in); break;
    default: ck_tile::FillUniformDistribution<InDataType>{-0.5, 0.5}(in); break;
    }

    ck_tile::DeviceMem in_device_buf(in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_device_buf(out_device.get_element_space_size_in_bytes());

    in_device_buf.ToDevice(in.data());

    image_to_column_traits traits{"fp16"};

    image_to_column_args<NDimSpatial> args{
        in_device_buf.GetDeviceBuffer(),
        out_device_buf.GetDeviceBuffer(),
        G,
        N,
        C,
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_spatial_lengths_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.filter_spatial_lengths_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.output_spatial_lengths_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial + 3>(in_desc.get_strides()),
        ck_tile::to_array<ck_tile::long_index_t, 3>(out_desc.get_strides()),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.conv_filter_strides_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.conv_filter_dilations_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_left_pads_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_right_pads_)};

    float ave_time =
        image_to_column(traits, args, ck_tile::stream_config{nullptr, config.time_kernel});

    if(config.time_kernel)
    {
        std::size_t num_btype = G * NHoWo * CYX * (sizeof(OutDataType) + sizeof(InDataType));
        float gb_per_sec      = num_btype / 1.E6 / ave_time;
        std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    }
    else
    {
        std::cout << "image_to_column: pass, No Perf generated due to config.time_kernel=0"
                  << std::endl;
    }

    bool pass = true;

    if(config.do_verification)
    {
        // reference
        ck_tile::reference_im2col<InDataType, OutDataType, NDimSpatial>(in, out_host, conv_params);

        out_device_buf.FromDevice(out_device.data());
        pass = ck_tile::check_err(out_device, out_host);

        std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    }

    return !pass;
}
