// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include <cstring>

// Parse command-line arguments for 3D pooling example
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("N", "2", "N dimension")
        .insert("H", "30", "H dimension")
        .insert("W", "30", "W dimension")
        .insert("C", "32", "C dimension")
        .insert("D", "30", "D dimension")
        .insert("Z", "2", "Z dimension")
        .insert("Y", "2", "Y dimension")
        .insert("X", "2", "X dimension")
        .insert("Sz", "2", "window stride d")
        .insert("Sy", "2", "window stride h")
        .insert("Sx", "2", "window stride w")
        .insert("Dz", "1", "window dilation d")
        .insert("Dy", "1", "window dilation h")
        .insert("Dx", "1", "window dilation w")
        .insert("LeftPz", "1", "left padding d")
        .insert("LeftPy", "1", "left padding h")
        .insert("LeftPx", "1", "left padding w")
        .insert("RightPz", "1", "right padding d")
        .insert("RightPy", "1", "right padding h")
        .insert("RightPx", "1", "right padding w")
        .insert("v", "1", "cpu validation or not")
        .insert("warmup", "20", "cold iter")
        .insert("repeat", "100", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename IndexDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{

    const ck_tile::index_t N = arg_parser.get_int("N");
    const ck_tile::index_t H = arg_parser.get_int("H");
    const ck_tile::index_t W = arg_parser.get_int("W");
    const ck_tile::index_t C = arg_parser.get_int("C");
    const ck_tile::index_t D = arg_parser.get_int("D");

    const ck_tile::index_t Z = arg_parser.get_int("Z");
    const ck_tile::index_t Y = arg_parser.get_int("Y");
    const ck_tile::index_t X = arg_parser.get_int("X");

    const ck_tile::index_t Sz = arg_parser.get_int("Sz");
    const ck_tile::index_t Sy = arg_parser.get_int("Sy");
    const ck_tile::index_t Sx = arg_parser.get_int("Sx");

    const ck_tile::index_t Dz = arg_parser.get_int("Dz");
    const ck_tile::index_t Dy = arg_parser.get_int("Dy");
    const ck_tile::index_t Dx = arg_parser.get_int("Dx");

    const ck_tile::index_t LeftPz  = arg_parser.get_int("LeftPz");
    const ck_tile::index_t LeftPy  = arg_parser.get_int("LeftPy");
    const ck_tile::index_t LeftPx  = arg_parser.get_int("LeftPx");
    const ck_tile::index_t RightPz = arg_parser.get_int("RightPz");
    const ck_tile::index_t RightPy = arg_parser.get_int("RightPy");
    const ck_tile::index_t RightPx = arg_parser.get_int("RightPx");

    const ck_tile::index_t Zs = (Z - 1) * Dz + 1;
    const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
    const ck_tile::index_t Xs = (X - 1) * Dx + 1;

    const ck_tile::index_t Do = (D + LeftPz + RightPz - Zs) / Sz + 1;
    const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
    const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

    printf("Input parameters:\n");
    printf("N: %d, D: %d, H: %d, W: %d, C: %d\n", N, D, H, W, C);
    printf("Window Z: %d, Y: %d, X: %d, Stride Z: %d, Y: %d, X: %d\n", Z, Y, X, Sz, Sy, Sx);
    printf("Output Do: %d, Ho: %d, Wo: %d\n", Do, Ho, Wo);

    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    constexpr bool OutputIndex  = true;
    constexpr bool PropagateNan = false;

    // Shapes / strides / parameters (NDHWC)
    const auto input_shape    = ck_tile::make_tuple(N, D, H, W, C);
    const auto output_shape   = ck_tile::make_tuple(N, Do, Ho, Wo, C);
    const auto input_strides  = ck_tile::make_tuple(D * H * W * C, H * W * C, W * C, C, 1);
    const auto output_strides = ck_tile::make_tuple(Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1);
    const auto window_spatial_lengths = ck_tile::make_tuple(Z, Y, X);
    const auto window_strides         = ck_tile::make_tuple(Sz, Sy, Sx);
    const auto window_dilations       = ck_tile::make_tuple(Dz, Dy, Dx);
    const auto input_left_pads        = ck_tile::make_tuple(LeftPz, LeftPy, LeftPx);
    const auto input_right_pads       = ck_tile::make_tuple(RightPz, RightPy, RightPx);

    ck_tile::HostTensor<InDataType> in({N, D, H, W, C}, {D * H * W * C, H * W * C, W * C, C, 1});
    ck_tile::HostTensor<OutDataType> out({N, Do, Ho, Wo, C},
                                         {Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1});
    ck_tile::HostTensor<OutDataType> out_ref({N, Do, Ho, Wo, C},
                                             {Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1});
    ck_tile::HostTensor<IndexDataType> out_index({N, Do, Ho, Wo, C},
                                                 {Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1});
    ck_tile::HostTensor<IndexDataType> out_ref_index({N, Do, Ho, Wo, C},
                                                     {Do * Ho * Wo * C, Ho * Wo * C, Wo * C, C, 1});

    ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(in);

    ck_tile::DeviceMem in_buf(in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_buf(out.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_index_buf(OutputIndex ? out_index.get_element_space_size_in_bytes() : 0);

    in_buf.ToDevice(in.data());

    using ReduceOp   = ck_tile::ReduceOp::Max;
    using BlockWarps = ck_tile::sequence<1, 1>;
    using BlockTile  = ck_tile::sequence<128, 1>;
    using WarpTile   = ck_tile::sequence<128, 1>;
    using ThreadTile = ck_tile::sequence<2, 1>;

    using Shape   = ck_tile::PoolShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;
    using Problem = ck_tile::PoolProblem<InDataType,
                                         OutDataType,
                                         ComputeDataType,
                                         IndexDataType,
                                         ReduceOp,
                                         OutputIndex,
                                         PropagateNan,
                                         Shape>;
    using Kernel  = ck_tile::PoolKernel<Problem>;

    constexpr ck_tile::index_t kBlockPerCu = 1;
    const ck_tile::index_t kBlockSize      = Kernel::BlockSize();

    auto host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_spatial_lengths)>{
        static_cast<InDataType*>(in_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_buf.GetDeviceBuffer()),
        OutputIndex ? static_cast<IndexDataType*>(out_index_buf.GetDeviceBuffer()) : nullptr,
        input_shape,
        output_shape,
        input_strides,
        output_strides,
        window_spatial_lengths,
        window_strides,
        window_dilations,
        input_left_pads,
        input_right_pads};

    auto kernel_args = Kernel::MakeKernelArgs(host_args);

    const ck_tile::index_t kGridSize = Kernel::CalculateGridSize(kernel_args);
    std::cout << "grid size " << kGridSize << std::endl;

    // Validate kernel can handle the given configuration
    if(!Kernel::IsSupportedArgument(kernel_args))
    {
        throw std::runtime_error("ERROR: Kernel arguments are not supported! \n");
    }

    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, kernel_args));

    std::size_t num_btype =
        sizeof(InDataType) * N * D * H * W * C + sizeof(OutDataType) * N * Do * Ho * Wo * C;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        out_buf.FromDevice(out.mData.data());

        ck_tile::reference_pool3d<InDataType,
                                  ComputeDataType,
                                  OutDataType,
                                  IndexDataType,
                                  ReduceOp,
                                  decltype(input_shape),
                                  decltype(window_spatial_lengths),
                                  OutputIndex>(in, out_ref, out_ref_index, kernel_args, ReduceOp{});

        if constexpr(OutputIndex)
        {
            out_index_buf.FromDevice(out_index.mData.data());
            pass = ck_tile::check_err(out, out_ref) && ck_tile::check_err(out_index, out_ref_index);
        }
        else
        {
            pass = ck_tile::check_err(out, out_ref);
        }

        std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    return run<ck_tile::half_t, ck_tile::half_t, float, ck_tile::index_t>(arg_parser) ? 0 : -2;
}
