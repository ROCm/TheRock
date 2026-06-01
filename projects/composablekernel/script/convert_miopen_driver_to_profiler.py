# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Convert miopen driver command to ck Profiler
# Example: python3 ../script/convert_miopen_driver_to_profiler.py
# /opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 -k 64 -y 3 -x 3
# -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32 -F 1 -t 1

import argparse
import subprocess


def init_const_args(args):
    args.ck_profiler_cmd = "../build/bin/ckProfiler"
    # use decimal values
    args.init_method = 2
    # don't print tensor values
    args.log_value = 0


def run_ck_profiler_cmd(cmd):
    print("ckProfiler command:")
    cmd_concatenated_str = ""
    for arg in cmd:
        cmd_concatenated_str += arg + " "
    print(cmd_concatenated_str)
    subprocess.run(cmd)


def parse_layouts(args):
    if args.in_layout == "NCW" or args.in_layout == "NCHW" or args.in_layout == "NCDHW":
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.layout = 4
        elif (
            args.ck_profier_op == "grouped_conv_fwd"
            or args.ck_profier_op == "grouped_conv_bwd_data"
        ):
            args.layout = 3
        else:
            print("Not supported layout for this op")
            exit(1)
    elif (
        args.in_layout == "NWC" or args.in_layout == "NHWC" or args.in_layout == "NDHWC"
    ):
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.layout = 2
        elif (
            args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.layout = 1
    else:
        print("Not supported layout for this op")
        exit(1)


def parse_data_type(args):
    if args.data_type == "fp32":
        if (
            args.ck_profier_op == "grouped_conv_bwd_weight"
            or args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.data_type = 0
    if args.data_type == "fp16":
        if (
            args.ck_profier_op == "grouped_conv_bwd_weight"
            or args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.data_type = 1
    if args.data_type == "int8":
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.data_type = 4
        if args.ck_profier_op == "grouped_conv_bwd_data":
            print("Not supported data type for grouped_conv_bwd_data")
            exit(1)
        if args.ck_profier_op == "grouped_conv_fwd":
            args.data_type = 3
    if args.data_type == "bfp16":
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.data_type = 5
        if (
            args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.data_type = 2


def add_conv_params_to_cmd(args, cmd):
    if args.spatial_dim == 1:
        cmd += [str(args.fil_w), str(args.in_w)]
        cmd += [str(args.conv_stride_w), str(args.dilation_w)]
        cmd += [str(args.pad_w), str(args.pad_w)]
    elif args.spatial_dim == 2:
        cmd += [str(args.fil_h), str(args.fil_w)]
        cmd += [str(args.in_h), str(args.in_w)]
        cmd += [str(args.conv_stride_h), str(args.conv_stride_w)]
        cmd += [str(args.dilation_h), str(args.dilation_w)]
        cmd += [str(args.pad_h), str(args.pad_w)]
        cmd += [str(args.pad_h), str(args.pad_w)]
    elif args.spatial_dim == 3:
        cmd += [str(args.fil_d), str(args.fil_h), str(args.fil_w)]
        cmd += [str(args.in_d), str(args.in_h), str(args.in_w)]
        cmd += [str(args.conv_stride_d), str(args.conv_stride_h)]
        cmd += [str(args.conv_stride_w)]
        cmd += [str(args.dilation_d), str(args.dilation_h), str(args.dilation_w)]
        cmd += [str(args.pad_d), str(args.pad_h), str(args.pad_w)]
        cmd += [str(args.pad_d), str(args.pad_h), str(args.pad_w)]
    else:
        print("Not supported spatial dim (supported: 1, 2, 3)")
        exit(1)


def run_ck_grouped_conv_fwd(args):
    args.ck_profier_op = "grouped_conv_fwd"
    parse_data_type(args)
    parse_layouts(args)
    # use int32 by default
    args.index_type = 0

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout), str(args.index_type)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    run_ck_profiler_cmd(cmd)


def run_ck_grouped_conv_bwd_data(args):
    args.ck_profier_op = "grouped_conv_bwd_data"
    parse_data_type(args)
    parse_layouts(args)
    # Test all split K value from the list {1, 2, 4, 8, 32, 64, 128}
    args.split_k_value = -1

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    cmd += [str(args.split_k_value)]
    run_ck_profiler_cmd(cmd)


def run_ck_grouped_conv_bwd_weight(args):
    args.ck_profier_op = "grouped_conv_bwd_weight"
    parse_data_type(args)
    parse_layouts(args)
    # Test all split K value from the list {1, 2, 4, 8, 32, 64, 128}
    args.split_k_value = "all"

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    cmd += [str(args.split_k_value)]
    run_ck_profiler_cmd(cmd)


# Get name of miopen driver, remove it from unknown
def process_miopen_driver_name(args, unknown):
    if "convint8" in unknown:
        args.data_type = "int8"
        unknown.remove("convint8")
    elif "convbfp16" in unknown:
        args.data_type = "bfp16"
        unknown.remove("convbfp16")
    elif "convfp16" in unknown:
        args.data_type = "fp16"
        unknown.remove("convfp16")
    elif "conv" in unknown:
        args.data_type = "fp32"
        unknown.remove("conv")
    else:
        print("Not supported driver (supported: conv, convfp16, convint8, convbfp16).")
        exit(1)


def run_ck_profiler(args):
    # MIOpen get number of channel per all groups, CK profiler get number of
    # channel per group
    args.in_channels = int(args.in_channels / args.group_count)
    args.out_channels = int(args.out_channels / args.group_count)

    if args.forw == 0 or args.forw == 1 or args.forw == 3 or args.forw == 5:
        run_ck_grouped_conv_fwd(args)
    if args.forw == 0 or args.forw == 2 or args.forw == 3 or args.forw == 6:
        run_ck_grouped_conv_bwd_data(args)
    if args.forw == 0 or args.forw == 4 or args.forw == 5 or args.forw == 6:
        run_ck_grouped_conv_bwd_weight(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="converter",
        description="Convert miopen driver command to ck Profiler"
        "\nExample: python3 "
        "../script/convert_miopen_driver_to_profiler.py "
        "/opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 "
        "-k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g "
        "32 -F 1 -t 1",
    )
    parser.add_argument(
        "-in_layout",
        "-I",
        "--in_layout",
        "--I",
        default="NCHW",
        type=str,
        required=False,
        help="Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
    )
    parser.add_argument(
        "-forw",
        "-F",
        "--forw",
        "--F",
        default=0,
        type=int,
        required=False,
        help="Flag enables fwd, bwd, wrw convolutions"
        "\n0 fwd+bwd+wrw (default)"
        "\n1 fwd only"
        "\n2 bwd only"
        "\n4 wrw only"
        "\n3 fwd+bwd"
        "\n5 fwd+wrw"
        "\n6 bwd+wrw",
    )
    parser.add_argument(
        "-spatial_dim",
        "-_",
        "--spatial_dim",
        "--_",
        default=2,
        type=int,
        required=False,
        help="convolution spatial dimension (Default-2)",
    )
    parser.add_argument(
        "-batchsize",
        "-n",
        "--batchsize",
        "--n",
        default=100,
        type=int,
        required=False,
        help="Mini-batch size (Default=100)",
    )
    parser.add_argument(
        "-in_channels",
        "-c",
        "--in_channels",
        "--c",
        default=3,
        type=int,
        required=False,
        help="Number of Input Channels (Default=3)",
    )
    parser.add_argument(
        "-in_d",
        "-!",
        "--in_d",
        "--!",
        default=32,
        type=int,
        required=False,
        help="Input Depth (Default=32)",
    )
    parser.add_argument(
        "-in_h",
        "-H",
        "--in_h",
        "--H",
        default=32,
        type=int,
        required=False,
        help="Input Height (Default=32)",
    )
    parser.add_argument(
        "-in_w",
        "-W",
        "--in_w",
        "--W",
        default=32,
        type=int,
        required=False,
        help="Input Width (Default=32)",
    )
    parser.add_argument(
        "-out_channels",
        "-k",
        "--out_channels",
        "--k",
        default=32,
        type=int,
        required=False,
        help="Number of Output Channels (Default=32)",
    )
    parser.add_argument(
        "-fil_d",
        "-@",
        "--fil_d",
        "--@",
        default=3,
        type=int,
        required=False,
        help="Filter Depth (Default=3)",
    )
    parser.add_argument(
        "-fil_h",
        "-y",
        "--fil_h",
        "--y",
        default=3,
        type=int,
        required=False,
        help="Filter Height (Default=3)",
    )
    parser.add_argument(
        "-fil_w",
        "-x",
        "--fil_w",
        "--x",
        default=3,
        type=int,
        required=False,
        help="Filter Width (Default=3)",
    )
    parser.add_argument(
        "-conv_stride_d",
        "-#",
        "--conv_stride_d",
        "--#",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Depth (Default=1)",
    )
    parser.add_argument(
        "-conv_stride_h",
        "-u",
        "--conv_stride_h",
        "--u",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Height (Default=1)",
    )
    parser.add_argument(
        "-conv_stride_w",
        "-v",
        "--conv_stride_w",
        "--v",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Width (Default=1)",
    )
    parser.add_argument(
        "-pad_d",
        "-$",
        "--pad_d",
        "--$",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Depth (Default=0)",
    )
    parser.add_argument(
        "-pad_h",
        "-p",
        "--pad_h",
        "--p",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Height (Default=0)",
    )
    parser.add_argument(
        "-pad_w",
        "-q",
        "--pad_w",
        "--q",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Width (Default=0)",
    )
    parser.add_argument(
        "-verify",
        "-V",
        "--verify",
        "--V",
        default=1,
        type=int,
        required=False,
        help="Verify Each Layer (Default=1)",
    )
    parser.add_argument(
        "-time",
        "-t",
        "--time",
        "--t",
        default=0,
        type=int,
        required=False,
        help="Time Each Layer (Default=0)",
    )
    parser.add_argument(
        "-dilation_d",
        "-^",
        "--dilation_d",
        "--^",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Depth (Default=1)",
    )
    parser.add_argument(
        "-dilation_h",
        "-l",
        "--dilation_h",
        "--l",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Height (Default=1)",
    )
    parser.add_argument(
        "-dilation_w",
        "-j",
        "--dilation_w",
        "--j",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Width (Default=1)",
    )
    parser.add_argument(
        "-group_count",
        "-g",
        "--group_count",
        "--g",
        type=int,
        default=1,
        required=False,
        help="Number of Groups (Default=1)",
    )

    args, unknown = parser.parse_known_args()
    init_const_args(args)
    process_miopen_driver_name(args, unknown)
    print("Ignored args:")
    print(unknown)
    run_ck_profiler(args)
