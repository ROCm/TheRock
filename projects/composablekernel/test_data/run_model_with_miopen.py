#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
PyTorch Model Runner with MIOpen Command Logging using torchvision models

Usage:
    MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py --model resnet18 2> miopen_commands.txt

Available 2D models: alexnet, vgg11, vgg16, resnet18, resnet50, mobilenet_v2, etc.
Available 3D models: r3d_18, mc3_18, r2plus1d_18
"""

import torch
import torchvision.models as models
import torchvision.models.video as video_models
import argparse
import os

# Define available models
MODELS_2D = [
    "alexnet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "inception_v3",
    "googlenet",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
    "squeezenet1_0",
    "squeezenet1_1",
]

MODELS_3D = ["r3d_18", "mc3_18", "r2plus1d_18"]

ALL_MODELS = MODELS_2D + MODELS_3D


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Model Runner with MIOpen Command Logging"
    )

    # Model selection
    parser.add_argument(
        "--model", choices=ALL_MODELS, default="resnet18", help="Model to run"
    )

    # Input tensor dimensions
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Input channels (e.g., 3 for RGB, 1 for grayscale)",
    )
    parser.add_argument("--height", type=int, default=224, help="Input height")
    parser.add_argument("--width", type=int, default=224, help="Input width")
    parser.add_argument(
        "--input-size",
        type=int,
        help="Input size (sets both height and width to same value)",
    )
    parser.add_argument(
        "--temporal-size", type=int, default=16, help="Temporal dimension for 3D models"
    )

    # Device and precision
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run on",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Floating point precision",
    )

    # Output control
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output except errors"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Handle input-size override
    if args.input_size:
        args.height = args.input_size
        args.width = args.input_size

    # Check MIOpen logging
    if not os.environ.get("MIOPEN_ENABLE_LOGGING_CMD") and not args.quiet:
        print("WARNING: Set MIOPEN_ENABLE_LOGGING_CMD=1 to capture commands")

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Check if actually running on GPU
    if device.type == "cpu":
        import sys

        print(
            "WARNING: Running on CPU, MIOpen commands will not be generated!",
            file=sys.stderr,
        )
        print(f"CUDA/ROCm available: {torch.cuda.is_available()}", file=sys.stderr)
        if torch.cuda.is_available():
            print(f"GPU device count: {torch.cuda.device_count()}", file=sys.stderr)
            print(
                f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}",
                file=sys.stderr,
            )
        # Continue anyway for testing purposes

    if not args.quiet:
        print(f"Using device: {device}")

    # Create model using torchvision
    if args.model in MODELS_3D:
        # 3D Video models
        model = getattr(video_models, args.model)(weights=None)
        # 3D input: (batch, channels, temporal, height, width)
        input_tensor = torch.randn(
            args.batch_size, args.channels, args.temporal_size, args.height, args.width
        )
        if not args.quiet:
            print(f"3D model: {args.model}")
            print(f"Input shape: {input_tensor.shape} (B, C, T, H, W)")
    else:
        # 2D Image models
        model = getattr(models, args.model)(weights=None)
        # 2D input: (batch, channels, height, width)
        input_tensor = torch.randn(
            args.batch_size, args.channels, args.height, args.width
        )
        if not args.quiet:
            print(f"2D model: {args.model}")
            print(f"Input shape: {input_tensor.shape} (B, C, H, W)")

    # Set precision
    if args.precision == "fp16":
        model = model.half()
        input_tensor = input_tensor.half()
    elif args.precision == "bf16":
        model = model.bfloat16()
        input_tensor = input_tensor.bfloat16()

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if not args.quiet:
        print(f"Running {args.model} model...")

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if not args.quiet:
            print(f"Output shape: {output.shape}")

    if not args.quiet:
        print("Done! MIOpen commands logged to stderr")


if __name__ == "__main__":
    main()
