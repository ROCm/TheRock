#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Generate Model Configuration Combinations for MIOpen Testing

This script generates all possible combinations of model parameters
and saves them as CSV files that can be read by the shell script.
"""

import csv
import argparse


def generate_2d_configs(mode="full"):
    """Generate all 2D model configuration combinations

    Args:
        mode: 'small' for minimal set (~50 configs), 'half' for reduced set (~250 configs), 'full' for comprehensive set (~500 configs)
    """

    # Define parameter ranges
    models_2d = [
        "resnet18",
        "resnet34",
        "resnet50",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "vgg11",
        "vgg16",
        "vgg19",
        "alexnet",
        "googlenet",
        "densenet121",
        "densenet161",
        "squeezenet1_0",
        "squeezenet1_1",
        "shufflenet_v2_x1_0",
    ]

    if mode == "small":
        # Minimal set for quick testing
        batch_sizes = [1, 8]  # Just two batch sizes
        # Very limited input dimensions - only 2 key sizes
        input_dims = [
            (224, 224),  # Standard (most common)
            (256, 256),  # Medium
        ]
        # Use only first 3 models for minimal testing
        models_2d = models_2d[:3]  # Only resnet18, resnet34, resnet50
    elif mode == "half":
        # Reduced set for faster testing
        batch_sizes = [1, 8, 32]  # Small, medium, large
        # Reduced input dimensions - 5 key sizes
        input_dims = [
            (64, 64),  # Small
            (224, 224),  # Standard (most common)
            (512, 512),  # Large
            (224, 320),  # Rectangular
            (227, 227),  # AlexNet preferred
        ]
    else:  # full mode
        # More comprehensive but still limited
        batch_sizes = [1, 4, 8, 16, 32]
        # More dimensions but skip some redundant ones
        input_dims = [
            (64, 64),
            (128, 128),
            (224, 224),
            (256, 256),
            (512, 512),  # Square
            (224, 320),
            (320, 224),  # Rectangular (reduced from 4)
            (227, 227),  # AlexNet preferred
            (299, 299),  # Inception preferred
        ]

    precisions = ["fp32"]  # , 'fp16', 'bf16']
    channels = [3]  # Most models expect RGB

    configs = []
    config_id = 1

    # Generate all combinations (but limit to reasonable subset)
    for model in models_2d:
        for batch_size in batch_sizes:
            for height, width in input_dims:
                for precision in precisions:
                    # Skip some combinations to keep dataset manageable
                    if batch_size > 16 and height > 256:
                        continue  # Skip large batch + large image combinations
                    if precision != "fp32" and batch_size < 8:
                        continue  # Skip mixed precision with tiny batches

                    config_name = f"{model}_b{batch_size}_{height}x{width}_{precision}"

                    config = {
                        "config_name": config_name,
                        "model": model,
                        "batch_size": batch_size,
                        "channels": channels[0],
                        "height": height,
                        "width": width,
                        "precision": precision,
                    }

                    configs.append(config)
                    config_id += 1

    return configs


def generate_3d_configs(mode="full"):
    """Generate all 3D model configuration combinations

    Args:
        mode: 'small' for minimal set (~10 configs), 'half' for reduced set (~50 configs), 'full' for comprehensive set (~100 configs)
    """

    models_3d = ["r3d_18", "mc3_18", "r2plus1d_18"]

    if mode == "small":
        # Minimal set for quick testing
        batch_sizes = [1, 4]  # Just two batch sizes
        temporal_sizes = [8]  # Only smallest temporal size
        # Very limited spatial dimensions
        input_dims = [
            (112, 112),  # Standard for 3D
        ]
        # Use only first model for minimal testing
        models_3d = models_3d[:1]  # Only r3d_18
    elif mode == "half":
        # Reduced set for faster testing
        batch_sizes = [1, 4, 8]  # Skip batch_size=2
        temporal_sizes = [8, 16]  # Skip 32 (most expensive)
        # Reduced spatial dimensions
        input_dims = [
            (112, 112),  # Small (common for video)
            (224, 224),  # Standard
            (224, 320),  # Rectangular
        ]
    else:  # full mode
        # More comprehensive but still reasonable
        batch_sizes = [1, 2, 4, 8]  # 3D models are more memory intensive
        temporal_sizes = [8, 16, 32]
        # More dimensions
        input_dims = [
            (112, 112),
            (224, 224),
            (256, 256),  # Standard sizes
            (224, 320),
            (320, 224),  # Rectangular
        ]

    precisions = ["fp32"]  # , 'fp16']  # Skip bf16 for 3D to reduce combinations
    channels = [3]

    configs = []

    for model in models_3d:
        for batch_size in batch_sizes:
            for temporal_size in temporal_sizes:
                for height, width in input_dims:
                    for precision in precisions:
                        # Skip very large combinations
                        if batch_size > 4 and temporal_size > 16:
                            continue
                        if batch_size > 2 and height > 224:
                            continue

                        config_name = f"{model}_b{batch_size}_t{temporal_size}_{height}x{width}_{precision}"

                        config = {
                            "config_name": config_name,
                            "model": model,
                            "batch_size": batch_size,
                            "channels": channels[0],
                            "temporal_size": temporal_size,
                            "height": height,
                            "width": width,
                            "precision": precision,
                        }

                        configs.append(config)

    return configs


def save_configs_to_csv(configs, filename, config_type):
    """Save configurations to CSV file"""

    if not configs:
        print(f"No {config_type} configurations generated")
        return

    fieldnames = list(configs[0].keys())

    with open(filename, "w", newline="\n", encoding="utf-8") as csvfile:
        csvfile.write(f"# {config_type} Model Configurations\n")
        csvfile.write(f"# Generated {len(configs)} configurations\n")

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for config in configs:
            writer.writerow(config)

    print(f"Generated {len(configs)} {config_type} configurations → {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model configuration combinations"
    )
    parser.add_argument(
        "--output-2d",
        type=str,
        default="model_configs_2d.csv",
        help="Output file for 2D configurations",
    )
    parser.add_argument(
        "--output-3d",
        type=str,
        default="model_configs_3d.csv",
        help="Output file for 3D configurations",
    )
    parser.add_argument(
        "--mode",
        choices=["small", "half", "full"],
        default="full",
        help="Configuration mode: small (~60 total), half (~300 total) or full (~600 total) (default: half)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of configurations per type (for testing)",
    )

    args = parser.parse_args()

    print(f"Generating {args.mode} model configurations...")

    print("Generating 2D model configurations...")
    configs_2d = generate_2d_configs(mode=args.mode)
    if args.limit:
        configs_2d = configs_2d[: args.limit]
    save_configs_to_csv(configs_2d, args.output_2d, "2D")

    print("Generating 3D model configurations...")
    configs_3d = generate_3d_configs(mode=args.mode)
    if args.limit:
        configs_3d = configs_3d[: args.limit]
    save_configs_to_csv(configs_3d, args.output_3d, "3D")

    print(
        f"\nTotal configurations: {len(configs_2d)} 2D + {len(configs_3d)} 3D = {len(configs_2d) + len(configs_3d)}"
    )
    print("\nTo use these configurations:")
    print("  Update generate_test_dataset.sh to read from these CSV files")


if __name__ == "__main__":
    main()
