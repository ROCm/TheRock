#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation utilities for GEMM kernel generation.
Extracted from tile_engine_develop for consistency.
"""

import subprocess
import re
from functools import lru_cache
import logging
from typing import Tuple, List

# Element size mapping for different data types
ELEMENT_SIZE_MAP = {
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
    "bf8": 1,
    "int4": 0.5,
    "int32": 4,
    "fp32": 4,
    "fp64": 8,
}

# Supported warp tile combinations for different GPU architectures and data types
WARP_TILE_SUPPORTED_COMBINATIONS = {
    "gfx90a": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32]],
        "bf8_bf8_fp16": [[32, 32, 16], [32, 32, 32]],
    },
    "gfx942": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
        "bf8_bf8_fp16": [[32, 32, 16], [32, 32, 32], [16, 16, 64], [16, 16, 32]],
        "int8_int8_int32": [[16, 16, 32], [32, 32, 16]],
    },
    "gfx950": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [
            [32, 32, 16],
            [32, 32, 32],
            [16, 16, 32],
            [16, 16, 64],
            [16, 16, 128],
            [32, 32, 64],
        ],
        "bf8_bf8_fp16": [
            [32, 32, 16],
            [32, 32, 32],
            [16, 16, 64],
            [16, 16, 32],
            [16, 16, 128],
            [32, 32, 64],
        ],
    },
}

# Unsupported trait combinations
TRAIT_UNSUPPORTED_COMBINATIONS = {
    ("compv3", "cshuffle", "interwave", "reduction"),
    ("compv3", "default", "interwave", "reduction"),
    ("compv3", "cshuffle", "interwave", "atomic"),
    ("compv3", "default", "interwave", "atomic"),
}


def element_size(data_type: str) -> float:
    """Calculate the size (in bytes) of a single element for given data type."""
    data_type = data_type.lower()
    if data_type not in ELEMENT_SIZE_MAP:
        raise ValueError(f"Unsupported data type: {data_type}")
    return ELEMENT_SIZE_MAP[data_type]


GPU_NAME_PATTERN = re.compile(r"Name:\s*(gfx\d+\w*)")


@lru_cache(maxsize=1)
def get_gpu_name_by_id(gpu_id: int = 0) -> str:
    """Retrieve GPU name (e.g. gfx90a) by device ID"""
    try:
        output = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.PIPE, timeout=5
        )
        if matches := GPU_NAME_PATTERN.finditer(output):
            gpu_list = [m.group(1) for m in matches]
            return gpu_list[gpu_id] if gpu_id < len(gpu_list) else ""

        return ""

    except subprocess.CalledProcessError as e:
        logging.debug(f"GPU query failed (exit {e.returncode}): {e.stderr.strip()}")
    except FileNotFoundError:
        logging.debug("ROCm tools not installed (requires rocminfo)")
    except subprocess.TimeoutExpired:
        logging.debug("GPU query timeout (5s)")
    except Exception as e:
        logging.debug(f"GPU detection error: {str(e)}")

    return ""


def is_trait_combination_valid(
    pipeline: str, epilogue: str, scheduler: str, reduction_strategy: str
) -> bool:
    """Check if a trait combination is valid."""
    return (
        pipeline,
        epilogue,
        scheduler,
        reduction_strategy,
    ) not in TRAIT_UNSUPPORTED_COMBINATIONS


def validate_warp_configuration(warp_m: int, warp_n: int, warp_k: int) -> bool:
    """Validate warp configuration."""
    return (warp_m, warp_n, warp_k) in [(1, 4, 1), (2, 2, 1), (4, 1, 1)]


def validate_dimension_alignment(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
) -> Tuple[bool, List[str]]:
    """Check if tile dimensions are properly aligned with warp dimensions."""
    alignment_issues = []

    if tile_m % (warp_m * warp_tile_m) != 0:
        alignment_issues.append(
            f"tile_m({tile_m}) % [{warp_m}x{warp_tile_m}] = {tile_m % (warp_m * warp_tile_m)}"
        )
    if tile_n % (warp_n * warp_tile_n) != 0:
        alignment_issues.append(
            f"tile_n({tile_n}) % [{warp_n}x{warp_tile_n}] = {tile_n % (warp_n * warp_tile_n)}"
        )
    if tile_k % (warp_k * warp_tile_k) != 0:
        alignment_issues.append(
            f"tile_k({tile_k}) % [{warp_k}x{warp_tile_k}] = {tile_k % (warp_k * warp_tile_k)}"
        )

    return len(alignment_issues) == 0, alignment_issues


def validate_lds_capacity(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_datatype: str,
    b_datatype: str,
    pipeline: str,
) -> Tuple[bool, str]:
    """Validate LDS capacity requirements."""
    matrix_a_size = (tile_m * tile_k) * element_size(a_datatype)
    matrix_b_size = (tile_n * tile_k) * element_size(b_datatype)
    total_tile_in_lds = matrix_a_size + matrix_b_size

    max_tile_size = 2**15 if pipeline == "compv4" else 2**16

    if total_tile_in_lds > max_tile_size:
        error_msg = (
            f"LDS capacity exceeded: Total required {total_tile_in_lds:,}B ({total_tile_in_lds / 1024:.1f}KB) > "
            f"maximum allowed {max_tile_size:,}B ({max_tile_size / 1024}KB). Breakdown:\n"
            f"- Matrix A ({a_datatype}): {tile_m}x{tile_k} = {matrix_a_size:,}B\n"
            f"- Matrix B ({b_datatype}): {tile_n}x{tile_k} = {matrix_b_size:,}B"
        )
        return False, error_msg

    return True, ""


def validate_warp_tile_combination(
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    gpu_name: str = None,
) -> Tuple[bool, str]:
    """Validate warp tile combination against GPU-specific supported combinations."""
    if gpu_name is None:
        gpu_name = get_gpu_name_by_id(0)

    # Construct the key for looking up supported combinations
    warp_tile_key = f"{a_datatype}_{b_datatype}_{c_datatype}"
    current_combination = [warp_tile_m, warp_tile_n, warp_tile_k]

    # Check if we have GPU-specific combinations
    gpu_warp_tile_combinations = WARP_TILE_SUPPORTED_COMBINATIONS.get(gpu_name, {})
    if not gpu_warp_tile_combinations:
        # If GPU not recognized, try to be permissive but log warning
        logging.warning(f"No warp tile combinations found for GPU: {gpu_name}")
        return True, ""

    # Check if we have combinations for this data type combination
    allowed_combinations = gpu_warp_tile_combinations.get(warp_tile_key, [])
    if not allowed_combinations:
        # For data type combinations not in the list, be permissive
        logging.debug(
            f"No warp tile combinations found for data types: {warp_tile_key}"
        )
        return True, ""

    # Check if current combination is in the allowed list
    if current_combination not in allowed_combinations:
        error_msg = (
            f"Invalid warp tile combination: {current_combination} not in allowed list. "
            f"Valid combinations for '{warp_tile_key}' on {gpu_name}: {allowed_combinations}"
        )
        return False, error_msg

    return True, ""


def is_tile_config_valid(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    pipeline: str,
    trait_name: str = None,
) -> bool:
    """
    Comprehensive tile configuration validation.
    Returns True if configuration is valid, False otherwise.
    """
    # Basic sanity checks
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        return False
    if warp_m <= 0 or warp_n <= 0 or warp_k <= 0:
        return False
    if warp_tile_m <= 0 or warp_tile_n <= 0 or warp_tile_k <= 0:
        return False

    # Check that warp tiles fit within block tiles
    if warp_m * warp_tile_m > tile_m:
        return False
    if warp_n * warp_tile_n > tile_n:
        return False
    if warp_k * warp_tile_k > tile_k:
        return False

    # Validate warp configuration
    if not validate_warp_configuration(warp_m, warp_n, warp_k):
        logging.debug(
            f"Invalid warp configuration: warp_m({warp_m}), warp_n({warp_n}), warp_k({warp_k})"
        )
        return False

    # Validate dimension alignment
    is_aligned, alignment_issues = validate_dimension_alignment(
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
    )
    if not is_aligned:
        logging.debug(
            f"Dimension alignment failed: {', '.join(alignment_issues)}. "
            f"Tile dimensions {tile_m}x{tile_n}x{tile_k} must be divisible by "
            f"[warp]: {warp_m}x{warp_n}x{warp_k} x [warp_tile]: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
        )
        return False

    # Validate LDS capacity
    lds_valid, lds_error = validate_lds_capacity(
        tile_m, tile_n, tile_k, a_datatype, b_datatype, pipeline
    )
    if not lds_valid:
        logging.debug(f"LDS validation failed: {lds_error}")
        return False

    # Validate warp tile combination
    warp_tile_valid, warp_tile_error = validate_warp_tile_combination(
        warp_tile_m, warp_tile_n, warp_tile_k, a_datatype, b_datatype, c_datatype
    )
    if not warp_tile_valid:
        logging.debug(f"Warp tile validation failed: {warp_tile_error}")
        return False

    return True
