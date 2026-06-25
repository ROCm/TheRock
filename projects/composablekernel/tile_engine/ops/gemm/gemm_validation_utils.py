# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
from typing import Tuple, List

GEMM_PIPELINES = ["mem", "compv3", "compv4"]

GEMM_PRESHUFFLE_PIPELINES = ["preshufflev2"]

LAYOUT_MAP = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

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

WARP_SUPPORTED_COMBINATIONS = {
    "gfx90a": [
        [1, 4, 1],
        [2, 2, 1],
        [4, 1, 1],
    ],
    "gfx942": [
        [1, 4, 1],
        [2, 2, 1],
        [4, 1, 1],
    ],
    "gfx950": [
        [1, 4, 1],
        [2, 2, 1],
        [4, 1, 1],
    ],
    "gfx1201": [
        [2, 4, 1],
        [1, 8, 1],
        [8, 1, 1],
        [4, 2, 1],
    ],
}

GEMM_PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS = {
    "gfx90a": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
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
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
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
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
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

GEMM_WARP_TILE_SUPPORTED_COMBINATIONS = {
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
    "gfx1201": {  # Check how to handle for GEMM and Multi D
        "fp16_fp16_fp16": [
            [16, 16, 16],
        ],
    },
}

TRAIT_UNSUPPORTED_COMBINATIONS = {
    ("compv3", "cshuffle", "interwave"),
    ("compv3", "default", "interwave"),
    ("compv4", "cshuffle", "interwave"),
    ("compv4", "default", "interwave"),
}


def element_size(data_type: str) -> float:
    """Calculate the size (in bytes) of a single element for given data type."""
    data_type = data_type.lower()
    if data_type not in ELEMENT_SIZE_MAP:
        raise ValueError(f"Unsupported data type: {data_type}")
    return ELEMENT_SIZE_MAP[data_type]


def is_trait_combination_valid(pipeline: str, epilogue: str, scheduler: str) -> bool:
    """Check if a trait combination is valid."""
    return (pipeline, epilogue, scheduler) not in TRAIT_UNSUPPORTED_COMBINATIONS


def validate_warp_configuration(
    warp_m: int,
    warp_n: int,
    warp_k: int,
    gpu_name: str,
) -> bool:
    """Validate warp configuration."""

    current_combination = [warp_m, warp_n, warp_k]

    allowed_combinations = WARP_SUPPORTED_COMBINATIONS.get(gpu_name, {})
    if not allowed_combinations:
        # If GPU not recognized, try to be permissive but log warning
        logging.warning(f"No warp_[m/n/k] combinations found for GPU: {gpu_name}")
        return True

    # Check if current combination is in the allowed list
    if current_combination not in allowed_combinations:
        return False

    return True


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

    max_tile_size = 2**15 if pipeline in ["preshufflev2", "compv4"] else 2**16

    if total_tile_in_lds > max_tile_size:
        error_msg = (
            f"LDS capacity exceeded: Total required {total_tile_in_lds:,}B ({total_tile_in_lds / 1024:.1f}KB) > "
            f"maximum allowed {max_tile_size:,}B ({max_tile_size / 1024}KB). Breakdown:\n"
            f"- Matrix A ({a_datatype}): {tile_m}x{tile_k} = {matrix_a_size:,}B\n"
            f"- Matrix B ({b_datatype}): {tile_n}x{tile_k} = {matrix_b_size:,}B"
        )
        return False, error_msg

    return True, ""


def validate_gemm_warp_tile_combination(
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    gpu_name: str,
) -> Tuple[bool, str]:
    """Validate warp tile combination against GPU-specific supported combinations."""

    # Construct the key for looking up supported combinations
    warp_tile_key = f"{a_datatype}_{b_datatype}_{c_datatype}"
    current_combination = [warp_tile_m, warp_tile_n, warp_tile_k]

    # Check if we have GPU-specific combinations
    gpu_warp_tile_combinations = GEMM_WARP_TILE_SUPPORTED_COMBINATIONS.get(gpu_name, {})
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


def validate_gemm_preshuffle_warp_tile_combination(
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    a_datatype: str,
    b_datatype: str,
    c_datatype: str,
    gpu_name: str,
) -> Tuple[bool, str]:
    """Validate warp tile combination against GPU-specific supported combinations."""

    # Construct the key for looking up supported combinations
    warp_tile_key = f"{a_datatype}_{b_datatype}_{c_datatype}"
    current_combination = [warp_tile_m, warp_tile_n, warp_tile_k]

    # Check if we have GPU-specific combinations
    gpu_warp_tile_combinations = GEMM_PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS.get(
        gpu_name, {}
    )
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
    layout: str,
    gpu_target: str,
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
    if not validate_warp_configuration(warp_m, warp_n, warp_k, gpu_target):
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

    if pipeline in GEMM_PIPELINES:
        gemm_valid, gemm_valid_error = validate_gemm(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            a_datatype,
            b_datatype,
            c_datatype,
            pipeline,
            layout,
            gpu_target,
        )
        if not gemm_valid:
            logging.debug(f"GEMM validation failed: {gemm_valid_error}")
            return False

        # Validate warp tile combination
        warp_tile_valid, warp_tile_error = validate_gemm_warp_tile_combination(
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            a_datatype,
            b_datatype,
            c_datatype,
            gpu_target,
        )
        if not warp_tile_valid:
            logging.debug(f"Warp tile validation failed: {warp_tile_error}")
            return False

    elif pipeline in GEMM_PRESHUFFLE_PIPELINES:
        preshuffle_valid, preshuffle_valid_error = validate_gemm_preshuffle(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            a_datatype,
            b_datatype,
            c_datatype,
            pipeline,
            layout,
            gpu_target,
        )
        if not preshuffle_valid:
            logging.debug(
                f"GEMM Preshuffle validation failed: {preshuffle_valid_error}"
            )
            return False

        # Validate warp tile combination
        warp_tile_valid, warp_tile_error = (
            validate_gemm_preshuffle_warp_tile_combination(
                warp_tile_m,
                warp_tile_n,
                warp_tile_k,
                a_datatype,
                b_datatype,
                c_datatype,
                gpu_target,
            )
        )
        if not warp_tile_valid:
            logging.debug(f"Warp tile validation failed: {warp_tile_error}")
            return False

    return True


# [TODO] Handle this while moving code to commons Add more datatype to this function if needed
def get_dtype_string(datatype: str) -> str:
    """Get C++ type string for datatype"""
    dtype_map = {
        "fp16": "ck_tile::fp16_t",
        "fp8": "ck_tile::fp8_t",
        "bf8": "ck_tile::bf8_t",
        "bf16": "ck_tile::bf16_t",
        "fp32": "float",
        "fp64": "double",
    }
    return dtype_map.get(datatype, "float")


def get_abc_layouts(layout_code: str) -> Tuple[str, str, str]:
    """
    Return (ALayout, BLayout, CLayout) from a 3-letter code like 'rcr', 'ccr', 'crr', 'rrr'.
    """
    code = str(layout_code).strip().lower()

    a_layout = LAYOUT_MAP[code[0]]
    b_layout = LAYOUT_MAP[code[1]]
    c_layout = LAYOUT_MAP[code[2]]
    return a_layout, b_layout, c_layout


def get_abcd_layouts(layout_code: str) -> Tuple[str, str, str, List[str]]:
    """
    Return (ALayout, BLayout, CLayout) from a 3-letter code like 'rcrr', 'ccrr', 'crrr', 'rrrr'.
    """
    code = str(layout_code).strip().lower()

    a_layout = LAYOUT_MAP[code[0]]
    b_layout = LAYOUT_MAP[code[1]]
    c_layout = LAYOUT_MAP[code[2]]
    d0_layout = LAYOUT_MAP[code[3]]
    d1_layout = LAYOUT_MAP[code[3]]
    return a_layout, b_layout, c_layout, [d0_layout, d1_layout]


def validate_whole_wg_cover_configuration(
    tile_m,
    tile_n,
    tile_k,
    warp_m,
    warp_n,
    warp_k,
    layout,
    a_datatype,
    b_datatype,
) -> Tuple[bool, str]:
    # Validate whole workgroup cover configuration

    warp_size = 64
    NumWarps = warp_m * warp_n * warp_k
    BlockSize = NumWarps * warp_size

    XPerTile = 0
    YPerTile = 0
    vector_load_size = 0

    # A matrix validation
    if layout[0] == "r":
        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, a_datatype, tile_m, tile_k
        )

        XPerTile = tile_k
        YPerTile = tile_m

    elif layout[0] == "c":
        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, a_datatype, tile_m, tile_m
        )

        # Validate distribution
        XPerTile = tile_k
        YPerTile = tile_m

        wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
            XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
        )

        if not wg_cover_core_valid:
            logging.debug(
                f"whole workgroup cover failed for Matrix A distribution: {wg_cover_core_error}"
            )
            return False, wg_cover_core_error

        XPerTile = tile_m
        YPerTile = tile_k

    wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
        XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
    )

    if not wg_cover_core_valid:
        logging.debug(
            f"whole workgroup cover failed for Matrix A: {wg_cover_core_error}"
        )
        return False, wg_cover_core_error

    # B matrix validation
    if layout[1] == "r":
        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, b_datatype, tile_n, tile_n
        )

        # Validate distribution
        XPerTile = tile_k
        YPerTile = tile_n

        wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
            XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
        )

        if not wg_cover_core_valid:
            logging.debug(
                f"whole workgroup cover failed for Matrix B distribution: {wg_cover_core_error}"
            )
            return False, wg_cover_core_error

        XPerTile = tile_n
        YPerTile = tile_k

    elif layout[1] == "c":
        XPerTile = tile_k
        YPerTile = tile_n

        vector_load_size = get_global_vector_load_size(
            BlockSize, tile_k, b_datatype, tile_n, tile_k
        )

    wg_cover_core_valid, wg_cover_core_error = wg_cover_core_validation(
        XPerTile, YPerTile, BlockSize, vector_load_size, warp_size
    )
    if not wg_cover_core_valid:
        logging.debug(
            f"whole workgroup cover failed for Matrix B: {wg_cover_core_error}"
        )
        return False, wg_cover_core_error

    return True, ""


def wg_cover_core_validation(
    XPerTile: int,
    YPerTile: int,
    BlockSize: int,
    vector_load_size: int,
    warp_size: int,
) -> Tuple[bool, str]:
    if XPerTile % vector_load_size != 0:
        return False, "XPerTile is not divisible by vector_load_size"

    num_warps = BlockSize / warp_size
    LargestVec = (XPerTile * YPerTile) / (num_warps * warp_size)

    X1 = LargestVec if vector_load_size > LargestVec else vector_load_size
    X0 = XPerTile / X1
    Y1 = warp_size // X0

    if X0 * Y1 != warp_size:
        return False, "X0 * Y1 != warp_size"

    return True, ""


def get_global_vector_load_size(
    BlockSize: int,
    KPerBlock: int,
    DataType: str,
    MNPerBlock: int,
    XPerTile: int,
) -> int:
    elements_per_thread = MNPerBlock * KPerBlock / BlockSize
    PackedSize = 1

    if (
        PackedSize == 2
        and XPerTile % (PackedSize * 32 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 32 / element_size(DataType)) == 0
    ):
        return PackedSize * 32 / element_size(DataType)
    elif (
        XPerTile % (PackedSize * 16 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 16 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 16 / element_size(DataType))

    elif (
        XPerTile % (PackedSize * 8 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 8 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 8 / element_size(DataType))
    elif (
        element_size(DataType) >= PackedSize * 4
        and XPerTile % (PackedSize * 4 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 4 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 4 / element_size(DataType))
    elif (
        element_size(DataType) >= PackedSize * 2
        and XPerTile % (PackedSize * 2 / element_size(DataType)) == 0
        and elements_per_thread % (PackedSize * 2 / element_size(DataType)) == 0
    ):
        return int(PackedSize * 2 / element_size(DataType))
    else:
        return PackedSize


def validate_gemm(
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
    layout: str,
    gpu_target: str,
    trait_name: str = None,
) -> bool:
    # GEMM Validation
    # Validate whole workgroup cover configuration
    whole_workgroup_cover_valid, whole_workgroup_cover_error = (
        validate_whole_wg_cover_configuration(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            layout,
            a_datatype,
            b_datatype,
        )
    )
    if not whole_workgroup_cover_valid:
        logging.debug(
            f"Whole workgroup cover configuration validation failed: {whole_workgroup_cover_error}"
        )
        return False, whole_workgroup_cover_error

    return True, ""


def validate_gemm_preshuffle(
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
    layout: str,
    gpu_target: str,
    trait_name: str = None,
) -> bool:
    # Preshuffle Validations
    # Validate vector load alignment
    m_iter_per_warp = tile_m / (warp_m * warp_tile_m)
    vector_valid, vector_error = validate_vector_load_alignment(
        warp_tile_m,
        warp_tile_k,
        a_datatype,
        m_iter_per_warp,
        wave_size=64,
        vector_load_size=16,
    )
    if not vector_valid:
        logging.debug(f"Vector load alignment failed: {vector_error}")
        return False, "vector load alignment error"

    # Validate M0, M1, M2 configuration for matrix A row-major layout
    m0_m1_m2_valid, m0_m1_m2_error = validate_m0_m1_m2_configuration(
        tile_m,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        a_datatype,
        vector_load_size=16,
        warp_size=64,
    )
    if not m0_m1_m2_valid:
        logging.debug(f"M0/M1/M2 configuration validation failed: {m0_m1_m2_error}")
        return False, m0_m1_m2_error

    return True, ""


def validate_vector_load_alignment(
    wg_m: int,
    wg_k: int,
    a_datatype: str,
    m_iter_per_warp: int,
    wave_size: int,
    vector_load_size: int,
) -> Tuple[bool, str]:
    try:
        # Calculate the memory access pattern size
        a_element_size = element_size(a_datatype)
        access_size = (wg_m * wg_k * a_element_size * m_iter_per_warp) / wave_size

        # Check if it's aligned to vector load size
        if access_size % vector_load_size != 0:
            error_msg = (
                f"Vector load alignment violation: "
                f"({wg_m} * {wg_k} * {a_element_size} * {m_iter_per_warp} / {wave_size}) "
                f"% {vector_load_size} = {access_size % vector_load_size} != 0. "
                f"Access size: {access_size} bytes"
            )
            return False, error_msg

        return True, ""

    except Exception as e:
        return False, f"Error in vector load validation: {str(e)}"


def validate_m0_m1_m2_configuration(
    tile_m: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    a_datatype: str,
    vector_load_size: int = 16,
    warp_size: int = 64,
) -> Tuple[bool, str]:
    """
    Validate M0, M1, M2 configuration for matrix A row-major layout.
    This ensures proper memory access pattern alignment.
    """
    try:
        # Validation for A as row-major
        MPerBlock = tile_m

        # Calculate K1 using element size
        K1 = vector_load_size / element_size(a_datatype)

        # Check if K1 is valid (must be integer)
        if K1 != int(K1):
            return (
                False,
                f"K1 = {K1} is not an integer. vector_load_size({vector_load_size}) must be divisible by element_size({a_datatype})",
            )
        K1 = int(K1)

        # Calculate K0
        if tile_k % K1 != 0:
            return False, f"tile_k({tile_k}) must be divisible by K1({K1})"
        K0 = tile_k // K1

        # Calculate M2
        if warp_size % K0 != 0:
            return False, f"warp_size({warp_size}) must be divisible by K0({K0})"
        M2 = warp_size // K0

        # Calculate number of warps and block size
        NumWarps = warp_m * warp_n * warp_k
        BlockSize = NumWarps * warp_size

        # Calculate M0 (assuming get_warp_size() returns warp_size)
        M0 = BlockSize // warp_size  # This should equal NumWarps

        # Calculate M1
        if (M2 * M0) == 0:
            return False, f"M2({M2}) * M0({M0}) cannot be zero"

        if MPerBlock % (M2 * M0) != 0:
            return (
                False,
                f"MPerBlock({MPerBlock}) must be divisible by M2({M2}) * M0({M0}) = {M2 * M0}",
            )
        M1 = MPerBlock // (M2 * M0)

        # Validate the assertion: M0 * M1 * M2 == MPerBlock
        calculated_m_per_block = M0 * M1 * M2
        if calculated_m_per_block != MPerBlock:
            error_msg = (
                f"Incorrect M0, M1, M2 configuration! "
                f"M0({M0}) * M1({M1}) * M2({M2}) = {calculated_m_per_block} != MPerBlock({MPerBlock}). "
                f"Configuration: K0={K0}, K1={K1}, NumWarps={NumWarps}, BlockSize={BlockSize}"
            )
            return False, error_msg

        return True, ""

    except ZeroDivisionError as e:
        return False, f"Division by zero in M0/M1/M2 calculation: {str(e)}"
    except Exception as e:
        return False, f"Error in M0/M1/M2 validation: {str(e)}"
