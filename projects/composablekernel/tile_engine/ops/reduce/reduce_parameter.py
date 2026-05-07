# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from itertools import product

from pyparsing import List

TYPE_MAP = {"fp16": "ck_tile::half_t", "float": "float"}


@dataclass
class ParametersBlockwise:
    tile_m: int
    tile_n: int
    warp_per_block_m: int
    warp_per_block_n: int
    warp_m: int
    warp_n: int
    thread_tile_m: int
    thread_tile_n: int
    input_shape: List[int]

    def __str__(self):
        tile_size = "x".join(str(i) for i in [self.tile_m, self.tile_n])
        warp_per_block = "x".join(
            str(i) for i in [self.warp_per_block_m, self.warp_per_block_n]
        )
        warp_size = "x".join(str(i) for i in [self.warp_m, self.warp_n])
        thread_tile_size = "x".join(
            str(i) for i in [self.thread_tile_m, self.thread_tile_n]
        )
        input_shape = "x".join(str(i) for i in self.input_shape)

        return "_".join(
            [tile_size, warp_per_block, warp_size, thread_tile_size, input_shape]
        )


def get_parameter_combinations(
    config_dict: dict,
) -> List[ParametersBlockwise]:
    input_shape_configs = config_dict["problem_size"]["input_shape"]

    fixed_configs = config_dict["tile_config"].get("fixed", None)

    seen_config = set()

    if fixed_configs is not None:
        for fixed in fixed_configs:
            tile_m_values = fixed["tile_m"]
            tile_n_values = fixed["tile_n"]
            warp_per_block_m_values = fixed["warp_per_block_m"]
            warp_per_block_n_values = fixed["warp_per_block_n"]
            warp_m_values = fixed["warp_tile_m"]
            warp_n_values = fixed["warp_tile_n"]
            thread_tile_m_values = fixed["thread_tile_m"]
            thread_tile_n_values = fixed["thread_tile_n"]
            for combo in product(
                [tile_m_values],
                [tile_n_values],
                [warp_per_block_m_values],
                [warp_per_block_n_values],
                [warp_m_values],
                [warp_n_values],
                [thread_tile_m_values],
                [thread_tile_n_values],
                input_shape_configs,
            ):
                p = ParametersBlockwise(*combo)
                if is_valid_combination(p):
                    hashable_combo = (tuple(combo[-1]),) + combo[0:-1]
                    seen_config.add(hashable_combo)
                    yield p

    combo_config = config_dict["tile_config"].get("combination", None)
    if combo_config is None:
        tile_m_values = combo_config["tile_m"]["values"]
        tile_n_values = combo_config["tile_n"]["values"]
        warp_per_block_m_values = combo_config["warp_per_block_m"]["values"]
        warp_per_block_n_values = combo_config["warp_per_block_n"]["values"]
        warp_m_values = combo_config["warp_tile_m"]["values"]
        warp_n_values = combo_config["warp_tile_n"]["values"]
        thread_tile_m_values = combo_config["thread_tile_m"]["values"]
        thread_tile_n_values = combo_config["tile_config"]["thread_tile_n"]["values"]

        for combo in product(
            tile_m_values,
            tile_n_values,
            warp_per_block_m_values,
            warp_per_block_n_values,
            warp_m_values,
            warp_n_values,
            thread_tile_m_values,
            thread_tile_n_values,
            input_shape_configs,
        ):
            if combo:
                p = ParametersBlockwise(*combo)
                hashable_combo = (tuple(combo[-1]),) + combo[0:-1]
                if is_valid_combination(p) and hashable_combo not in seen_config:
                    yield p


def is_valid_combination(p: ParametersBlockwise) -> bool:
    # Thread tile must be at least 1
    if p.thread_tile_m < 1 or p.thread_tile_n < 1:
        return False

    # Alignment check
    if p.tile_m % (p.warp_per_block_m * p.warp_m) != 0:
        return False
    if p.tile_n % (p.warp_per_block_n * p.warp_n) != 0:
        return False

    # Reduction dimension size must be divisible by tile size
    if len(p.input_shape) == 4 and (
        p.input_shape[2] * p.input_shape[3] % p.thread_tile_n != 0
    ):
        return False

    if len(p.input_shape) == 3 and (
        p.input_shape[1] * p.input_shape[2] % p.thread_tile_n != 0
    ):
        return False

    return True
