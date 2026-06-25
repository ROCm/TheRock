# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from reduce_config import ReduceConfig
from reduce_parameter import get_parameter_combinations, TYPE_MAP


class MultiReduceBase:
    def __init__(self, working_path, gpu_target, datatype, config_json=None):
        self.working_path = Path(working_path)
        self.gpu_target = gpu_target
        self.datatype = datatype
        self.output_type = self.datatype
        self.config = ReduceConfig(config_json) if config_json else None
        self.name = "multiops_base"

        self.signature_test = {
            3: "Test3D_KeepDim0_ReduceDim12",
            4: "Test4D_KeepDim01_ReduceDim23",
        }
        self.header = "test_multi_reduce2d_multiblock_impl.hpp"
        self.test_type = "TestCkTileMultiReduce2D"

    def _generate_instances(self):
        if not self.config:
            raise ValueError("Configuration not provided.")

        instances = []
        for params in get_parameter_combinations(self.config.config_dict):
            instance = self._create_instance(params)
            instances.append((instance, params))
        return instances

    def _create_instance(self, parameters):
        generated_test = self._get_test(parameters)

        return generated_test

    def do_list_blobs(self):
        with open(
            self.working_path / Path(f"reduce_{self.name}_blobs_list.txt"), "w"
        ) as f:
            combos_str = [
                f"{self.name}_{params}"
                for params in get_parameter_combinations(self.config.config_dict)
            ]
            f.write("\n".join(combos_str))
            f.write("\n")

    def do_generate_blobs(self):
        instances = self._generate_instances()
        for instance_code, params in instances:
            blob_filename = self.working_path / Path(f"test_{self.name}_{params}.cpp")
            with open(blob_filename, "w") as f:
                f.write(instance_code)

    def _get_test(self, params):
        dimension = len(params.input_shape)
        signature = self.signature_test.get(dimension, None)

        if not signature:
            raise ValueError(
                f"No test signature found for input shape dimension: {dimension}"
            )

        shape_str = [str(i) for i in params.input_shape]
        input_shape_arg_str = ",".join(shape_str)
        input_shape_str = "x".join(shape_str)

        t = f"""#include "{self.header}"

using Shape_BlockWarps = ck_tile::sequence<{params.warp_per_block_m}, {params.warp_per_block_n}>;
using Shape_BlockTile  = ck_tile::sequence<{params.tile_m}, {params.tile_n}>;
using Shape_WarpTile   = ck_tile::sequence<{params.warp_m}, {params.warp_n}>;
using Shape_ThreadTile = ck_tile::sequence<{params.thread_tile_m}, {params.thread_tile_n}>;

using TestConfig =
    std::tuple<{TYPE_MAP[self.datatype]},
               float,
               {TYPE_MAP[self.output_type]},
               ck_tile::tuple<ck_tile::ReduceOp::Add, ck_tile::ReduceOp::Add>, // Intra block reductions
               ck_tile::tuple<ck_tile::element_wise::PassThrough, ck_tile::element_wise::UnarySquare>, // Elementwise ops
               ck_tile::tuple<ck_tile::element_wise::PassThrough, ck_tile::element_wise::UnaryDivide>, // Accumulator Elementiwise ops, intra block
               ck_tile::tuple<ck_tile::ReduceOp::Add, ck_tile::ReduceOp::Add>, // Inter block reduction
               Shape_BlockWarps,
               Shape_BlockTile,
               Shape_WarpTile,
               Shape_ThreadTile>;

// Register the type(s) for the typed test suite
typedef ::testing::Types<TestConfig> TestTypes;
TYPED_TEST_SUITE({self.test_type}, TestTypes);

TYPED_TEST({self.test_type}, {signature}_{input_shape_str})
{{
    this->Run{signature}({input_shape_arg_str});
}}
"""

        return t


class MultiReduceThreadwiseKernelBuilder(MultiReduceBase):
    def __init__(self, working_path, gpu_target, datatype, config_json=None):
        super().__init__(working_path, gpu_target, datatype, config_json)

        self.name = "multiops_threadwise"

        self.header = "test_multi_reduce2d_threadwise_impl.hpp"
        self.test_type = "TestCkTileMultiReduceThreadwise"


class MultiReduceMultiBlockKernelBuilder(MultiReduceBase):
    def __init__(self, working_path, gpu_target, datatype, config_json=None):
        super().__init__(working_path, gpu_target, datatype, config_json)

        self.name = "multiops_multiblock"

        self.output_type = (
            "float"  # Force float to be used as the output is also used as accumulator
        )

        self.header = "test_multi_reduce2d_multiblock_impl.hpp"
        self.test_type = "TestCkTileMultiReduceMultiblock"


def main(args):
    variants = {
        "multiops_threadwise": {"class": MultiReduceThreadwiseKernelBuilder},
        "multiops_multiblock": {"class": MultiReduceMultiBlockKernelBuilder},
    }
    if not (args.list_blobs or args.gen_blobs):
        raise ValueError("Please provide a list or generate blobs.")

    builder = variants.get(args.variant)
    builder_instance = builder["class"](
        working_path=args.working_path,
        gpu_target=args.gpu_target,
        datatype=args.datatype,
        config_json=args.config_json,
    )

    if args.list_blobs:
        builder_instance.do_list_blobs()
    if args.gen_blobs:
        builder_instance.do_generate_blobs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce Instance Builder")

    parser.add_argument(
        "--working_path", type=str, required=True, help="Working directory path"
    )
    parser.add_argument("--datatype", type=str, required=True, help="Data type")
    parser.add_argument(
        "--variant", type=str, required=True, help="Variant: multiblock or threadwise"
    )
    parser.add_argument(
        "--config_json", type=str, required=True, help="Path to config JSON blob"
    )
    parser.add_argument("--list_blobs", action="store_true", help="List blobs")
    parser.add_argument("--gen_blobs", action="store_true", help="Generate blobs")
    parser.add_argument("--gpu_target", type=str, required=True, help="GPU target")

    args = parser.parse_args()

    main(args)
