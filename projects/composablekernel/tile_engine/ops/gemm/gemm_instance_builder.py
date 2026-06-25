# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import json
from pathlib import Path
import importlib.util
import itertools
import logging


def _import_validation_utils():
    """Import validation utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "validation_utils",
        os.path.join(parent_dir, "gemm", "gemm_validation_utils.py"),
    )
    validation_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_utils)

    return validation_utils


# Import validation functions
_validation_utils = _import_validation_utils()
is_tile_config_valid = _validation_utils.is_tile_config_valid
is_trait_combination_valid = _validation_utils.is_trait_combination_valid
get_abc_layouts = _validation_utils.get_abc_layouts
get_abcd_layouts = _validation_utils.get_abcd_layouts
get_dtype_string = _validation_utils.get_dtype_string


class GemmKernelBuilder:
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json=None,
    ):
        self.kernel_name_prefix = kernel_name_prefix
        self.working_path = Path(working_path)
        self.gpu_target = gpu_target
        self.datatype = datatype
        self.layout = layout
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)

    def _list_kernels(self):
        """Write kernel list to file for CMake to read (with comprehensive validation)"""
        # Get configurations using comprehensive validation
        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                (
                    pipeline,
                    epilogue,
                    scheduler,
                    pad_m,
                    pad_n,
                    pad_k,
                    persistent,
                ) = trait_combo

                # Create kernel name with proper boolean capitalization
                kernel_name = f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

                # Create tile configuration string
                tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

                kernel_name += f"_{tile_str}"

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                    }
                )

        # Write kernel count
        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_count.txt", "w"
        ) as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_list.txt", "w"
        ) as f:
            for kernel in kernel_list:
                # Format: kernel_name|tile_config|trait_combo
                tile_config = kernel["tile_config"]
                trait_combo = kernel["trait_combo"]

                tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

                trait_str = (
                    f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_"
                    + "_".join(str(x) for x in trait_combo[3:])
                )

                f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _get_tile_configs(self):
        """Get tile configurations for the current datatype and layout"""

        tile_config = self.config["tile_config"]

        # Generate values in the config if default range is given
        if tile_config.get("tile_m").get("values") is None:
            tile_config.get("tile_m")["values"] = self._generate_values(
                tile_config.get("tile_m").get("min"),
                tile_config.get("tile_m").get("max"),
                tile_config.get("tile_m").get("step"),
            )
        if tile_config.get("tile_n").get("values") is None:
            tile_config.get("tile_n")["values"] = self._generate_values(
                tile_config.get("tile_n").get("min"),
                tile_config.get("tile_n").get("max"),
                tile_config.get("tile_n").get("step"),
            )
        if tile_config.get("tile_k").get("values") is None:
            tile_config.get("tile_k")["values"] = self._generate_values(
                tile_config.get("tile_k").get("min"),
                tile_config.get("tile_k").get("max"),
                tile_config.get("tile_k").get("step"),
            )

        # Get all possible values for each parameter
        tile_m_values = tile_config.get("tile_m").get("values")
        tile_n_values = tile_config.get("tile_n").get("values")
        tile_k_values = tile_config.get("tile_k").get("values")
        warp_m_values = tile_config.get("warp_m").get("values")
        warp_n_values = tile_config.get("warp_n").get("values")
        warp_k_values = tile_config.get("warp_k").get("values")
        warp_tile_m_values = tile_config.get("warp_tile_m").get("values")
        warp_tile_n_values = tile_config.get("warp_tile_n").get("values")
        warp_tile_k_values = tile_config.get("warp_tile_k").get("values")

        # Generate all combinations
        default_pipeline = ""
        if self.kernel_name_prefix == "gemm_universal":
            default_pipeline = "compv4"
        elif self.kernel_name_prefix == "gemm_multi_d":
            default_pipeline = "compv4"
        elif self.kernel_name_prefix == "gemm_preshuffle":
            default_pipeline = "preshufflev2"

        configs = []
        for tile_m in tile_m_values:
            for tile_n in tile_n_values:
                for tile_k in tile_k_values:
                    for warp_m in warp_m_values:
                        for warp_n in warp_n_values:
                            for warp_k in warp_k_values:
                                for warp_tile_m in warp_tile_m_values:
                                    for warp_tile_n in warp_tile_n_values:
                                        for warp_tile_k in warp_tile_k_values:
                                            # Validate configuration
                                            if self._validate_tile_config(
                                                tile_m,
                                                tile_n,
                                                tile_k,
                                                warp_m,
                                                warp_n,
                                                warp_k,
                                                warp_tile_m,
                                                warp_tile_n,
                                                warp_tile_k,
                                                default_pipeline,
                                            ):
                                                configs.append(
                                                    {
                                                        "tile_m": tile_m,
                                                        "tile_n": tile_n,
                                                        "tile_k": tile_k,
                                                        "warp_m": warp_m,
                                                        "warp_n": warp_n,
                                                        "warp_k": warp_k,
                                                        "warp_tile_m": warp_tile_m,
                                                        "warp_tile_n": warp_tile_n,
                                                        "warp_tile_k": warp_tile_k,
                                                    }
                                                )
        return configs

    def _generate_values(self, min_val, max_val, step):
        """Generate a list of values from min to max with the given step"""
        values = []
        val = min_val
        while val <= max_val:
            values.append(val)
            val += step
        return values

    def _validate_tile_config(
        self,
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        pipeline,
    ):
        """Validate that tile configuration is reasonable"""
        # Validate preshuffle specific constraints
        if (
            self.config.get("permute_n") is not None
            and self.config.get("permute_n") is True
        ):
            valid = (tile_n / warp_tile_n / warp_n) % 2 == 0
            if not valid:
                return False

        # Determine data types for validation
        a_datatype = self.datatype
        b_datatype = self.datatype
        c_datatype = self.datatype

        layout = self.layout

        # Special handling for certain data types
        if self.datatype in ["fp8", "bf8"]:
            c_datatype = "fp16"

        # Use the comprehensive validation function
        return is_tile_config_valid(
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
            self.gpu_target,
        )

    def _generate_trait_combinations(self):
        """Generate all combinations of traits"""

        trait_config = self.config["trait_config"]

        pipelines = trait_config.get("pipeline").get("values")
        epilogues = trait_config.get("epilogue").get("values")
        schedulers = trait_config.get("scheduler").get("values")
        pad_m_values = trait_config.get("pad_m").get("values")
        pad_n_values = trait_config.get("pad_n").get("values")
        pad_k_values = trait_config.get("pad_k").get("values")
        persistent_values = trait_config.get("persistent").get("values")

        all_combinations = list(
            itertools.product(
                pipelines,
                epilogues,
                schedulers,
                pad_m_values,
                pad_n_values,
                pad_k_values,
                persistent_values,
            )
        )

        # Filter out unsupported trait combinations
        combinations = []
        for combo in all_combinations:
            pipeline, epilogue, scheduler = combo[:3]
            if is_trait_combination_valid(pipeline, epilogue, scheduler):
                combinations.append(combo)
            else:
                logging.debug(
                    f"Skipping unsupported trait combination: {pipeline}-{epilogue}-{scheduler}"
                )
        return combinations

    def _generate_kernel_instance(self, tile_config, trait_combo):
        """Generate a single kernel instance"""

        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            persistent,
        ) = trait_combo

        # Create kernel name with proper boolean capitalization
        kernel_name = f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

        # Create tile configuration string
        tile_str = (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
        )
        tile_str += (
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
        )
        tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

        kernel_name += f"_{tile_str}"

        if self.kernel_name_prefix in ["gemm_universal", "gemm_multi_d"]:
            # Map pipeline names to the correct pipeline implementation
            pipeline_impl_map = {
                "mem": "ck_tile::GemmPipelineAgBgCrMem",
                "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
                "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
            }
            # Map pipeline names to base pipeline for hot loop detection
            base_pipeline_map = {
                "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
                "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
                "compv4": "ck_tile::BaseGemmPipelineAgBgCrCompV4",
            }
        elif self.kernel_name_prefix == "gemm_preshuffle":
            # Map pipeline names to the correct pipeline implementation
            pipeline_impl_map = {
                "preshufflev2": "ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2",
            }
            # Map pipeline names to base pipeline for hot loop detection
            base_pipeline_map = {
                "preshufflev2": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
            }

        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        instance_code = self.populate_kernel_header(kernel_name)
        instance_code += self.populate_kernel_dtype_layout()
        instance_code += self.populate_strut_begin(kernel_name)
        instance_code += self.populate_tile_config(tile_config)
        instance_code += self.populate_trait_config(trait_combo)
        instance_code += self.populate_initialization(base_pipeline_map, pipeline)
        instance_code += self.populate_launch(
            scheduler_type_map,
            scheduler,
            pipeline_impl_map,
            pipeline,
            epilogue,
            k_block_per_cu,
            persistent,
        )

        # Write into a file
        simplified_name = kernel_name
        if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
            simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

        header_file = (
            self.working_path
            / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

        return kernel_name, instance_code

    def populate_kernel_header(self, kernel_name):
        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
"""
        return instance_code

    def populate_kernel_dtype_layout(self):
        # Determine accumulator type based on datatype
        acc_type = "float"

        # Determine output type
        c_type = self.datatype
        if self.datatype in ["fp8", "bf8"]:
            c_type = "fp16"

        # Assign layouts based on self.layout
        if self.kernel_name_prefix == "gemm_multi_d":
            a_layout, b_layout, c_layout, ds_layout = get_abcd_layouts(self.layout)
        elif (
            self.kernel_name_prefix == "gemm_universal"
            or self.kernel_name_prefix == "gemm_preshuffle"
        ):
            a_layout, b_layout, c_layout = get_abc_layouts(self.layout)

        instance_code = f"""
using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
using AccDataType = {acc_type};
using CDataType = {get_dtype_string(c_type)};"""

        if self.kernel_name_prefix == "gemm_multi_d":
            instance_code += f"""
using D0DataType = {get_dtype_string(self.datatype)};
using D1DataType = {get_dtype_string(self.datatype)};
using DsDataType = ck_tile::tuple<D0DataType, D1DataType>;"""

        instance_code += f"""
using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};
"""
        if self.kernel_name_prefix == "gemm_multi_d":
            instance_code += f"""
using D0Layout = {ds_layout[0]};
using D1Layout = {ds_layout[1]};
using DsLayout = ck_tile::tuple<D0Layout, D1Layout>;

using ElementWiseFn = ck_tile::element_wise::{self.elementwise_function};"""

        return instance_code

    def populate_strut_begin(self, kernel_name):
        instance_code = f"""
// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";

// Wrapper for simplified launch interface
struct SelectedKernel {{
    """
        return instance_code

    def populate_tile_config(self, tile_config):
        instance_code = f"""// Tile configuration
    static constexpr ck_tile::index_t BlockSize = 256;
    static constexpr ck_tile::index_t TileM = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t TileN = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t TileK = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t WarpTileM = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t WarpTileK = {tile_config["warp_tile_k"]};"""
        return instance_code

    def populate_trait_config(self, trait_combo):
        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            persistent,
        ) = trait_combo

        instance_code = f"""
    
    // Traits configurations
    static constexpr bool kPadM = {"true" if pad_m in [True, "true"] else "false"};
    static constexpr bool kPadN = {"true" if pad_n in [True, "true"] else "false"};
    static constexpr bool kPadK = {"true" if pad_k in [True, "true"] else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool DoubleSmemBuffer = {"true" if pipeline in ["compv4", "preshufflev2"] else "false"};"""

        if self.kernel_name_prefix in ["gemm_universal", "gemm_preshuffle"]:
            instance_code += f"""    
    static constexpr bool UsePersistentKernel = {"true" if persistent in [True, "true"] else "false"};
    static constexpr bool UseStructuredSparsity = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;"""

            if self.kernel_name_prefix == "gemm_preshuffle":
                instance_code += f"""
    static constexpr bool Preshuffle = true;
    static constexpr bool PermuteN     = {"true" if self.config.get("permute_n") else "false"};"""
            else:
                instance_code += """
    static constexpr bool Preshuffle = false;"""
        return instance_code

    def populate_initialization(self, base_pipeline_map, pipeline):
        # Tile Shape
        if self.kernel_name_prefix == "gemm_multi_d":
            instance_code = """  

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;"""

        elif self.kernel_name_prefix in ["gemm_universal", "gemm_preshuffle"]:
            instance_code = """
            
    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;"""

        # Tile partitioner
        instance_code += """

    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;"""

        # Traits
        if self.kernel_name_prefix == "gemm_multi_d":
            instance_code += """

    // Traits
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;"""
        elif self.kernel_name_prefix == "gemm_preshuffle":
            instance_code += """

    // Traits
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, NumWaveGroups>;"""

        # Pipeline problem
        if self.kernel_name_prefix in ["gemm_preshuffle", "gemm_multi_d"]:
            instance_code += """

    // Pipeline problem
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        TileShape,
        Traits>;"""

        # Base pipeline for hot loop detection
        if self.kernel_name_prefix == "gemm_preshuffle":
            instance_code += f"""

    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline, "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2")}<GemmPipelineProblem>;"""

        elif self.kernel_name_prefix == "gemm_multi_d":
            instance_code += f"""

    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline)}<GemmPipelineProblem>;"""

        return instance_code

    def populate_launch(
        self,
        scheduler_type_map,
        scheduler,
        pipeline_impl_map,
        pipeline,
        epilogue,
        k_block_per_cu,
        persistent,
    ):
        # Function Signature
        if self.kernel_name_prefix == "gemm_multi_d":
            instance_code = """

    // Launch function
    static float launch(const ck_tile::GemmMultiDHostArgs<DsDataType::size()>&  args, const ck_tile::stream_config& stream) {"""
        elif self.kernel_name_prefix in ["gemm_universal", "gemm_preshuffle"]:
            instance_code = """

    // Launch function
    static float launch(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {"""

        # Scheduler initialization
        if self.kernel_name_prefix in ["gemm_preshuffle", "gemm_multi_d"]:
            instance_code += f"""

        constexpr auto scheduler = {scheduler_type_map.get(scheduler)};"""

        # Problem Initialization
        if self.kernel_name_prefix == "gemm_preshuffle":
            instance_code += """

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, CLayout, TransposeC,
                                                UseStructuredSparsity, UsePersistentKernel,
                                            NumWaveGroups, Preshuffle>,
                scheduler>;"""
        elif self.kernel_name_prefix == "gemm_multi_d":
            instance_code += """

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, CLayout, TransposeC>,
                scheduler>;"""

        # GemmPipeline
        if self.kernel_name_prefix in ["gemm_preshuffle", "gemm_multi_d"]:
            instance_code += f"""
            
        using GemmPipeline = {pipeline_impl_map.get(pipeline)}<UniversalGemmProblem>;"""

        # Runfunction body
        instance_code += """

        const auto Run = [&](const auto memory_operation_) {"""

        # Scheduler initialization
        if self.kernel_name_prefix in ["gemm_universal"]:
            instance_code += f"""
            constexpr auto scheduler = {scheduler_type_map.get(scheduler)};"""

        # Memory operation
        instance_code += """
            [[maybe_unused]] constexpr auto memory_operation = memory_operation_.value;"""

        # UniversalGemmProblem
        if self.kernel_name_prefix in ["gemm_universal"]:
            instance_code += """

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, CLayout, TransposeC,
                                                UseStructuredSparsity, UsePersistentKernel,
                                                NumWaveGroups, Preshuffle>,
                scheduler>;"""

        # GemmPipeline
        if self.kernel_name_prefix in ["gemm_universal"]:
            instance_code += f"""

            using GemmPipeline = {pipeline_impl_map.get(pipeline)}<UniversalGemmProblem>;"""

        # Epilogue
        instance_code += self.populate_epilogue(epilogue)

        # Kernel type
        if self.kernel_name_prefix == "gemm_multi_d":
            instance_code += """
            
            // Kernel type
            using GemmKernelMultiD = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
            
            // Kernel arguments
            auto kargs = GemmKernelMultiD::MakeKernelArgs(args);
            
            if (!GemmKernelMultiD::IsSupportedArgument(kargs)) {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
            }

            // Get grid and block sizes
            const dim3 grids = GemmKernelMultiD::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = GemmKernelMultiD::BlockSize();
            
            if(stream.log_level_ > 0) {
                std::cout << "Launching kernel with args: " << GemmKernelMultiD::GetName() << '\\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                          << std::endl;
            }"""

            instance_code += f"""    
                // Launch kernel
                constexpr int kBlockPerCu = {k_block_per_cu};
                float ave_time = ck_tile::launch_kernel(
                    stream,
                    ck_tile::make_kernel<kBlockPerCu>(GemmKernelMultiD{{}}, grids, blocks, 0, kargs));
                
                return ave_time;
            }};"""

        elif self.kernel_name_prefix in ["gemm_universal", "gemm_preshuffle"]:
            instance_code += f"""
            
        // Kernel type
        using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            
            // Kernel arguments
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
            }}

            // Get grid and block sizes
            const dim3 grids = {"GemmKernel::MaxOccupancyGridSize(stream)" if persistent in [True, "true"] else "GemmKernel::GridSize(args.M, args.N, args.k_batch)"};
            const dim3 blocks = GemmKernel::BlockSize();
            
            if(stream.log_level_ > 0) {{
                std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\\n'
                          << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }}"""

            instance_code += f"""    
                // Launch kernel
                constexpr int kBlockPerCu = {k_block_per_cu};
                float ave_time = ck_tile::launch_kernel(
                    stream,
                    ck_tile::make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
                
                return ave_time;
            }};"""

        # Run SplitK handler

        instance_code += """

        float ave_time = 0.f;
        if(args.k_batch == 1) {
            ave_time = Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                        ck_tile::memory_operation_enum::set>{});
        } else {
            ave_time = Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                        ck_tile::memory_operation_enum::atomic_add>{});
        }
        return ave_time;
    }
};
"""
        return instance_code

    def populate_epilogue(self, epilogue):
        instance_code = """

            // Epilogue
            """

        if epilogue == "cshuffle":
            if self.kernel_name_prefix == "gemm_universal":
                instance_code += self.populate_cshuffle_gemm_universal()
            elif self.kernel_name_prefix == "gemm_multi_d":
                instance_code += self.populate_cshuffle_gemm_multi_d()
            elif self.kernel_name_prefix == "gemm_preshuffle":
                instance_code += self.populate_cshuffle_gemm_preshuffle()
        else:  # default epilogue
            if self.kernel_name_prefix == "gemm_universal":
                instance_code += self.populate_default_gemm_universal()
            elif self.kernel_name_prefix == "gemm_multi_d":
                instance_code += self.populate_default_gemm_multi_d()
            elif self.kernel_name_prefix == "gemm_preshuffle":
                instance_code += self.populate_default_gemm_preshuffle()

        return instance_code

    def populate_cshuffle_gemm_universal(self):
        instance_code = """            
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TileM,  // kM_
                TileN,  // kN_
                WarpPerBlock_M,              // MWave_
                WarpPerBlock_N,              // NWave_
                WarpTileM,                   // MPerXdl_
                WarpTileN,                   // NPerXdl_
                WarpTileK,                   // KPerXdl_
                TransposeC,                  // isCTransposed_
                NumWaveGroups>;              // kNumWaveGroups_
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        return instance_code

    def populate_cshuffle_gemm_multi_d(self):
        instance_code = """            
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType,
                BDataType,
                DsDataType,
                AccDataType,
                CDataType,
                DsLayout,
                CLayout,
                ElementWiseFn,
                TileM,  // kM_
                TileN,  // kN_
                WarpPerBlock_M,              // MWave_
                WarpPerBlock_N,              // NWave_
                WarpTileM,                   // MPerXdl_
                WarpTileN,                   // NPerXdl_
                WarpTileK,                   // KPerXdl_
                TransposeC>;                  // isCTransposed_
       
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        return instance_code

    def populate_cshuffle_gemm_preshuffle(self):
        instance_code = """            
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TileM,  // kM_
                TileN,  // kN_
                WarpPerBlock_M,              // MWave_
                WarpPerBlock_N,              // NWave_
                WarpTileM,                   // MPerXdl_
                WarpTileN,                   // NPerXdl_
                WarpTileK,                   // KPerXdl_
                TransposeC,                  // isCTransposed_
                NumWaveGroups,               // kNumWaveGroups_
                false,                       // FixedVectorSize_
                1,                           // VectorSizeC_
                PermuteN>;                   // isPermuteN_
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        return instance_code

    def populate_default_gemm_universal(self):
        instance_code = """            
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TileM,  // kM_
                TileN,  // kN_
                kPadM,
                kPadN,
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_
            
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""
        return instance_code

    def populate_default_gemm_multi_d(self):
        instance_code = """            
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                DsDataType,
                AccDataType,
                CDataType,
                DsLayout,
                CLayout,
                ElementWiseFn,
                TileM,  // kM_
                TileN,  // kN_
                kPadM,
                kPadN,
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_
            
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""
        return instance_code

    def populate_default_gemm_preshuffle(self):
        instance_code = """            
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TileM,  // kM_
                TileN,  // kN_
                kPadM,
                kPadN,
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_
            
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""
        return instance_code

    def _generate_cmake_individual_targets(self, kernel_list):
        """Generate CMake include file that creates individual targets"""
        cmake_code = f"""# Generated CMake file for individual {self.kernel_name_prefix} targets
        # Datatype: {self.datatype}, Layout: {self.layout}
        """

        for kernel_name, trait_combo, tile_config in kernel_list:
            pipeline, epilogue, scheduler = trait_combo[:3]

            # Format tile config for CMake function
            tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
            tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
            tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

            trait_str = f"{pipeline}_{epilogue}_{scheduler}_" + "_".join(
                str(x) for x in trait_combo[3:]
            )

            cmake_code += f'create_individual_{self.kernel_name_prefix}_target("{self.datatype}" "{self.layout}" "{trait_str}" "{tile_str}")\n'

        # Write CMake include file
        with open(
            self.working_path / f"{self.kernel_name_prefix}_individual_targets.cmake",
            "w",
        ) as f:
            f.write(cmake_code)
