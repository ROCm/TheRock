"""
RCCL Benchmark Test

Runs RCCL collective communication benchmarks, collects results, and uploads to results API.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class RCCLBenchmark(BenchmarkBase):
    """RCCL benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="rccl", display_name="RCCL")
        self.log_file = self.script_dir / "rccl_bench.log"
        self.ngpu = self._detect_gpu_count()

        # Validate OpenMPI is available (from base class)
        self._validate_openmpi()

    def run_benchmarks(self) -> None:
        """Run RCCL benchmarks and save output to log file."""
        # Load benchmark configuration
        config_file = self.script_dir.parent / "configs" / "rccl.json"
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Get configuration
        benchmarks = config_data.get("benchmarks")
        data_types = config_data.get("data_types")
        min_size = config_data.get("min_size")
        max_size = config_data.get("max_size")
        step_factor = config_data.get("step_factor")
        warmup_iters = config_data.get("warmup_iters")
        test_iters = config_data.get("test_iters")
        operations = config_data.get("operations")

        # Apply GPU-specific overrides
        if self.amdgpu_families:
            overrides = config_data.get("gpu_overrides", {}).get(
                self.amdgpu_families, {}
            )
            max_size = overrides.get("max_size", max_size)

        log.info("Running RCCL Benchmarks")

        with open(self.log_file, "w+") as f:
            for benchmark in benchmarks:
                bench_binary = Path(self.therock_bin_dir) / benchmark

                if not bench_binary.exists():
                    log.warning(f"Benchmark binary not found: {bench_binary}")
                    continue

                for dtype in data_types:
                    for operation in operations:
                        # Write section header to log file for parsing
                        section_header = f"\n{'='*80}\nBenchmark: {benchmark}\nDataType: {dtype}\nOperation: {operation}\n\n"
                        f.write(section_header)

                        # Build environment variables
                        env_vars = {"HSA_FORCE_FINE_GRAIN_PCIE": "1"}

                        # Construct benchmark command with MPI
                        cmd = [
                            "mpirun",
                            "--np",
                            "1",
                            str(bench_binary),
                            "--minbytes",
                            min_size,
                            "--maxbytes",
                            max_size,
                            "--stepfactor",
                            step_factor,
                            "--ngpus",
                            str(self.ngpu),
                            "--op",
                            operation,
                            "--datatype",
                            dtype,
                            "--iters",
                            test_iters,
                            "--warmup_iters",
                            warmup_iters,
                        ]

                        self.execute_command(cmd, f, env=env_vars)

        log.info("RCCL benchmarks execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse benchmark results from log file.

        Returns:
            tuple: (test_results list, PrettyTable object)
        """
        # Regex patterns for parsing
        pattern_benchmark = re.compile(r"Benchmark:\s*(\S+)")
        pattern_dtype = re.compile(r"DataType:\s*(\S+)")
        pattern_operation = re.compile(r"Operation:\s*(\S+)")
        pattern_bandwidth = re.compile(r"#\s+Avg bus bandwidth\s+:\s+(\d+\.\d+)")

        log.info("Parsing Results")

        # Setup table
        field_names = [
            "TestName",
            "SubTests",
            "nGPU",
            "Result",
            "Scores",
            "Units",
            "Flag",
        ]
        table = PrettyTable(field_names)

        test_results = []

        try:
            with open(self.log_file, "r") as log_fp:
                content = log_fp.read()
            # Split by benchmark sections
            sections = content.split("=" * 80)

            for section in sections:
                if not section.strip():
                    continue

                # Extract metadata
                benchmark_match = re.search(pattern_benchmark, section)
                dtype_match = re.search(pattern_dtype, section)
                operation_match = re.search(pattern_operation, section)
                bandwidth_match = re.search(pattern_bandwidth, section)

                if not (benchmark_match and dtype_match and bandwidth_match):
                    continue

                benchmark_name = benchmark_match.group(1)
                dtype = dtype_match.group(1)
                operation = operation_match.group(1) if operation_match else "sum"
                bandwidth = float(bandwidth_match.group(1))

                # Determine status
                status = "PASS" if bandwidth > 0 else "FAIL"

                # Build subtest name
                subtest_name = f"{benchmark_name}_{dtype}_{operation}"

                # Add to table and results
                table.add_row(
                    [
                        self.benchmark_name,
                        subtest_name,
                        self.ngpu,
                        status,
                        bandwidth,
                        "GB/s",
                        "H",
                    ]
                )

                test_results.append(
                    self.create_test_result(
                        self.benchmark_name,
                        subtest_name,
                        status,
                        bandwidth,
                        "GB/s",
                        "H",
                        ngpu=self.ngpu,
                        dtype=dtype,
                        operation=operation,
                    )
                )

        except OSError as e:
            raise ValueError(f"IO Error in Score Extractor: {e}")

        return test_results, table


if __name__ == "__main__":
    run_benchmark_main(RCCLBenchmark())
