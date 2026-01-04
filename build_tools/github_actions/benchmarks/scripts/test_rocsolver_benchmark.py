"""
ROCsolver Benchmark Test

Runs ROCsolver benchmarks, collects results, and uploads to results API.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class ROCsolverBenchmark(BenchmarkBase):
    """ROCsolver benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="rocsolver", display_name="ROCsolver")
        self.log_file = self.script_dir / "rocsolver_bench.log"
        self.therock_dir = self.script_dir.parent.parent.parent.parent

    def run_benchmarks(self) -> None:
        """Run ROCsolver benchmarks and save output to log file."""
        log.info("Running ROCsolver Benchmarks")

        with open(self.log_file, "w+") as f:
            cmd = [
                f"{self.therock_bin_dir}/rocsolver-bench",
                "-f",
                "gesvd",
                "--precision",
                "d",
                "--left_svect",
                "S",
                "--right_svect",
                "S",
                "-m",
                "250",
                "-n",
                "250",
            ]

            log.info(f"++ Exec [{self.therock_dir}]$ {shlex.join(cmd)}")
            f.write(f"{shlex.join(cmd)}\n")

            process = subprocess.Popen(
                cmd,
                cwd=self.therock_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                log.info(line.strip())
                f.write(f"{line}\n")

            process.wait()

        log.info("Benchmark execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse benchmark results from log file.

        Returns:
            tuple: (test_results list, PrettyTable object)
        """
        # Regex patterns for parsing
        # Pattern to match timing results: "cpu_time_us  gpu_time_us"
        gpu_pattern = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")
        # Pattern to detect device ID
        device_pattern = re.compile(r"Device\s+ID\s*\d+")

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
        score = 0
        num_gpus = 0

        # Test configuration from command
        subtest_name = "rocsolver_gesvd_d_S_S_250_250"

        try:
            with open(self.log_file, "r") as fp:
                for line in fp:
                    # Check for GPU device lines
                    if re.search(device_pattern, line):
                        num_gpus += 1

                    # Extract timing score - try new 2-column format first
                    gpu_match = re.search(gpu_pattern, line)
                    if gpu_match:
                        # Group 2 contains gpu_time_us in new format
                        score = float(gpu_match.group(2))
                        log.debug(
                            f"Matched 2-column format: cpu_time={gpu_match.group(1)}, gpu_time={gpu_match.group(2)}"
                        )

            # Determine status
            if score > 0:
                status = "PASS"
            else:
                status = "FAIL"
                log.warning(f"No valid score extracted from log file. Score = {score}")

            log.info(f"Extracted score: {score} us")

            # Default to 1 GPU if none detected
            if num_gpus == 0:
                num_gpus = 1

            # Add to table
            table.add_row(
                [self.benchmark_name, subtest_name, num_gpus, status, score, "us", "L"]
            )

            # Add to test results
            test_results.append(
                self.create_test_result(
                    self.benchmark_name,
                    subtest_name,
                    status,
                    score,
                    "us",
                    "L",  # Lower is better for time
                    ngpu=num_gpus,
                )
            )

        except OSError as e:
            raise ValueError(f"IO Error in Score Extractor: {e}")

        return test_results, table


if __name__ == "__main__":
    run_benchmark_main(ROCsolverBenchmark())
