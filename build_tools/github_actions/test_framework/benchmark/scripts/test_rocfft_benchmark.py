"""
ROCfft Benchmark Test

Runs ROCfft benchmarks, collects results, and uploads to results API.
"""

import json
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


class ROCfftBenchmark(BenchmarkBase):
    """ROCfft benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="rocfft", display_name="ROCfft")
        self.log_file = self.script_dir / "rocfft_bench.log"
        self.therock_dir = self.script_dir.parent.parent.parent.parent

    def run_benchmarks(self) -> None:
        """Run ROCfft benchmarks and save output to log file."""
        DEFAULT_BATCH_SIZE = 10  # Default batch size for benchmarks
        NUM_ITERATIONS = 20  # Number of benchmark iterations

        # Load benchmark configuration
        config_file = self.script_dir.parent / "configs" / "rocfft.json"
        with open(config_file, "r") as f:
            data = json.load(f)

        test_cases = data.get("generic", [])
        if self.amdgpu_families:
            test_cases.extend(data.get(self.amdgpu_families, []))

        log.info("Running ROCfft Benchmarks")

        with open(self.log_file, "w+") as f:
            for test_case in test_cases:
                # Extract batch size from test case string (if specified)
                pattern_batch_size = re.compile(r"-b\s+(\d+)")
                explicit_batch = re.search(pattern_batch_size, test_case)

                if explicit_batch:
                    batch_size = int(explicit_batch.group(1))
                    cleaned_case = re.sub(r"-b\s+\d+", "", test_case)
                else:
                    batch_size = DEFAULT_BATCH_SIZE
                    cleaned_case = test_case

                cmd = [
                    f"{self.therock_bin_dir}/rocfft-bench",
                    "--length",
                    *cleaned_case.split(),
                    "-b",
                    str(batch_size),
                    "-N",
                    str(NUM_ITERATIONS),
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
        default_batch_size = 10

        # Regex patterns for parsing
        pattern_test_case = re.compile(r"\s*--(length)\s*(\d+.*)")
        pattern_gpu_time = re.compile(r"(\s*Execution gpu time:\s*)(\s*.*)")
        pattern_gflops = re.compile(r"(\s*Execution gflops:\s*)(\s*.*)")
        pattern_batch_size = re.compile(r"-b\s+(\d+)")

        log.info("Parsing Results")

        # Setup table
        field_names = [
            "TestName",
            "SubTests",
            "BatchSize",
            "nGPU",
            "Result",
            "Scores",
            "Units",
            "Flag",
        ]
        table = PrettyTable(field_names)

        test_results = []
        num_gpus = 1
        batch_size = default_batch_size

        try:
            with open(self.log_file, "r") as log_fp:

                for line in log_fp:
                    # Extract batch size from command line
                    batch_match = re.search(pattern_batch_size, line)
                    if batch_match:
                        batch_size = int(batch_match.group(1))

                    # Check if this is a test case line
                    test_case_match = re.search(pattern_test_case, line)
                    if not test_case_match:
                        continue

                    # Build subtest identifier
                    length_type = test_case_match.group(1)
                    dimensions = (
                        test_case_match.group(2).replace(" ", "_").replace("-", "")
                    )
                    subtest_id = f"{length_type}={dimensions}"

                    # Parse test results
                    gpu_time = None
                    gflops = None

                    for result_line in log_fp:
                        if re.search(pattern_gpu_time, result_line):
                            gpu_time = float(result_line.split()[-2])
                        elif re.search(pattern_gflops, result_line):
                            gflops = float(result_line.split()[-1])
                            break  # Found both metrics
                        elif "--length" in result_line:
                            # Next test case started, this one failed
                            break

                    # Determine if test passed or failed
                    status = "PASS" if (gpu_time and gflops) else "FAIL"
                    gpu_time = gpu_time or 0.0
                    gflops = gflops or 0.0

                    # Add GPU time result
                    time_testname = f"rider_{subtest_id}_time"
                    table.add_row(
                        [
                            self.benchmark_name,
                            time_testname,
                            batch_size,
                            num_gpus,
                            status,
                            gpu_time,
                            "ms",
                            "L",
                        ]
                    )
                    test_results.append(
                        self.create_test_result(
                            self.benchmark_name,
                            time_testname,
                            status,
                            gpu_time,
                            "ms",
                            "L",
                            batch_size=batch_size,
                            ngpu=num_gpus,
                        )
                    )

                    # Add GFLOPS result
                    gflops_testname = f"rider_{subtest_id}_gflops"
                    table.add_row(
                        [
                            self.benchmark_name,
                            gflops_testname,
                            batch_size,
                            num_gpus,
                            status,
                            gflops,
                            "GFLOPS",
                            "H",
                        ]
                    )
                    test_results.append(
                        self.create_test_result(
                            self.benchmark_name,
                            gflops_testname,
                            status,
                            gflops,
                            "GFLOPS",
                            "H",
                            batch_size=batch_size,
                            ngpu=num_gpus,
                        )
                    )

        except OSError as e:
            raise ValueError(f"IO Error in Score Extractor: {e}")

        return test_results, table


if __name__ == "__main__":
    run_benchmark_main(ROCfftBenchmark())
