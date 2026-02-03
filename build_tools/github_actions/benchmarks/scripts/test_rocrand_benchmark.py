"""
ROCrand Benchmark Test

Runs ROCrand benchmarks, collects results, and uploads to results API.
"""

import csv
import io
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class ROCrandBenchmark(BenchmarkBase):
    """ROCrand benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="rocrand", display_name="ROCrand")
        self.therock_dir = self.script_dir.parent.parent.parent.parent
        self.bench_bins = ["benchmark_rocrand_host_api", "benchmark_rocrand_device_api"]

    def run_benchmarks(self) -> None:
        """Run ROCrand benchmarks and save output to log files."""
        NUM_TRIALS = 1000  # Number of benchmark trials

        log.info("Running ROCrand Benchmarks")

        for bench_bin in self.bench_bins:
            # Extract benchmark type from binary name
            match = re.search(r"benchmark_(.*?)_api", bench_bin)
            if not match:
                log.warning(f"Could not parse benchmark name from: {bench_bin}")
                continue

            bench_type = match.group(1)
            log_file = self.script_dir / f"{bench_type}_bench.log"

            # Run benchmark
            with open(log_file, "w+") as f:
                cmd = [
                    f"{self.therock_bin_dir}/{bench_bin}",
                    "--trials",
                    str(NUM_TRIALS),
                    "--benchmark_color=false",
                    "--benchmark_format=csv",
                ]

                self.execute_command(cmd, f)

        log.info("Benchmark execution complete")

    def parse_results(self) -> List[Dict[str, Any]]:
        """Parse benchmark results from log files.

        Note: rocrand benchmarks support --benchmark_format=csv/json/console.
        Currently using CSV format (--benchmark_format=csv).

        Returns:
            List[Dict[str, Any]]: test_results list
        """
        log.info("Parsing Results")

        # Regex pattern to match CSV section in benchmark output
        csv_pattern = re.compile(
            r"^engine,distribution,mode,name,iterations,real_time,cpu_time,time_unit,bytes_per_second,throughput_gigabytes_per_second,lambda,items_per_second,label,error_occurred,error_message\n(?:[^\n]*\n)+$",
            re.MULTILINE,
        )

        bench_types = ["rocrand_host", "rocrand_device"]

        test_results = []

        for bench_type in bench_types:
            log_file = self.script_dir / f"{bench_type}_bench.log"

            if not log_file.exists():
                log.warning(f"Log file not found: {log_file}")
                continue

            log.info(f"Parsing {bench_type} results")

            with open(log_file, "r") as f:
                data = f.read()

            # Find the CSV data in the file
            csv_match = csv_pattern.search(data)
            if not csv_match:
                log.warning(f"No CSV data found in {log_file}")
                continue

            csv_data = csv_match.group()
            lines = csv_data.strip().split("\n")

            # Parse CSV data
            csv_reader = csv.DictReader(io.StringIO("\n".join(lines)))

            for row in csv_reader:
                engine = row.get("engine", "")
                distribution = row.get("distribution", "")
                mode = row.get("mode", "")
                throughput = row.get("throughput_gigabytes_per_second", "0")

                try:
                    throughput_val = float(throughput)
                except (ValueError, TypeError):
                    log.warning(f"Invalid throughput value: {throughput}, skipping")
                    continue

                # Build subtest identifier
                subtest_id = f"{engine}_{distribution}"

                # Determine status
                status = "PASS" if throughput_val > 0 else "FAIL"

                test_results.append(
                    self.create_test_result(
                        self.benchmark_name,
                        subtest_id,
                        status,
                        throughput_val,
                        "GB/s",
                        "H",
                        mode=mode,
                    )
                )

        return test_results


if __name__ == "__main__":
    run_benchmark_main(ROCrandBenchmark())
