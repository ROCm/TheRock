"""
hipBLASLt Benchmark Test

Runs hipBLASLt benchmarks, collects results, and uploads to results API.
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


class HipblasltBenchmark(BenchmarkBase):
    """hipBLASLt benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="hipblaslt", display_name="hipBLASLt")
        self.log_file = self.script_dir / "hipblaslt_bench.log"
        self.therock_dir = self.script_dir.parent.parent.parent.parent

    def run_benchmarks(self) -> None:
        """Run hipBLASLt benchmarks and save output to log file."""
        BETA = 0
        ITERATIONS = 1000
        COLD_ITERATIONS = 1000
        PRECISION = "f16_r"
        COMPUTE_TYPE = "f32_r"
        ACTIVATION_TYPE = "none"

        # Load benchmark configuration
        config_file = self.script_dir.parent / "configs" / "hipblaslt.json"
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Combine test configurations: (shapes_list, transB_value)
        test_configs = [
            (config_data.get("input_shapes", []), "N"),  # transB = N
            (config_data.get("ntinput_shapes", []), "T"),  # transB = T
        ]

        log.info("Running hipBLASLt Benchmarks")

        with open(self.log_file, "w+") as f:
            for shapes_list, transB in test_configs:
                for input_shape in shapes_list:
                    M, N, K, B = input_shape.split()

                    # Calculate matrix strides
                    stride_a = int(M) * int(K)
                    stride_b = int(K) * int(N)
                    stride_c = int(M) * int(N)
                    stride_d = int(M) * int(N)

                    cmd = [
                        f"{self.therock_bin_dir}/hipblaslt-bench",
                        "-v",
                        "--transA",
                        "N",
                        "--transB",
                        transB,
                        "-m",
                        M,
                        "-n",
                        N,
                        "-k",
                        K,
                        "--alpha",
                        "1",
                        "--lda",
                        M,
                        "--stride_a",
                        str(stride_a),
                        "--beta",
                        str(BETA),
                        "--ldb",
                        K,
                        "--stride_b",
                        str(stride_b),
                        "--ldc",
                        M,
                        "--stride_c",
                        str(stride_c),
                        "--ldd",
                        M,
                        "--stride_d",
                        str(stride_d),
                        "--precision",
                        PRECISION,
                        "--compute_type",
                        COMPUTE_TYPE,
                        "--activation_type",
                        ACTIVATION_TYPE,
                        "--iters",
                        str(ITERATIONS),
                        "--cold_iters",
                        str(COLD_ITERATIONS),
                        "--batch_count",
                        B,
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
        log.info("Parsing Results")

        field_names = [
            "TestName",
            "SubTests",
            "BatchCount",
            "nGPU",
            "Result",
            "Scores",
            "Units",
            "Flag",
        ]
        table = PrettyTable(field_names)

        test_results = []
        num_gpus = 1

        def get_param(params_dict: Dict[str, str], key: str, default: str = "") -> str:
            """Extract and strip parameter value from params dictionary."""
            return params_dict.get(key, default).strip()

        def create_subtest_name(params: Dict[str, str], batch_count: int) -> str:
            """Create comprehensive subtest name from parameters."""
            return (
                f"gemm_{get_param(params, 'transA')}"
                f"_{get_param(params, 'transB')}"
                f"_{get_param(params, 'm')}"
                f"_{get_param(params, 'n')}"
                f"_{get_param(params, 'k')}"
                f"_{get_param(params, 'alpha')}"
                f"_{get_param(params, 'lda')}"
                f"_{get_param(params, 'stride_a')}"
                f"_{get_param(params, 'beta')}"
                f"_{get_param(params, 'ldb')}"
                f"_{get_param(params, 'stride_b')}"
                f"_{get_param(params, 'ldc')}"
                f"_{get_param(params, 'stride_c')}"
                f"_{get_param(params, 'ldd')}"
                f"_{get_param(params, 'stride_d')}"
                f"_{get_param(params, 'a_type')}"
                f"_{get_param(params, 'compute_type')}"
                f"_{get_param(params, 'activation_type')}"
                f"_{batch_count}"
            )

        try:
            with open(self.log_file, "r") as log_fp:
                data = log_fp.readlines()

            # Find CSV header line
            header_line = None
            header_index = -1

            for i, line in enumerate(data):
                if "transA" in line and "transB" in line and "hipblaslt-Gflops" in line:
                    header_line = line.replace("[0]:", "").strip().split(",")
                    header_index = i
                    break

            if not header_line or header_index == -1:
                log.warning("CSV header not found in log file")
                return test_results, table

            for line in data[header_index + 1 :]:
                line = line.strip()

                # Skip empty or header lines
                if (
                    not line
                    or len(line.split(",")) < 2
                    or "transA" in line
                    or "transB" in line
                ):
                    continue

                # Remove [0]: prefix and parse values
                line = re.sub(r"^\[\d+\]:\s*", "", line)
                values = line.split(",")

                if len(values) != len(header_line):
                    continue

                params = dict(zip(header_line, values))

                # Validate batch_count
                try:
                    batch_count = int(get_param(params, "batch_count", "0") or "0")
                except (ValueError, TypeError):
                    log.warning(f"Invalid batch_count, skipping line")
                    continue

                # Validate Gflops score
                try:
                    score = float(get_param(params, "hipblaslt-Gflops", "0"))
                    status = "PASS" if score > 0 else "FAIL"
                except (ValueError, TypeError):
                    score = 0.0
                    status = "FAIL"

                # Create subtest name
                subtest_name = create_subtest_name(params, batch_count)

                table.add_row(
                    [
                        self.benchmark_name,
                        subtest_name,
                        batch_count,
                        num_gpus,
                        status,
                        score,
                        "Gflops",
                        "H",
                    ]
                )
                test_results.append(
                    self.create_test_result(
                        self.benchmark_name,
                        subtest_name,
                        status,
                        score,
                        "Gflops",
                        "H",
                        batch_size=batch_count,
                        ngpu=num_gpus,
                    )
                )

        except FileNotFoundError:
            log.error(f"Log file not found: {self.log_file}")
        except OSError as e:
            log.error(f"Failed to read log file: {e}")
            raise

        return test_results, table


if __name__ == "__main__":
    run_benchmark_main(HipblasltBenchmark())
