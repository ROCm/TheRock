"""
ROCblas Benchmark Test

Runs ROCblas benchmarks, collects results, and uploads to results API.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, IO
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class ROCblasBenchmark(BenchmarkBase):
    """ROCblas benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="rocblas", display_name="ROCblas")

    def run_benchmarks(self) -> None:
        """Run ROCblas benchmarks and save output to log files."""
        config_file = self.script_dir.parent / "configs" / "rocblas.json"
        with open(config_file) as f:
            config_data = json.load(f)

        benchmark_config = config_data.get("benchmark_config", {})
        iterations = benchmark_config.get("iterations", 1000)
        cold_iterations = benchmark_config.get("cold_iterations", 1000)
        benchmarks = benchmark_config.get("benchmarks", [])

        log.info("Running ROCblas Benchmarks")

        # Run each benchmark suite based on configuration
        for bench_meta in benchmarks:
            bench_config = config_data.get(bench_meta["name"], {})
            self._run_benchmark(bench_meta, bench_config, iterations, cold_iterations)

        log.info("ROCblas benchmarks execution complete")

    def _build_base_cmd(self, function: str, precision: str) -> List[str]:
        """Build base rocblas-bench command with common arguments."""
        return [
            f"{self.therock_bin_dir}/rocblas-bench",
            "--function",
            function,
            "--precision",
            precision,
            "--initialization",
            "rand_int",
        ]

    def _run_benchmark(
        self,
        bench_meta: Dict[str, Any],
        bench_config: Dict[str, Any],
        iterations: int,
        cold_iterations: int,
    ) -> None:
        """Generic benchmark runner for any ROCblas benchmark type.

        Args:
            bench_meta: Benchmark metadata (name, function, dimensions, etc.)
            bench_config: Benchmark-specific configuration (sizes, precision, etc.)
            iterations: Number of iterations
            cold_iterations: Number of cold iterations
        """
        try:
            name = bench_meta["name"]
            function = bench_meta["function"]
            log_file = self.script_dir / bench_meta["log_file"]
            dimensions = bench_meta.get("dimensions", "sizes")
            extra_args = bench_meta.get("extra_args", {})
            has_compute_type = bench_meta.get("has_compute_type", False)

            log.info(f"Running rocBLAS-{name.upper()} Benchmarks")

            # Apply GPU-specific overrides
            if self.amdgpu_families and "gpu_overrides" in bench_config:
                overrides = bench_config.get("gpu_overrides", {}).get(
                    self.amdgpu_families, {}
                )
                bench_config.update(overrides)

            with open(log_file, "w+") as f:
                precision_values = bench_config.get("precision", ["s"])
                if not isinstance(precision_values, list):
                    precision_values = [precision_values]

                for precision in precision_values:
                    if dimensions == "sizes":
                        # GEMM-style: single size for all dimensions
                        self._run_sizes_benchmark(
                            f,
                            bench_config,
                            function,
                            precision,
                            iterations,
                            cold_iterations,
                            extra_args,
                            has_compute_type,
                        )
                    elif dimensions == "separate":
                        # GEMV/GER-style: separate m, n, lda values
                        self._run_separate_dims_benchmark(
                            f,
                            bench_config,
                            function,
                            precision,
                            iterations,
                            cold_iterations,
                        )
                    elif dimensions == "simple":
                        # DOT-style: single dimension (n)
                        self._run_simple_benchmark(
                            f,
                            bench_config,
                            function,
                            precision,
                            iterations,
                            cold_iterations,
                        )

        except Exception as e:
            log.error(f"{name.upper()} benchmark failed: {e}")
            log.warning("Continuing with next benchmark...")

    def _run_sizes_benchmark(
        self,
        f: IO[str],
        config: Dict[str, Any],
        function: str,
        precision: str,
        iterations: int,
        cold_iterations: int,
        extra_args: Dict[str, Any],
        has_compute_type: bool,
    ) -> None:
        """Run benchmarks that use 'sizes' parameter (GEMM, GEMM_HPA_HGEMM)."""
        sizes = config.get("sizes", [])
        transpose_values = config.get("transpose", ["N"])

        for size in sizes:
            for trans in transpose_values:
                # Build base command
                cmd = self._build_base_cmd(function, precision)

                # Add dimension arguments
                cmd.extend(
                    [
                        "-m",
                        str(size),
                        "-n",
                        str(size),
                        "-k",
                        str(size),
                        "--lda",
                        str(size),
                        "--ldb",
                        str(size),
                        "--ldc",
                        str(size),
                    ]
                )

                # Add ldd for gemm_ex
                if has_compute_type:
                    cmd.extend(["--ldd", str(size)])
                    compute_type = config.get("compute_type", "s")
                    cmd.extend(["--compute_type", compute_type])

                # Add transpose
                cmd.extend(["--transposeB", trans])

                # Add precision-specific extra args
                if precision in extra_args:
                    for key, val in extra_args[precision].items():
                        cmd.extend([f"--{key}", val])

                # Add iteration args
                cmd.extend(
                    ["--iters", str(iterations), "--cold_iters", str(cold_iterations)]
                )

                self.execute_command(cmd, f)

    def _run_separate_dims_benchmark(
        self,
        f: IO[str],
        config: Dict[str, Any],
        function: str,
        precision: str,
        iterations: int,
        cold_iterations: int,
    ) -> None:
        """Run benchmarks with separate m, n, lda parameters (GEMV, GER)."""
        m_values = config.get("m", [])
        n_values = config.get("n", [])
        lda_values = config.get("lda", [])
        transpose_values = config.get("transpose", ["N"])

        # Validate lengths match
        if not (len(m_values) == len(n_values) == len(lda_values)):
            log.warning(
                f"Config length mismatch: m={len(m_values)}, n={len(n_values)}, lda={len(lda_values)}"
            )

        for m, n, lda in zip(m_values, n_values, lda_values):
            # GEMV has transpose, GER doesn't
            transpose_loop = transpose_values if function == "gemv" else [None]

            for trans in transpose_loop:
                cmd = self._build_base_cmd(function, precision)
                cmd.extend(
                    [
                        "-m",
                        str(m),
                        "-n",
                        str(n),
                        "--lda",
                        str(lda),
                    ]
                )

                # Add transpose for GEMV
                if trans is not None:
                    cmd.extend(["--transposeA", trans])

                cmd.extend(
                    [
                        "--iters",
                        str(iterations),
                        "--cold_iters",
                        str(cold_iterations),
                    ]
                )
                self.execute_command(cmd, f)

    def _run_simple_benchmark(
        self,
        f: IO[str],
        config: Dict[str, Any],
        function: str,
        precision: str,
        iterations: int,
        cold_iterations: int,
    ) -> None:
        """Run benchmarks with single dimension n (DOT)."""
        n_values = config.get("n", [])

        for n in n_values:
            cmd = self._build_base_cmd(function, precision)
            cmd.extend(
                [
                    "-n",
                    str(n),
                    "--iters",
                    str(iterations),
                    "--cold_iters",
                    str(cold_iterations),
                ]
            )
            self.execute_command(cmd, f)

    def parse_results(self) -> Tuple[List[Dict[str, Any]], List[PrettyTable]]:
        """Parse benchmark results from log files.

        Parses CSV output from rocBLAS-bench for GEMM, GEMV, GER, DOT, and GEMM_HPA_HGEMM suites.
        Only rocblas-Gflops metric is captured.
        """
        log.info("Parsing Results")

        # Setup field names for tables
        field_names = [
            "TestName",
            "SubTests",
            "nGPU",
            "Result",
            "Scores",
            "Units",
            "Flag",
        ]

        # List to store all suite-specific tables
        all_tables = []

        test_results = []
        num_gpus = 1

        # List of log files to parse with suite names
        log_files = [
            (self.script_dir / "rocblas-gemm_bench.log", "GEMM"),
            (self.script_dir / "rocblas-gemv_bench.log", "GEMV"),
            (self.script_dir / "rocblas-ger_bench.log", "GER"),
            (self.script_dir / "rocblas-dot_bench.log", "DOT"),
            (self.script_dir / "rocblas-gemm_hpa_hgemm_bench.log", "GEMM_HPA_HGEMM"),
        ]

        for log_file, suite_name in log_files:
            if not log_file.exists():
                log.warning(f"Log file not found: {log_file}, skipping")
                continue

            log.info(f"Parsing {suite_name} results from {log_file.name}")

            # Create suite-specific table
            suite_table = PrettyTable(field_names)
            suite_table.title = f"ROCblas {suite_name} Benchmark Results"

            try:
                with open(log_file, "r") as log_fp:
                    lines = log_fp.readlines()

                # Parse line by line, looking for CSV header followed by data
                i = 0
                current_precision = None  # Track precision from command line

                while i < len(lines):
                    line = lines[i].strip()

                    # Extract precision from command line (e.g., "-r s")
                    if "rocblas-bench" in line and "-r" in line:
                        parts = line.split()
                        try:
                            idx = parts.index("-r")
                            current_precision = (
                                parts[idx + 1] if idx + 1 < len(parts) else None
                            )
                        except (ValueError, IndexError):
                            pass

                    # Look for CSV header line
                    if "rocblas-Gflops" in line:
                        header = [col.strip() for col in line.split(",")]

                        i += 1
                        if i >= len(lines):
                            break

                        data_line = lines[i].strip()
                        if not data_line or "rocblas-Gflops" in data_line:
                            i += 1
                            continue

                        values = [val.strip() for val in data_line.split(",")]
                        if len(values) != len(header):
                            log.warning(
                                f"Column mismatch: expected {len(header)}, got {len(values)}"
                            )
                            i += 1
                            continue

                        params = dict(zip(header, values))

                        # Add precision from command line if not in CSV
                        if current_precision and not any(
                            k in params for k in ["a_type", "precision"]
                        ):
                            params["precision"] = current_precision

                        function_type = self._determine_function_type(params)
                        subtest_name = self._build_subtest_name_from_params(
                            function_type, params
                        )

                        try:
                            gflops = float(params.get("rocblas-Gflops", "0"))
                            status = "PASS" if gflops > 0 else "FAIL"

                            row_data = [
                                self.benchmark_name,
                                subtest_name,
                                num_gpus,
                                status,
                                gflops,
                                "rocblas-Gflops",
                                "H",
                            ]
                            suite_table.add_row(row_data)

                            test_results.append(
                                self.create_test_result(
                                    self.benchmark_name,
                                    subtest_name,
                                    status,
                                    gflops,
                                    "rocblas-Gflops",
                                    "H",
                                    ngpu=num_gpus,
                                )
                            )
                        except (ValueError, TypeError) as e:
                            log.warning(f"Failed to parse metrics: {e}")
                            i += 1
                            continue

                    i += 1

            except OSError as e:
                log.error(f"IO Error reading {log_file}: {e}")
                continue

            # Add suite table to the list of tables
            all_tables.append(suite_table)

        return test_results, all_tables

    def _determine_function_type(self, params: Dict[str, str]) -> str:
        """Determine ROCblas function type from parameters."""
        if "ldd" in params or "stride_d" in params:
            return "gemm_hpa_hgemm"
        if "transA" in params and "transB" in params and "K" in params:
            return "gemm"
        if "transA" in params and "incx" in params:
            return "gemv"
        if "incx" in params and "incy" in params:
            return "dot" if "algo" in params else "ger"
        return "unknown"

    def _build_subtest_name_from_params(
        self, function_type: str, params: Dict[str, str]
    ) -> str:
        """Build descriptive subtest name from parameters."""

        # Helper to safely get and clean parameter values
        def get_param(key: str, default: str = "") -> str:
            return params.get(key, default).strip()

        # Get precision/data type (ROCblas uses different column names)
        def get_precision() -> str:
            """Extract precision from various possible column names."""
            precision = (
                get_param("a_type")
                or get_param("precision")
                or get_param("compute_type")
            )
            return f"_{precision}" if precision else ""

        # Build name based on function type with all available parameters
        # Format: operation_precision_parameters
        if function_type == "gemm":
            # Format: gemm_precision_transA_transB_M_N_K_alpha_lda_beta_ldb_ldc
            precision = get_precision()
            return (
                f"gemm{precision}"
                f"_{get_param('transA')}{get_param('transB')}"
                f"_{get_param('M')}_{get_param('N')}_{get_param('K')}"
                f"_{get_param('alpha')}_{get_param('lda')}"
                f"_{get_param('beta')}_{get_param('ldb')}_{get_param('ldc')}"
            )
        elif function_type == "gemv":
            # Format: gemv_precision_transA_M_N_alpha_lda_incx_beta_incy
            precision = get_precision()
            return (
                f"gemv{precision}"
                f"_{get_param('transA')}"
                f"_{get_param('M')}_{get_param('N')}"
                f"_{get_param('alpha')}_{get_param('lda')}"
                f"_{get_param('incx')}_{get_param('beta')}_{get_param('incy')}"
            )
        elif function_type == "ger":
            # Format: ger_precision_M_N_alpha_lda_incx_incy
            precision = get_precision()
            return (
                f"ger{precision}"
                f"_{get_param('M')}_{get_param('N')}"
                f"_{get_param('alpha')}_{get_param('lda')}"
                f"_{get_param('incx')}_{get_param('incy')}"
            )
        elif function_type == "dot":
            # Format: dot_precision_N_incx_incy_algo
            precision = get_precision()
            return (
                f"dot{precision}"
                f"_{get_param('N')}"
                f"_{get_param('incx')}_{get_param('incy')}"
                f"_{get_param('algo', '0')}"
            )
        elif function_type == "gemm_hpa_hgemm":
            # Format: gemm_hpa_hgemm_a_type_compute_type_transA_transB_M_N_K_alpha_lda_beta_ldb_ldc_ldd_batch_count
            batch_count = get_param("batch_count", "1")
            a_type = get_param("a_type", "h")
            compute_type = get_param("compute_type", "s")

            # Include strides if available (for strided batched operations)
            stride_a = get_param("stride_a")
            stride_b = get_param("stride_b")
            stride_c = get_param("stride_c")
            stride_d = get_param("stride_d")

            strides = ""
            if stride_a:
                strides = f"_sa{stride_a}_sb{stride_b}_sc{stride_c}_sd{stride_d}"

            return (
                f"gemm_hpa_hgemm_{a_type}_{compute_type}"
                f"_{get_param('transA')}{get_param('transB')}"
                f"_{get_param('M')}_{get_param('N')}_{get_param('K')}"
                f"_{get_param('alpha')}_{get_param('lda')}"
                f"_{get_param('beta')}_{get_param('ldb')}"
                f"_{get_param('ldc')}_{get_param('ldd')}"
                f"_bc{batch_count}"
                f"{strides}"
            )
        else:
            # Fallback for unknown functions - include all non-metric params
            excluded_keys = {"rocblas-Gflops", "rocblas-GB/s", "us", "function"}
            param_str = "_".join(
                [f"{k}{v}" for k, v in params.items() if k not in excluded_keys and v]
            )
            return f"{function_type}_{param_str}"[:150]  # Limit length


if __name__ == "__main__":
    run_benchmark_main(ROCblasBenchmark())
