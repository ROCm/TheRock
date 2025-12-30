"""
ROCblas Benchmark Test

Runs ROCblas benchmarks, collects results, and uploads to results API.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
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

        log.info("Running ROCblas Benchmarks")

        # Run each benchmark suite with its own log file
        self._run_gemm_benchmarks(config_data, iterations, cold_iterations)
        self._run_gemv_benchmarks(config_data, iterations, cold_iterations)
        self._run_ger_benchmarks(config_data, iterations, cold_iterations)
        self._run_dot_benchmarks(config_data, iterations, cold_iterations)
        self._run_gemm_hpa_hgemm_benchmarks(config_data, iterations, cold_iterations)

        log.info("Benchmark execution complete")

    def _run_gemm_benchmarks(
        self, config_data: Dict, iterations: int, cold_iterations: int
    ) -> None:
        """Run GEMM benchmarks."""
        try:
            log.info("Running rocBLAS-GEMM Benchmarks")
            log_file = self.script_dir / "rocblas-gemm_bench.log"

            with open(log_file, "w+") as f:
                # Get GEMM configuration
                gemm_config = config_data.get("gemm", {})
                gemm_sizes = gemm_config.get("sizes", [])
                gemm_trans = gemm_config.get("transpose", [])
                gemm_precision = gemm_config.get("precision", [])

                for precision in gemm_precision:
                    for size in gemm_sizes:
                        m = n = k = lda = ldb = ldc = size
                        for trans in gemm_trans:
                            # Build ROCblas benchmark command
                            cmd = [
                                f"{self.therock_bin_dir}/rocblas-bench",
                                "-f",
                                "gemm",
                                "-r",
                                precision,
                                "--initialization",
                                "rand_int",
                                "-m",
                                str(m),
                                "-n",
                                str(n),
                                "-k",
                                str(k),
                                "--lda",
                                str(lda),
                                "--ldb",
                                str(ldb),
                                "--ldc",
                                str(ldc),
                                "--transposeB",
                                trans,
                                "-i",
                                str(iterations),
                                "-j",
                                str(cold_iterations),
                            ]

                            # Add precision-specific parameters
                            if precision == "h":
                                cmd.extend(
                                    ["--alpha", "1", "--beta", "0", "--transposeA", "N"]
                                )

                            self.execute_command(cmd, f)
        except Exception as e:
            log.error(f"GEMM benchmark failed: {e}")
            log.warning("Continuing with next benchmark...")

    def _run_gemv_benchmarks(
        self, config_data: Dict, iterations: int, cold_iterations: int
    ) -> None:
        """Run GEMV benchmarks."""
        try:
            log.info("Running rocBLAS-GEMV Benchmarks")
            log_file = self.script_dir / "rocblas-gemv_bench.log"

            with open(log_file, "w+") as f:
                # Get GEMV configuration
                gemv_config = config_data.get("gemv", {})
                gemv_m = gemv_config.get("m", [])
                gemv_n = gemv_config.get("n", [])
                gemv_lda = gemv_config.get("lda", [])
                gemv_trans = gemv_config.get("transpose", [])
                gemv_precision = gemv_config.get("precision", [])

                # Validate configuration lengths match
                if not (len(gemv_m) == len(gemv_n) == len(gemv_lda)):
                    log.warning(
                        f"GEMV config length mismatch: m={len(gemv_m)}, n={len(gemv_n)}, lda={len(gemv_lda)}"
                    )

                for precision in gemv_precision:
                    for m, n, lda in zip(gemv_m, gemv_n, gemv_lda):
                        for trans in gemv_trans:
                            # Build ROCblas GEMV benchmark command
                            cmd = [
                                f"{self.therock_bin_dir}/rocblas-bench",
                                "-f",
                                "gemv",
                                "-r",
                                precision,
                                "--initialization",
                                "rand_int",
                                "-m",
                                str(m),
                                "-n",
                                str(n),
                                "--lda",
                                str(lda),
                                "--transposeA",
                                trans,
                                "-i",
                                str(iterations),
                                "-j",
                                str(cold_iterations),
                            ]
                            self.execute_command(cmd, f)
        except Exception as e:
            log.error(f"GEMV benchmark failed: {e}")
            log.warning("Continuing with next benchmark...")

    def _run_ger_benchmarks(
        self, config_data: Dict, iterations: int, cold_iterations: int
    ) -> None:
        """Run GER benchmarks."""
        try:
            log.info("Running rocBLAS-GER Benchmarks")
            log_file = self.script_dir / "rocblas-ger_bench.log"

            with open(log_file, "w+") as f:
                # Get GER configuration
                ger_config = config_data.get("ger", {})
                ger_m = ger_config.get("m", [])
                ger_n = ger_config.get("n", [])
                ger_lda = ger_config.get("lda", [])
                ger_precision = ger_config.get("precision", [])

                # Validate configuration lengths match
                if not (len(ger_m) == len(ger_n) == len(ger_lda)):
                    log.warning(
                        f"GER config length mismatch: m={len(ger_m)}, n={len(ger_n)}, lda={len(ger_lda)}"
                    )

                for precision in ger_precision:
                    for m, n, lda in zip(ger_m, ger_n, ger_lda):
                        # Build ROCblas GER benchmark command
                        cmd = [
                            f"{self.therock_bin_dir}/rocblas-bench",
                            "-f",
                            "ger",
                            "-r",
                            precision,
                            "--initialization",
                            "rand_int",
                            "-m",
                            str(m),
                            "-n",
                            str(n),
                            "--lda",
                            str(lda),
                            "-i",
                            str(iterations),
                            "-j",
                            str(cold_iterations),
                        ]
                        self.execute_command(cmd, f)
        except Exception as e:
            log.error(f"GER benchmark failed: {e}")
            log.warning("Continuing with next benchmark...")

    def _run_dot_benchmarks(
        self, config_data: Dict, iterations: int, cold_iterations: int
    ) -> None:
        """Run DOT benchmarks."""
        try:
            log.info("Running rocBLAS-DOT Benchmarks")
            log_file = self.script_dir / "rocblas-dot_bench.log"

            with open(log_file, "w+") as f:
                # Get DOT configuration
                dot_config = config_data.get("dot", {})
                dot_n = dot_config.get("n", [])
                dot_precision = dot_config.get("precision", [])

                for precision in dot_precision:
                    for n in dot_n:
                        # Build ROCblas DOT benchmark command
                        cmd = [
                            f"{self.therock_bin_dir}/rocblas-bench",
                            "-f",
                            "dot",
                            "-r",
                            precision,
                            "--initialization",
                            "rand_int",
                            "-n",
                            str(n),
                            "-i",
                            str(iterations),
                            "-j",
                            str(cold_iterations),
                        ]
                        self.execute_command(cmd, f)
        except Exception as e:
            log.error(f"DOT benchmark failed: {e}")
            log.warning("Continuing with next benchmark...")

    def _run_gemm_hpa_hgemm_benchmarks(
        self, config_data: Dict, iterations: int, cold_iterations: int
    ) -> None:
        """Run GEMM_HPA_HGEMM (HPA/Half-Precision Accumulate) benchmarks."""
        try:
            log.info("Running rocBLAS-GEMM_HPA_HGEMM Benchmarks")
            log_file = self.script_dir / "rocblas-gemm_hpa_hgemm_bench.log"

            with open(log_file, "w+") as f:
                # Get GEMM_HPA_HGEMM configuration
                gemm_hpa_hgemm_config = config_data.get("gemm_hpa_hgemm", {})
                gemm_hpa_hgemm_sizes = gemm_hpa_hgemm_config.get("sizes", [])
                gemm_hpa_hgemm_trans = gemm_hpa_hgemm_config.get("transpose", [])
                gemm_hpa_hgemm_precision = gemm_hpa_hgemm_config.get("precision", "h")
                gemm_hpa_hgemm_compute_type = gemm_hpa_hgemm_config.get(
                    "compute_type", "s"
                )

                for size in gemm_hpa_hgemm_sizes:
                    m = n = k = lda = ldb = ldc = ldd = size
                    for trans in gemm_hpa_hgemm_trans:
                        # Build ROCblas GEMM_HPA_HGEMM benchmark command
                        cmd = [
                            f"{self.therock_bin_dir}/rocblas-bench",
                            "-f",
                            "gemm_ex",
                            "-r",
                            gemm_hpa_hgemm_precision,
                            "-m",
                            str(m),
                            "-n",
                            str(n),
                            "-k",
                            str(k),
                            "--lda",
                            str(lda),
                            "--ldb",
                            str(ldb),
                            "--ldc",
                            str(ldc),
                            "--ldd",
                            str(ldd),
                            "--compute_type",
                            gemm_hpa_hgemm_compute_type,
                            "--transposeB",
                            trans,
                            "-i",
                            str(iterations),
                            "-j",
                            str(cold_iterations),
                        ]

                        self.execute_command(cmd, f)
        except Exception as e:
            log.error(f"GEMM_HPA_HGEMM benchmark failed: {e}")
            log.warning("Continuing with next benchmark...")

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
