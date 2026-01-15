"""
Kokkos HPC Benchmark Test

Builds and runs Kokkos benchmarks, collects results, and uploads to results API.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log
from utils.system.hardware import HardwareDetector


class KokkosBenchmark(BenchmarkBase):
    """Kokkos benchmark test."""

    def __init__(self):
        super().__init__(benchmark_name="kokkos", display_name="Kokkos")
        self.log_file = self.script_dir / "kokkos_bench.log"
        self.therock_dir = self.script_dir.parent.parent.parent.parent

        # Load configuration
        config_file = self.script_dir.parent / "configs" / "kokkos.json"
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.repo_path = Path.cwd() / self.benchmark_name
        self.build_dir = self.repo_path / "build"
        self.benchmark_json_path = self.build_dir / "results.json"

        # Setup environment with ROCm paths
        self.env = os.environ.copy()
        self.env["PATH"] = f"{self.therock_bin_dir}:{self.env.get('PATH', '')}"
        self.env["LD_LIBRARY_PATH"] = (
            f"{Path(self.therock_bin_dir).parent / 'lib'}:{self.env.get('LD_LIBRARY_PATH', '')}"
        )
        self.env["HIP_PLATFORM"] = "amd"

    def build(self) -> None:
        """Build Kokkos from source."""
        log.info("Starting Kokkos build process...")

        # Detect GPU architecture using HardwareDetector
        detector = HardwareDetector()
        gfx_id = detector.get_gpu_architecture()

        # Clone repository
        success, message = self.clone_repository(
            self.config["repository_url"],
            self.repo_path,
            branch=self.config["repository_branch"],
        )
        if not success:
            raise ValueError(f"Failed to clone repository: {message}")

        # Remove existing build directory
        if self.build_dir.exists():
            log.info(f"Removing existing build directory: {self.build_dir}")
            shutil.rmtree(self.build_dir)

        self.build_dir.mkdir(parents=True)

        # Configure with CMake
        log.info("Configuring Kokkos with CMake...")
        cmake_cmd = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_CXX_COMPILER=hipcc",
            f"-DCMAKE_INSTALL_PREFIX={self.repo_path / 'install'}",
            "-DKokkos_ENABLE_BENCHMARKS=On",
            "-DKokkos_ENABLE_HIP=On",
            "-DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=On",
            "-DKokkos_ENABLE_SERIAL=On",
            "-DKokkos_ENABLE_TESTS=On",
            "-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=Off",
            f"-DKokkos_ARCH_AMD_{gfx_id.upper()}=On",
        ]

        log.info(f"++ Exec [{self.build_dir}]$ {shlex.join(cmake_cmd)}")
        subprocess.run(cmake_cmd, cwd=self.build_dir, env=self.env, check=True)

        # Build
        log.info("Building Kokkos...")
        nproc = os.cpu_count() or 4
        make_cmd = ["make", "-j", str(nproc)]

        log.info(f"++ Exec [{self.build_dir}]$ {shlex.join(make_cmd)}")
        subprocess.run(make_cmd, cwd=self.build_dir, env=self.env, check=True)

        # Install
        log.info("Installing Kokkos...")
        install_cmd = ["make", "-j", str(nproc), "install"]

        log.info(f"++ Exec [{self.build_dir}]$ {shlex.join(install_cmd)}")
        subprocess.run(install_cmd, cwd=self.build_dir, env=self.env, check=True)

        log.info("Kokkos build completed successfully")

    def run_benchmarks(self) -> None:
        """Run Kokkos benchmarks and save output to log file."""
        log.info("Running Kokkos Benchmarks")

        # Check if build exists
        if not self.build_dir.exists():
            raise ValueError(f"Kokkos build directory not found: {self.build_dir}")

        # Check if benchmark executable exists
        benchmark_exe = (
            self.build_dir / "core" / "perf_test" / "Kokkos_PerformanceTest_Benchmark"
        )
        if not benchmark_exe.exists():
            raise ValueError(f"Benchmark executable not found: {benchmark_exe}")

        # Run benchmark
        cmd = [
            str(benchmark_exe),
            "--benchmark_counters_tabular=true",
            "--benchmark_out_format=json",
            f"--benchmark_out={self.benchmark_json_path}",
        ]

        with open(self.log_file, "w+") as f:
            log.info(f"++ Exec [{self.build_dir}]$ {shlex.join(cmd)}")
            f.write(f"{shlex.join(cmd)}\n")

            process = subprocess.Popen(
                cmd,
                cwd=self.build_dir,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                log.info(line.strip())
                f.write(f"{line}")

            process.wait()

            if process.returncode != 0:
                raise ValueError(
                    f"Benchmark execution failed with return code {process.returncode}"
                )

        log.info("Benchmark execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse benchmark results from JSON file.

        Filters for configured test cases and kernel size from kokkos.json.

        Returns:
            tuple: (test_results list, PrettyTable object)
        """
        log.info("Parsing Results")

        # Setup table
        field_names = [
            "TestName",
            "SubTests",
            "Status",
            "Scores",
            "Units",
            "Flag",
        ]
        table = PrettyTable(field_names)

        test_results = []

        if not self.benchmark_json_path.exists():
            raise ValueError(f"Results file not found: {self.benchmark_json_path}")

        log.info(f"Parsing benchmark results from {self.benchmark_json_path}")

        try:
            with open(self.benchmark_json_path, "r") as benchmark_file:
                benchmark_data = json.load(benchmark_file)

            benchmarks = benchmark_data.get("benchmarks")
            if not benchmarks:
                raise ValueError(
                    f"No 'benchmarks' key found in results JSON {self.benchmark_json_path}"
                )

            # Get kernel size string once (outside loop for efficiency)
            kernel_size_str = str(self.config["kernel_size"])

            for result in benchmarks:
                result_name = result["name"]

                # Filter for configured test cases and kernel size
                is_target_test = any(
                    test in result_name for test in self.config["test_cases"]
                )
                has_target_kernel_size = kernel_size_str in result_name

                if not (is_target_test and has_target_kernel_size):
                    continue

                # Extract metrics
                test_name = result_name
                time_normalized = result.get("Time normalized", 0.0)
                time_unit = result.get("time_unit", "ns")
                status = "PASS" if time_normalized > 0 else "FAIL"

                log.info(
                    f"Found {test_name} with kernel size {kernel_size_str}, score: {time_normalized} {time_unit}"
                )

                # Add to table
                table.add_row(
                    [
                        self.benchmark_name,
                        test_name,
                        status,
                        time_normalized,
                        time_unit,
                        "L",
                    ]
                )

                # Add to results list
                test_results.append(
                    self.create_test_result(
                        self.benchmark_name,
                        test_name,
                        status,
                        time_normalized,
                        time_unit,
                        "L",
                    )
                )

        except OSError as e:
            raise ValueError(f"IO Error parsing results: {e}")
        except KeyError as e:
            raise ValueError(f"Missing key in results JSON: {e}")

        return test_results, table


if __name__ == "__main__":
    run_benchmark_main(KokkosBenchmark())
