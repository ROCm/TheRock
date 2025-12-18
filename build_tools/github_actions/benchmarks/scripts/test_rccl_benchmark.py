"""
RCCL Benchmark Test

Runs RCCL collective communication benchmarks, collects results, and uploads to results API.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent.parent))  # For utils
sys.path.insert(0, str(Path(__file__).parent))         # For benchmark_base
from benchmark_base import BenchmarkBase, run_benchmark_main
from utils.logger import log


class RCCLBenchmark(BenchmarkBase):
    """RCCL benchmark test."""
    
    def __init__(self):
        super().__init__(benchmark_name='rccl', display_name='RCCL')
        self.log_file = self.script_dir / "rccl_bench.log"
        self.mpi_path = None
        self.ngpu = self._detect_gpu_count()
    
    def _detect_gpu_count(self) -> int:
        """Detect the number of available GPUs."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showgpus'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Count GPU lines in output
            gpu_count = len([line for line in result.stdout.split('\n') if 'GPU[' in line])
            return max(1, gpu_count)
        except Exception as e:
            log.warning(f"Could not detect GPU count: {e}. Defaulting to 1.")
            return 1
    
    def _setup_mpi(self) -> bool:
        """Setup MPI - check for existing installation or install if needed."""
        # Check system MPI installations
        for mpi_bin in ["/usr/bin", "/usr/lib64/openmpi/bin", "/opt/openmpi/bin"]:
            if Path(mpi_bin, "mpirun").exists():
                self.mpi_path = mpi_bin
                log.info(f"Found system MPI at: {self.mpi_path}")
                return True
        
        # Check local build or build if needed
        mpi_install_dir = self.script_dir / "openmpi"
        mpirun = mpi_install_dir / "bin" / "mpirun"
        
        if mpirun.exists():
            self.mpi_path = str(mpirun.parent)
            log.info(f"Using local MPI at: {self.mpi_path}")
            return True
        
        # Build MPI locally
        log.info("Building OpenMPI locally (this may take several minutes)...")
        mpi_version = "4.1.4"
        mpi_url = f"https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{mpi_version}.tar.gz"
        mpi_src_dir = self.script_dir / f"openmpi-{mpi_version}"
        
        try:
            # Download and extract
            tar_file = self.script_dir / f"openmpi-{mpi_version}.tar.gz"
            subprocess.run(["wget", "-q", "-O", str(tar_file), mpi_url], check=True, timeout=300)
            subprocess.run(["tar", "-xzf", str(tar_file), "-C", str(self.script_dir)], check=True, timeout=60)
            tar_file.unlink()
            
            # Configure, build, and install
            ncpus = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip() or "4"
            common_opts = {"cwd": str(mpi_src_dir), "stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
            
            subprocess.run(["./configure", f"--prefix={mpi_install_dir}", "--enable-mpi-cxx", 
                          "--with-rocm", "--disable-man-pages"], check=True, timeout=600, **common_opts)
            subprocess.run(["make", f"-j{ncpus}"], check=True, timeout=1800, **common_opts)
            subprocess.run(["make", "install"], check=True, timeout=300, **common_opts)
            
            # Cleanup
            subprocess.run(["rm", "-rf", str(mpi_src_dir)], check=False)
            
            if mpirun.exists():
                self.mpi_path = str(mpirun.parent)
                log.info(f"OpenMPI built successfully at: {self.mpi_path}")
                return True
            
            log.error("MPI build failed - mpirun not found after installation")
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.error(f"MPI build failed: {e}")
        except Exception as e:
            log.error(f"Unexpected error during MPI build: {e}")
        
        log.warning("MPI unavailable - RCCL tests may not run optimally")
        return False
    
    def run_benchmarks(self) -> None:
        """Run RCCL benchmarks and save output to log file."""
        # Load benchmark configuration
        config_file = self.script_dir.parent / 'configs' / 'rccl.json'
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        # Setup MPI
        self._setup_mpi()
        
        # Get configuration
        benchmarks = config_data.get("benchmarks", [])
        data_types = config_data.get("data_types", ["float"])
        min_size = config_data.get("min_size", "8")
        max_size = config_data.get("max_size", "128M")
        step_factor = config_data.get("step_factor", "2")
        warmup_iters = config_data.get("warmup_iters", "75")
        test_iters = config_data.get("test_iters", "1000")
        operations = config_data.get("operations", ["sum"])
        
        # Apply GPU-specific overrides
        if self.amdgpu_families:
            overrides = config_data.get("gpu_overrides", {}).get(self.amdgpu_families, {})
            max_size = overrides.get("max_size", max_size)
        
        log.info(f"Running RCCL Benchmarks with {self.ngpu} GPU(s)")
        log.info(f"Size range: {min_size} to {max_size}, step factor: {step_factor}")
        
        with open(self.log_file, "w+") as f:
            for benchmark in benchmarks:
                bench_binary = Path(self.therock_bin_dir) / benchmark
                
                if not bench_binary.exists():
                    log.warning(f"Benchmark binary not found: {bench_binary}")
                    continue
                
                for dtype in data_types:
                    for operation in operations:
                        log.info(f"Running {benchmark} with dtype={dtype}, operation={operation}")
                        
                        # Write section header to log
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Benchmark: {benchmark}\n")
                        f.write(f"DataType: {dtype}\n")
                        f.write(f"Operation: {operation}\n")
                        f.write(f"{'='*80}\n\n")
                        f.flush()
                        
                        # Build environment variables
                        env_vars = {"HSA_FORCE_FINE_GRAIN_PCIE": "1"}
                        
                        # Construct benchmark command with MPI
                        if self.mpi_path:
                            # Update environment with MPI paths
                            mpi_lib_path = str(Path(self.mpi_path).parent / "lib")
                            env_vars["PATH"] = f"{self.mpi_path}:{os.environ.get('PATH', '')}"
                            env_vars["LD_LIBRARY_PATH"] = f"{mpi_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                            
                            # Run with MPI - use one process per GPU
                            mpirun_binary = str(Path(self.mpi_path) / "mpirun")
                            cmd = [
                                mpirun_binary,
                                "-np", str(self.ngpu),
                                "--allow-run-as-root",
                                str(bench_binary),
                                "-b", min_size,
                                "-e", max_size,
                                "-f", step_factor,
                                "-g", "1",  # Each MPI process handles 1 GPU
                                "-o", operation,
                                "-d", dtype,
                                "-n", test_iters,
                                "-w", warmup_iters
                            ]
                        else:
                            # Fallback: run without MPI
                            cmd = [
                                str(bench_binary),
                                "-b", min_size,
                                "-e", max_size,
                                "-f", step_factor,
                                "-g", str(self.ngpu),
                                "-o", operation,
                                "-d", dtype,
                                "-n", test_iters,
                                "-w", warmup_iters
                            ]
                        
                        self.execute_command(cmd, f, env=env_vars)
        
        log.info("Benchmark execution complete")
    
    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse benchmark results from log file.
        
        Returns:
            tuple: (test_results list, PrettyTable object)
        """
        # Regex patterns for parsing
        pattern_benchmark = re.compile(r'Benchmark:\s*(\S+)')
        pattern_dtype = re.compile(r'DataType:\s*(\S+)')
        pattern_operation = re.compile(r'Operation:\s*(\S+)')
        pattern_bandwidth = re.compile(r'#\s+Avg bus bandwidth\s+:\s+(\d+\.\d+)')
        
        log.info("Parsing Results")
        
        # Setup table
        field_names = ['TestName', 'SubTests', 'nGPU', 'Result', 'Scores', 'Units', 'Flag']
        table = PrettyTable(field_names)
        
        test_results = []
        
        try:
            with open(self.log_file, 'r') as log_fp:
                content = log_fp.read()
            
            # Split by benchmark sections
            sections = content.split('=' * 80)
            
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
                table.add_row([
                    self.benchmark_name,
                    subtest_name,
                    self.ngpu,
                    status,
                    bandwidth,
                    'GB/s',
                    'H'  # Higher is better
                ])
                
                test_results.append(self.create_test_result(
                    self.benchmark_name,
                    subtest_name,
                    status,
                    bandwidth,
                    "GB/s",
                    "H",
                    ngpu=self.ngpu,
                    dtype=dtype,
                    operation=operation
                ))
        
        except FileNotFoundError:
            log.error(f"Log file not found: {self.log_file}")
            raise
        except Exception as e:
            log.error(f"Error parsing results: {e}")
            raise ValueError(f"IO Error in Score Extractor: {e}")
        
        return test_results, table


if __name__ == '__main__':
    run_benchmark_main(RCCLBenchmark())
