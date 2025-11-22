"""
ROCrand Benchmark Test

Runs ROCrand benchmarks, collects results, and uploads to results API.
"""

import csv
import io
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import TestClient
from utils.logger import log

# Constants
BENCHMARK_NAME = 'rocrand'

# Environment variables
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
ARTIFACT_RUN_ID = os.getenv("ARTIFACT_RUN_ID")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent


def run_benchmarks() -> None:
    """Run ROCrand benchmarks and save output to log files."""
    NUM_TRIALS = 1000  # Number of benchmark trials
    
    bench_bins = ['benchmark_rocrand_host_api', 'benchmark_rocrand_device_api']
    
    log.info("Running ROCrand Benchmarks")
    
    for bench_bin in bench_bins:
        # Extract benchmark type from binary name
        match = re.search(r'benchmark_(.*?)_api', bench_bin)
        if not match:
            log.warning(f"Could not parse benchmark name from: {bench_bin}")
            continue
        
        bench_type = match.group(1)
        log_file = SCRIPT_DIR / f"{bench_type}_bench.log"
        
        # Check if binary exists
        bench_path = Path(THEROCK_BIN_DIR) / bench_bin
        if not bench_path.exists():
            log.error(f"Benchmark binary not found: {bench_bin}")
            continue
        
        # Run benchmark
        with open(log_file, "w+") as f:
            cmd = [
                str(bench_path),
                "--trials", str(NUM_TRIALS),
                "--benchmark_color=false",
                "--benchmark_format=csv"
            ]
            
            log.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
            f.write(f"{shlex.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                cwd=THEROCK_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                log.info(line.strip())
                f.write(f"{line}\n")
            
            process.wait()
    
    log.info("Benchmark execution complete")


def create_test_result(test_name: str,
                       subtest_name: str,
                       mode: str,
                       status: str,
                       score: float,
                       unit: str,
                       flag: str) -> Dict[str, Any]:
    """Create a test result dictionary.
    
    Args:
        test_name: Benchmark name
        subtest_name: Specific test name
        mode: Benchmark mode
        status: PASS/FAIL
        score: Metric value
        unit: Unit of measurement
        flag: H (higher better) or L (lower better)
        
    Returns:
        dict: Test result dictionary
    """
    return {
        "test_name": test_name,
        "subtest": subtest_name,
        "batch_size": 0,
        "ngpu": 1,
        "status": status,
        "score": float(score) if score else 0.0,
        "unit": unit,
        "flag": flag,
        "test_config": {
            "test_name": test_name,
            "sub_test_name": subtest_name,
            "mode": mode,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "environment_dependencies": [],
            "batch_size": 0,
            "ngpu": 1
        }
    }


def parse_results() -> Tuple[List[Dict[str, Any]], PrettyTable]:
    """Parse benchmark results from log files.
    
    Returns:
        tuple: (test_results list, PrettyTable object)
    """
    log.info("Parsing Results")
    
    # Regex pattern to match CSV section in benchmark output
    csv_pattern = re.compile(
        r'^engine,distribution,mode,name,iterations,real_time,cpu_time,time_unit,bytes_per_second,throughput_gigabytes_per_second,lambda,items_per_second,label,error_occurred,error_message\n(?:[^\n]*\n)+$',
        re.MULTILINE
    )
    
    bench_types = ['rocrand_host', 'rocrand_device']
    
    # Setup table
    field_names = ['TestName', 'SubTests', 'Mode', 'Result', 'Scores', 'Units', 'Flag']
    table = PrettyTable(field_names)
    
    test_results = []
    num_gpus = 1
    
    for bench_type in bench_types:
        log_file = SCRIPT_DIR / f"{bench_type}_bench.log"
        
        if not log_file.exists():
            log.warning(f"Log file not found: {log_file}")
            continue
        
        log.info(f"Parsing {bench_type} results")
        
        try:
            with open(log_file, 'r') as f:
                data = f.read()
            
            # Find the CSV data in the file
            csv_match = csv_pattern.search(data)
            if not csv_match:
                log.warning(f"No CSV data found in {log_file}")
                continue
            
            csv_data = csv_match.group()
            lines = csv_data.strip().split('\n')
            
            # Parse CSV data
            csv_reader = csv.DictReader(io.StringIO('\n'.join(lines)))
            
            for row in csv_reader:
                engine = row.get('engine', '')
                distribution = row.get('distribution', '')
                mode = row.get('mode', '')
                throughput = row.get('throughput_gigabytes_per_second', '0')
                
                # Create subtest name
                subtest_name = f"{engine}-{distribution}"
                
                # Determine status
                try:
                    throughput_val = float(throughput)
                    status = "PASS" if throughput_val > 0 else "FAIL"
                except (ValueError, TypeError):
                    throughput_val = 0.0
                    status = "FAIL"
                
                # Add to table
                table.add_row([bench_type, subtest_name, mode, status, throughput_val, 'GB/s', 'H'])
                
                # Add to test results
                test_results.append(create_test_result(
                    bench_type,
                    subtest_name,
                    mode,
                    status,
                    throughput_val,
                    "GB/s",
                    "H"  # Higher is better for throughput
                ))
        
        except OSError as e:
            log.error(f"Error reading {log_file}: {e}")
            continue
    
    return test_results, table


def main():
    """Main execution function."""
    log.info("Initializing ROCrand Benchmark Test")
    
    # Initialize test client and print system info
    client = TestClient(auto_detect=True)
    client.print_system_summary()
    
    # Run benchmarks
    run_benchmarks()
    
    # Parse results
    test_results, table = parse_results()
    
    if not test_results:
        log.error("No test results found")
        return 1
    
    # Calculate statistics
    passed = sum(1 for r in test_results if r['status'] == 'PASS')
    failed = sum(1 for r in test_results if r['status'] == 'FAIL')
    overall_status = "PASS" if failed == 0 else "FAIL"
    
    log.info(f"Test Summary: {passed} passed, {failed} failed")
    
    # Upload results
    log.info("Uploading Results to API")
    success = client.upload_results(
        test_name="rocrand_benchmark",
        test_results=test_results,
        test_status=overall_status,
        test_metadata={
            "artifact_run_id": ARTIFACT_RUN_ID,
            "amdgpu_families": AMDGPU_FAMILIES,
            "benchmark_name": BENCHMARK_NAME,
            "total_subtests": len(test_results),
            "passed_subtests": passed,
            "failed_subtests": failed
        },
        save_local=True,
        output_dir=str(SCRIPT_DIR / "results")
    )
    
    if success:
        log.info("✓ Results uploaded successfully")
    else:
        log.info("⚠ Results saved locally only (API upload disabled or failed)")
    
    # Compare with LKG
    log.info("Comparing results with LKG")
    final_table = client.compare_results(test_name=BENCHMARK_NAME, table=table)
    log.info(f"\n{final_table}")
    
    # Determine final status
    if 'FinalResult' not in final_table.field_names:
        raise ValueError("The table does not have a 'FinalResult' column.")
    
    final_result_index = final_table.field_names.index('FinalResult')
    final_status = 'FAIL' if any(row[final_result_index] == 'FAIL' for row in final_table._rows) else 'PASS'
    
    log.info(f"Final Status: {final_status}")
    
    return 0 if final_status == "PASS" else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warning("\nExecution interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
