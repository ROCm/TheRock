"""
ROCfft Benchmark Test

Runs ROCfft benchmarks, collects results, and uploads to results API.
"""

import json
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
BENCHMARK_NAME = 'rocfft'

# Environment variables
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
ARTIFACT_RUN_ID = os.getenv("ARTIFACT_RUN_ID")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent


def run_benchmarks() -> None:
    """Run ROCfft benchmarks and save output to log file."""
    DEFAULT_BATCH_SIZE = 10  # Default batch size for benchmarks
    NUM_ITERATIONS = 20  # Number of benchmark iterations
    
    # Load benchmark configuration
    config_file = SCRIPT_DIR.parent / 'configs/benchmarks/rocfft.json'
    with open(config_file, "r") as f:
        data = json.load(f)
    
    test_cases = data.get("generic", [])
    if AMDGPU_FAMILIES:
        test_cases.extend(data.get(AMDGPU_FAMILIES, []))
    
    log_file = SCRIPT_DIR / "rocfft_bench.log"
    
    log.info("Running ROCfft Benchmarks")
    
    with open(log_file, "w+") as f:
        for test_case in test_cases:
            # Extract batch size from test case string (if specified)
            pattern_batch_size = re.compile(r'-b\s+(\d+)')
            explicit_batch = re.search(pattern_batch_size, test_case)
            
            if explicit_batch:
                batch_size = int(explicit_batch.group(1))
                cleaned_case = re.sub(r'-b\s+\d+', "", test_case)
            else:
                batch_size = DEFAULT_BATCH_SIZE
                cleaned_case = test_case
            
            cmd = [
                f"{THEROCK_BIN_DIR}/rocfft-bench",
                "--length",
                *cleaned_case.split(),
                "-b", str(batch_size),
                "-N", str(NUM_ITERATIONS)
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
                       batch_size: int,
                       num_gpus: int,
                       status: str,
                       score: float,
                       unit: str,
                       flag: str) -> Dict[str, Any]:
    """Create a test result dictionary.
    
    Args:
        test_name: Benchmark name
        subtest_name: Specific test identifier
        batch_size: Batch size used in test
        num_gpus: Number of GPUs used
        status: Test status ('PASS' or 'FAIL')
        score: Performance metric value
        unit: Unit of measurement (e.g., 'ms', 'GFLOPS')
        flag: 'H' (higher better) or 'L' (lower better)
        
    Returns:
        Dict[str, Any]: Test result dictionary with test data and configuration
    """
    return {
        "test_name": test_name,
        "subtest": subtest_name,
        "batch_size": batch_size,
        "ngpu": num_gpus,
        "status": status,
        "score": float(score),
        "unit": unit,
        "flag": flag,
        "test_config": {
            "test_name": test_name,
            "sub_test_name": subtest_name,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "environment_dependencies": [],
            "batch_size": batch_size,
            "ngpu": num_gpus
        }
    }


def parse_results() -> Tuple[List[Dict[str, Any]], PrettyTable]:
    """Parse benchmark results from log file.
    
    Returns:
        tuple: (test_results list, PrettyTable object)
    """
    default_batch_size = 10
    
    # Regex patterns for parsing
    pattern_test_case = re.compile(r'\s*--(length)\s*(\d+.*)')
    pattern_gpu_time = re.compile(r'(\s*Execution gpu time:\s*)(\s*.*)')
    pattern_gflops = re.compile(r'(\s*Execution gflops:\s*)(\s*.*)')
    pattern_batch_size = re.compile(r'-b\s+(\d+)')
    
    log.info("Parsing Results")
    
    # Setup table
    field_names = ['TestName', 'SubTests', 'BatchSize', 'nGPU', 'Result', 'Scores', 'Units', 'Flag']
    table = PrettyTable(field_names)
    
    test_results = []
    num_gpus = 1
    batch_size = default_batch_size
    
    log_file = SCRIPT_DIR / "rocfft_bench.log"
    
    try:
        with open(log_file, 'r') as log_fp:
            
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
                dimensions = test_case_match.group(2).replace(" ", "_").replace("-", "")
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
                    elif '--length' in result_line:
                        # Next test case started, this one failed
                        break
                
                # Determine if test passed or failed
                status = "PASS" if (gpu_time and gflops) else "FAIL"
                gpu_time = gpu_time or 0.0
                gflops = gflops or 0.0
                
                # Add GPU time result
                time_testname = f"rider_{subtest_id}_time"
                table.add_row([BENCHMARK_NAME, time_testname, batch_size, num_gpus, status, gpu_time, 'ms', 'L'])
                test_results.append(create_test_result(
                    BENCHMARK_NAME, time_testname, batch_size, num_gpus, status, gpu_time, "ms", "L"
                ))
                
                # Add GFLOPS result
                gflops_testname = f"rider_{subtest_id}_gflops"
                table.add_row([BENCHMARK_NAME, gflops_testname, batch_size, num_gpus, status, gflops, 'GFLOPS', 'H'])
                test_results.append(create_test_result(
                    BENCHMARK_NAME, gflops_testname, batch_size, num_gpus, status, gflops, "GFLOPS", "H"
                ))
    
    except OSError as e:
        raise ValueError(f"IO Error in Score Extractor: {e}")
    
    return test_results, table


def main():
    """Main execution function."""
    log.info("Initializing ROCfft Benchmark Test")
    
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
        test_name="rocfft_benchmark",
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
    has_fail = any(row[final_result_index] == 'FAIL' for row in final_table._rows)
    has_unknown = any(row[final_result_index] == 'UNKNOWN' for row in final_table._rows)
    
    final_status = 'FAIL' if has_fail else ('UNKNOWN' if has_unknown else 'PASS')
    if has_unknown and not has_fail:
        log.warning("Some results have UNKNOWN status (no LKG data available for comparison)")
    
    log.info(f"Final Status: {final_status}")
    
    # Return 0 only if PASS, otherwise return 1 (for FAIL or UNKNOWN)
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

