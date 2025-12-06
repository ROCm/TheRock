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

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import TestClient
from utils.logger import log

# Constants
BENCHMARK_NAME = 'rocsolver'

# Environment variables
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
ARTIFACT_RUN_ID = os.getenv("ARTIFACT_RUN_ID")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent


def run_benchmarks() -> None:
    """Run ROCsolver benchmarks and save output to log file."""
    
    log_file = SCRIPT_DIR / "rocsolver_bench.log"
    log.info("Running ROCsolver Benchmarks")
    
    # ROCsolver benchmark command
    
    with open(log_file, "w+") as f:
        cmd = [
            f"{THEROCK_BIN_DIR}/rocsolver-bench",
            "-f", "gesvd",
            "--precision", "d",
            "--left_svect", "S",
            "--right_svect", "S",
            "-m", "250",
            "-n", "250"
        ]
        
        # Set environment variable
        env = os.environ.copy()
        env['HIP_VISIBLE_DEVICES'] = '0'
        log.info(f"++ Exec [{THEROCK_DIR}]$ HIP_VISIBLE_DEVICES=0 {shlex.join(cmd)}")
        f.write(f"HIP_VISIBLE_DEVICES=0 {shlex.join(cmd)}\n")
        
        process = subprocess.Popen(
            cmd,
            cwd=THEROCK_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        for line in process.stdout:
            log.info(line.strip())
            f.write(f"{line}\n")
        
        process.wait()
    
    log.info("Benchmark execution complete")


def create_test_result(test_name: str,
                       subtest_name: str,
                       num_gpus: int,
                       status: str,
                       score: float,
                       unit: str,
                       flag: str) -> Dict[str, Any]:
    """Create a test result dictionary.
    
    Args:
        test_name: Benchmark name
        subtest_name: Specific test identifier
        num_gpus: Number of GPUs used
        status: Test status ('PASS' or 'FAIL')
        score: Performance metric value
        unit: Unit of measurement (e.g., 'us')
        flag: 'H' (higher better) or 'L' (lower better)
        
    Returns:
        Dict[str, Any]: Test result dictionary with test data and configuration
    """
    return {
        "test_name": test_name,
        "subtest": subtest_name,
        "batch_size": 0,
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
            "batch_size": 0,
            "ngpu": num_gpus
        }
    }


def parse_results() -> Tuple[List[Dict[str, Any]], PrettyTable]:
    """Parse benchmark results from log file.
    
    Returns:
        tuple: (test_results list, PrettyTable object)
    """
    # Regex patterns for parsing
    # Pattern to match timing results: "cpu_time_us  gpu_time_us"
    gpu_pattern = re.compile(r'^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$')
    # Pattern to detect device ID
    device_pattern = re.compile(r'Device\s+ID\s*\d+')
    
    log.info("Parsing Results")
    # Setup table
    field_names = ['TestName', 'SubTests', 'nGPU', 'Result', 'Scores', 'Units', 'Flag']
    table = PrettyTable(field_names)
    
    test_results = []
    score = 0
    num_gpus = 0

    log_file = SCRIPT_DIR / "rocsolver_bench.log"
    
    # Test configuration from command
    subtest_name = "rocsolver_gesvd_d_S_S_250_250"
    
    try:
        with open(log_file, 'r') as fp:
            for line in fp:
                # Check for GPU device lines
                if re.search(device_pattern, line):
                    num_gpus += 1
                
                # Extract timing score - try new 2-column format first
                gpu_match = re.search(gpu_pattern, line)
                if gpu_match:
                    # Group 2 contains gpu_time_us in new format
                    score = float(gpu_match.group(2))
                    log.debug(f"Matched 2-column format: cpu_time={gpu_match.group(1)}, gpu_time={gpu_match.group(2)}")
        
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
        table.add_row([BENCHMARK_NAME, subtest_name, num_gpus, status, score, 'us', 'L'])
        
        # Add to test results
        test_results.append(create_test_result(
            BENCHMARK_NAME,
            subtest_name,
            num_gpus,
            status,
            score,
            "us",
            "L"  # Lower is better for time
        ))
    
    except OSError as e:
        raise ValueError(f"IO Error in Score Extractor: {e}")
    
    return test_results, table


def main():
    """Main execution function."""
    log.info("Initializing ROCsolver Benchmark Test")
    
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
        test_name="rocsolver_benchmark",
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
