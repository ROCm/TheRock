"""
hipBLASLt Benchmark Test

Runs hipBLASLt benchmarks, collects results, and uploads to results API.
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
BENCHMARK_NAME = 'hipblaslt'

# Environment variables
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
ARTIFACT_RUN_ID = os.getenv("ARTIFACT_RUN_ID")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent


def run_benchmarks() -> None:
    """Run hipBLASLt benchmarks and save output to log file."""
    BETA = 0
    ITERATIONS = 1000
    COLD_ITERATIONS = 1000
    PRECISION = "f16_r"
    COMPUTE_TYPE = "f32_r"
    ACTIVATION_TYPE = "none"
    
    # Load benchmark configuration
    config_file = SCRIPT_DIR.parent / 'configs/benchmarks/hipblaslt.json'
    with open(config_file, "r") as f:
        config_data = json.load(f)
    
    # Combine test configurations: (shapes_list, transB_value)
    test_configs = [
        (config_data.get("input_shapes", []), "N"),      # transB = N
        (config_data.get("ntinput_shapes", []), "T")     # transB = T
    ]
    
    log_file = SCRIPT_DIR / "hipblaslt_bench.log"
    
    log.info("Running hipBLASLt Benchmarks")
    
    with open(log_file, "w+") as f:
        for shapes_list, transB in test_configs:
            for input_shape in shapes_list:
                M, N, K, B = input_shape.split()
                
                # Calculate matrix strides
                stride_a = int(M) * int(K)
                stride_b = int(K) * int(N)
                stride_c = int(M) * int(N)
                stride_d = int(M) * int(N)
                
                cmd = [
                    f"{THEROCK_BIN_DIR}/hipblaslt-bench",
                    "-v",
                    "--transA", "N",
                    "--transB", transB,
                    "-m", M,
                    "-n", N,
                    "-k", K,
                    "--alpha", "1",
                    "--lda", M,
                    "--stride_a", str(stride_a),
                    "--beta", str(BETA),
                    "--ldb", K,
                    "--stride_b", str(stride_b),
                    "--ldc", M,
                    "--stride_c", str(stride_c),
                    "--ldd", M,
                    "--stride_d", str(stride_d),
                    "--precision", PRECISION,
                    "--compute_type", COMPUTE_TYPE,
                    "--activation_type", ACTIVATION_TYPE,
                    "--iters", str(ITERATIONS),
                    "--cold_iters", str(COLD_ITERATIONS),
                    "--batch_count", B
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
                       params: Dict[str, str],
                       num_gpus: int,
                       status: str,
                       score: float,
                       unit: str,
                       flag: str) -> Dict[str, Any]:
    """Create a test result dictionary.
    
    Args:
        test_name: Benchmark name
        subtest_name: Specific test identifier
        params: Dictionary of test parameters (transA, transB, m, n, k, etc.)
        num_gpus: Number of GPUs used
        status: Test status ('PASS' or 'FAIL')
        score: Performance metric value (Gflops)
        unit: Unit of measurement (e.g., 'Gflops')
        flag: 'H' (higher better) or 'L' (lower better)
        
    Returns:
        Dict[str, Any]: Test result dictionary with test data and configuration
    """
    try:
        batch_count = int(params.get("batch_count", 0))
    except (ValueError, TypeError):
        batch_count = 0
    
    return {
        "test_name": test_name,
        "subtest": subtest_name,
        "batch_size": batch_count,
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
            "batch_count": batch_count,
            "ngpu": num_gpus
        }
    }


def parse_results() -> Tuple[List[Dict[str, Any]], PrettyTable]:
    """Parse benchmark results from log file.
    
    Returns:
        tuple: (test_results list, PrettyTable object)
    """
    log.info("Parsing Results")
    
    log_file = SCRIPT_DIR / "hipblaslt_bench.log"
    field_names = ['TestName', 'SubTests', 'BatchCount', 'nGPU', 'Result', 'Scores', 'Units', 'Flag']
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
        with open(log_file, 'r') as log_fp:
            data = log_fp.readlines()
        
        # Find CSV header line
        header_line = None
        header_index = -1
        
        for i, line in enumerate(data):
            if "transA" in line and "transB" in line and "hipblaslt-Gflops" in line:
                header_line = line.replace('[0]:', '').strip().split(',')
                header_index = i
                break
        
        if not header_line or header_index == -1:
            log.warning("CSV header not found in log file")
            return test_results, table
        
        for line in data[header_index + 1:]:
            line = line.strip()
            
            # Skip empty or header lines
            if not line or len(line.split(',')) < 2 or "transA" in line or "transB" in line:
                continue
            
            # Remove [0]: prefix and parse values
            line = re.sub(r'^\[\d+\]:\s*', '', line)
            values = line.split(',')
            
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
            
            table.add_row([BENCHMARK_NAME, subtest_name, batch_count, num_gpus, status, score, 'Gflops', 'H'])
            test_results.append(create_test_result(
                BENCHMARK_NAME, subtest_name, params, num_gpus, status, score, "Gflops", "H"
            ))
    
    except FileNotFoundError:
        log.error(f"Log file not found: {log_file}")
    except OSError as e:
        log.error(f"Failed to read log file: {e}")
        raise
    
    return test_results, table


def main():
    """Main execution function."""
    log.info("Initializing hipBLASLt Benchmark Test")
    
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
        test_name="hipblaslt_benchmark",
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
