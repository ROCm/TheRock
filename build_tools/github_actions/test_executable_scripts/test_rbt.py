#!/usr/bin/env python3
"""
RBT Test Runner - Execute ROCm Bandwidth Test validation suite

Usage:
  # Run with default settings (uses THEROCK_BIN_DIR env var)
  python test_rbt.py

  # Run with custom binary path
  python test_rbt.py --bin-dir /opt/rocm/bin

  # Smoke test (quick validation)
  python test_rbt.py --test-type smoke

  # Full test suite
  python test_rbt.py --test-type full

Environment Variables:
  THEROCK_BIN_DIR  - Path to directory containing rocm-bandwidth-test binary
  TEST_TYPE        - Test type: smoke, standard, full (default: full)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def get_gpu_info(env):
    """Get GPU count and architecture info"""
    gpu_count = 0
    has_mi300 = False
    
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            gpu_count = sum(1 for line in lines if 'Device Type:' in line and 'GPU' in line)
            has_mi300 = 'gfx94' in result.stdout
    except Exception as e:
        log.warning(f"Could not get GPU info: {e}")
    
    return max(gpu_count, 1), has_mi300


def run_test(name, cmd, env, timeout, cwd):
    """Run a single test and return (passed, output)"""
    log.info(f"Running {name}: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        # Check for error patterns in output
        error_patterns = ["error:", "failed", "not found", "cannot"]
        output = result.stdout + result.stderr
        has_error = any(p in output.lower() for p in error_patterns)
        
        if result.returncode != 0:
            log.error(f"✗ {name} failed (exit code: {result.returncode})")
            if result.stderr:
                log.error(f"stderr: {result.stderr[:500]}")
            return False, output
        elif has_error:
            log.warning(f"⚠ {name} completed but output contains errors")
            return False, output
        else:
            log.info(f"✓ {name} passed")
            return True, output
            
    except subprocess.TimeoutExpired:
        log.error(f"✗ {name} timed out after {timeout}s")
        return False, "TIMEOUT"


def main():
    parser = argparse.ArgumentParser(
        description="Run ROCm Bandwidth Test validation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bin-dir",
        help="Path to RBT binary directory (default: $THEROCK_BIN_DIR)"
    )
    parser.add_argument(
        "--test-type",
        choices=["smoke", "standard", "full"],
        default=os.getenv("TEST_TYPE", "full"),
        help="Test type to run (default: full)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show test output"
    )
    
    args = parser.parse_args()
    
    # Determine binary directory
    bin_dir = args.bin_dir or os.getenv("THEROCK_BIN_DIR")
    if not bin_dir:
        log.error("Binary directory not specified. Use --bin-dir or set THEROCK_BIN_DIR")
        sys.exit(1)
    
    bin_dir = Path(bin_dir).resolve()
    rbt_binary = bin_dir / "rocm-bandwidth-test"
    
    if not rbt_binary.exists():
        log.error(f"RBT binary not found at {rbt_binary}")
        sys.exit(1)
    
    # Setup environment
    lib_dir = bin_dir.parent / "lib"
    lib64_dir = bin_dir.parent / "lib64"
    
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{lib64_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Get GPU info
    num_gpus, has_mi300 = get_gpu_info(env)
    log.info(f"Detected {num_gpus} GPU(s), MI300: {has_mi300}")
    
    # Define test suite
    # Format: (command, requirement, extra_env, timeout_override)
    rbt = str(rbt_binary)
    timeout = args.timeout
    
    all_tests = {
        # Basic tests
        "Plugin_List": (f"{rbt} plugin --list", None, None, 60),
        "Plugin_Info": (f"{rbt} plugin -i", None, None, 60),
        "Hello": (f"{rbt} run Hello", None, None, 60),
        
        # TransferBench tests
        "TB": (f"{rbt} run tb", None, None, timeout),
        "TB_P2P": (f"{rbt} run tb p2p", "multi_gpu", None, timeout),
        "TB_Scaling": (f"{rbt} run tb scaling", "multi_gpu", None, timeout),
        "TB_Schmoo": (f"{rbt} run tb schmoo", "multi_gpu", None, timeout),
        "TB_Sweep": (f"{rbt} run tb sweep", None, {"SWEEP_TIME_LIMIT": "10"}, timeout),
        "TB_RSweep": (f"{rbt} run tb rsweep", None, {"SWEEP_TIME_LIMIT": "10"}, timeout),
        "TB_One2All": (f"{rbt} run tb one2all", "multi_gpu", {"NUM_GPU_DEVICES": "2"}, timeout),
        "TB_Healthcheck": (f"{rbt} run tb healthcheck", "mi300", None, timeout),
    }
    
    # Select tests based on test type
    if args.test_type == "smoke":
        tests_to_run = ["Plugin_List", "Hello"]
    elif args.test_type == "standard":
        tests_to_run = ["Plugin_List", "Plugin_Info", "Hello", "TB"]
    else:  # full
        tests_to_run = list(all_tests.keys())
    
    # Run tests
    log.info("=" * 60)
    log.info(f"RBT Test Suite - {args.test_type.upper()}")
    log.info("=" * 60)
    log.info(f"Binary: {rbt_binary}")
    log.info(f"Tests:  {len(tests_to_run)}")
    log.info("=" * 60)
    
    results = {"passed": 0, "skipped": 0, "failed": 0}
    cwd = bin_dir.parent
    
    for test_name in tests_to_run:
        cmd, requirement, extra_env, test_timeout = all_tests[test_name]
        
        # Check requirements
        skip_reason = None
        if requirement == "multi_gpu" and num_gpus < 2:
            skip_reason = f"requires 2+ GPUs (have {num_gpus})"
        elif requirement == "mi300" and not has_mi300:
            skip_reason = "requires MI300 GPU"
        
        if skip_reason:
            log.info(f"⊘ Skipping {test_name}: {skip_reason}")
            results["skipped"] += 1
            continue
        
        # Setup test environment
        test_env = env.copy()
        if extra_env:
            test_env.update(extra_env)
        
        # Run test
        passed, output = run_test(test_name, cmd, test_env, test_timeout, cwd)
        
        if args.verbose and output:
            log.info(f"Output:\n{output[:1000]}")
        
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("RBT Test Results")
    log.info("=" * 60)
    log.info(f"  Passed:  {results['passed']}")
    log.info(f"  Skipped: {results['skipped']}")
    log.info(f"  Failed:  {results['failed']}")
    log.info("=" * 60)
    
    if results["failed"] > 0:
        log.error(f"✗ {results['failed']} test(s) failed!")
        sys.exit(1)
    
    log.info("✓ All tests completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
