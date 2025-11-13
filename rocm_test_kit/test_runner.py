#!/usr/bin/env python3
"""
ROCm Component Test Kit - Main Test Runner

Orchestrates component-level tests for MI300/MI350 hardware.
"""
import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import from test scripts
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent
TEST_SCRIPTS_DIR = THEROCK_DIR / "build_tools" / "github_actions" / "test_executable_scripts"

sys.path.insert(0, str(TEST_SCRIPTS_DIR))

# Import our modules
from hardware_detector import detect_hardware, check_compatibility


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


logger = logging.getLogger(__name__)


class TestResult:
    """Stores result of a single component test."""

    def __init__(self, component: str):
        self.component = component
        self.status = "pending"  # pending, running, passed, failed, skipped
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.error_message: Optional[str] = None
        self.log_file: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> dict:
        return {
            'component': self.component,
            'status': self.status,
            'duration': self.duration,
            'error_message': self.error_message,
            'log_file': self.log_file
        }


class TestRunner:
    """Main test orchestration class."""

    def __init__(self, config_path: Path):
        """Initialize test runner with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def _load_config(self) -> dict:
        """Load component configuration from YAML."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_components_to_test(self,
                               components: Optional[List[str]] = None,
                               category: Optional[str] = None,
                               preset: Optional[str] = None) -> List[str]:
        """
        Determine which components to test based on arguments.

        Args:
            components: Specific component names
            category: Test a category (e.g., 'blas', 'deep_learning')
            preset: Use a preset (e.g., 'quick', 'core', 'full')

        Returns:
            List of component names to test
        """
        if preset:
            if preset not in self.config.get('presets', {}):
                raise ValueError(f"Unknown preset: {preset}")
            return self.config['presets'][preset]['components']

        if category:
            if category not in self.config.get('categories', {}):
                raise ValueError(f"Unknown category: {category}")
            return self.config['categories'][category]['components']

        if components:
            # Validate components exist
            available = self.config.get('components', {}).keys()
            invalid = [c for c in components if c not in available]
            if invalid:
                raise ValueError(f"Unknown components: {', '.join(invalid)}")
            return components

        # Default: test all
        return list(self.config.get('components', {}).keys())

    def run_test(self, component: str, test_type: str = "smoke",
                 log_dir: Optional[Path] = None) -> TestResult:
        """
        Run test for a single component.

        Args:
            component: Component name
            test_type: Test type ('smoke' or 'full')
            log_dir: Directory to store logs

        Returns:
            TestResult object
        """
        result = TestResult(component)
        comp_config = self.config['components'].get(component)

        if not comp_config:
            result.status = "skipped"
            result.error_message = f"Component {component} not found in config"
            return result

        test_script = comp_config.get('test_script')
        if not test_script:
            result.status = "skipped"
            result.error_message = f"No test script defined for {component}"
            return result

        script_path = TEST_SCRIPTS_DIR / test_script

        if not script_path.exists():
            result.status = "skipped"
            result.error_message = f"Test script not found: {script_path}"
            return result

        # Setup environment
        env = os.environ.copy()
        env['TEST_TYPE'] = test_type
        env['THEROCK_DIR'] = str(THEROCK_DIR)

        # Setup logging
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{component}_{test_type}.log"
            result.log_file = str(log_file)
        else:
            log_file = None

        logger.info(f"Running {component} ({test_type} tests)...")
        result.status = "running"
        result.start_time = time.time()

        try:
            # Determine how to run the test based on type
            if comp_config.get('test_type') == 'pytest':
                cmd = ['pytest', '-v', str(script_path)]
            else:
                cmd = ['python3', str(script_path)]

            # Run the test
            if log_file:
                with open(log_file, 'w') as f:
                    proc = subprocess.run(
                        cmd,
                        env=env,
                        cwd=THEROCK_DIR,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=3600  # 1 hour timeout
                    )
            else:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    cwd=THEROCK_DIR,
                    capture_output=True,
                    text=True,
                    timeout=3600
                )

            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            if proc.returncode == 0:
                result.status = "passed"
                logger.info(f"✓ {component} passed ({result.duration:.1f}s)")
            else:
                result.status = "failed"
                result.error_message = f"Test failed with exit code {proc.returncode}"
                logger.error(f"✗ {component} failed ({result.duration:.1f}s)")
                if not log_file and proc.stderr:
                    logger.debug(f"Error output: {proc.stderr[:500]}")

        except subprocess.TimeoutExpired:
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            result.status = "failed"
            result.error_message = "Test timed out after 1 hour"
            logger.error(f"✗ {component} timed out")

        except Exception as e:
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time if result.start_time else 0
            result.status = "failed"
            result.error_message = str(e)
            logger.error(f"✗ {component} error: {e}")

        return result

    def run_tests_sequential(self, components: List[str], test_type: str = "smoke",
                            log_dir: Optional[Path] = None) -> List[TestResult]:
        """Run tests sequentially (one at a time)."""
        results = []

        for component in components:
            result = self.run_test(component, test_type, log_dir)
            results.append(result)
            self.results.append(result)

        return results

    def run_tests_parallel(self, components: List[str], test_type: str = "smoke",
                          log_dir: Optional[Path] = None, max_workers: int = 4) -> List[TestResult]:
        """Run tests in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_component = {
                executor.submit(self.run_test, comp, test_type, log_dir): comp
                for comp in components
            }

            # Collect results as they complete
            for future in as_completed(future_to_component):
                result = future.result()
                results.append(result)
                self.results.append(result)

        return results

    def run(self, components: Optional[List[str]] = None,
            category: Optional[str] = None,
            preset: Optional[str] = None,
            test_type: str = "smoke",
            parallel: bool = False,
            max_workers: int = 4,
            log_dir: Optional[Path] = None) -> Dict:
        """
        Main test execution method.

        Args:
            components: Specific components to test
            category: Test a category
            preset: Use a preset
            test_type: 'smoke' or 'full'
            parallel: Run tests in parallel
            max_workers: Max parallel workers
            log_dir: Directory for logs

        Returns:
            Summary dictionary
        """
        self.start_time = time.time()
        self.results = []

        # Determine components to test
        components_to_test = self.get_components_to_test(components, category, preset)

        logger.info("=" * 70)
        logger.info("ROCm Component Test Kit")
        logger.info("=" * 70)
        logger.info(f"Test Type: {test_type}")
        logger.info(f"Components: {len(components_to_test)}")
        logger.info(f"Parallel: {parallel} (workers: {max_workers})" if parallel else "Sequential execution")
        logger.info("=" * 70)

        # Run tests
        if parallel:
            self.run_tests_parallel(components_to_test, test_type, log_dir, max_workers)
        else:
            self.run_tests_sequential(components_to_test, test_type, log_dir)

        self.end_time = time.time()

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict:
        """Generate test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if r.failed)
        skipped = sum(1 for r in self.results if r.status == "skipped")

        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0

        summary = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'duration': total_duration,
            'results': [r.to_dict() for r in self.results]
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print test summary to console."""
        logger.info("=" * 70)
        logger.info("Test Summary")
        logger.info("=" * 70)

        # Overall stats
        logger.info(f"Total Tests:    {summary['total']}")
        logger.info(f"Passed:         {summary['passed']} ✓")
        logger.info(f"Failed:         {summary['failed']} ✗")
        logger.info(f"Skipped:        {summary['skipped']}")
        logger.info(f"Duration:       {summary['duration']:.1f}s")
        logger.info("=" * 70)

        # Failed tests
        if summary['failed'] > 0:
            logger.info("Failed Components:")
            for result in summary['results']:
                if result['status'] == 'failed':
                    logger.error(f"  ✗ {result['component']}: {result.get('error_message', 'Unknown error')}")
                    if result.get('log_file'):
                        logger.error(f"    Log: {result['log_file']}")
            logger.info("=" * 70)

        # Success rate
        if summary['total'] > 0:
            success_rate = (summary['passed'] / (summary['total'] - summary['skipped'])) * 100 if (summary['total'] - summary['skipped']) > 0 else 0
            logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROCm Component Test Kit for MI300/MI350",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (default)
  %(prog)s --preset quick

  # Test all BLAS libraries
  %(prog)s --category blas

  # Test specific components
  %(prog)s --components rocblas hipblas miopen

  # Full test suite in parallel
  %(prog)s --preset full --test-type full --parallel

  # Test with detailed logs
  %(prog)s --preset core --log-dir ./test_logs --verbose
        """
    )

    parser.add_argument(
        '--components', '-c',
        nargs='+',
        help='Specific components to test'
    )

    parser.add_argument(
        '--category', '-C',
        help='Test a component category (blas, deep_learning, etc.)'
    )

    parser.add_argument(
        '--preset', '-p',
        choices=['quick', 'core', 'full'],
        default='quick',
        help='Use a test preset (default: quick)'
    )

    parser.add_argument(
        '--test-type', '-t',
        choices=['smoke', 'full'],
        default='smoke',
        help='Test type: smoke (fast) or full (comprehensive)'
    )

    parser.add_argument(
        '--parallel', '-P',
        action='store_true',
        help='Run tests in parallel'
    )

    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=4,
        help='Maximum parallel workers (default: 4)'
    )

    parser.add_argument(
        '--log-dir', '-l',
        type=Path,
        help='Directory to store test logs'
    )

    parser.add_argument(
        '--check-hardware',
        action='store_true',
        help='Check hardware compatibility and exit'
    )

    parser.add_argument(
        '--list-components',
        action='store_true',
        help='List all available components and exit'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check hardware compatibility
    if args.check_hardware:
        info = detect_hardware()
        print("=" * 70)
        print("Hardware Detection Results")
        print("=" * 70)
        print(info)
        print("=" * 70)
        sys.exit(0 if info.compatible else 1)

    # List components
    config_path = SCRIPT_DIR / "components.yaml"
    runner = TestRunner(config_path)

    if args.list_components:
        print("=" * 70)
        print("Available ROCm Components")
        print("=" * 70)
        for comp_name, comp_info in runner.config['components'].items():
            print(f"{comp_name:20s} - {comp_info['name']} ({comp_info['category']})")
        print("=" * 70)
        sys.exit(0)

    # Verify hardware compatibility (warn if not compatible)
    check_compatibility(verbose=args.verbose)

    # Run tests
    try:
        summary = runner.run(
            components=args.components,
            category=args.category,
            preset=args.preset if not (args.components or args.category) else None,
            test_type=args.test_type,
            parallel=args.parallel,
            max_workers=args.max_workers,
            log_dir=args.log_dir
        )

        runner.print_summary(summary)

        # Exit with error code if any tests failed
        sys.exit(1 if summary['failed'] > 0 else 0)

    except KeyboardInterrupt:
        logger.error("\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test run failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
