#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TheRock Logging Demo Test Runner
=================================

Demonstrates the logging framework by running GTest and CTest suites
with comprehensive logging and reporting.

Usage:
    python run_logging_demo.py --config build_tools/logging_demo.yaml
    python run_logging_demo.py --config logging_demo.yaml --dry-run
    python run_logging_demo.py --build-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Import TheRock logging framework
from _therock_utils.logging_config import get_logger, configure_root_logger, LogLevel


class TestResult:
    """Container for test execution results"""
    
    def __init__(self, name: str, success: bool, duration: float, 
                 output: str = "", error: str = ""):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output
        self.error = error


class LoggingDemoRunner:
    """Main test runner with logging integration"""
    
    def __init__(self, config_path: Path):
        """Initialize the test runner"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = None
        self.results: List[TestResult] = []
        self.start_time = None
        
        # Initialize logging
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging based on YAML config"""
        log_config = self.config.get('logging', {})
        
        # Get log level
        level_str = log_config.get('level', 'INFO')
        level = getattr(LogLevel, level_str.upper(), LogLevel.INFO)
        
        # Get log file path
        log_file = None
        if log_config.get('log_file', {}).get('enabled', False):
            log_file = log_config['log_file'].get('path')
            if log_file:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        use_colors = log_config.get('use_colors', True)
        configure_root_logger(level=level, log_file=log_file, use_colors=use_colors)
        
        # Create component logger
        self.logger = get_logger(__name__, component="test_runner", operation="demo")
        self.logger.info("Logging configured successfully")
        self.logger.debug(f"Log level: {level_str}")
        if log_file:
            self.logger.debug(f"Log file: {log_file}")
    
    def _setup_environment(self):
        """Setup environment variables from config"""
        env_config = self.config.get('environment', {})
        
        # Add custom environment variables
        variables = env_config.get('variables', {})
        for key, value in variables.items():
            # Expand environment variables in values
            expanded_value = os.path.expandvars(value)
            os.environ[key] = expanded_value
            self.logger.debug(f"Set environment: {key}={expanded_value}")
        
        # Add paths to PATH
        paths = env_config.get('paths', [])
        if paths:
            path_separator = ';' if sys.platform == 'win32' else ':'
            current_path = os.environ.get('PATH', '')
            expanded_paths = [os.path.expandvars(p) for p in paths]
            new_path = path_separator.join(expanded_paths + [current_path])
            os.environ['PATH'] = new_path
            self.logger.debug(f"Updated PATH with {len(expanded_paths)} additional paths")
    
    def build_tests(self) -> bool:
        """Build test executables using CMake"""
        build_config = self.config.get('build', {})
        
        if not build_config.get('enabled', False):
            self.logger.info("Build step disabled, skipping...")
            return True
        
        self.logger.info("=" * 70)
        self.logger.info("Starting build process")
        self.logger.info("=" * 70)
        
        build_dir = Path(build_config.get('build_directory', 'build'))
        build_dir.mkdir(parents=True, exist_ok=True)
        
        cmake_config = build_config.get('cmake', {})
        
        with self.logger.timed_operation("cmake_configure"):
            # Configure CMake
            configure_cmd = [
                'cmake',
                '-S', 'tests/gtest_samples',
                '-B', str(build_dir),
                '-G', cmake_config.get('generator', 'Ninja'),
                f"-DCMAKE_BUILD_TYPE={cmake_config.get('build_type', 'Debug')}"
            ]
            
            # Add additional options
            options = cmake_config.get('options', [])
            configure_cmd.extend(options)
            
            self.logger.debug(f"CMake configure command: {' '.join(configure_cmd)}")
            result = subprocess.run(configure_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error("CMake configuration failed!")
                self.logger.error(result.stderr)
                return False
            
            self.logger.info("CMake configuration successful")
        
        with self.logger.timed_operation("cmake_build"):
            # Build tests
            build_cmd = ['cmake', '--build', str(build_dir)]
            
            parallel_jobs = build_config.get('parallel_jobs', 0)
            if parallel_jobs > 0:
                build_cmd.extend(['--parallel', str(parallel_jobs)])
            
            targets = build_config.get('targets', [])
            for target in targets:
                build_cmd.extend(['--target', target])
            
            self.logger.debug(f"CMake build command: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error("Build failed!")
                self.logger.error(result.stderr)
                return False
            
            self.logger.info("Build successful")
        
        return True
    
    def run_gtest(self) -> List[TestResult]:
        """Run GTest executables"""
        gtest_config = self.config.get('tests', {}).get('gtest', {})
        
        if not gtest_config.get('enabled', False):
            self.logger.info("GTest execution disabled, skipping...")
            return []
        
        self.logger.info("=" * 70)
        self.logger.info("Running GTest executables")
        self.logger.info("=" * 70)
        
        results = []
        test_executables = gtest_config.get('test_executables', [])
        working_dir = Path(self.config.get('tests', {}).get('working_directory', 'build'))
        
        # Create test results directory
        results_dir = working_dir / 'test_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        for test_config in test_executables:
            name = test_config['name']
            path = test_config['path']
            args = test_config.get('args', [])
            timeout = test_config.get('timeout', 60)
            
            test_logger = get_logger(__name__, component="gtest", operation=name)
            test_logger.info(f"Starting test: {name}")
            
            # Build full command
            test_executable = working_dir / path
            if sys.platform == 'win32':
                test_executable = test_executable.with_suffix('.exe')
            
            cmd = [str(test_executable)] + args
            test_logger.debug(f"Command: {' '.join(cmd)}")
            
            try:
                with test_logger.timed_operation(f"execute_{name}"):
                    start = time.time()
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=str(working_dir)
                    )
                    duration = time.time() - start
                
                success = result.returncode == 0
                
                if success:
                    test_logger.info(f"✅ Test passed: {name}")
                else:
                    test_logger.error(f"❌ Test failed: {name}")
                    test_logger.error(f"Exit code: {result.returncode}")
                
                # Show output in debug mode
                if result.stdout:
                    test_logger.debug("STDOUT:")
                    for line in result.stdout.splitlines()[:20]:  # Limit output
                        test_logger.debug(f"  {line}")
                
                if result.stderr:
                    test_logger.warning("STDERR:")
                    for line in result.stderr.splitlines()[:20]:  # Limit output
                        test_logger.warning(f"  {line}")
                
                results.append(TestResult(
                    name=name,
                    success=success,
                    duration=duration,
                    output=result.stdout,
                    error=result.stderr
                ))
                
            except subprocess.TimeoutExpired:
                test_logger.error(f"❌ Test timed out: {name}")
                results.append(TestResult(
                    name=name,
                    success=False,
                    duration=timeout,
                    output="",
                    error=f"Test timed out after {timeout} seconds"
                ))
            except Exception as e:
                test_logger.log_exception(e, f"Test execution failed: {name}")
                results.append(TestResult(
                    name=name,
                    success=False,
                    duration=0,
                    output="",
                    error=str(e)
                ))
        
        return results
    
    def run_ctest(self) -> List[TestResult]:
        """Run tests using CTest"""
        ctest_config = self.config.get('tests', {}).get('ctest', {})
        
        if not ctest_config.get('enabled', False):
            self.logger.info("CTest execution disabled, skipping...")
            return []
        
        self.logger.info("=" * 70)
        self.logger.info("Running CTest")
        self.logger.info("=" * 70)
        
        working_dir = Path(self.config.get('tests', {}).get('working_directory', 'build'))
        
        ctest_logger = get_logger(__name__, component="ctest", operation="run")
        
        # Build CTest command
        cmd = ['ctest']
        cmd.extend(ctest_config.get('args', []))
        
        timeout = ctest_config.get('timeout', 300)
        
        ctest_logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            with ctest_logger.timed_operation("ctest_execution"):
                start = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(working_dir)
                )
                duration = time.time() - start
            
            success = result.returncode == 0
            
            if success:
                ctest_logger.info("✅ CTest passed")
            else:
                ctest_logger.error("❌ CTest failed")
            
            # Show output
            if result.stdout:
                ctest_logger.info("CTest output:")
                for line in result.stdout.splitlines():
                    ctest_logger.info(f"  {line}")
            
            return [TestResult(
                name="ctest",
                success=success,
                duration=duration,
                output=result.stdout,
                error=result.stderr
            )]
            
        except subprocess.TimeoutExpired:
            ctest_logger.error("❌ CTest timed out")
            return [TestResult(
                name="ctest",
                success=False,
                duration=timeout,
                output="",
                error=f"CTest timed out after {timeout} seconds"
            )]
        except Exception as e:
            ctest_logger.log_exception(e, "CTest execution failed")
            return [TestResult(
                name="ctest",
                success=False,
                duration=0,
                output="",
                error=str(e)
            )]
    
    def generate_report(self):
        """Generate test execution report"""
        self.logger.info("=" * 70)
        self.logger.info("Test Execution Summary")
        self.logger.info("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.results)
        
        self.logger.info(f"Total tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Total duration: {total_duration:.2f}s")
        
        if failed_tests > 0:
            self.logger.error("Failed tests:")
            for result in self.results:
                if not result.success:
                    self.logger.error(f"  - {result.name}")
        
        # Generate JSON summary
        reporting_config = self.config.get('reporting', {})
        json_config = reporting_config.get('json_summary', {})
        
        if json_config.get('enabled', False):
            summary = {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'total_duration': total_duration,
                'results': [
                    {
                        'name': r.name,
                        'success': r.success,
                        'duration': r.duration
                    }
                    for r in self.results
                ]
            }
            
            output_path = Path(json_config.get('output_path', 'test_results/summary.json'))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"JSON summary written to: {output_path}")
        
        self.logger.info("=" * 70)
        
        return failed_tests == 0
    
    def run(self, build_only: bool = False, dry_run: bool = False) -> bool:
        """Run the complete test suite"""
        self.start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("TheRock Logging Demo - Test Runner")
        self.logger.info("=" * 70)
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Platform: {sys.platform}")
        self.logger.info(f"Python: {sys.version.split()[0]}")
        
        if dry_run:
            self.logger.warning("DRY RUN MODE - No tests will be executed")
        
        try:
            # Setup environment
            self._setup_environment()
            
            # Build tests
            if not dry_run:
                if not self.build_tests():
                    self.logger.error("Build failed, aborting...")
                    return False
            
            if build_only:
                self.logger.info("Build-only mode, skipping test execution")
                return True
            
            # Run tests
            if not dry_run:
                gtest_results = self.run_gtest()
                self.results.extend(gtest_results)
                
                ctest_results = self.run_ctest()
                self.results.extend(ctest_results)
            
            # Generate report
            if self.results:
                success = self.generate_report()
            else:
                self.logger.warning("No tests were executed")
                success = True
            
            total_duration = time.time() - self.start_time
            self.logger.info(f"Total execution time: {total_duration:.2f}s")
            
            if success:
                self.logger.info("✅ All tests passed!")
            else:
                self.logger.error("❌ Some tests failed!")
            
            return success
            
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            return False
        except Exception as e:
            self.logger.log_exception(e, "Unexpected error during test execution")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TheRock Logging Demo Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('build_tools/logging_demo.yaml'),
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--build-only',
        action='store_true',
        help='Only build tests, do not run them'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1
    
    runner = LoggingDemoRunner(args.config)
    success = runner.run(build_only=args.build_only, dry_run=args.dry_run)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

