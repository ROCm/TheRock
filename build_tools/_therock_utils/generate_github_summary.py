#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GitHub Actions Summary Generator
=================================
Generates markdown summary for GitHub Actions from test results.

Usage:
    python3 generate_github_summary.py --results-file test_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def generate_github_summary(results_file: Path):
    """
    Generate markdown summary for GitHub Actions.
    
    Parameters:
    -----------
    results_file : Path
        Path to aggregated test results JSON file
    """
    with open(results_file) as f:
        results = json.load(f)
    
    summary = results['summary']
    
    # Determine overall status
    overall_status = "✅ PASSED" if summary['failed'] == 0 else "❌ FAILED"
    status_emoji = "✅" if summary['failed'] == 0 else "❌"
    
    # Generate markdown
    print(f"# {status_emoji} Test Results Summary")
    print()
    print("## Overall Results")
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| **Status** | **{overall_status}** |")
    print(f"| Total Tests | {summary['total_tests']} |")
    print(f"| ✅ Passed | {summary['passed']} |")
    print(f"| ❌ Failed | {summary['failed']} |")
    print(f"| ⏭️ Skipped | {summary['skipped']} |")
    print(f"| Success Rate | {summary['success_rate_pct']:.2f}% |")
    print(f"| Total Duration | {summary['total_duration_sec']:.2f}s |")
    print()
    
    # Test suites breakdown
    print("## Test Suites")
    print()
    print("| Component | Framework | Type | Total | Passed | Failed | Skipped | Status |")
    print("|-----------|-----------|------|-------|--------|--------|---------|--------|")
    
    for suite in results.get('test_suites', []):
        suite_status = "✅ PASS" if suite.get('failed', 0) == 0 else "❌ FAIL"
        print(f"| {suite.get('component', 'Unknown')} | "
              f"{suite.get('framework', 'Unknown').upper()} | "
              f"{suite.get('test_type', 'Unknown')} | "
              f"{suite.get('total', 0)} | "
              f"{suite.get('passed', 0)} | "
              f"{suite.get('failed', 0)} | "
              f"{suite.get('skipped', 0)} | "
              f"{suite_status} |")
    
    print()
    
    # Failed tests details
    failed_tests_exist = False
    for suite in results.get('test_suites', []):
        if suite.get('failed_tests'):
            failed_tests_exist = True
            break
    
    if failed_tests_exist:
        print("## Failed Tests Details")
        print()
        for suite in results.get('test_suites', []):
            failed_tests = suite.get('failed_tests', [])
            if failed_tests:
                print(f"### {suite.get('component', 'Unknown')} - {suite.get('framework', 'Unknown').upper()}")
                print()
                for test in failed_tests:
                    print(f"- ❌ `{test}`")
                print()
    
    # Links
    run_id = os.getenv('GITHUB_RUN_ID', 'unknown')
    repo = os.getenv('GITHUB_REPOSITORY', 'ROCm/TheRock')
    
    print("## Links")
    print()
    print(f"- 📊 [Download HTML Report](https://github.com/{repo}/actions/runs/{run_id})")
    print(f"- 📄 [Download JSON Report](https://github.com/{repo}/actions/runs/{run_id})")
    print(f"- 🔗 [View Workflow Run](https://github.com/{repo}/actions/runs/{run_id})")
    print()
    
    print(f"---")
    print(f"*Generated: {summary['timestamp']}*")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate GitHub Actions summary from test results"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Path to aggregated test results JSON file"
    )
    
    args = parser.parse_args(argv)
    
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)
    
    generate_github_summary(args.results_file)


if __name__ == "__main__":
    main(sys.argv[1:])

