#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test Report Generator
=====================
Aggregates test results from multiple test suites and generates comprehensive reports.

Usage:
    python3 generate_test_report.py \
        --results-dir ./test_results \
        --output-html report.html \
        --output-json report.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def aggregate_test_results(results_dir: Path) -> Dict:
    """
    Aggregate all test result JSON files from a directory.
    
    Parameters:
    -----------
    results_dir : Path
        Directory containing test result JSON files
    
    Returns:
    --------
    dict
        Aggregated test results
    """
    results = {
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "success_rate_pct": 0.0,
            "total_duration_sec": 0.0,
            "timestamp": datetime.now().isoformat()
        },
        "test_suites": []
    }
    
    # Collect all test_results*.json files
    result_files = list(results_dir.glob("test_results_*.json"))
    
    if not result_files:
        print(f"Warning: No test result files found in {results_dir}")
        return results
    
    print(f"Found {len(result_files)} test result files")
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
                
                # Add to test suites
                results["test_suites"].append(data)
                
                # Aggregate summary
                results["summary"]["total_tests"] += data.get("total", 0)
                results["summary"]["passed"] += data.get("passed", 0)
                results["summary"]["failed"] += data.get("failed", 0)
                results["summary"]["skipped"] += data.get("skipped", 0)
                results["summary"]["total_duration_sec"] += data.get("duration_sec", 0)
                
                print(f"  Processed: {result_file.name}")
        except Exception as e:
            print(f"  Error processing {result_file.name}: {e}")
    
    # Calculate success rate
    if results["summary"]["total_tests"] > 0:
        results["summary"]["success_rate_pct"] = round(
            (results["summary"]["passed"] / results["summary"]["total_tests"]) * 100, 2
        )
    
    return results


def generate_html_report(results: Dict, output_file: Path):
    """
    Generate HTML test report.
    
    Parameters:
    -----------
    results : dict
        Aggregated test results
    output_file : Path
        Output HTML file path
    """
    summary = results["summary"]
    
    # Generate test suites table rows
    suite_rows = ""
    for suite in results["test_suites"]:
        status_color = "green" if suite.get("failed", 0) == 0 else "red"
        suite_rows += f"""
        <tr>
            <td>{suite.get('component', 'Unknown')}</td>
            <td>{suite.get('framework', 'Unknown').upper()}</td>
            <td>{suite.get('test_type', 'Unknown')}</td>
            <td>{suite.get('total', 0)}</td>
            <td class="pass">{suite.get('passed', 0)}</td>
            <td class="fail">{suite.get('failed', 0)}</td>
            <td>{suite.get('skipped', 0)}</td>
            <td style="color: {status_color}; font-weight: bold;">
                {'✅ PASS' if suite.get('failed', 0) == 0 else '❌ FAIL'}
            </td>
            <td>{suite.get('duration_sec', 0):.2f}s</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Results - {summary['timestamp']}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-card.passed {{
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            }}
            .metric-card.failed {{
                background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            }}
            .metric-card.rate {{
                background: linear-gradient(135deg, #4776e6 0%, #8e54e9 100%);
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
            table {{ 
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            .pass {{ color: #4CAF50; font-weight: bold; }}
            .fail {{ color: #f44336; font-weight: bold; }}
            .timestamp {{
                color: #888;
                font-size: 14px;
                margin-top: 20px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Test Results Summary</h1>
            
            <div class="summary">
                <div class="metric-card">
                    <div class="metric-label">Total Tests</div>
                    <div class="metric-value">{summary['total_tests']}</div>
                </div>
                <div class="metric-card passed">
                    <div class="metric-label">✅ Passed</div>
                    <div class="metric-value">{summary['passed']}</div>
                </div>
                <div class="metric-card failed">
                    <div class="metric-label">❌ Failed</div>
                    <div class="metric-value">{summary['failed']}</div>
                </div>
                <div class="metric-card rate">
                    <div class="metric-label">Success Rate</div>
                    <div class="metric-value">{summary['success_rate_pct']:.1f}%</div>
                </div>
            </div>
            
            <h2>📋 Test Suites</h2>
            <table>
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Framework</th>
                        <th>Type</th>
                        <th>Total</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Skipped</th>
                        <th>Status</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {suite_rows}
                </tbody>
            </table>
            
            <div class="timestamp">
                Generated: {summary['timestamp']}<br>
                Total Duration: {summary['total_duration_sec']:.2f} seconds
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_file}")


def main(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test reports from test results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing test result JSON files"
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default="test_report.html",
        help="Output HTML report file (default: test_report.html)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default="test_report.json",
        help="Output JSON report file (default: test_report.json)"
    )
    
    args = parser.parse_args(argv)
    
    # Validate results directory
    if not args.results_dir.exists():
        print(f"Error: Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    print(f"Aggregating test results from: {args.results_dir}")
    
    # Aggregate results
    results = aggregate_test_results(args.results_dir)
    
    # Save JSON report
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON report saved: {args.output_json}")
    
    # Generate HTML report
    generate_html_report(results, args.output_html)
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    print(f"Total Tests:   {summary['total_tests']}")
    print(f"Passed:        {summary['passed']}")
    print(f"Failed:        {summary['failed']}")
    print(f"Skipped:       {summary['skipped']}")
    print(f"Success Rate:  {summary['success_rate_pct']:.2f}%")
    print(f"Duration:      {summary['total_duration_sec']:.2f}s")
    print("="*60)
    
    # Exit with non-zero if any tests failed
    if summary['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])

