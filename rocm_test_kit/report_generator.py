"""
Report generator for ROCm Test Kit.

Generates HTML and JSON reports from test results.
"""
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_html_report(summary: Dict, output_path: Path, hardware_info: Dict = None):
    """
    Generate an HTML report from test results.

    Args:
        summary: Test summary dictionary
        output_path: Path to output HTML file
        hardware_info: Optional hardware information
    """
    # Calculate stats
    total = summary['total']
    passed = summary['passed']
    failed = summary['failed']
    skipped = summary['skipped']
    duration = summary['duration']

    success_rate = (passed / (total - skipped)) * 100 if (total - skipped) > 0 else 0

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROCm Component Test Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f7f9fc;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card.passed .value {{ color: #10b981; }}
        .stat-card.failed .value {{ color: #ef4444; }}
        .stat-card.total .value {{ color: #667eea; }}
        .stat-card.skipped .value {{ color: #f59e0b; }}

        .success-bar {{
            padding: 20px 30px;
            background: white;
        }}

        .progress-bar {{
            width: 100%;
            height: 40px;
            background: #e5e7eb;
            border-radius: 20px;
            overflow: hidden;
            position: relative;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}

        .hardware-info {{
            padding: 20px 30px;
            background: #f7f9fc;
            border-top: 1px solid #e5e7eb;
        }}

        .hardware-info h2 {{
            margin-bottom: 15px;
            color: #333;
        }}

        .hardware-info ul {{
            list-style: none;
            padding-left: 20px;
        }}

        .hardware-info li {{
            padding: 5px 0;
            color: #666;
        }}

        .results {{
            padding: 30px;
        }}

        .results h2 {{
            margin-bottom: 20px;
            color: #333;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        thead {{
            background: #667eea;
            color: white;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
        }}

        tbody tr:nth-child(even) {{
            background: #f7f9fc;
        }}

        tbody tr:hover {{
            background: #e5e7eb;
        }}

        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85em;
            text-transform: uppercase;
        }}

        .status.passed {{
            background: #d1fae5;
            color: #065f46;
        }}

        .status.failed {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .status.skipped {{
            background: #fef3c7;
            color: #92400e;
        }}

        .footer {{
            padding: 20px 30px;
            background: #f7f9fc;
            border-top: 1px solid #e5e7eb;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        .log-link {{
            color: #667eea;
            text-decoration: none;
        }}

        .log-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ROCm Component Test Report</h1>
            <div class="subtitle">MI300/MI350 Hardware Test Results</div>
            <div class="subtitle">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="summary">
            <div class="stat-card total">
                <div class="label">Total Tests</div>
                <div class="value">{total}</div>
            </div>
            <div class="stat-card passed">
                <div class="label">Passed</div>
                <div class="value">{passed}</div>
            </div>
            <div class="stat-card failed">
                <div class="label">Failed</div>
                <div class="value">{failed}</div>
            </div>
            <div class="stat-card skipped">
                <div class="label">Skipped</div>
                <div class="value">{skipped}</div>
            </div>
        </div>

        <div class="success-bar">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {success_rate}%">
                    {success_rate:.1f}% Success Rate
                </div>
            </div>
        </div>
"""

    # Add hardware info if available
    if hardware_info:
        html += """
        <div class="hardware-info">
            <h2>Hardware Information</h2>
            <ul>
"""
        html += f"                <li><strong>GPU Count:</strong> {hardware_info.get('gpu_count', 'Unknown')}</li>\n"
        html += f"                <li><strong>MI300 Series:</strong> {'Yes' if hardware_info.get('is_mi300_series') else 'No'}</li>\n"
        html += f"                <li><strong>MI350 Series:</strong> {'Yes' if hardware_info.get('is_mi350_series') else 'No'}</li>\n"

        for i, gpu in enumerate(hardware_info.get('gpus', [])):
            html += f"                <li><strong>GPU {i}:</strong> {gpu.get('name', 'Unknown')} (gfx{gpu.get('gfx_version', '?')})</li>\n"

        html += """
            </ul>
        </div>
"""

    # Add test results table
    html += """
        <div class="results">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add each test result
    for result in summary['results']:
        component = result['component']
        status = result['status']
        duration = f"{result['duration']:.1f}s" if result['duration'] else "N/A"
        error = result.get('error_message', '')
        log_file = result.get('log_file', '')

        details = ""
        if status == "failed" and error:
            details = f"<span style='color: #991b1b;'>{error}</span>"
        if log_file:
            details += f" <a href='file://{log_file}' class='log-link'>[View Log]</a>"

        html += f"""
                    <tr>
                        <td><strong>{component}</strong></td>
                        <td><span class="status {status}">{status}</span></td>
                        <td>{duration}</td>
                        <td>{details}</td>
                    </tr>
"""

    html += f"""
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by ROCm Component Test Kit | Total Duration: {duration:.1f}s</p>
            <p>For more information, visit <a href="https://github.com/ROCm/TheRock" class="log-link">ROCm/TheRock</a></p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


def generate_json_report(summary: Dict, output_path: Path, hardware_info: Dict = None):
    """
    Generate a JSON report from test results.

    Args:
        summary: Test summary dictionary
        output_path: Path to output JSON file
        hardware_info: Optional hardware information
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'hardware': hardware_info,
        'summary': summary
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Example usage
    from hardware_detector import detect_hardware

    # Sample test results
    sample_summary = {
        'total': 5,
        'passed': 3,
        'failed': 1,
        'skipped': 1,
        'duration': 125.5,
        'results': [
            {'component': 'rocblas', 'status': 'passed', 'duration': 25.3, 'error_message': None, 'log_file': None},
            {'component': 'hipblas', 'status': 'passed', 'duration': 22.1, 'error_message': None, 'log_file': None},
            {'component': 'miopen', 'status': 'failed', 'duration': 45.2, 'error_message': 'Test timeout', 'log_file': '/tmp/miopen.log'},
            {'component': 'rocprim', 'status': 'passed', 'duration': 32.9, 'error_message': None, 'log_file': None},
            {'component': 'rccl', 'status': 'skipped', 'duration': None, 'error_message': 'Test script not found', 'log_file': None},
        ]
    }

    # Get hardware info
    hw_info = detect_hardware()
    hw_dict = {
        'gpu_count': hw_info.gpu_count,
        'is_mi300_series': hw_info.is_mi300_series,
        'is_mi350_series': hw_info.is_mi350_series,
        'gpus': hw_info.gpus
    }

    # Generate reports
    generate_html_report(sample_summary, Path('/tmp/test_report.html'), hw_dict)
    generate_json_report(sample_summary, Path('/tmp/test_report.json'), hw_dict)

    print("Sample reports generated:")
    print("  HTML: /tmp/test_report.html")
    print("  JSON: /tmp/test_report.json")
