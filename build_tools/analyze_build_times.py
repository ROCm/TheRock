#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
import re

def parse_ninja_log(log_path):
    """
    Parses .ninja_log file.
    Format: start_time \t end_time \t mtime \t output_path \t command_hash
    Times are in milliseconds.
    """
    tasks = []
    with open(log_path, 'r') as f:
        header = f.readline() # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            start, end, _, output, _ = parts[:5]
            tasks.append({
                'start': int(start),
                'end': int(end),
                'output': output
            })
    return tasks

def parse_output_path(output_path):
    """
    Parses the output path to identify:
    - Clean Name
    - Category (ROCm Component vs Dependency)
    - Phase (Configure, Build, Install, Package)
    """
    parts = output_path.split('/')

    # Phase Detection
    phase = None
    if output_path.endswith('/stamp/configure.stamp'):
        phase = 'Configure'
    elif output_path.endswith('/stamp/build.stamp'):
        phase = 'Build'
    elif output_path.endswith('/stamp/stage.stamp'):
        phase = 'Install'
    elif output_path.startswith('artifacts/') and output_path.endswith('.tar.xz'):
        phase = 'Package'
    elif 'download' in output_path and ('stamp' in output_path or output_path.endswith('.stamp')):
        phase = 'Download'
    elif 'update' in output_path and 'stamp' in output_path:
        phase = 'Update'

    if not phase:
        return None, None, None

    # Name and Category Detection
    name = "Unknown"
    category = "ROCm Component" # Default

    if output_path.startswith('artifacts/'):
        # Artifacts: artifacts/name_variant.tar.xz
        filename = parts[1]
        # Remove extension
        base = filename.replace('.tar.xz', '')
        # Heuristic: split by underscore, take first part as name usually
        # But some names have hyphens: core-runtime, amd-llvm
        # Some have underscores: sysdeps_doc_generic
        # Let's try to match against known patterns or just take the prefix before the first underscore that looks like a variant
        # Common variants: dbg, dev, doc, lib, run, test
        # Regex to capture name and variant, ignoring the specific architecture/platform suffix (e.g., _generic, _gfx1151)
        m = re.match(r'(.+)_(dbg|dev|doc|lib|run|test)(_.+)?', base)
        if m:
            name = m.group(1)
        else:
            name = base

        # Exclude 'base' artifact as it is a directory/meta-package, not a specific component
        if name == 'base' or name == 'sysdeps':
            return None, None, None

        # Categorize artifacts
        if 'sysdeps' in name or 'fftw3' in name or name.startswith('host-'):
            category = "Dependency"

    elif parts[0] == 'third-party':
        category = "Dependency"
        if len(parts) > 3 and parts[1] == 'sysdeps' and parts[2] in ['linux', 'common']:
            # third-party/sysdeps/linux/zstd -> zstd
            # third-party/sysdeps/common/something -> something
            name = parts[3]
        elif len(parts) > 1:
            # third-party/boost -> boost
            name = parts[1]

        if name == 'sysdeps':
             return None, None, None

    elif parts[0] in ['rocm-libraries', 'rocm-systems']:
        category = "ROCm Component"
        if len(parts) > 2 and parts[1] == 'projects':
            name = parts[2]

    elif parts[0] in ['base', 'compiler', 'core', 'comm-libs', 'dctools', 'profiler', 'ml-libs']:
        category = "ROCm Component"
        if len(parts) > 1:
            name = parts[1]

    elif parts[0] == 'math-libs':
        category = "ROCm Component"
        if len(parts) > 1:
            if parts[1] == 'BLAS':
                 if len(parts) > 2:
                    name = parts[2]
                 else:
                     return None, None, None # Skip the parent BLAS directory itself
            elif parts[1] == 'support' and len(parts) > 2:
                name = parts[2]
            else:
                name = parts[1]

    else:
        return None, None, None

    # Mapping for components with different build/package names
    NAME_MAPPING = {
        'clr': 'core-hip',
        'ocl-clr': 'core-ocl',
        'ROCR-Runtime': 'core-runtime',
        'blas': 'rocBLAS',
        'prim': 'rocPRIM',
        'fft': 'rocFFT',
        'rand': 'rocRAND',
        'miopen': 'MIOpen',
        'hipdnn': 'hipDNN',
        'composable-kernel': 'composable_kernel',
        'support': 'mxDataGenerator',
        'host-suite-sparse': 'SuiteSparse',
        'rocwmma': 'rocWMMA',
        'miopen-plugin': 'miopen_plugin'
    }

    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]

    return name, category, phase

def analyze_tasks(tasks, build_dir):
    # Structure: projects[category][name][phase] = duration
    projects = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    seen_tasks = set()

    build_dir_abs = str(build_dir.resolve())

    for task in tasks:
        output_path = task['output']
        start = task['start']
        end = task['end']

        # Normalize absolute paths by stripping build_dir
        if output_path.startswith(build_dir_abs):
            output_path = output_path[len(build_dir_abs):].lstrip('/')

        # Deduplicate based on normalized path and timestamps
        # Some entries in .ninja_log might be duplicated with absolute/relative paths
        task_key = (output_path, start, end)
        if task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)

        name, category, phase = parse_output_path(output_path)
        if not name:
            continue

        duration = task['end'] - task['start']
        # We assume one task per phase per component usually, or we sum them if multiple (e.g. multiple artifacts for one component)
        projects[category][name][phase] += duration

    return projects

def format_duration(ms):
    if ms == 0:
        return "-"
    seconds = ms / 1000.0
    return f"{seconds:.2f}"

def generate_html(projects, output_file):
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROCm Build Time Analysis</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }
        h1 { text-align: center; color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); background-color: white; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:hover { background-color: #f1f1f1; }
        .total-col { font-weight: bold; color: #27ae60; }
        .footer { margin-top: 30px; text-align: center; font-size: 0.9em; color: #777; }
    </style>
</head>
<body>
    <h1>ROCm Build Time Analysis</h1>

    {tables}

    <div class="footer">Generated by analyze_build_times.py</div>
</body>
</html>
    """

    tables_html = ""

    # Order: ROCm Components, then Dependencies
    categories = ["ROCm Component", "Dependency"]

    for cat in categories:
        if cat not in projects:
            continue

        tables_html += f"<h2>{cat}s</h2>" if cat != "Dependency" else f"<h2>Dependencies</h2>"

        if cat == "ROCm Component":
            tables_html += """
            <table>
                <thead>
                    <tr>
                        <th>Sub-Project</th>
                        <th>Configure (s)</th>
                        <th>Build (s)</th>
                        <th>Install (s)</th>
                        <th>Package (s)</th>
                        <th>Total Time (s)</th>
                    </tr>
                </thead>
                <tbody>
            """
        else: # Dependency
            tables_html += """
            <table>
                <thead>
                    <tr>
                        <th>Sub-Project</th>
                        <th>Download (s)</th>
                        <th>Configure (s)</th>
                        <th>Build (s)</th>
                        <th>Install (s)</th>
                        <th>Total Time (s)</th>
                    </tr>
                </thead>
                <tbody>
            """

        # Sort by Total Time descending
        comps = []
        for name, phases in projects[cat].items():
            total = sum(phases.values())
            comps.append((name, phases, total))

        comps.sort(key=lambda x: x[2], reverse=True)

        for name, phases, total in comps:
            if cat == "ROCm Component":
                tables_html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{format_duration(phases['Configure'])}</td>
                        <td>{format_duration(phases['Build'])}</td>
                        <td>{format_duration(phases['Install'])}</td>
                        <td>{format_duration(phases['Package'])}</td>
                        <td class="total-col">{format_duration(total)}</td>
                    </tr>
                """
            else: # Dependency
                download_time = phases['Download'] + phases['Update']
                tables_html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{format_duration(download_time)}</td>
                        <td>{format_duration(phases['Configure'])}</td>
                        <td>{format_duration(phases['Build'])}</td>
                        <td>{format_duration(phases['Install'])}</td>
                        <td class="total-col">{format_duration(total)}</td>
                    </tr>
                """

        tables_html += """
            </tbody>
        </table>
        """

    try:
        with open(output_file, 'w') as f:
            f.write(html_template.replace("{tables}", tables_html))
        print(f"HTML report generated at: {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Ninja build times")
    parser.add_argument("--build-dir", type=Path, required=True, help="Path to build directory")
    parser.add_argument("--output", type=Path, help="Path to output HTML file")
    args = parser.parse_args()

    ninja_log = args.build_dir / ".ninja_log"
    if not ninja_log.exists():
        print(f"Error: {ninja_log} not found.")
        sys.exit(1)

    tasks = parse_ninja_log(ninja_log)
    projects = analyze_tasks(tasks, args.build_dir)

    if args.output:
        output_html = args.output
    else:
        output_html = args.build_dir / "logs" / "build_time_analysis.html"
        # Ensure logs dir exists
        output_html.parent.mkdir(parents=True, exist_ok=True)

    generate_html(projects, output_html)

if __name__ == "__main__":
    main()
