#!/usr/bin/env python3

import json
import pathlib
import subprocess
import sys


def run(cmd, cwd):
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def main() -> int:
    script_path = pathlib.Path(__file__).resolve()
    mirage_root = script_path.parents[2]
    build_dir = mirage_root / "build" / "coverage"
    reports_dir = mirage_root / "reports" / "coverage"
    snapshot_dir = mirage_root / "tests" / "data" / "coverage"
    build_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    probe_source = mirage_root / "tools" / "coverage" / "gfx950_coverage_probe.cc"
    probe_binary = build_dir / "gfx950_coverage_probe"

    compile_cmd = [
        "c++",
        "-std=c++20",
        "-O2",
        f"-I{mirage_root}",
        str(probe_source),
        str(mirage_root / "native" / "generated" / "gfx950_instruction_catalog.cc"),
        str(mirage_root / "native" / "src" / "isa" / "gfx950" / "interpreter.cc"),
        str(mirage_root / "native" / "src" / "isa" / "gfx950" / "binary_decoder.cc"),
        "-o",
        str(probe_binary),
    ]
    run(compile_cmd, cwd=mirage_root)
    report = json.loads(run([str(probe_binary)], cwd=mirage_root))

    report_json = reports_dir / "gfx950_support_report.json"
    snapshot_json = snapshot_dir / "gfx950_support_snapshot.json"
    report_markdown = reports_dir / "gfx950_support_report.md"

    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_json.write_text(json_text, encoding="utf-8")
    snapshot_json.write_text(json_text, encoding="utf-8")

    summary = report["summary"]
    instructions = report["instructions"]
    semantic_only = [
        item["instruction_name"]
        for item in instructions
        if item["semantic_supported"] and not item["decode_supported"]
    ]
    decode_only = [
        item["instruction_name"]
        for item in instructions
        if item["decode_supported"] and not item["semantic_supported"]
    ]
    unsupported = [
        item["instruction_name"]
        for item in instructions
        if not item["semantic_supported"] and not item["decode_supported"]
    ]

    lines = [
        "# GFX950 Coverage Report",
        "",
        f"- Catalog instructions: {report['catalog']['instruction_count']}",
        f"- Semantic support: {summary['semantic_supported']} ({summary['semantic_supported_percent_total']:.1%})",
        f"- Raw decode support: {summary['decode_supported']} ({summary['decode_supported_percent_total']:.1%} of total, {summary['decode_supported_percent_measured']:.1%} of measured)",
        f"- Raw decode measurable instructions: {summary['decode_measured']}",
        "",
        "## Gaps",
        "",
        f"- Semantic-only coverage without measured decode: {len(semantic_only)}",
        f"- Decode-only without semantic support: {len(decode_only)}",
        f"- Missing both semantic and decode support: {len(unsupported)}",
        "",
        "## Unmeasured Encoding Families",
        "",
    ]
    for encoding_name in report["unmeasured_encoding_families"]:
        lines.append(f"- `{encoding_name}`")
    if not report["unmeasured_encoding_families"]:
        lines.append("- None")

    def append_sample(title, values):
        lines.extend(["", f"## {title}", ""])
        if not values:
            lines.append("- None")
            return
        for value in values[:25]:
            lines.append(f"- `{value}`")

    append_sample("Semantic-Only Sample", semantic_only)
    append_sample("Decode-Only Sample", decode_only)
    append_sample("Missing-Both Sample", unsupported)

    report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_json}")
    print(f"Wrote {snapshot_json}")
    print(f"Wrote {report_markdown}")
    print(
        "Semantic support:",
        summary["semantic_supported"],
        "/",
        report["catalog"]["instruction_count"],
    )
    print(
        "Raw decode support:",
        summary["decode_supported"],
        "/",
        report["catalog"]["instruction_count"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
