#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate llvm profraw output from test shards into one coverage report.

Components in rocm-libraries generate coverage with an in-tree `coverage` make
target that runs the tests and reports on them in the same CMake tree. TheRock
splits build and test across nodes and shards the test suite, so that target has
nothing to run against: the tree is gone by test time and no single node has run
more than a slice of the suite. This script performs the equivalent steps from
the artifacts instead, on a node that has collected the profraw files from every
shard:

    llvm-profdata merge -sparse <all shards>/*.profraw -o coverage.profdata
    llvm-cov export -format=lcov -object <objects> -instr-profile=coverage.profdata

Merging across shards is what makes the number meaningful. Each shard only runs
its slice of the suite, so a per-shard report would claim roughly
100/<shard count> percent coverage no matter how complete the suite is.

Objects are discovered rather than configured. A component that installs a
shared library is reported against that library. Header-only components (such as
rocPRIM and hipCUB) have no library to point at because their code is compiled
into the test binaries, so those binaries become the objects, and test sources
are filtered out of the report so that testing code does not count as covered
product code.

Usage:
    python coverage_report.py \\
        --component hiprand \\
        --rocm-dir build \\
        --profraw-dir coverage-profraw \\
        --output-dir coverage-report
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_actions_api import gha_append_step_summary

# Job names do not always match the directory a component installs its tests
# into. Only the mismatches need an entry here.
COMPONENT_TEST_DIR_OVERRIDES = {
    "hipcub": "hipcub",
    "hiprand": "hipRAND",
    "rocprim": "rocprim",
    "rocrand": "rocRAND",
    "rocthrust": "rocthrust",
}

# Paths that describe how a component is tested rather than what it ships.
# Counting them would let a component raise its coverage by adding test code.
DEFAULT_IGNORE_REGEX = r"(/test/|/tests/|/googletest/|/benchmark/|/_deps/)"


def log(*args):
    print(*args)
    sys.stdout.flush()


def find_llvm_tool(rocm_dir: Path, name: str) -> Path:
    """Locate an LLVM tool, preferring the toolchain that built the artifacts.

    Coverage data is only readable by a tool at least as new as the compiler
    that produced it, so the bundled toolchain is the correct one to use and a
    system llvm-cov is a fallback for local runs.
    """
    bundled = rocm_dir / "lib" / "llvm" / "bin" / name
    if bundled.is_file():
        return bundled
    system = shutil.which(name)
    if system:
        log(f"[WARN] {bundled} not found, falling back to {system}")
        return Path(system)
    raise FileNotFoundError(
        f"Could not find {name} in {bundled.parent} or on PATH. The coverage "
        f"report needs the LLVM tools from the instrumented build's artifacts."
    )


def collect_profraw_files(profraw_dir: Path) -> list[Path]:
    """Collect profraw files from every shard's uploaded directory."""
    files = sorted(p for p in profraw_dir.rglob("*.profraw") if p.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(
            f"No non-empty .profraw files found under {profraw_dir}. Either the "
            f"binaries under test were not instrumented, or LLVM_PROFILE_FILE "
            f"did not point at the collected directory during the test run."
        )
    log(f"[INFO] Found {len(files)} profraw file(s) under {profraw_dir}")
    return files


def merge_profraw(
    llvm_profdata: Path, profraw_files: list[Path], output_file: Path
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(llvm_profdata), "merge", "-sparse", "-o", str(output_file)]
    cmd += [str(p) for p in profraw_files]
    log(f"[INFO] Merging profraw into {output_file}")
    subprocess.run(cmd, check=True)
    return output_file


def resolve_test_dir(rocm_dir: Path, component: str) -> Path:
    dir_name = COMPONENT_TEST_DIR_OVERRIDES.get(component, component)
    return rocm_dir / "bin" / dir_name


def discover_objects(rocm_dir: Path, component: str) -> list[Path]:
    """Find the binaries to report coverage for.

    A shared library is preferred: it is the component's product and reporting
    against it keeps test binaries out of the numbers automatically. Only when
    no library exists (header-only components) do the test binaries stand in for
    it, since that is where the component's code actually got compiled.
    """
    lib_dir = rocm_dir / "lib"
    library = lib_dir / f"lib{component}.so"
    if library.is_file():
        log(f"[INFO] Reporting against shared library {library}")
        return [library]

    test_dir = resolve_test_dir(rocm_dir, component)
    # llvm-cov's -object flag takes one path and does not expand globs, so every
    # test binary has to be listed explicitly.
    tests = sorted(
        p
        for p in test_dir.glob("test_*")
        if p.is_file() and not p.suffix and p.stat().st_mode & 0o111
    )
    if tests:
        log(
            f"[INFO] No lib{component}.so found; reporting against "
            f"{len(tests)} test binaries in {test_dir}"
        )
        return tests

    raise FileNotFoundError(
        f"Found neither {library} nor test binaries in {test_dir}. Cannot "
        f"determine what to report coverage for."
    )


def run_llvm_cov(
    llvm_cov: Path,
    subcommand: str,
    objects: list[Path],
    profdata: Path,
    ignore_regex: str,
    extra_args: list[str] | None = None,
) -> str:
    cmd = [str(llvm_cov), subcommand]
    for obj in objects:
        cmd += ["-object", str(obj)]
    cmd += [f"-instr-profile={profdata}"]
    if ignore_regex:
        cmd += [f"--ignore-filename-regex={ignore_regex}"]
    cmd += extra_args or []
    log(f"[INFO] Running llvm-cov {subcommand}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # llvm-cov explains mismatched profiles and unreadable objects on
        # stderr, which is captured here because stdout is the report itself.
        log(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.stdout


def extract_total_line(report_text: str) -> str:
    for line in report_text.splitlines():
        if line.strip().startswith("TOTAL"):
            return re.sub(r"\s+", " ", line.strip())
    return ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        required=True,
        help="Component under test, as named by the test job (e.g. hiprand).",
    )
    parser.add_argument(
        "--rocm-dir",
        type=Path,
        default=Path("build"),
        help="Directory the instrumented ROCm artifacts were unpacked into.",
    )
    parser.add_argument(
        "--profraw-dir",
        type=Path,
        required=True,
        help="Directory holding the profraw files downloaded from every shard.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coverage-report"),
        help="Directory to write coverage.profdata and coverage.info into.",
    )
    parser.add_argument(
        "--object",
        dest="objects",
        type=Path,
        action="append",
        default=[],
        help="Binary to report against, repeatable. Overrides auto-discovery.",
    )
    parser.add_argument(
        "--ignore-filename-regex",
        default=DEFAULT_IGNORE_REGEX,
        help="Exclude matching source paths from the report.",
    )
    args = parser.parse_args(argv)

    rocm_dir = args.rocm_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    llvm_profdata = find_llvm_tool(rocm_dir, "llvm-profdata")
    llvm_cov = find_llvm_tool(rocm_dir, "llvm-cov")

    profraw_files = collect_profraw_files(args.profraw_dir.resolve())
    profdata = merge_profraw(
        llvm_profdata, profraw_files, output_dir / "coverage.profdata"
    )

    objects = args.objects or discover_objects(rocm_dir, args.component)

    lcov = run_llvm_cov(
        llvm_cov,
        "export",
        objects,
        profdata,
        args.ignore_filename_regex,
        ["--format=lcov"],
    )
    lcov_file = output_dir / "coverage.info"
    lcov_file.write_text(lcov)
    log(f"[INFO] Wrote {lcov_file} ({len(lcov.splitlines())} lines)")

    report = run_llvm_cov(
        llvm_cov, "report", objects, profdata, args.ignore_filename_regex
    )
    (output_dir / "coverage.txt").write_text(report)
    log(report)

    total = extract_total_line(report)
    summary = [f"### Code coverage: {args.component}", ""]
    if total:
        summary.append(f"`{total}`")
        summary.append("")
    summary.append(
        f"Merged {len(profraw_files)} profraw file(s) from all test shards over "
        f"{len(objects)} object(s)."
    )
    gha_append_step_summary("\n".join(summary))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
