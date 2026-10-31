#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""TheRock-only actual-object ccache input comparison; compilation stays blocked."""

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import profile_offload_trial as preparation
from profile_cache_hash_inputs import run_probe, sha256

ROOT = Path(__file__).resolve().parents[1]


def ignored(directory, names, root):
    relative = Path(directory).relative_to(root)
    result = {".git"} & set(names)
    if relative == Path("."):
        result |= {".container-cache"} & set(names)
    if relative == Path("build"):
        result |= {"logs"} & set(names)
    return result


def clone_prefix(source, destination):
    """Real directories, hardlinked immutable regular inputs, preserved symlinks."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("Second checkout prefix must be new")
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError(
            "Both prefixes must share a filesystem for byte-identical hardlinks"
        )
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        copy_function=os.link,
        ignore=lambda directory, names: ignored(directory, names, source),
    )
    remapped = []
    for directory, directories, files in os.walk(destination, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            target = os.readlink(path)
            if target == str(source) or target.startswith(str(source) + "/"):
                replacement = str(destination) + target[len(str(source)) :]
                path.unlink()  # Only the new clone's symlink, never a regular input.
                path.symlink_to(replacement)
                remapped.append(
                    {
                        "relative_path": str(path.relative_to(destination)),
                        "original_target": target,
                        "clone_target": replacement,
                    }
                )
    return remapped


def content_manifest(root, hash_cache):
    rows = []
    for directory, directories, files in os.walk(root, followlinks=False):
        skip = ignored(directory, directories + files, root)
        directories[:] = sorted(name for name in directories if name not in skip)
        for name in sorted(set(directories + files) - skip):
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                target = os.readlink(path)
                if target == str(root) or target.startswith(str(root) + "/"):
                    target = "$CHECKOUT" + target[len(str(root)) :]
                rows.append({"path": relative, "kind": "symlink", "target": target})
            elif path.is_file():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if identity not in hash_cache:
                    hash_cache[identity] = sha256(path)
                if path.stat().st_mtime_ns != stat.st_mtime_ns:
                    raise ValueError("Input changed during hashing: " + relative)
                rows.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": hash_cache[identity],
                    }
                )
    return sorted(rows, key=lambda row: row["path"])


def original_arguments(entry):
    tokens = entry.get("arguments") or shlex.split(entry["command"])
    if tokens and Path(tokens[0]).name == "ccache":
        tokens = tokens[1:]
    if (
        not tokens
        or not Path(tokens[0]).is_absolute()
        or Path(tokens[0]).name != "clang++"
    ):
        raise ValueError(
            "Expected a direct absolute clang++ command after the ccache launcher"
        )
    if any(
        token in ("&&", ";", "|", "||") or token.startswith("@") for token in tokens
    ):
        raise ValueError("Use the expanded single compiler command")
    arches = [
        arg.split("=", 1)[1] for arg in tokens if arg.startswith("--offload-arch=")
    ]
    if len(arches) != 7 or set(arches) != preparation.ARCHES:
        raise ValueError("The exact original seven architectures are required")
    if any("offload-jobs=" in arg or "parallel-jobs=" in arg for arg in tokens):
        raise ValueError(
            "Do not combine hash diagnosis with offload concurrency trials"
        )
    if any(flag not in tokens for flag in ("-O3", "-std=c++17", "-c", "-MD", "-MF")):
        raise ValueError("Original optimization/language/dependency flags drifted")
    sources = [
        arg
        for arg in tokens
        if arg.endswith("benchmark_device_segmented_radix_sort_pairs.cpp")
    ]
    if len(sources) != 1:
        raise ValueError("Expected exactly the giant pairs source")
    return tokens


def compiler_check(ccache, env):
    command = preparation.capture(
        [ccache, "--get-config", "compiler_check"], ROOT, env
    ).strip()
    args = shlex.split(command)
    if len(args) != 4 or args[-1] != "%compiler%":
        raise ValueError("Unexpected compiler-check command shape")
    helper = Path(args[1])
    if not helper.is_file() or sha256(helper) != sha256(
        ROOT / "build_tools/posix_ccache_compiler_check.py"
    ):
        raise ValueError(
            "Compiler check differs from the reviewed non-compiling fingerprint helper"
        )
    return args


def consumed_inputs(raw_directory, cwd, prefix):
    # The default second-preprocessor mode strips dependency-generation flags.
    # Preserve its actual raw output in the guard before ccache cleans up, then
    # enumerate the included source/header paths in compiler line markers.
    names = set()
    files = list(raw_directory.iterdir())
    if not files:
        raise ValueError("No raw preprocessor output survived the guard")
    for file in files:
        for line in file.read_text(errors="replace").splitlines():
            match = re.match(r'^\s*#\s+\d+\s+("(?:[^"\\]|\\.)*")', line)
            if match:
                name = json.loads(match[1])
                if not name.startswith("<"):
                    names.add(name)
    if not names:
        raise ValueError("No source/header line markers in raw preprocessor output")
    rows = []
    for raw in sorted(names):
        path = Path(raw)
        if not path.is_absolute():
            path = cwd / path
        stat = path.stat()
        rows.append(
            {
                "path": str(path).replace(str(prefix), "$CHECKOUT"),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def compare_inputs(output):
    evidence = {}
    for case in ("A1", "B1", "B2", "A2"):
        files = {}
        for kind in ("c", "d", "p", "text"):
            matches = list((output / case / "debug").rglob("*.ccache-input-" + kind))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {kind} input stream in {case}")
            files[kind] = {
                "path": str(matches[0].relative_to(output)),
                "bytes": matches[0].stat().st_size,
                "sha256": sha256(matches[0]),
            }
        evidence[case] = files
    for a, b in (("A1", "A2"), ("B1", "B2")):
        if any(
            evidence[a][kind]["sha256"] != evidence[b][kind]["sha256"]
            for kind in evidence[a]
        ):
            raise ValueError(f"Within-prefix input instability: {a} vs {b}")
    a = (
        (output / evidence["A1"]["text"]["path"])
        .read_text(errors="replace")
        .splitlines(keepends=True)
    )
    b = (
        (output / evidence["B1"]["text"]["path"])
        .read_text(errors="replace")
        .splitlines(keepends=True)
    )
    with (output / "A1-vs-B1-input-text.diff").open("w") as stream:
        stream.writelines(difflib.unified_diff(a, b, fromfile="A1", tofile="B1"))
    return {
        "cases": evidence,
        "within_prefix_stable": True,
        "cross_prefix_equal": {
            kind: evidence["A1"][kind]["sha256"] == evidence["B1"][kind]["sha256"]
            for kind in ("c", "d", "p", "text")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT.parent / "cache-probe-diagnostics"
    )
    args = parser.parse_args()
    preparation.require_ci(os.environ)
    if sys.platform != "linux" or os.environ.get("PROFILE_OFFLOAD_TRIAL") == "true":
        raise RuntimeError(
            "Linux cache diagnosis must run separately from offload trials"
        )
    output = args.output.absolute()
    second = ROOT.parent / (
        ROOT.name
        + "-cache-probe-B-"
        + os.environ["GITHUB_RUN_ID"]
        + "-"
        + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    )
    if output.resolve().is_relative_to(
        ROOT.resolve()
    ) or output.resolve().is_relative_to(second.resolve()):
        raise ValueError("All diagnostics must be outside both input trees")
    output.mkdir(parents=True, exist_ok=True)
    build = ROOT / "build"
    ccache, ninja = shutil.which("ccache"), shutil.which("ninja")
    if not ccache or not ninja:
        raise RuntimeError("ccache and Ninja are required")
    if (
        preparation.capture([ccache, "--version"], ROOT, os.environ).splitlines()[0]
        != "ccache version 4.12.1"
    ):
        raise ValueError("The reviewed guard requires ccache 4.12.1")
    env = os.environ.copy()
    nested = preparation.prepare(build, output / "preparation", env)
    (output / "source-submodule-status.txt").write_text(
        preparation.capture(["git", "submodule", "status", "--recursive"], ROOT, env)
    )
    # prepare() configures this target and builds prerequisites only. Never call
    # run_trial(), force_object_rebuild(), capability_probe(), or build OBJECT.
    compdb = json.loads(
        preparation.capture([ninja, "-C", str(nested), "-t", "compdb", "-x"], ROOT, env)
    )
    entries = [
        entry
        for entry in compdb
        if entry.get("output", "").removeprefix(str(nested) + "/") == preparation.OBJECT
    ]
    if len(entries) != 1 or Path(entries[0]["directory"]).resolve() != nested.resolve():
        raise ValueError(
            "Expected one exact expanded giant-object entry in the original nested build"
        )
    arguments = original_arguments(entries[0])
    preparation.write_json(output / "original-compilation-entry.json", entries[0])
    if (nested / preparation.OBJECT).exists():
        raise ValueError("Preparation unexpectedly produced the heavy object")
    original_config = Path(env["CCACHE_CONFIGPATH"])
    check = compiler_check(ccache, env)
    remapped = clone_prefix(ROOT, second)
    hash_cache = {}
    before_a, before_b = content_manifest(ROOT, hash_cache), content_manifest(
        second, hash_cache
    )
    if before_a != before_b:
        raise ValueError(
            "Prepared input content/size/mtime manifests differ between prefixes"
        )
    preparation.write_json(output / "input-content-manifest.json", before_a)
    preparation.write_json(output / "clone-symlink-remapping.json", remapped)
    diagnostic_env = preparation.trial_env(env)
    (output / "tmp").mkdir(exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        diagnostic_env[key] = str(output / "tmp")
    records, fingerprints = {}, {}
    for name, prefix in (("A", ROOT), ("B", second)):
        mapped = lambda value: value.replace(str(ROOT), str(prefix))
        check_args = [
            mapped(check[0]),
            mapped(check[1]),
            str(output / "fingerprint-cache" / name),
            "%compiler%",
        ]
        record = {
            "directory": mapped(str(nested)),
            "arguments": [mapped(arg) for arg in arguments],
            "config_path": mapped(str(original_config)),
            "compiler_check": shlex.join(check_args),
        }
        fingerprint = subprocess.check_output(
            [*check_args[:-1], record["arguments"][0]], cwd=prefix, env=diagnostic_env
        )
        (output / (name + "-compiler-fingerprint.txt")).write_bytes(fingerprint)
        fingerprints[name] = fingerprint
        records[name] = record
        preparation.write_json(output / (name + "-record.json"), record)
    if fingerprints["A"] != fingerprints["B"]:
        raise ValueError("Compiler/shared-library fingerprints differ")
    results, consumed = {}, {}
    for case in ("A1", "B1", "B2", "A2"):
        record = records[case[0]]
        result = run_probe(
            Path(ccache),
            record,
            output / case,
            output / "shared",
            300,
            env=diagnostic_env,
        )
        if not result["valid_hash_probe"]:
            raise ValueError("Invalid guarded preprocessing case: " + case)
        results[case] = result["keys"]
        consumed[case] = consumed_inputs(
            output / case / "raw-preprocessor",
            Path(record["directory"]),
            ROOT if case[0] == "A" else second,
        )
        preparation.write_json(
            output / (case + "-consumed-input-manifest.json"), consumed[case]
        )
    if results["A1"] != results["A2"] or results["B1"] != results["B2"]:
        raise ValueError("Within-prefix key instability")
    if any(rows != consumed["A1"] for rows in consumed.values()):
        raise ValueError("Actual consumed source/header content/size/mtime differs")
    comparisons = compare_inputs(output)
    # Fresh hash cache: re-read content after the probe, not just file metadata.
    after_cache = {}
    after_a, after_b = content_manifest(ROOT, after_cache), content_manifest(
        second, after_cache
    )
    if after_a != before_a or after_b != before_b:
        raise ValueError("Input content/size/mtime changed while diagnostics ran")
    if (nested / preparation.OBJECT).exists() or (
        Path(records["B"]["directory"]) / preparation.OBJECT
    ).exists():
        raise ValueError("The forbidden heavy object was produced")
    preparation.write_json(
        output / "summary.json",
        {
            "status": "valid_guarded_hash_input_comparison",
            "repository": env["GITHUB_REPOSITORY"],
            "run_id": env["GITHUB_RUN_ID"],
            "diagnostic_sha": env["GITHUB_SHA"],
            "recorded_product_head": preparation.SOURCE_BASELINE,
            "libraries_pin": preparation.LIBRARIES_PIN,
            "sdk_run": preparation.SDK_RUN,
            "container_image": "ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:d666e4613c1ca919bc7d1a31a4c368740b66c34b5c19ef95ba15cd31db7783fb",
            "runner_label": "aws-linux-scale-rocm-prod",
            "runner_name": env.get("RUNNER_NAME"),
            "prefixes": {"A": str(ROOT), "B": str(second)},
            "input_content_rows": len(before_a),
            "input_manifest_sha256": sha256(output / "input-content-manifest.json"),
            "regular_files_hardlinked": True,
            "input_content_size_mtime_stable": True,
            "consumed_source_header_manifests_identical": True,
            "compiler_shared_library_fingerprints_identical": True,
            "order": ["A1", "B1", "B2", "A2"],
            "keys": results,
            "inputs": comparisons,
            "heavy_object_compiled": False,
            "scope": "Private compile-blocked preprocessing/hash diagnostics; not a cache-hit or speedup benchmark",
            "controlled_changes": [
                "real checkout prefix",
                "same private external -o/-MF paths for all cases",
                "same compiler-check algorithm with private external memoization caches",
            ],
        },
    )


if __name__ == "__main__":
    main()
