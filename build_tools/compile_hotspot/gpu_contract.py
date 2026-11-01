# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Link and exercise exact benchmark objects on the matching GPU/runtime."""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / "ck-objects"
EVIDENCE = ROOT / "ck-evidence"
OUT = ROOT / "ck-gpu-evidence"
SDK = ROOT / "ck-sdk"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def save(name: str, data: object) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def run(argv: list[str], label: str, env: dict | None = None) -> None:
    save(label + "-command.json", argv)
    print(shlex.join(argv), flush=True)
    with (OUT / (label + ".log")).open("w") as log:
        started = time.monotonic()
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        elapsed = time.monotonic() - started
    save(label + "-result.json", {"exit_code": result.returncode, "elapsed_s": elapsed})
    result.check_returncode()


def verify_producer(context: dict) -> tuple[dict, dict]:
    benchmark_run = os.environ.get("BENCHMARK_RUN_ID") or os.environ["GITHUB_RUN_ID"]
    if not benchmark_run.isdecimal():
        raise RuntimeError("BENCHMARK_RUN_ID must be a numeric GitHub Actions run ID")
    for key, expected in (
        ("run_id", benchmark_run),
        ("variant", os.environ["BUILD_VARIANT"]),
        ("baseline_run_id", os.environ["BASELINE_RUN_ID"]),
    ):
        if str(context[key]) != expected:
            raise RuntimeError(f"Build artifact provenance mismatch: {key}")
    if benchmark_run == os.environ["GITHUB_RUN_ID"]:
        if context["instrumentation_sha"] != os.environ["GITHUB_SHA"]:
            raise RuntimeError("Same-run producer instrumentation SHA mismatch")
    request = urllib.request.Request(
        f"https://api.github.com/repos/ROCm/TheRock/actions/runs/{benchmark_run}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    if (
        str(metadata["id"]) != benchmark_run
        or metadata["head_sha"] != context["instrumentation_sha"]
        or metadata["repository"]["full_name"].lower() != "rocm/therock"
    ):
        raise RuntimeError("Producer context differs from GitHub run metadata")
    # Overall conclusion may be failure because the original GPU job failed;
    # benchmark-complete.json and trial measurements validate the build below.
    producer = {
        key: metadata[key]
        for key in (
            "id",
            "head_sha",
            "head_branch",
            "run_attempt",
            "status",
            "conclusion",
            "html_url",
        )
    }
    validation = {
        "run_id": os.environ["GITHUB_RUN_ID"],
        "instrumentation_sha": os.environ["GITHUB_SHA"],
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "benchmark_run_id": benchmark_run,
        "repository": os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock"),
    }
    save("producer-run-metadata.json", producer)
    save("validation-context.json", validation)
    return producer, validation


def verify_trials(
    protocol: dict, summary: list[dict], complete: dict
) -> tuple[str, dict, dict]:
    """Validate the fixed three-arm protocol and hash every retained arm."""
    order = ["B1", "T1", "C1", "C2", "T2", "B2"]
    counts = {"baseline": 1, "reference": 3, "candidate": 2}
    variants = {"B": "baseline", "T": "reference", "C": "candidate"}
    if protocol["order"] != order or [row["trial"] for row in summary] != order:
        raise RuntimeError("Three-arm benchmark order/inventory mismatch")
    if protocol["variant_object_counts"] != counts or complete["status"] != "complete":
        raise RuntimeError("Three-arm benchmark protocol/counts incomplete")
    original_source = next(iter(protocol["baseline_sources"].values()))["path"]
    if "/rocm-libraries/" not in original_source:
        raise RuntimeError("Unrecognized baseline source root")
    old_root = original_source.split("/rocm-libraries/")[0]
    selected, verified = {}, {}
    for row in summary:
        trial = row["trial"]
        variant = variants[trial[0]]
        commands = protocol[variant + "_commands"]
        expected_names = {Path(name).with_suffix(".o").name for name in commands}
        if len(commands) != counts[variant] or len(expected_names) != counts[variant]:
            raise RuntimeError(f"Unexpected {variant} compile command inventory")
        if (
            row["variant"] != variant
            or len(row["objects"]) != counts[variant]
            or len(row["objects_measured"]) != counts[variant]
            or row["group_measurement"]["exit_code"] != 0
            or any(item["exit_code"] != 0 for item in row["objects_measured"])
        ):
            raise RuntimeError(f"Unsuccessful/incomplete benchmark trial: {trial}")
        names = [Path(item["path"]).name for item in row["objects"]]
        if len(set(names)) != len(names) or set(names) != expected_names:
            raise RuntimeError(f"Object/command inventory mismatch: {trial}")
        objects, records = [], []
        for item in row["objects"]:
            relative = Path(item["path"]).relative_to(old_root)
            expected_parent = Path("compile-hotspot-evidence/benchmarks") / trial
            if relative.parent != expected_parent:
                raise RuntimeError(
                    f"Object is not in its isolated trial directory: {trial}"
                )
            if trial not in {"B2", "C2", "T1", "T2"}:
                continue
            path = BUNDLE / relative
            if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"Object digest/size mismatch: {path}")
            objects.append(path)
            records.append(
                {
                    "path": str(relative),
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                }
            )
        if objects:
            verified[trial] = {"variant": variant, "objects": records}
        if trial in {"B2", "C2"}:
            selected[trial] = objects
    return old_root, selected, verified


def main() -> None:
    OUT.mkdir(exist_ok=True)
    contexts = list(EVIDENCE.rglob("context.json"))
    if len(contexts) != 1:
        raise RuntimeError("Expected exactly one build context")
    evidence = contexts[0].parent
    context = json.loads(contexts[0].read_text())
    producer_metadata, validation_context = verify_producer(context)
    if context["source_sha"] != "5b7cb88373362ab80dc0f19605915e534f4158aa":
        raise RuntimeError("Product source changed")
    protocol = json.loads((evidence / "benchmark-protocol.json").read_text())
    summary = json.loads((evidence / "benchmark-summary.json").read_text())
    complete = json.loads((evidence / "benchmark-complete.json").read_text())
    old_root, selected, verified_arms = verify_trials(protocol, summary, complete)
    save("verified-benchmark-arms.json", verified_arms)
    for relative, item in protocol["generated_headers"].items():
        path = BUNDLE / "build/math-libs/composable_kernel/build" / relative
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"Generated header digest mismatch: {path}")
    for relative, item in protocol["source_headers"].items():
        path = BUNDLE / relative
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"Source header digest mismatch: {path}")
    run(
        [
            sys.executable,
            "build_tools/artifact_manager.py",
            "fetch",
            f"--run-id={context['baseline_run_id']}",
            "--run-github-repo=ROCm/TheRock",
            "--stage=math-libs",
            "--amdgpu-families=gfx950",
            f"--output-dir={SDK}",
            "--flatten",
        ],
        "fetch-runtime",
    )
    compiler = SDK / "lib/llvm/bin/clang++"
    if digest(compiler) != protocol["compiler"]["sha256"]:
        raise RuntimeError("GPU linker compiler differs from benchmark compiler")
    utility = BUNDLE / "compile-hotspot-evidence/link-support/libutility.a"
    utility_manifest = json.loads((utility.parent / "manifest.json").read_text())
    if digest(utility) != utility_manifest["utility"]["sha256"]:
        raise RuntimeError("CK utility digest mismatch")

    def relocate(value: str) -> str:
        if old_root not in value:
            return value
        prefix, relative = value.split(old_root, 1)
        bundled = BUNDLE / relative.lstrip("/")
        if bundled.exists():
            return prefix + str(bundled)
        for marker in ("/build/core/clr/dist/", "/stage/", "/dist/"):
            if marker in relative:
                remainder = relative.rsplit(marker, 1)[1]
                target = SDK / remainder
                if not target.exists():
                    raise RuntimeError(f"Relocated compiler path missing: {target}")
                return prefix + str(target)
        if relative == "/build/core/clr/dist":
            return prefix + str(SDK)
        raise RuntimeError(f"Unrecognized producer path: {value}")

    entry = next(iter(protocol["baseline_commands"].values()))
    argv = shlex.split(entry["command"])
    if Path(argv[0]).name == "ccache":
        argv = argv[1:]
    common = []
    index = 0
    while index < len(argv):
        if argv[index] in {"-o", "-MF", "-MT", "-c"}:
            index += 2
        elif argv[index] == "-MD":
            index += 1
        else:
            common.append(relocate(argv[index]))
            index += 1
    arches = [arg for arg in common if arg.startswith("--offload-arch=")]
    if not arches or any(not arg.startswith("--offload-arch=gfx950") for arg in arches):
        raise RuntimeError("Unexpected GPU architecture in compiler flags")
    if context["variant"] == "asan-debug" and any(
        ":xnack+" not in arg for arg in arches
    ):
        raise RuntimeError("ASan objects do not request xnack+")
    environment = dict(os.environ)
    library_dirs = [
        SDK / "lib",
        SDK / "lib64",
        SDK / "lib/rocm_sysdeps/lib",
        SDK / "lib/llvm/lib",
    ]
    sanitizer_link_flags = []
    sanitizer_runtime = None
    if context["variant"] == "asan-debug":
        environment["HSA_XNACK"] = "1"
        # Use the exact repository's standard ASan settings; retain leak checks.
        sys.path.insert(0, str(ROOT / "build_tools/github_actions"))
        from configure_asan_env import STATIC_ASAN_ENV

        environment.update(STATIC_ASAN_ENV)
        environment["ASAN_SYMBOLIZER_PATH"] = str(SDK / "lib/llvm/bin/llvm-symbolizer")
        sanitizer_link_flags = [
            "-shared-libsan"
        ]  # therock_sanitizers.cmake link policy.
        for name in ("libclang_rt.asan.so", "libclang_rt.asan-x86_64.so"):
            path = Path(
                subprocess.check_output(
                    [str(compiler), f"-print-file-name={name}"], text=True
                ).strip()
            )
            if path.is_absolute() and path.is_file():
                sanitizer_runtime = path
                library_dirs.append(path.parent)
                break
        if sanitizer_runtime is None:
            raise RuntimeError("Matching shared ASan runtime unavailable")
    environment["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in library_dirs if path.exists()
    )
    harness = OUT / "contract.o"
    run(
        [*common, "-c", str(HERE / "ck_f32_contract.cpp"), "-o", str(harness)],
        "compile-contract",
        environment,
    )
    link_common = []
    index = 0
    while index < len(common):
        if common[index] == "-x":
            index += 2
        else:
            link_common.append(common[index])
            index += 1
    save(
        "provenance.json",
        {
            "producer_context": context,
            "producer_run_metadata": producer_metadata,
            "validation_context": validation_context,
            "benchmark_protocol_sha256": digest(evidence / "benchmark-protocol.json"),
            "verified_benchmark_arms": verified_arms,
            "combined_object_bytes": {
                label: sum(path.stat().st_size for path in selected[trial])
                for trial, label in (("B2", "baseline"), ("C2", "candidate"))
            },
            "producer_root": old_root,
            "compiler_sha256": digest(compiler),
            "utility_sha256": digest(utility),
            "harness_source_sha256": digest(HERE / "ck_f32_contract.cpp"),
            "comparator_sha256": digest(HERE / "compare_contract.py"),
            "architecture_flags": arches,
            "harness_object_sha256": digest(harness),
            "sanitizer_link_flags": sanitizer_link_flags,
            "sanitizer_runtime_sha256": (
                digest(sanitizer_runtime) if sanitizer_runtime else None
            ),
            "runtime_env": {
                key: environment.get(key)
                for key in (
                    "HSA_XNACK",
                    "LD_LIBRARY_PATH",
                    "ASAN_OPTIONS",
                    "ASAN_SYMBOLIZER_PATH",
                )
            },
            "scope": "Exact isolated benchmark objects, not complete packaged CK/consumer validation.",
        },
    )
    link_summary = {}
    for trial, label in (("B2", "baseline"), ("C2", "candidate")):
        binary = OUT / ("contract-" + label)
        run(
            [
                *link_common,
                *sanitizer_link_flags,
                "--hip-link",
                str(harness),
                *map(str, selected[trial]),
                str(utility),
                f"-L{SDK / 'lib'}",
                "-lamdhip64",
                "-o",
                str(binary),
            ],
            "link-" + label,
            environment,
        )
        save(
            label + "-binary.json",
            {
                "path": str(binary),
                "sha256": digest(binary),
                "bytes": binary.stat().st_size,
                "objects": [
                    {
                        "path": str(path),
                        "sha256": digest(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in selected[trial]
                ],
            },
        )
        link_summary[label] = {
            "link_elapsed_s": json.loads(
                (OUT / ("link-" + label + "-result.json")).read_text()
            )["elapsed_s"],
            "combined_object_bytes": sum(
                path.stat().st_size for path in selected[trial]
            ),
            "final_binary_bytes": binary.stat().st_size,
            "scope": "Compiler-driver link command wall time; same harness object, exact measured CK objects and flags. No CK source recompilation.",
        }
        save("link-summary.json", link_summary)
        provenance = json.loads((OUT / "provenance.json").read_text())
        provenance["link_summary"] = link_summary
        save("provenance.json", provenance)
        run(["ldd", str(binary)], "dependencies-" + label, environment)
        run([str(binary), "--inventory"], label + "-inventory", environment)
        run([str(binary)], label + "-numeric", environment)
    run(
        [
            sys.executable,
            str(HERE / "compare_contract.py"),
            str(OUT / "baseline-inventory.log"),
            str(OUT / "candidate-inventory.log"),
            "--inventory-only",
            "--output",
            str(OUT / "inventory-comparison.json"),
        ],
        "compare-inventory",
    )
    run(
        [
            sys.executable,
            str(HERE / "compare_contract.py"),
            str(OUT / "baseline-numeric.log"),
            str(OUT / "candidate-numeric.log"),
            "--output",
            str(OUT / "numeric-comparison.json"),
        ],
        "compare-numeric",
    )


if __name__ == "__main__":
    main()
