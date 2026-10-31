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
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    save(label + "-result.json", {"exit_code": result.returncode})
    result.check_returncode()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    contexts = list(EVIDENCE.rglob("context.json"))
    if len(contexts) != 1:
        raise RuntimeError("Expected exactly one build context")
    evidence = contexts[0].parent
    context = json.loads(contexts[0].read_text())
    for key, expected in (
        ("run_id", os.environ["GITHUB_RUN_ID"]),
        ("instrumentation_sha", os.environ["GITHUB_SHA"]),
        ("variant", os.environ["BUILD_VARIANT"]),
        ("baseline_run_id", os.environ["BASELINE_RUN_ID"]),
    ):
        if str(context[key]) != expected:
            raise RuntimeError(f"Build artifact provenance mismatch: {key}")
    if context["source_sha"] != "5b7cb88373362ab80dc0f19605915e534f4158aa":
        raise RuntimeError("Product source changed")
    protocol = json.loads((evidence / "benchmark-protocol.json").read_text())
    summary = json.loads((evidence / "benchmark-summary.json").read_text())
    if [row["trial"] for row in summary] != ["B1", "C1", "C2", "B2"]:
        raise RuntimeError("Paired benchmark incomplete")
    original_source = next(iter(protocol["baseline_sources"].values()))["path"]
    old_root = original_source.split("/rocm-libraries/")[0]
    selected = {}
    for trial in ("B2", "C2"):
        row = next(row for row in summary if row["trial"] == trial)
        objects = []
        for item in row["objects"]:
            relative = Path(item["path"]).relative_to(old_root)
            path = BUNDLE / relative
            if digest(path) != item["sha256"]:
                raise RuntimeError(f"Object digest mismatch: {path}")
            objects.append(path)
        if len(objects) != (1 if trial == "B2" else 3):
            raise RuntimeError("Unexpected object count")
        selected[trial] = objects
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
                "objects": [
                    {"path": str(path), "sha256": digest(path)}
                    for path in selected[trial]
                ],
            },
        )
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
