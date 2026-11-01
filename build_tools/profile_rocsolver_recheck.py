# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixed-input required-test recheck; observations never change test execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from normal_ci_input_identity import file_identity, loader_dependencies, query_loader

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "build_tools/diagnostics/rocsolver_recheck_contract.json"


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(env, contract, root):
    expected = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "ROCm/TheRock",
        "GITHUB_ACTOR": "sa-faizal",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/" + contract["branch"],
        "GITHUB_WORKFLOW_REF": "ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/"
        + contract["branch"],
    }
    if any(env.get(key) != value for key, value in expected.items()):
        raise ValueError("Recheck must use its fixed TheRock context")
    payload = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
    inputs = payload.get("inputs", {})
    if inputs.get("profile_rocsolver_recheck") != "true" or any(
        inputs.get(key, "false") != "false"
        for key in ("profile_native_fixture_only", "profile_normal_validation")
    ):
        raise ValueError("Recheck-only mode conflicts with normal/native work")
    head = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if head != env.get("GITHUB_SHA"):
        raise ValueError("Checkout differs from authenticated CI SHA")
    if env.get("INPUT_PROFILE_NORMAL_VALIDATION") == "true":
        raise ValueError("Required-test recheck is not a normal comparison arm")
    for path, expected_hash in contract["source_files"].items():
        if fingerprint(root / path) != expected_hash:
            raise ValueError(f"Original test driver differs: {path}")
    return head


def selected_files(build):
    """Bounded candidate-file selection; loader query below is separate evidence."""
    selected = {build / "bin/rocsolver-test"}
    roots = [build / "lib", build / "lib64", build / "bin/rocsolver", build / ".kpack"]
    visited = 0
    for root in roots:
        if not root.is_dir():
            continue
        for directory, directories, names in os.walk(root, followlinks=False):
            # Package payloads may place references under rocblas/rocsolver.
            directories[:] = sorted(
                name for name in directories if name in {"rocblas", "rocsolver"}
            )
            for name in sorted(names):
                visited += 1
                if visited > 10000:
                    raise ValueError("Required-test input scan exceeded bound")
                if (
                    name.startswith(
                        (
                            "librocsolver",
                            "librocblas",
                            "libopenblas",
                            "liblapack",
                            "libblas",
                            "libflame",
                        )
                    )
                    or name.endswith(".cmake")
                    or (
                        root.name == ".kpack"
                        and name.startswith(("solver_lib_", "blas_lib_"))
                        and name.endswith(".kpack")
                    )
                ):
                    selected.add(Path(directory) / name)
    return sorted(selected)


def capture_inputs(root, component):
    paths = selected_files(root / "build") if component == "rocsolver" else []
    hashes = {}

    def shared_fingerprint(path):
        resolved = path.resolve(strict=True)
        if resolved not in hashes:
            hashes[resolved] = fingerprint(resolved)
        return hashes[resolved]

    rows = [file_identity(path, shared_fingerprint) for path in paths]
    if any(row["status"] != "file" for row in rows):
        raise ValueError("Selected required-test input is missing or unreadable")
    loaders = []
    queried = set()
    if component == "rocsolver":
        for name in ("solver_lib_gfx942.kpack", "blas_lib_gfx942.kpack"):
            if root / "build/.kpack" / name not in paths:
                raise ValueError(f"Original gfx942 code pack is missing: {name}")
        for path in paths:
            if path.name == "rocsolver-test" or path.name.startswith(
                ("librocsolver", "librocblas")
            ):
                if path.resolve() not in queried:
                    queried.add(path.resolve())
                    loaders.append(
                        loader_dependencies(path, shared_fingerprint, query_loader)
                    )
        names = [path.name for path in paths]
        if not any(name.startswith("librocsolver") for name in names) or not any(
            name.startswith("librocblas") for name in names
        ):
            raise ValueError("Required rocSOLVER/rocBLAS candidates not retained")
    return dict(
        files=rows,
        loader_queries=loaders,
        unique_hashed_files=len(hashes),
        scope="Selected SDK files and loader-query providers, not a complete SDK/dlopen or active-process map. No compilation or execution of test binaries during capture.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["context", "before", "after"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    contract = json.loads(CONTRACT.read_text())
    output = args.output.resolve()
    expected_output = ROOT / "required-recheck" / (args.phase + ".json")
    if output != expected_output:
        raise ValueError("Receipt must use its fixed retained path")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(phase=args.phase, complete=False)
    try:
        receipt.update(
            checkout_sha=guard(os.environ, contract, ROOT),
            run_id=os.environ["GITHUB_RUN_ID"],
            run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
            contract_sha256=fingerprint(CONTRACT),
            source_files=contract["source_files"],
        )
        if args.phase != "context":
            component = json.loads(os.environ["RECHECK_COMPONENT"])
            name = component["job_name"]
            if component != contract["components"].get(name):
                raise ValueError("Component selection/options differ from original")
            expected = {
                "ARTIFACT_RUN_ID": contract["artifact_run_id"],
                "NORMAL_REQUESTED_RUNNER": contract["runner"],
                "NORMAL_IMAGE_PIN": component.get("container_image", contract["image"]),
                "AMDGPU_FAMILIES": contract["family"],
                "AMDGPU_TARGETS": contract["targets"],
                "BUILD_VARIANT": "release",
            }
            if any(os.environ.get(key) != value for key, value in expected.items()):
                raise ValueError("Artifact, runner, image or family input changed")
            if int(os.environ["RECHECK_SHARD"]) not in component["shard_arr"]:
                raise ValueError("Unexpected required-test shard")
            receipt.update(
                component=component,
                shard=os.environ["RECHECK_SHARD"],
                artifact_run_id=contract["artifact_run_id"],
                runner_name=os.environ.get("RUNNER_NAME"),
                thread_limits={
                    key: os.environ.get(key)
                    for key in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "KUBE_CPU_REQUEST",
                    )
                },
                inputs=capture_inputs(ROOT, name),
            )
            if args.phase == "after":
                before = json.loads(output.with_name("before.json").read_text())
                for field in (
                    "checkout_sha",
                    "contract_sha256",
                    "component",
                    "shard",
                    "artifact_run_id",
                    "thread_limits",
                    "inputs",
                ):
                    if receipt[field] != before[field]:
                        raise ValueError(
                            f"Required-test inputs changed during execution: {field}"
                        )
        receipt["complete"] = True
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt.update(
            capture_wall_s=time.monotonic() - started,
            capture_cpu_s=time.process_time() - cpu,
        )
        output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
