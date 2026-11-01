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
import sys

from normal_ci_input_identity import file_identity, loader_dependencies, query_loader

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "build_tools/diagnostics/profiler_lifecycle_contract.json"


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
    if inputs.get("profile_profiler_lifecycle") != "true" or any(
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
    if env.get("GITHUB_WORKFLOW_SHA") != head:
        raise ValueError("Primary workflow differs from authenticated CI SHA")
    if env.get("INPUT_PROFILE_NORMAL_VALIDATION") == "true":
        raise ValueError("Required-test recheck is not a normal comparison arm")
    for path, expected_hash in contract["source_files"].items():
        if fingerprint(root / path) != expected_hash:
            raise ValueError(f"Original test driver differs: {path}")
    return head


def selected_files(build):
    selected = {build / "bin/rocprofv3"}
    visited = 0
    for root in [
        build / "bin",
        build / "lib",
        build / "lib64",
        build / ".kpack",
        build / "share/rocprofiler-sdk/tests",
    ]:
        if not root.is_dir():
            continue
        for directory, directories, names in os.walk(root, followlinks=False):
            directories[:] = sorted(
                n
                for n in directories
                if n not in {"Testing", "__pycache__"}
                and not (root == build / "share/rocprofiler-sdk/tests" and n == "build")
            )
            for name in sorted(names):
                visited += 1
                if visited > 50000:
                    raise ValueError("Input scan exceeded its bound")
                if root in [
                    build / ".kpack",
                    build / "share/rocprofiler-sdk/tests",
                ] or name.startswith(
                    (
                        "rocprofv3",
                        "librocprofiler",
                        "libamdhip",
                        "libamd_comgr",
                        "libhiprtc",
                        "libhsa-runtime",
                    )
                ):
                    selected.add(Path(directory) / name)
    return sorted(selected)


def capture_inputs(root, component, previous=None):
    paths = selected_files(root / "build")
    if previous is not None:
        paths = [Path(row["path"]) for row in previous["files"]]
        if any(not path.is_relative_to(root / "build") for path in paths):
            raise ValueError("Prior source path escaped the SDK")
    hashes = {}

    def cached(path):
        resolved = path.resolve(strict=True)
        if resolved not in hashes:
            hashes[resolved] = fingerprint(resolved)
        return hashes[resolved]

    rows = [file_identity(path, cached) for path in paths]
    if any(row["status"] != "file" for row in rows):
        raise ValueError("Required producer SDK input missing or unreadable")
    loaders, seen = [], set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.name.startswith(
            (
                "librocprofiler",
                "libamdhip",
                "libamd_comgr",
                "libhiprtc",
                "libhsa-runtime",
            )
        ):
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        seen.add(resolved)
        loaders.append(loader_dependencies(path, cached, query_loader))
    return dict(
        files=rows,
        loader_queries=loaders,
        unique_hashed_files=len(hashes),
        scope="Selected shipped inputs and capture-environment loader queries, not complete active-process/dlopen/cache identity. Consumer-built app is separately hashed after stock configure/build.",
    )


ENVIRONMENT_KEYS = (
    "HOME",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_XNACK",
    "HSA_ENABLE_SDMA",
    "HSA_ENABLE_PEER_SDMA",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "KUBE_CPU_REQUEST",
    "HIPRTC_COMPILE_OPTIONS_APPEND",
    "HIPRTC_LINK_OPTIONS_APPEND",
    "AMD_COMGR_CACHE",
    "AMD_COMGR_CACHE_DIR",
    "AMD_COMGR_CACHE_POLICY",
    "AMD_COMGR_DRIVER_OPTIONS_APPEND",
    "ROCPROFILER_LIBRARY_VERBOSE",
    "ROCPROFILER_LOG_LEVEL",
    "MIOPEN_FIND_MODE",
    "MIOPEN_FIND_ENFORCE",
    "MIOPEN_SYSTEM_DB_PATH",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "MIOPEN_DEBUG_DISABLE_FIND_DB",
    "MIOPEN_DEBUG_DISABLE_FIND_DB_SOLVER_FILTERING",
    "MIOPEN_DEBUG_FIND_ONLY_SOLVER",
    "MIOPEN_DEBUG_DISABLE_CACHE",
    "MIOPEN_ENABLE_LOGGING",
    "MIOPEN_ENABLE_LOGGING_CMD",
    "MIOPEN_LOG_LEVEL",
)


def runtime_context(env):
    """Observe explicit nonsecret knobs and visible hardware without changing policy."""
    files = {}
    for path in (
        Path("/sys/module/amdgpu/version"),
        Path("/proc/driver/amdgpu/version"),
    ):
        try:
            files[str(path)] = path.read_text()[:65536]
        except OSError:
            files[str(path)] = None
    topology = Path("/sys/class/kfd/kfd/topology/nodes")
    if topology.is_dir():
        for path in sorted(topology.glob("*/gpu_id")):
            files[str(path)] = path.read_text()[:256]
    return dict(
        environment={key: env.get(key) for key in ENVIRONMENT_KEYS},
        logical_cpus=os.cpu_count(),
        affinity=(
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        hardware_files=files,
        scope="Explicit environment and visible static hardware; original driver/GPU sanity logs retain actual device inventory. Unset cache knobs do not prove cache disabled; mutable cache contents/active-process providers are not fully captured.",
    )


def cache_metadata(env):
    rows = []
    home = Path(env.get("HOME", str(ROOT)))
    for root in [home / ".cache/miopen", home / ".cache/comgr"]:
        entries = []
        if root.is_dir():
            for directory, dirs, names in os.walk(root, followlinks=False):
                dirs.sort()
                for name in sorted(names):
                    path = Path(directory) / name
                    stat = path.lstat()
                    entries.append(
                        dict(
                            path=str(path.relative_to(root)),
                            bytes=stat.st_size,
                            mtime_ns=stat.st_mtime_ns,
                        )
                    )
                    if len(entries) >= 2000:
                        break
                if len(entries) >= 2000:
                    break
        rows.append(
            dict(
                path=str(root),
                exists=root.is_dir(),
                entries=entries,
                truncated=len(entries) >= 2000,
            )
        )
    return dict(
        candidate_locations=rows,
        scope="Read-only metadata for two HOME cache candidates, not proof of effective paths. No cache setting/content changes. Ordinary test writes are observed rather than rejected.",
    )


def main():
    if sys.argv[1:] == ["run"]:
        contract = json.loads(CONTRACT.read_text())
        guard(os.environ, contract, ROOT)
        from profile_profiler_lifecycle_trial import run_trial

        return run_trial(ROOT, contract)
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["context", "before", "after"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    contract = json.loads(CONTRACT.read_text())
    output = args.output.resolve()
    expected_output = ROOT / "profiler-lifecycle" / (args.phase + ".json")
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
            before = (
                json.loads(output.with_name("before.json").read_text())
                if args.phase == "after"
                else None
            )
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
                inputs=capture_inputs(ROOT, name, before["inputs"] if before else None),
                runtime_context=runtime_context(os.environ),
                cache_metadata=cache_metadata(os.environ),
            )
            if args.phase == "after":
                for field in (
                    "checkout_sha",
                    "contract_sha256",
                    "component",
                    "shard",
                    "artifact_run_id",
                    "thread_limits",
                    "inputs",
                    "runtime_context",
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
    raise SystemExit(main())
