# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Opt-in functional validation of already-produced rocSHMEM binaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runtime-probe"
BRANCH = "users/sa-faizal/build-profile-41147801272-rocshmem-runtime"
RUNNER = "linux-gfx942-8gpu-ossci-rocm"
CONTRACT = ROOT / "diagnostics/rocshmem_runtime_contract.json"


def save(name: str, value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(inputs: dict, env: dict, gpu: bool = False) -> None:
    expected = inputs.get("profile_rocshmem_expected_sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("Exact frozen diagnostic SHA required")
    if inputs.get("profile_rocshmem_runtime_probe") not in (True, "true"):
        raise ValueError("Explicit diagnostic enable required")
    for key in ("profile_normal_validation", "profile_native_fixture_only"):
        if key in inputs and inputs[key] not in (False, "false"):
            raise ValueError("Conflicting mode: " + key)
    required = {
        "GITHUB_REPOSITORY": "ROCm/TheRock",
        "GITHUB_ACTOR": "sa-faizal",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_SHA": expected,
        "GITHUB_WORKFLOW_SHA": expected,
        "GITHUB_REF": "refs/heads/" + BRANCH,
        "GITHUB_WORKFLOW_REF": f"ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/{BRANCH}",
    }
    if any(env.get(key) != value for key, value in required.items()):
        raise ValueError("Repository/actor/source/workflow mismatch")
    if gpu and (
        env.get("RUNNER_OS") != "Linux"
        or not env.get("RUNNER_NAME", "").startswith(RUNNER + "-")
        or env.get("NORMAL_REQUESTED_RUNNER") != RUNNER
    ):
        raise ValueError("Unexpected runtime allocation")


def command(argv: list[str], name: str, env: dict) -> dict:
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=60)
    if len(result.stdout) + len(result.stderr) > 8 * 1024**2:
        raise RuntimeError("Diagnostic output exceeds bound")
    (OUT / (name + ".stdout")).write_bytes(result.stdout)
    (OUT / (name + ".stderr")).write_bytes(result.stderr)
    return dict(
        argv=argv,
        returncode=result.returncode,
        wall_s=time.monotonic() - started,
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
    )


def identities(contract: dict) -> dict:
    files, providers = [], {}
    for expected in contract["installed_files"]:
        relative = Path(expected["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid installed-file contract")
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Installed-file alias escapes the workspace")
        item = dict(
            path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha(path)
        )
        if any(item[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError("Installed artifact mismatch: " + str(relative))
        files.append(item)
        if not expected.get("images"):
            continue
        if (
            len(expected["images"]) != 31
            or len({x["target_id"] for x in expected["images"]}) != 31
        ):
            raise ValueError("Incomplete native device-image identity")
        name = "ldd-" + path.name
        result = command(["ldd", str(path)], name, provider_env())
        output = (OUT / (name + ".stdout")).read_text()
        if result["returncode"] or "not found" in output:
            raise RuntimeError("Unresolved native loader provider")
        for line in output.splitlines():
            match = re.search(r"(?:=>\s+|^\s*)(/\S+)", line)
            if match:
                provider = Path(match[1]).resolve()
                providers[str(provider)] = dict(
                    bytes=provider.stat().st_size, sha256=sha(provider)
                )
    return dict(files=files, loader_providers=providers)


def corrected_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = str(ROOT / "build/bin") + os.pathsep + env.get("PATH", "")
    return env


def provider_env() -> dict:
    env = corrected_env()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(ROOT / "build/lib"),
                str(ROOT / "build/lib/rocm_sysdeps/lib"),
                env.get("LD_LIBRARY_PATH", ""),
            ],
        )
    )
    return env


def parse_ctest(text: str) -> dict:
    summaries = re.findall(r"100% tests passed, 0 tests failed out of (\d+)", text)
    passed = len(re.findall(r"\bPassed\s+[0-9.]+ sec", text))
    skipped = len(re.findall(r"\*\*\*Skipped\s+[0-9.]+ sec", text))
    if len(summaries) != 1 or passed + skipped != int(summaries[0]) or not passed:
        raise ValueError("Failed, incomplete, or zero-executed test selection")
    return dict(selected=int(summaries[0]), passed=passed, skipped=skipped)


def prepare(contract: dict) -> None:
    record = dict(
        contract_sha256=sha(CONTRACT),
        producer=contract,
        ci={
            k: os.getenv(k)
            for k in [
                "GITHUB_RUN_ID",
                "GITHUB_SHA",
                "RUNNER_NAME",
                "NORMAL_REQUESTED_RUNNER",
            ]
        },
        environment={
            k: os.getenv(k)
            for k in [
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "KUBE_CPU_REQUEST",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
            ]
        },
        ambient_tools={
            name: shutil.which(name) for name in ["amd-smi", "rocm-smi", "rocminfo"]
        },
        affinity=sorted(os.sched_getaffinity(0)),
        cpu_model=next(
            (
                line.partition(":")[2].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        cgroups={
            name: (
                (Path("/sys/fs/cgroup") / name).read_text()
                if (Path("/sys/fs/cgroup") / name).exists()
                else None
            )
            for name in ["cpu.max", "cpuset.cpus.effective", "memory.max", "pids.max"]
        },
        difference="Only installed build/bin prepended to rocSHMEM subprocess PATH; no installed source, test, CTest or skip-policy changes.",
    )
    save("before.json", record)
    record["identities"] = identities(contract)
    env = corrected_env()
    record["corrected_tools"] = {
        name: shutil.which(name, path=env["PATH"]) for name in ["amd-smi", "rocminfo"]
    }
    for tool in ("amd-smi", "rocminfo"):
        if record["corrected_tools"][tool] != str(ROOT / "build/bin" / tool):
            raise RuntimeError("Installed prerequisite missing")
    record["amd_smi_version"] = command(
        [str(ROOT / "build/bin/amd-smi"), "version"], "amd-smi-version", env
    )
    record["amd_smi_list"] = command(
        [str(ROOT / "build/bin/amd-smi"), "list"], "amd-smi-list", env
    )
    record["rocminfo"] = command([str(ROOT / "build/bin/rocminfo")], "rocminfo", env)
    hsa = (OUT / "rocminfo.stdout").read_text()
    record["hsa_gfx942_agents"] = len(
        re.findall(r"^\s*Name:\s+gfx942\s*$", hsa, re.MULTILINE)
    )
    query = "import ctypes,json,sys; h=ctypes.CDLL(sys.argv[1]); n=ctypes.c_int(); h.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)]; h.hipGetDeviceCount.restype=ctypes.c_int; r=h.hipGetDeviceCount(ctypes.byref(n)); print(json.dumps(dict(returncode=r,count=n.value)))"
    record["hip_query"] = command(
        [sys.executable, "-c", query, str(ROOT / "build/lib/libamdhip64.so")],
        "hip-device-count",
        provider_env(),
    )
    record["hip_device_count"] = json.loads(
        (OUT / "hip-device-count.stdout").read_text()
    )
    save("before.json", record)
    if (
        any(
            record[k]["returncode"]
            for k in ["amd_smi_version", "amd_smi_list", "rocminfo", "hip_query"]
        )
        or record["hip_device_count"]["returncode"]
        or record["hip_device_count"]["count"] < 8
        or record["hsa_gfx942_agents"] < 8
    ):
        raise RuntimeError(
            "Actual HSA/HIP/management prerequisites do not expose8 GPUs"
        )


def run_test() -> None:
    if not (OUT / "before.json").exists():
        raise RuntimeError("Missing preflight")
    started = time.monotonic()
    argv = [
        sys.executable,
        "build_tools/github_actions/test_executable_scripts/test_runner.py",
    ]
    attempt = len(list(OUT.glob("test-exit-*.json"))) + 1
    output_name = f"original-test-{attempt}.stdout"
    with (OUT / output_name).open("w") as stream:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=corrected_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            stream.write(line)
            print(line, end="", flush=True)
        process.stdout.close()
        code = process.wait()
    result = dict(
        argv=argv,
        returncode=code,
        wall_s=time.monotonic() - started,
        stdout=output_name,
    )
    save(f"test-exit-{attempt}.json", result)
    save("test-exit.json", result)
    if code:
        raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["guard", "prepare", "test", "finalize"])
    args = parser.parse_args()
    guard(json.loads(os.environ["PROBE_INPUTS"]), os.environ, args.mode != "guard")
    contract = json.loads(CONTRACT.read_text())
    if (
        type(contract["artifact_run"]) is not int
        or contract["artifact_run"] <= 0
        or contract["artifact_source"] != "60faab8d3d433d009f154c3e30f9fd8ac5cfda59"
    ):
        raise ValueError("Unexpected producer contract")
    if args.mode == "guard":
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write(f"artifact_run={contract['artifact_run']}\n")
        return
    if os.environ.get("ARTIFACT_RUN_ID") != str(contract["artifact_run"]):
        raise ValueError("Fetched producer differs from the frozen contract")
    if args.mode == "prepare":
        prepare(contract)
    elif args.mode == "test":
        run_test()
    else:
        after = identities(contract)
        save("after.json", after)
        before = json.loads((OUT / "before.json").read_text())
        if before["identities"] != after:
            raise RuntimeError("Native files or loader providers changed")
        code = json.loads((OUT / "test-exit.json").read_text())
        if code["returncode"]:
            raise RuntimeError("Original test command failed")
        outcomes = parse_ctest((OUT / code["stdout"]).read_text())
        if outcomes["selected"] != 296:
            raise RuntimeError("Original comprehensive selection changed")
        save(
            "result.json",
            dict(
                success=True,
                **outcomes,
                scope="Functional check only; no performance claim. Executed gfx942 coverage only; other30 images have static identity checks.",
            ),
        )


if __name__ == "__main__":
    operation = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1] in ["guard", "prepare", "test", "finalize"]
        else "entry"
    )
    started_wall, started_cpu = time.monotonic(), time.process_time()
    try:
        main()
    except Exception as error:
        save(
            "failure-" + operation + ".json",
            dict(type=type(error).__name__, message=str(error)[:4096]),
        )
        raise
    finally:
        save(
            "accounting-" + operation + ".json",
            dict(
                wall_s=time.monotonic() - started_wall,
                own_cpu_s=time.process_time() - started_cpu,
                scope="Controller own CPU only; wall includes child tools/tests. Metadata operations and functional test are separate modes; no performance-gain inference.",
            ),
        )
