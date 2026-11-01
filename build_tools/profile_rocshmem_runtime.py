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
BRANCH = "users/sa-faizal/build-profile-41147801272-rocshmem-runtime-heap"
RUNNER = "linux-gfx942-8gpu-ossci-rocm"
CONTRACT = ROOT / "diagnostics/rocshmem_runtime_contract.json"
CTEST_PATH = "build/bin/rocshmem/tests/functional/CTestTestfile.cmake"
CTEST_SHA = "d4965a4ed15fe00b4a17ba6777147ba85ac31cd0e68ddd95c02287f813b500f4"
HEAP_BYTES = 24 * 1024**3
MIN_FREE_BYTES = HEAP_BYTES + 8 * 1024**3
HEAP_TESTS = (
    "waveget_n2_w1_z64_v1073741824B_uuid",
    "waveget_n2_w2_z64_v1073741824B_uuid",
    "waveget_n2_w16_z1024_v1073741824B_uuid",
    "wgget_n2_w1_z1024_v1073741824B_uuid",
    "wgget_n2_w16_z1024_v1073741824B_uuid",
    "waveput_n2_w1_z64_v1073741824B_uuid",
    "waveput_n2_w2_z64_v1073741824B_uuid",
    "waveput_n2_w16_z1024_v1073741824B_uuid",
    "wgput_n2_w1_z1024_v1073741824B_uuid",
    "wgput_n2_w16_z1024_v1073741824B_uuid",
)


def capacity_text(original: bytes) -> bytes:
    """Change only ten infeasible heap capacities in the exact installed graph."""
    if hashlib.sha256(original).hexdigest() != CTEST_SHA:
        raise ValueError("Unexpected original installed CTest graph")
    lines = original.splitlines(keepends=True)
    for name in HEAP_TESTS:
        selected = [
            i
            for i, line in enumerate(lines)
            if line.startswith(f"add_test({name} ".encode())
        ]
        if len(selected) != 1:
            raise ValueError("Missing or duplicated exact capacity test")
        index = selected[0]
        old = b'"ROCSHMEM_HEAP_SIZE=6442450944"'
        if lines[index].count(old) != 1:
            raise ValueError("Unexpected original per-test heap capacity")
        lines[index] = lines[index].replace(
            old, f'"ROCSHMEM_HEAP_SIZE={HEAP_BYTES}"'.encode()
        )
    return b"".join(lines)


def capacity_file() -> Path:
    path = ROOT / CTEST_PATH
    if any(
        part.is_symlink() for part in [path, *path.parents] if part.is_relative_to(ROOT)
    ):
        raise ValueError("Aliased installed CTest graph")
    if (
        not path.is_file()
        or path.stat().st_nlink != 1
        or not path.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ValueError("Missing or escaping installed CTest graph")
    return path


def prepare_capacity() -> None:
    path = capacity_file()
    if (OUT / "heap-capacity.json").exists():
        raise ValueError("Refuse to adopt a prior capacity mutation")
    original = path.read_bytes()
    updated = capacity_text(original)
    (OUT / "original-CTestTestfile.cmake").write_bytes(original)
    path.write_bytes(updated)
    (OUT / "capacity-CTestTestfile.cmake").write_bytes(updated)
    save(
        "heap-capacity.json",
        dict(
            path=CTEST_PATH,
            original_sha256=CTEST_SHA,
            capacity_sha256=hashlib.sha256(updated).hexdigest(),
            tests=list(HEAP_TESTS),
            old_heap_bytes=6 * 1024**3,
            heap_bytes=HEAP_BYTES,
            payload_bytes_per_rank=20 * 1024**3,
            ranks=2,
            rank_device_mapping="SLR rank0->logical HIP0, rank1->logical HIP1; original CTest --parallel1",
            scope="Only ten generated add_test heap fields; original native files, arguments, iteration counts, test properties and skip policy unchanged.",
        ),
    )


def verify_capacity() -> None:
    receipt = json.loads((OUT / "heap-capacity.json").read_text())
    expected = capacity_text((OUT / "original-CTestTestfile.cmake").read_bytes())
    if (
        capacity_file().read_bytes() != expected
        or hashlib.sha256(expected).hexdigest() != receipt["capacity_sha256"]
    ):
        raise ValueError("Generated capacity graph changed after preparation")


def require_memory_budget(
    record: dict, minimum_free_bytes: int = MIN_FREE_BYTES
) -> None:
    devices = record.get("devices", [])
    if (
        record.get("count_returncode") != 0
        or record.get("count") != 8
        or len(devices) != 8
    ):
        raise ValueError("Exactly eight measured HIP devices required")
    for index, device in enumerate(devices):
        if (
            device.get("device") != index
            or device.get("set_returncode") != 0
            or device.get("memory_returncode") != 0
            or type(device.get("free_bytes")) is not int
            or type(device.get("total_bytes")) is not int
            or not minimum_free_bytes <= device["free_bytes"] <= device["total_bytes"]
        ):
            raise ValueError("Insufficient or unavailable per-device heap headroom")


def memory_budget(name: str, minimum_free_bytes: int = MIN_FREE_BYTES) -> dict:
    # Separate child exits before SLR forks; never initialize HIP in its parent.
    query = """
import ctypes,json,sys
h=ctypes.CDLL(sys.argv[1]); n=ctypes.c_int()
h.hipGetDeviceCount.argtypes=[ctypes.POINTER(ctypes.c_int)]
h.hipGetDeviceCount.restype=ctypes.c_int
h.hipSetDevice.argtypes=[ctypes.c_int]; h.hipSetDevice.restype=ctypes.c_int
h.hipMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
h.hipMemGetInfo.restype=ctypes.c_int
r=h.hipGetDeviceCount(ctypes.byref(n)); devices=[]
for index in range(min(max(n.value,0),8)):
    free=ctypes.c_size_t(); total=ctypes.c_size_t()
    selected=h.hipSetDevice(index)
    measured=h.hipMemGetInfo(ctypes.byref(free),ctypes.byref(total)) if selected==0 else None
    devices.append(dict(device=index,set_returncode=selected,memory_returncode=measured,free_bytes=free.value,total_bytes=total.value))
print(json.dumps(dict(count_returncode=r,count=n.value,devices=devices)))
"""
    result = command(
        [sys.executable, "-c", query, str(ROOT / "build/lib/libamdhip64.so")],
        name,
        provider_env(),
    )
    record = dict(
        command=result,
        observation=json.loads((OUT / (name + ".stdout")).read_text()),
        required_free_bytes=minimum_free_bytes,
        heap_bytes=HEAP_BYTES,
        scope="Point-in-time free/total memory per logical HIP device, after stock cleanup.8GiB outside-heap margin; concurrent external allocations remain possible.",
    )
    save(name + ".json", record)
    if result["returncode"]:
        raise ValueError("HIP memory observation failed")
    require_memory_budget(record["observation"], minimum_free_bytes)
    return record


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


def require_original_case_coverage(text: str) -> None:
    original = json.loads(
        (
            ROOT / "build_tools/tests/fixtures/rocshmem-runtime/actual-45-outcomes.json"
        ).read_text()
    )["rows"]
    pattern = r"^\s*\d+/\d+\s+Test\s+#(?P<id>\d+):\s+(?P<name>\S+)\s+\.+\s*(?P<status>Passed|\*{3}Skipped|\*{3}Failed)\s+"
    actual = [
        dict(id=int(m["id"]), name=m["name"], status=m["status"])
        for m in re.finditer(pattern, text, re.MULTILINE)
    ]
    expected = {(row["id"], row["name"]): row["status"] for row in original}
    if (
        len(expected) != 296
        or len(actual) != 296
        or {(row["id"], row["name"]) for row in actual} != set(expected)
    ):
        raise ValueError("Original296 selected case identities changed")
    for row in actual:
        prior = expected[(row["id"], row["name"])]
        if row["status"] != "Passed" and not (
            prior == "***Skipped" and row["status"] == "***Skipped"
        ):
            raise ValueError(
                "Required executed case failed or became skipped: " + row["name"]
            )


def prepare(contract: dict) -> None:
    if os.getenv("ROCSHMEM_HEAP_ALLOCATOR_TYPE") not in (None, ""):
        raise ValueError(
            "Capacity diagnostic requires the unchanged default heap allocator"
        )
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
                "ROCSHMEM_HEAP_ALLOCATOR_TYPE",
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
        difference="Installed build/bin PATH discovery plus generated CTest heap6GiB->24GiB for exactlyten1GiB-volume cases. Native bytes, original argv/loops/volume, other286 selected cases and test/security/skip policies unchanged.",
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
    record["memory_headroom"] = memory_budget("hip-memory-prepare")
    prepare_capacity()
    save("before.json", record)


def run_test() -> None:
    if not (OUT / "before.json").exists():
        raise RuntimeError("Missing preflight")
    if os.getenv("ROCSHMEM_HEAP_ALLOCATOR_TYPE") not in (None, ""):
        raise ValueError("Default heap allocator changed after preparation")
    verify_capacity()
    memory_budget("hip-memory-before-test")
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
        verify_capacity()
        after = identities(contract)
        save("after.json", after)
        memory_budget("hip-memory-after-test", minimum_free_bytes=0)
        before = json.loads((OUT / "before.json").read_text())
        if before["identities"] != after:
            raise RuntimeError("Native files or loader providers changed")
        code = json.loads((OUT / "test-exit.json").read_text())
        if code["returncode"]:
            raise RuntimeError("Original test command failed")
        outcomes = parse_ctest((OUT / code["stdout"]).read_text())
        require_original_case_coverage((OUT / code["stdout"]).read_text())
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
