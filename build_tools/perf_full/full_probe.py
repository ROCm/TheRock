#!/usr/bin/env python3
"""Full-CI experiment provenance/guards; product execution only in TheRock CI."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
import zipfile

BASE = "5b7cb88373362ab80dc0f19605915e534f4158aa"
LIBS = "f4e24d58b3f313f1a31068def1e35376e43c2f49"
SYSTEMS = "a10dca087065a82f15ea89dc7d6fdd3c6bb2c246"
CONFIG = "7624a5e209cec569bbed38dad7071fa7393cb52b"
HERE = Path(__file__).resolve().parent
OUTPUT = Path(".perf-full")


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={Path.cwd()}", *args], text=True).strip()


def save(name, data):
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def guard():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "ROCm/TheRock":
        raise RuntimeError("Full experiment execution is restricted to ROCm/TheRock Actions")
    if git("rev-parse", "HEAD") != BASE:
        raise RuntimeError("Source checkout does not match frozen HEAD")
    mode = os.environ.get("PERF_MODE", "")
    if mode not in ("baseline", "joint"): raise RuntimeError("Missing explicit baseline/joint mode")
    return mode


def limits():
    result = {"logical_cpus": os.cpu_count(), "platform": platform.platform(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "KUBE_CPU_REQUEST": os.environ.get("KUBE_CPU_REQUEST"),
              "limits": "Affinity is visibility; hidden quotas/siblings and filesystem residency are not established."}
    if sys.platform == "linux":
        spec = importlib.util.spec_from_file_location("full_observer", HERE / "collect_build_perf.py")
        observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
        result["visible_cgroups"] = observer.groups_snapshot(observer.cgroup_dirs())
        result["cpu_models"] = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)))
    return result


def preflight(label):
    mode = guard()
    ci_config = None
    if label in ("setup", "configure_test_matrix", "focused_configure_test_matrix"):
        if not Path("ci-config/.git").exists() or git("-C", "ci-config", "rev-parse", "HEAD") != CONFIG:
            raise RuntimeError("Required pinned external CI config checkout is absent or mismatched")
        if git("-C", "ci-config", "diff", "--name-only") or git("-C", "ci-config", "diff", "--cached", "--name-only"):
            raise RuntimeError("External CI config has tracked modifications")
        ci_config = {"head": CONFIG, "tree": git("-C", "ci-config", "rev-parse", "HEAD^{tree}"),
                     "files": {name: digest(Path("ci-config") / name) for name in git("-C", "ci-config", "ls-files").splitlines() if (Path("ci-config") / name).is_file()}}
        if "runner-config-v2.json" not in ci_config["files"]: raise RuntimeError("Missing required runner matrix")
    context = os.environ.get("PERF_CONTEXT_JSON", "") + os.environ.get("PERF_MATRIX_JSON", "")
    artifact = "full-perf-" + mode + "-" + re.sub(r"[^A-Za-z0-9_.-]", "-", label)[:60] + "-" + hashlib.sha256(context.encode()).hexdigest()[:10]
    inputs = json.loads(os.environ.get("PERF_CONTEXT_JSON") or "{}")
    selected = {k: inputs.get(k) for k in ("stage_name", "amdgpu_family", "amdgpu_families", "dist_amdgpu_families", "build_variant", "build_variant_cmake_preset", "rocm_package_version", "release_type", "repository", "ref", "native_package_type", "os_profile", "default_container_image")}
    if inputs.get("component"):
        component = json.loads(inputs["component"])
        selected["component"] = {k: component.get(k) for k in ("job_name", "total_shards", "test_type", "container_image", "container_options", "linux_cpu_runner", "multi_gpu_runner")}
    save("preflight-" + re.sub(r"[^A-Za-z0-9_.-]", "-", label) + ".json", {
        "utc": utc(), "mode": mode, "source_sha": BASE, "instrumentation_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"), "requested_runner": os.environ.get("PERF_REQUESTED_RUNNER"),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "job": os.environ.get("GITHUB_JOB"),
        "stage": os.environ.get("STAGE_NAME"), "label": label, "resources": limits(),
        "selected_inputs": selected, "matrix": json.loads(os.environ.get("PERF_MATRIX_JSON") or "null"), "ci_config": ci_config,
        "cache_contract": "CCACHE_DISABLE=1; filesystem cache state unknown; no OS cache drop is claimed.",
        "environment": {k: os.environ.get(k) for k in ("KUBE_CPU_REQUEST", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_XNACK", "CCACHE_DISABLE", "ROCM_PACKAGE_VERSION", "BUILD_VARIANT")}})
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f: f.write("artifact_name=" + artifact + "\n")
    if os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f: f.write("CCACHE_DISABLE=1\nPERF_MODE=" + mode + "\n")


def guard_ci(label):
    guard()
    if os.environ.get("RELEASE_TYPE") != "ci": raise RuntimeError("Non-CI release type is forbidden")
    sys.path.insert(0, str(Path("build_tools").resolve()))
    from _therock_utils.workflow_outputs import WorkflowOutputRoot
    run = os.environ["GITHUB_RUN_ID"]
    root = WorkflowOutputRoot.from_workflow_run(run_id=run, platform="linux")
    if root.bucket != "therock-ci-artifacts" or root.prefix != run + "-linux":
        raise RuntimeError(f"Unexpected destination {root.bucket}/{root.prefix}")
    save("ci-destination-" + label + ".json", {"utc": utc(), "bucket": root.bucket, "prefix": root.prefix, "mode": os.environ["PERF_MODE"]})


def verify_module(module, expected):
    entry = git("ls-tree", BASE, "--", module).split()
    if len(entry) < 3 or entry[2] != expected: raise RuntimeError("Frozen gitlink differs: " + module)
    head = git("-C", module, "rev-parse", "HEAD")
    stock = sorted((Path("patches/amd-mainline") / module).glob("*.patch"))
    def patch_id(data):
        value = subprocess.check_output(["git", "patch-id", "--stable"], input=data).decode().split()
        if not value: raise RuntimeError("Empty stock patch diff")
        return value[0]
    expected_ids = [patch_id(p.read_bytes()) for p in stock]
    commits = git("-C", module, "rev-list", "--reverse", expected + ".." + head).splitlines()
    actual_ids = [patch_id(subprocess.check_output(["git", "-C", module, "show", "--format=", "--binary", commit])) for commit in commits]
    if expected_ids != actual_ids: raise RuntimeError("Effective stock patch chain differs: " + module)
    if subprocess.run(["git", "-C", module, "merge-base", "--is-ancestor", expected, head]).returncode:
        raise RuntimeError("Effective source is not based on frozen gitlink")
    if git("-C", module, "diff", "--name-only") or git("-C", module, "diff", "--cached", "--name-only"):
        raise RuntimeError("Unexpected pre-experiment tracked source changes: " + module)
    return {"module": module, "requested_gitlink": expected, "effective_stock_head": head,
            "effective_stock_tree": git("-C", module, "rev-parse", "HEAD^{tree}"),
            "stock_patch_ids": actual_ids, "stock_patch_files": [str(p) for p in stock]}


def prepare_stage():
    mode = guard(); stage = os.environ["STAGE_NAME"]
    topology = tomllib.loads(Path("BUILD_TOPOLOGY.toml").read_text())
    groups = topology["build_stages"][stage]["artifact_groups"]
    sdk_group = topology["artifacts"]["rocprofiler-sdk"]["artifact_group"]
    patches, effective_sources = [], []
    if sdk_group in groups:
        if Path("build/profiler/rocprofiler-sdk/stage.prebuilt").exists():
            raise RuntimeError("SDK producer unexpectedly imported SDK rather than building it")
        effective_sources.append(verify_module("rocm-systems", SYSTEMS))
        patch = HERE / "bootstrap-codeobj.patch"
        subprocess.run(["git", "-C", "rocm-systems", "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", "rocm-systems", "apply", str(patch)], check=True)
        target = Path("profiler/pre_hook_rocprofiler-sdk.cmake")
        if target.exists(): raise RuntimeError("Unexpected pre-existing SDK pre-hook")
        shutil.copy2(HERE / "pre_hook_rocprofiler-sdk.cmake", target)
        patches.append({"kind": "common SDK fixture repair", "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "common_hook_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "prefix_strategy": "Generated toolchain HIP flags, verified at SDK configure time"})
    if stage == "math-libs":
        effective_sources.append(verify_module("rocm-libraries", LIBS))
        if mode == "joint":
            patch = HERE / "rocsolver-specialized-shards.patch"
            subprocess.run(["git", "-C", "rocm-libraries", "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", "rocm-libraries", "apply", str(patch)], check=True)
            patches.append({"kind": "joint solver patch", "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()})
    extra = ["-DCMAKE_PROJECT_THEROCK_INCLUDE_BEFORE=" + str(HERE / "full_project_hook.cmake"),
             "-DTHEROCK_BACKGROUND_BUILD_JOBS=9", "-DLLVM_PARALLEL_LINK_JOBS=4"]
    if stage == "math-libs":
        launcher = "python3;" + str(HERE / "command_trace.py") + ";ccache"
        extra += ["-DCMAKE_" + lang + "_COMPILER_LAUNCHER=" + launcher for lang in ("C", "CXX", "HIP")]
        commands = (Path.cwd() / OUTPUT / "compiler-commands").resolve()
        commands.mkdir(parents=True, exist_ok=True)
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
            f.write("THEROCK_PERF_COMMAND_DIR=" + str(commands) + "\n")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "configure-args.txt").write_text("\n".join(extra) + "\n", encoding="utf-8")
    save("stage-source-manifest.json", {"utc": utc(), "source_sha": BASE, "mode": mode, "stage": stage,
        "package_version": os.environ.get("ROCM_PACKAGE_VERSION"), "patches": patches, "effective_stock_sources": effective_sources,
        "submodules": git("submodule", "status", "--recursive"), "configure_args": extra,
        "limits": "Common repair and instrumentation are separate from candidate; source HEAD remains frozen."})
    for module in ("rocm-systems", "rocm-libraries"):
        if (Path(module) / ".git").exists():
            (OUTPUT / (module + "-experimental.diff")).write_text(git("-C", module, "diff"), encoding="utf-8")


def stage_build():
    guard(); stage = os.environ["STAGE_NAME"]; build = os.environ.get("BUILD_DIR", "build")
    command = ["cmake", "--build", build, "--target", "stage-" + stage, "therock-artifacts", "--", "-k", "0"]
    if stage == "math-libs":
        commands = Path.cwd() / OUTPUT / "compiler-commands"
        commands.mkdir(parents=True, exist_ok=True)
        os.environ["THEROCK_PERF_COMMAND_DIR"] = str(commands)
        command = [sys.executable, str(HERE / "collect_build_perf.py"), "--output", str(OUTPUT / "math-observer"),
                   "--build-dir", build, "--interval", "1", "--", *command]
    start = time.monotonic(); code = subprocess.run(command).returncode
    save("build-result.json", {"utc": utc(), "argv": command, "exit_code": code,
        "observed_wrapper_wall_s_including_export": time.monotonic() - start,
        "limits": "Math observer metadata separates command span/export; observer CPU is not elapsed perturbation."})
    return code


def matrix(kind):
    guard()
    if kind == "build":
        config = json.loads(os.environ["PERF_BUILD_CONFIG"])
        assert config["build_variant_label"] == "asan-debug"
        assert config["build_variant_cmake_preset"] == "linux-release-asan-debug"
        assert config["dist_amdgpu_families"] == "gfx950-dcgpu"
        assert len(config["per_family_info"]) == 1
        assert not config.get("skip_stages") and not config.get("prebuilt_stages")
        assert not config["build_python_packages"] and not config["build_pytorch"] and not config["build_jax"]
        assert config["build_native_linux"]
        save("build-matrix.json", {"config": config, "manual_gpu_runner_override": "linux-gfx950-1gpu-ccs-ossci-rocm", "test_type": os.environ.get("PERF_TEST_TYPE")})
    else:
        components = json.loads(os.environ["PERF_TEST_COMPONENTS"])
        sanity = json.loads(os.environ["PERF_TEST_SANITY"])
        assert components and sanity and sanity.get("test_runner")
        assert any(c["job_name"] == "rocsolver" and c.get("test_runner") and c.get("shard_arr") for c in components)
        assert all(c.get("test_runner") or c.get("multi_gpu_runner") for c in components)
        save("test-matrix.json", {"components": components, "sanity": sanity})


def finalize():
    guard()
    save("final-provenance.json", {"utc": utc(), "mode": os.environ["PERF_MODE"], "source_sha": git("rev-parse", "HEAD"),
        "submodules": git("submodule", "status", "--recursive"), "tracked_diff": git("diff", "--submodule=short"),
        "resources": limits(), "limits": "GPU numerical, ASan and artifact checks remain the stock workflow steps; no optimization acceptance is inferred here."})
    prefix = Path("build/profiler/rocprofiler-sdk/build/perf-sdk-prefixes.txt")
    if prefix.exists(): shutil.copy2(prefix, OUTPUT / "sdk-actual-prefixes.txt")
    cache = Path("build/CMakeCache.txt")
    if cache.exists(): shutil.copy2(cache, OUTPUT / "top-CMakeCache.txt")
    results = []
    for path in Path("build").rglob("LastTest.log"):
        entry = {"path": str(path), "bytes": path.stat().st_size}
        if path.stat().st_size <= 64 * 1024 * 1024:
            dest = OUTPUT / "test-results" / path
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
            entry["retained"] = str(dest)
        else: entry["omitted"] = "64MiB log cap; complete Actions job log must also be retained"
        results.append(entry)
    save("test-result-inventory.json", {"logs": results, "limits": "Existing test status and raw Actions logs remain authoritative; log presence is not numerical coverage."})
    if os.environ.get("STAGE_NAME") == "math-libs" and Path("build/math-libs/BLAS/rocSOLVER/stage").exists():
        stage = Path("build/math-libs/BLAS/rocSOLVER/stage")
        libraries = [p for p in stage.rglob("librocsolver.so*") if p.is_file() and not p.is_symlink()]
        if len(libraries) != 1: raise RuntimeError("Expected one real staged solver DSO")
        lib = libraries[0]
        nm = shutil.which("nm")
        if not nm: raise RuntimeError("nm unavailable for required public symbol inventory")
        lines = subprocess.check_output([nm, "--dynamic", "--defined-only", "--format=posix", "--no-demangle", str(lib)], text=True).splitlines()
        save("solver-public-symbols.json", {"library": str(lib), "defined_symbols": sorted(" ".join(l.split()[:2]) for l in lines if len(l.split()) >= 2),
            "limits": "Compare baseline/joint symbol inventories; equality is not numerical, device-registration or ODR validation."})
        solver_identity()
        toolchain_identity()


def solver_identity():
    # Kpack may transform stage binaries: attest post-split consumer bytes.
    distribution = Path("build/dist/rocm")
    artifact_files = [p for d in Path("build/artifacts").glob("solver_*") for p in d.rglob("*") if p.is_file() and not p.is_symlink()]
    files = []
    for role, pattern in (("rocsolver_bench", "rocsolver-bench"), ("rocsolver_test", "rocsolver-test"), ("librocsolver", "librocsolver.so*")):
        found = [p for p in distribution.rglob(pattern) if p.is_file() and not p.is_symlink() and not any(x.endswith((".debug", ".dbg")) for x in p.parts)]
        if len(found) != 1: raise RuntimeError("Cannot identify unique post-split SDK " + role)
        path = found[0]; sha = digest(path)
        matches = [p for p in artifact_files if p.name == path.name and digest(p) == sha]
        if not matches: raise RuntimeError("Distribution bytes do not match exported solver artifacts: " + role)
        files.append({"role": role, "basename": path.name, "stage_relative_path": str(path.relative_to(distribution)), "sha256": sha, "bytes": path.stat().st_size, "artifact_sources": list(map(str,matches))})
    payloads = []
    for path in distribution.rglob("*"):
        if not path.is_file() or path.is_symlink() or ".kpack" not in path.parts or "solver" not in str(path.relative_to(distribution)): continue
        sha = digest(path)
        if not any(p.name == path.name and digest(p) == sha for p in artifact_files): raise RuntimeError("Device payload differs from exported artifacts")
        row = {"role": "solver_device_payload", "basename": path.name, "stage_relative_path": str(path.relative_to(distribution)), "sha256": sha, "bytes": path.stat().st_size}
        files.append(row); payloads.append(row)
    save("solver-sdk-identity.json", {"source_sha": BASE, "rocm_libraries_gitlink": LIBS,
        "effective_stock_libraries_head": git("-C", "rocm-libraries", "rev-parse", "HEAD"),
        "mode": os.environ["PERF_MODE"], "joint_patch_sha256": digest(HERE / "rocsolver-specialized-shards.patch") if os.environ["PERF_MODE"] == "joint" else None,
        "producer_run_id": os.environ["GITHUB_RUN_ID"], "instrumentation_sha": os.environ["GITHUB_SHA"],
        "distribution_root": str(distribution), "files": files, "solver_device_payloads": payloads})


def toolchain_identity():
    cache = Path("build/math-libs/BLAS/rocSOLVER/build/CMakeCache.txt").read_text()
    match = re.search(r"^CMAKE_CXX_COMPILER:[^=]+=(.+)$", cache, re.M)
    if match: compiler = Path(match.group(1))
    else:
        database = json.loads(Path("build/math-libs/BLAS/rocSOLVER/build/compile_commands.json").read_text())
        compiler = Path(next(t for t in shlex.split(database[0]["command"]) if Path(t).name in ("clang++", "clang", "amdclang++")))
    started = time.monotonic(); tools = []
    for path in (compiler, compiler.with_name("ld.lld"), compiler.with_name("clang-linker-wrapper")):
        if not path.is_file(): raise RuntimeError("Missing actual math toolchain file " + str(path))
        env = os.environ.copy()
        for key in ("ROCM_PATH", "ROCM_DIR", "HIP_PATH", "HIP_DIR"): env.pop(key, None)
        version = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10, env=env)
        tools.append({"path": str(path), "resolved_path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size, "version_exit": version.returncode, "version_stdout": version.stdout, "version_stderr": version.stderr})
    save("math-toolchain-identity.json", {"tools": tools, "wall_s_outside_stage_build": time.monotonic() - started,
        "limits": "Compare actual tool bytes and versions across modes; source/argv equality is insufficient."})


def solver_tests():
    guard()
    path = Path("build/bin/rocsolver/Testing/Temporary/LastTest.log")
    if not path.exists(): raise RuntimeError("Missing rocSOLVER CTest execution log")
    counts = [int(x) for x in re.findall(r"\[=+\]\s+(\d+)\s+tests?\s+from", path.read_text(errors="replace"))]
    save("solver-executed-tests.json", {"path": str(path), "observed_gtest_counts": counts,
        "limits": "Positive observed counts only; not a proof of every new dimension/type/pivot/batch case."})
    if not counts or not any(counts): raise RuntimeError("No positive executed rocSOLVER GTest count")


def focused_component():
    guard()
    rows = [row for row in json.loads(os.environ["PERF_TEST_COMPONENTS"]) if row.get("job_name") == "rocsolver"]
    if len(rows) != 1: raise RuntimeError("Focused numerical route requires one real configured solver component")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.write("component=" + json.dumps(rows[0], separators=(",", ":")) + "\n")


def package_version():
    guard()
    sys.path.insert(0, str(Path("build_tools").resolve()))
    import compute_rocm_package_version as versions
    versions.get_current_date = lambda: "20261031"
    original_msi = versions.compute_msi_build_field
    versions.compute_msi_build_field = lambda: original_msi(datetime.date(2026, 10, 31))
    values = {}
    for package in ("wheel", "deb", "rpm", "msi"):
        key = "rocm_package_version" if package == "wheel" else "rocm_" + package + "_package_version"
        values[key] = versions.compute_version(package_type=package, release_type="ci", build_variant="asan-debug", override_git_sha=BASE)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for key, value in values.items(): stream.write(key + "=" + value + "\n")
    save("matched-package-versions.json", {"frozen_date": "20261031", "source": "stock compute_rocm_package_version.py functions with fixed date and source SHA", "versions": values})


def fetch_identity():
    mode = guard(); run = os.environ["GITHUB_RUN_ID"]
    family = os.environ["AMDGPU_FAMILIES"]
    name = "full-solver-identity-" + mode + "-" + family
    headers = {"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"], "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            redirected = super().redirect_request(req, fp, code, msg, hdrs, newurl)
            if redirected is not None: redirected.remove_header("Authorization")
            return redirected
    opener = urllib.request.build_opener(SafeRedirect())
    def fetch(url):
        with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response: return response.read(4 * 1024 * 1024 + 1)
    found = []
    for page in range(1, 101):
        data = json.loads(fetch(f"https://api.github.com/repos/ROCm/TheRock/actions/runs/{run}/artifacts?per_page=100&page={page}"))
        found.extend(row for row in data["artifacts"] if row["name"] == name and not row["expired"])
        if len(data["artifacts"]) < 100: break
    if len(found) != 1: raise RuntimeError("Missing/ambiguous producer SDK identity artifact")
    row = found[0]
    if row.get("workflow_run", {}).get("id") not in (None, int(run)): raise RuntimeError("Producer run mismatch")
    raw = fetch(f"https://api.github.com/repos/ROCm/TheRock/actions/artifacts/{row['id']}/zip")
    if len(raw) > 4 * 1024 * 1024: raise RuntimeError("Unexpectedly large SDK identity sidecar")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [p for p in archive.infolist() if p.filename == "solver-sdk-identity.json"]
        if len(members) != 1 or members[0].file_size > 1024 * 1024: raise RuntimeError("Invalid SDK identity archive")
        manifest = archive.read(members[0])
    identity = json.loads(manifest)
    expected = {"source_sha": BASE, "mode": mode, "producer_run_id": run, "instrumentation_sha": os.environ["GITHUB_SHA"]}
    if any(str(identity.get(k)) != str(v) for k, v in expected.items()): raise RuntimeError("Producer identity differs from this full run")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "solver-sdk-identity.json").write_bytes(manifest)
    save("producer-sidecar-transport.json", {"utc": utc(), "artifact_id": row["id"], "artifact_name": name, "archive_sha256": hashlib.sha256(raw).hexdigest(), "manifest_sha256": hashlib.sha256(manifest).hexdigest(), "producer_run_id": run})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "inner-limits", "guard-ci", "prepare-stage", "stage-build", "assert-build-matrix", "assert-test-matrix", "verify-solver-tests", "focused-component", "fetch-identity", "package-version", "finalize"))
    p.add_argument("--label", default="job")
    a = p.parse_args()
    if a.action == "preflight": preflight(a.label)
    elif a.action == "inner-limits":
        guard(); save("inner-" + a.label + ".json", {"utc": utc(), "source_sha": BASE, "mode": os.environ["PERF_MODE"], "resources": limits()})
    elif a.action == "guard-ci": guard_ci(a.label)
    elif a.action == "prepare-stage": prepare_stage()
    elif a.action == "stage-build": return stage_build()
    elif a.action == "verify-solver-tests": solver_tests()
    elif a.action == "focused-component": focused_component()
    elif a.action == "fetch-identity": fetch_identity()
    elif a.action == "package-version": package_version()
    elif a.action.startswith("assert-"): matrix("build" if a.action == "assert-build-matrix" else "test")
    else: finalize()
    return 0


if __name__ == "__main__": raise SystemExit(main())
