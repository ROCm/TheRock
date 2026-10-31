# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One default-off, fixed-input TheRock profiler readiness diagnostic."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

try:
    import resource
except ImportError:
    resource = None

from normal_ci_input_identity import file_identity, sdk_inputs, visible_cgroups
from rocprofiler_attachment_observer import TargetObserver, PROVIDERS
from rocprofiler_attachment_sink import LoopbackSink
from rocprofiler_attachment_entry import validate_dashboard

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "build_tools/diagnostics/rocprofiler_attachment_contract.json"
HOOK = ROOT / "build_tools/diagnostics/rocprofiler_attachment_hook.sh"
TEST_SCRIPT = "python build_tools/github_actions/test_executable_scripts/test_rocprofiler_sdk.py --enable-cdash"
TEST_ROOT = Path("build/share/rocprofiler-sdk/tests")
APP_PATHS = [
    TEST_ROOT / "build/bin/attachment-test/attachment-test",
    TEST_ROOT / "build/bin/roctx-attach-pause-resume/roctx-attach-pause-resume",
]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_guard(inputs: dict, env: dict) -> dict:
    contract = json.loads(CONTRACT.read_text())
    sha = env.get("GITHUB_SHA", "")
    expected = {
        "GITHUB_REPOSITORY": "ROCm/TheRock",
        "GITHUB_ACTOR": "sa-faizal",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/" + contract["branch"],
        "GITHUB_WORKFLOW_REF": "ROCm/TheRock/.github/workflows/test_artifacts.yml@refs/heads/"
        + contract["branch"],
    }
    for key, value in expected.items():
        if env.get(key) != value:
            raise ValueError("diagnostic context mismatch: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or git("rev-parse", "HEAD") != sha:
        raise ValueError("checkout and current-run source SHA do not match")
    if inputs.get("ref") != sha or inputs.get("repository") != "ROCm/TheRock":
        raise ValueError("explicit checkout repository/ref mismatch")
    run_id = str(env.get("GITHUB_RUN_ID", ""))
    if (
        not run_id.isdecimal()
        or int(run_id) <= 0
        or run_id == contract["artifact_run_id"]
    ):
        raise ValueError("diagnostic must not impersonate the artifact-producing run")
    if (
        git("ls-tree", "HEAD", "rocm-systems").split()[2]
        != contract["source_rocm_systems_sha"]
    ):
        raise ValueError("rocm-systems source Gitlink changed")
    return dict(
        contract=contract,
        source_sha=sha,
        ci={
            key: env.get(key)
            for key in [
                *expected,
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "RUNNER_NAME",
                "RUNNER_OS",
                "RUNNER_ARCH",
            ]
        },
    )


def validate_component(selected: dict, expected: dict) -> None:
    for name in [
        "job_name",
        "test_script",
        "fetch_artifact_args",
        "timeout_minutes",
        "container_image",
        "container_options",
        "test_type",
        "total_shards",
        "shard_arr",
        "test_runner",
        "additional_requirements_files",
        "platform",
    ]:
        if selected.get(name) != expected[name]:
            raise ValueError("component input changed: " + name)
    for name, default in [
        ("expect_failure", False),
        ("rocm_kpack_debug", "1"),
        ("gpu_enable_pal", "0"),
    ]:
        if selected.get(name, default) != expected.get(name, default):
            raise ValueError("component failure/environment policy changed: " + name)


def guard(inputs: dict, env: dict, component: bool = False) -> dict:
    state = source_guard(inputs, env)
    contract = state["contract"]
    fixed = dict(
        profile_attachment_diagnostic=True,
        artifact_run_id=contract["artifact_run_id"],
        artifact_group="multi-arch-release",
        amdgpu_families="gfx94X-dcgpu",
        amdgpu_targets="gfx942",
        test_runs_on=contract["runner"],
        release_type="ci",
        build_variant="release",
        quartz_tracking_id="",
    )
    if component:
        fixed["platform"] = "linux"
        selected = json.loads(inputs["component"])
        validate_component(selected, contract["component"])
        if env.get("NORMAL_IMAGE_PIN") != selected["container_image"]:
            raise ValueError("effective component image is not pinned")
        if selected["test_script"] != TEST_SCRIPT:
            raise ValueError("test command changed")
    else:
        fixed.update(
            test_runs_on_cpu="",
            tests_enabled=True,
            sanity_check_only_for_family=False,
            test_type="comprehensive",
            changed_projects="",
            skip_stages="",
        )
        if json.loads(inputs.get("test_labels", "null")) != ["test:rocprofiler-sdk"]:
            raise ValueError("diagnostic selection is not profiler-only")
    for name, value in fixed.items():
        # GitHub omits empty strings from toJSON(inputs) on workflow_dispatch.
        # Explicit null, nonempty replacements and missing nonempty values fail.
        actual = inputs.get(name, "" if value == "" else None)
        if (isinstance(value, bool) and actual is not value) or actual != value:
            raise ValueError("fixed diagnostic input mismatch: " + name)
    state["validated_inputs"] = inputs
    return state


def guard_matrix(inputs: dict, env: dict, components: list, sanity: dict) -> dict:
    state = guard(inputs, env)
    if len(components) != 1:
        raise ValueError("diagnostic must select exactly one profiler component")
    validate_component(components[0], state["contract"]["component"])
    validate_component(sanity, state["contract"]["sanity_component"])
    state["matrix_validated_before_container_scheduling"] = True
    state["selected_component_digests"] = [
        hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        for value in [components[0], sanity]
    ]
    return state


class Fingerprints:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.cache, self.records = {}, []
        self.bytes = 0

    def __call__(self, path: Path) -> str:
        path = path.resolve(strict=True)
        allowed = [
            self.workspace / "build",
            Path("/usr/lib"),
            Path("/usr/local/lib"),
            Path("/lib"),
            Path("/lib64"),
            Path("/opt/rocm"),
        ]
        if not any(path.is_relative_to(root.resolve()) for root in allowed):
            raise ValueError("fingerprint path outside SDK/system-provider allowlist")
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if key in self.cache:
            return self.cache[key]
        if before.st_size > 4 * 1024**3 or self.bytes + before.st_size > 16 * 1024**3:
            raise ValueError("unique fingerprint byte budget exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("input changed while fingerprinting")
        self.bytes += before.st_size
        self.cache[key] = digest.hexdigest()
        self.records.append(
            dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest())
        )
        return digest.hexdigest()


def sdk_snapshot(workspace: Path, output: Path, contract: dict) -> dict:
    start = time.monotonic()
    fingerprints = Fingerprints(workspace)
    data = sdk_inputs(workspace / "build", fingerprints)
    data["profiler_providers"] = []
    for relative in ["lib", "lib64"]:
        base = workspace / "build" / relative
        if not base.exists():
            continue
        for directory, subdirs, names in os.walk(base, followlinks=False):
            if len(Path(directory).relative_to(base).parts) >= 4:
                subdirs[:] = []
            for name in names:
                if PROVIDERS.fullmatch(name):
                    data["profiler_providers"].append(
                        file_identity(Path(directory) / name, fingerprints)
                    )
    data["installed_scripts"] = []
    data["installed_target_sources"] = []
    for row in contract["source_script_records"]:
        target_source = row["path"] in {
            "tests/bin/attachment-test/attachment_test.cpp",
            "tests/bin/roctx-attach-pause-resume/roctx-attach-pause-resume.cpp",
        }
        if not row["path"].endswith(".sh") and not target_source:
            continue
        path = workspace / TEST_ROOT / Path(row["path"]).relative_to("tests")
        identity = file_identity(path, fingerprints)
        identity["expected_source_sha256"] = row["sha256"]
        identity["matches_pinned_source"] = identity.get("sha256") == row["sha256"]
        data[
            "installed_target_sources" if target_source else "installed_scripts"
        ].append(identity)
    entry = file_identity(workspace / "build/bin/rocprofv3", fingerprints)
    data["rocprofv3_entry"] = entry
    data["required_test_compilers"] = [
        file_identity(workspace / "build/llvm/bin" / name, fingerprints)
        for name in ["amdclang", "amdclang++"]
    ]
    for row in [
        *data["files"],
        *data["profiler_providers"],
        *data["installed_scripts"],
        *data["installed_target_sources"],
        *data["required_test_compilers"],
        entry,
    ]:
        path = Path(row["path"])
        try:
            current = path.lstat()
            row["entry_mode"] = stat.S_IMODE(current.st_mode)
            row["resolved_mode"] = stat.S_IMODE(path.stat().st_mode)
            row["entry_kind"] = (
                "symlink"
                if stat.S_ISLNK(current.st_mode)
                else "file" if stat.S_ISREG(current.st_mode) else "other"
            )
            if path.is_symlink():
                row["link_target"] = os.readlink(path)
        except OSError as error:
            row.update(status="unavailable", error=str(error))
            data["errors"].append(dict(path=str(path), error=str(error)))
    required = [
        "libamdhip64.so",
        "libhsa-runtime64.so",
        "librocprofiler-sdk.so",
        "librocprofiler-sdk-attach.so",
        "librocprofiler-register.so",
    ]
    data["required_provider_presence"] = {
        name: [
            row["path"]
            for row in data["profiler_providers"]
            if Path(row["path"]).name.startswith(name) and row.get("status") == "file"
        ]
        for name in required
    }
    data["unique_fingerprint_bytes"] = fingerprints.bytes
    data["wall_s"] = time.monotonic() - start
    data["complete"] = bool(
        data["complete"]
        and not data["errors"]
        and all(data["required_provider_presence"].values())
        and all(row.get("status") == "file" for row in data["profiler_providers"])
        and entry.get("status") == "file"
        and all(row.get("status") == "file" for row in data["required_test_compilers"])
        and data["installed_scripts"]
        and all(row["matches_pinned_source"] for row in data["installed_scripts"])
        and all(
            row["matches_pinned_source"] for row in data["installed_target_sources"]
        )
    )
    write_json(output, data)
    return data


def usage(kind) -> float:
    value = resource.getrusage(kind)
    return value.ru_utime + value.ru_stime


def selected_input_bindings(data: dict) -> dict:
    rows = [data["rocprofv3_entry"]]
    for key in [
        "files",
        "profiler_providers",
        "installed_scripts",
        "installed_target_sources",
        "required_test_compilers",
    ]:
        rows.extend(data[key])
    return {
        row["path"]: {
            name: row.get(name)
            for name in [
                "sha256",
                "status",
                "resolved_path",
                "entry_kind",
                "entry_mode",
                "resolved_mode",
                "link_target",
                "bytes",
            ]
        }
        for row in rows
    }


def loopback_bypass(env: dict) -> dict:
    result = env.copy()
    for name in ["no_proxy", "NO_PROXY"]:
        inherited = result.get(name, "")
        result[name] = inherited + ("," if inherited else "") + "127.0.0.1,localhost"
    return result


def run(inputs: dict, env: dict) -> int:
    state = guard(inputs, env, component=True)
    if (
        sys.platform != "linux"
        or resource is None
        or env.get("BASH_ENV")
        or "BASH_FUNC_grep%%" in env
    ):
        raise ValueError(
            "probe requires Linux and an unmodified noninteractive Bash environment"
        )
    workspace = Path(env["GITHUB_WORKSPACE"]).resolve()
    if workspace != ROOT:
        raise ValueError("unexpected checkout workspace")
    output = (
        workspace
        / "attachment-diagnostic"
        / f"execution-{os.getpid()}-{time.monotonic_ns()}"
    )
    output.mkdir(parents=True, mode=0o700)
    (output / "poll-errors").mkdir(mode=0o700)
    began, own_cpu = time.monotonic(), usage(resource.RUSAGE_SELF)
    summary = dict(
        state=state,
        original_test_command=TEST_SCRIPT.split(),
        test_returncode=None,
        diagnostic_complete=False,
        scope="Read-only readiness diagnostic, not a speedup experiment",
        child_scope="Unchanged installed-test driver including configure/build/CTest; not compilation-only or CTest-only CPU",
    )
    write_json(output / "summary.json", summary)
    child, sink, observer = None, None, None
    try:
        before = sdk_snapshot(workspace, output / "sdk-before.json", state["contract"])
        if not before["complete"]:
            raise ValueError(
                "selected installed SDK/script input capture is incomplete"
            )
        dependencies = []
        for relative in state["contract"]["component"]["additional_requirements_files"]:
            path = workspace / relative
            dependencies.append(
                dict(
                    path=relative,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    bytes=path.stat().st_size,
                )
            )
        write_json(
            output / "test-dependencies.json",
            dict(
                requirements=dependencies,
                installed_distributions=sorted(
                    [
                        dict(name=dist.metadata["Name"], version=dist.version)
                        for dist in importlib.metadata.distributions()
                    ],
                    key=lambda row: (row["name"] or "", row["version"]),
                ),
                scope="Package names/versions and exact declared requirements files only; no direct URLs or environment dump",
            ),
        )
        visible, scope = visible_cgroups()
        limits = []
        for directory in visible:
            files = {}
            for name in ["cpu.max", "cpuset.cpus.effective", "cpu.stat", "memory.max"]:
                try:
                    files[name] = (directory / name).read_text().strip()
                except OSError as error:
                    files[name] = {"error": str(error)}
            limits.append(dict(path=str(directory), files=files))
        write_json(
            output / "allocation.json",
            dict(
                logical_cpus=os.cpu_count(),
                clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
                affinity=sorted(os.sched_getaffinity(0)),
                cgroup_resolution=scope,
                visible_limits=limits,
                scope="Visible bounds only; hidden ancestors and dedicated CPU entitlement are unknown",
            ),
        )
        hook = output / "grep-hook.sh"
        hook.write_bytes(HOOK.read_bytes())
        sink = LoopbackSink(output / "local-cdash")
        sink.start()
        dashboard = workspace / TEST_ROOT / "build/dashboard.cmake"
        if dashboard.exists():
            raise ValueError("stale dashboard is not current-run evidence")
        child_env = loopback_bypass(env)
        summary["loopback_proxy_bypass"] = dict(
            appended_hosts=["127.0.0.1", "localhost"],
            inherited_values_preserved=True,
            inherited_values_recorded=False,
        )
        child_env.update(
            BASH_ENV=str(hook),
            ATTACHMENT_DIAG_ROOT=str(output),
            THEROCK_CDASH_SUBMIT_URL=sink.url,
        )
        summary["controlled_environment_changes"] = {
            name: child_env[name]
            for name in ["BASH_ENV", "ATTACHMENT_DIAG_ROOT", "THEROCK_CDASH_SUBMIT_URL"]
        }
        summary["preparation_wall_s"] = time.monotonic() - began
        summary["preparation_own_cpu_s"] = usage(resource.RUSAGE_SELF) - own_cpu
        summary["hook_sha256"] = hashlib.sha256(hook.read_bytes()).hexdigest()
        child_start, child_cpu = time.monotonic(), usage(resource.RUSAGE_CHILDREN)
        observation_cpu = usage(resource.RUSAGE_SELF)
        actual_bootstrap = [
            sys.executable,
            str(ROOT / "build_tools/rocprofiler_attachment_entry.py"),
        ]
        summary["actual_bootstrap_argv"] = actual_bootstrap
        child = subprocess.Popen(actual_bootstrap, cwd=workspace, env=child_env)
        observer = TargetObserver(
            output,
            child.pid,
            {workspace / path for path in APP_PATHS},
            Fingerprints(workspace),
            marker_root=workspace / TEST_ROOT / "build",
        )
        interrupted = []

        def forward(signum, frame):
            interrupted.append(signum)
            if child.poll() is None:
                child.send_signal(signum)

        previous = {
            name: signal.signal(name, forward)
            for name in [signal.SIGTERM, signal.SIGINT]
        }
        observer_failed = False
        observer_active_wall = 0.0
        try:
            while child.poll() is None:
                if not observer_failed:
                    observe_start = time.monotonic()
                    try:
                        observer.consume()
                        observer.snapshot()
                    except (OSError, ValueError) as error:
                        observer.errors.append({"fatal_observation_error": str(error)})
                        observer_failed = True
                    finally:
                        observer_active_wall += time.monotonic() - observe_start
                time.sleep(0.5)
            result = child.wait()
            if not observer_failed:
                observe_start = time.monotonic()
                try:
                    observer.consume()
                    observer.snapshot()
                except (OSError, ValueError) as error:
                    observer.errors.append({"final_observation_error": str(error)})
                finally:
                    observer_active_wall += time.monotonic() - observe_start
        finally:
            for name, handler in previous.items():
                signal.signal(name, handler)
        summary.update(
            test_returncode=result,
            test_wall_s=time.monotonic() - child_start,
            exact_posix_waited_child_cpu_s=usage(resource.RUSAGE_CHILDREN) - child_cpu,
            observer_and_sink_own_cpu_s=usage(resource.RUSAGE_SELF) - observation_cpu,
            observer_active_wall_s=observer_active_wall,
            configured_poll_sleep_s=0.5,
            interrupted_signals=interrupted,
            overhead_scope="Own CPU includes observer, hashing and loopback sink. Child CPU includes the grep hook. No A/B total-perturbation estimate is claimed.",
        )
        summary["observer"] = observer.summary()
        entry = output / "ctest-entry.json"
        summary["generated_loopback_dashboard"] = (
            json.loads(entry.read_text()) if entry.exists() else None
        )
        summary["local_cdash"] = sink.close()
        sink = None
        after = sdk_snapshot(workspace, output / "sdk-after.json", state["contract"])
        summary["selected_sdk_and_original_scripts_unchanged"] = (
            selected_input_bindings(before) == selected_input_bindings(after)
        )
        evidence = output / "generated-test-metadata"
        build = workspace / TEST_ROOT / "build"
        for path in build.rglob("*"):
            if path.is_file() and path.name in {
                "CTestTestfile.cmake",
                "CMakeCache.txt",
                "build.ninja",
                "dashboard.cmake",
                "dashboard-notes.txt",
            }:
                target = evidence / path.relative_to(build)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        summary["diagnostic_complete"] = bool(
            summary["observer"]["complete"]
            and summary.get("generated_loopback_dashboard")
            and summary["local_cdash"]["complete"]
            and after["complete"]
            and summary["selected_sdk_and_original_scripts_unchanged"]
        )
        return result if result >= 0 else 128 - result
    except Exception as error:
        summary["collector_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if child is not None:
            result = child.wait()
            summary["test_returncode"] = result
            return result if result >= 0 else 128 - result
        raise
    finally:
        if sink is not None:
            summary["local_cdash"] = sink.close()
        summary["total_controller_wall_s"] = time.monotonic() - began
        summary["total_controller_own_cpu_s"] = usage(resource.RUSAGE_SELF) - own_cpu
        write_json(output / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["guard", "matrix", "run", "check"])
    parser.add_argument("--inputs-json")
    parser.add_argument("--components-json")
    parser.add_argument("--sanity-json")
    args = parser.parse_args()
    if args.mode == "check":
        summaries = sorted(
            (ROOT / "attachment-diagnostic").glob("execution-*/summary.json")
        )
        if (
            list((ROOT / "attachment-diagnostic").glob("controller-error-*.json"))
            or not summaries
            or not all(
                json.loads(path.read_text())["diagnostic_complete"]
                for path in summaries
            )
        ):
            raise ValueError(
                "diagnostic evidence is absent or incomplete; original test result is separate"
            )
        return 0
    try:
        inputs = json.loads(args.inputs_json)
        if args.mode == "guard":
            write_json(
                ROOT / "attachment-matrix-contract.json",
                guard(inputs, dict(os.environ)),
            )
            return 0
        if args.mode == "matrix":
            write_json(
                ROOT / "attachment-matrix-contract.json",
                guard_matrix(
                    inputs,
                    dict(os.environ),
                    json.loads(args.components_json),
                    json.loads(args.sanity_json),
                ),
            )
            return 0
        return run(inputs, dict(os.environ))
    except Exception as error:
        target = (
            ROOT / "attachment-matrix-contract.json"
            if args.mode in {"guard", "matrix"}
            else ROOT
            / "attachment-diagnostic"
            / f"controller-error-{os.getpid()}-{time.monotonic_ns()}.json"
        )
        write_json(
            target,
            dict(
                validated=False,
                diagnostic_complete=False,
                test_acceptance=False,
                error=dict(type=type(error).__name__, message=str(error)),
                ci={
                    name: os.environ.get(name)
                    for name in [
                        "GITHUB_REPOSITORY",
                        "GITHUB_SHA",
                        "GITHUB_RUN_ID",
                        "GITHUB_RUN_ATTEMPT",
                        "GITHUB_REF",
                        "RUNNER_NAME",
                    ]
                },
            ),
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
