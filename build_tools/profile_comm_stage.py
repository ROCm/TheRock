#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fresh complete COMM-stage link-only control, exclusively in TheRock CI.

Parent dispatches independent1/2/2/1 allocations at one exact diagnostic SHA.
Both settings contain the identical common upstream patch, default-one option
and source-version metadata. Never reuse the older full-stage/link-only runs as
the performance baseline for this product candidate.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import comm_stage_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]
BASE = "82d5e8c176ce0cc00b5fd1cf631ce7790be8992b"
PRODUCT = "5b7cb88373362ab80dc0f19605915e534f4158aa"
SYSTEMS = "a10dca087065a82f15ea89dc7d6fdd3c6bb2c246"
LLVM = "e095f1b3becf0c1bfdc5e7647003d60976a2460b"
SDK_RUN = "41189458499"
VERSION = "10.3.0.dev0+" + BASE
PATCH = ROOT / "patches/amd-mainline/rocm-systems/0002-rocshmem-native-link-jobs.patch"


def capture(command: list[str], env: dict, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True)


def git(command: list[str], env: dict, cwd: Path = ROOT) -> str:
    return capture(["git", "-c", f"safe.directory={cwd}", *command], env, cwd).strip()


def require_controls(env: dict, width: int) -> None:
    evidence.require_rock_ci()
    if sys.platform != "linux":
        raise RuntimeError("Full COMM trial requires Linux")
    if width not in [1, 2]:
        raise RuntimeError("Only native link widths1 and2 are reviewed")
    for key, expected in {
        "RELEASE_TYPE": "ci",
        "CCACHE_READONLY": "1",
        "TEATIME_PROFILE_CPU": "1",
    }.items():
        if env.get(key) != expected:
            raise RuntimeError(f"Required common control:{key}={expected}")
    if env.get("HIP_CLANG_NUM_PARALLEL_JOBS") not in [None, "", "1"]:
        raise RuntimeError("Do not change the HIP-wide compilation/link knob")
    if env.get("CCACHE_DISABLE"):
        raise RuntimeError("Compiler cache must retain the enabled read-only CI policy")
    git(["merge-base", "--is-ancestor", BASE, "HEAD"], env)
    for path, pin in [("rocm-systems", SYSTEMS), ("compiler/amd-llvm", LLVM)]:
        if git(["rev-parse", f"HEAD:{path}"], env) != pin:
            raise RuntimeError(f"Source pin drift:{path}")


def prepare(build: Path, output: Path, width: int, env: dict, deadline: float) -> None:
    for name, command in [
        (
            "fetch-sdk",
            [
                sys.executable,
                "build_tools/artifact_manager.py",
                "fetch",
                "--run-id=" + SDK_RUN,
                "--run-github-repo=ROCm/TheRock",
                "--stage=comm-libs",
                "--amdgpu-families=",
                "--output-dir=" + str(build),
                "--bootstrap",
            ],
        ),
        (
            "fetch-sources",
            [
                sys.executable,
                "build_tools/fetch_sources.py",
                "--stage",
                "comm-libs",
                "--jobs",
                "12",
                "--depth",
                "1",
            ],
        ),
    ]:
        evidence.run_logged(command, ROOT, output / (name + ".log"), deadline, env)
    systems = ROOT / "rocm-systems"
    git(["merge-base", "--is-ancestor", SYSTEMS, "HEAD"], env, systems)
    if git(["diff", "--name-only", "HEAD"], env, systems):
        raise RuntimeError(
            "Unexpected dirty rocm-systems source after normal patch application"
        )
    patch_state = (ROOT / ".rocm-systems.smrev").read_text()
    if SYSTEMS + "+PATCHED:" not in patch_state:
        raise RuntimeError("Required normal common-patch source metadata absent")
    fingerprint = hashlib.sha1(usedforsecurity=False)
    for path in sorted(PATCH.parent.glob("*.patch")):
        fingerprint.update(path.read_bytes())
    if f"{SYSTEMS}+PATCHED:{fingerprint.hexdigest()}" not in patch_state:
        raise RuntimeError(
            "Recorded patch-state fingerprint does not match the common source patch series"
        )
    git(["apply", "--reverse", "--check", str(PATCH)], env, systems)
    # Both modes use the same full patch series and source HEAD/dirty state.
    evidence.save(
        output / "source-identity.json",
        dict(
            systems_pin=SYSTEMS,
            systems_patched_head=git(["rev-parse", "HEAD"], env, systems),
            systems_smrev=patch_state,
            llvm_pin=LLVM,
            product_head=PRODUCT,
            diagnostic_head=git(["rev-parse", "HEAD"], env),
            patch_series=[
                dict(path=str(p.relative_to(ROOT)), sha256=evidence.sha256(p, deadline))
                for p in sorted(PATCH.parent.glob("*.patch"))
            ],
            patch_sha256=evidence.sha256(PATCH, deadline),
        ),
    )
    (output / "rocm-systems-common-source.diff").write_text(
        git(["diff", SYSTEMS, "HEAD"], env, systems)
    )
    from configure_stage import get_topology

    dependencies = get_topology().get_python_requires_for_stage("comm-libs")
    if dependencies:
        evidence.run_logged(
            [sys.executable, "-m", "pip", "install", *dependencies],
            ROOT,
            output / "python-deps.log",
            deadline,
            env,
        )
    args = capture(
        [
            sys.executable,
            "build_tools/configure_stage.py",
            "--stage=comm-libs",
            "--amdgpu-families=",
            "--dist-amdgpu-families=gfx94X-dcgpu",
            "--manylinux",
            "--oneline",
        ],
        env,
    )
    if any(
        flag not in args
        for flag in [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_RCCL=ON",
            "-DTHEROCK_ENABLE_ROCSHMEM=ON",
        ]
    ):
        raise RuntimeError("Full COMM feature selection drift")
    (output / "stage-cmake-args.txt").write_text(args)
    command = [
        "cmake",
        "-B",
        str(build),
        "-S",
        str(ROOT),
        "-GNinja",
        "-DTHEROCK_PACKAGE_VERSION=" + VERSION,
        "-DTHEROCK_ROCJITSU_TEST_JOBS=16",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
        f"-DTHEROCK_ROCSHMEM_NATIVE_LINK_JOBS={width}",
        *shlex.split(args),
    ]
    evidence.save(output / "configure-command.json", command)
    evidence.run_logged(command, ROOT, output / "configure.log", deadline, env)
    values = evidence.cache_values(build / "CMakeCache.txt")
    if values.get("THEROCK_PACKAGE_VERSION") != VERSION or values.get(
        "THEROCK_ROCSHMEM_NATIVE_LINK_JOBS"
    ) != str(width):
        raise RuntimeError("Root configuration/version control drift")
    # No nested +configure or object preparation: all seven project phases
    # remain fresh within the measured full-stage build.
    if any(
        (build / relative / ".ninja_log").exists()
        for relative in evidence.PROJECTS.values()
    ):
        raise RuntimeError("Preparation unexpectedly built a measured COMM project")


def retain_logs(build: Path, output: Path) -> None:
    destination = output / "native-logs"
    destination.mkdir(exist_ok=True)
    for phase in ["configure", "build", "install", "build_test"]:
        for path in (build / "logs").glob(f"*_{phase}.log"):
            if path.is_file():
                shutil.copyfile(path, destination / path.name)


def retain_cache_identity(output: Path, env: dict, deadline: float) -> None:
    from setup_ccache import CACHE_SRV_DEV

    config = Path(env["CCACHE_CONFIGPATH"])
    compiler_check = config.parent / "compiler_check.py"
    effective = {
        key: capture(["ccache", "--get-config", key], env).strip()
        for key in [
            "read_only",
            "namespace",
            "compiler_check",
            "sloppiness",
            "base_dir",
            "hash_dir",
            "remote_only",
            "cache_dir",
            "direct_mode",
        ]
    }
    remote_matches = (
        capture(["ccache", "--get-config", "remote_storage"], env).strip()
        == CACHE_SRV_DEV
    )
    evidence.save(
        output / "cache-identity.json",
        dict(
            config_sha256=evidence.sha256(config, deadline),
            compiler_check_sha256=evidence.sha256(compiler_check, deadline),
            compiler_check_source_sha256=evidence.sha256(
                ROOT / "build_tools/posix_ccache_compiler_check.py", deadline
            ),
            effective=effective,
            remote_matches_existing_ci_dev_endpoint=remote_matches,
            scope="Same intentional CI trust domain; remote auth/config dumps are not exported",
        ),
    )
    if (
        effective["read_only"] != "true"
        or effective["namespace"] != "therock-v4"
        or not remote_matches
        or compiler_check.read_bytes()
        != (ROOT / "build_tools/posix_ccache_compiler_check.py").read_bytes()
    ):
        raise RuntimeError("Effective read-only CI cache identity drift")


def retain_cache_log(build: Path, output: Path, deadline: float) -> None:
    source = build / "comm-ccache.log"
    if not source.is_file():
        raise RuntimeError("Missing per-transaction cache log")
    count = 0
    # Retain original timestamp/PID/line text for relevant transactions, without
    # exporting config dumps or remote-auth settings. Raw SHA binds the source.
    needles = [
        "Command line:",
        "Working directory:",
        "Manifest key:",
        "Result key:",
        "Result:",
    ]
    with source.open(errors="replace") as stream, (
        output / "ccache-transactions.log"
    ).open("w") as target:
        for number, line in enumerate(stream, 1):
            if any(needle in line for needle in needles):
                target.write(f"{number}\t" + line)
                count += 1
    evidence.save(
        output / "ccache-log-provenance.json",
        dict(
            source=str(source.relative_to(build)),
            raw_sha256=evidence.sha256(source, deadline),
            retained_lines=count,
            fields=needles,
        ),
    )


def collect_artifacts(build: Path, output: Path, deadline: float) -> None:
    root = build / "artifacts"
    if not root.is_dir():
        raise RuntimeError("Missing complete COMM artifacts")
    import hashlib
    import stat

    hashed = hashlib.sha256()
    count = 0
    with (output / "artifact-files.jsonl").open("w") as stream:
        for path in sorted(root.rglob("*")):
            evidence.check_deadline(deadline)
            info = path.lstat()
            row = dict(
                path=str(path.relative_to(root)), mode=stat.S_IMODE(info.st_mode)
            )
            if path.is_symlink():
                row.update(kind="symlink", target=os.readlink(path))
            elif path.is_file():
                row.update(
                    kind="file",
                    bytes=info.st_size,
                    sha256=evidence.sha256(path, deadline),
                )
            elif path.is_dir():
                row.update(kind="directory")
            else:
                raise RuntimeError("Unexpected artifact file type")
            encoded = json.dumps(row, sort_keys=True) + "\n"
            stream.write(encoded)
            hashed.update(encoded.encode())
            count += 1
    evidence.save(
        output / "artifacts-summary.json",
        dict(
            entries=count,
            content_mode_link_manifest_sha256=hashed.hexdigest(),
            scope="Artifact file bytes/modes/links, not archive mtimes or GPU execution",
        ),
    )


def packaging_tests(build: Path, output: Path, env: dict, deadline: float) -> None:
    inventory = capture(["ctest", "--test-dir", str(build), "--show-only=json-v1"], env)
    (output / "packaging-ctest-inventory.json").write_text(inventory)
    if not json.loads(inventory).get("tests"):
        raise RuntimeError("No root packaging checks were registered")
    evidence.run_logged(
        ["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"],
        ROOT,
        output / "packaging-ctest.log",
        deadline,
        env,
    )


def postchecks(
    build: Path, output: Path, width: int, env: dict, deadline: float
) -> list[dict]:
    checks = [
        ("phase_cpu", lambda: evidence.collect_phase_cpu(build, output)),
        (
            "graphs_compile_headers_objects",
            lambda: evidence.collect_graphs(build, output, width, deadline),
        ),
        (
            "rocshmem_fixed_link_inputs",
            lambda: evidence.retain_link_inputs(build, output, deadline),
        ),
        (
            "rocshmem_native_images",
            lambda: evidence.save(
                output / "rocshmem-native.json",
                evidence.retain_outputs(
                    build / evidence.PROJECTS["rocshmem"], output, deadline
                ),
            ),
        ),
        ("rccl_native_images", lambda: evidence.retain_rccl(build, output, deadline)),
        ("artifact_contents", lambda: collect_artifacts(build, output, deadline)),
        ("packaging_ctest", lambda: packaging_tests(build, output, env, deadline)),
    ]
    results = []
    for name, check in checks:
        started = time.monotonic()
        try:
            check()
            row = dict(name=name, status="passed")
        except Exception as error:
            row = dict(name=name, status="failed", error=repr(error))
        row["collector_wall_s"] = time.monotonic() - started
        results.append(row)
        evidence.save(output / "postchecks.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-jobs", type=int, choices=[1, 2], required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=10800)
    args = parser.parse_args()
    env = dict(os.environ)
    require_controls(env, args.link_jobs)
    build = ROOT / "build"
    if build.exists() or build.is_symlink():
        raise RuntimeError(
            "Full-stage trial requires a fresh build directory; no reuse/deletion"
        )
    output = build / "logs/comm-stage"
    output.mkdir(parents=True)
    deadline = time.monotonic() + args.max_wall_seconds
    manifest = dict(
        mode=args.link_jobs,
        source_sha=os.environ["GITHUB_SHA"],
        product_head=PRODUCT,
        reference_head=BASE,
        sdk_run=SDK_RUN,
        sdk_repository="ROCm/TheRock",
        version=VERSION,
        runner_label="aws-linux-scale-rocm-prod",
        runner_name=env.get("RUNNER_NAME"),
        run_id=env.get("GITHUB_RUN_ID"),
        container_digest="sha256:d666e4613c1ca919bc7d1a31a4c368740b66c34b5c19ef95ba15cd31db7783fb",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cache_controls={
            k: env.get(k)
            for k in [
                "CCACHE_READONLY",
                "CCACHE_CONFIGPATH",
                "HIP_CLANG_NUM_PARALLEL_JOBS",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "OMP_NUM_THREADS",
                "CMAKE_BUILD_PARALLEL_LEVEL",
                "MAKEFLAGS",
                "NINJAFLAGS",
            ]
        },
        scope="Fresh full COMM stage with actual RCCL competition and dependency readiness; only two native link flags differ; setup/verification separate",
        previous_runs="Older full baseline and isolated link trial are contextual evidence, not controls for this common source patch",
        correctness="GPU runtime and performance unvalidated until controlled outputs/checks and repetitions agree",
    )
    evidence.save(output / "manifest.json", manifest)
    evidence.save(
        output / "allocation-hardware.json",
        dict(
            uname=list(os.uname()),
            cpuinfo=Path("/proc/cpuinfo").read_text(),
            scope="Host-visible model/topology; effective allocation and cgroups are separate profile metadata",
        ),
    )
    n = int(env.get("GIT_CONFIG_COUNT", "0"))
    for index, path in enumerate([ROOT, ROOT / "rocm-systems"]):
        env[f"GIT_CONFIG_KEY_{n+index}"] = "safe.directory"
        env[f"GIT_CONFIG_VALUE_{n+index}"] = str(path)
    env["GIT_CONFIG_COUNT"] = str(n + 2)
    env["CCACHE_LOGFILE"] = str(build / "comm-ccache.log")
    failures = []
    try:
        prepare(build, output, args.link_jobs, env, deadline)
        retain_cache_identity(output, env, deadline)
        (output / "ccache-before.txt").write_text(
            capture(["ccache", "--show-stats"], env)
        )
        evidence.save(
            output / "python-packages.json",
            sorted(
                [
                    dict(name=d.metadata.get("Name", "unknown"), version=d.version)
                    for d in importlib.metadata.distributions()
                ],
                key=lambda row: (row["name"], row["version"]),
            ),
        )
        evidence.save(
            output / "tool-versions.json",
            dict(
                python=sys.version,
                tools={
                    name: capture([name, "--version"], env).splitlines()[0]
                    for name in ["cmake", "ninja", "ccache"]
                },
            ),
        )
        child_env = {
            k: v
            for k, v in env.items()
            if k
            not in [
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            ]
        }
        started = time.monotonic()
        try:
            evidence.run_logged(
                [
                    sys.executable,
                    "build_tools/profile_build.py",
                    "--output",
                    str(output / "profile"),
                    "--",
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "stage-comm-libs",
                    "therock-artifacts",
                    "--",
                    "-k",
                    "0",
                ],
                ROOT,
                output / "stage-build.log",
                deadline,
                child_env,
            )
            manifest["build_succeeded"] = True
        except Exception as error:
            failures.append(dict(scope="full_stage_build", error=repr(error)))
            manifest["build_succeeded"] = False
        manifest["full_stage_command_wall_s"] = time.monotonic() - started
        results = postchecks(build, output, args.link_jobs, child_env, deadline)
        failures.extend(row for row in results if row["status"] != "passed")
    except Exception as error:
        failures.append(dict(scope="preparation_or_controller", error=repr(error)))
    finally:
        for name, check in [
            ("native_logs", lambda: retain_logs(build, output)),
            (
                "ccache_after",
                lambda: (output / "ccache-after.txt").write_text(
                    capture(["ccache", "--show-stats"], env)
                ),
            ),
            ("cache_transactions", lambda: retain_cache_log(build, output, deadline)),
        ]:
            try:
                check()
            except Exception as error:
                failures.append(dict(scope=name, error=repr(error)))
        manifest.update(
            failures=failures,
            complete=not failures,
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        evidence.save(output / "manifest.json", manifest)
    if failures:
        raise RuntimeError("COMM-stage trial failed; independent evidence retained")


if __name__ == "__main__":
    main()
