#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""TheRock-only full DEB AB/BA diagnostic; never compile or publish.

Separate actions let CI retain each complete repository before exact owned
removal. Reset/hash/check/repo work stays outside the package-build timer.
"""

import argparse
from dataclasses import replace
import gzip
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import benchmark_package_preprocess as pre
import deb_validation_support as support
import package_preprocess_snapshot as snapshots

SCRIPT = Path(__file__).resolve()
FAMILIES = "gfx94X-dcgpu;gfx110X-all;gfx1151;gfx120X-all;gfx125X-dcgpu;gfx90a;gfx950-dcgpu;gfx900;gfx90c;gfx906;gfx908;gfx101X-dgpu;gfx103X-all;gfx1150;gfx1152;gfx1153;gfx117X-all"
EXPECTED_CONTENT = "c6b6b5a996686d55eb2a5fbd505a21b592e962f31f48340ed4c120ea25629ad1"
GIB = 1024**3


def owner() -> dict:
    return {
        key: os.environ[key]
        for key in [
            "GITHUB_REPOSITORY",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_SHA",
            "GITHUB_WORKSPACE",
        ]
    }


def paths() -> tuple[Path, Path, Path, Path]:
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    return (
        workspace / "output/artifacts",
        workspace / "output/preprocess-work",
        workspace / "output/scan-diagnostics/deb",
        workspace / "output/deb-case",
    )


def read_state() -> dict:
    root, work, output, case = paths()
    state = json.loads((output / "state.json").read_text())
    if state["owner"] != owner():
        raise RuntimeError(
            "Full-DEB state belongs to a different run/SHA/attempt/workspace"
        )
    if state["paths"] != list(map(str, [root, work, output, case])):
        raise RuntimeError("Full-DEB state paths changed")
    if tuple(state["output_inode"]) != snapshots.SnapshotReset.inode(output):
        raise RuntimeError("Diagnostic output inode changed")
    if state["helper_sha256"] != support.file_hash(SCRIPT):
        raise RuntimeError("Diagnostic controller changed between steps")
    return state


def save_state(state: dict) -> None:
    pre.dump(paths()[2] / "state.json", state)


def load_snapshot(state: dict):
    root, work, output, _ = paths()
    data = state["snapshot"]
    result = snapshots.SnapshotReset.__new__(snapshots.SnapshotReset)
    result.root, result.work, result.output = root, work, output / "snapshot"
    result.archive = root.parent / "preprocess-pristine.tar.zst"
    result.initial, result.deadline = state["initial"], state["deadline"]
    result.records = {
        r["path"]: r
        for r in map(
            json.loads, (output / "pristine-before.jsonl").read_text().splitlines()
        )
    }
    result.owned = {Path(path): tuple(inode) for path, inode in data["owned"].items()}
    if any(path not in [root, work] for path in result.owned):
        raise RuntimeError("Unexpected owned snapshot tree")
    result.original_removed = data["original_removed"]
    result.archive_sha256, result.operations = (
        data["archive_sha256"],
        data["operations"],
    )
    result.floor = (
        max(5 * GIB, state["initial"]["counts"]["bytes"] // 10)
        + state["initial"]["largest_elf_bytes"]
    )
    if result.floor != state["reserve_bytes"]:
        raise RuntimeError("Snapshot reserve changed")
    return result


def save_snapshot(state: dict, snapshot) -> None:
    state["snapshot"] = dict(
        owned={str(p): list(v) for p, v in snapshot.owned.items()},
        original_removed=snapshot.original_removed,
        archive_sha256=snapshot.archive_sha256,
        operations=snapshot.operations,
    )
    save_state(state)


def remove_owned(path: Path, inode: list[int], state: dict) -> None:
    root, work, _, case = paths()
    pre.check_deadline(state["deadline"])
    if path not in [root, work, case] or path.resolve() != path or path.is_symlink():
        raise RuntimeError("Refuse deletion outside exact owned diagnostic roots")
    if snapshots.SnapshotReset.inode(path) != tuple(inode):
        raise RuntimeError("Owned diagnostic directory inode changed")
    support.floor_check(path.parent, state["reserve_bytes"])
    shutil.rmtree(path)


def config(root: Path, destination: Path):
    import build_package as build

    return build.create_package_config(
        argparse.Namespace(
            dest_dir=destination,
            target=None,
            rocm_version=support.VERSION,
            artifacts_dir=root,
            pkg_type="deb",
            version_suffix=support.SUFFIX,
            build_variant="release",
            install_prefix=build.DEFAULT_INSTALL_PREFIX,
        )
    )


def component_plan(root: Path, initial_manifest: Path, output_bytes: int) -> dict:
    """Stock selectors; every variant source tree persists until task exit.

    dh_strip is disabled at pinned HEAD. dh_install normally hardlinks within
    each temp tree. A common fail-fast diagnostic guard rejects its copy
    fallback before duplicate allocation; guard firing invalidates the case.
    """
    import build_package as build
    import deb_package as deb

    if "disable_dh_strip = True" not in inspect.getsource(deb.generate_rules_file):
        raise RuntimeError("Review changed stripping/storage behavior")
    records = list(map(json.loads, initial_manifest.read_text().splitlines()))
    files = {r["path"]: r for r in records if r["kind"] == "file"}
    cfg = config(root, root.parent / "deb-case/packages")
    names, skipped = build.parse_input_package_list(None, root)
    rows = []
    for name in names:
        info = build.get_package_info(name)
        scopes = (
            []
            if build.is_meta_package(info)
            else (
                [build.GFX_HOST, *build.group_package_targets(cfg.gfxarch_list)]
                if build.is_gfxarch_package(info, root)
                else [""]
            )
        )
        variants = []
        for scope in scopes:
            selected = replace(cfg, gfx_arch=scope)
            roots = build.filter_components_fromartifactory(
                name, root, scope, target_members=build.package_target_members(selected)
            )
            size = largest_file = 0
            prefixes = []
            for source in roots:
                if not source.is_dir():
                    continue
                resolved = source.resolve()
                if root not in resolved.parents:
                    raise RuntimeError("Selected package root escapes owned artifacts")
                prefix = resolved.relative_to(root).as_posix().rstrip("/") + "/"
                prefixes.append(prefix)
                for path, record in files.items():
                    if path.startswith(prefix):
                        size += record["size"]
                        largest_file = max(largest_file, record["size"])
            variants.append(
                dict(
                    scope=scope,
                    selected_bytes=size,
                    largest_file_bytes=largest_file,
                    roots=prefixes,
                )
            )
        rows.append(
            dict(
                component=name,
                retained_source_bytes=sum(v["selected_bytes"] for v in variants),
                max_current_variant_bytes=max(
                    [v["selected_bytes"] for v in variants], default=0
                ),
                largest_file_bytes=max(
                    [v["largest_file_bytes"] for v in variants], default=0
                ),
                variants=variants,
            )
        )
    # Independent maxima are conservative, rather than guessing which tasks overlap.
    retained = sum(sorted([r["retained_source_bytes"] for r in rows], reverse=True)[:2])
    fallback = sum(
        sorted([r["max_current_variant_bytes"] for r in rows], reverse=True)[:2]
    )
    scratch = sum(sorted([r["largest_file_bytes"] for r in rows], reverse=True)[:2])
    estimated_output = output_bytes * 5 // 4 + GIB
    return dict(
        components=rows,
        skipped=skipped,
        retained_two_worker_bytes=retained,
        two_current_variant_copy_fallback_bytes=0,
        rejected_copy_fallback_upper_bound_bytes=fallback,
        two_worker_file_scratch_bytes=scratch,
        metadata_allowance_bytes=GIB,
        historical_deb_output_bytes=output_bytes,
        output_estimate_bytes=estimated_output,
        extra_case_headroom_bytes=retained + scratch + GIB + estimated_output,
        output_caveat="Historical output125%+1GiB estimate, not compression bound; live floor at0.2s",
        staging_caveat="All variant source trees retained; dh_strip disabled; common guard rejects cp-al fallback BEFORE duplicate allocation, invalidating case; file scratch reserved",
    )


def fetch_command() -> list[str]:
    return [
        sys.executable,
        str(SCRIPT.parents[2] / "artifact_manager.py"),
        "fetch",
        "--run-id=41147801272",
        "--run-github-repo=ROCm/rockrel",
        "--stage=all",
        "--amdgpu-families=" + FAMILIES,
        "--expand-family-to-targets",
        "--output-dir=" + str(paths()[0].parent),
    ]


def restore_refetch_times(root: Path, original: Path) -> None:
    # Byte/mode/link equality must be checked BEFORE this metadata restoration.
    records = list(map(json.loads, original.read_text().splitlines()))
    for record in reversed(records):
        path = root / record["path"]
        os.utime(
            path, ns=(record["mtime_ns"], record["mtime_ns"]), follow_symlinks=False
        )


def fetch_owned(state: dict, name: str) -> None:
    root, _, output, _ = paths()
    if root.exists() or root.is_symlink():
        raise RuntimeError("Refuse to fetch into a pre-existing input directory")
    root.mkdir()
    state["root_inode"] = list(snapshots.SnapshotReset.inode(root))
    state["root_verified"] = False
    save_state(state)
    env = dict(os.environ, RELEASE_TYPE="nightly", AWS_EC2_METADATA_DISABLED="true")
    support.monitored_run(
        fetch_command(),
        output / name,
        state["deadline"],
        state["reserve_bytes"],
        root.parent,
        env,
    )
    state["root_inode"] = list(snapshots.SnapshotReset.inode(root))
    save_state(state)
    pre.check_links(root)
    before = pre.manifest(
        root, output / f"{name}-before-times.jsonl", state["deadline"]
    )
    if before["content_sha256"] != state["initial"]["content_sha256"]:
        raise RuntimeError("Re-fetched content/modes/links differ from original inputs")
    restore_refetch_times(root, output / "pristine-before.jsonl")
    state["root_verified"] = True
    save_state(state)


def prepare(args) -> None:
    root, work, output, case = paths()
    snapshots.validate_ownership(
        root, root.parent / "scan-diagnostics/input-ownership.json"
    )
    if output.exists() or work.exists() or case.exists():
        raise RuntimeError("Fresh diagnostic/work/case paths required; no adoption")
    output.mkdir()
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    pre.check_links(root)
    initial = pre.manifest(root, output / "pristine-before.jsonl", deadline)
    if (
        initial["content_sha256"] != EXPECTED_CONTENT
        or initial["counts"]["files"] != 61567
        or initial["counts"]["bytes"] != 51399860152
    ):
        raise RuntimeError("Historical783-input content/mode/link identity differs")
    sources = pre.prepare_sources(output)
    hardlinks = support.hardlink_groups(root, output / "pristine-before.jsonl")
    pre.dump(output / "pristine-hardlinks.json", hardlinks)
    with urllib.request.urlopen(support.INDEX_URL, timeout=60) as response:
        data = response.read()
    (output / "historical-Packages.gz").write_bytes(data)
    entries = support.deb822(gzip.decompress(data).decode())
    plan = component_plan(
        root, output / "pristine-before.jsonl", sum(int(e["Size"]) for e in entries)
    )
    pre.dump(output / "storage-plan.json", plan)
    reserve = (
        max(5 * GIB, initial["counts"]["bytes"] // 10) + initial["largest_elf_bytes"]
    )
    state = dict(
        owner=owner(),
        paths=list(map(str, paths())),
        output_inode=list(snapshots.SnapshotReset.inode(output)),
        helper_sha256=support.file_hash(SCRIPT),
        deadline=deadline,
        started=started,
        initial=initial,
        root_inode=list(snapshots.SnapshotReset.inode(root)),
        reset_mode=args.reset_mode,
        reserve_bytes=reserve,
        order=args.order,
        completed=[],
        retained=[],
        plan=plan,
        source_sha256={k: support.file_hash(v) for k, v in sources.items()},
        expected_hardlink_groups=hardlinks if args.reset_mode == "refetch" else [],
    )
    save_state(state)
    pre.dump(
        output / "scope.json",
        dict(
            recorded_product_head="5b7cb88373362ab80dc0f19605915e534f4158aa",
            input_run="ROCm/rockrel41147801272 read-only",
            historical_index_packages=len(entries),
            historical_index_sha256=support.file_hash(
                output / "historical-Packages.gz"
            ),
            worker_count=2,
            version=support.VERSION,
            suffix=support.SUFFIX,
            reset_mode=args.reset_mode,
            cache_policy="Full resets and complete input hashing precede BOTH cases; no cache dropping/cold claim",
            expected_build_wall_s=2 * 5780,
            planned_case_budget_s=9000,
            max_wall_seconds=args.max_wall_seconds,
            retained_outputs="All raw DEBs and local repo,7days, uploaded before exact owned removal",
            limits="No compilation, S3 upload, release publication, notification or real installation",
        ),
    )
    (output / "installed-packages.txt").write_bytes(
        subprocess.check_output(
            ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n"]
        )
    )
    apt_files = []
    for directory in [Path("/etc/apt"), Path("/var/lib/apt/lists")]:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink():
                apt_files.append(
                    dict(
                        path=str(path),
                        size=path.stat().st_size,
                        sha256=support.file_hash(path),
                    )
                )
    pre.dump(output / "apt-state-file-hashes.json", apt_files)
    versions = {}
    for tool in [
        "dpkg-buildpackage",
        "dpkg-deb",
        "dpkg-scanpackages",
        "xz",
        "gzip",
        "patchelf",
        "readelf",
        "tar",
        "zstd",
    ]:
        versions[tool] = subprocess.check_output(
            [tool, "--version"], stderr=subprocess.STDOUT
        ).decode(errors="replace")
    pre.dump(output / "tool-versions.json", versions)
    pre.dump(
        output / "runner.json",
        {
            key: os.getenv(key)
            for key in [
                "RUNNER_NAME",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
            ]
        },
    )
    if args.reset_mode == "snapshot":
        snapshot = snapshots.SnapshotReset(
            root,
            work,
            output / "snapshot",
            initial,
            output / "pristine-before.jsonl",
            deadline,
        )
        try:
            snapshot.prepare()
        finally:
            save_snapshot(state, snapshot)
    else:
        support.floor_check(root.parent, reserve, plan["extra_case_headroom_bytes"])


def worker(args) -> None:
    _, work, output, case = paths()
    state = read_state()
    diag = output / f"case-{len(state['completed'])}-{args.method}"
    import build_package as build
    import deb_package as deb
    import packaging_summary as summary
    import build_package_repo as repo

    pre.dump(
        diag / "metadata-controls.json",
        support.metadata_controls(deb, summary, repo, diag / "metadata-templates"),
    )
    source = output / "sources" / f"runpath_{args.method}.py"
    if support.file_hash(source) != state["source_sha256"][args.method]:
        raise RuntimeError("Product variant source identity changed")
    spec = importlib.util.spec_from_file_location("diagnostic_runpath", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.convert_runpath_to_rpath

    def convert(path):
        import resource

        start, cpu = time.monotonic(), time.process_time()
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            original(path)
        finally:
            end = resource.getrusage(resource.RUSAGE_CHILDREN)
            pre.dump(
                diag / "conversion.json",
                dict(
                    wall_s=time.monotonic() - start,
                    own_cpu_s=time.process_time() - cpu,
                    waited_child_cpu_s=end.ru_utime
                    + end.ru_stime
                    - child.ru_utime
                    - child.ru_stime,
                ),
            )

    build.convert_runpath_to_rpath = convert
    started = time.monotonic()
    try:
        build.main(
            [
                "--dest-dir",
                str(case / "packages"),
                "--rocm-version",
                support.VERSION,
                "--artifacts-dir",
                str(work),
                "--pkg-type",
                "deb",
                "--version-suffix",
                support.SUFFIX,
                "--build-variant",
                "release",
                "--parallel",
                "2",
            ]
        )
    finally:
        pre.dump(
            diag / "build-wall.json",
            dict(
                wall_s=time.monotonic() - started,
                scope="Stock build_package.main incl config, conversion,2workerassembly andsummary; observer external",
            ),
        )


def case_run(args) -> None:
    root, work, output, case = paths()
    state = read_state()
    index = len(state["completed"])
    if index >= 2 or args.method != state["order"][index] or case.exists():
        raise RuntimeError("Unexpected case order or existing case output")
    if state["deadline"] - time.monotonic() < (2 - index) * 9000 + 1800:
        raise RuntimeError("Insufficient runtime for complete cases and recovery")
    if state["reset_mode"] == "snapshot":
        snapshot = load_snapshot(state)
        try:
            if work.exists():
                snapshot.remove_owned(work)
            support.floor_check(
                root.parent,
                state["reserve_bytes"],
                state["initial"]["counts"]["bytes"]
                + state["plan"]["extra_case_headroom_bytes"],
            )
            snapshot.restore(work)
        finally:
            save_snapshot(state, snapshot)
    else:
        # Re-fetch BOTH cases, including the first; reset policy is matched.
        if root.exists():
            remove_owned(root, state["root_inode"], state)
        fetch_owned(state, f"fetch-case-{index}")
        support.floor_check(
            root.parent,
            state["reserve_bytes"],
            state["plan"]["extra_case_headroom_bytes"],
        )
        root.rename(work)
        state["work_inode"] = list(snapshots.SnapshotReset.inode(work))
        state["root_inode"] = None
        save_state(state)
    before = pre.manifest(
        work, output / f"case-{index}-before.jsonl", state["deadline"]
    )
    if any(
        before[k] != state["initial"][k]
        for k in ["content_sha256", "mtime_sha256", "counts"]
    ):
        raise RuntimeError("Case reset differs in input content/modes/links/mtime")
    pre.check_links(work)
    groups = support.hardlink_groups(work, output / f"case-{index}-before.jsonl")
    pre.dump(output / f"case-{index}-hardlinks.json", groups)
    if groups != state["expected_hardlink_groups"]:
        raise RuntimeError("Restored regular-file hardlink topology differs")
    case.mkdir()
    state["case_inode"] = list(snapshots.SnapshotReset.inode(case))
    state["current_method"] = args.method
    save_state(state)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ["ROCM_PATH", "ROCM_DIR", "HIP_PATH", "HIP_DIR"]
    }
    env.update(SOURCE_DATE_EPOCH=str(support.EPOCH), TMPDIR=str(case / "temp"))
    (case / "temp").mkdir()
    diag = output / f"case-{index}-{args.method}"
    diag.mkdir()
    support.hardlink_capability(case, diag / "hardlink-capability.json")
    profile = SCRIPT.parents[2] / "profile_build.py"
    command = [
        sys.executable,
        str(profile),
        "--output",
        str(diag / "profile"),
        "--",
        sys.executable,
        str(SCRIPT),
        "--action",
        "worker",
        "--method",
        args.method,
    ]
    support.monitored_run(
        command,
        diag / "build",
        min(state["deadline"] - 1800, time.monotonic() + 9000),
        state["reserve_bytes"] + GIB,
        root.parent,
        env,
    )
    errors, converted = [], 0
    with (diag / "build/stdout.log").open(errors="replace") as log:
        for line in log:
            if "patchelf failed" in line or "Couldn't update rocm.cfg" in line:
                errors.append(line.rstrip())
            converted += "DT_RUNPATH changed to DT_RPATH:" in line
    pre.dump(
        diag / "conversion-log-verification.json",
        dict(errors=errors, converted=converted),
    )
    if errors:
        raise RuntimeError(
            "Stock conversion reported errors despite package command success"
        )
    package_errors = []
    pattern = re.compile(
        r"\[\s*ERROR\s*\]|^dpkg-buildpackage: error:|^THEROCK_DEB_PROFILE_FALLBACK_REJECTED$|Failed to build (host|meta|versioned|non-versioned|device)"
    )
    for log_path in [diag / "build/stdout.log", *sorted((case / "logs").glob("*.log"))]:
        with log_path.open(errors="replace") as log:
            for number, line in enumerate(log, 1):
                if pattern.search(line):
                    package_errors.append(
                        dict(log=str(log_path), line=number, text=line.rstrip())
                    )
    pre.dump(diag / "package-errors.json", package_errors)
    if package_errors:
        raise RuntimeError(
            "Stock package logs contain failures despite successful top-level exit"
        )
    for name, command in [
        (
            "presence",
            [
                sys.executable,
                str(SCRIPT.with_name("native_linux_package_pre_upload_test.py")),
                "--packages-dir",
                str(case / "packages"),
                "--pkg-type",
                "deb",
            ],
        ),
        (
            "apt-simulate",
            [
                sys.executable,
                str(SCRIPT.with_name("native_linux_package_install_test.py")),
                "--test-type",
                "simulate",
                "--packages-dir",
                str(case / "packages"),
                "--pkg-type",
                "deb",
            ],
        ),
        ("inspect", [sys.executable, str(SCRIPT), "--action", "inspect"]),
        ("repo", [sys.executable, str(SCRIPT), "--action", "repo"]),
    ]:
        support.monitored_run(
            command,
            diag / name,
            state["deadline"] - 1200,
            state["reserve_bytes"] + GIB,
            root.parent,
            env,
        )
    raw = (case / "packages/built_packages.txt").read_bytes()
    pre.dump(
        diag / "build-manifest.json",
        dict(
            sha256=support.file_hash(case / "packages/built_packages.txt"),
            text=raw.decode(),
        ),
    )
    inventory = json.loads((diag / "package-inspection/inventory.json").read_text())
    pre.dump(
        diag / "repository-verification.json",
        support.verify_repository(case / "packages", inventory),
    )
    after = pre.manifest(
        work, output / f"case-{index}-post.jsonl", state["deadline"], dynamic=True
    )
    pre.dump(diag / "post-input-summary.json", after)
    if after["readelf_errors"]:
        raise RuntimeError("Post-conversion dynamic ELF inspection reported errors")
    if index:
        first = output / f"case-0-{state['order'][0]}"
        comparisons = [
            (diag / name, first / name)
            for name in [
                "package-inspection/inventory.json",
                "build-manifest.json",
                "repository-verification.json",
            ]
        ]
        if any(a.read_bytes() != b.read_bytes() for a, b in comparisons):
            raise RuntimeError(
                "Complete raw package/control/payload/repository comparison differs"
            )
        previous = json.loads((first / "post-input-summary.json").read_text())
        if any(
            after[k] != previous[k]
            for k in ["content_sha256", "dynamic_sha256", "readelf_errors", "counts"]
        ):
            raise RuntimeError("Post-conversion artifact trees differ")
    state["completed"].append(args.method)
    pre.dump(
        diag / "validated.json",
        dict(
            complete=True,
            method=args.method,
            packages=len(inventory),
            exact_comparison=index == 1,
        ),
    )
    save_state(state)


def cleanup(args) -> None:
    _, work, _, case = paths()
    state = read_state()
    if not args.artifact_id or not args.artifact_id.isdigit() or not state["completed"]:
        raise RuntimeError(
            "Successful package retention artifact ID required before removal"
        )
    state["retained"].append(
        dict(method=state["current_method"], artifact_id=args.artifact_id)
    )
    save_state(state)
    remove_owned(case, state["case_inode"], state)
    if state["reset_mode"] == "refetch":
        remove_owned(work, state["work_inode"], state)
        state["work_inode"] = None
    save_state(state)


def finish() -> None:
    root, work, output, _ = paths()
    state = read_state()
    if state["reset_mode"] == "snapshot":
        snapshot = load_snapshot(state)
        try:
            snapshot.recover_pristine()
        finally:
            save_snapshot(state, snapshot)
    else:
        if work.exists():
            remove_owned(work, state["work_inode"], state)
        if root.exists() and not state.get("root_verified", True):
            remove_owned(root, state["root_inode"], state)
        if not root.exists():
            fetch_owned(state, "fetch-recovery")
    recovered = pre.manifest(
        root, output / "pristine-recovered.jsonl", state["deadline"]
    )
    if any(
        recovered[k] != state["initial"][k]
        for k in ["content_sha256", "mtime_sha256", "counts"]
    ):
        raise RuntimeError("Recovered pristine input identity differs")
    pre.dump(
        output / "summary.json",
        dict(
            complete=len(state["completed"]) == 2,
            order=state["order"],
            retained=state["retained"],
            pristine_recovered=True,
            scope="Full native DEB paired comparison; not whole-CI speedup",
            total_diagnostic_wall_s=time.monotonic() - state["started"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=["prepare", "case", "worker", "inspect", "repo", "cleanup", "finish"],
        required=True,
    )
    parser.add_argument("--method", choices=["A", "B"])
    parser.add_argument("--order", choices=["AB", "BA"], default="AB")
    parser.add_argument(
        "--reset-mode", choices=["snapshot", "refetch"], default="refetch"
    )
    parser.add_argument("--max-wall-seconds", type=int, default=21600)
    parser.add_argument("--artifact-id")
    args = parser.parse_args()
    pre.require_rock_ci()
    if sys.platform != "linux":
        parser.error("Native DEB diagnostic requires Linux")
    if args.action in ["case", "worker"] and args.method is None:
        parser.error("Case/worker requires --method")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "case":
        case_run(args)
    elif args.action == "worker":
        worker(args)
    elif args.action == "cleanup":
        cleanup(args)
    elif args.action == "finish":
        finish()
    else:
        _, _, output, case = paths()
        state = read_state()
        diag = output / f"case-{len(state['completed'])}-{state['current_method']}"
        if args.action == "inspect":
            support.inspect_packages(case / "packages", diag / "package-inspection")
        else:
            import deb_package as deb
            import packaging_summary as summary
            import build_package_repo as repo

            pre.dump(
                diag / "repo-clock-controls.json",
                support.metadata_controls(
                    deb, summary, repo, diag / "repo-metadata-templates"
                ),
            )
            repo.main(
                [
                    "--package-dir",
                    str(case / "packages"),
                    "--pkg-type",
                    "deb",
                    "--release-type",
                    "ci",
                ]
            )


if __name__ == "__main__":
    main()
