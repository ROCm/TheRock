#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Opt-in I-029 support; workflow wiring and common controls remain separate."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = "f4e24d58b3f313f1a31068def1e35376e43c2f49"
ARCHES = [f"gfx103{i}" for i in range(7)]
ENTRIES = 17204
GENERATOR_SUM = "5a96d3dd1f2fdcbfdacf246a427050105a2b7b3b68b73c6a387d343db92d8653"
PATCH = "patches/build-profile/rocm-libraries/0004-rocfft-normal-private-aot.patch"
HEADER = "build_tools/diagnostics/rocfft_normal_aot.h"
COPIED_HEADER = "projects/rocfft/shared/normal_aot_transaction.h"
HOST_FILES = {
    "projects/rocfft/library/src/rocfft_aot_helper.cpp",
    "projects/rocfft/library/src/rtc_cache.cpp",
    "projects/rocfft/library/src/include/rtc_cache.h",
}
KNOBS = [
    "OMP_NUM_THREADS",
    "ROCFFT_RTC_MAX_SUBPROCESSES",
    "ROCFFT_RTC_PROCESS",
    "ROCFFT_RTC_CACHE_READ_DISABLE",
    "ROCFFT_RTC_CACHE_WRITE_DISABLE",
    "ROCFFT_LAYER",
    "ROCFFT_RTC_CACHE_PATH",
    "ROCFFT_RTC_SYS_CACHE_PATH",
    "ROCFFT_AOT_BATCH_TRANSACTION",
    "ROCFFT_AOT_PRIVATE_CACHE_PATH",
    "ROCFFT_AOT_PRIVATE_CACHE_ID",
    "ROCFFT_AOT_RECEIPT_DIR",
    "AMD_COMGR_CACHE",
    "AMD_COMGR_CACHE_DIR",
    "AMD_COMGR_CACHE_POLICY",
    "AMD_COMGR_DRIVER_OPTIONS_APPEND",
    "AMD_COMGR_EMIT_VERBOSE_LOGS",
    "AMD_COMGR_USE_EMBEDDED_LIBCXX",
    "HIPRTC_COMPILE_OPTIONS_APPEND",
    "HIPRTC_LINK_OPTIONS_APPEND",
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def source_identity(repository: Path) -> dict:
    return dict(
        head=git(repository, "rev-parse", "HEAD"),
        branch=git(repository, "branch", "--show-current"),
        shallow=git(repository, "rev-parse", "--is-shallow-repository"),
        head_and_parents=git(repository, "log", "-1", "--format=%H %P"),
    )


def selected(enabled: bool, env: dict, platform: str) -> bool:
    if not enabled:
        return False
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
    ):
        raise ValueError("I-029 executes only in ROCm/TheRock Actions")
    if env.get("NORMAL_VALIDATION") != "true":
        raise ValueError("Explicit normal-validation mode is required")
    if (
        env.get("STAGE_NAME") != "math-libs"
        or env.get("AMDGPU_FAMILIES", "").lower() != "gfx103x-all"
    ):
        return False
    if platform != "linux" or env.get("RELEASE_TYPE") != "ci":
        raise ValueError("The selected product requires Linux and CI destinations")
    for name in [
        "ROCFFT_LAYER",
        "AMD_COMGR_DRIVER_OPTIONS_APPEND",
        "HIPRTC_COMPILE_OPTIONS_APPEND",
        "HIPRTC_LINK_OPTIONS_APPEND",
    ]:
        if env.get(name):
            raise ValueError(f"Uncontrolled source logging/compiler override: {name}")
    if env.get("AMD_COMGR_EMIT_VERBOSE_LOGS") not in (None, "", "0"):
        raise ValueError("Verbose COMGR logging is not part of this comparison")
    return True


def generator_inputs(source: Path) -> list[dict]:
    cmake = (source / "library/src/CMakeLists.txt").read_text(encoding="utf-8")
    paths = {"shared/embed-to-header-rtc.py"}
    for name in ["kgen_embed_files", "kgen_logic_files"]:
        match = re.search(r"set\(\s*" + name + r"\s+(.*?)\n\)", cmake, re.DOTALL)
        if not match:
            raise ValueError(f"Unknown generator checksum input list: {name}")
        paths.update(re.findall(r"\$\{CMAKE_SOURCE_DIR\}/([^\s)]+)", match[1]))
    return [{"path": name, "sha256": digest(source / name)} for name in sorted(paths)]


def apply_product(
    libraries: Path, output: Path, *, expected_head: str = LIBRARIES
) -> dict:
    before = source_identity(libraries)
    if before["head"] != expected_head or git(libraries, "diff", "--name-only", "HEAD"):
        raise ValueError("Expected pristine pinned source; do not rewrite HEAD/history")
    copied = libraries / COPIED_HEADER
    if copied.exists() or copied.is_symlink():
        raise ValueError("Refuse to replace a preexisting product header")
    patch, header = ROOT / PATCH, ROOT / HEADER
    paths = re.findall(
        r"^diff --git a/(\S+) b/\1$", patch.read_text(encoding="utf-8"), re.MULTILINE
    )
    if set(paths) != HOST_FILES or len(paths) != len(HOST_FILES):
        raise ValueError("Product patch may touch only the reviewed three host files")
    generator_before = generator_inputs(libraries / "projects/rocfft")
    original_files = {name: digest(libraries / name) for name in HOST_FILES}
    subprocess.run(
        ["git", "-C", str(libraries), "apply", "--check", str(patch)], check=True
    )
    subprocess.run(["git", "-C", str(libraries), "apply", str(patch)], check=True)
    shutil.copyfile(header, copied)
    if (
        source_identity(libraries) != before
        or generator_inputs(libraries / "projects/rocfft") != generator_before
    ):
        raise ValueError("Source provenance or generator inputs changed")
    if set(git(libraries, "diff", "--name-only", "HEAD").splitlines()) != HOST_FILES:
        raise ValueError("Unexpected tracked source changes")
    receipt = dict(
        schema_version=1,
        identity=before,
        patch=PATCH,
        patch_sha256=digest(patch),
        header_source=HEADER,
        header_target=COPIED_HEADER,
        header_sha256=digest(copied),
        original_host_files=original_files,
        applied_host_files={name: digest(libraries / name) for name in HOST_FILES},
        generator_inputs=generator_before,
        aot_cmake_sha256=digest(
            libraries / "projects/rocfft/library/src/CMakeLists.txt"
        ),
        product="rocfft-private-batch",
        scope="linux/math-libs/gfx103X-all",
        note="Same source in BOTH arms; apply/copy only, no commit/history or compile-flag change",
    )
    save(output / "source-control.json", receipt, exclusive=True)
    shutil.copyfile(patch, output / patch.name)
    shutil.copyfile(header, output / header.name)
    return receipt


def verify_source(libraries: Path, source: dict) -> None:
    if source_identity(libraries) != source["identity"]:
        raise ValueError("Source HEAD/branch/history changed")
    for name, expected in {
        **source["applied_host_files"],
        source["header_target"]: source["header_sha256"],
    }.items():
        path = libraries / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Product source input changed: {name}")
    if generator_inputs(libraries / "projects/rocfft") != source["generator_inputs"]:
        raise ValueError("Generator checksum inputs changed")
    if (
        digest(libraries / "projects/rocfft/library/src/CMakeLists.txt")
        != source["aot_cmake_sha256"]
    ):
        raise ValueError("Original AOT commands/dependencies changed")


def require_fresh_outputs(nested: Path) -> None:
    for name in [
        "rocfft_temp_cache.db",
        *[f"rocfft_kernel_cache_{arch}.db" for arch in ARCHES],
    ]:
        path = nested / "library/src" / name
        if path.exists() or path.is_symlink():
            raise ValueError(
                "Fresh original temp/shipped caches required; nothing is deleted"
            )


def check_toolchain(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'set\(GPU_TARGETS\s+"?([^\s"]+)"?\s+CACHE STRING', text)
    if not match or match[1].split(";") != ARCHES:
        raise ValueError("Generated rocFFT targets differ from the validated seven")


def cache_inventory(path: Path) -> dict:
    # Complete private metadata: deliberately no generic100k cap.
    records = []
    entries = 0
    for entry in path.rglob("*"):
        entries += 1
        if entry.is_symlink():
            raise ValueError("Unexpected private COMGR symlink")
        if entry.is_file():
            records.append(
                dict(path=str(entry.relative_to(path)), bytes=entry.stat().st_size)
            )
    records.sort(key=lambda r: r["path"])
    manifest = hashlib.sha256()
    for record in records:
        manifest.update(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return dict(
        path=str(path),
        device=path.stat().st_dev,
        complete=True,
        entry_count=entries,
        file_count=len(records),
        bytes=sum(r["bytes"] for r in records),
        files=records,
        metadata_sha256=manifest.hexdigest(),
        scope="All private entry names/sizes, not content hashes",
    )


def prepare_product(
    build: Path, output: Path, libraries: Path, case: str, env: dict
) -> dict:
    if case not in ("baseline", "candidate"):
        raise ValueError("Explicit baseline/candidate case is required")
    if (output / "prepared.json").exists():
        raise ValueError("Refuse to adopt an existing prepared state")
    source = json.loads((output / "source-control.json").read_text())
    verify_source(libraries, source)
    nested = build / "math-libs/rocFFT/build"
    check_toolchain(build / "math-libs/rocFFT/build_toolchain.cmake")
    require_fresh_outputs(nested)
    parent = nested / "library/src"
    parent.mkdir(parents=True, exist_ok=True)
    comgr = build / "rocfft-private-comgr-cache"
    comgr.mkdir()  # Exclusive fresh cache; never clear or adopt a previous one.
    owner = Path(tempfile.mkdtemp(prefix="rocfft-normal-chain-", dir=parent)).resolve()
    receipts = output / "receipts"
    receipts.mkdir(mode=0o700)
    owner_id = uuid.uuid4().hex
    marker = owner / ".rocfft-aot-owner"
    with marker.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(owner_id + "\n")
    if len({p.stat().st_dev for p in [parent, owner, comgr]}) != 1:
        raise ValueError("Private caches must stay on the original cache filesystem")
    private = owner / "rocfft_temp_cache.db"
    runtime = dict(
        ROCFFT_AOT_BATCH_TRANSACTION="all" if case == "candidate" else "0",
        ROCFFT_AOT_PRIVATE_CACHE_PATH=str(private),
        ROCFFT_AOT_PRIVATE_CACHE_ID=owner_id,
        ROCFFT_AOT_RECEIPT_DIR=str(receipts.resolve()),
        AMD_COMGR_CACHE="1",
        AMD_COMGR_CACHE_DIR=str(comgr.resolve()),
        AMD_COMGR_CACHE_POLICY="",
    )
    state = dict(
        schema_version=1,
        case=case,
        run_id=env.get("GITHUB_RUN_ID"),
        run_attempt=env.get("GITHUB_RUN_ATTEMPT"),
        build=str(build),
        nested=str(nested),
        owner_id=owner_id,
        path=str(private),
        marker=str(marker),
        marker_sha256=digest(marker),
        device=owner.stat().st_dev,
        original_parent_device=parent.stat().st_dev,
        receipts=str(receipts.resolve()),
        comgr=str(comgr.resolve()),
        runtime_environment=runtime,
        original_runtime_knobs={name: env.get(name) for name in KNOBS},
        source_control_sha256=digest(output / "source-control.json"),
        cache_contract="fresh private COMGR + fresh owned AOT DB growing across seven original helpers; ccache is separate",
        initial_private_exists=private.exists(),
        initial_comgr=cache_inventory(comgr),
    )
    save(output / "runtime-environment.json", runtime, exclusive=True)
    save(output / "prepared.json", state, exclusive=True)
    return state


def export_environment(state: dict, path: Path) -> None:
    # Explicit GITHUB_ENV opt-in only; no eval, no credential/environment dump.
    if any(
        "\n" in value or "\r" in value
        for value in state["runtime_environment"].values()
    ):
        raise ValueError("Multiline runtime value rejected")
    with path.open("a", encoding="utf-8") as stream:
        for key, value in state["runtime_environment"].items():
            stream.write(f"{key}={value}\n")


def regular_owned_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Expected a regular single-link owned file: {path}")


def inventory_database(path: Path, destination: Path) -> dict:
    regular_owned_file(path)
    semantic = hashlib.sha256()
    count = 0
    arches, generators, versions = set(), set(), set()
    code_bytes = 0
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        integrity = db.execute("PRAGMA quick_check").fetchall()
        with destination.open("w", encoding="utf-8") as stream:
            for kernel, arch, version, generator, code, timestamp in db.execute(
                "SELECT kernel_name,arch,hip_version,generator_sum,code,timestamp FROM cache_v1 "
                "ORDER BY kernel_name,arch,hip_version,generator_sum"
            ):
                record = dict(
                    kernel_name=kernel,
                    arch=arch,
                    hip_version=version,
                    generator_sum=generator.hex(),
                    code_bytes=len(code),
                    code_sha256=hashlib.sha256(code).hexdigest(),
                    timestamp=timestamp,
                )
                encoded = (
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
                stream.write(encoded)
                semantic.update(encoded.encode())
                count += 1
                code_bytes += len(code)
                arches.add(arch)
                generators.add(generator.hex())
                versions.add(version)
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        file_sha256=digest(path),
        semantic_sha256=semantic.hexdigest(),
        entries=count,
        arches=sorted(arches),
        generator_sums=sorted(generators),
        hip_versions=sorted(versions),
        code_bytes=code_bytes,
        quick_check=integrity,
    )


def generated_commands(nested: Path) -> dict:
    import shlex

    commands = {}
    for index, arch in enumerate(ARCHES):
        target = f"library/src/rocfft_kernel_cache_{arch}.db"
        command = subprocess.check_output(
            ["ninja", "-C", str(nested), "-t", "commands", "-s", target], text=True
        ).strip()
        tokens = shlex.split(command)
        helper = str(nested / "library/src/rocfft_aot_helper")
        if tokens.count(helper) != 1:
            raise ValueError(f"Missing original AOT command: {arch}")
        at = tokens.index(helper)
        expected = [
            "",
            str(nested / target),
            str(nested / "library/src/rocfft_rtc_helper"),
            arch,
        ]
        if tokens[at + 1 :] != expected:
            raise ValueError(f"Changed original AOT arguments: {arch}")
        query = subprocess.check_output(
            [
                "ninja",
                "-C",
                str(nested),
                "-t",
                "query",
                f"library/src/rocfft_kernel_cache_{arch}_target",
            ],
            text=True,
        )
        if index and f"rocfft_kernel_cache_{ARCHES[index-1]}_target" not in query:
            raise ValueError(f"Original sequential dependency is absent: {arch}")
        commands[arch] = dict(command=command, query=query)
    return commands


def validate_product(
    output: Path, libraries: Path, case: str, env: dict, *, entries: int = ENTRIES
) -> dict:
    state = json.loads((output / "prepared.json").read_text())
    if (
        state["case"] != case
        or state["run_id"] != env.get("GITHUB_RUN_ID")
        or state["run_attempt"] != env.get("GITHUB_RUN_ATTEMPT")
    ):
        raise ValueError("Prepared state belongs to another case/run/attempt")
    source = json.loads((output / "source-control.json").read_text())
    if digest(output / "source-control.json") != state["source_control_sha256"]:
        raise ValueError("Source receipt changed")
    verify_source(libraries, source)
    cache, marker = Path(state["path"]), Path(state["marker"])
    regular_owned_file(marker)
    regular_owned_file(cache)
    if (
        marker.read_bytes() != (state["owner_id"] + "\n").encode()
        or digest(marker) != state["marker_sha256"]
        or cache.stat().st_dev != state["original_parent_device"]
    ):
        raise ValueError("Owned cache marker/filesystem changed")
    for suffix in ["-wal", "-shm"]:
        path = Path(str(cache) + suffix)
        if path.exists() or path.is_symlink():
            regular_owned_file(path)
    receipts = Path(state["receipts"])
    if receipts.is_symlink() or {p.name for p in receipts.iterdir()} != {
        f"{a}.json" for a in ARCHES
    }:
        raise ValueError("Missing, duplicate or partial completion receipts")
    nested = Path(state["nested"])
    expected_batch = case == "candidate"
    inventories = []
    for index, arch in enumerate(ARCHES):
        receipt_path = receipts / f"{arch}.json"
        regular_owned_file(receipt_path)
        row = json.loads(receipt_path.read_text())
        expected_path = nested / f"library/src/rocfft_kernel_cache_{arch}.db"
        if (
            row.get("schema_version") != 1
            or row.get("success") is not True
            or row.get("arch") != arch
            or row.get("owner_id") != state["owner_id"]
            or row.get("cache_path") != state["path"]
            or row.get("mode") != ("all" if expected_batch else "0")
            or row.get("begin_succeeded") != expected_batch
            or row.get("commit_succeeded") != expected_batch
            or row.get("cache_existed_before") != (index != 0)
            or row.get("output_path") != str(expected_path)
            or row.get("subprocess_fallback_errors") != 0
        ):
            raise ValueError(f"Invalid ownership/transaction/workflow receipt: {arch}")
        for name in ["policy_before", "policy_after"]:
            policy = row.get(name, {})
            if (
                policy.get("journal_mode") != "wal"
                or policy.get("synchronous") != "2"
                or policy.get("autocommit") is not True
            ):
                raise ValueError(f"WAL/FULL/autocommit policy changed: {arch}")
        if row["policy_before"] != row["policy_after"]:
            raise ValueError(f"SQLite policy changed during helper: {arch}")
        for field, key in [
            ("comgr_cache", "AMD_COMGR_CACHE"),
            ("comgr_cache_dir", "AMD_COMGR_CACHE_DIR"),
            ("comgr_cache_policy", "AMD_COMGR_CACHE_POLICY"),
        ]:
            if row.get(field) != state["runtime_environment"][key]:
                raise ValueError(f"Actual helper COMGR control differs: {arch}/{field}")
        inventory = inventory_database(
            expected_path, output / f"inventory-{arch}.jsonl"
        )
        if (
            inventory["quick_check"] != [("ok",)]
            or inventory["entries"] != entries
            or inventory["arches"] != [arch]
            or inventory["generator_sums"] != [GENERATOR_SUM]
            or row.get("output_bytes") != inventory["bytes"]
        ):
            raise ValueError(f"Incomplete/changed shipped cache: {arch}")
        inventories.append(inventory)
    with closing(
        sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        counts = dict(db.execute("SELECT arch,count(*) FROM cache_v1 GROUP BY arch"))
        if db.execute("PRAGMA quick_check").fetchall() != [
            ("ok",)
        ] or counts != dict.fromkeys(ARCHES, entries):
            raise ValueError("Final growing DB did not retain all seven worksets")
    save(output / "cache-inventories.json", inventories)
    save(output / "original-helper-commands.json", generated_commands(nested))
    helpers = [
        dict(path=str(path), sha256=digest(path))
        for path in [
            nested / "library/src/rocfft_aot_helper",
            nested / "library/src/rocfft_rtc_helper",
        ]
    ]
    save(output / "helper-binaries.json", helpers)
    after = cache_inventory(Path(state["comgr"]))
    save(output / "comgr-after.json", after)
    result = dict(
        success=True,
        scope="linux/math-libs/gfx103X-all",
        case=case,
        run_id=state["run_id"],
        entries_by_arch=counts,
        total_entries=sum(counts.values()),
        owned_db_bytes=cache.stat().st_size,
        owned_db_file_sha256=digest(cache),
        comgr_metadata_complete=after["complete"],
        source_control_sha256=state["source_control_sha256"],
        caveat="Local output/transaction gate only; compare both arms' payload/tool/cache identities and normal checks before claiming gain",
    )
    save(output / "validation.json", result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["apply", "prepare", "validate"])
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Explicit product opt-in; otherwise no mutation",
    )
    parser.add_argument("--case", choices=["baseline", "candidate"])
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--github-env",
        action="store_true",
        help="prepare only: export nonsecret controls to GITHUB_ENV",
    )
    args = parser.parse_args()
    env = dict(os.environ)
    if not selected(args.enable, env, sys.platform):
        print(
            json.dumps(
                dict(
                    selected=False,
                    reason="disabled or outside Linux gfx103X math scope",
                )
            )
        )
        return
    if args.case is None:
        raise ValueError("Explicit baseline/candidate case is required")
    build = args.build_dir.resolve()
    output = (args.output or build / "logs/normal-validation/rocfft-aot").resolve()
    if not build.is_relative_to(ROOT) or not output.is_relative_to(build):
        raise ValueError(
            "Owned product paths must remain under this checkout's build directory"
        )
    if args.github_env and args.command != "prepare":
        raise ValueError("Only prepare can export runtime controls")
    libraries = ROOT / "rocm-libraries"
    if args.command == "apply":
        result = apply_product(libraries, output)
    elif args.command == "prepare":
        result = prepare_product(build, output, libraries, args.case, env)
        if args.github_env:
            export_environment(result, Path(env["GITHUB_ENV"]))
    else:
        try:
            result = validate_product(output, libraries, args.case, env)
        except Exception as error:
            save(
                output / "validation-error.json",
                dict(
                    success=False,
                    error=repr(error),
                    case=args.case,
                    run_id=env.get("GITHUB_RUN_ID"),
                ),
            )
            raise
    print(
        json.dumps(
            dict(
                selected=True,
                command=args.command,
                case=args.case,
                output=str(output),
                success=result.get("success"),
                source_head=LIBRARIES,
            )
        )
    )


if __name__ == "__main__":
    main()
