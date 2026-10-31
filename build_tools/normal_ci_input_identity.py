# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Read-only generated-tool, RTC/cache and explicitly declared source receipts.

The normal controller guards CI execution. These functions are also exercised by
small inert file fixtures; no compiler or cache behavioral experiment is needed.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

RTC_KNOBS = [
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
    "AMD_COMGR_CACHE",
    "AMD_COMGR_CACHE_DIR",
    "AMD_COMGR_CACHE_POLICY",
    "AMD_COMGR_DRIVER_OPTIONS_APPEND",
    "AMD_COMGR_EMIT_VERBOSE_LOGS",
    "HIPRTC_COMPILE_OPTIONS_APPEND",
    "HIPRTC_LINK_OPTIONS_APPEND",
    "LLVM_PATH",
    "AMD_COMGR_USE_EMBEDDED_LIBCXX",
    "AMD_COMGR_USE_VFS",
    "AMD_COMGR_SAVE_TEMPS",
    "AMD_COMGR_SAVE_LLVM_TEMPS",
    "AMD_COMGR_REDIRECT_LOGS",
    "AMD_COMGR_TIME_STATISTICS",
    "AMD_COMGR_TIME_STATISTICS_GRANULARITY",
    "AMD_COMGR_LOG_LEVEL",
]
SDK_ROOTS = [
    "compiler/amd-llvm/dist",
    "compiler/amd-comgr/dist",
    "core/clr/dist",
    "core/ROCR-Runtime/dist",
]
COMPILER_KEY = re.compile(
    r"^CMAKE_(?:(?:C|CXX|HIP|ASM)_COMPILER(?:_AR|_RANLIB|_CLANG_SCAN_DEPS)?|LINKER|AR|RANLIB)$"
)
PROVIDER_KEYS = {
    "amd_comgr_DIR",
    "hiprtc_DIR",
    "hip_DIR",
    "hsa-runtime64_DIR",
    "HIP_PATH",
    "ROCM_PATH",
    "CMAKE_PREFIX_PATH",
}


def file_identity(path, fingerprint):
    p = Path(path)
    row = {"path": str(p)}
    try:
        row["resolved_path"] = str(p.resolve(strict=True))
        if p.is_dir():
            row["status"] = "directory-provider; files captured separately"
        elif p.is_file():
            row.update(status="file", bytes=p.stat().st_size, sha256=fingerprint(p))
        else:
            row["status"] = "not-regular-file"
    except (OSError, ValueError) as error:
        row.update(status="unavailable", error=str(error))
    return row


def build_metadata_files(build):
    if not build.is_dir():
        return
    for directory, names, files in os.walk(build, followlinks=False):
        names[:] = [
            name
            for name in names
            if name not in {".git", "dist", "stage", "artifacts", "logs", "Testing"}
        ]
        for name in files:
            if name in {"CMakeCache.txt", "build.ninja"} or re.fullmatch(
                r"CMake(?:C|CXX|HIP|ASM)Compiler\.cmake", name
            ):
                yield Path(directory) / name


def generated_inputs(build, fingerprint):
    fields, commands, errors = [], [], []
    for path in build_metadata_files(build):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
            continue
        for number, line in enumerate(lines, 1):
            if path.name == "build.ninja":
                if "COMMAND =" not in line or not (
                    "rocfft_aot_helper" in line
                    or any(key + "=" in line for key in RTC_KNOBS)
                ):
                    continue
                try:
                    tokens = shlex.split(line.split("COMMAND =", 1)[1])
                except ValueError as error:
                    errors.append(dict(path=str(path), line=number, error=str(error)))
                    continue
                overrides = {
                    key: value
                    for token in tokens
                    for key, sep, value in [token.partition("=")]
                    if sep and key in RTC_KNOBS
                }
                record = dict(
                    origin=str(path), line=number, command_knob_assignments=overrides
                )
                for index, token in enumerate(tokens):
                    if (
                        Path(token).name == "rocfft_aot_helper"
                        and Path(token).is_absolute()
                        and len(tokens) > index + 3
                        and (index == 0 or tokens[index - 1] != "-o")
                    ):
                        cwd = (
                            Path(tokens[tokens.index("cd") + 1])
                            if "cd" in tokens
                            else path.parent
                        )
                        temporary = tokens[index + 1] or "rocfft_temp_cache.db"
                        temporary = Path(temporary)
                        if not temporary.is_absolute():
                            temporary = cwd / temporary
                        record["aot_source_defined_overrides"] = dict(
                            ROCFFT_RTC_CACHE_PATH=str(temporary),
                            ROCFFT_RTC_SYS_CACHE_PATH=":memory:",
                            rtc_helper=tokens[index + 3],
                            output_database=tokens[index + 2],
                            architectures=tokens[index + 4 :],
                            scope="inferred from pinned normal AOT argv/source, not observed child environment; future private-AOT source overrides require declared inputs",
                        )
                commands.append(record)
                continue
            if path.name == "CMakeCache.txt":
                match = re.match(r"([^#/:][^:]*):[^=]+=(.*)$", line)
                if not match:
                    continue
                key, value = match.groups()
            else:
                match = re.match(r'set\((\w+) "(.*)"\)', line)
                if not match:
                    continue
                key, value = match.groups()
                value = value.replace('\\"', '"').replace("\\\\", "\\")
            is_compiler = bool(COMPILER_KEY.fullmatch(key))
            is_knob = key.startswith(("ROCFFT_", "AMD_COMGR_", "HIPRTC_")) and any(
                word in key for word in ["CACHE", "RTC", "COMPILE", "PROCESS"]
            )
            if not (is_compiler or key in PROVIDER_KEYS or is_knob):
                continue
            record = dict(
                origin=str(path),
                line=number,
                key=key,
                value=value,
                role="generated-compiler" if is_compiler else "provider-or-RTC-setting",
            )
            if value and not any(
                marker in value for marker in [";", "${", "-NOTFOUND"]
            ):
                candidate = Path(value)
                if candidate.is_absolute():
                    record["identity"] = file_identity(candidate, fingerprint)
                elif is_compiler:
                    record["identity"] = dict(
                        status="unresolved-generated-name; PATH is not substituted"
                    )
            fields.append(record)
    return dict(
        fields=fields,
        rtc_commands=commands,
        errors=errors,
        scope="generated CMake selections; ambient PATH tools are separate and not active-compiler proof",
    )


def sdk_inputs(build, fingerprint):
    rows, roots = [], []
    executables = {
        "clang",
        "clang++",
        "amdclang",
        "amdclang++",
        "ld.lld",
        "lld",
        "clang-linker-wrapper",
        "rocfft_rtc_helper",
        "rocfft_aot_helper",
    }
    for relative in SDK_ROOTS:
        root = build / relative
        roots.append(dict(path=str(root), exists=root.is_dir()))
        if not root.is_dir():
            continue
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                p = Path(directory) / name
                if (
                    name in executables
                    or name.startswith("clang-")
                    or p.suffix in {".bc", ".so", ".dll", ".dylib"}
                    or ".so." in name
                ):
                    row = file_identity(p, fingerprint)
                    row["sdk_root"] = relative
                    rows.append(row)
    for directory, _, names in (
        os.walk(build / "math-libs/rocFFT", followlinks=False)
        if (build / "math-libs/rocFFT").is_dir()
        else []
    ):
        for name in names:
            if name in {"rocfft_rtc_helper", "rocfft_aot_helper"}:
                row = file_identity(Path(directory) / name, fingerprint)
                row["sdk_root"] = "rocFFT active runtime helper"
                rows.append(row)
    return dict(
        roots=roots,
        files=rows,
        scope="LLVM plus separate COMGR/CLR/HIPRTC/HSA runtime candidate closure; active loader selection queried separately",
    )


def loader_dependencies(binary, fingerprint, inspect):
    result = inspect(binary)
    providers, unresolved = [], []
    for line in result.stdout.splitlines():
        if "=> not found" in line:
            unresolved.append(line.strip())
            continue
        match = re.search(
            r"(?:=>\s+)?((?:[A-Za-z]:)?/[^\n]+?)\s+\(0x[0-9a-fA-F]+\)", line
        )
        if match:
            providers.append(file_identity(Path(match.group(1).strip()), fingerprint))
    return dict(
        binary=str(binary),
        returncode=result.returncode,
        providers=providers,
        unresolved=unresolved,
        stderr=result.stderr[:16384],
        scope="loader query in capture environment; command-local LD_LIBRARY_PATH/dlopen overrides not established",
    )


def query_loader(binary):
    if sys.platform != "linux" or not shutil.which("ldd"):
        return subprocess.CompletedProcess(
            [], 127, "", "loader query unavailable on this platform"
        )
    try:
        return subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess([], 124, "", str(error))


def cache_inventory(path, limit=100000):
    p = Path(path)
    if p.resolve() == Path(p.anchor) or p.name in {"", ".", ".."}:
        return dict(path=str(p), status="unsafe-broad-path; no scan", complete=False)
    if not p.exists():
        return dict(path=str(p), status="absent", complete=True, entries=[])
    if p.is_file():
        stat = p.stat()
        return dict(
            path=str(p),
            status="file",
            complete=True,
            entries=[dict(path=p.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)],
        )
    rows, complete = [], True
    try:
        for directory, dirs, files in os.walk(p, followlinks=False):
            for name in sorted(dirs + files):
                child = Path(directory) / name
                stat = child.lstat()
                rows.append(
                    dict(
                        path=child.relative_to(p).as_posix(),
                        bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        symlink=child.is_symlink(),
                    )
                )
                if len(rows) >= limit:
                    complete = False
                    break
            if not complete:
                break
    except OSError as error:
        return dict(
            path=str(p),
            status="inventory-error",
            error=str(error),
            complete=False,
            entries=rows,
        )
    rows.sort(key=lambda row: row["path"])
    return dict(
        path=str(p),
        status="directory",
        complete=complete,
        entries=rows,
        inventory_sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        scope="metadata only; cache contents not hashed or modified",
    )


def runtime_cache_state(workspace, env, platform):
    knobs = {key: env.get(key) for key in RTC_KNOBS}
    caches = []
    if env.get("AMD_COMGR_CACHE_DIR"):
        caches.append(
            dict(
                origin="explicit ambient AMD_COMGR_CACHE_DIR",
                inventory=cache_inventory(env["AMD_COMGR_CACHE_DIR"]),
            )
        )
    elif platform == "linux":
        base = env.get("XDG_CACHE_HOME") or (
            str(Path(env["HOME"]) / ".cache") if env.get("HOME") else None
        )
        if base:
            caches.append(
                dict(
                    origin="Linux default candidate from pinned COMGR source; local-filesystem enablement unverified",
                    inventory=cache_inventory(Path(base) / "comgr"),
                )
            )
    for key in [
        "ROCFFT_RTC_CACHE_PATH",
        "ROCFFT_RTC_SYS_CACHE_PATH",
        "ROCFFT_AOT_PRIVATE_CACHE_PATH",
    ]:
        value = env.get(key)
        if value and value != ":memory:":
            p = Path(value)
            caches.append(
                dict(
                    origin="ambient " + key,
                    inventory=cache_inventory(p if p.is_absolute() else workspace / p),
                )
            )
    normal_aot = (
        workspace / "build/math-libs/rocFFT/build/library/src/rocfft_temp_cache.db"
    )
    caches.append(
        dict(
            origin="pinned normal AOT default candidate; generated argv confirms later",
            inventory=cache_inventory(normal_aot),
        )
    )
    return dict(
        ambient_knobs=knobs,
        caches=caches,
        comgr_policy="unset CACHE means enabled by default; default location is disabled on nonlocal filesystems; unset POLICY uses default pruning",
        scope="snapshot at capture time, not a guarantee of per-helper initial state; generated commands and source overrides are separate; ccache regime does not control RTC/COMGR",
        mutation="none; no cache creation, deletion or policy override",
    )


def declared_sources(workspace, manifest, case, fingerprint, stage=None):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    receipts, valid = [], True
    for kind in ["patches", "copied_inputs"]:
        for entry in data.get(kind, []):
            if case not in entry.get("cases", ["baseline", "candidate"]):
                continue
            if "stages" in entry and stage not in entry["stages"]:
                receipts.append(
                    dict(
                        kind=kind,
                        path=entry["path"],
                        status="outside declared stage scope",
                        stages=entry["stages"],
                    )
                )
                continue
            relative = Path(entry["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or any(
                    part in {".git", ".ccache", ".normal-validation-cache", ".env"}
                    for part in relative.parts
                )
            ):
                raise ValueError(
                    "declared source input must be a scoped workspace source path"
                )
            path = workspace / relative
            if workspace.resolve() not in path.resolve().parents:
                raise ValueError("declared source input escapes workspace")
            expected = entry["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("declared source input needs an explicit SHA256")
            receipt = file_identity(path, fingerprint)
            receipt.update(
                kind=kind,
                declared_sha256=expected,
                matches_declared_sha256=receipt.get("sha256") == expected,
            )
            valid = valid and receipt["matches_declared_sha256"]
            receipts.append(receipt)
    return dict(
        manifest=str(manifest),
        manifest_sha256=fingerprint(manifest),
        stage=stage,
        valid=valid,
        files=receipts,
        scope="explicit reviewed source/patch paths only, including untracked copied headers; no untracked-tree or secret sweep",
    )


def observed_rtc_process(process, command):
    """Narrow opt-in observer: metadata only, no live library hashing or general env dump."""
    if not command or Path(command[0]).name not in {
        "rocfft_aot_helper",
        "rocfft_rtc_helper",
    }:
        return None
    row = dict(
        command_executable=command[0], knobs={}, loaded_shared_libraries=[], errors=[]
    )
    try:
        environment = process.environ()
        row["knobs"] = {key: environment.get(key) for key in RTC_KNOBS}
        row["loader_environment"] = {
            key: environment.get(key) for key in ["LD_LIBRARY_PATH", "LD_PRELOAD"]
        }
        row["executable"] = process.exe()
    except Exception as error:
        row["errors"].append("environment/executable: " + type(error).__name__)
    try:
        paths = {
            mapping.path
            for mapping in process.memory_maps(grouped=True)
            if Path(mapping.path).is_absolute() and (".so" in Path(mapping.path).name)
        }
        for path in sorted(paths):
            item = dict(path=path)
            try:
                stat = Path(path).stat()
                item.update(
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    device=stat.st_dev,
                    inode=stat.st_ino,
                )
            except OSError as error:
                item["error"] = type(error).__name__
            row["loaded_shared_libraries"].append(item)
    except Exception as error:
        row["errors"].append("memory maps: " + type(error).__name__)
    row["scope"] = (
        "sampled process snapshot; short-lived helpers may be missed; Linux environ may omit post-exec setenv, while RTC child exec inherits parent overrides; live provider contents not read"
    )
    return row


def observed_provider_identities(profile, fingerprint):
    records, errors = {}, []
    if not profile.is_file():
        return dict(status="no live RTC capture available", files=[], errors=[])
    with profile.open(encoding="utf-8") as stream:
        for line in stream:
            if '"rtc_process_inputs"' not in line:
                continue
            row = json.loads(line)
            for entry in row.get("receipt", {}).get("loaded_shared_libraries", []):
                path = entry["path"]
                if path not in records:
                    record = file_identity(Path(path), fingerprint)
                    record["observed_stats"] = []
                    records[path] = record
                records[path]["observed_stats"].append(
                    {
                        key: entry.get(key)
                        for key in ["bytes", "mtime_ns", "device", "inode"]
                    }
                )
    for path, record in records.items():
        try:
            stat = Path(path).stat()
            current = dict(
                bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev,
                inode=stat.st_ino,
            )
            record["postbuild_stat_matches_all_observations"] = all(
                item == current for item in record["observed_stats"]
            )
        except OSError as error:
            errors.append(dict(path=path, error=type(error).__name__))
            record["postbuild_stat_matches_all_observations"] = False
    return dict(
        status="observed provider paths hashed outside build command",
        files=list(records.values()),
        errors=errors,
        scope="positive sampled loaded-provider evidence; no completeness claim for missed processes/dlopen intervals",
    )
