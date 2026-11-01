# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Read-only generated-tool, RTC/cache and explicitly declared source receipts.

The normal controller guards CI execution. These functions are also exercised by
small inert file fixtures; no compiler or cache behavioral experiment is needed.
"""

import hashlib
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request


def scoped_git(workspace, repository, *args):
    """Trust only this verified repository, only for this child invocation."""
    workspace, repository = workspace.resolve(strict=True), repository.resolve(
        strict=True
    )
    if repository != workspace and workspace not in repository.parents:
        raise ValueError("Git repository escapes verified workspace")
    if not (repository / ".git").exists():
        raise ValueError("Expected checked-out Git repository")
    return [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
        *args,
    ]


def download_checkout_archive(sha):
    """Public immutable commit URL; no token, new dependency or extraction."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Full CI commit SHA required")
    url = f"https://codeload.github.com/ROCm/TheRock/tar.gz/{sha}"
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.geturl() != url:
            raise ValueError("Unexpected commit-archive redirect")
        data = response.read(32 * 1024**2 + 1)
    if len(data) > 32 * 1024**2:
        raise ValueError("Commit archive exceeds compressed capture bound")
    return data


def compare_checkout_archive(workspace, sha, data, fingerprint):
    """Compare all archive-declared files, not an untracked/secret tree sweep."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or len(data) > 32 * 1024**2:
        raise ValueError("Invalid commit archive identity/size")
    workspace = workspace.resolve(strict=True)
    prefix = f"TheRock-{sha}"
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        decoded = stream.read(128 * 1024**2 + 1)
    if len(decoded) > 128 * 1024**2:
        raise ValueError("Commit archive exceeds decoded capture bound")
    rows, seen, total = [], set(), 0
    with tarfile.open(fileobj=io.BytesIO(decoded), mode="r|") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != prefix
                or any(p in ("..", ".git") for p in path.parts)
                or "\\" in member.name
                or len(seen) >= 10000
            ):
                raise ValueError("Unsafe or oversized commit archive inventory")
            relative = PurePosixPath(*path.parts[1:])
            name = relative.as_posix()
            if name in seen:
                raise ValueError("Duplicate commit archive member")
            seen.add(name)
            if member.isdir():
                continue
            if not relative.parts or not (member.isfile() or member.issym()):
                raise ValueError("Unsupported commit archive member")
            total += member.size
            if member.size > 16 * 1024**2 or total > 128 * 1024**2:
                raise ValueError("Commit archive exceeds decoded capture bound")
            local = workspace.joinpath(*relative.parts)
            if any(
                parent.is_symlink()
                for parent in local.parents
                if parent != workspace and workspace in parent.parents
            ):
                raise ValueError("Declared source has a symlink ancestor")
            row = dict(path=name, archive_mode=member.mode)
            if member.issym():
                if not local.is_symlink() or str(local.readlink()) != member.linkname:
                    raise ValueError(
                        f"Checkout symlink differs from commit archive: {name}"
                    )
                row.update(kind="symlink", link_target=member.linkname)
            else:
                if local.is_symlink() or not local.is_file():
                    raise ValueError(f"Checkout file missing/type changed: {name}")
                hashed = hashlib.sha256()
                stream = archive.extractfile(member)
                count = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hashed.update(chunk)
                    count += len(chunk)
                actual = fingerprint(local)
                executable = bool(local.stat().st_mode & stat.S_IXUSR)
                if (
                    count != member.size
                    or local.stat().st_size != count
                    or actual != hashed.hexdigest()
                    or executable != bool(member.mode & stat.S_IXUSR)
                ):
                    raise ValueError(
                        f"Checkout bytes/mode differ from commit archive: {name}"
                    )
                row.update(
                    kind="file", bytes=count, sha256=actual, executable=executable
                )
            rows.append(row)
        if any(decoded[archive.offset :]):
            raise ValueError("Unexpected trailing commit archive contents")
    if not rows:
        raise ValueError("Empty commit archive cannot verify a checkout")
    return dict(
        complete=True,
        verified_files=len(rows),
        logical_file_bytes=total,
        archive_sha256=hashlib.sha256(data).hexdigest(),
        files=rows,
        caps=dict(
            compressed_bytes=32 * 1024**2,
            decoded_bytes=128 * 1024**2,
            member_bytes=16 * 1024**2,
            members=10000,
        ),
        scope="All exact-commit archive-declared root files/symlinks and executable bits; untracked files, Git history/HEAD, submodule contents and dirty state are not measured",
    )


def gitless_consumer_source(workspace, env, fingerprint, fetch=None):
    """Only unchanged consumer checkouts may use an independently compared archive."""
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
        or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or env.get("GITHUB_JOB")
        not in ("test_wheels", "test_native_linux_packages_install")
    ):
        raise ValueError(
            "Missing Git is supported only for declared normal consumer jobs"
        )
    sha, run_id = env.get("GITHUB_SHA", ""), env.get("GITHUB_RUN_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not run_id.isdecimal():
        raise ValueError("Gitless source needs full CI commit and run identity")
    links = {}
    for key in ("REPO_URL", "PACKAGE_FIND_LINKS_URL"):
        value = env.get(key, "")
        if not value:
            continue
        parsed = urllib.parse.urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "therock-ci-artifacts.s3.amazonaws.com"
            or not parsed.path.startswith(f"/{run_id}-linux/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Consumer artifact provenance is not this CI run")
        links[key] = value
    if not links:
        raise ValueError(
            "Gitless consumer requires same-run producer artifact reference"
        )
    compared = compare_checkout_archive(
        workspace, sha, (fetch or download_checkout_archive)(sha), fingerprint
    )
    return dict(
        mode="git-unavailable; exact-commit archive comparison",
        measured_git_head=None,
        ci_declared_sha=sha,
        checkout_archive=compared,
        archive_url=f"https://codeload.github.com/ROCm/TheRock/tar.gz/{sha}",
        producer_references=dict(
            run_id=run_id,
            run_attempt=env.get("GITHUB_RUN_ATTEMPT"),
            artifact_urls=links,
            scope="CI-bound same-run artifact references; producer Git/SDK/output receipts must be reviewed separately",
        ),
        repositories=[],
        working_changes=None,
        submodule_status=None,
        git_unavailable_scope="No Git HEAD/history/dirty/submodule claim; verified root files only",
    )


def capture_source(workspace, gitlinks, fingerprint, env):
    """Measured Git source, or a strictly scoped verified consumer archive."""
    workspace = workspace.resolve(strict=True)
    if shutil.which("git") is None:
        result = gitless_consumer_source(workspace, env, fingerprint)
        result["recorded_product_gitlinks"] = gitlinks
        result["recorded_product_gitlinks_scope"] = (
            "Declared expected producer pins; submodule contents not verified in this Gitless consumer"
        )
        return result

    def checked(repository, *args, text=True):
        command = scoped_git(workspace, repository, *args)
        return subprocess.run(command, capture_output=True, text=text, check=True)

    top = checked(workspace, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(top).resolve() != workspace:
        raise ValueError("Git toplevel does not match verified workspace")
    source = dict(
        mode="measured-git", repositories=[], recorded_product_gitlinks=gitlinks
    )
    for name, args in {
        "checkout_sha": ["rev-parse", "HEAD"],
        "working_changes": ["status", "--porcelain", "--untracked-files=no"],
        "submodule_status": ["submodule", "status", "--recursive"],
        "checkout_tree": ["ls-tree", "-r", "HEAD"],
    }.items():
        result = checked(workspace, *args)
        source[name] = dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
    head = source["checkout_sha"]["stdout"].strip()
    if env.get("GITHUB_SHA") and env["GITHUB_SHA"] != head:
        raise ValueError("Measured Git HEAD differs from CI triggering SHA")
    source.update(measured_git_head=head, ci_declared_sha=env.get("GITHUB_SHA"))
    for relative in ["."] + list(gitlinks):
        repository = workspace / relative
        if not (repository / ".git").exists():
            continue
        head = checked(repository, "rev-parse", "HEAD").stdout.strip()
        diff = checked(repository, "diff", "--binary", "HEAD", text=False).stdout
        names = checked(repository, "diff", "--name-only", "-z", "HEAD").stdout.split(
            "\0"
        )
        files = []
        for name in names:
            path = repository / name
            if name and path.is_file() and not path.is_symlink():
                files.append(
                    dict(path=name, sha256=fingerprint(path), bytes=path.stat().st_size)
                )
        row = dict(
            path=relative,
            head=head,
            diff_sha256=hashlib.sha256(diff).hexdigest(),
            changed_files=files,
            is_shallow=checked(
                repository, "rev-parse", "--is-shallow-repository"
            ).stdout.strip(),
        )
        if relative == "rocm-systems":
            commit = checked(
                repository, "log", "-1", "--format=%H", "--", "projects/clr"
            )
            row["clr_last_directory_commit"] = dict(
                returncode=commit.returncode, stdout=commit.stdout.strip()
            )
        source["repositories"].append(row)
    return source


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


def visible_cgroups():
    """Resolve visible cgroup-v2 ancestors using only the Python standard library."""

    def read(path):
        try:
            return Path(path).read_text().strip()
        except (OSError, UnicodeError):
            return None

    membership, mounts = read("/proc/self/cgroup"), read("/proc/self/mountinfo")
    if not membership or not mounts:
        return [], {"status": "unavailable_or_non_linux"}
    for line in membership.splitlines():
        hierarchy, controllers, member = line.split(":", 2)
        if hierarchy != "0" or controllers:
            continue
        for line in mounts.splitlines():
            left, _, right = line.partition(" - ")
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            root, mount = Path(fields[3].replace("\\040", " ")), Path(
                fields[4].replace("\\040", " ")
            )
            try:
                relative = Path(member).relative_to(root)
            except ValueError:
                if member == "/":
                    relative = Path(".")
                else:
                    continue
            current, paths = mount / relative, []
            while current == mount or mount in current.parents:
                paths.append(current)
                if current == mount:
                    return paths, {
                        "status": "v2_visible_hierarchy",
                        "membership": member,
                        "mount_root": str(root),
                        "mountpoint": str(mount),
                    }
                current = current.parent
    return [], {"status": "v1_or_unresolved", "membership": membership}


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


def sdk_inputs(build, fingerprint, max_entries=100000, max_bytes=16 * 1024**3):
    rows, roots, errors, limits_reached = [], [], [], []
    visited, selected_bytes, complete = 0, 0, True
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
        "rocgdb",
        "gdb",
    }
    scan_roots = [(relative, "nested-dist", True) for relative in SDK_ROOTS]
    scan_roots += [("math-libs/rocFFT", "runtime-helper", True)]
    scan_roots += [
        (relative, "installed-tools", False)
        for relative in ["bin", "llvm/bin", "lib/llvm/bin"]
    ]
    scan_roots += [
        (relative, "installed-providers", False) for relative in ["lib", "lib64"]
    ]
    scan_roots += [
        (relative, "installed-providers", True)
        for relative in ["llvm/lib", "lib/llvm/lib"]
    ]
    scan_roots += [("lib/llvm/amdgcn/bitcode", "installed-bitcode", True)]
    for relative, layout, recursive in scan_roots:
        root = build / relative
        roots.append(dict(path=str(root), exists=root.is_dir(), layout=layout))
        if not root.is_dir():
            continue

        def walk_error(error):
            errors.append(dict(path=str(error.filename), error=str(error)))

        for directory, directories, names in os.walk(
            root, followlinks=False, onerror=walk_error
        ):
            directories[:] = sorted(directories) if recursive else []
            for name in sorted(names):
                visited += 1
                if visited > max_entries:
                    complete = False
                    limits_reached.append("max_entries")
                    break
                p = Path(directory) / name
                tool = name.removesuffix(".exe") in executables or name.startswith(
                    "clang-"
                )
                library = p.suffix in {".so", ".dll", ".dylib"} or ".so." in name
                selected = tool or library or p.suffix == ".bc"
                if layout == "runtime-helper":
                    selected = name in {"rocfft_rtc_helper", "rocfft_aot_helper"}
                elif layout == "installed-tools":
                    selected = name.removesuffix(".exe") in executables or bool(
                        re.fullmatch(r"clang-\d+(?:\.exe)?", name)
                    )
                elif layout == "installed-providers":
                    selected = library and name.startswith(
                        (
                            "libLLVM",
                            "libclang",
                            "libamd_comgr",
                            "libhiprtc",
                            "libamdhip64",
                            "libhsa-runtime64",
                        )
                    )
                elif layout == "installed-bitcode":
                    selected = p.suffix == ".bc"
                if not selected:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                if selected_bytes + size > max_bytes:
                    complete = False
                    limits_reached.append("max_selected_logical_bytes")
                    break
                row = file_identity(p, fingerprint)
                row.update(
                    sdk_root=(
                        "rocFFT active runtime helper"
                        if layout == "runtime-helper"
                        else relative
                    ),
                    layout=layout,
                )
                row["entry_kind"] = "symlink" if p.is_symlink() else "file"
                if p.is_symlink():
                    row["link_target"] = os.readlink(p)
                if row["status"] != "file":
                    errors.append(
                        dict(path=str(p), error=row.get("error", row["status"]))
                    )
                selected_bytes += size
                rows.append(row)
            if not complete:
                break
        if not complete:
            break
    return dict(
        roots=roots,
        files=rows,
        complete=complete and not errors,
        errors=errors,
        limits=dict(max_entries=max_entries, max_selected_logical_bytes=max_bytes),
        limits_reached=limits_reached,
        scanned_entries=visited,
        selected_logical_bytes=selected_bytes,
        tool_presence={
            name: [
                row["path"]
                for row in rows
                if Path(row["path"]).name.removesuffix(".exe") == name
            ]
            for name in [
                "rocgdb",
                "clang",
                "clang++",
                "clang-linker-wrapper",
                "rocfft_rtc_helper",
            ]
        },
        scope="Selected SDK candidate files in nested dist and installed views; installed scans select tools, LLVM bitcode and compiler/RTC/HSA providers, excluding unrelated math libraries. Completeness covers this selection only; absent roots/tools remain explicit. Actual active loader selection is separate.",
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


def workspace_storage_identity(workspace, sysfs_root=Path("/sys/dev/block")):
    """Narrow device identity only; no mount or unrelated environment sweep."""
    row = dict(path=str(workspace))
    try:
        device = workspace.stat().st_dev
        row["st_dev"] = device
        if not hasattr(os, "major"):
            row["status"] = "device major/minor unavailable on this platform"
            return row
        major, minor = os.major(device), os.minor(device)
        entry = sysfs_root / f"{major}:{minor}"
        row.update(major=major, minor=minor, sysfs_path=str(entry))
        if entry.exists():
            row.update(
                status="visible sysfs block device",
                resolved_sysfs_path=str(entry.resolve()),
            )
        else:
            row["status"] = (
                "no sysfs block device for workspace st_dev; backing storage unresolved"
            )
    except OSError as error:
        row.update(status="unavailable", error=str(error))
    return row


def io_limit_files(paths):
    """Every visible cgroup io.max, including explicit absent/denied receipts."""
    rows = []
    for parent in paths:
        path = parent / "io.max"
        try:
            rows.append(dict(path=str(path), status="read", text=path.read_text()))
        except OSError as error:
            rows.append(dict(path=str(path), status="unavailable", error=str(error)))
    return dict(
        files=rows,
        scope="visible cgroup hierarchy only; hidden ancestor limits remain unknown",
    )


def declared_sources(
    workspace, manifest, case, fingerprint, stage=None, family=None, product="none"
):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    receipts, valid = [], True
    for kind in ["patches", "copied_inputs"]:
        for entry in data.get(kind, []):
            if case not in entry.get("cases", ["baseline", "candidate"]):
                continue
            if "products" in entry and product not in entry["products"]:
                receipts.append(
                    dict(
                        kind=kind,
                        path=entry["path"],
                        status="outside declared product scope",
                        products=entry["products"],
                    )
                )
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
            if "families" in entry and (family or "").casefold() not in {
                value.casefold() for value in entry["families"]
            }:
                receipts.append(
                    dict(
                        kind=kind,
                        path=entry["path"],
                        status="outside declared family scope",
                        families=entry["families"],
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
        family=family,
        product=product,
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
