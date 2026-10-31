# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Read-only COMM-stage evidence and reviewed rocSHMEM link verification.

Link argument/binary checks are reused from the controlled link-only diagnostic
333ebe0b2. Their functions do not build or reconfigure any product target.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("tools/rocshmem_info", "tests/functional_tests/rocshmem_functional_tests")
ARCHITECTURES = "gfx900 gfx90c gfx906 gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1170 gfx1171 gfx1200 gfx1201 gfx1250-strict gfx1250".split()
RCCL_ARCHITECTURES = [
    a
    for a in ARCHITECTURES
    if a not in {"gfx1103", "gfx1150", "gfx1152", "gfx1153", "gfx1170", "gfx1171"}
]
PROJECTS = {
    "rccl": "comm-libs/rccl/build",
    "rocshmem": "comm-libs/rocshmem/build",
    "rccl-tests": "comm-libs/rccl-tests/build",
    "rccl-translate": "comm-libs/rccl-translate/build",
    "therock-fmt": "third-party/fmt/build",
    "therock-googletest": "third-party/googletest/build",
    "therock-yaml-cpp": "third-party/yaml-cpp/build",
}


def open_archive_for_write(*args, **kwargs):
    # Declared production dependency; argument/metadata unit tests need no codec.
    from _therock_utils.archive_util import open_archive_for_write as open_archive

    return open_archive(*args, **kwargs)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RuntimeError("Link replay runtime budget exhausted")


def require_rock_ci() -> None:
    if (
        os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_REPOSITORY") != "ROCm/TheRock"
    ):
        raise RuntimeError("Full link replay executes only in ROCm/TheRock CI")
    current = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if current != os.environ["GITHUB_SHA"]:
        raise RuntimeError("Checkout does not match triggering TheRock SHA")


def sha256(path: Path, deadline: float) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_deadline(deadline)
            hashed.update(chunk)
    return hashed.hexdigest()


def driver_arguments(command: str, target: str) -> list[str]:
    args = shlex.split(command)
    if args[:2] == [":", "&&"] and args[-2:] == ["&&", ":"]:
        args = args[2:-2]
    if (
        not args
        or Path(args[0]).name != "clang++"
        or any(a in ["&&", ";", "|", ">"] for a in args)
    ):
        raise RuntimeError("Expected one generated clang++ link edge")
    if "-c" in args or "-o" not in args or args[args.index("-o") + 1] != target:
        raise RuntimeError("Unexpected generated output or compile action")
    if any(
        a.lower().endswith((".cpp", ".cc", ".cxx", ".hip", ".cu", ".c")) for a in args
    ):
        raise RuntimeError(
            "Measured native link edge must not recompile source translation units"
        )
    if any(a.startswith("@") for a in args):
        raise RuntimeError(
            "Generated link response file was not expanded by Ninja compdb -x"
        )
    arches = [
        a.removeprefix("--offload-arch=")
        for a in args
        if a.startswith("--offload-arch=")
    ]
    if arches != ARCHITECTURES or "--hip-link" not in args or "-fgpu-rdc" not in args:
        raise RuntimeError(
            "Link must retain the exact baseline 31-target RDC configuration"
        )
    if any(a in ["-v", "-###"] for a in args):
        raise RuntimeError("Timed replay must not add verbose/dry-run driver options")
    return args


def validate_candidate(baseline: list[str], candidate: list[str]) -> None:
    if any(a.startswith("--offload-jobs=") for a in baseline):
        raise RuntimeError(
            "Expected the recorded baseline with no offload-jobs override"
        )
    if (
        candidate.count("--offload-jobs=2") != 1
        or [a for a in candidate if a != "--offload-jobs=2"] != baseline
    ):
        raise RuntimeError(
            "Candidate changed more than the generated width-two link flag"
        )


def capture_edges(tree: Path) -> dict[str, list[str]]:
    rows = json.loads(
        subprocess.check_output(
            ["ninja", "-C", str(tree), "-t", "compdb", "-x"], text=True
        )
    )
    edges = {}
    for row in rows:
        if row.get("output") in TARGETS:
            target = row["output"]
            if target in edges:
                raise RuntimeError(
                    "Duplicate final edge in generated compilation database"
                )
            edges[target] = driver_arguments(row["command"], target)
    if set(edges) != set(TARGETS):
        raise RuntimeError("Missing expected final rocSHMEM links")
    return edges


def stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_logged(
    command: list[str], cwd: Path, log: Path, deadline: float, env=None
) -> float:
    check_deadline(deadline)
    started = time.monotonic()
    own_cpu = time.process_time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    before = None
    if sys.platform == "linux":
        import resource

        before = resource.getrusage(resource.RUSAGE_CHILDREN)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            dict(
                event="start",
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                log=str(log),
                command=command,
            )
        ),
        flush=True,
    )
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            while True:
                check_deadline(deadline)
                try:
                    code = process.wait(
                        timeout=min(30, max(1, deadline - time.monotonic()))
                    )
                    break
                except subprocess.TimeoutExpired:
                    print(
                        json.dumps(
                            dict(
                                event="running",
                                log=log.name,
                                wall_s=time.monotonic() - started,
                            )
                        ),
                        flush=True,
                    )
        except (subprocess.TimeoutExpired, RuntimeError):
            stop_group(process)
            raise RuntimeError(f"Command exceeded deadline; see {log}")
        finally:
            child = None
            if before is not None:
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                child = dict(
                    user_s=after.ru_utime - before.ru_utime,
                    system_s=after.ru_stime - before.ru_stime,
                )
            save(
                log.with_suffix(".timing.json"),
                dict(
                    command=command,
                    started_utc=started_utc,
                    ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    wall_s=time.monotonic() - started,
                    controller_cpu_s=time.process_time() - own_cpu,
                    waited_child_cpu=child,
                    exit_code=process.returncode,
                    scope="This command's wait chain; profiler-wrapped build includes profiler CPU here. Do not add to profile/phase totals.",
                ),
            )
    if code:
        stop_group(process)
        raise RuntimeError(f"Command failed with exit {code}; see {log}")
    return time.monotonic() - started


def explicit_inputs(tree: Path, edges: dict[str, list[str]]) -> list[Path]:
    paths = set()
    for args in edges.values():
        paths.add(Path(args[0]).resolve())
        for arg in args[1:]:
            if arg.startswith("-"):
                continue
            path = Path(arg) if Path(arg).is_absolute() else tree / arg
            if path.is_file() and any(
                token in path.name for token in [".o", ".a", ".so"]
            ):
                paths.add(path.absolute())
    return sorted(paths)


def input_manifest(paths: list[Path], deadline: float) -> list[dict[str, Any]]:
    return [
        dict(
            path=str(p),
            resolved=str(p.resolve()),
            size=p.stat().st_size,
            sha256=sha256(p, deadline),
        )
        for p in paths
    ]


def retain_outputs(tree: Path, output: Path, deadline: float) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "rocm-systems/shared/kpack/python"))
    from rocm_kpack.binutils import BundledBinary

    rows = {}
    with open_archive_for_write(
        output / "linked-binaries.tar.zst", "zstd", 3
    ) as archive:
        for target in TARGETS:
            path = tree / target
            row = dict(bytes=path.stat().st_size, sha256=sha256(path, deadline))
            archive.add(path, arcname=target, recursive=False)
            rows[target] = row
    save(output / "output-sha256-before-inspection.json", rows)
    for target in TARGETS:
        path = tree / target
        row = rows[target]
        images = []
        with BundledBinary(path).unbundle() as contents:
            for target_id, filename in contents.target_list:
                if not target_id.startswith("hip"):
                    continue
                image = contents.dest_dir / filename
                images.append(
                    dict(
                        target_id=target_id,
                        architecture=target_id.rpartition("--")[2],
                        bytes=image.stat().st_size,
                        sha256=sha256(image, deadline),
                    )
                )
        if sorted(image["architecture"] for image in images) != sorted(ARCHITECTURES):
            raise RuntimeError(
                "Retained native binary does not contain every full architecture ID"
            )
        row["images"] = sorted(images, key=lambda image: image["target_id"])
        for option in ["-h", "-SW", "-d", "-n", "-sW"]:
            run_logged(
                ["readelf", option, str(path)],
                tree,
                output / (Path(target).name + option.replace("-", "_") + ".txt"),
                deadline,
            )
    return rows


def cache_values(path: Path) -> dict[str, str]:
    import re

    return {
        m[1]: m[2]
        for line in path.read_text().splitlines()
        if (m := re.match(r"([^/#][^:]*):[^=]+=(.*)", line))
    }


def inventory_files(tree: Path, output: Path, deadline: float) -> dict:
    """Hash generated source/headers and all object/static-library inputs.

    SDK/imported files are separate provenance. No timestamps/inodes enter the
    content inventory; raw modes and symlink targets remain explicit.
    """
    import stat

    records = []
    for path in sorted(tree.rglob("*")):
        check_deadline(deadline)
        if (
            path.suffix.lower()
            not in {
                ".o",
                ".obj",
                ".a",
                ".h",
                ".hpp",
                ".hxx",
                ".inc",
                ".cuh",
                ".cpp",
                ".cc",
                ".cu",
                ".hip",
                ".hsaco",
                ".co",
                ".bc",
                ".kpack",
            }
            or not path.is_file()
        ):
            continue
        record = dict(
            path=str(path.relative_to(tree)), mode=stat.S_IMODE(path.lstat().st_mode)
        )
        if path.is_symlink():
            record.update(kind="symlink", target=os.readlink(path))
        else:
            record.update(
                kind="file", bytes=path.stat().st_size, sha256=sha256(path, deadline)
            )
        records.append(record)
    output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    return dict(
        entries=len(records),
        sha256=sha256(output, deadline),
        bytes=sum(r.get("bytes", 0) for r in records),
    )


def collect_phase_cpu(build: Path, output: Path) -> dict:
    rows = []
    errors = []
    for project in PROJECTS:
        for phase in ["configure", "build", "install"]:
            source = build / "logs" / f"{project}_{phase}.log"
            if not source.is_file():
                errors.append(f"Missing phase log:{project}/{phase}")
                continue
            records = []
            with source.open(errors="replace") as stream:
                for number, line in enumerate(stream, 1):
                    if line.startswith("CPU_PROFILE\t"):
                        record = json.loads(line.split("\t", 1)[1])
                        record.update(
                            project=project,
                            phase=phase,
                            source_file=str(source.relative_to(build)),
                            line=number,
                        )
                        records.append(record)
            if len(records) != 1:
                errors.append(
                    f"Expected one CPU record:{project}/{phase},got{len(records)}"
                )
            rows.extend(records)
    expected = {
        f"{project}_{phase}.log"
        for project in PROJECTS
        for phase in ["configure", "build", "install"]
    }
    for phase in ["configure", "build", "install"]:
        for path in (build / "logs").glob(f"*_{phase}.log"):
            if path.name not in expected:
                with path.open(errors="replace") as stream:
                    if any(line.startswith("CPU_PROFILE\t") for line in stream):
                        errors.append(
                            f"Unexpected additional measured project phase:{path.name}"
                        )
    exact = sum(r.get("child_total_s") or 0 for r in rows)
    summary_path = output / "profile/summary.json"
    stage = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    accounted = stage.get("posix_waited_child_cpu") or {}
    stage_cpu = (
        sum(accounted.get(k, 0) for k in ["user_s", "system_s"]) if accounted else None
    )
    result = dict(
        records=rows,
        record_count=len(rows),
        errors=errors,
        phase_waited_child_cpu_s=exact,
        stage_waited_child_cpu_s=stage_cpu,
        residual_unallocated_cpu_s=stage_cpu - exact if stage_cpu is not None else None,
        scope="21 disjoint project phase wait chains; not pure compilation; no sampled CPU rescaling; residual unallocated",
    )
    save(output / "phase-cpu.json", result)
    if errors or any(
        r.get("exit_code") != 0 or r.get("child_total_s") is None for r in rows
    ):
        raise RuntimeError(
            "Missing/failed exact phase CPU records; partial evidence retained"
        )
    return result


def collect_graphs(build: Path, output: Path, width: int, deadline: float) -> dict:
    """Retain all edges, including RCCL custom commands bypassing compdb launchers."""
    rows = []
    errors = []
    directory = output / "graphs"
    directory.mkdir(exist_ok=True)
    # Copy hidden Ninja logs through a tar so permission/hidden-file upload
    # defaults do not discard the original native evidence.
    with open_archive_for_write(output / "ninja-graphs.tar.zst", "zstd", 3) as archive:
        trees = {
            "root": build,
            **{name: build / relative for name, relative in PROJECTS.items()},
        }
        for name, tree in trees.items():
            try:
                target = directory / name
                target.mkdir(exist_ok=True)
                for filename in [
                    "CMakeCache.txt",
                    "build.ninja",
                    "compile_commands.json",
                    ".ninja_log",
                    "CTestTestfile.cmake",
                ]:
                    source = tree / filename
                    if source.is_file():
                        archive.add(
                            source, arcname=f"{name}/{filename}", recursive=False
                        )
                        if filename != ".ninja_log":
                            shutil.copyfile(source, target / filename)
                for source in [
                    tree / "CMakeFiles/rules.ninja",
                    *(tree / "CMakeFiles").glob("*/*Compiler.cmake"),
                ]:
                    if source.is_file():
                        relative = source.relative_to(tree)
                        archive.add(
                            source,
                            arcname=f"{name}/{relative.as_posix()}",
                            recursive=False,
                        )
                        destination = target / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
                commands = json.loads(
                    subprocess.check_output(
                        ["ninja", "-C", str(tree), "-t", "compdb", "-x"], text=True
                    )
                )
                save(target / "all-edges.json", commands)
                (target / "ninja-commands.txt").write_bytes(
                    subprocess.check_output(
                        ["ninja", "-C", str(tree), "-t", "commands"]
                    )
                )
                (target / "ninja-graph.dot").write_bytes(
                    subprocess.check_output(["ninja", "-C", str(tree), "-t", "graph"])
                )
                (target / "ninja-targets.txt").write_bytes(
                    subprocess.check_output(
                        ["ninja", "-C", str(tree), "-t", "targets", "all"]
                    )
                )
                values = cache_values(tree / "CMakeCache.txt")
                if name == "rocshmem":
                    if values.get("ROCSHMEM_NATIVE_LINK_JOBS") != str(width):
                        raise RuntimeError("Nested native link width drift")
                    if values.get("HIP_CLANG_NUM_PARALLEL_JOBS", "1") != "1":
                        raise RuntimeError("HIP-wide compile knob changed")
                    edges = capture_edges(tree)
                    for argv in edges.values():
                        flags = [a for a in argv if a.startswith("--offload-jobs=")]
                        if flags != ([] if width == 1 else ["--offload-jobs=2"]):
                            raise RuntimeError("Unexpected native link override")
                    save(output / "final-link-edges.json", edges)
                if name in ["rccl", "rocshmem"]:
                    expected = RCCL_ARCHITECTURES if name == "rccl" else ARCHITECTURES
                    if values.get("GPU_TARGETS", "").split(";") != expected:
                        raise RuntimeError(f"{name} architecture list drift")
                if name != "root":
                    inventory_files(
                        tree, target / "generated-and-object-manifest.jsonl", deadline
                    )
                rows.append(
                    dict(
                        project=name,
                        tree=str(tree.relative_to(build)),
                        edges=len(commands),
                        status="passed",
                    )
                )
            except Exception as error:
                errors.append(dict(project=name, error=repr(error)))
    save(output / "graph-collection.json", dict(projects=rows, errors=errors))
    if errors:
        raise RuntimeError(
            "One or more graph/object collectors failed; independent evidence retained"
        )
    return dict(projects=rows)


def retain_link_inputs(build: Path, output: Path, deadline: float) -> None:
    tree = build / PROJECTS["rocshmem"]
    edges = capture_edges(tree)
    inputs = explicit_inputs(tree, edges)
    tools = Path(edges[TARGETS[0]][0]).parent
    for name in ["ld.lld", "clang-linker-wrapper", "clang-offload-bundler"]:
        inputs.append(tools / name)
    inputs.extend(tools.glob("*.cfg"))
    inputs.extend((tools.parent / "lib").glob("libLLVM*.so*"))
    inputs.extend((tools.parent / "lib").glob("libclang-cpp*.so*"))
    for argv in edges.values():
        for arg in argv:
            if arg.startswith("--hip-device-lib-path="):
                inputs.extend(Path(arg.split("=", 1)[1]).rglob("*.bc"))
    inputs = sorted(set(inputs))
    save(output / "link-inputs.json", input_manifest(inputs, deadline))
    with open_archive_for_write(
        output / "local-link-inputs.tar.zst", "zstd", 3
    ) as archive:
        for path in inputs:
            if tree in path.parents:
                archive.add(
                    path, arcname=path.relative_to(tree).as_posix(), recursive=False
                )


def retain_rccl(build: Path, output: Path, deadline: float) -> dict:
    sys.path.insert(0, str(ROOT / "rocm-systems/shared/kpack/python"))
    from rocm_kpack.binutils import BundledBinary

    tree = build / PROJECTS["rccl"]
    paths = [p for p in tree.rglob("librccl.so*") if p.is_file() and not p.is_symlink()]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one native RCCL library,found{len(paths)}")
    path = paths[0]
    # Retain native bytes BEFORE the independent image parser can fail.
    with open_archive_for_write(output / "rccl-native.tar.zst", "zstd", 3) as archive:
        archive.add(path, arcname=path.relative_to(tree).as_posix(), recursive=False)
    result = dict(
        path=str(path.relative_to(tree)),
        bytes=path.stat().st_size,
        sha256=sha256(path, deadline),
        images=[],
    )
    save(output / "rccl-native.json", result)
    with BundledBinary(path).unbundle() as contents:
        for target_id, filename in contents.target_list:
            if target_id.startswith("hip"):
                image = contents.dest_dir / filename
                result["images"].append(
                    dict(
                        target_id=target_id,
                        architecture=target_id.rpartition("--")[2],
                        bytes=image.stat().st_size,
                        sha256=sha256(image, deadline),
                    )
                )
    result["images"].sort(key=lambda r: (r["target_id"], r["sha256"]))
    save(output / "rccl-native.json", result)
    if {r["architecture"] for r in result["images"]} != set(RCCL_ARCHITECTURES):
        raise RuntimeError("RCCL native images do not preserve25full target IDs")
    return result
