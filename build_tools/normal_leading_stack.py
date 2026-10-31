#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Default-off leading-stack source support; no workflow dispatch or build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = "a10dca087065a82f15ea89dc7d6fdd3c6bb2c246"
PRODUCT = "5b7cb88373362ab80dc0f19605915e534f4158aa"
PRODUCT_SELECTION = "leading-build-stack"
PAYLOADS = Path("patches/build-profile/normal-stack")
ROOT_CMAKE = "comm-libs/CMakeLists.txt"
ELF_MODULE = "build_tools/packaging/linux/runpath_to_rpath.py"
SYSTEM_FILES = (
    "projects/rocshmem/CMakeLists.txt",
    "projects/rocshmem/tools/CMakeLists.txt",
    "projects/rocshmem/tests/functional_tests/CMakeLists.txt",
)
TARGETS = ("tools/rocshmem_info", "tests/functional_tests/rocshmem_functional_tests")
ARCHES = "gfx900 gfx90c gfx906 gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1170 gfx1171 gfx1200 gfx1201 gfx1250-strict gfx1250".split()
PATCH_HASHES = {
    "bounded-elf-probe.patch": "777edbf4581659833190bc047c4e6314ba25381b7bf72f0bfd07b7a1a05b98b1",
    "rocshmem-native-link-control.patch": "8b21193ef9f4287f522b8bd8648e7487a183165422460e6ffa8eb84e02676746",
    "therock-native-link-control.patch": "e9c8de9b3f3804fe88186a4b7453bf4fa43c6388d8524171347a65d1361e621a",
}
ORIGINAL_HASHES = {
    "projects/rocshmem/CMakeLists.txt": "074ae96c5748cc8819fd3bc03df0b3f2d57a24d44c01e98d1ff63e435fb01baf",
    "projects/rocshmem/tools/CMakeLists.txt": "a05c5437702afdd20a79bcfde6e74e5efbd6d428cc96848241cff0348ad8fd24",
    "projects/rocshmem/tests/functional_tests/CMakeLists.txt": "ed60e8cae07bb4c6b5b2b4207085e1f49d1bb9e24a39e01a76d4efbbb16a88c3",
    "comm-libs/CMakeLists.txt": "4556ed7961f3c941d99e753f91e2e491dca22820133545ffa05767ff03cbec4c",
    "build_tools/packaging/linux/runpath_to_rpath.py": "251718ab87ed42be9dc425870e1d88d73fde388455e76c1aae5307079314ee54",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def regular(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Expected regular single-link source/output: {path}")


def git(repository, *args):
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def identity(repository):
    return dict(
        head=git(repository, "rev-parse", "HEAD"),
        branch=git(repository, "branch", "--show-current"),
        shallow=git(repository, "rev-parse", "--is-shallow-repository"),
        head_and_parents=git(repository, "log", "-1", "--format=%H %P"),
    )


def payload_path(workspace, name):
    path = workspace / PAYLOADS / name
    regular(path)
    if digest(path) != PATCH_HASHES[name]:
        raise ValueError("Reviewed LF patch payload changed")
    return path


def patch(repository, payload):
    regular(payload)
    subprocess.run(
        ["git", "-C", str(repository), "apply", "--check", str(payload)], check=True
    )
    subprocess.run(["git", "-C", str(repository), "apply", str(payload)], check=True)


def require_original(repository, names):
    for name in names:
        path = repository / name
        regular(path)
        if digest(path) != ORIGINAL_HASHES[name]:
            raise ValueError(
                f"Selected original source differs from pinned product: {name}"
            )
        if git(repository, "diff", "--name-only", "HEAD", "--", name):
            raise ValueError(f"Selected source is already dirty: {name}")


def require_case(case):
    if case not in ("baseline", "candidate"):
        raise ValueError("Explicit baseline/candidate required")


def controls(env, event, case, command, platform):
    require_case(case)
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
        or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or platform != "linux"
        or env.get("NORMAL_VALIDATION") != "true"
        or env.get("RELEASE_TYPE") != "ci"
    ):
        raise ValueError(
            "Leading-stack support executes only in normal Linux TheRock CI"
        )
    if not env.get("GITHUB_RUN_ID") or not env.get("GITHUB_RUN_ATTEMPT"):
        raise ValueError("Current run/attempt identity required")
    inputs = event.get("inputs", {})
    if (
        str(inputs.get("profile_normal_validation")).lower() != "true"
        or inputs.get("profile_normal_product") != PRODUCT_SELECTION
        or inputs.get("profile_normal_case", "baseline") != case
    ):
        raise ValueError("Current normal product/case does not match this invocation")
    if env.get("HIP_CLANG_NUM_PARALLEL_JOBS") not in (None, "", "1"):
        raise ValueError("Global HIP compilation width must remain unchanged")
    if command.endswith("comm"):
        return env.get("STAGE_NAME") == "comm-libs" and not env.get("AMDGPU_FAMILIES")
    return env.get("NATIVE_PACKAGE_TYPE") in ("deb", "rpm")


def stock_patch_state(workspace, expected_base):
    marker = workspace / ".rocm-systems.smrev"
    regular(marker)
    series = []
    fingerprint = hashlib.sha1(usedforsecurity=False)
    for path in sorted(
        (workspace / "patches/amd-mainline/rocm-systems").glob("*.patch")
    ):
        regular(path)
        fingerprint.update(path.read_bytes())
        series.append(
            dict(path=path.relative_to(workspace).as_posix(), sha256=digest(path))
        )
    text = marker.read_text(encoding="utf-8")
    if (
        not series
        or len(text.splitlines()) != 2
        or text.splitlines()[1] != expected_base + "+PATCHED:" + fingerprint.hexdigest()
    ):
        raise ValueError(
            "Original stock patched-source .smrev does not match its series"
        )
    return dict(
        path=str(marker), sha256=digest(marker), text=text, stock_patch_series=series
    )


def state_common(workspace, case, env):
    require_case(case)
    return dict(
        schema_version=1,
        case=case,
        run_id=env.get("GITHUB_RUN_ID"),
        run_attempt=env.get("GITHUB_RUN_ATTEMPT"),
        root_identity=identity(workspace),
    )


def apply_comm(workspace, output, case, env, *, expected_base=SYSTEMS):
    systems = workspace / "rocm-systems"
    state = state_common(workspace, case, env)
    if (output / "source-control.json").exists():
        raise ValueError("Refuse to adopt existing COMM product state")
    subprocess.run(
        [
            "git",
            "-C",
            str(systems),
            "merge-base",
            "--is-ancestor",
            expected_base,
            "HEAD",
        ],
        check=True,
    )
    require_original(systems, SYSTEM_FILES)
    require_original(workspace, [ROOT_CMAKE])
    stock = stock_patch_state(workspace, expected_base)
    systems_identity = identity(systems)
    root_payload = payload_path(workspace, "therock-native-link-control.patch")
    systems_payload = payload_path(workspace, "rocshmem-native-link-control.patch")
    # Fail both applicability checks before changing either repository.
    for repository, payload in [(workspace, root_payload), (systems, systems_payload)]:
        subprocess.run(
            ["git", "-C", str(repository), "apply", "--check", str(payload)], check=True
        )
    patch(systems, systems_payload)
    patch(workspace, root_payload)
    if case == "candidate":
        path = workspace / ROOT_CMAKE
        text = path.read_text(encoding="utf-8")
        original = 'set(THEROCK_ROCSHMEM_NATIVE_LINK_JOBS "1" CACHE STRING'
        if text.count(original) != 1:
            raise ValueError("Expected exactly one reviewed ROOT default-width field")
        path.write_text(
            text.replace(original, original.replace('"1"', '"2"')),
            encoding="utf-8",
            newline="\n",
        )
    if (
        identity(workspace) != state["root_identity"]
        or identity(systems) != systems_identity
        or stock_patch_state(workspace, expected_base) != stock
    ):
        raise ValueError(
            "COMM apply changed Git identity or original stock patch metadata"
        )
    state.update(
        product="native-link-width",
        systems_base=expected_base,
        systems_identity=systems_identity,
        stock_patch_state=stock,
        width=1 if case == "baseline" else 2,
        source_hashes={
            ROOT_CMAKE: digest(workspace / ROOT_CMAKE),
            **{"rocm-systems/" + n: digest(systems / n) for n in SYSTEM_FILES},
        },
        payloads=[
            dict(path=p.relative_to(workspace).as_posix(), sha256=digest(p))
            for p in [root_payload, systems_payload]
        ],
        source_git_commit_mutations=False,
        candidate_change="Only ROOT default1->2; identical three-file rocSHMEM patch in both arms",
    )
    save(output / "source-control.json", state)
    return state


def apply_elf(workspace, output, case, env):
    state = state_common(workspace, case, env)
    if (output / "source-control.json").exists():
        raise ValueError("Refuse to adopt existing ELF product state")
    require_original(workspace, [ELF_MODULE])
    payload = payload_path(workspace, "bounded-elf-probe.patch")
    regular(payload)
    if case == "candidate":
        patch(workspace, payload)
    if identity(workspace) != state["root_identity"]:
        raise ValueError("ELF apply changed source Git identity")
    state.update(
        product="bounded-elf-probe",
        native_package_type=env.get("NATIVE_PACKAGE_TYPE"),
        original_sha256=ORIGINAL_HASHES[ELF_MODULE],
        source_hashes={ELF_MODULE: digest(workspace / ELF_MODULE)},
        payloads=[
            dict(path=payload.relative_to(workspace).as_posix(), sha256=digest(payload))
        ],
        source_git_commit_mutations=False,
    )
    save(output / "source-control.json", state)
    return state


def verify_source(workspace, output, case, env):
    state = json.loads((output / "source-control.json").read_text(encoding="utf-8"))
    if (
        state["case"] != case
        or state["run_id"] != env.get("GITHUB_RUN_ID")
        or state["run_attempt"] != env.get("GITHUB_RUN_ATTEMPT")
    ):
        raise ValueError("Source receipt belongs to another run/attempt/case")
    if identity(workspace) != state["root_identity"]:
        raise ValueError("ROOT Git identity changed")
    for name, expected in state["source_hashes"].items():
        path = workspace / name
        regular(path)
        if digest(path) != expected:
            raise ValueError(f"Selected source changed after apply: {name}")
    for row in state["payloads"]:
        regular(workspace / row["path"])
        if digest(workspace / row["path"]) != row["sha256"]:
            raise ValueError("Reviewed patch payload changed")
    if state["product"] == "native-link-width":
        if (
            identity(workspace / "rocm-systems") != state["systems_identity"]
            or stock_patch_state(workspace, state["systems_base"])
            != state["stock_patch_state"]
        ):
            raise ValueError("Patched systems HEAD/history or stock .smrev changed")
    return state


def cache_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("//", "#")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def native_link_arguments(command, target, width):
    args = shlex.split(command)
    if args[:2] == [":", "&&"] and args[-2:] == ["&&", ":"]:
        args = args[2:-2]
    if (
        not args
        or Path(args[0]).name != "clang++"
        or any(
            v in ["&&", ";", "|", ">", "-c", "-v", "-###"] or v.startswith("@")
            for v in args
        )
    ):
        raise ValueError("Expected one expanded nonverbose native clang++ link")
    if "-o" not in args or args[args.index("-o") + 1] != target:
        raise ValueError("Unexpected native output")
    if any(
        v.lower().endswith((".cpp", ".cc", ".cxx", ".hip", ".cu", ".c")) for v in args
    ):
        raise ValueError("Native edge would compile a source")
    if (
        [
            v.removeprefix("--offload-arch=")
            for v in args
            if v.startswith("--offload-arch=")
        ]
        != ARCHES
        or "--hip-link" not in args
        or "-fgpu-rdc" not in args
    ):
        raise ValueError("Native link must retain all31 original RDC architectures")
    if [v for v in args if v.startswith(("--offload-jobs=", "-parallel-jobs="))] != (
        [] if width == 1 else ["--offload-jobs=2"]
    ):
        raise ValueError("Native link width/mode drift")
    return args


def validate_comm(workspace, build, output, case, env):
    state = verify_source(workspace, output, case, env)
    if state["product"] != "native-link-width":
        raise ValueError("Not a COMM product receipt")
    width = 1 if case == "baseline" else 2
    nested = build / "comm-libs/rocshmem/build"
    root_cache, cache = cache_values(build / "CMakeCache.txt"), cache_values(
        nested / "CMakeCache.txt"
    )
    if (
        root_cache.get("THEROCK_ROCSHMEM_NATIVE_LINK_JOBS") != str(width)
        or cache.get("ROCSHMEM_NATIVE_LINK_JOBS") != str(width)
        or cache.get("HIP_CLANG_NUM_PARALLEL_JOBS", "1") != "1"
    ):
        raise ValueError("Root/nested native or global HIP width changed")
    if width == 2 and cache.get("HIP_CLANG_SUPPORTS_PARALLEL_JOBS", "").upper() not in (
        "1",
        "ON",
        "TRUE",
        "YES",
    ):
        raise ValueError(
            "Candidate native offload width lacks generated capability support"
        )
    if cache.get("GPU_TARGETS", "").split(";") != ARCHES:
        raise ValueError("rocSHMEM target coverage changed")
    rows = json.loads(
        subprocess.check_output(
            ["ninja", "-C", str(nested), "-t", "compdb", "-x"], text=True
        )
    )
    edges = {}
    for row in rows:
        if row.get("output") in TARGETS:
            target = row["output"]
            if target in edges:
                raise ValueError("Duplicate native link edge")
            edges[target] = dict(
                command=row["command"],
                argv=native_link_arguments(row["command"], target, width),
            )
        elif " -c " in row["command"] and any(
            flag in row["command"] for flag in ["--offload-jobs=", "-parallel-jobs="]
        ):
            raise ValueError("Native link control leaked into compilation")
    if set(edges) != set(TARGETS):
        raise ValueError("Missing native final link targets")
    executed = {target: [] for target in TARGETS}
    for line in (nested / ".ninja_log").read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) == 5 and fields[3] in TARGETS:
            if int(fields[0]) > int(fields[1]) or not fields[4]:
                raise ValueError("Invalid executed native link record")
            executed[fields[3]].append(
                dict(
                    start_ms=int(fields[0]),
                    end_ms=int(fields[1]),
                    mtime=fields[2],
                    ninja_command_hash=fields[4],
                )
            )
    if not all(executed.values()):
        raise ValueError("Native final link has no executed Ninja record")
    outputs = []
    for target in TARGETS:
        path = nested / target
        regular(path)
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("Native output is not ELF")
        outputs.append(
            dict(target=target, bytes=path.stat().st_size, sha256=digest(path))
        )
    result = dict(
        success=True,
        run_id=state["run_id"],
        run_attempt=state["run_attempt"],
        case=case,
        width=width,
        generated_controls={
            name: cache.get(name)
            for name in [
                "ROCSHMEM_NATIVE_LINK_JOBS",
                "HIP_CLANG_NUM_PARALLEL_JOBS",
                "HIP_CLANG_SUPPORTS_PARALLEL_JOBS",
                "GPU_TARGETS",
            ]
        },
        final_links=edges,
        executed_native_links=executed,
        outputs=outputs,
        source_control_sha256=digest(output / "source-control.json"),
        scope="Source/cache/generated+executed native-link/file-hash gate; stock checks, full output parity and GPU correctness remain required",
    )
    save(output / "validation.json", result)
    return result


def validate_elf(workspace, output, case, env):
    state = verify_source(workspace, output, case, env)
    if state["product"] != "bounded-elf-probe" or state[
        "native_package_type"
    ] != env.get("NATIVE_PACKAGE_TYPE"):
        raise ValueError("Not the matching native ELF product receipt")
    result = dict(
        success=True,
        case=case,
        run_id=state["run_id"],
        run_attempt=state["run_attempt"],
        source_control_sha256=digest(output / "source-control.json"),
        source_hashes=state["source_hashes"],
        scope="Unchanged source gate only; package manifests/stock install and correctness checks remain required",
    )
    save(output / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["apply-comm", "validate-comm", "apply-elf", "validate-elf"]
    )
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--case", choices=["baseline", "candidate"])
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.enable:
        print(json.dumps(dict(selected=False, reason="default off")))
        return
    env = dict(os.environ)
    # Reject foreign/local execution before reading any event file.
    if (
        env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
        or env.get("GITHUB_ACTIONS") != "true"
    ):
        raise ValueError("TheRock Actions only")
    event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
    if not controls(env, event, args.case, args.command, sys.platform):
        print(
            json.dumps(dict(selected=False, reason="outside selected stage/native job"))
        )
        return
    output, build = args.output.resolve(), args.build_dir.resolve()
    if ROOT not in output.parents or ROOT not in build.parents:
        raise ValueError("Output/build must be contained in this checkout")
    if args.command == "apply-comm":
        result = apply_comm(ROOT, output, args.case, env)
    elif args.command == "validate-comm":
        result = validate_comm(ROOT, build, output, args.case, env)
    elif args.command == "apply-elf":
        result = apply_elf(ROOT, output, args.case, env)
    else:
        result = validate_elf(ROOT, output, args.case, env)
    print(
        json.dumps(
            dict(
                selected=True,
                command=args.command,
                case=args.case,
                success=result.get("success"),
            )
        )
    )


if __name__ == "__main__":
    main()
