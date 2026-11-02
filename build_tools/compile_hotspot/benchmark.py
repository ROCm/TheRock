# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Same-runner CK object-group trials; not a whole-stage benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import profile as diagnostic  # Sibling script, not the stdlib profiler.

HERE = Path(__file__).resolve().parent
LIBRARIES = diagnostic.ROOT / "rocm-libraries"
CK_SOURCE = LIBRARIES / "projects/composablekernel"
PATCH = HERE / "candidate.patch"
REFERENCE_PATCH = HERE / "three-tu-reference.patch"
ORIGINAL = (
    "projects/composablekernel/library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/nhwgc/xdl/"
    + diagnostic.SOURCE
)


def object_run(path: Path) -> int:
    import resource

    spec = json.loads(path.read_text())
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with Path(spec["log"]).open("w") as stream:
        code = subprocess.run(
            spec["argv"], cwd=spec["cwd"], stdout=stream, stderr=subprocess.STDOUT
        ).returncode
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "exit_code": code,
        "wall_s": time.monotonic() - started,
        "user_s": usage.ru_utime - before.ru_utime,
        "system_s": usage.ru_stime - before.ru_stime,
        "max_child_rss_kib": usage.ru_maxrss,
    }
    Path(spec["measurement"]).write_text(json.dumps(result, indent=2) + "\n")
    return code


def group_run(path: Path) -> int:
    paths = json.loads(path.read_text())["object_specs"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                lambda p: subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "object",
                        "--spec",
                        p,
                    ]
                ).returncode,
                paths,
            )
        )
    return int(any(results))


def arguments(command: str) -> list[str]:
    argv = shlex.split(command)
    if any(token in {"&&", "||", ";", "|"} for token in argv):
        raise RuntimeError("Expected direct compiler command")
    return argv[1:] if Path(argv[0]).name == "ccache" else argv


def common_flags(argv: list[str]) -> list[str]:
    result, index = [], 0
    while index < len(argv):
        if argv[index] in {"-o", "-MF", "-MT", "-c"}:
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


def entries(names: set[str]) -> dict:
    database = json.loads(
        diagnostic.run(["ninja", "-t", "compdb", "-x"], cwd=diagnostic.CK)
    )
    selected = {}
    rejected = []
    for entry in database:
        name = Path(entry.get("file", "")).name
        if name in names:
            if not diagnostic.is_object_compilation(entry, name):
                rejected.append(entry)
                continue
            if name in selected:
                raise RuntimeError(f"Duplicate command: {name}")
            selected[name] = entry
    diagnostic.save(
        f"benchmark-compdb-selection-{len(names)}.json",
        {"selected": selected, "rejected": rejected},
    )
    if set(selected) != names:
        raise RuntimeError(f"Missing CMake-discovered sources: {names - set(selected)}")
    return selected


def reconfigure_ck() -> None:
    """Replay TheRock's configure edge with a fresh cache, retaining its inputs.

    A direct second configure reused Threads detection state and added -pthread.
    The root edge supplies the original environment, init/toolchain files and
    project arguments. Only these two owned files are reset; no build is run.
    """
    expected = diagnostic.BUILD / "math-libs/composable_kernel/build"
    if diagnostic.CK.resolve() != expected.resolve():
        raise RuntimeError("Refusing to reset an unexpected CK build directory")
    cache = diagnostic.CK / "CMakeCache.txt"
    stamp = diagnostic.CK.parent / "stamp/configure.stamp"
    if not cache.is_file() or not stamp.is_file():
        raise RuntimeError("Baseline CK configuration must exist before reset")
    if cache.is_symlink() or stamp.is_symlink():
        raise RuntimeError("Expected regular owned CK configuration files")
    edge = diagnostic.run(
        [
            "ninja",
            "-C",
            str(diagnostic.BUILD),
            "-t",
            "commands",
            "composable_kernel+configure",
        ]
    )
    for flag, filename in (
        ("-DCMAKE_TOOLCHAIN_FILE=", "_toolchain.cmake"),
        ("-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=", "_init.cmake"),
    ):
        if flag + str(diagnostic.CK.parent / filename) not in edge:
            raise RuntimeError("TheRock CK configure edge is missing its pinned inputs")
    (diagnostic.OUT / "candidate-configure-edge.txt").write_text(edge)
    shutil.copy2(cache, diagnostic.OUT / "candidate-pre-reset-CMakeCache.txt")
    diagnostic.save(
        "candidate-configure-reset.json",
        {
            "removed_files": [
                diagnostic.fingerprint(cache),
                diagnostic.fingerprint(stamp),
            ]
        },
    )
    cache.unlink()
    stamp.unlink()
    diagnostic.run(
        [
            "cmake",
            "--build",
            str(diagnostic.BUILD),
            "--target",
            "composable_kernel+configure",
            "--",
            "-v",
        ],
        log="candidate-configure.log",
    )
    if not cache.is_file() or not stamp.is_file():
        raise RuntimeError("TheRock did not regenerate the CK configuration")
    shutil.copy2(cache, diagnostic.OUT / "candidate-post-reset-CMakeCache.txt")


def benchmark() -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Run trials on the dedicated CI runner")
    if not PATCH.is_file():
        raise RuntimeError("Reviewed candidate.patch has not been supplied")
    git = ["git", "-C", str(LIBRARIES)]
    if diagnostic.run([*git, "status", "--porcelain"]).strip():
        raise RuntimeError("Library source must start clean")
    paths = re.findall(r"^\+\+\+ b/(.+)$", PATCH.read_text(), re.MULTILINE)
    cmake_path = (Path(ORIGINAL).parent.parent / "CMakeLists.txt").as_posix()
    cpp_paths = [p for p in paths if p.endswith(".cpp")]
    expected_paths = {
        ORIGINAL,
        ORIGINAL.replace("_instance.cpp", "_kernel_group1_instance.cpp"),
        ORIGINAL.replace("_instance.cpp", "_nongrouped_match_instance.cpp"),
        ORIGINAL.replace("_instance.cpp", "_kernel_groups.hpp"),
        cmake_path,
    }
    if len(paths) != 5 or set(paths) != expected_paths or len(cpp_paths) != 3:
        raise RuntimeError(
            "Expected three kernel-group sources, private header and companion CMakeLists"
        )
    if any(Path(p).parent != Path(ORIGINAL).parent for p in cpp_paths):
        raise RuntimeError("Patch outside selected CK source directory")
    reference_paths = re.findall(
        r"^\+\+\+ b/(.+)$", REFERENCE_PATCH.read_text(), re.MULTILINE
    )
    reference_cpp_paths = [p for p in reference_paths if p.endswith(".cpp")]
    if (
        len(reference_paths) != 4
        or len(set(reference_paths)) != 4
        or ORIGINAL not in reference_paths
        or cmake_path not in reference_paths
        or len(reference_cpp_paths) != 3
        or any(Path(p).parent != Path(ORIGINAL).parent for p in reference_cpp_paths)
    ):
        raise RuntimeError("Unexpected three-TU reference patch paths")
    if (
        diagnostic.fingerprint(REFERENCE_PATCH)["sha256"]
        != "2c00276ac5cbb6a374f83201fa38879cbd59fb654bf8c619a5f72bc37715445d"
    ):
        raise RuntimeError(
            "Three-TU reference differs from the previously measured patch"
        )
    baseline = entries({diagnostic.SOURCE})
    baseline_argv = arguments(baseline[diagnostic.SOURCE]["command"])
    historical = json.loads(
        (HERE / "baseline-argv.json").read_text(encoding="utf-8-sig")
    )[1:]
    normalize = lambda values: [
        value.replace(str(diagnostic.ROOT), "<ROOT>").replace(
            "/__w/TheRock/TheRock", "<ROOT>"
        )
        for value in values
    ]
    if os.environ.get("BUILD_VARIANT") == "asan-debug" and normalize(
        baseline_argv
    ) != normalize(historical):
        diagnostic.save(
            "benchmark-baseline-mismatch.json",
            {"historical": historical, "current": baseline_argv},
        )
        raise RuntimeError("ASan baseline command differs from the historical command")
    generated_headers = {
        str(p.relative_to(diagnostic.CK)): diagnostic.fingerprint(p)
        for p in (diagnostic.CK / "include").rglob("*")
        if p.is_file()
    }
    baseline_sources = {
        p: diagnostic.fingerprint(LIBRARIES / p) for p in (ORIGINAL, cmake_path)
    }
    source_headers = {
        str(p.relative_to(diagnostic.ROOT)): diagnostic.fingerprint(p)
        for include_root in (CK_SOURCE / "include", CK_SOURCE / "library/include")
        for p in include_root.rglob("*")
        if p.is_file()
    }
    state = "baseline"

    def set_state(wanted: str) -> None:
        nonlocal state
        if wanted == state:
            return
        patches = {"candidate": PATCH, "reference": REFERENCE_PATCH}
        if state != "baseline":
            diagnostic.run([*git, "apply", "--reverse", "--check", str(patches[state])])
            diagnostic.run([*git, "apply", "--reverse", str(patches[state])])
            state = "baseline"
        if wanted != "baseline":
            diagnostic.run([*git, "apply", "--check", str(patches[wanted])])
            diagnostic.run([*git, "apply", str(patches[wanted])])
        state = wanted

    set_state("candidate")
    reconfigure_ck()
    candidate = entries({Path(p).name for p in cpp_paths})
    candidate_headers = {
        str(p.relative_to(diagnostic.CK)): diagnostic.fingerprint(p)
        for p in (diagnostic.CK / "include").rglob("*")
        if p.is_file()
    }
    if generated_headers != candidate_headers:
        diagnostic.save(
            "generated-header-mismatch.json",
            {"baseline": generated_headers, "candidate": candidate_headers},
        )
        raise RuntimeError("Candidate regeneration changed generated CK headers")
    for name, entry in candidate.items():
        if common_flags(arguments(entry["command"])) != common_flags(baseline_argv):
            diagnostic.save(
                "candidate-flag-mismatch.json",
                {"baseline": baseline, "candidate": candidate},
            )
            raise RuntimeError(f"Compiler flags changed: {name}")
    candidate_sources = {p: diagnostic.fingerprint(LIBRARIES / p) for p in paths}
    set_state("reference")
    reconfigure_ck()
    reference = entries({Path(p).name for p in reference_cpp_paths})
    reference_headers = {
        str(p.relative_to(diagnostic.CK)): diagnostic.fingerprint(p)
        for p in (diagnostic.CK / "include").rglob("*")
        if p.is_file()
    }
    if reference_headers != generated_headers:
        raise RuntimeError("Reference regeneration changed generated CK headers")
    for name, entry in reference.items():
        if common_flags(arguments(entry["command"])) != common_flags(baseline_argv):
            raise RuntimeError(f"Reference compiler flags changed: {name}")
    reference_sources = {
        p: diagnostic.fingerprint(LIBRARIES / p) for p in reference_paths
    }
    trials = [
        ("B1", "baseline"),
        ("T1", "reference"),
        ("C1", "candidate"),
        ("C2", "candidate"),
        ("T2", "reference"),
        ("B2", "baseline"),
    ]
    sources_by_variant = {
        "baseline": baseline_sources,
        "candidate": candidate_sources,
        "reference": reference_sources,
    }
    commands_by_variant = {
        "baseline": baseline,
        "candidate": candidate,
        "reference": reference,
    }
    diagnostic.save(
        "benchmark-protocol.json",
        {
            "order": [trial for trial, _ in trials],
            "candidate_parallelism_cap": 3,
            "variant_object_counts": {
                name: len(commands) for name, commands in commands_by_variant.items()
            },
            "compiler_instrumentation": False,
            "compiler": {
                **diagnostic.fingerprint(Path(baseline_argv[0])),
                "version": diagnostic.run([baseline_argv[0], "--version"]),
            },
            "cache": "CCACHE_DISABLE=1; OS cache uncontrolled",
            "patch": diagnostic.fingerprint(PATCH),
            "reference_patch": diagnostic.fingerprint(REFERENCE_PATCH),
            "baseline_commands": baseline,
            "candidate_commands": candidate,
            "reference_commands": reference,
            "baseline_sources": baseline_sources,
            "candidate_sources": candidate_sources,
            "reference_sources": reference_sources,
            "generated_headers": generated_headers,
            "source_headers": source_headers,
            "limits": "Isolated object groups; no stage improvement inferred. Identical wrappers/RSS sampling in all three arms; two observations per arm, mirrored order, OS cache uncontrolled.",
        },
    )
    # Reject host/template diagnostics before spending time on control trials.
    set_state("candidate")
    dispatcher = candidate[diagnostic.SOURCE]
    syntax_argv = [
        value
        for value in common_flags(arguments(dispatcher["command"]))
        if value not in {"-MD", "-MMD", "-MP"}
    ] + ["-fsyntax-only", dispatcher["file"]]
    diagnostic.save(
        "candidate-syntax-preflight.json",
        {
            "argv": syntax_argv,
            "cwd": dispatcher["directory"],
            "scope": "Untimed dispatcher host/device syntax and warning check; no flag relaxation.",
        },
    )
    diagnostic.run(
        syntax_argv,
        cwd=Path(dispatcher["directory"]),
        log="candidate-syntax-preflight.log",
    )
    summary = []
    for trial, variant in trials:
        set_state(variant)
        expected = sources_by_variant[variant]
        actual = {p: diagnostic.fingerprint(LIBRARIES / p) for p in expected}
        if actual != expected:
            raise RuntimeError(f"Source hash mismatch: {trial}")
        if any(
            (LIBRARIES / p).exists()
            for p in set(paths + reference_paths)
            if p not in expected
        ):
            raise RuntimeError(f"Other variant sources survived restoration: {trial}")
        directory = diagnostic.OUT / "benchmarks" / trial
        directory.mkdir(parents=True, exist_ok=False)
        commands = commands_by_variant[variant]
        specs = []
        for name, entry in sorted(commands.items()):
            stem = Path(name).stem
            argv = arguments(entry["command"])
            for flag, extension in (("-o", ".o"), ("-MF", ".d"), ("-MT", ".o")):
                if flag in argv:
                    argv[argv.index(flag) + 1] = str(directory / (stem + extension))
            spec = {
                "argv": argv,
                "cwd": entry["directory"],
                "log": str(directory / (stem + ".log")),
                "measurement": str(directory / (stem + "-measurement.json")),
            }
            path = directory / (stem + "-command.json")
            path.write_text(json.dumps(spec, indent=2) + "\n")
            specs.append(str(path))
        group = directory / "group-spec.json"
        group.write_text(json.dumps({"object_specs": specs}, indent=2) + "\n")
        diagnostic.save(f"benchmarks/{trial}/sources.json", actual)
        diagnostic.measure(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "group",
                "--spec",
                str(group),
            ],
            diagnostic.ROOT,
            f"benchmarks/{trial}/group",
        )
        objects = [diagnostic.fingerprint(p) for p in directory.glob("*.o")]
        if len(objects) != len(commands):
            raise RuntimeError(f"Missing object output: {trial}")
        results = [
            json.loads(Path(json.loads(Path(p).read_text())["measurement"]).read_text())
            for p in specs
        ]
        summary.append(
            {
                "trial": trial,
                "variant": variant,
                "objects": objects,
                "group_measurement": json.loads(
                    (directory / "group-measurement.json").read_text()
                ),
                "group_gnu_time": (directory / "group-time.txt").read_text(),
                "aggregate_object_user_s": sum(row["user_s"] for row in results),
                "aggregate_object_system_s": sum(row["system_s"] for row in results),
                "longest_object_wall_s": max(row["wall_s"] for row in results),
                "objects_measured": results,
            }
        )
        diagnostic.save("benchmark-summary.json", summary)
    set_state("candidate")
    reconfigure_ck()
    # Small host utility support is compiled outside all timed trials.
    diagnostic.run(
        ["ninja", "-C", str(diagnostic.CK), "utility"], log="utility-build.log"
    )
    utilities = list(diagnostic.CK.rglob("libutility.a"))
    if len(utilities) != 1:
        raise RuntimeError(f"Expected one CK utility archive, found {utilities}")
    support = diagnostic.OUT / "link-support"
    support.mkdir(exist_ok=True)
    shutil.copy2(utilities[0], support / "libutility.a")
    diagnostic.save(
        "link-support/manifest.json",
        {
            "utility": diagnostic.fingerprint(utilities[0]),
            "scope": "Built after all measured trials; excluded from compile-group metrics.",
        },
    )
    diagnostic.save(
        "benchmark-complete.json",
        {
            "status": "complete",
            "source_state": state,
            "followup": "GPU contract, actual archive rebuild and consumer quick/standard tests still required.",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "group", "object"])
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    os.environ["CCACHE_DISABLE"] = "1"
    if args.action == "object":
        return object_run(args.spec)
    if args.action == "group":
        return group_run(args.spec)
    benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
