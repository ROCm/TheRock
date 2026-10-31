# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pinned-source and inert Ninja/CMake fixtures; never run a ROCm compiler."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "normal_leading_stack", ROOT / "build_tools/normal_leading_stack.py"
)
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)
SOURCE = json.loads(
    (ROOT / "build_tools/tests/fixtures/normal_stack_source.json").read_text()
)
LINKS = json.loads(
    (ROOT / "build_tools/tests/fixtures/normal_stack_rocshmem_links.json").read_text()
)
ENV = dict(
    GITHUB_RUN_ID="fixture-run", GITHUB_RUN_ATTEMPT="1", NATIVE_PACKAGE_TYPE="deb"
)


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "user.name=Build Profile",
            "-c",
            "user.email=build-profile@example.invalid",
            "-C",
            str(root),
            *args,
        ],
        text=True,
    ).strip()


def repository(root):
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")


def fixture(root):
    for name in [stack.ROOT_CMAKE, stack.ELF_MODULE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(SOURCE[name].encode())
    for path in (ROOT / stack.PAYLOADS).glob("*.patch"):
        target = root / stack.PAYLOADS / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    stock = root / "patches/amd-mainline/rocm-systems/0001-stock-fixture.patch"
    stock.parent.mkdir(parents=True)
    stock.write_bytes(
        b"fixture for unchanged stock patch fingerprint; no product patch is added to this series\n"
    )
    repository(root)
    git(root, "add", "comm-libs", "build_tools", "patches")
    git(root, "commit", "-qm", "fixture original ROOT source")
    systems = root / "rocm-systems"
    for name in stack.SYSTEM_FILES:
        target = systems / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(SOURCE[name].encode())
    repository(systems)
    git(systems, "add", "projects")
    git(systems, "commit", "-qm", "fixture exact a10 rocSHMEM files")
    base = git(systems, "rev-parse", "HEAD")
    (systems / "ROCR-stock-patch-marker").write_text(
        "inert unrelated stock-patch history fixture\n"
    )
    git(systems, "add", "ROCR-stock-patch-marker")
    git(systems, "commit", "-qm", "fixture existing stock ROCR patch commit")
    (root / ".rocm-systems.smrev").write_text(
        "https://github.com/ROCm/rocm-systems\n"
        + base
        + "+PATCHED:"
        + hashlib.sha1(stock.read_bytes(), usedforsecurity=False).hexdigest()
        + "\n"
    )
    return base


def graph(root, case):
    build = root / "build"
    tree = build / "comm-libs/rocshmem/build"
    tree.mkdir(parents=True)
    width = 1 if case == "baseline" else 2
    (build / "CMakeCache.txt").write_text(
        f"THEROCK_ROCSHMEM_NATIVE_LINK_JOBS:STRING={width}\n"
    )
    (tree / "CMakeCache.txt").write_text(
        f"ROCSHMEM_NATIVE_LINK_JOBS:STRING={width}\nHIP_CLANG_NUM_PARALLEL_JOBS:STRING=1\nHIP_CLANG_SUPPORTS_PARALLEL_JOBS:BOOL=TRUE\nGPU_TARGETS:STRING="
        + ";".join(stack.ARCHES)
        + "\n"
    )
    ninja, log = [], ["# ninja log v5"]
    for index, row in enumerate(LINKS["links"]):
        command = row["command"].replace(LINKS["workspace_prefix"], root.as_posix())
        assert (
            hashlib.sha256(row["command"].encode()).hexdigest() == row["command_sha256"]
        )
        if case == "candidate":
            command = command.replace(" -o ", " --offload-jobs=2 -o ", 1)
        target = row["output"]
        ninja += [
            f"rule final{index}",
            f"  command = {command}",
            f"build {target}: final{index} input{index}.o",
        ]
        path = tree / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFinert native hash fixture, never executed\n")
        log.append(f"1\t2\t0\t{target}\t{index+1:016x}")
    (tree / "build.ninja").write_text("\n".join(ninja) + "\n")
    (tree / ".ninja_log").write_text("\n".join(log) + "\n")
    return build, tree


class LeadingStackTest(unittest.TestCase):
    def test_default_disabled_cli_and_selected_execution_guards(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "absent"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "build_tools/normal_leading_stack.py"),
                    "apply-comm",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(json.loads(result.stdout)["selected"])
            self.assertFalse(output.exists())
        env = {
            **ENV,
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "ROCm/TheRock",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "NORMAL_VALIDATION": "true",
            "RELEASE_TYPE": "ci",
            "STAGE_NAME": "comm-libs",
            "AMDGPU_FAMILIES": "",
        }
        event = dict(
            inputs=dict(
                profile_normal_validation="true",
                profile_normal_product="leading-build-stack",
                profile_normal_case="baseline",
            )
        )
        self.assertTrue(stack.controls(env, event, "baseline", "apply-comm", "linux"))
        self.assertFalse(
            stack.controls(
                {**env, "STAGE_NAME": "math-libs"},
                event,
                "baseline",
                "apply-comm",
                "linux",
            )
        )
        self.assertTrue(stack.controls(env, event, "baseline", "apply-elf", "linux"))
        for name, value in [
            ("GITHUB_REPOSITORY", "ROCm/rockrel"),
            ("RELEASE_TYPE", "nightly"),
            ("HIP_CLANG_NUM_PARALLEL_JOBS", "2"),
            ("GITHUB_RUN_ID", ""),
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                stack.controls(
                    {**env, name: value}, event, "baseline", "apply-comm", "linux"
                )
        with self.assertRaises(ValueError):
            stack.controls(env, event, "candidate", "apply-comm", "linux")

    def test_exact_source_common_patch_candidate_only_root_default_and_metadata_preserved(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            states, root_texts = [], []
            for case in ["baseline", "candidate"]:
                root = Path(raw) / case
                base = fixture(root)
                marker = (root / ".rocm-systems.smrev").read_bytes()
                before = stack.identity(root / "rocm-systems")
                state = stack.apply_comm(
                    root, root / "evidence", case, ENV, expected_base=base
                )
                states.append(state)
                root_texts.append((root / stack.ROOT_CMAKE).read_text())
                self.assertEqual(stack.identity(root / "rocm-systems"), before)
                self.assertEqual((root / ".rocm-systems.smrev").read_bytes(), marker)
                self.assertFalse(state["source_git_commit_mutations"])
                stack.verify_source(root, root / "evidence", case, ENV)
                with self.assertRaisesRegex(ValueError, "another run"):
                    stack.verify_source(
                        root,
                        root / "evidence",
                        case,
                        {**ENV, "GITHUB_RUN_ATTEMPT": "2"},
                    )
                with self.assertRaises(ValueError):
                    stack.apply_comm(
                        root, root / "second", case, ENV, expected_base=base
                    )
                git(
                    root / "rocm-systems",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "fixture identity mutation",
                )
                with self.assertRaisesRegex(ValueError, "HEAD/history"):
                    stack.verify_source(root, root / "evidence", case, ENV)
            self.assertEqual(
                root_texts[0].replace(
                    'set(THEROCK_ROCSHMEM_NATIVE_LINK_JOBS "1"',
                    'set(THEROCK_ROCSHMEM_NATIVE_LINK_JOBS "2"',
                ),
                root_texts[1],
            )
            self.assertEqual(
                {
                    k: v
                    for k, v in states[0]["source_hashes"].items()
                    if k.startswith("rocm-systems/")
                },
                {
                    k: v
                    for k, v in states[1]["source_hashes"].items()
                    if k.startswith("rocm-systems/")
                },
            )

    def test_changed_original_payload_or_stock_marker_rejects_before_root_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            for kind in ["source", "payload", "stock-marker"]:
                root = Path(raw) / kind
                base = fixture(root)
                if kind == "source":
                    path = root / "rocm-systems" / stack.SYSTEM_FILES[0]
                elif kind == "payload":
                    path = root / stack.PAYLOADS / "rocshmem-native-link-control.patch"
                else:
                    path = root / ".rocm-systems.smrev"
                path.write_bytes(path.read_bytes() + b"changed fixture input\n")
                original_root = (root / stack.ROOT_CMAKE).read_bytes()
                with self.assertRaises(ValueError):
                    stack.apply_comm(
                        root, root / "evidence", "candidate", ENV, expected_base=base
                    )
                self.assertEqual((root / stack.ROOT_CMAKE).read_bytes(), original_root)
                self.assertFalse((root / "evidence/source-control.json").exists())

    @unittest.skipUnless(shutil.which("ninja"), "Ninja required; no compiler execution")
    def test_actual_expanded_native_edges_full_targets_executed_records_and_width_gate(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            results = []
            for case in ["baseline", "candidate"]:
                root = Path(raw) / case
                base = fixture(root)
                output = root / "evidence"
                stack.apply_comm(root, output, case, ENV, expected_base=base)
                build, tree = graph(root, case)
                result = stack.validate_comm(root, build, output, case, ENV)
                results.append(result)
                self.assertEqual(len(result["final_links"]), 2)
                self.assertTrue(
                    all(
                        len([a for a in r["argv"] if a.startswith("--offload-arch=")])
                        == 31
                        for r in result["final_links"].values()
                    )
                )
                self.assertTrue(all(result["executed_native_links"].values()))
                log = tree / ".ninja_log"
                original = log.read_bytes()
                log.write_bytes(b"# ninja log v5\n")
                with self.assertRaisesRegex(ValueError, "no executed"):
                    stack.validate_comm(root, build, output, case, ENV)
                log.write_bytes(original)
                cache = tree / "CMakeCache.txt"
                cache.write_text(
                    cache.read_text().replace(
                        "HIP_CLANG_NUM_PARALLEL_JOBS:STRING=1",
                        "HIP_CLANG_NUM_PARALLEL_JOBS:STRING=2",
                    )
                )
                with self.assertRaisesRegex(ValueError, "width changed"):
                    stack.validate_comm(root, build, output, case, ENV)
            self.assertEqual(results[0]["outputs"], results[1]["outputs"])
            for target in stack.TARGETS:
                aa = results[0]["final_links"][target]["argv"]
                bb = results[1]["final_links"][target]["argv"]
                # Only fixture workspace prefixes differ between these independent roots.
                aa = [
                    a.replace((Path(raw) / "baseline").as_posix(), "WORKSPACE")
                    for a in aa
                ]
                bb = [
                    a.replace((Path(raw) / "candidate").as_posix(), "WORKSPACE")
                    for a in bb
                    if a != "--offload-jobs=2"
                ]
                self.assertEqual(aa, bb)
                for bad in [
                    " ".join(aa).replace("--offload-arch=gfx1250-strict ", ""),
                    " ".join(aa) + " --offload-jobs=2",
                    " ".join(aa) + " -c",
                ]:
                    with self.assertRaises(ValueError):
                        stack.native_link_arguments(bad, target, 1)

    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("ninja") and os.name != "nt",
        "Host CMake/Ninja CXX generation fixture",
    )
    def test_actual_cmake_targets_keep_37_compiles_identical_and_only_two_link_flags_change(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "source"
            base = fixture(root)
            stack.apply_comm(
                root, root / "evidence", "baseline", ENV, expected_base=base
            )
            candidate = root / "rocm-systems/projects/rocshmem"
            host = Path(raw) / "host"
            host.mkdir()
            cmake = (candidate / "CMakeLists.txt").read_text()
            start = cmake.index("# Limit this control")
            block = cmake[
                start : cmake.index("endfunction()", start) + len("endfunction()")
            ]
            (host / "native-options.cmake").write_text(block + "\n")
            for relative in ["tools", "tests/functional_tests"]:
                destination = host / relative
                destination.mkdir(parents=True)
                text = (candidate / relative / "CMakeLists.txt").read_text()
                (destination / "CMakeLists.txt").write_text(text)
                for filename in set(re.findall(r"[A-Za-z0-9_]+\.(?:cpp|hpp)", text)):
                    (destination / filename).write_text("int fixture();\n")
            (
                host / "tests/functional_tests/CMakeDeviceBitcodeTester.cmake"
            ).write_bytes(
                SOURCE[
                    "projects/rocshmem/tests/functional_tests/CMakeDeviceBitcodeTester.cmake"
                ].encode()
            )
            for relative in [
                "cmake/setup_project.cmake",
                "scripts/functional_tests/driver.sh",
                "tests/functional_tests/test_wrapper.sh",
                "tests/functional_tests/ctest_with_retry.sh",
            ]:
                path = host / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# inert host fixture\n")
            flags = " ".join("--offload-arch=" + a for a in stack.ARCHES)
            (host / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.16)\nproject(native_link_fixture LANGUAGES CXX)\nset(CMAKE_EXPORT_COMPILE_COMMANDS ON)\nset(CMAKE_INSTALL_BINDIR bin)\nset(CMAKE_INSTALL_DATADIR share)\nset(BUILD_TESTS_ONLY OFF)\nset(BUILD_CTESTS OFF)\nset(USE_GDA ON)\nset(USE_SDMA ON)\ninclude(native-options.cmake)\nfunction(rocm_install)\nendfunction()\nadd_library(fixture_rocshmem INTERFACE)\nadd_library(roc::rocshmem ALIAS fixture_rocshmem)\ntarget_compile_options(fixture_rocshmem INTERFACE -fgpu-rdc "
                + flags
                + ")\ntarget_link_options(fixture_rocshmem INTERFACE --hip-link -fgpu-rdc "
                + flags
                + ")\nadd_subdirectory(tools)\nadd_subdirectory(tests/functional_tests)\n"
            )

            def configure(options, success=True):
                result = subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(host),
                        "-B",
                        str(host / "build"),
                        "-GNinja",
                        "-DHIP_CLANG_SUPPORTS_PARALLEL_JOBS=TRUE",
                        "-DHIP_CLANG_NUM_PARALLEL_JOBS=1",
                        *options,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode == 0, success, result.stdout + result.stderr
                )

            captures = []
            for width in [1, 2]:
                configure([f"-DROCSHMEM_NATIVE_LINK_JOBS={width}"])
                compile_bytes = (host / "build/compile_commands.json").read_bytes()
                self.assertEqual(len(json.loads(compile_bytes)), 37)
                rows = json.loads(
                    subprocess.check_output(
                        ["ninja", "-C", str(host / "build"), "-t", "compdb", "-x"],
                        text=True,
                    )
                )
                links = {
                    r["output"]: stack.native_link_arguments(
                        r["command"]
                        .replace("/usr/bin/c++", "/fixture/clang++")
                        .replace("/usr/bin/g++", "/fixture/clang++"),
                        r["output"],
                        width,
                    )
                    for r in rows
                    if r.get("output") in stack.TARGETS
                }
                self.assertEqual(set(links), set(stack.TARGETS))
                captures.append((compile_bytes, links))
            self.assertEqual(captures[0][0], captures[1][0])
            for target in stack.TARGETS:
                self.assertEqual(
                    captures[0][1][target],
                    [a for a in captures[1][1][target] if a != "--offload-jobs=2"],
                )
            for flags in [
                ["-DROCSHMEM_NATIVE_LINK_JOBS=0"],
                [
                    "-DROCSHMEM_NATIVE_LINK_JOBS=2",
                    "-DHIP_CLANG_SUPPORTS_PARALLEL_JOBS=FALSE",
                ],
                ["-DROCSHMEM_NATIVE_LINK_JOBS=2", "-DHIP_CLANG_NUM_PARALLEL_JOBS=2"],
            ]:
                configure(flags, False)

    def test_bounded_elf_patch_only_candidate_same_tool_decisions_and_source_gate(self):
        calls = []
        with tempfile.TemporaryDirectory() as raw:
            for case in ["baseline", "candidate"]:
                root = Path(raw) / case
                fixture(root)
                output = root / "evidence"
                state = stack.apply_elf(root, output, case, ENV)
                self.assertEqual(
                    state["original_sha256"], stack.ORIGINAL_HASHES[stack.ELF_MODULE]
                )
                path = root / stack.ELF_MODULE
                spec = importlib.util.spec_from_file_location(
                    "fixture_rpath_" + case, path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                payload = root / "payload"
                payload.mkdir()
                (payload / "elf").write_bytes(b"\x7fELF" + b"x" * 8192)
                (payload / "text").write_bytes(b"notELF")
                with patch.object(
                    module, "_get_rpath", return_value="$ORIGIN"
                ), patch.object(module.subprocess, "check_call") as tool:
                    module.update_rpath(payload, [])
                calls.append(
                    [
                        [str(a).replace(str(root), "ROOT") for a in call.args[0]]
                        for call in tool.call_args_list
                    ]
                )
                stack.validate_elf(root, output, case, ENV)
                path.write_bytes(path.read_bytes() + b"\n# changed source\n")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    stack.validate_elf(root, output, case, ENV)
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(len(calls[0]), 1)


if __name__ == "__main__":
    unittest.main()
