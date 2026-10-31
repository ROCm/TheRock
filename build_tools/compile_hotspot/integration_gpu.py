# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate rebuilt CK/MIOpen products using pinned official consumer tests."""

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

SOURCE_SHA = "5b7cb88373362ab80dc0f19605915e534f4158aa"
FOCUS = (
    "Smoke/GPU_UnitTestConvSolverImplicitGemmGroupBwdXdlops_FP32."
    "ConvHipImplicitGemmGroupBwdXdlops/*"
)
HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def contained(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise RuntimeError(f"Artifact path escapes its root: {relative}")
    return path


def resolve_asan_runtime(compiler, sdk):
    runtimes = set()
    for name in ("libclang_rt.asan.so", "libclang_rt.asan-x86_64.so"):
        path = Path(
            subprocess.check_output(
                [str(compiler), f"-print-file-name={name}"], text=True
            ).strip()
        )
        if path.is_absolute() and path.is_file():
            path = path.resolve()
            if not path.is_relative_to(sdk.resolve()):
                raise RuntimeError(
                    "Compiler resolved an ASan runtime outside the pinned SDK"
                )
            runtimes.add(path)
    if len(runtimes) != 1:
        raise RuntimeError(
            f"Expected one pinned shared ASan runtime, got {sorted(runtimes)}"
        )
    return runtimes.pop()


class Validation:
    def __init__(self, args):
        self.args = args
        self.root = Path.cwd().resolve()
        self.bundle = args.bundle.resolve()
        self.out = args.output.resolve()
        self.sdk = args.sdk.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.issues = []

    def save(self, name, data):
        (self.out / name).write_text(json.dumps(data, indent=2) + "\n")

    def run(self, argv, label, env=None, check=True):
        argv = list(map(str, argv))
        self.save(label + "-command.json", argv)
        print(shlex.join(argv), flush=True)
        with (self.out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.save(label + "-result.json", {"exit_code": result.returncode})
        if check:
            result.check_returncode()
        return result.returncode

    def verify_bundle(self):
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())
        m = self.manifest
        if m["source_sha"] != SOURCE_SHA or m["variant"] != self.args.variant:
            raise RuntimeError("Bundle source/variant mismatch")
        if str(m["baseline_run_id"]) != self.args.baseline_run_id:
            raise RuntimeError("Bundle baseline runtime mismatch")
        if str(m["producer_run_id"]) != self.args.producer_run_id:
            raise RuntimeError("Bundle does not belong to the requested producer run")
        self.files, self.symlinks = m["files"], m["symlinks"]
        actual = {
            str(path.relative_to(self.bundle))
            for path in self.bundle.rglob("*")
            if path.is_file() or path.is_symlink()
        } - {"manifest.json"}
        if actual != set(self.files) | set(self.symlinks):
            raise RuntimeError("Bundle has missing or unmanifested files")
        for relative, expected in self.files.items():
            path = contained(self.bundle, relative)
            if path.is_symlink() or digest(path) != expected:
                raise RuntimeError(f"Bundle file hash mismatch: {relative}")
        for relative, target in self.symlinks.items():
            path = contained(self.bundle, relative)
            if not path.is_symlink() or os.readlink(path) != target:
                raise RuntimeError(f"Bundle symlink mismatch: {relative}")
        self.save("producer-manifest.json", m)
        baseline = json.loads(self.args.baseline_contract_provenance.read_text())
        self.baseline_provenance = baseline
        context = baseline["producer_context"]
        for key in ("source_sha", "variant", "baseline_run_id"):
            if str(context[key]) != str(m[key]):
                raise RuntimeError(f"Contract baseline provenance mismatch: {key}")
        if str(context["run_id"]) != self.args.baseline_benchmark_run_id:
            raise RuntimeError("Baseline contract came from a different benchmark run")
        validation = baseline.get("validation_context")
        if validation is None:
            if (
                self.args.baseline_contract_run_id
                != self.args.baseline_benchmark_run_id
            ):
                raise RuntimeError(
                    "Cross-run GPU contract has no validation provenance"
                )
        else:
            producer = baseline["producer_run_metadata"]
            if (
                str(validation["run_id"]) != self.args.baseline_contract_run_id
                or str(validation["benchmark_run_id"])
                != self.args.baseline_benchmark_run_id
                or not validation["instrumentation_sha"]
                or validation["repository"].lower() != "rocm/therock"
                or str(producer["id"]) != self.args.baseline_benchmark_run_id
                or producer["head_sha"] != context["instrumentation_sha"]
            ):
                raise RuntimeError(
                    "GPU validation and benchmark producer provenance mismatch"
                )
            if baseline["benchmark_protocol_sha256"] != digest(
                self.args.baseline_benchmark_protocol
            ):
                raise RuntimeError("GPU validation used a different benchmark protocol")
        if baseline["harness_source_sha256"] != digest(HERE / "ck_f32_contract.cpp"):
            raise RuntimeError("Baseline used a different contract source")
        if baseline["compiler_sha256"] != m["compiler_sha256"]:
            raise RuntimeError("Baseline and integration compilers differ")
        if digest(HERE / "candidate.patch") != m["patch_sha256"]:
            raise RuntimeError(
                "Integration source patch differs from the benchmark patch"
            )
        protocol = json.loads(self.args.baseline_benchmark_protocol.read_text())
        if protocol["patch"]["sha256"] != m["patch_sha256"]:
            raise RuntimeError("Paired benchmark used a different candidate patch")
        if protocol["compiler"]["sha256"] != m["compiler_sha256"]:
            raise RuntimeError("Paired benchmark used a different compiler")
        if m["harness_source_sha256"] != baseline["harness_source_sha256"]:
            raise RuntimeError("Archive contract uses a different harness source")
        if m["rocm_libraries_sha"] != "f4e24d58b3f313f1a31068def1e35376e43c2f49":
            raise RuntimeError("Integration library revision changed")
        self.save("baseline-benchmark-protocol.json", protocol)
        self.save("baseline-contract-provenance.json", baseline)
        shutil.copy2(self.args.baseline_contract_log, self.out / "baseline-numeric.log")

    def fetch(self, names, components, destination, label):
        topology = tomllib.loads((self.root / "BUILD_TOPOLOGY.toml").read_text())
        all_names = set(topology["artifacts"])
        if not names or not set(names) <= all_names:
            raise RuntimeError(f"Invalid artifact allowlist: {names}")
        self.run(
            [
                sys.executable,
                "build_tools/artifact_manager.py",
                "fetch",
                f"--run-id={self.args.baseline_run_id}",
                "--run-github-repo=ROCm/TheRock",
                "--stage=all",
                "--amdgpu-families=gfx950",
                "--flatten",
                f"--output-dir={destination}",
                "--exclude-artifacts=" + ",".join(sorted(all_names - set(names))),
                "--exclude-components="
                + ",".join(
                    sorted(
                        {"lib", "run", "dev", "dbg", "doc", "test"} - set(components)
                    )
                ),
            ],
            label,
        )

    def check_overlay(self):
        for relative, expected in self.files.items():
            if (
                relative.startswith("sdk/")
                and digest(self.sdk / relative[4:]) != expected
            ):
                raise RuntimeError(f"Rebuilt SDK file mismatch: {relative}")
        for relative, target in self.symlinks.items():
            if (
                relative.startswith("sdk/")
                and os.readlink(self.sdk / relative[4:]) != target
            ):
                raise RuntimeError(f"Rebuilt SDK symlink mismatch: {relative}")

    def prepare_sdk(self):
        if self.sdk.exists():
            raise RuntimeError("SDK directory must be fresh to exclude stale libraries")
        names = self.manifest["imported_artifacts"]
        sys.path.insert(0, str(self.root / "build_tools"))
        from _therock_utils.build_topology import BuildTopology

        topology = BuildTopology(str(self.root / "BUILD_TOPOLOGY.toml"))
        logical = (
            topology.get_inbound_artifacts("math-libs")
            | topology.get_produced_artifacts("math-libs")
        ) - {"miopen", "composable-kernel"}
        logical_only = {"core-hipinfo", "spdlog", "sysdeps-libmnl", "sysdeps-libnl"}
        reasons = self.manifest["logical_only_artifacts"]
        if (
            len(logical) != 37
            or set(self.manifest["requested_logical_artifacts"]) != logical
            or set(reasons) != logical_only
            or not all(
                isinstance(value, str) and value.strip() for value in reasons.values()
            )
            or set(names) != logical - logical_only
            or len(names) != 33
            or "sysdeps" not in names
        ):
            raise RuntimeError(
                "Dependency manifest differs from pinned Linux physical artifact closure"
            )
        self.fetch(names, {"lib", "run", "dev"}, self.sdk, "fetch-untouched")
        self.fetch(["miopen"], {"run", "test"}, self.sdk, "fetch-miopen-tests")
        data_sdk = self.sdk.with_name(self.sdk.name + "-baseline-miopen-data")
        self.fetch(["miopen"], {"lib"}, data_sdk, "fetch-miopen-data")
        data = data_sdk / "share/miopen"
        if data.is_dir():
            for path in data.rglob("*"):
                if path.is_symlink() and not path.resolve().is_relative_to(
                    data.resolve()
                ):
                    raise RuntimeError(
                        "MIOpen data symlink escapes whitelisted data tree"
                    )
            shutil.copytree(
                data, self.sdk / "share/miopen", dirs_exist_ok=True, symlinks=True
            )
        old_libraries = list(self.sdk.rglob("libMIOpen*.so*"))
        if old_libraries:
            raise RuntimeError(
                f"Baseline changed libraries entered SDK: {old_libraries}"
            )
        shutil.copytree(
            self.bundle / "sdk", self.sdk, dirs_exist_ok=True, symlinks=True
        )
        self.plugin = contained(self.sdk, self.manifest["plugin_path"]).resolve()
        self.miopen = contained(
            self.sdk, self.manifest["miopen_library_path"]
        ).resolve()
        self.check_overlay()
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("GTEST_") or key in {
                "LD_PRELOAD",
                "LD_DEBUG",
                "LD_DEBUG_OUTPUT",
            }:
                self.env.pop(key)
        libdirs = [
            self.sdk / p
            for p in ("lib", "lib64", "lib/rocm_sysdeps/lib", "lib/llvm/lib")
        ]
        if self.args.variant == "asan-debug":
            self.env["HSA_XNACK"] = "1"
            sys.path.insert(0, str(self.root / "build_tools/github_actions"))
            from configure_asan_env import STATIC_ASAN_ENV

            self.env.update(STATIC_ASAN_ENV)
            self.env["ASAN_SYMBOLIZER_PATH"] = str(
                self.sdk / "lib/llvm/bin/llvm-symbolizer"
            )
            compiler = self.sdk / "lib/llvm/bin/clang++"
            if digest(compiler) != self.manifest["compiler_sha256"]:
                raise RuntimeError(
                    "Fetched compiler does not match the producer compiler"
                )
            runtime = resolve_asan_runtime(compiler, self.sdk)
            runtime_hash = digest(runtime)
            if runtime_hash != self.baseline_provenance["sanitizer_runtime_sha256"]:
                raise RuntimeError(
                    "Selected ASan runtime differs from the paired baseline"
                )
            self.save(
                "asan-runtime.json", {"path": str(runtime), "sha256": runtime_hash}
            )
            libdirs.append(runtime.parent)
        self.env.update(
            {
                "LD_LIBRARY_PATH": ":".join(str(p) for p in libdirs if p.is_dir()),
                "PATH": str(self.sdk / "bin") + ":" + self.env.get("PATH", ""),
                "ROCM_PATH": str(self.sdk),
                "THEROCK_BIN_DIR": str(self.sdk / "bin"),
                "MIOPEN_CK_LIB_PATH": str(self.plugin.parent),
                "TEST_COMPONENT": "miopen",
                "AMDGPU_FAMILIES": "gfx950",
                "SHARD_INDEX": "1",
                "TOTAL_SHARDS": "1",
            }
        )
        self.save(
            "runtime-environment.json",
            {
                k: self.env.get(k)
                for k in (
                    "LD_LIBRARY_PATH",
                    "ROCM_PATH",
                    "MIOPEN_CK_LIB_PATH",
                    "HSA_XNACK",
                    "ASAN_OPTIONS",
                    "ROCR_VISIBLE_DEVICES",
                    "SHARD_INDEX",
                    "TOTAL_SHARDS",
                )
            },
        )

    def xml_report(self, path, label, expected=None):
        cases = list(ET.parse(path).getroot().iter("testcase"))
        rows = [
            {
                "name": case.get("name"),
                "class": case.get("classname"),
                "status": case.get("status"),
                "result": case.get("result"),
                "skipped": case.find("skipped") is not None,
                "failed": case.find("failure") is not None
                or case.find("error") is not None,
            }
            for case in cases
        ]
        self.save(label + "-identities.json", rows)
        if not rows or (expected is not None and len(rows) != expected):
            raise RuntimeError(f"{label}: missing test executions")
        if any(
            r["failed"]
            or r["skipped"]
            or r["status"] in {"notrun", "disabled"}
            or r["result"] in {"suppressed", "skipped"}
            for r in rows
        ):
            raise RuntimeError(
                f"{label}: failures or skipped tests; see identity report"
            )

    def official_tier(self, tier):
        env = dict(self.env, TEST_TYPE=tier)
        old_env = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(env)
            path = (
                self.root
                / "build_tools/github_actions/test_executable_scripts/test_runner.py"
            )
            spec = importlib.util.spec_from_file_location(
                "official_miopen_" + tier, path
            )
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            archs, excludes = runner.check_available_labels()
            command = runner.build_ctest_command(
                tier,
                "gfx950",
                archs,
                excludes,
                runner.generate_resource_spec(),
                "gfx950",
            )
            actual_env = runner.environ_vars.copy()
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        enum = [arg for arg in command if arg not in {"-V", "--output-on-failure"}]
        self.run([*enum, "--show-only=json-v1"], tier + "-enumeration", actual_env)
        tests = json.loads((self.out / (tier + "-enumeration.log")).read_text())[
            "tests"
        ]
        self.save(tier + "-selected-tests.json", tests)
        if not tests:
            raise RuntimeError(f"No official {tier} tests selected")
        for test in tests:
            for prop in test.get("properties", []):
                if prop["name"] in {"ENVIRONMENT", "ENVIRONMENT_MODIFICATION"}:
                    if any(
                        key in str(prop["value"])
                        for key in (
                            "MIOPEN_CK_LIB_PATH",
                            "LD_LIBRARY_PATH",
                            "LD_PRELOAD",
                        )
                    ):
                        raise RuntimeError(
                            f"Test overrides validated library selection: {test['name']}"
                        )
        xml = self.out / (tier + ".xml")
        code = self.run(
            [*command, "--output-junit", xml], tier, actual_env, check=False
        )
        self.xml_report(xml, tier, expected=len(tests))
        actual_names = [
            case.get("name") for case in ET.parse(xml).getroot().iter("testcase")
        ]
        if sorted(actual_names) != sorted(test["name"] for test in tests):
            raise RuntimeError(
                f"{tier}: enumerated and executed CTest identities differ"
            )
        skips = [
            line
            for line in (self.out / (tier + ".log"))
            .read_text(errors="replace")
            .splitlines()
            if re.search(r"\[\s*SKIPPED\s*\]", line)
        ]
        self.save(tier + "-reported-gtest-skips.json", skips)
        if code or skips:
            raise RuntimeError(
                f"Official {tier}: exit={code}; skipped GTests={len(skips)}"
            )

    def focused_solver(self):
        binary = self.sdk / "bin/miopen_gtest"
        self.run(
            [binary, "--gtest_list_tests", "--gtest_filter=" + FOCUS],
            "focus-list",
            self.env,
        )
        names, suite = [], None
        for line in (self.out / "focus-list.log").read_text().splitlines():
            token = line.split("#", 1)[0].strip()
            if token.endswith(".") and not line.startswith(" "):
                suite = token
            elif line.startswith("  ") and suite and token:
                name = suite + token
                if not fnmatch.fnmatchcase(name, FOCUS):
                    raise RuntimeError(f"Unexpected focused GTest: {name}")
                names.append(name)
        self.save("focus-selected-tests.json", names)
        if not names or len(names) != len(set(names)):
            raise RuntimeError("Focused CK solver filter has no unique tests")
        env = dict(
            self.env,
            MIOPEN_LOG_LEVEL="6",
            LD_DEBUG="libs,files",
            LD_DEBUG_OUTPUT=str(self.out / "focus-loader"),
        )
        xml = self.out / "focus.xml"
        code = self.run(
            [binary, "--gtest_filter=" + FOCUS, "--gtest_output=xml:" + str(xml)],
            "focus",
            env,
            check=False,
        )
        self.xml_report(xml, "focus", expected=len(names))
        actual = [
            case.get("classname") + "." + case.get("name")
            for case in ET.parse(xml).getroot().iter("testcase")
        ]
        if sorted(actual) != sorted(names):
            raise RuntimeError(
                "Focused test identities changed between enumeration/execution"
            )
        if code:
            raise RuntimeError(f"Focused solver failed: {code}")
        log = (self.out / "focus.log").read_text(errors="replace")
        expected = "Loaded CK grouped conv library from env path: " + str(self.plugin)
        if expected not in log:
            raise RuntimeError("MIOpen did not confirm loading the candidate CK plugin")
        loaded = set()
        for path in self.out.glob("focus-loader.*"):
            for line in path.read_text(errors="replace").splitlines():
                match = re.search(r"calling init:\s*(\S+)", line)
                if match:
                    loaded.add(str(Path(match.group(1)).resolve()))
        self.save("focus-loaded-libraries.json", sorted(loaded))
        if not {str(self.plugin), str(self.miopen)} <= loaded:
            raise RuntimeError(
                "Loader trace does not prove both rebuilt libraries initialized"
            )

    def archive_contract(self):
        contract = self.manifest["contract"]
        archive = contained(self.bundle, self.manifest["ck_archive_path"])
        binary = contained(self.bundle, contract["binary"])
        if (
            digest(archive) != contract["archive_sha256"]
            or digest(binary) != contract["binary_sha256"]
        ):
            raise RuntimeError("Contract/archive provenance mismatch")
        command = json.loads(
            contained(self.bundle, contract["link_command_file"]).read_text()
        )
        if command["archive_sha256"] != contract["archive_sha256"] or not any(
            Path(arg).name == "libdevice_conv_operations.a" for arg in command["argv"]
        ):
            raise RuntimeError(
                "Contract link command did not use the rebuilt CK archive"
            )
        self.run([binary, "--inventory"], "archive-inventory", self.env)
        self.run([binary], "archive-numeric", self.env)
        self.run(
            [
                sys.executable,
                HERE / "compare_contract.py",
                self.out / "baseline-numeric.log",
                self.out / "archive-numeric.log",
                "--output",
                self.out / "archive-contract-comparison.json",
            ],
            "compare-contract",
        )

    def main(self):
        self.verify_bundle()
        self.prepare_sdk()
        for label, task in (
            ("quick", lambda: self.official_tier("quick")),
            ("standard", lambda: self.official_tier("standard")),
            ("focused-solver", self.focused_solver),
            ("archive-contract", self.archive_contract),
        ):
            try:
                task()
            except Exception as error:
                self.issues.append({"stage": label, "error": str(error)})
                print(f"{label}: {error}", file=sys.stderr, flush=True)
            self.save(
                "validation-progress.json", {"last_stage": label, "issues": self.issues}
            )
        self.check_overlay()
        self.save(
            "validation-result.json",
            {
                "pass": not self.issues,
                "issues": self.issues,
                "scope": "Rebuilt full CK archive and MIOpen plugin; official quick/standard and focused FP32 backward-data solver.",
                "coverage_limit": "gfx950 only; no claim for other GPU architectures or other CK families.",
            },
        )
        if self.issues:
            raise RuntimeError(
                "Consumer validation incomplete or failed; inspect retained evidence"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("integration-bundle"))
    parser.add_argument("--sdk", type=Path, default=Path("integration-sdk"))
    parser.add_argument("--output", type=Path, default=Path("integration-gpu-evidence"))
    parser.add_argument("--variant", choices=("release", "asan-debug"), required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--producer-run-id", required=True)
    parser.add_argument(
        "--baseline-contract-run-id", default=os.getenv("BASELINE_CONTRACT_RUN_ID")
    )
    parser.add_argument("--baseline-contract-log", type=Path, required=True)
    parser.add_argument("--baseline-contract-provenance", type=Path, required=True)
    parser.add_argument("--baseline-benchmark-protocol", type=Path, required=True)
    parser.add_argument(
        "--baseline-benchmark-run-id", default=os.getenv("BASELINE_BENCHMARK_RUN_ID")
    )
    args = parser.parse_args()
    if not args.baseline_contract_run_id:
        parser.error(
            "--baseline-contract-run-id or BASELINE_CONTRACT_RUN_ID is required"
        )
    if not args.baseline_benchmark_run_id:
        args.baseline_benchmark_run_id = args.baseline_contract_run_id
    Validation(args).main()


if __name__ == "__main__":
    main()
