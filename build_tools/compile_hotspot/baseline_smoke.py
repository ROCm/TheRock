# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Preflight existing CK harness API/cases against untouched release artifacts.

This is baseline harness validation only, not candidate or performance evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE_SHA = "5b7cb88373362ab80dc0f19605915e534f4158aa"
BASELINE_RUN = "41149779244"
BASELINE_SHA = "3fa23328ec595fe1304c98d9e8c074b3ead97dd0"
SOURCE_TREE = "5e50660a18cb52c7db2e676b8a2618672da3cef9"
HERE = Path(__file__).resolve().parent


def fingerprint(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return {"path": str(path), "size": path.stat().st_size, "sha256": hasher.hexdigest()}


class Smoke:
    def __init__(self, args):
        self.root = Path.cwd().resolve()
        self.out = args.output.resolve()
        self.sdk = args.sdk.resolve()
        self.cache = args.download_cache.resolve()

    def save(self, name, data):
        (self.out / name).write_text(json.dumps(data, indent=2) + "\n")

    def run(self, argv, label, env=None):
        argv = list(map(str, argv))
        self.save(label + "-command.json", {"argv": argv, "cwd": str(self.root)})
        print(shlex.join(argv), flush=True)
        with (self.out / (label + ".log")).open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
        self.save(label + "-result.json", {"exit_code": result.returncode})
        result.check_returncode()

    def main(self):
        for directory in (self.out, self.sdk, self.cache):
            if directory.exists():
                raise RuntimeError(f"Preflight requires a fresh directory: {directory}")
        self.out.mkdir(parents=True)
        self.save("result.json", {"pass": False, "status": "preflight started", "scope": __doc__})
        if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != SOURCE_SHA:
            raise RuntimeError("Product checkout differs from pinned source")
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
        if tree != SOURCE_TREE:
            raise RuntimeError("Product tree differs from normal baseline")
        sys.path.insert(0, str(self.root / "build_tools"))
        from _therock_utils.build_topology import BuildTopology

        topology = BuildTopology(str(self.root / "BUILD_TOPOLOGY.toml"))
        imported = topology.get_inbound_artifacts("math-libs") | {"composable-kernel"}
        excluded = set(topology.artifacts) - imported
        self.save("context.json", {
            "baseline_run_id": BASELINE_RUN, "baseline_sha": BASELINE_SHA,
            "source_sha": SOURCE_SHA, "source_tree": tree, "variant": "release",
            "instrumentation_sha": os.getenv("GITHUB_SHA"), "smoke_run_id": os.getenv("GITHUB_RUN_ID"),
            "imported_artifacts": sorted(imported), "architecture": "gfx950",
            "harness": fingerprint(HERE / "ck_f32_contract.cpp"),
            "comparator": fingerprint(HERE / "compare_contract.py"), "scope": __doc__,
        })
        self.run([
            sys.executable, "build_tools/artifact_manager.py", "fetch",
            "--run-id=" + BASELINE_RUN, "--run-github-repo=ROCm/TheRock",
            "--stage=all", "--platform=linux", "--amdgpu-families=gfx950", "--amdgpu-targets=gfx950",
            "--flatten", "--output-dir=" + str(self.sdk), "--download-cache-dir=" + str(self.cache),
            "--exclude-artifacts=" + ",".join(sorted(excluded)), "--exclude-components=dbg,doc,test",
        ], "fetch-baseline")
        self.save("artifact-archives.json", [fingerprint(p) for p in sorted(self.cache.rglob("*.tar.*"))])
        compiler = self.sdk / "lib/llvm/bin/clang++"
        archive = self.sdk / "lib/libdevice_conv_operations.a"
        utility = self.sdk / "lib/libutility.a"
        for path in (compiler, archive, utility, self.sdk / "include/ck/ck.hpp"):
            if not path.is_file():
                raise RuntimeError(f"Baseline SDK is missing {path}")
        environment = dict(os.environ)
        for key in ("LD_PRELOAD", "LD_DEBUG", "LD_DEBUG_OUTPUT", "HSA_XNACK"):
            environment.pop(key, None)
        environment.update({
            "LD_LIBRARY_PATH": ":".join(str(self.sdk / p) for p in
                ("lib", "lib64", "lib/rocm_sysdeps/lib", "lib/llvm/lib") if (self.sdk / p).is_dir()),
            "ROCM_PATH": str(self.sdk), "CCACHE_DISABLE": "1",
        })
        self.run([compiler, "--version"], "compiler-version", environment)
        self.run([self.sdk / "lib/llvm/bin/llvm-ar", "t", archive], "archive-members", environment)
        members = (self.out / "archive-members.log").read_text().splitlines()
        original = "device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp.o"
        if members.count(original) != 1 or any(
            name in members for name in (
                original.replace("_instance.cpp.o", "_noshuffle_instance.cpp.o"),
                original.replace("_instance.cpp.o", "_nongrouped_match_instance.cpp.o"),
            )
        ):
            raise RuntimeError("Expected untouched baseline CK archive membership")
        binary = self.out / "baseline-contract"
        # Standalone harness flags do not claim benchmark flag parity. The archive
        # is untouched; this does not rebuild the 90 registered instantiations.
        command = [compiler, "-std=c++20", "-O2", "-DNDEBUG", "-x", "hip",
                   "--offload-arch=gfx950", "--hip-path=" + str(self.sdk),
                   "-I" + str(self.sdk / "include"), HERE / "ck_f32_contract.cpp",
                   "-x", "none", archive, utility, "--hip-link",
                   "-L" + str(self.sdk / "lib"), "-lamdhip64", "-o", binary]
        self.save("input-provenance.json", {
            "compiler": fingerprint(compiler), "ck_archive": fingerprint(archive),
            "utility_archive": fingerprint(utility),
            "ck_headers": [fingerprint(p) for p in sorted((self.sdk / "include/ck").rglob("*")) if p.is_file()],
            "runtime_env": {key: environment.get(key) for key in
                            ("LD_LIBRARY_PATH", "ROCM_PATH", "HSA_XNACK", "ROCR_VISIBLE_DEVICES")},
        })
        self.run(command, "compile-and-link", environment)
        self.save("binary.json", fingerprint(binary))
        self.run(["ldd", binary], "dependencies", environment)
        self.run([binary, "--inventory"], "baseline-inventory", environment)
        self.run([binary], "baseline-numeric", environment)
        self.run([sys.executable, HERE / "compare_contract.py",
                  self.out / "baseline-inventory.log", self.out / "baseline-numeric.log",
                  "--inventory-only", "--output", self.out / "inventory-comparison.json"], "compare-inventory")
        # Self-comparison invokes all numeric, architecture, and execution checks.
        # It cannot establish equivalence to a candidate.
        self.run([sys.executable, HERE / "compare_contract.py",
                  self.out / "baseline-numeric.log", self.out / "baseline-numeric.log",
                  "--output", self.out / "numeric-self-comparison.json"], "compare-numeric-self")
        self.save("result.json", {"pass": True, "scope": __doc__, "baseline_run_id": BASELINE_RUN,
                                 "candidate_tested": False, "performance_comparison": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("baseline-smoke-evidence"))
    parser.add_argument("--sdk", type=Path, default=Path("baseline-smoke-sdk"))
    parser.add_argument("--download-cache", type=Path, default=Path("baseline-smoke-downloads"))
    smoke = Smoke(parser.parse_args())
    try:
        smoke.main()
    except Exception as error:
        if smoke.out.is_dir():
            smoke.save("failure.json", {"error": str(error), "scope": __doc__})
        raise


if __name__ == "__main__":
    main()
