#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit a Linux DASan install and run an explicitly selected lit test scope.

The source inventory is pinned to llvm-project b07aa864f5b2bf648e6a61e2902cbfde83192481:
3 host tests, 21 HIP tests, and 3 OpenMP tests. This script runs the host tests,
optionally runs HIP and OpenMP tests. Run after building both
compiler-rt runtime targets and HIP. A successful host run is not GPU validation.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HOST_TESTS = {"hsa-load-order.c", "hsa-missing.c", "initialize.c"}
HIP_TESTS = {
    "AMDGPU/" + name + ".hip"
    for name in (
        "access-size",
        "clean",
        "flags",
        "global-bounds",
        "global-linkage",
        "global-overflow",
        "heap-bounds",
        "heap-overflow",
        "host-overflow",
        "module-unload",
        "multi-device",
        "no-scratch",
        "quarantine",
        "recover",
        "report-location",
        "shared-bounds",
        "shared-overlay",
        "size-classes",
        "stress",
        "symbol-abi",
        "two-modules",
    )
}
DEVICE_TRIPLES = {"amdgpu-amd-amdhsa", "amdgcn-amd-amdhsa"}
OPENMP_TESTS = {
    "AMDGPU/openmp-device-alloc.cpp",
    "AMDGPU/openmp-overflow.cpp",
    "AMDGPU/openmp-section-map.c",
}


def run_command(
    command: list[str], log: Path, env: dict[str, str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(command), flush=True)
    result = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    log.write_text(result.stdout, encoding="utf-8")
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result


def check_lit_results(
    report_path: Path, expected: set[str], allowed_unsupported: set[str]
) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results: dict[str, str] = {}
    for test in report["tests"]:
        if test["code"] == "EXCLUDED":
            continue
        name = test["name"].split(" :: ", 1)[-1].replace("\\", "/")
        if name in results:
            raise RuntimeError(
                f"Duplicate lit test {name}; select one host configuration"
            )
        results[name] = test["code"]
    if results.keys() != expected:
        raise RuntimeError(
            "lit inventory mismatch; missing="
            f"{sorted(expected - results.keys())}, "
            f"unexpected={sorted(results.keys() - expected)}. "
            "GPU tests can disappear when the device runtime is missing."
        )
    failures = {
        name: code
        for name, code in results.items()
        if code != "PASS"
        and not (code == "UNSUPPORTED" and name in allowed_unsupported)
    }
    if failures:
        raise RuntimeError(f"lit tests did not pass: {failures}")
    return results


def audit(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output = args.output_dir.resolve()
    rocm = args.rocm_path.resolve()
    clang = args.clang.resolve() if args.clang else rocm / "lib/llvm/bin/clang"
    if not clang.is_file():
        raise RuntimeError(f"Compiler missing: {clang}; pass --clang explicitly")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(rocm / "lib"), str(rocm / "lib/rocm_sysdeps/lib")]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    env["PATH"] = str(clang.parent) + os.pathsep + env.get("PATH", "")
    resource = Path(
        run_command(
            [str(clang), "-print-resource-dir"], output / "resource.log", env
        ).stdout.strip()
    ).resolve()
    libraries = sorted(resource.rglob("libclang_rt.dasan*"))
    host = [p for p in libraries if not DEVICE_TRIPLES.intersection(p.parts)]
    device = [p for p in libraries if DEVICE_TRIPLES.intersection(p.parts)]
    required = {
        "host static": [p for p in host if p.name == "libclang_rt.dasan.a"],
        "host shared": [p for p in host if p.name == "libclang_rt.dasan.so"],
        "host preinit": [p for p in host if p.name == "libclang_rt.dasan-preinit.a"],
        "GPU static": [p for p in device if p.name == "libclang_rt.dasan.a"],
    }
    summary["resource_dir"] = str(resource)
    summary["libraries"] = {k: [str(p) for p in v] for k, v in required.items()}
    for kind, paths in required.items():
        if not paths or not all(p.is_file() and p.stat().st_size for p in paths):
            raise RuntimeError(f"Missing or empty {kind} runtime under {resource}")

    host_source = output / "initialize.c"
    host_source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    spellings = {}
    for spelling in ("daddress", "dasan"):
        result = run_command(
            [str(clang), f"-fsanitize={spelling}", "-fsyntax-only", str(host_source)],
            output / f"driver-{spelling}.log",
            env,
            check=False,
        )
        spellings[spelling] = result.returncode == 0
    summary["driver_spellings"] = spellings
    if not spellings["daddress"]:
        raise RuntimeError("Compiler does not accept -fsanitize=daddress")
    if not spellings["dasan"]:
        print("NOTE: -fsanitize=dasan is unsupported; branch uses -fsanitize=daddress.")

    # Exercise the installed runtime as well as the build-tree runtimes used by lit.
    for linkage in ("static", "shared"):
        executable = output / f"initialize-{linkage}"
        flags = []
        if linkage == "shared":
            flags = [
                "-shared-libsan",
                "-Wl,-rpath," + str(required["host shared"][0].parent),
            ]
        run_command(
            [str(clang), "-fsanitize=daddress", str(host_source), "-o", str(executable)]
            + flags,
            output / f"initialize-{linkage}-build.log",
            env,
        )
        result = run_command(
            [str(executable)],
            output / f"initialize-{linkage}-run.log",
            {**env, "DASAN_OPTIONS": "verbosity=1"},
        )
        if "DeviceAddressSanitizer: initialized" not in result.stdout:
            raise RuntimeError(
                f"Installed {linkage} runtime did not report initialization"
            )
    summary["installed_host_initialization"] = "static and shared passed"

    hip_source = output / "device-smoke.hip"
    hip_source.write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void write_element(int *p, int i) { p[i] = 1; }\n"
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    hip_flags = [
        "-xhip",
        "--rocm-path=" + str(rocm),
        "--hip-path=" + str(rocm),
        "--hip-device-lib-path=" + str(rocm / "lib/llvm/amdgcn/bitcode"),
        "-isystem",
        str(rocm / "include"),
    ]
    compiled_architectures: list[str] = []
    summary["compile_link_architectures"] = compiled_architectures
    for arch in args.compile_arch:
        ir_path = output / f"device-{arch}.ll"
        run_command(
            [
                str(clang),
            ]
            + hip_flags
            + [
                f"--offload-arch={arch}",
                "-fsanitize=daddress",
                "-O1",
                "--offload-device-only",
                "-S",
                "-emit-llvm",
                str(hip_source),
                "-o",
                str(ir_path),
            ],
            output / f"instrumentation-{arch}.log",
            env,
        )
        if "__dasan_report_store" not in ir_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"No DASan store instrumentation for {arch}: {ir_path}")
        command = (
            [
                str(clang),
            ]
            + hip_flags
            + [
                f"--offload-arch={arch}",
                "-fsanitize=daddress",
                "-O1",
                "-g",
                str(hip_source),
                "-L" + str(rocm / "lib"),
                "-lamdhip64",
                "-Wl,-rpath," + str(rocm / "lib"),
                "-o",
                str(output / f"smoke-{arch}"),
            ]
        )
        run_command(command, output / f"compile-link-{arch}.log", env)
        compiled_architectures.append(arch)

    lit_command = [str(args.llvm_lit.resolve())]
    if args.llvm_lit.suffix == ".py":
        lit_command.insert(0, sys.executable)
    lit_command += ["-sv", "-D", "rocm_root=" + str(rocm)]
    lit_suite = args.lit_suite.resolve()
    if args.openmp or args.gpu_device_count is not None:
        # The branch probes GPU access during CMake configuration. A build-only
        # container may see no GPUs even when the tests later run on a GPU host.
        # Its configured fields use set_default(), so an explicit site overlay
        # can supply the execution host's capabilities without editing sources.
        sites = list(lit_suite.rglob("lit.site.cfg.py"))
        if len(sites) != 1:
            raise RuntimeError(
                "Expected exactly one host lit.site.cfg.py under --lit-suite; "
                f"found {len(sites)}"
            )
        overlay = output / "lit-overlay"
        overlay.mkdir(exist_ok=True)
        settings = []
        if args.openmp:
            settings.append("config.can_run_openmp_offload = True")
        if args.gpu_device_count is not None:
            settings.append(f"config.gpu_device_count = {args.gpu_device_count}")
        settings.append(f"lit_config.load_config(config, {str(sites[0])!r})")
        (overlay / "lit.site.cfg.py").write_text(
            "\n".join(settings) + "\n", encoding="utf-8"
        )
        lit_suite = overlay
    gpu_parameters = []
    if args.gpu_arch:
        gpu_parameters = [
            "-D",
            "amdgpu_arch=" + args.gpu_arch,
            "-D",
            "hip_lib_path=" + str(rocm / "lib"),
        ]
    host_report = output / "host-lit.json"
    host_result = run_command(
        lit_command
        + gpu_parameters
        + [
            "--filter",
            r":: (hsa-load-order|hsa-missing|initialize)\.c$",
            "-o",
            str(host_report),
            str(lit_suite),
        ],
        output / "host-lit.log",
        env,
        check=False,
    )
    summary["host_tests"] = check_lit_results(host_report, HOST_TESTS, set())
    if host_result.returncode:
        raise RuntimeError("Host lit returned nonzero; see host-lit.log")

    if args.gpu_arch:
        gpu_report = output / "gpu-lit.json"
        gpu_result = run_command(
            lit_command
            + gpu_parameters
            + [
                "--filter",
                r":: AMDGPU/.*\.hip$",
                "-o",
                str(gpu_report),
                str(lit_suite),
            ],
            output / "gpu-lit.log",
            env,
            check=False,
        )
        allowed = set() if args.require_multi_device else {"AMDGPU/multi-device.hip"}
        summary["gpu_tests"] = check_lit_results(gpu_report, HIP_TESTS, allowed)
        if gpu_result.returncode:
            raise RuntimeError("GPU lit returned nonzero; see gpu-lit.log")
    if args.openmp:
        openmp_report = output / "openmp-lit.json"
        openmp_result = run_command(
            lit_command
            + gpu_parameters
            + [
                "--filter",
                r":: AMDGPU/openmp-.*\.(c|cpp)$",
                "-o",
                str(openmp_report),
                str(lit_suite),
            ],
            output / "openmp-lit.log",
            env,
            check=False,
        )
        summary["openmp_tests"] = check_lit_results(openmp_report, OPENMP_TESTS, set())
        if openmp_result.returncode:
            raise RuntimeError("OpenMP lit returned nonzero; see openmp-lit.log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-path", type=Path, required=True)
    parser.add_argument("--clang", type=Path, help="Default: ROCM/lib/llvm/bin/clang")
    parser.add_argument("--llvm-lit", type=Path, required=True)
    parser.add_argument(
        "--lit-suite",
        type=Path,
        required=True,
        help="Configured HOST compiler-rt test/dasan directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compile-arch",
        action="append",
        help="Compile/link target (repeatable); default: gfx942 and gfx950",
    )
    parser.add_argument(
        "--gpu-arch", help="Explicit local GPU target; enables HIP execution"
    )
    parser.add_argument(
        "--require-multi-device",
        action="store_true",
        help="Reject multi-device test being unsupported",
    )
    parser.add_argument(
        "--gpu-device-count",
        type=int,
        help="Execution host GPU count when build-time detection differed",
    )
    parser.add_argument(
        "--openmp",
        action="store_true",
        help="Run all three OpenMP tests; requires working built offload runtimes",
    )
    args = parser.parse_args()
    if args.require_multi_device and not args.gpu_arch:
        parser.error("--require-multi-device requires --gpu-arch")
    if (args.openmp or args.gpu_device_count is not None) and not args.gpu_arch:
        parser.error("--openmp and --gpu-device-count require --gpu-arch")
    if args.gpu_device_count is not None and args.gpu_device_count < 1:
        parser.error("--gpu-device-count must be positive")
    args.compile_arch = args.compile_arch or ["gfx942", "gfx950"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "status": "failed",
        "gpu_tests": "not completed" if args.gpu_arch else "not run (no --gpu-arch)",
        "openmp_tests": "not completed" if args.openmp else "not run (no --openmp)",
    }
    try:
        audit(args, summary)
        summary["status"] = "passed"
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        summary["error"] = str(error)
        print(f"ERROR: {error}", file=sys.stderr)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
