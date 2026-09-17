#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the fail-closed hipDNN CPU-only host-ASAN admissions.

The integration component is discovered from its installed CTest file so this
runner audits the artifact that it actually executes. Each positive selector
must have the reviewed count and exact-name digest before any case runs.
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from host_asan_instrumentation import (
    native_host_asan_environment,
    require_direct_clang_asan,
)


UNIT_FILTER = (
    "TestIsMetaJsonFile.*:TestMetaJsonPath.*:TestLoadBundleMetadata.*:"
    "TestCheckVramRequirement.*:TestCheckArchCompatibility.*:"
    "TestParseSupportClaimsJson.*:TestSupportJsonPath.*:"
    "TestLoadSupportClaims.*:TestParseSweepSupportClaimsJson.*:"
    "TestLoadSweepSupportClaims.*:TestSupportClaimLocator.*:"
    "TestGraphDescription.*:TestNodeTypeToString.*:TestGlobMatch.*:"
    "TestCurrentPlatform.*:TestSupportMatrixCollector.*:"
    "TestConfigUninitialized.*:TestParseVerificationMode.*:"
    "TestResolveVerificationMode.*:TestResolveGoldenDataDir.*:"
    "TestConfigInitialized.*:TestSettingsParser.*:TestTomlGuards.*:"
    "TestGpuConvolutionFwdSignatureKey.*:TestGpuConvolutionFwdPlanBuilder.*:"
    "TestGpuLayernormBwdSignatureKey.*:TestGpuLayernormBwdPlanBuilder.*:"
    "TestGpuLayernormFwdSignatureKey.*:TestGpuLayernormFwdPlanBuilder.*:"
    "TestGpuSdpaFwdSignatureKey.*:TestGpuRMSNormFwdSignatureKey.*:"
    "TestGpuRMSNormBwdSignatureKey.*:TestGpuRMSNormFwdPlanBuilder.*:"
    "TestGpuRMSNormBwdPlanBuilder.*:TestReferenceGraphExecutorFactory.*:"
    "TestBundleDiscoveryFixture.*:TestGraphFile.*:TestSanitizeForGtest.*:"
    "TestGoldenHarnessFixture.*:TestInputFillRecipes.*:TestFillInputs.*:"
    "TestToleranceResolver.*:TestVerificationModePathsFixture.*:"
    "TestErrorPaths.*:TestLoadedEngineTable.*:TestReferenceOpCoverage.*:"
    "TestSupportClaimReport.*:TestSupportClaimCoverageRules.*:"
    "TestSupportClaimSummary.*:TestSupportClaimEnforcement.*:"
    "TestEnforcementRungs.*:TestSupportVerdict.*:TestUnpinnedRunEnforcement.*:"
    "TestProductionPolicy.*:"
    "TestGraphSession.*:TestOutputComparison.*:"
    "TestGpuReferenceGraphExecutor.IsNotApplicableForRuntimePassByValueGraph:"
    "TestGpuReferenceGraphExecutor.IsApplicableForBakedScalarGraph:"
    "TestGpuReferenceGraphExecutor."
    "IsApplicableWhenRuntimePassByValueTensorIsNotConsumed:"
    "TestGpuReferenceGraphExecutor."
    "IsNotApplicableWhenSupportedOpTensorIsRuntimePassByValue:"
    "TestGpuReferenceGraphExecutor."
    "IsNotApplicableWhenSupportedOpOutputIsRuntimePassByValue:"
    "TestGpuReferenceGraphExecutor."
    "IsNotApplicableWhenConvInputIsRuntimePassByValue:"
    "TestGpuReferenceGraphExecutor."
    "IsNotApplicableWhenOptionalPointwiseOperandIsRuntimePassByValue:"
    "TestGpuSdpaFwdPlanBuilder.PlanConstruction:"
    "TestGpuSdpaFwdPlanBuilder.IsApplicable:"
    "TestGpuSdpaFwdPlanBuilder.IsNotApplicableForUnsupportedModes:"
    "TestGpuSdpaFwdPlanBuilder.IsApplicableWithStatsOutput:"
    "TestGpuSdpaFwdPlanBuilder.ProbabilityModeKeyedOnInputsNotOutput:"
    "TestGpuSdpaFwdPlanBuilder.ThrowsOnBothCausalFlags:"
    "TestGpuSdpaFwdPlanBuilder.ThrowsOnInvalidBounds:"
    "TestVerificationRouting.BundleDiscoveryFindsOnlyAuthoredBundleData:"
    "TestBundleReferenceValidationHarness."
    "SetUpFailsForABundleRegisteredWithNoGoldenData:"
    "TestBundleReferenceValidationHarness."
    "SetUpFailsForABundleRegisteredWithNoTensorData:"
    "TestBundleReferenceValidationHarness."
    "UseDeviceStaysOffTheDeviceWhenTheExecutorReportsNoDeviceNeed:"
    "TestBundleReferenceValidationHarness.InapplicableReferenceFailsRatherThanSkips:"
    "TestBundleReferenceValidationHarness.CapabilityErrorFromTheReferenceFails:"
    "TestBundleReferenceValidationHarness.ReferenceThatThrowsIsReportedWithItsMessage:"
    "TestBundleReferenceValidationHarness.MatchingReferenceOutputPasses:"
    "TestBundleReferenceValidationHarness.DriftedReferenceOutputFailsNamingTheTensor"
)

GPU_REF_FILTER = "TestConvolutionValidation.*:TestGpuRefHipError.*"

GOLDEN_FILTER = (
    "Inference_CpuRef.small_fp32_nchw:Inference_CpuRef.large_fp32_nchw:"
    "Inference_CpuRef.tiny_fp32_nchw:Inference_CpuRef.small_fp16_nchw:"
    "Inference_CpuRef.small_bfp16_nchw:Inference_CpuRef.small_fp32_ncdhw:"
    "ncdhw_fp32_Small_CpuRef.Small:nchw_bfp16_Small_CpuRef.Small:"
    "nchw_fp16_Small_CpuRef.Small:nchw_fp32_Large_CpuRef.Large:"
    "nchw_fp32_MIOpen_CpuRef.MIOpen:nchw_fp32_Small_CpuRef.Small"
)

INTEGRATION_SELECTIONS = {
    "hipdnn_integration_tests_unit_tests_host-asan_suite": {
        "binary": "hipdnn_integration_tests_unit_tests",
        "filter": UNIT_FILTER,
        "count": 515,
        "sha256": "f9d25ca6b657f7af5f9bae6411f1a3620d39b220d74adbe58d8ea2e7edc42a67",
    },
    "hipdnn_gpu_ref_tests_host-asan_suite": {
        "binary": "hipdnn_gpu_ref_tests",
        "filter": GPU_REF_FILTER,
        "count": 14,
        "sha256": "68e333b6d9877a26f1071f856a719674a33225be53e2b458ab922546e6a4e445",
    },
    "hipdnn_golden_data_tests_host-asan_suite": {
        "binary": "hipdnn_golden_data_tests",
        "filter": GOLDEN_FILTER,
        "count": 12,
        "sha256": "7f247adfa0b386f26fe050b10f856064ad9c7a45df2936e2cd4c39c0e15c87f4",
        "golden_data": "lib/integration-test-bundles/quick/BatchnormFwdInference",
    },
}

EXPECTED_ENGINE = {
    "name": "TEST_GOOD_DEFAULT_ENGINE",
    "id": "0x5FB06B52DB2039AC",
    "plugin": "test_good_default_plugin",
    "version": "1.0.0",
    "type": "HIPDNN_PLUGIN_TYPE_ENGINE",
}

_ROCRAND_NEEDED_RE = re.compile(r"NEEDED.*librocrand\.so(?:\.\d+)*")


def _with_option(value: str, option: str) -> str:
    return f"{value}:{option}" if value else option


def _test_environment(prefix: Path) -> dict[str, str]:
    env = native_host_asan_environment()
    detect_leaks = (
        "detect_leaks=0"
        if env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1"
        else "detect_leaks=1"
    )
    env["ASAN_OPTIONS"] = _with_option(env.get("ASAN_OPTIONS", ""), detect_leaks)
    env["ASAN_OPTIONS"] = _with_option(env["ASAN_OPTIONS"], "halt_on_error=1")
    env["LSAN_OPTIONS"] = _with_option(env.get("LSAN_OPTIONS", ""), "exitcode=23")
    library_dirs = (
        prefix / "lib",
        prefix / "lib" / "rocm_sysdeps" / "lib",
        prefix / "lib" / "llvm" / "lib",
    )
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(path) for path in library_dirs), existing]
    ).rstrip(os.pathsep)
    return env


def _require_no_gpu_nodes() -> None:
    present = [path for path in (Path("/dev/kfd"), Path("/dev/dri")) if path.exists()]
    if present:
        raise RuntimeError(f"host-only runner unexpectedly exposes GPU nodes: {present}")


def _read_dynamic(executable: Path, env: dict[str, str]) -> str:
    if not executable.is_file():
        raise RuntimeError(f"required host-ASAN artifact is missing: {executable}")
    result = subprocess.run(
        ["readelf", "-d", str(executable)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout


def _require_direct_asan(executable: Path, env: dict[str, str]) -> None:
    require_direct_clang_asan(executable, env)


def _require_rocrand_closure(
    executable: Path, prefix: Path, env: dict[str, str]
) -> None:
    if not _ROCRAND_NEEDED_RE.search(_read_dynamic(executable, env)):
        raise RuntimeError(f"GPU-reference test no longer declares rocRAND: {executable}")
    result = subprocess.run(
        ["ldd", str(executable)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    match = re.search(r"librocrand\.so(?:\.\d+)*\s+=>\s+(\S+)", result.stdout)
    if not match or match.group(1) == "not":
        raise RuntimeError("installed hipDNN integration artifact cannot resolve rocRAND")
    resolved = Path(match.group(1)).resolve()
    if prefix not in resolved.parents:
        raise RuntimeError(f"rocRAND resolved outside the installed artifact: {resolved}")


def _parse_gtest_names(output: str) -> list[str]:
    names: list[str] = []
    suite = None
    for raw_line in output.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line[0].isspace() and raw_line.rstrip().endswith("."):
            suite = raw_line.split("#", 1)[0].strip()
            continue
        if suite is not None and raw_line[0].isspace():
            case = raw_line.split("#", 1)[0].strip()
            if case:
                names.append(f"{suite}{case}")
    return names


def _inventory_digest(names: list[str]) -> str:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _validate_inventory(names: list[str], expected: dict) -> None:
    digest = _inventory_digest(names)
    if len(names) != expected["count"] or digest != expected["sha256"]:
        raise RuntimeError(
            "hipDNN host-ASAN inventory changed before execution: "
            f"expected count={expected['count']}, sha256={expected['sha256']}; "
            f"got count={len(names)}, sha256={digest}"
        )


def _installed_ctest_commands(
    bin_dir: Path, env: dict[str, str]
) -> dict[str, dict]:
    test_dir = bin_dir / "hipdnn_integration_tests_ctest"
    if not (test_dir / "CTestTestfile.cmake").is_file():
        raise RuntimeError(f"installed hipDNN integration CTest file is missing: {test_dir}")
    result = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(test_dir),
            "-L",
            "^host-asan$",
            "--show-only=json-v1",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    try:
        tests = json.loads(result.stdout).get("tests", [])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid installed CTest JSON inventory: {error}") from error
    commands = {test.get("name"): test for test in tests}
    if set(commands) != set(INTEGRATION_SELECTIONS):
        raise RuntimeError(
            "installed hipDNN host-ASAN categories changed: "
            f"expected {sorted(INTEGRATION_SELECTIONS)}; got {sorted(commands)}"
        )
    return commands


def _validate_installed_command(
    test: dict, expected: dict, prefix: Path
) -> tuple[list[str], Path]:
    command = list(test.get("command") or [])
    if not command or Path(command[0]).name != expected["binary"]:
        raise RuntimeError(f"unexpected installed host-ASAN command: {command}")
    properties = {
        prop.get("name"): prop.get("value") for prop in test.get("properties", [])
    }
    cwd = Path(
        properties.get(
            "WORKING_DIRECTORY", prefix / "bin" / "hipdnn_integration_tests_ctest"
        )
    )
    executable = Path(command[0])
    if not executable.is_absolute():
        executable = (cwd / executable).resolve()
    command[0] = str(executable)
    expected_filter = f"--gtest_filter={expected['filter']}"
    if command.count(expected_filter) != 1:
        raise RuntimeError(
            f"installed {expected['binary']} selector does not match the reviewed filter"
        )
    if "golden_data" in expected:
        data_dir = (prefix / expected["golden_data"]).resolve()
        if not data_dir.is_dir():
            raise RuntimeError(f"required installed BatchNorm golden data is missing: {data_dir}")
        try:
            reference_index = command.index("--reference")
            data_index = command.index("--gd")
        except ValueError as error:
            raise RuntimeError("golden host-ASAN command lost its CPU/data arguments") from error
        command_data = Path(command[data_index + 1])
        if not command_data.is_absolute():
            command_data = (cwd / command_data).resolve()
        if command[reference_index + 1] != "cpu" or command_data != data_dir:
            raise RuntimeError("golden host-ASAN command does not use the exact CPU data root")
    return command, cwd


def _run_gtest(
    command: list[str], cwd: Path, expected: dict, env: dict[str, str]
) -> None:
    executable = Path(command[0])
    _require_direct_asan(executable, env)
    listed = subprocess.run(
        [*command, "--gtest_list_tests"],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    names = _parse_gtest_names(listed.stdout)
    _validate_inventory(names, expected)

    with tempfile.TemporaryDirectory(prefix="hipdnn-host-asan-") as tmp:
        xml_path = Path(tmp) / "gtest.xml"
        run_command = [*command, f"--gtest_output=xml:{xml_path}"]
        print(f"++ Exec {shlex.join(run_command)}", flush=True)
        result = subprocess.run(
            run_command, cwd=cwd, capture_output=True, text=True, env=env
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, run_command)
        if not xml_path.is_file():
            raise RuntimeError(f"GTest did not produce its required XML: {executable}")
        root = ET.parse(xml_path).getroot()
        cases = list(root.iter("testcase"))
        bad = [
            case
            for case in cases
            if case.find("failure") is not None
            or case.find("error") is not None
            or case.find("skipped") is not None
            or case.get("status") == "notrun"
        ]
        if len(cases) != expected["count"] or bad:
            raise RuntimeError(
                f"unexpected GTest XML result for {executable.name}: "
                f"expected {expected['count']} executed with no failures/skips; "
                f"got {len(cases)} cases and {len(bad)} bad"
            )


def _run_integration(bin_dir: Path, prefix: Path, env: dict[str, str]) -> None:
    tests = _installed_ctest_commands(bin_dir, env)
    for name, expected in INTEGRATION_SELECTIONS.items():
        command, cwd = _validate_installed_command(tests[name], expected, prefix)
        if expected["binary"] == "hipdnn_gpu_ref_tests":
            _require_rocrand_closure(Path(command[0]), prefix, env)
        _run_gtest(command, cwd, expected, env)


def _run_install_engine(bin_dir: Path, prefix: Path, env: dict[str, str]) -> None:
    executable = bin_dir / "hipdnn_list_engines"
    plugin_dir = prefix / "lib" / "test_plugins" / "default"
    plugin = plugin_dir / "libtest_good_default_plugin.so"
    _require_direct_asan(executable, env)
    _require_direct_asan(plugin, env)
    command = [str(executable), "--plugin-dir", str(plugin_dir)]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    output = result.stdout + result.stderr
    engines = re.findall(r"^  (\S+) \((0x[0-9A-F]+)\)$", output, re.MULTILINE)
    if engines != [(EXPECTED_ENGINE["name"], EXPECTED_ENGINE["id"])]:
        raise RuntimeError(f"unexpected synthetic hipDNN engine inventory: {engines}")
    required = (
        f"Plugin:  {EXPECTED_ENGINE['plugin']}",
        f"Version: {EXPECTED_ENGINE['version']}",
        f"Type:    {EXPECTED_ENGINE['type']}",
    )
    if any(value not in output for value in required):
        raise RuntimeError("synthetic hipDNN engine metadata changed")


def main() -> int:
    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    prefix = bin_dir.parent
    component = os.environ["TEST_COMPONENT"]
    _require_no_gpu_nodes()
    env = _test_environment(prefix)
    if component == "hipdnn-integration-tests":
        _run_integration(bin_dir, prefix, env)
    elif component in {"hipdnn_install", "hipdnn-install"}:
        _run_install_engine(bin_dir, prefix, env)
    else:
        raise RuntimeError(f"unsupported hipDNN host-ASAN component: {component!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
