"""
code object version  functional test.

Build default bit_extract, detect code object version n, rebuild with n-1/n-2,
then run all binaries and verify they print "PASSED!".
"""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.exceptions import TestExecutionError
from utils.logger import log


class TargetIdBitExtractTest(FunctionalBase):
    """Validate bit_extract code object backward compatibility."""

    def __init__(self):
        super().__init__(
            test_name="cov_backward_comptability",
            display_name="Cov Backward Comptability",
        )
        self.results_json = self.script_dir / "cov_results.json"
        self.test_results: List[Dict[str, Any]] = []

        config = self.load_config("cov_backward_comp.json")
        self.sample_relative_path = config.get(
            "sample_relative_path", "projects/hip-tests/samples/0_Intro/bit_extract"
        )
        self.include_relative_path = config.get(
            "include_relative_path", "projects/hip-tests/samples/common"
        )
        self.source_file = config.get("source_file", "bit_extract.cpp")
        self.binary_prefix = config.get("binary_prefix", "bit_extract")

        self.rocm_systems_dir = self.therock_dir / "rocm-systems"
        self.sample_dir = self.rocm_systems_dir / self.sample_relative_path
        self.include_dir = self.rocm_systems_dir / self.include_relative_path

    def _execute_command_with_output(
        self, cmd: List[str], cwd: Path = None, env: Dict[str, str] = None
    ) -> Tuple[int, str]:
        work_dir = cwd or self.therock_dir
        log.info(f"++ Exec [{work_dir}]$ {shlex.join(cmd)}")

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env,
        )

        output_lines = []
        for line in process.stdout:
            line_text = line.rstrip()
            log.info(line_text)
            output_lines.append(line_text)

        process.wait()
        return process.returncode, "\n".join(output_lines)

    def _get_rocm_env_with_path(self) -> Dict[str, str]:
        env = self.get_rocm_env()
        rocm_bin = str(self.rocm_path / "bin")
        env["PATH"] = f"{rocm_bin}:{env.get('PATH', '')}".rstrip(":")
        env["HIP_PLATFORM"] = "amd"
        return env

    def _ensure_sources_ready(self) -> None:
        if not self.rocm_systems_dir.exists():
            raise TestExecutionError(
                f"rocm-systems directory not found at {self.rocm_systems_dir}\n"
                "Ensure rocm-systems is present in TheRock directory"
            )

        if not self.sample_dir.exists():
            raise TestExecutionError(f"bit_extract sample directory not found: {self.sample_dir}")
        if not (self.sample_dir / self.source_file).exists():
            raise TestExecutionError(f"Source file not found: {self.sample_dir / self.source_file}")
        if not self.include_dir.exists():
            raise TestExecutionError(f"Include path not found: {self.include_dir}")

    def _resolve_tool(self, candidates: List[Path], default_tool: str) -> str:
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return default_tool

    def _compile_sample(self, output_binary: Path, code_obj_version: int = None) -> int:
        hipcc = self._resolve_tool(
            [self.rocm_path / "bin" / "hipcc", self.rocm_path / "hip" / "bin" / "hipcc"],
            "hipcc",
        )
        cmd = [hipcc, self.source_file, "-I", str(self.include_dir)]
        if code_obj_version is not None:
            cmd.append(f"-mcode-object-version={code_obj_version}")
        cmd += ["-o", str(output_binary)]

        env = self._get_rocm_env_with_path()
        rc, _ = self._execute_command_with_output(cmd, cwd=self.sample_dir, env=env)
        return rc

    def _extract_code_object_version(self, binary_path: Path) -> int:
        """Extract code object version from bundled/offloaded binary."""
        llvm_objdump = self._resolve_tool(
            [
                self.rocm_path / "llvm" / "bin" / "llvm-objdump",
                self.rocm_path / "lib" / "llvm" / "bin" / "llvm-objdump",
            ],
            "llvm-objdump",
        )
        llvm_readelf = self._resolve_tool(
            [
                self.rocm_path / "llvm" / "bin" / "llvm-readelf",
                self.rocm_path / "lib" / "llvm" / "bin" / "llvm-readelf",
            ],
            "llvm-readelf",
        )
        env = self._get_rocm_env_with_path()

        def _summarize_output(output: str, max_lines: int = 6) -> str:
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if not lines:
                return "no output"
            if len(lines) <= max_lines:
                return " | ".join(lines)
            return " | ".join(lines[:max_lines]) + " | ..."

        def abi_to_code_object_version(abi_version: int, source_desc: str) -> int:
            # AMDGPU HSA ABI uses a fixed offset:
            #   ELFABIVERSION_AMDGPU_HSA_V2 -> ABI Version 0
            #   ELFABIVERSION_AMDGPU_HSA_V3 -> ABI Version 1
            #   ...
            # So code object major version is (ABI Version + 2), which naturally
            # supports future versions (for example ABI 5 -> V7) without code edits.
            abi_symbol_version = abi_version + 2
            if abi_version < 0 or abi_symbol_version < 2:
                raise TestExecutionError(
                    f"Invalid AMDGPU ELF ABI Version {abi_version} from {source_desc}"
                )

            abi_symbol = f"ELFABIVERSION_AMDGPU_HSA_V{abi_symbol_version}"
            code_object_version = abi_symbol_version
            log.info(
                "Resolved AMDGPU ABI Version %s (%s) to code object version %s",
                abi_version,
                abi_symbol,
                code_object_version,
            )
            return code_object_version

        offloading_reason = "offloading bundle inspection did not produce an ABI Version"
        with tempfile.TemporaryDirectory(prefix="targetid_bit_extract_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            rc, offloading_out = self._execute_command_with_output(
                [llvm_objdump, "--offloading", str(binary_path)],
                cwd=tmpdir_path,
                env=env,
            )
            if rc != 0:
                offloading_reason = (
                    f"`llvm-objdump --offloading` failed with exit code {rc}: "
                    f"{_summarize_output(offloading_out)}"
                )
            else:
                bundle_names = re.findall(
                    r"Extracting offload bundle:\s*(\S*amdgcn-amd-amdhsa--\S+)",
                    offloading_out,
                    re.IGNORECASE,
                )
                if not bundle_names:
                    offloading_reason = (
                        "`llvm-objdump --offloading` succeeded but found no AMDGPU "
                        "offload bundles"
                    )
                else:
                    bundle_errors: List[str] = []
                    for bundle_name in bundle_names:
                        bundle_path = tmpdir_path / bundle_name
                        if not bundle_path.exists():
                            bundle_errors.append(
                                f"bundle {bundle_name} was referenced but not extracted"
                            )
                            continue
                        readelf_rc, elf_header = self._execute_command_with_output(
                            [llvm_readelf, "-h", str(bundle_path)],
                            cwd=tmpdir_path,
                            env=env,
                        )
                        if readelf_rc != 0:
                            bundle_errors.append(
                                f"`llvm-readelf -h {bundle_name}` failed with exit code "
                                f"{readelf_rc}: {_summarize_output(elf_header)}"
                            )
                            continue

                        match = re.search(r"ABI Version:\s*(\d+)", elf_header)
                        if match:
                            abi_version = int(match.group(1))
                            return abi_to_code_object_version(abi_version, str(bundle_path))
                        bundle_errors.append(
                            f"`llvm-readelf -h {bundle_name}` did not contain ABI Version"
                        )

                    if bundle_errors:
                        offloading_reason = "; ".join(bundle_errors[:3])
                        if len(bundle_errors) > 3:
                            offloading_reason += "; ..."

        raise TestExecutionError(
            "Could not determine code object version from binary "
            f"{binary_path}. Offloading path: {offloading_reason}. "
            "Host-binary readelf fallback is disabled."
        )

    def _run_and_validate_binary(self, binary_path: Path) -> Tuple[int, str, str]:
        env = self._get_rocm_env_with_path()
        rc, output = self._execute_command_with_output(
            [str(binary_path)], cwd=self.sample_dir, env=env
        )
        if rc != 0:
            return rc, output, f"Binary execution failed with return code {rc}"
        if "PASSED!" not in output:
            return rc, output, "Binary did not print expected token: PASSED!"
        return rc, output, ""

    def _record_result(
        self,
        variant: str,
        command: str,
        status: str,
        return_code: int,
        detected_version: int = None,
        error: str = None,
    ) -> None:
        result: Dict[str, Any] = {
            "test_suite": "targetid_bit_extract",
            "test_case": variant,
            "command": command,
            "return_code": return_code,
            "status": status,
        }
        if detected_version is not None:
            result["detected_code_object_version"] = detected_version
        if error:
            result["error"] = error
        self.test_results.append(result)
        if status.upper() == "SKIP":
            if error:
                log.warning(f"Skipping {variant}: {error}")
            else:
                log.warning(f"Skipping {variant}: no reason provided")

    def run_tests(self) -> None:
        log.info(f"Running {self.display_name} Tests")
        self._ensure_sources_ready()

        # 1) Build with default compiler behavior (no explicit code object version),
        #    then detect current code object version n from the produced binary.
        default_variant = "default_build_detect_n"
        default_compile_cmd = f"hipcc {self.source_file} -I {self.include_dir}"
        default_binary = self.sample_dir / f"{self.binary_prefix}_default"

        rc = self._compile_sample(default_binary, code_obj_version=None)
        if rc != 0:
            self._record_result(
                default_variant,
                default_compile_cmd,
                "FAIL",
                rc,
                error="Compile failed for default build (no -mcode-object-version)",
            )
            with open(self.results_json, "w") as f:
                json.dump(self.test_results, f, indent=2)
            log.info(f"{self.display_name} results saved to {self.results_json}")
            return

        try:
            base_version = self._extract_code_object_version(default_binary)
        except TestExecutionError as e:
            self._record_result(
                default_variant,
                default_compile_cmd,
                "SKIP",
                0,
                error=(
                    "Code object version detection is not supported on this "
                    f"platform/toolchain: {e}"
                ),
            )
            with open(self.results_json, "w") as f:
                json.dump(self.test_results, f, indent=2)
            log.info(f"{self.display_name} results saved to {self.results_json}")
            return

        run_rc, _, run_err = self._run_and_validate_binary(default_binary)
        if run_err:
            self._record_result(
                default_variant,
                default_compile_cmd,
                "FAIL",
                run_rc,
                detected_version=base_version,
                error=run_err,
            )
        else:
            self._record_result(
                default_variant,
                default_compile_cmd,
                "PASS",
                0,
                detected_version=base_version,
            )

        # 2) Rebuild with n-1 and n-2 and validate detected versions + execution.
        checks = [
            ("backward_compat_n_minus_1", base_version - 1),
            ("backward_compat_n_minus_2", base_version - 2),
        ]
        for variant, target in checks:
            compile_cmd = (
                f"hipcc {self.source_file} -I {self.include_dir} "
                f"-mcode-object-version={target}"
            )

            if target < 0:
                self._record_result(
                    variant,
                    compile_cmd,
                    "SKIP",
                    0,
                    error=f"Invalid target version {target} derived from n={base_version}",
                )
                continue

            binary = self.sample_dir / f"{self.binary_prefix}_cov{target}"
            rc = self._compile_sample(binary, code_obj_version=target)
            if rc != 0:
                self._record_result(
                    variant,
                    compile_cmd,
                    "FAIL",
                    rc,
                    error=f"Compile failed for -mcode-object-version={target}",
                )
                continue

            try:
                detected = self._extract_code_object_version(binary)
            except TestExecutionError as e:
                self._record_result(
                    variant,
                    compile_cmd,
                    "SKIP",
                    0,
                    error=(
                        "Code object version detection is not supported on this "
                        f"platform/toolchain: {e}"
                    ),
                )
                continue
            if detected != target:
                self._record_result(
                    variant,
                    compile_cmd,
                    "FAIL",
                    0,
                    detected_version=detected,
                    error=f"Expected code object version {target}, but detected {detected}",
                )
                continue

            run_rc, _, run_err = self._run_and_validate_binary(binary)
            if run_err:
                self._record_result(
                    variant,
                    compile_cmd,
                    "FAIL",
                    run_rc,
                    detected_version=detected,
                    error=run_err,
                )
            else:
                self._record_result(
                    variant,
                    compile_cmd,
                    "PASS",
                    0,
                    detected_version=detected,
                )

        with open(self.results_json, "w") as f:
            json.dump(self.test_results, f, indent=2)
        log.info(f"{self.display_name} results saved to {self.results_json}")

    def parse_results(self) -> List[Dict[str, Any]]:
        log.info(f"Parsing {self.display_name} Results")
        try:
            with open(self.results_json, "r") as f:
                json_results = json.load(f)
        except FileNotFoundError:
            raise TestExecutionError(f"Results JSON file not found: {self.results_json}")
        except json.JSONDecodeError as e:
            raise TestExecutionError(f"Invalid JSON in results file: {e}")

        parsed_results = []
        for result in json_results:
            status = result.get("status", "ERROR").upper()
            if status not in ["PASS", "FAIL", "ERROR", "SKIP"]:
                status = "ERROR"

            parsed_results.append(
                self.create_test_result(
                    test_name=self.test_name,
                    subtest_name=result["test_case"],
                    status=status,
                    suite=result["test_suite"],
                    command=result.get("command", ""),
                )
            )

        return parsed_results


if __name__ == "__main__":
    run_functional_main(TargetIdBitExtractTest())
