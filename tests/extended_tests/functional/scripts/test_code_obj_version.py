"""
AMDGPU Code Object Version Backward Compatibility Functional Test.

Validates that the ROCm toolchain can produce and execute HIP binaries
targeting older code object versions, ensuring backward compatibility
across the AMDGPU runtime.

Test procedure:
  1. Compile a self-contained HIP program (sources/cov_vecadd.hip) with
     hipcc using default compiler settings (no explicit
     -mcode-object-version flag).
  2. Detect the code object version (n) of the resulting binary.
  3. Recompile with -mcode-object-version=n-1 and n-2 to produce binaries
     targeting the two prior code object versions.
  4. For each variant, verify:
     - The detected code object version in the binary matches the requested
       version.
     - The binary executes successfully (exit code 0).

The HIP source lives in the test tree (sources/cov_vecadd.hip) so the test
does not depend on the rocm-systems submodule being initialised.
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.exceptions import TestExecutionError
from utils.logger import log


class CovBackwardCompatibilityTest(FunctionalBase):
    """Validate code-object-version backward compatibility."""

    def __init__(self):
        super().__init__(
            test_name="cov_backward_compatibility",
            display_name="Cov Backward Compatibility",
        )
        self.test_results: List[Dict[str, Any]] = []

        self.source_file = "cov_vecadd.hip"
        self.binary_prefix = "cov_vecadd"

        # Source lives alongside the test scripts in the repo, under sources/.
        self.sources_dir = Path(__file__).resolve().parents[1] / "sources"

        # Build directory for compiled binaries (created/cleaned in run_tests).
        self.build_dir = Path(__file__).resolve().parents[1] / "build" / "cov"

    def _resolve_tool(self, candidates: List[Path], default_tool: str) -> str:
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return default_tool

    def _compile_sample(self, output_binary: Path, code_obj_version: int = None) -> int:
        hipcc = self._resolve_tool(
            [
                self.rocm_path / "bin" / "hipcc",
                self.rocm_path / "hip" / "bin" / "hipcc",
            ],
            "hipcc",
        )
        source_path = str(self.sources_dir / self.source_file)
        cmd = [hipcc, source_path]
        if code_obj_version is not None:
            cmd.append(f"-mcode-object-version={code_obj_version}")
        cmd += ["-o", str(output_binary)]

        env = self.get_rocm_env_with_path()
        rc, _ = self.execute_command_with_output(cmd, cwd=self.build_dir, env=env)
        return rc

    def _extract_code_object_version(self, binary_path: Path) -> int:
        """Extract code object version from bundled/offloaded binary."""
        llvm_objdump = self._resolve_tool(
            [self.rocm_path / "llvm" / "bin" / "llvm-objdump"],
            "llvm-objdump",
        )
        llvm_readelf = self._resolve_tool(
            [
                self.rocm_path / "llvm" / "bin" / "llvm-readelf",
            ],
            "llvm-readelf",
        )
        env = self.get_rocm_env_with_path()

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

        with tempfile.TemporaryDirectory(prefix="cov_backward_compat_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            rc, offloading_out = self.execute_command_with_output(
                [llvm_objdump, "--offloading", str(binary_path)],
                cwd=tmpdir_path,
                env=env,
            )
            if rc != 0:
                raise TestExecutionError(
                    f"Could not determine code object version from {binary_path}: "
                    f"`llvm-objdump --offloading` failed with exit code {rc}: "
                    f"{_summarize_output(offloading_out)}"
                )

            bundle_names = re.findall(
                r"Extracting offload bundle:\s*(\S*amdgcn-amd-amdhsa--\S+)",
                offloading_out,
                re.IGNORECASE,
            )
            if not bundle_names:
                raise TestExecutionError(
                    f"Could not determine code object version from {binary_path}: "
                    "`llvm-objdump --offloading` succeeded but found no AMDGPU "
                    "offload bundles"
                )

            bundle_errors: List[str] = []
            for bundle_name in bundle_names:
                bundle_path = tmpdir_path / bundle_name
                if not bundle_path.exists():
                    bundle_errors.append(
                        f"bundle {bundle_name} was referenced but not extracted"
                    )
                    continue

                readelf_rc, elf_header = self.execute_command_with_output(
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

            diag = "; ".join(bundle_errors[:3])
            if len(bundle_errors) > 3:
                diag += "; ..."
            raise TestExecutionError(
                f"Could not determine code object version from {binary_path}: "
                f"{diag or 'no ABI Version found in any bundle'}"
            )

    def _run_and_validate_binary(self, binary_path: Path) -> Tuple[int, str, str]:
        """Run a compiled sample binary and determine pass/fail from its exit code.

        Returns (exit_code, captured_output, error_message).
        error_message is empty on success (exit code 0).

        We rely solely on the exit code rather than grepping for a specific
        token (e.g. "PASSED!") in the output.  Exit codes are a stable
        contract: the HIP samples return 0 on success and non-zero on failure,
        whereas output strings may change across SDK versions.
        """
        env = self.get_rocm_env_with_path()
        rc, output = self.execute_command_with_output(
            [str(binary_path)], cwd=self.build_dir, env=env
        )
        if rc != 0:
            return rc, output, f"Binary execution failed with return code {rc}"
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
            "test_suite": "cov_backward_compat",
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
            log.warning(f"Skipping {variant}: {error or 'no reason provided'}")

    def _test_variant(
        self,
        variant: str,
        compile_cmd: str,
        binary: Path,
        code_obj_version: int = None,
        expected_version: int = None,
    ) -> int | None:
        """Compile, detect version, run binary, and record the result.

        Returns the detected code object version on success, or None when the
        variant could not complete (compile failure, detection not supported,
        version mismatch, or runtime failure).  The result is always recorded
        via _record_result before returning.
        """
        rc = self._compile_sample(binary, code_obj_version=code_obj_version)
        if rc != 0:
            self._record_result(
                variant,
                compile_cmd,
                "FAIL",
                rc,
                error=f"Compile failed for {compile_cmd}",
            )
            return None

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
            return None

        if expected_version is not None and detected != expected_version:
            self._record_result(
                variant,
                compile_cmd,
                "FAIL",
                0,
                detected_version=detected,
                error=f"Expected code object version {expected_version}, but detected {detected}",
            )
            return None

        run_rc, _, run_err = self._run_and_validate_binary(binary)
        self._record_result(
            variant,
            compile_cmd,
            "FAIL" if run_err else "PASS",
            run_rc if run_err else 0,
            detected_version=detected,
            error=run_err or None,
        )
        return detected

    def run_tests(self) -> None:
        log.info(f"Running {self.display_name} Tests")

        source_path = self.sources_dir / self.source_file
        if not source_path.exists():
            raise TestExecutionError(
                f"Source file not found: {source_path}\n"
                f"Expected in test sources directory: {self.sources_dir}"
            )

        # Prepare a clean build directory for compiled binaries.
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True)

        # 1) Build with default compiler behavior (no explicit code object version),
        #    then detect current code object version n from the produced binary.
        base_version = self._test_variant(
            "default_build_detect_n",
            f"hipcc {self.source_file}",
            self.build_dir / f"{self.binary_prefix}_default",
        )
        if base_version is None:
            return

        # 2) Rebuild with n-1 and n-2 and validate detected versions + execution.
        for label, offset in [("n_minus_1", 1), ("n_minus_2", 2)]:
            target = base_version - offset
            variant = f"backward_compat_{label}"
            compile_cmd = (
                f"hipcc {self.source_file} "
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

            self._test_variant(
                variant,
                compile_cmd,
                self.build_dir / f"{self.binary_prefix}_cov{target}",
                code_obj_version=target,
                expected_version=target,
            )

    def parse_results(self) -> List[Dict[str, Any]]:
        log.info(f"Parsing {self.display_name} Results")
        if not self.test_results:
            raise TestExecutionError("No test results collected during run_tests()")

        parsed_results = []
        for result in self.test_results:
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
    run_functional_main(CovBackwardCompatibilityTest())
