"""
AMDGPU Code Object Version Backward Compatibility Functional Test.

Validates that the ROCm runtime can load and execute HIP kernels compiled
with older code object versions, ensuring backward compatibility across
the AMDGPU runtime.

This test invokes a pre-built hipRTC-based executable
(bin/cov-tests/cov_backward_compat) that:
  1. Compiles a vector-add kernel at runtime via hipRTC with the default
     code object version (no explicit flag).
  2. Detects the code object version (n) from the compiled blob.
  3. Recompiles with -mcode-object-version=n-1 and n-2.
  4. For each variant, verifies:
     - The detected code object version matches the requested version.
     - The kernel executes successfully and produces correct results.

The executable outputs one JSON object per line (JSONL) for each variant,
which this script parses into the standard test result format.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For utils
sys.path.insert(0, str(Path(__file__).resolve().parent))  # For functional_base
from functional_base import FunctionalBase, run_functional_main
from utils.exceptions import TestExecutionError
from utils.logger import log


class CovBackwardCompatibilityTest(FunctionalBase):
    """Validate code-object-version backward compatibility via hipRTC."""

    BINARY_NAME = "cov_backward_compat"
    BINARY_SUBDIR = Path("bin") / "cov-tests"

    def __init__(self):
        super().__init__(
            test_name="cov_backward_compatibility",
            display_name="Cov Backward Compatibility",
        )
        self.test_results: List[Dict[str, Any]] = []

    def _find_binary(self) -> Path:
        """Locate the pre-built hipRTC test executable."""
        candidates = [
            self.rocm_path / self.BINARY_SUBDIR / self.BINARY_NAME,
            self.rocm_path / "bin" / self.BINARY_NAME,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                log.info("Found test binary: %s", candidate)
                return candidate

        searched = "\n  ".join(str(c) for c in candidates)
        raise TestExecutionError(
            f"Pre-built test binary '{self.BINARY_NAME}' not found.\n"
            f"Searched:\n  {searched}\n"
            "Ensure the cov-backward-compat artifact was built and installed.\n"
            "The binary is built as part of TheRock when THEROCK_BUILD_TESTING "
            "is enabled and is included in the core-cov-tests artifact."
        )

    def run_tests(self) -> None:
        log.info("Running %s Tests", self.display_name)

        binary = self._find_binary()
        env = self.get_rocm_env_with_path()

        log.info("Executing: %s", binary)
        rc, output = self.execute_command_with_output(
            [str(binary)], cwd=binary.parent, env=env,
        )

        # Parse JSONL output — each line is one variant result.
        parsed_any = False
        for line in output.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            parsed_any = True
            variant = result.get("variant", "unknown")
            status = result.get("status", "ERROR").upper()
            detected_cov = result.get("detected_cov", -1)
            requested_cov = result.get("requested_cov", -1)
            error = result.get("error")

            cov_info = f"detected_cov={detected_cov}"
            if requested_cov >= 0:
                cov_info = f"requested_cov={requested_cov}, {cov_info}"

            self.test_results.append({
                "test_suite": "cov_backward_compat",
                "test_case": variant,
                "command": str(binary),
                "return_code": 0 if status == "PASS" else 1,
                "status": status,
                "detected_code_object_version": detected_cov,
                "info": cov_info,
                "error": error,
            })

        if not parsed_any:
            status = "FAIL" if rc != 0 else "ERROR"
            self.test_results.append({
                "test_suite": "cov_backward_compat",
                "test_case": "execution",
                "command": str(binary),
                "return_code": rc,
                "status": status,
                "error": f"Binary exited with code {rc}, no parseable JSONL output",
            })

    def parse_results(self) -> List[Dict[str, Any]]:
        log.info("Parsing %s Results", self.display_name)
        if not self.test_results:
            raise TestExecutionError("No test results collected during run_tests()")

        parsed_results = []
        for result in self.test_results:
            status = result.get("status", "ERROR").upper()
            if status not in ("PASS", "FAIL", "ERROR", "SKIP"):
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
