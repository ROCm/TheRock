"""Regression checks for false-positive contract acceptance."""

from pathlib import Path
import tempfile
import unittest

from compare_contract import EXPECTED_CASES, compare_contracts, read_contract


def good_log() -> str:
    lines = ["COUNT\t90"]
    for index in range(90):
        lines.append(
            f'INSTANCE\t{index}\tfamily\tspec\t"type{index}"\t"display{index}"'
        )
    lines.append("GPU\tgfx950:xnack+")
    for name in sorted(EXPECTED_CASES):
        for index in range(90):
            lines.append(f"CASE\t{name}\t{index}\tPASS\t0")
    lines.extend(f"EXECUTED\t{index}\t4" for index in range(90))
    lines.append("RESULT\tPASS")
    return "\n".join(lines) + "\n"


class ContractTests(unittest.TestCase):
    def compare(self, candidate: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.log"
            candidate_path = Path(directory) / "candidate.log"
            baseline_path.write_text(good_log())
            candidate_path.write_text(candidate)
            return compare_contracts(
                read_contract(baseline_path), read_contract(candidate_path)
            )

    def test_complete_matching_contract(self):
        self.assertTrue(self.compare(good_log())["pass"])

    def test_support_loss_despite_other_successful_cases(self):
        candidate = (
            good_log()
            .replace(
                "CASE\tgrouped_1x1\t0\tPASS\t0", "CASE\tgrouped_1x1\t0\tUNSUPPORTED"
            )
            .replace("EXECUTED\t0\t4", "EXECUTED\t0\t3")
        )
        report = self.compare(candidate)
        self.assertTrue(report["candidate"]["numeric_pass"])
        self.assertFalse(report["case_support_and_results_identical"])
        self.assertFalse(report["pass"])

    def test_truncated_case_matrix(self):
        candidate = good_log().replace("CASE\tgrouped_1x1\t0\tPASS\t0\n", "")
        self.assertFalse(self.compare(candidate)["pass"])

    def test_duplicate_case(self):
        self.assertFalse(
            self.compare(good_log() + "CASE\tgrouped_1x1\t0\tPASS\t0\n")["pass"]
        )

    def test_duplicate_control_records(self):
        for extra in (
            "COUNT\t90",
            "RESULT\tPASS",
            "GPU\tgfx950:xnack+",
            "EXECUTED\t0\t4",
        ):
            with self.subTest(extra=extra):
                self.assertFalse(self.compare(good_log() + extra + "\n")["pass"])

    def test_wrong_gpu_and_gpu_feature_mismatch(self):
        for gpu in ("gfx942", "gfx950:xnack-"):
            with self.subTest(gpu=gpu):
                self.assertFalse(
                    self.compare(good_log().replace("gfx950:xnack+", gpu))["pass"]
                )

    def test_execution_count_must_match_case_results(self):
        self.assertFalse(
            self.compare(good_log().replace("EXECUTED\t0\t4", "EXECUTED\t0\t3"))["pass"]
        )

    def test_unknown_status(self):
        self.assertFalse(
            self.compare(
                good_log().replace(
                    "CASE\tgrouped_1x1\t0\tPASS", "CASE\tgrouped_1x1\t0\tOK"
                )
            )["pass"]
        )


if __name__ == "__main__":
    unittest.main()
