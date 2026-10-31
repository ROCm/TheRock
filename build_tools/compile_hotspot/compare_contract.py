"""Compare ordered runtime identities and require numeric execution coverage."""

import argparse
import json
from pathlib import Path
import shlex

EXPECTED_CASES = {
    "grouped_3x3_padded",
    "grouped_1x1",
    "nongrouped_1x1",
    "grouped_3x3_stride2",
}


def read_contract(path: Path) -> dict:
    inventory = []
    executed = {}
    cases = []
    result = None
    count = None
    gpu = None
    errors = []
    seen = {"COUNT": 0, "RESULT": 0, "GPU": 0}
    for line in path.read_text().splitlines():
        if line.startswith("INSTANCE\t"):
            fields = shlex.split(line)
            if len(fields) != 6:
                raise ValueError(f"Malformed identity record: {line}")
            inventory.append(
                {
                    "index": int(fields[1]),
                    "family": fields[2],
                    "specialization": fields[3],
                    "rtti": fields[4],
                    "type_string": fields[5],
                }
            )
        elif line.startswith("EXECUTED\t"):
            _, index, times = line.split("\t")
            index = int(index)
            if index in executed:
                errors.append(f"Duplicate EXECUTED index {index}")
            executed[index] = int(times)
        elif line.startswith("CASE\t"):
            fields = line.split("\t")
            cases.append(
                {"case": fields[1], "index": int(fields[2]), "status": fields[3]}
            )
        elif line.startswith("RESULT\t"):
            seen["RESULT"] += 1
            result = line.split("\t")[1]
        elif line.startswith("COUNT\t"):
            seen["COUNT"] += 1
            count = int(line.split("\t")[1])
        elif line.startswith("GPU\t"):
            seen["GPU"] += 1
            gpu = line.split("\t")[1]
    valid_inventory = (
        seen["COUNT"] == 1
        and count == 90
        and len(inventory) == 90
        and [item["index"] for item in inventory] == list(range(90))
        and len({item["rtti"] for item in inventory}) == 90
    )
    coverage = {}
    for item in inventory:
        key = item["family"] + "/" + item["specialization"]
        coverage.setdefault(key, {"registered": 0, "executed": 0})
        coverage[key]["registered"] += 1
        coverage[key]["executed"] += executed.get(item["index"], 0) > 0
    case_keys = [(case["case"], case["index"]) for case in cases]
    expected_keys = {(name, index) for name in EXPECTED_CASES for index in range(90)}
    valid_cases = (
        len(case_keys) == 360
        and len(set(case_keys)) == 360
        and set(case_keys) == expected_keys
        and all(case["status"] in {"PASS", "FAIL", "UNSUPPORTED"} for case in cases)
    )
    successful = {
        index: sum(
            case["index"] == index and case["status"] == "PASS" for case in cases
        )
        for index in range(90)
    }
    numeric_pass = (
        not errors
        and seen["RESULT"] == 1
        and result == "PASS"
        and seen["GPU"] == 1
        and gpu.split(":")[0] == "gfx950"
        and valid_cases
        and set(executed) == set(range(90))
        and all(times > 0 for times in executed.values())
        and executed == successful
        and not any(case["status"] == "FAIL" for case in cases)
        and any(case["status"] == "PASS" for case in cases)
    )
    return {
        "path": str(path),
        "valid_inventory": valid_inventory,
        "inventory": inventory,
        "numeric_pass": numeric_pass,
        "coverage": coverage,
        "cases": cases,
        "gpu": gpu,
        "errors": errors,
    }


def compare_contracts(
    baseline: dict, candidate: dict, inventory_only: bool = False
) -> dict:
    identical = baseline["inventory"] == candidate["inventory"]
    # Compare sorted matrices so harmless output interleaving cannot mask changed support.
    case_matrix = lambda result: sorted(
        (case["case"], case["index"], case["status"]) for case in result["cases"]
    )
    cases_identical = case_matrix(baseline) == case_matrix(candidate)
    gpu_identical = baseline["gpu"] == candidate["gpu"]
    passed = baseline["valid_inventory"] and candidate["valid_inventory"] and identical
    if not inventory_only:
        passed = (
            passed
            and baseline["numeric_pass"]
            and candidate["numeric_pass"]
            and cases_identical
            and gpu_identical
        )
    return {
        "pass": passed,
        "mode": "inventory-only" if inventory_only else "numeric",
        "ordered_inventory_identical": identical,
        "case_support_and_results_identical": cases_identical,
        "gpu_identical": gpu_identical,
        "baseline": baseline,
        "candidate": candidate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    baseline, candidate = read_contract(args.baseline), read_contract(args.candidate)
    report = compare_contracts(baseline, candidate, args.inventory_only)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "ordered_inventory_identical": report["ordered_inventory_identical"],
                "report": str(args.output),
            }
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
