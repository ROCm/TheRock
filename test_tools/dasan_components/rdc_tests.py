# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run packaged RDC unit binaries, excluding daemon/management functional driver."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument("--prefix", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
directory = a.prefix / "share/rdc/rdctst_tests"
binaries = sorted(path for path in directory.glob("rdc_*_tests") if path.is_file())
if not binaries:
    raise SystemExit("No installed RDC unit binaries discovered")
status = 0
records = []
for binary in binaries:
    command = [str(binary), f"--gtest_output=xml:{a.output / (binary.name + '.xml')}"]
    print(command, flush=True)
    result = subprocess.run(command, cwd=directory, timeout=300)
    status |= result.returncode != 0
    counts = dict(passed=0, failed=0, skipped=0)
    for case in ET.parse(a.output / (binary.name + ".xml")).iter("testcase"):
        key = (
            "failed"
            if case.find("failure") is not None or case.find("error") is not None
            else "skipped" if case.find("skipped") is not None else "passed"
        )
        counts[key] += 1
    if not sum(counts.values()) or counts["failed"]:
        status = 1
    records.append(
        dict(binary=binary.name, returncode=result.returncode, counts=counts)
    )
(a.output / "rdc-summary.json").write_text(
    json.dumps(
        dict(status="failed" if status else "passed", binaries=records), indent=2
    )
    + "\n"
)
sys.exit(status)
