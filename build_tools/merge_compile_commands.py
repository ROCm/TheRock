#!/usr/bin/env python
# Merges a number of compile_commands.json files.
# Usage:
#   python merge_compile_commands.py {output} {inputs...}
# If any input file is empty or missing, it will be silently ignored. This is
# so that the build graph doesn't need to conditionally deal with sub projects
# that do not populate it.

from pathlib import Path
import sys
import json

all_files = sys.argv[1:]
if len(all_files) < 1:
    print("Usage: merge_compile_commands.py {output_file} {inputs...}")
output_file = all_files[0]
input_files = all_files[1:]

records = []

for input_file in input_files:
    input_path = Path(input_file)
    if not input_path.exists() or input_path.stat().st_size == 0:
        continue
    with open(input_path, "rb") as f:
        this_records = json.load(f)
    records.extend(this_records)

with open(output_file, "wt") as f:
    json.dump(records, f, check_circular=False, indent="  ")
