# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import sys


def read_json_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format: {e}", e.doc, e.pos)
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_json.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    try:
        parsed_data = read_json_file(json_file_path)
        print("JSON parsed successfully!")
        threshold = 15  # max number of minutes for compilation
        for i in range(len(parsed_data)):
            if parsed_data[i]["dur"] > threshold * 60000000:
                print(
                    f"build duration of {parsed_data[i]['name']}  exceeds {threshold} minutes! actual build time: {parsed_data[i]['dur'] / 60000000:.2f} minutes!"
                )

    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except json.JSONDecodeError as json_err:
        print(f"Error: {json_err}")
    except Exception as e:
        print(f"Unexpected error: {e}")
