#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convert raw HTML mermaid blocks to commented format for SVG conversion."""

import os
import re


def convert_raw_html_to_commented(content):
    """Convert raw HTML mermaid blocks to commented mermaid format."""

    # Pattern to match raw HTML mermaid blocks
    pattern = r'\.\. raw:: html\n\n   <div class="mermaid"[^>]*>\n(.*?)\n   </div>'

    def replace_block(match):
        mermaid_code = match.group(1)
        # The mermaid code in HTML has 3-space indentation, keep it
        # but add 3 more spaces for .. mermaid:: indentation
        mermaid_lines = mermaid_code.split("\n")
        properly_indented = []
        for line in mermaid_lines:
            if line.strip():  # Non-empty line
                # Line already has 3 spaces from HTML, add 3 more for mermaid block
                properly_indented.append("   " + line)
            else:
                properly_indented.append("")

        indented_code = "\n".join(properly_indented)

        # Create commented format matching the expected pattern
        commented = f""".. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
   .. mermaid::
   
{indented_code}
   
   
"""
        return commented

    return re.sub(pattern, replace_block, content, flags=re.DOTALL)


def main():
    """Process files with raw HTML mermaid blocks."""

    files_to_convert = [
        "introduction_motivation.rst",
        "buffer_views.rst",
        "tensor_views.rst",
        "coordinate_systems.rst",
        "tile_distribution.rst",
    ]

    converted_files = []

    for filename in files_to_convert:
        if not os.path.exists(filename):
            print(f"Skipping {filename} - not found")
            continue

        with open(filename, "r", encoding="utf-8") as f:
            original = f.read()

        converted = convert_raw_html_to_commented(original)

        if converted != original:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(converted)

            blocks_converted = original.count(".. raw:: html")
            converted_files.append((filename, blocks_converted))
            print(f"✓ Converted {filename}: {blocks_converted} blocks")
        else:
            print(f"  {filename}: no raw HTML blocks found")

    print("\n=== CONVERSION COMPLETE ===")
    print(f"Files converted: {len(converted_files)}")
    print(f"Total blocks: {sum(c for _, c in converted_files)}")
    print("\nNext: Run convert_mermaid_to_svg.py to generate SVG files")


if __name__ == "__main__":
    main()
