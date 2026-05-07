#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Script to convert all mermaid diagrams in CK Tile docs to SVGs.
This script:
1. Finds all mermaid blocks in RST files
2. Converts them to SVG using mmdc
3. Updates RST files to use SVG images with commented mermaid source
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

# Configuration
DOCS_DIR = Path(__file__).parent
DIAGRAMS_DIR = DOCS_DIR / "diagrams"
RST_FILES = [
    "convolution_example.rst",
    "encoding_internals.rst",
    "lds_index_swapping.rst",
    "space_filling_curve.rst",
    "sweep_tile.rst",
    "tensor_coordinates.rst",
    "thread_mapping.rst",
    "static_distributed_tensor.rst",
    "load_store_traits.rst",
    "tile_window.rst",
    "transforms.rst",
    "descriptors.rst",
    "coordinate_movement.rst",
    "adaptors.rst",
    "introduction_motivation.rst",
    "buffer_views.rst",
    "tensor_views.rst",
    "coordinate_systems.rst",
    "tile_distribution.rst",
]

# Pattern to find mermaid blocks (can be indented with 3 spaces for commented blocks)
MERMAID_PATTERN = re.compile(
    r"^(?:   )?\.\. mermaid::\s*\n((?:(?:\n|   .*))*)", re.MULTILINE
)


def extract_mermaid_content(block):
    """Extract the actual mermaid code from the block, removing RST indentation."""
    lines = block.split("\n")
    # Remove the leading spaces (RST indentation)
    content_lines = []
    for line in lines:
        if line.startswith("   "):
            content_lines.append(line[3:])  # Remove 3 spaces
        elif line.strip() == "":
            content_lines.append("")
    return "\n".join(content_lines).strip()


def generate_diagram_name(file_path, diagram_index, total_in_file):
    """Generate a descriptive name for the diagram."""
    base_name = file_path.stem
    if total_in_file == 1:
        return f"{base_name}.svg"
    else:
        return f"{base_name}_{diagram_index + 1}.svg"


def convert_mermaid_to_svg(mermaid_code, output_path):
    """Convert mermaid code to SVG using mmdc."""
    # Create a temporary file for the mermaid code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(mermaid_code)
        tmp_path = tmp.name

    try:
        # Run mmdc to convert to SVG (use shell=True on Windows for .cmd files)
        subprocess.run(
            [
                "mmdc",
                "-i",
                tmp_path,
                "-o",
                str(output_path),
                "-t",
                "neutral",
                "-b",
                "transparent",
            ],
            capture_output=True,
            text=True,
            check=True,
            shell=True,  # Required for Windows .cmd files
        )
        print(f"  ✓ Generated: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error converting diagram: {e.stderr}")
        return False
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def update_rst_file(file_path, diagrams_info):
    """Update RST file to replace mermaid blocks with commented source + image reference."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Sort diagrams by position (reverse order to maintain positions)
    diagrams_info.sort(key=lambda x: x["position"], reverse=True)

    for info in diagrams_info:
        # Find the mermaid block
        match = info["match"]
        start_pos = match.start()
        end_pos = match.end()

        # Create the replacement text
        mermaid_block = match.group(0)

        # Create commented mermaid block
        commented_lines = [
            ".. ",
            "   Original mermaid diagram (edit here, then run update_diagrams.py)",
            "   ",
        ]
        for line in mermaid_block.split("\n"):
            commented_lines.append(f"   {line}")

        # Add image reference
        svg_rel_path = f"diagrams/{info['svg_name']}"
        image_block = [
            "",
            f".. image:: {svg_rel_path}",
            "   :alt: Diagram",
            "   :align: center",
            "",
        ]

        replacement = "\n".join(commented_lines + image_block)

        # Replace in content
        content = content[:start_pos] + replacement + content[end_pos:]

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✓ Updated: {file_path.name}")


def process_file(file_path):
    """Process a single RST file."""
    print(f"\nProcessing {file_path.name}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all mermaid blocks
    matches = list(MERMAID_PATTERN.finditer(content))

    if not matches:
        print("  No mermaid diagrams found.")
        return

    print(f"  Found {len(matches)} diagram(s)")

    diagrams_info = []

    # Process each mermaid block
    for idx, match in enumerate(matches):
        mermaid_content = extract_mermaid_content(match.group(1))
        svg_name = generate_diagram_name(file_path, idx, len(matches))
        svg_path = DIAGRAMS_DIR / svg_name

        # Convert to SVG
        if convert_mermaid_to_svg(mermaid_content, svg_path):
            diagrams_info.append(
                {"match": match, "svg_name": svg_name, "position": match.start()}
            )

    # Update the RST file
    if diagrams_info:
        update_rst_file(file_path, diagrams_info)


def main():
    """Main function."""
    print("CK Tile Mermaid to SVG Converter")
    print("=" * 50)

    # Verify mmdc is available
    try:
        subprocess.run(
            ["mmdc", "--version"], capture_output=True, check=True, shell=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: mermaid-cli (mmdc) not found. Please install it:")
        print("  npm install -g @mermaid-js/mermaid-cli")
        return 1

    # Ensure diagrams directory exists
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    # Process each file
    for rst_file in RST_FILES:
        file_path = DOCS_DIR / rst_file
        if file_path.exists():
            process_file(file_path)
        else:
            print(f"\n⚠ Warning: {rst_file} not found")

    print("\n" + "=" * 50)
    print("✓ Conversion complete!")
    print(f"SVG files saved to: {DIAGRAMS_DIR}")

    return 0


if __name__ == "__main__":
    exit(main())
