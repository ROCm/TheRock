#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Helper script to update SVG diagrams from commented mermaid sources in RST files.

This script scans RST files for commented mermaid blocks (created by convert_mermaid_to_svg.py)
and regenerates the corresponding SVG files when the source has been modified.

Usage:
    python update_diagrams.py              # Update all diagrams
    python update_diagrams.py <file.rst>   # Update diagrams in a specific file
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
DOCS_DIR = Path(__file__).parent
DIAGRAMS_DIR = DOCS_DIR / "diagrams"

# Pattern to find commented mermaid blocks followed by image references
COMMENTED_MERMAID_PATTERN = re.compile(
    r"\.\.\s*\n"  # Comment start
    r"(?:   .*\n|\s*\n)*?"  # Comment description lines (may have blank lines)
    r"(   \.\. mermaid::\s*\n"  # Commented mermaid directive
    r"(?:   \n|   .*\n|\s*\n)*?)"  # Mermaid content (including blank lines)
    r"\.\. image:: diagrams/([^\s]+)",  # Image reference
    re.MULTILINE,
)


def extract_mermaid_from_comment(commented_block):
    """Extract mermaid code from a commented block."""
    # Remove the comment indentation (3 spaces at start of each line)
    lines = commented_block.split("\n")
    content_lines = []

    for line in lines:
        if line.startswith("   "):
            # Remove the 3-space comment indentation
            content_lines.append(line[3:])
        elif line.strip() == "":
            content_lines.append("")

    # Now we have the mermaid block, extract the actual mermaid code
    mermaid_content = "\n".join(content_lines)

    # Remove the ".. mermaid::" directive and extract the indented content
    mermaid_match = re.search(
        r"\.\. mermaid::\s*\n((?:(?:\n|   .*))*)", mermaid_content
    )
    if mermaid_match:
        mermaid_code = mermaid_match.group(1)
        # Remove RST indentation from mermaid code
        code_lines = []
        for line in mermaid_code.split("\n"):
            if line.startswith("   "):
                code_lines.append(line[3:])
            elif line.strip() == "":
                code_lines.append("")
        return "\n".join(code_lines).strip()

    return None


def convert_mermaid_to_svg(mermaid_code, output_path):
    """Convert mermaid code to SVG using mmdc."""
    # Create a temporary file for the mermaid code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(mermaid_code)
        tmp_path = tmp.name

    try:
        # Run mmdc to convert to SVG
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
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def process_file(file_path, force_update=False):
    """Process a single RST file to update diagrams."""
    print(f"Checking {file_path.name}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all commented mermaid blocks
    matches = list(COMMENTED_MERMAID_PATTERN.finditer(content))

    if not matches:
        print("  No commented mermaid diagrams found.")
        return 0, 0

    updated_count = 0
    error_count = 0

    for match in matches:
        commented_mermaid = match.group(1)
        svg_filename = match.group(2)
        svg_path = DIAGRAMS_DIR / svg_filename

        # Extract mermaid code
        mermaid_code = extract_mermaid_from_comment(commented_mermaid)
        if not mermaid_code:
            print(f"  ⚠ Could not extract mermaid code for {svg_filename}")
            error_count += 1
            continue

        # Check if SVG needs updating
        needs_update = force_update or not svg_path.exists()

        if not needs_update:
            # For a more sophisticated check, we could hash the mermaid code
            # and compare with a stored hash, but for simplicity we just check existence
            print(f"  ✓ {svg_filename} exists (use --force to regenerate)")
            continue

        # Generate SVG
        success, error = convert_mermaid_to_svg(mermaid_code, svg_path)

        if success:
            print(f"  ✓ Updated: {svg_filename}")
            updated_count += 1
        else:
            print(f"  ✗ Error updating {svg_filename}: {error}")
            error_count += 1

    return updated_count, error_count


def find_rst_files():
    """Find all RST files in the CK tile docs directory."""
    return list(DOCS_DIR.glob("*.rst"))


def main():
    """Main function."""
    print("CK Tile Diagram Updater")
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

    # Parse command line arguments
    force_update = "--force" in sys.argv or "-f" in sys.argv
    specific_file = None

    for arg in sys.argv[1:]:
        if arg not in ["--force", "-f"] and arg.endswith(".rst"):
            specific_file = DOCS_DIR / arg
            if not specific_file.exists():
                print(f"Error: File not found: {arg}")
                return 1

    # Get files to process
    if specific_file:
        files_to_process = [specific_file]
    else:
        files_to_process = find_rst_files()

    # Process files
    total_updated = 0
    total_errors = 0

    for file_path in files_to_process:
        updated, errors = process_file(file_path, force_update)
        total_updated += updated
        total_errors += errors

    print("\n" + "=" * 50)
    print("✓ Update complete!")
    print(f"  Updated: {total_updated} diagram(s)")
    if total_errors > 0:
        print(f"  Errors: {total_errors}")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit(main())
