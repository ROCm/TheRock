#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Purpose:
  Normalize and enforce AMD two-line copyright + SPDX headers across files.

Target files:
  - C/C++-style: .cpp, .hpp, .inc  -> uses "//" comment style
  - Hash-style:  .py, .cmake, .sh, and CMakeLists.txt -> uses "#" style

Header formats inserted (top of file, followed by exactly one blank line):
  C/C++  :
    // Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
    // SPDX-License-Identifier: MIT
    <blank>
  Hash   :
    <blank>

Shebang special case (hash-style only):
  - If line 1 starts with "#!", keep shebang, then a blank line, then the
    two hash-style header lines, then a blank line.

Removal rules:
  - Remove any comment lines (anywhere in file) containing the keywords
    "copyright" or "spdx" (case-insensitive). Blank lines are preserved.
  - Remove long-form MIT license block comment when:
      a) The file starts with the block (absolute top), OR
      b) The block appears immediately after the AMD header position
         (i.e., when remainder at insertion point begins with "/*" and
         the first content line is "* The MIT License (MIT)").

Blank-line normalization:
  - Enforce exactly ONE blank line immediately after the AMD header.
    (Drop only the leading blank lines at the insertion point before
     re-inserting the header.)
  - Do not change blank lines between other non-copyright comments.

Preservation:
  - Preserve original newline style: CRLF (\r\n) vs LF (\n).
  - Preserve UTF-8 BOM if present.
  - Do not modify non-comment code lines.

Idempotency:
  - Running this script multiple times does not further modify files.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import List, Tuple

AMD_CPP_HEADER_TEXT = [
    "// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
    "// SPDX-License-Identifier: MIT",
]
AMD_HASH_HEADER_TEXT = [
    "# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
    "# SPDX-License-Identifier: MIT",
]

CPP_EXTS = {".cpp", ".hpp", ".inc"}
HASH_EXTS = {".py", ".cmake", ".sh"}

# --- Encoding helpers -------------------------------------------------------


def has_bom(raw: bytes) -> bool:
    return raw.startswith(b"\xef\xbb\xbf")


def decode_text(raw: bytes) -> str:
    return raw.decode("utf-8-sig", errors="replace")


def encode_text(text: str, bom: bool) -> bytes:
    data = text.encode("utf-8")
    return (b"\xef\xbb\xbf" + data) if bom else data


# --- Newline detection ------------------------------------------------------


def detect_newline_sequence(raw: bytes) -> str:
    if b"\r\n" in raw:
        return "\r\n"
    elif b"\n" in raw:
        return "\n"
    else:
        return "\n"


# --- Utilities --------------------------------------------------------------


def is_comment_line(line: str, style: str) -> bool:
    stripped = line.lstrip()
    if style == "cpp":
        return (
            stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.startswith("*/")
        )
    elif style == "hash":
        return stripped.startswith("#")
    return False


def has_keywords(line: str) -> bool:
    lower_line = line.lower()
    return ("copyright" in lower_line) or ("spdx" in lower_line)


# --- MIT License banner detection ------------------------------
MIT_C_FIRST_LINE_RE = re.compile(r"^\s*\*\s*The MIT License \(MIT\)")
MIT_HASH_FIRST_LINE_RE = re.compile(r"^\s*#\s*The MIT License \(MIT\)")


def remove_top_mit_block(lines: List[str]) -> Tuple[List[str], bool]:
    """
    Unified MIT banner removal at the top of 'lines'.
    Supports:
      - C-style block starting with '/*' and ending with '*/'; removes only if
        a line within the block matches MIT_C_FIRST_LINE_RE.
      - Hash-style banner: contiguous top run of lines starting with '#';
        removes only if any line in that run matches MIT_HASH_FIRST_LINE_RE.
    Returns (new_lines, removed_flag). Preserves EOLs.
    """
    if not lines:
        return lines, False

    first = lines[0].lstrip()

    # C-style block
    if first.startswith("/*"):
        end_idx, saw_mit = None, False
        for i, line in enumerate(lines[1:], 1):
            if not saw_mit and MIT_C_FIRST_LINE_RE.match(line):
                saw_mit = True
            s = line.lstrip()
            if s.startswith("*/") or s.rstrip().endswith("*/"):
                end_idx = i + 1
                break
        if end_idx is not None and saw_mit:
            return lines[end_idx:], True
        return lines, False

    # Hash-style contiguous banner
    if first.startswith("#"):
        end_idx, saw_mit = 0, False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                if not saw_mit and MIT_HASH_FIRST_LINE_RE.match(line):
                    saw_mit = True
                end_idx = i + 1
            else:
                break
        if saw_mit:
            return lines[end_idx:], True
        return lines, False

    return lines, False


# --- Removal + normalization helpers ---------------------------------------


def remove_keyword_comment_lines_globally(lines: List[str], style: str) -> List[str]:
    """Remove comment lines containing keywords anywhere in the file.
    **Do not** remove blank lines; preserve all other lines as-is."""
    out: List[str] = []
    for line in lines:
        if is_comment_line(line, style) and has_keywords(line):
            continue
        out.append(line)
    return out


def drop_leading_blank_lines(lines: List[str]) -> List[str]:
    """Drop only the leading blank lines at the start of the given list."""
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return lines[i:]


# --- Header builder ---------------------------------------------------------


def build_header_lines(style: str, nl: str) -> List[str]:
    base = AMD_CPP_HEADER_TEXT if style == "cpp" else AMD_HASH_HEADER_TEXT
    return [base[0] + nl, base[1] + nl, nl]  # header + exactly one blank


# --- Main transforms --------------------------------------------------------


def process_cpp(text: str, nl: str) -> str:
    lines = text.splitlines(True)

    # Remove MIT block if it is at the *absolute* top
    lines, _ = remove_top_mit_block(lines)

    # Remove keyworded comment lines globally (blank lines preserved)
    lines = remove_keyword_comment_lines_globally(lines, style="cpp")

    # Normalize insertion point and remove MIT block if it appears *after header*
    lines = drop_leading_blank_lines(lines)
    lines, _ = remove_top_mit_block(lines)

    # Prepend AMD header (guarantee exactly one blank after)
    return "".join(build_header_lines("cpp", nl) + lines)


def process_hash(text: str, nl: str) -> str:
    lines = text.splitlines(True)
    if not lines:
        return "".join(build_header_lines("hash", nl))

    shebang = lines[0].startswith("#!")

    if shebang:
        remainder = remove_keyword_comment_lines_globally(lines[1:], style="hash")
        remainder = drop_leading_blank_lines(remainder)
        remainder, _ = remove_top_mit_block(remainder)  # remove MIT block after header
        new_top = [lines[0], nl] + build_header_lines("hash", nl)
        return "".join(new_top + remainder)
    else:
        remainder = remove_keyword_comment_lines_globally(lines, style="hash")
        remainder = drop_leading_blank_lines(remainder)
        remainder, _ = remove_top_mit_block(remainder)  # remove MIT block after header
        return "".join(build_header_lines("hash", nl) + remainder)


# --- File processing & CLI --------------------------------------------------


def process_file(path: Path) -> bool:
    name = path.name
    suffix = path.suffix.lower()
    if suffix in CPP_EXTS:
        style = "cpp"
    elif suffix in HASH_EXTS or name == "CMakeLists.txt":
        style = "hash"
    else:
        return False

    raw = path.read_bytes()
    bom = has_bom(raw)
    nl = detect_newline_sequence(raw)
    text = decode_text(raw)

    updated = process_cpp(text, nl) if style == "cpp" else process_hash(text, nl)
    if updated != text:
        path.write_bytes(encode_text(updated, bom))
        return True
    return False


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    changed = 0
    skipped = 0
    errors: List[str] = []
    for arg in argv[1:]:
        p = Path(arg)
        try:
            if not p.exists():
                errors.append(f"Not found: {p}")
                continue
            if p.is_dir():
                errors.append(f"Is a directory (pass specific files): {p}")
                continue
            if process_file(p):
                changed += 1
                print(f"Updated: {p}")
            else:
                skipped += 1
                print(f"Skipped (no change needed or unsupported type): {p}")
        except Exception as e:
            errors.append(f"Error processing {p}: {e}")
    print(f"\nSummary: {changed} updated, {skipped} skipped, {len(errors)} errors")
    for msg in errors:
        print(f" - {msg}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
