# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
import dataclasses
import os.path as path
import textwrap


def update_file(file_path, content):
    """Update the file at file_path with the given content if it differs from the existing content.

    It avoids unnecessary touching of the file which triggers rebuilds
    """

    existing_content = ""
    if path.exists(file_path):
        with open(file_path, "r") as file:
            existing_content = file.read()
    if existing_content == content:
        return
    with open(file_path, "w") as file:
        file.write(content)


def indent(code: str, indent: str = "    ") -> str:
    return textwrap.indent(code, indent)


def if_(i: int) -> str:
    return "if" if i == 0 else "else if"


def check_duplicates_and_paddings(traits, trait):
    """Check
    * if the traits list does not contain a trait with the same parameters;
    * if paddings are consitent: the previous kernel can be incorrectly called before the new one,
      for example, f, _t_, f, t cannot be before f, _f_, f, t.
    """

    fields = [f.name for f in dataclasses.fields(trait)]
    pad_fields = [f for f in fields if "pad" in f]
    non_pad_fields = [f for f in fields if "pad" not in f]
    for prev_trait in traits:
        if any(getattr(trait, f) != getattr(prev_trait, f) for f in non_pad_fields):
            continue
        if all(getattr(trait, f) == getattr(prev_trait, f) for f in pad_fields):
            raise Exception(f"Duplicate found {trait}")
        # Check if the previous kernel can be incorrectly used before the current one
        # for example, f, _t_, f, t cannot be before f, _f_, f, t
        is_prev_more_restrictive = False
        is_curr_more_restrictive = False
        for f in pad_fields:
            prev_pad = getattr(prev_trait, f)
            pad = getattr(trait, f)
            if isinstance(prev_pad, str):
                prev_pad = 1000000 if prev_pad == "f" else 1
                pad = 1000000 if pad == "f" else 1
            elif isinstance(prev_pad, int):
                prev_pad = 1000000 if prev_pad == 0 else prev_pad
                pad = 1000000 if pad == 0 else pad
            else:
                assert False
            if prev_pad < pad:
                is_prev_more_restrictive = True
            elif prev_pad > pad:
                is_curr_more_restrictive = True
        if is_prev_more_restrictive and not is_curr_more_restrictive:
            raise Exception(
                f"Kernel will never be used because paddings are not ordered correctly:\n{prev_trait} supersedes\n{trait}"
            )
