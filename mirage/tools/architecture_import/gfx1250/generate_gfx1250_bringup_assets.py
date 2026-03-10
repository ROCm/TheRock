#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


FOCUS_AREAS = {
    "kVop3p": "vop3p",
    "kWmma": "wmma",
    "kFp8Bf8": "fp8_bf8",
    "kScalePaired": "scale_paired",
}

FOCUS_LIMITS = {
    "vop3p": 24,
    "wmma": 24,
    "fp8_bf8": 32,
    "scale_paired": 24,
}

PRIORITY_INSTRUCTIONS = {
    "vop3p": [
        "V_PK_ADD_BF16",
        "V_PK_FMA_BF16",
    ],
    "wmma": [
        "V_WMMA_F32_16X16X4_F32_w32",
        "V_WMMA_BF16F32_16X16X32_BF16_w32",
        "V_SWMMAC_F32_16X16X64_F16_w32",
        "TENSOR_LOAD_TO_LDS",
    ],
    "fp8_bf8": [
        "V_CVT_F16_FP8",
        "V_CVT_F16_BF8",
        "V_CVT_PK_FP8_F16",
    ],
    "scale_paired": [
        "V_WMMA_LD_SCALE_PAIRED_B32",
        "V_WMMA_LD_SCALE16_PAIRED_B64",
    ],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_string_array(name: str, values: list[str]) -> str:
    lines = [f"constexpr std::array<std::string_view, {len(values)}> {name}{{{{"]
    for value in values:
        lines.append(f'    "{value}",')
    lines.append("}};")
    return "\n".join(lines)


def render_cc(summary: dict, category_values: dict[str, list[str]]) -> str:
    arrays = []
    for enum_name, category_name in FOCUS_AREAS.items():
        values = category_values[category_name]
        arrays.append(format_string_array(f"{enum_name}Instructions", values))

    switch_cases = []
    for enum_name in FOCUS_AREAS:
        switch_cases.append(
            f"    case BringupFocusArea::{enum_name}:\n"
            f"      return {enum_name}Instructions;"
        )

    return f"""#include "lib/sim/isa/gfx1250/bringup_profile.h"

#include <array>

namespace mirage::sim::isa::gfx1250 {{
namespace {{

constexpr BringupSummary kBringupSummary{{
    {summary["rdna4_instruction_count"]},
    {summary["gfx950_instruction_count"]},
    {summary["shared_instruction_count"]},
    {summary["rdna4_only_instruction_count"]},
    {summary["gfx950_only_instruction_count"]},
    {summary["llvm_normalized_symbol_count"]},
    {summary["llvm_target_specific_count"]},
    {summary["vop3p_instruction_count"]},
    {summary["wmma_instruction_count"]},
    {summary["fp8_bf8_instruction_count"]},
    {summary["scale_paired_instruction_count"]},
}};

{chr(10).join(arrays)}

}}  // namespace

const BringupSummary& GetBringupSummary() {{
  return kBringupSummary;
}}

std::span<const std::string_view> GetFocusInstructions(BringupFocusArea area) {{
  switch (area) {{
{chr(10).join(switch_cases)}
  }}
  return std::span<const std::string_view>();
}}

bool IsFocusInstruction(std::string_view instruction_name) {{
  for (const BringupFocusArea area : {{
           BringupFocusArea::kVop3p,
           BringupFocusArea::kWmma,
           BringupFocusArea::kFp8Bf8,
           BringupFocusArea::kScalePaired,
       }}) {{
    for (const std::string_view candidate : GetFocusInstructions(area)) {{
      if (candidate == instruction_name) {{
        return true;
      }}
    }}
  }}
  return false;
}}

}}  // namespace mirage::sim::isa::gfx1250
"""


def render_target_catalog_cc(llvm_inventory: dict, rdna4_inventory: dict, delta: dict) -> str:
    rdna4_names = {
        instruction["instruction_name"] for instruction in rdna4_inventory["instructions"]
    }
    normalized_entries = {}
    for entry in llvm_inventory["entries"]:
        info = normalized_entries.setdefault(
            entry["normalized_symbol"],
            {
                "instruction_name": entry["normalized_symbol"],
                "llvm_file": entry["file"],
                "llvm_line": entry["line"],
                "is_vop3": False,
                "is_vop3p": False,
                "is_wmma": False,
                "is_fp8_bf8": False,
                "is_scale_paired": False,
            },
        )
        info["llvm_line"] = min(info["llvm_line"], entry["line"])
        for category in entry["categories"]:
            if category == "vop3":
                info["is_vop3"] = True
            elif category == "vop3p":
                info["is_vop3p"] = True
            elif category == "wmma":
                info["is_wmma"] = True
            elif category == "fp8_bf8":
                info["is_fp8_bf8"] = True
            elif category == "scale_paired":
                info["is_scale_paired"] = True

    sorted_entries = []
    for name in sorted(normalized_entries):
        info = normalized_entries[name]
        info["appears_in_rdna4_xml"] = name in rdna4_names
        info["is_target_specific"] = name not in rdna4_names
        sorted_entries.append(info)

    entry_lines = []
    for info in sorted_entries:
        entry_lines.append(
            "    {"
            f'"{info["instruction_name"]}", '
            f'"{info["llvm_file"]}", '
            f'{info["llvm_line"]}, '
            f'{"true" if info["appears_in_rdna4_xml"] else "false"}, '
            f'{"true" if info["is_target_specific"] else "false"}, '
            f'{"true" if info["is_vop3"] else "false"}, '
            f'{"true" if info["is_vop3p"] else "false"}, '
            f'{"true" if info["is_wmma"] else "false"}, '
            f'{"true" if info["is_fp8_bf8"] else "false"}, '
            f'{"true" if info["is_scale_paired"] else "false"}'
            "},"
        )

    shared_sample_lines = "\n".join(
        f'    "{value}",' for value in delta["shared_sample"][:32]
    )
    rdna4_only_lines = "\n".join(
        f'    "{value}",' for value in delta["rdna4_only_sample"][:32]
    )

    return f"""#include "lib/sim/isa/gfx1250/target_catalog.h"

#include <array>

namespace mirage::sim::isa::gfx1250 {{
namespace {{

constexpr std::array<TargetOpcodeInfo, {len(sorted_entries)}> kTargetOpcodeInfos{{{{
{chr(10).join(entry_lines)}
}}}};

constexpr std::array<std::string_view, {min(32, len(delta["shared_sample"]))}> kSharedInstructionSample{{{{
{shared_sample_lines}
}}}};

constexpr std::array<std::string_view, {min(32, len(delta["rdna4_only_sample"]))}> kRdna4OnlyInstructionSample{{{{
{rdna4_only_lines}
}}}};

}}  // namespace

std::span<const TargetOpcodeInfo> GetTargetOpcodeInfos() {{
  return kTargetOpcodeInfos;
}}

const TargetOpcodeInfo* FindTargetOpcodeInfo(std::string_view instruction_name) {{
  for (const TargetOpcodeInfo& info : kTargetOpcodeInfos) {{
    if (info.instruction_name == instruction_name) {{
      return &info;
    }}
  }}
  return nullptr;
}}

std::span<const std::string_view> GetSharedInstructionSample() {{
  return kSharedInstructionSample;
}}

std::span<const std::string_view> GetRdna4OnlyInstructionSample() {{
  return kRdna4OnlyInstructionSample;
}}

}}  // namespace mirage::sim::isa::gfx1250
"""


def render_report(summary: dict, category_values: dict[str, list[str]]) -> str:
    lines = [
        "# gfx1250 Bring-up Plan",
        "",
        "## Summary",
        "",
        f"- RDNA4 instruction count: `{summary['rdna4_instruction_count']}`",
        f"- Shared with current gfx950 catalog: `{summary['shared_instruction_count']}`",
        f"- RDNA4-only vs gfx950: `{summary['rdna4_only_instruction_count']}`",
        f"- gfx950-only vs RDNA4: `{summary['gfx950_only_instruction_count']}`",
        f"- LLVM gfx1250 normalized symbols: `{summary['llvm_normalized_symbol_count']}`",
        f"- LLVM gfx1250 target-specific symbols not verbatim in RDNA4 XML: `{summary['llvm_target_specific_count']}`",
        "",
        "## Bring-up Priority",
        "",
        "1. `VOP3P` packed/vector arithmetic families",
        "2. `WMMA` / `SWMMAC` / tensor-load-store paths",
        "3. `FP8` / `BF8` / `FP6` / `BF6` / `FP4` conversion families",
        "4. scale-select and paired-load forms",
        "",
    ]

    for enum_name, category_name in FOCUS_AREAS.items():
        lines.extend(
            [
                f"## {category_name}",
                "",
                f"- Focus instruction count: `{summary[f'{category_name}_instruction_count']}`",
            ]
        )
        for value in category_values[category_name]:
            lines.append(f"- `{value}`")
        lines.append("")

    lines.extend(
        [
            "## Recommended Next Slice",
            "",
            "- Add a gfx1250-local instruction catalog wrapper that overlays LLVM-derived target deltas on top of the RDNA4 baseline inventory.",
            "- Start decoder/catalog scaffolding with the `VOP3P` and `WMMA` paths before generic long-tail RDNA4 ALU coverage.",
            "- Keep `gfx1250` execution semantics separate from `gfx950` until the architecture-local opcode families are cataloged and testable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    script_path = Path(__file__).resolve()
    mirage_root = script_path.parents[3]

    delta_path = mirage_root / "tests" / "data" / "architecture_import" / "gfx1250" / "gfx1250_delta_vs_gfx950.json"
    llvm_path = mirage_root / "tests" / "data" / "architecture_import" / "gfx1250" / "gfx1250_llvm_inventory.json"
    rdna4_path = mirage_root / "tests" / "data" / "architecture_import" / "gfx1250" / "rdna4_instruction_inventory.json"

    delta = read_json(delta_path)
    llvm_inventory = read_json(llvm_path)
    rdna4_inventory = read_json(rdna4_path)

    category_values: dict[str, list[str]] = {}
    for category_name in FOCUS_AREAS.values():
        ordered_values = []
        seen = set()
        for value in PRIORITY_INSTRUCTIONS[category_name]:
            if value not in seen:
                ordered_values.append(value)
                seen.add(value)
        for value in delta["llvm_category_membership"][category_name]["not_in_rdna4_xml"]:
            if value in seen:
                continue
            ordered_values.append(value)
            seen.add(value)
        category_values[category_name] = ordered_values[: FOCUS_LIMITS[category_name]]

    summary = {
        "rdna4_instruction_count": delta["rdna4_instruction_count"],
        "gfx950_instruction_count": delta["gfx950_instruction_count"],
        "shared_instruction_count": delta["shared_instruction_count"],
        "rdna4_only_instruction_count": delta["rdna4_only_count"],
        "gfx950_only_instruction_count": delta["gfx950_only_count"],
        "llvm_normalized_symbol_count": llvm_inventory["normalized_symbol_count"],
        "llvm_target_specific_count": delta["llvm_target_specific_count"],
        "vop3p_instruction_count": delta["llvm_category_membership"]["vop3p"]["count"],
        "wmma_instruction_count": delta["llvm_category_membership"]["wmma"]["count"],
        "fp8_bf8_instruction_count": delta["llvm_category_membership"]["fp8_bf8"]["count"],
        "scale_paired_instruction_count": delta["llvm_category_membership"]["scale_paired"]["count"],
    }

    cc_path = mirage_root / "native" / "src" / "isa" / "gfx1250" / "bringup_profile.cc"
    target_catalog_cc_path = mirage_root / "native" / "src" / "isa" / "gfx1250" / "target_catalog.cc"
    report_path = mirage_root / "reports" / "architectures" / "gfx1250" / "gfx1250_bringup_plan.md"

    cc_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cc_path.write_text(render_cc(summary, category_values), encoding="utf-8")
    target_catalog_cc_path.write_text(
        render_target_catalog_cc(llvm_inventory, rdna4_inventory, delta),
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary, category_values), encoding="utf-8")

    print(f"Wrote {cc_path}")
    print(f"Wrote {target_catalog_cc_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
