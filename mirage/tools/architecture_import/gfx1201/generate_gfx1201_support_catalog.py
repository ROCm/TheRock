#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
MIRAGE_ROOT = SCRIPT_PATH.parents[3]

INVENTORY_PATH = (
    MIRAGE_ROOT / "tests" / "data" / "architecture_import" / "gfx1201" /
    "gfx1201_instruction_inventory.json"
)
GFX950_CATALOG_PATH = MIRAGE_ROOT / "tests" / "data" / "gfx950_instruction_catalog.json"
GFX950_SUPPORT_PATH = (
    MIRAGE_ROOT / "tests" / "data" / "coverage" / "gfx950_support_snapshot.json"
)

SUPPORT_CATALOG_CC_PATH = (
    MIRAGE_ROOT / "native" / "src" / "isa" / "gfx1201" / "support_catalog.cc"
)
REPORT_DIR = MIRAGE_ROOT / "reports" / "architectures" / "gfx1201"
SUPPORT_MATRIX_JSON_PATH = REPORT_DIR / "gfx1201_support_matrix.json"
SUPPORT_MATRIX_MD_PATH = REPORT_DIR / "gfx1201_support_matrix.md"

ROLLUP_ORDER = [
    "transferable_as_is",
    "transferable_with_decoder_work",
    "transferable_with_semantic_work",
    "gfx1201_specific",
]

STATE_ORDER = [
    "transferable_as_is",
    "transferable_with_decoder_work",
    "transferable_with_semantic_work",
    "transferable_with_decoder_and_semantic_work",
    "gfx1201_specific",
]

ROLLUP_DESCRIPTIONS = {
    "transferable_as_is": (
        "Instruction name is already proven in gfx950 with both raw decode "
        "coverage and interpreter semantics."
    ),
    "transferable_with_decoder_work": (
        "Instruction name exists in gfx950 lineage, but Mirage still needs "
        "decoder work before the local target can rely on it. This rollup is "
        "decoder-first and includes opcodes that still need both decoder and "
        "semantic work."
    ),
    "transferable_with_semantic_work": (
        "Binary decode precedent exists in gfx950 coverage, but execution "
        "semantics still need to be carried over."
    ),
    "gfx1201_specific": (
        "Instruction name is absent from the gfx950 catalog and needs "
        "gfx1201-local handling."
    ),
}

STATE_DESCRIPTIONS = {
    "transferable_as_is": (
        "Both raw decode coverage and semantics already exist in gfx950."
    ),
    "transferable_with_decoder_work": (
        "Semantics exist in gfx950 coverage, but raw decode support is still missing."
    ),
    "transferable_with_semantic_work": (
        "Raw decode support exists in gfx950 coverage, but semantics are still missing."
    ),
    "transferable_with_decoder_and_semantic_work": (
        "Instruction name exists in gfx950, but neither decode nor semantics are ready."
    ),
    "gfx1201_specific": (
        "Instruction name is new relative to gfx950."
    ),
}

ROLLUP_ENUM = {
    "transferable_as_is": "Gfx1201SupportRollup::kTransferableAsIs",
    "transferable_with_decoder_work": "Gfx1201SupportRollup::kTransferableWithDecoderWork",
    "transferable_with_semantic_work": "Gfx1201SupportRollup::kTransferableWithSemanticWork",
    "gfx1201_specific": "Gfx1201SupportRollup::kGfx1201Specific",
}

STATE_ENUM = {
    "transferable_as_is": "Gfx1201SupportState::kTransferableAsIs",
    "transferable_with_decoder_work": "Gfx1201SupportState::kTransferableWithDecoderWork",
    "transferable_with_semantic_work": "Gfx1201SupportState::kTransferableWithSemanticWork",
    "transferable_with_decoder_and_semantic_work": (
        "Gfx1201SupportState::kTransferableWithDecoderAndSemanticWork"
    ),
    "gfx1201_specific": "Gfx1201SupportState::kGfx1201Specific",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def family_for_instruction(name: str) -> str:
    if "_" not in name:
        return name.lower()
    return name.split("_", 1)[0].lower()


def classify_instruction(
    instruction_name: str,
    gfx950_names: set[str],
    support_by_name: dict[str, dict],
) -> tuple[str, str, bool, bool, bool]:
    if instruction_name not in gfx950_names:
        return "gfx1201_specific", "gfx1201_specific", False, False, False

    support = support_by_name.get(instruction_name)
    decode_supported = bool(support and support["decode_supported"])
    semantic_supported = bool(support and support["semantic_supported"])
    if decode_supported and semantic_supported:
        return (
            "transferable_as_is",
            "transferable_as_is",
            True,
            True,
            True,
        )
    if not decode_supported and semantic_supported:
        return (
            "transferable_with_decoder_work",
            "transferable_with_decoder_work",
            True,
            False,
            True,
        )
    if decode_supported and not semantic_supported:
        return (
            "transferable_with_semantic_work",
            "transferable_with_semantic_work",
            True,
            True,
            False,
        )
    return (
        "transferable_with_decoder_work",
        "transferable_with_decoder_and_semantic_work",
        True,
        False,
        False,
    )


def compute_ranges(records: list[dict], key: str, ordered_values: list[str]) -> dict[str, dict]:
    ranges: dict[str, dict] = {}
    for value in ordered_values:
        begin = None
        count = 0
        for index, record in enumerate(records):
            if record[key] != value:
                continue
            if begin is None:
                begin = index
            count += 1
        ranges[value] = {
            "begin": 0 if begin is None else begin,
            "count": count,
        }
    return ranges


def cxx_string(value: str) -> str:
    return json.dumps(value)


def cxx_bool(value: bool) -> str:
    return "true" if value else "false"


def render_support_catalog_cc(matrix: dict) -> str:
    metadata = matrix["metadata"]
    records = matrix["instructions"]
    encodings = matrix["encodings"]

    lines = [
        '#include "lib/sim/isa/gfx1201/support_catalog.h"',
        "",
        "#include <array>",
        "",
        "namespace mirage::sim::isa {",
        "namespace {",
        "",
        "// Generated by tools/architecture_import/gfx1201/generate_gfx1201_support_catalog.py",
        f"constexpr InstructionCatalogMetadata kCatalogMetadata{{{cxx_string(metadata['gfx_target'])}, "
        f"{cxx_string(metadata['architecture_name'])}, {cxx_string(metadata['release_date'])}, "
        f"{cxx_string(metadata['schema_version'])}, {cxx_string(metadata['source_xml'])}, "
        f"{metadata['instruction_count']}u, {metadata['encoding_count']}u}};",
        "",
        "struct SupportRange {",
        "  std::uint32_t begin = 0;",
        "  std::uint32_t count = 0;",
        "};",
        "",
        f"constexpr std::array<InstructionEncodingSpec, {len(encodings)}> kEncodingSpecs{{{{",
    ]

    for encoding in encodings:
        lines.append(
            f"    {{{cxx_string(encoding['encoding_name'])}, "
            f"{cxx_string(encoding['encoding_condition'])}, {encoding['opcode']}u, "
            f"{encoding['operand_count']}u}},"
        )

    lines.extend(
        [
            "}};",
            "",
            f"constexpr std::array<Gfx1201InstructionSupportInfo, {len(records)}> kInstructions{{{{",
        ]
    )

    for record in records:
        flags = record["flags"]
        lines.append(
            f"    {{{cxx_string(record['instruction_name'])}, "
            f"{ROLLUP_ENUM[record['rollup']]}, {STATE_ENUM[record['state']]}, "
            f"{{{cxx_bool(flags['is_branch'])}, {cxx_bool(flags['is_conditional_branch'])}, "
            f"{cxx_bool(flags['is_indirect_branch'])}, "
            f"{cxx_bool(flags['is_program_terminator'])}, "
            f"{cxx_bool(flags['is_immediately_executed'])}}}, "
            f"{cxx_bool(record['known_in_gfx950_catalog'])}, "
            f"{cxx_bool(record['decoder_supported_in_gfx950'])}, "
            f"{cxx_bool(record['semantic_supported_in_gfx950'])}, "
            f"{record['encoding_begin']}u, {record['encoding_count']}u}},"
        )

    lines.extend(
        [
            "}};",
            "",
            f"constexpr std::array<SupportRange, {len(ROLLUP_ORDER)}> kRollupRanges{{{{",
        ]
    )

    for rollup in ROLLUP_ORDER:
        rng = matrix["rollup_ranges"][rollup]
        lines.append(f"    {{{rng['begin']}u, {rng['count']}u}},")

    lines.extend(
        [
            "}};",
            "",
            f"constexpr std::array<SupportRange, {len(STATE_ORDER)}> kStateRanges{{{{",
        ]
    )

    for state in STATE_ORDER:
        rng = matrix["state_ranges"][state]
        lines.append(f"    {{{rng['begin']}u, {rng['count']}u}},")

    lines.extend(
        [
            "}};",
            "",
            f"constexpr std::array<Gfx1201SupportSummary, {len(ROLLUP_ORDER)}> kRollupSummaries{{{{",
        ]
    )

    for rollup in ROLLUP_ORDER:
        summary = next(
            item for item in matrix["rollup_summaries"] if item["rollup"] == rollup
        )
        lines.append(
            f"    {{{ROLLUP_ENUM[rollup]}, {summary['instruction_count']}u, "
            f"{cxx_string(summary['description'])}}},"
        )

    lines.extend(
        [
            "}};",
            "",
            f"constexpr std::array<Gfx1201SupportStateSummary, {len(STATE_ORDER)}> kStateSummaries{{{{",
        ]
    )

    for state in STATE_ORDER:
        summary = next(item for item in matrix["state_summaries"] if item["state"] == state)
        lines.append(
            f"    {{{STATE_ENUM[state]}, {summary['instruction_count']}u, "
            f"{cxx_string(summary['description'])}}},"
        )

    lines.extend(
        [
            "}};",
            "",
            "std::span<const Gfx1201InstructionSupportInfo> MakeInstructionSpan(",
            "    const SupportRange& range) {",
            "  return std::span<const Gfx1201InstructionSupportInfo>(",
            "      kInstructions.data() + range.begin, range.count);",
            "}",
            "",
            "}  // namespace",
            "",
            "const InstructionCatalogMetadata& GetGfx1201SupportCatalogMetadata() {",
            "  return kCatalogMetadata;",
            "}",
            "",
            "std::span<const Gfx1201InstructionSupportInfo>",
            "GetGfx1201InstructionSupportCatalog() {",
            "  return kInstructions;",
            "}",
            "",
            "std::span<const InstructionEncodingSpec> GetGfx1201InstructionSupportEncodings() {",
            "  return kEncodingSpecs;",
            "}",
            "",
            "const Gfx1201InstructionSupportInfo* FindGfx1201InstructionSupport(",
            "    std::string_view instruction_name) {",
            "  for (const Gfx1201InstructionSupportInfo& instruction : kInstructions) {",
            "    if (instruction.instruction_name == instruction_name) {",
            "      return &instruction;",
            "    }",
            "  }",
            "  return nullptr;",
            "}",
            "",
            "std::span<const Gfx1201InstructionSupportInfo> GetGfx1201InstructionsByRollup(",
            "    Gfx1201SupportRollup rollup) {",
            "  switch (rollup) {",
            "    case Gfx1201SupportRollup::kTransferableAsIs:",
            "      return MakeInstructionSpan(kRollupRanges[0]);",
            "    case Gfx1201SupportRollup::kTransferableWithDecoderWork:",
            "      return MakeInstructionSpan(kRollupRanges[1]);",
            "    case Gfx1201SupportRollup::kTransferableWithSemanticWork:",
            "      return MakeInstructionSpan(kRollupRanges[2]);",
            "    case Gfx1201SupportRollup::kGfx1201Specific:",
            "      return MakeInstructionSpan(kRollupRanges[3]);",
            "  }",
            "  return {};",
            "}",
            "",
            "std::span<const Gfx1201InstructionSupportInfo> GetGfx1201InstructionsByState(",
            "    Gfx1201SupportState state) {",
            "  switch (state) {",
            "    case Gfx1201SupportState::kTransferableAsIs:",
            "      return MakeInstructionSpan(kStateRanges[0]);",
            "    case Gfx1201SupportState::kTransferableWithDecoderWork:",
            "      return MakeInstructionSpan(kStateRanges[1]);",
            "    case Gfx1201SupportState::kTransferableWithSemanticWork:",
            "      return MakeInstructionSpan(kStateRanges[2]);",
            "    case Gfx1201SupportState::kTransferableWithDecoderAndSemanticWork:",
            "      return MakeInstructionSpan(kStateRanges[3]);",
            "    case Gfx1201SupportState::kGfx1201Specific:",
            "      return MakeInstructionSpan(kStateRanges[4]);",
            "  }",
            "  return {};",
            "}",
            "",
            "std::span<const InstructionEncodingSpec> GetGfx1201Encodings(",
            "    const Gfx1201InstructionSupportInfo& instruction) {",
            "  return std::span<const InstructionEncodingSpec>(",
            "      kEncodingSpecs.data() + instruction.encoding_begin,",
            "      instruction.encoding_count);",
            "}",
            "",
            "std::span<const Gfx1201SupportSummary> GetGfx1201SupportRollupSummaries() {",
            "  return kRollupSummaries;",
            "}",
            "",
            "std::span<const Gfx1201SupportStateSummary> GetGfx1201SupportStateSummaries() {",
            "  return kStateSummaries;",
            "}",
            "",
            "std::string_view ToString(Gfx1201SupportRollup rollup) {",
            "  switch (rollup) {",
            "    case Gfx1201SupportRollup::kTransferableAsIs:",
            '      return "transferable_as_is";',
            "    case Gfx1201SupportRollup::kTransferableWithDecoderWork:",
            '      return "transferable_with_decoder_work";',
            "    case Gfx1201SupportRollup::kTransferableWithSemanticWork:",
            '      return "transferable_with_semantic_work";',
            "    case Gfx1201SupportRollup::kGfx1201Specific:",
            '      return "gfx1201_specific";',
            "  }",
            '  return "unknown";',
            "}",
            "",
            "std::string_view ToString(Gfx1201SupportState state) {",
            "  switch (state) {",
            "    case Gfx1201SupportState::kTransferableAsIs:",
            '      return "transferable_as_is";',
            "    case Gfx1201SupportState::kTransferableWithDecoderWork:",
            '      return "transferable_with_decoder_work";',
            "    case Gfx1201SupportState::kTransferableWithSemanticWork:",
            '      return "transferable_with_semantic_work";',
            "    case Gfx1201SupportState::kTransferableWithDecoderAndSemanticWork:",
            '      return "transferable_with_decoder_and_semantic_work";',
            "    case Gfx1201SupportState::kGfx1201Specific:",
            '      return "gfx1201_specific";',
            "  }",
            '  return "unknown";',
            "}",
            "",
            "}  // namespace mirage::sim::isa",
            "",
        ]
    )
    return "\n".join(lines)


def render_support_matrix_markdown(matrix: dict) -> str:
    metadata = matrix["metadata"]
    lines = [
        "# gfx1201 Support Matrix",
        "",
        "## Imported Baseline",
        "",
        f"- Target: `{metadata['gfx_target']}`",
        f"- Architecture source: `{metadata['architecture_name']}`",
        f"- Release date: `{metadata['release_date']}`",
        f"- Source XML: `{metadata['source_xml']}`",
        f"- Imported instructions: `{metadata['instruction_count']}`",
        f"- Imported encodings: `{metadata['encoding_count']}`",
        "",
        "## Rollup Categories",
        "",
        (
            "- `transferable_with_decoder_work` is decoder-first: the "
            "`transferable_with_decoder_and_semantic_work` subset stays in this "
            "rollup because decode is still the first local integration blocker."
        ),
        "",
    ]

    for summary in matrix["rollup_summaries"]:
        top_families = ", ".join(
            f"{item['family_name']}({item['instruction_count']})"
            for item in summary["top_families"]
        )
        samples = ", ".join(f"`{value}`" for value in summary["sample_instructions"])
        lines.extend(
            [
                f"### {summary['rollup']}",
                "",
                f"- Instruction count: `{summary['instruction_count']}`",
                f"- Meaning: {summary['description']}",
                f"- Top families: {top_families or 'None'}",
                f"- Sample instructions: {samples or 'None'}",
                "",
            ]
        )

    lines.extend(["## Exact States", ""])
    for summary in matrix["state_summaries"]:
        samples = ", ".join(f"`{value}`" for value in summary["sample_instructions"])
        lines.extend(
            [
                f"### {summary['state']}",
                "",
                f"- Instruction count: `{summary['instruction_count']}`",
                f"- Meaning: {summary['description']}",
                f"- Sample instructions: {samples or 'None'}",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUPPORT_CATALOG_CC_PATH.parent.mkdir(parents=True, exist_ok=True)

    inventory = read_json(INVENTORY_PATH)
    gfx950_catalog = read_json(GFX950_CATALOG_PATH)
    gfx950_support = read_json(GFX950_SUPPORT_PATH)

    gfx950_names = {
        instruction["instruction_name"] for instruction in gfx950_catalog["instructions"]
    }
    support_by_name = {
        item["instruction_name"]: item for item in gfx950_support["instructions"]
    }

    rollup_index = {name: index for index, name in enumerate(ROLLUP_ORDER)}
    state_index = {name: index for index, name in enumerate(STATE_ORDER)}

    records = []
    rollup_family_counts: dict[str, Counter] = defaultdict(Counter)
    state_counts = Counter()
    rollup_counts = Counter()
    for instruction in inventory["instructions"]:
        instruction_name = instruction["instruction_name"]
        rollup, state, known_in_gfx950, decoder_supported, semantic_supported = (
            classify_instruction(instruction_name, gfx950_names, support_by_name)
        )
        family_name = family_for_instruction(instruction_name)
        record = {
            "instruction_name": instruction_name,
            "rollup": rollup,
            "state": state,
            "family_name": family_name,
            "flags": instruction["flags"],
            "known_in_gfx950_catalog": known_in_gfx950,
            "decoder_supported_in_gfx950": decoder_supported,
            "semantic_supported_in_gfx950": semantic_supported,
            "encodings": instruction["encodings"],
        }
        records.append(record)
        rollup_counts[rollup] += 1
        state_counts[state] += 1
        rollup_family_counts[rollup][family_name] += 1

    records.sort(
        key=lambda item: (
            rollup_index[item["rollup"]],
            state_index[item["state"]],
            item["instruction_name"],
        )
    )

    encoding_rows = []
    for record in records:
        record["encoding_begin"] = len(encoding_rows)
        record["encoding_count"] = len(record["encodings"])
        for encoding in record["encodings"]:
            encoding_rows.append(
                {
                    "encoding_name": encoding["encoding_name"],
                    "encoding_condition": encoding["encoding_condition"],
                    "opcode": encoding["opcode"],
                    "operand_count": encoding["operand_count"],
                }
            )
        del record["encodings"]

    rollup_ranges = compute_ranges(records, "rollup", ROLLUP_ORDER)
    state_ranges = compute_ranges(records, "state", STATE_ORDER)

    matrix = {
        "metadata": inventory["metadata"],
        "rollup_ranges": rollup_ranges,
        "state_ranges": state_ranges,
        "rollup_summaries": [
            {
                "rollup": rollup,
                "instruction_count": rollup_counts[rollup],
                "description": ROLLUP_DESCRIPTIONS[rollup],
                "top_families": [
                    {
                        "family_name": family_name,
                        "instruction_count": count,
                    }
                    for family_name, count in rollup_family_counts[rollup].most_common(8)
                ],
                "sample_instructions": [
                    record["instruction_name"]
                    for record in records
                    if record["rollup"] == rollup
                ][:12],
            }
            for rollup in ROLLUP_ORDER
        ],
        "state_summaries": [
            {
                "state": state,
                "instruction_count": state_counts[state],
                "description": STATE_DESCRIPTIONS[state],
                "sample_instructions": [
                    record["instruction_name"]
                    for record in records
                    if record["state"] == state
                ][:12],
            }
            for state in STATE_ORDER
        ],
        "instructions": records,
        "encodings": encoding_rows,
    }

    SUPPORT_MATRIX_JSON_PATH.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SUPPORT_MATRIX_MD_PATH.write_text(
        render_support_matrix_markdown(matrix),
        encoding="utf-8",
    )
    SUPPORT_CATALOG_CC_PATH.write_text(
        render_support_catalog_cc(matrix),
        encoding="utf-8",
    )

    print(f"Wrote {SUPPORT_MATRIX_JSON_PATH.relative_to(MIRAGE_ROOT)}")
    print(f"Wrote {SUPPORT_MATRIX_MD_PATH.relative_to(MIRAGE_ROOT)}")
    print(f"Wrote {SUPPORT_CATALOG_CC_PATH.relative_to(MIRAGE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
