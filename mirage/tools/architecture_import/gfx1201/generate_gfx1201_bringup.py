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
REPORT_DIR = MIRAGE_ROOT / "reports" / "architectures" / "gfx1201"
PLAN_JSON_PATH = REPORT_DIR / "gfx1201_bringup_plan.json"
PLAN_MD_PATH = REPORT_DIR / "gfx1201_bringup_plan.md"

BUCKET_ORDER = [
    "transferable_full",
    "transferable_decode_only",
    "transferable_semantic_only",
    "known_but_unsupported",
    "new_vs_gfx950",
]

BUCKET_DESCRIPTIONS = {
    "transferable_full": (
        "Present in gfx950 with both raw decode coverage and interpreter "
        "semantics already implemented."
    ),
    "transferable_decode_only": (
        "Binary shape already appears in gfx950 coverage, but execution "
        "semantics are still missing there."
    ),
    "transferable_semantic_only": (
        "Execution semantics exist in gfx950 coverage, but binary decode "
        "coverage has not been wired up yet."
    ),
    "known_but_unsupported": (
        "Opcode name already exists in the gfx950 catalog, but neither "
        "decoder nor interpreter support is complete."
    ),
    "new_vs_gfx950": (
        "Opcode name is absent from gfx950 and needs RDNA4-local decode "
        "and semantic work."
    ),
}

PHASE0_ENCODINGS = [
    ("ENC_SOPP", "S_ENDPGM", "Program control, wait/event sequencing, and branch bring-up."),
    ("ENC_SOP1", "S_MOV_B32", "Scalar move, bit-manipulation, and exec-mask setup precedent."),
    ("ENC_SOP2", "S_AND_B32", "Scalar binary core needed before wider control-flow work."),
    ("ENC_SOPC", "S_CMP_EQ_U32", "Scalar compare path for branch and predicate plumbing."),
    ("ENC_SOPK", "S_MOVK_I32", "Small scalar-immediate surface used by early control kernels."),
    ("ENC_SMEM", "S_LOAD_B32", "RDNA4 scalar memory is a first architecture-local blocker."),
    ("ENC_VOP1", "V_MOV_B32", "Vector move and conversion baseline reused across many programs."),
    ("ENC_VOP2", "V_ADD_F32", "Core vector arithmetic used by both compute and graphics paths."),
    ("ENC_VOPC", "V_CMP_EQ_F32", "Vector compare path for control, masking, and shader predicates."),
    ("ENC_VOP3", "V_ADD3_U32", "Largest instruction family and the main overlap with gfx950 precedent."),
    ("ENC_VDS", "DS_ADD_U32", "LDS data path with a small fully-supported carry-over subset."),
    ("ENC_VGLOBAL", "GLOBAL_LOAD_B32", "Global memory load/store family that replaces gfx950-specific naming."),
]

PHASE1_ENCODINGS = [
    ("ENC_VBUFFER", "BUFFER_LOAD_FORMAT_X", "RDNA4 buffer resource path and typed buffer forms."),
    ("ENC_VFLAT", "FLAT_LOAD_B32", "Flat memory path layered after scalar/global addressing is stable."),
    ("ENC_VSCRATCH", "SCRATCH_LOAD_B32", "Scratch memory path is target-local and can come after flat/global."),
    ("ENC_VSAMPLE", "IMAGE_SAMPLE", "Graphics sampling path unique to the RDNA4 import."),
    ("ENC_VIMAGE", "IMAGE_LOAD", "Image load/store/atomic path separate from sampled operations."),
    ("ENC_VEXPORT", "EXPORT", "Graphics export path with no gfx950 precedent."),
    ("ENC_VINTERP", "V_INTERP_P10_F32", "Shader interpolation path that anchors graphics-local vector work."),
    ("ENC_VOP3P", "V_PK_ADD_F16", "Packed/vector math delta visible in the RDNA4 import."),
]

CARRY_OVER_FAMILIES = [
    ("v", "transferable_full", "V_MOV_B32", "Largest fully-supported carry-over bucket from gfx950."),
    ("s", "transferable_full", "S_MOV_B32", "Scalar control and ALU subset with direct gfx950 precedent."),
    ("ds", "transferable_full", "DS_ADD_U32", "Small LDS subset already proven end-to-end on gfx950."),
    ("global", "transferable_full", "GLOBAL_ATOMIC_ADD_F32", "Only a narrow global atomic subset currently has full precedent."),
    ("v", "transferable_decode_only", "V_CVT_F32_FP8", "Binary decode precedent exists, but execution semantics still need work."),
    ("v", "transferable_semantic_only", "V_CMP_LT_F16", "Interpreter precedent exists, but raw binary decode work is missing."),
    ("s", "transferable_decode_only", "S_GETPC_B64", "Scalar control opcodes appear in coverage but are not executable yet."),
]

RDNA4_DELTA_FAMILIES = [
    ("s", "new_vs_gfx950", "S_LOAD_B32", "Scalar memory and wait/barrier control differ from gfx950."),
    ("v", "new_vs_gfx950", "V_INTERP_P10_F32", "Graphics-local vector forms expand the RDNA4 surface."),
    ("image", "new_vs_gfx950", "IMAGE_LOAD", "Image pipeline has no direct gfx950 baseline."),
    ("buffer", "new_vs_gfx950", "BUFFER_LOAD_FORMAT_X", "Buffer load/store/atomic naming and resource handling are RDNA4-local."),
    ("global", "new_vs_gfx950", "GLOBAL_LOAD_B32", "Global memory family uses the RDNA4 ISA surface instead of gfx950 forms."),
    ("ds", "new_vs_gfx950", "DS_LOAD_B32", "LDS load/store and atomic expansion beyond the gfx950 subset."),
    ("flat", "new_vs_gfx950", "FLAT_LOAD_B32", "Flat addressing path needs RDNA4-specific decode and operand wiring."),
    ("scratch", "new_vs_gfx950", "SCRATCH_LOAD_B32", "Scratch memory is absent from the current gfx950-local layout."),
    ("tbuffer", "new_vs_gfx950", "TBUFFER_LOAD_FORMAT_X", "Typed buffer graphics path should layer on top of buffer support."),
    ("export", "new_vs_gfx950", "EXPORT", "Graphics export path is unique to the RDNA4 import."),
]

BLOCKERS = [
    (
        "GPUOpen publishes gfx1201 through the architecture-level RDNA4 XML, "
        "so target mapping cannot assume one XML per target."
    ),
    (
        "gfx1201 barrier/wait control differs from gfx950: the import contains "
        "`S_BARRIER_WAIT` and `S_WAIT_*`, while `S_BARRIER` is absent."
    ),
    (
        "RDNA4-only graphics and memory encodings (`ENC_VBUFFER`, `ENC_VIMAGE`, "
        "`ENC_VSAMPLE`, `ENC_VFLAT`, `ENC_VSCRATCH`, `ENC_VEXPORT`, "
        "`ENC_VINTERP`, `ENC_VOP3P`) need architecture-local plumbing before "
        "gfx950 logic can be reused."
    ),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def family_for_instruction(name: str) -> str:
    if "_" not in name:
        return name.lower()
    return name.split("_", 1)[0].lower()


def classify_bucket(name: str, gfx950_names: set[str], support_by_name: dict[str, dict]) -> str:
    if name not in gfx950_names:
        return "new_vs_gfx950"
    coverage = support_by_name.get(name)
    if coverage is None:
        return "known_but_unsupported"
    if coverage["semantic_supported"] and coverage["decode_supported"]:
        return "transferable_full"
    if coverage["decode_supported"]:
        return "transferable_decode_only"
    if coverage["semantic_supported"]:
        return "transferable_semantic_only"
    return "known_but_unsupported"


def build_focus_rows(
    definitions: list[tuple[str, str, str]],
    counts_by_name: dict[str, int],
    known_instruction_names: set[str],
) -> list[dict]:
    rows = []
    for name, example_instruction, rationale in definitions:
        if example_instruction not in known_instruction_names:
            raise RuntimeError(f"missing representative instruction: {example_instruction}")
        count = counts_by_name.get(name, 0)
        if count == 0:
            raise RuntimeError(f"missing count for focus item: {name}")
        rows.append(
            {
                "name": name,
                "instruction_count": count,
                "example_instruction": example_instruction,
                "rationale": rationale,
            }
        )
    return rows


def build_family_rows(
    definitions: list[tuple[str, str, str, str]],
    counts_by_bucket: dict[str, Counter],
    known_instruction_names: set[str],
) -> list[dict]:
    rows = []
    for family_name, bucket, example_instruction, rationale in definitions:
        if example_instruction not in known_instruction_names:
            raise RuntimeError(f"missing representative instruction: {example_instruction}")
        count = counts_by_bucket[bucket][family_name]
        if count == 0:
            raise RuntimeError(
                f"missing family count for {family_name} in bucket {bucket}"
            )
        rows.append(
            {
                "family_name": family_name,
                "bucket": bucket,
                "instruction_count": count,
                "example_instruction": example_instruction,
                "rationale": rationale,
            }
        )
    return rows


def render_markdown(plan: dict) -> str:
    metadata = plan["metadata"]
    lines = [
        "# gfx1201 Bring-up Plan",
        "",
        "## Imported Baseline",
        "",
        f"- Imported target: `{metadata['gfx_target']}`",
        f"- Architecture source: `{metadata['architecture_name']}`",
        f"- Release date: `{metadata['release_date']}`",
        f"- Schema version: `{metadata['schema_version']}`",
        f"- Source XML: `{metadata['source_xml']}`",
        f"- Imported instructions: `{metadata['instruction_count']}`",
        f"- Imported encodings: `{metadata['encoding_count']}`",
        "",
        "## Support Buckets",
        "",
    ]

    for bucket in plan["support_buckets"]:
        top_families = ", ".join(
            f"{item['family_name']}({item['instruction_count']})"
            for item in bucket["top_families"]
        )
        top_encodings = ", ".join(
            f"{item['encoding_name']}({item['instruction_count']})"
            for item in bucket["top_encodings"]
        )
        lines.extend(
            [
                f"### {bucket['bucket']}",
                "",
                f"- Instruction count: `{bucket['instruction_count']}`",
                f"- Meaning: {bucket['description']}",
                f"- Top families: {top_families or 'None'}",
                f"- Top encodings: {top_encodings or 'None'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Phase 0 Decoder Focus",
            "",
        ]
    )
    for item in plan["phase0_decoder_focus"]:
        lines.append(
            f"- `{item['name']}` ({item['instruction_count']} instructions, "
            f"example `{item['example_instruction']}`): {item['rationale']}"
        )

    lines.extend(
        [
            "",
            "## Phase 1 Decoder Focus",
            "",
        ]
    )
    for item in plan["phase1_decoder_focus"]:
        lines.append(
            f"- `{item['name']}` ({item['instruction_count']} instructions, "
            f"example `{item['example_instruction']}`): {item['rationale']}"
        )

    lines.extend(
        [
            "",
            "## Carry-over Family Focus",
            "",
        ]
    )
    for item in plan["carry_over_family_focus"]:
        lines.append(
            f"- `{item['family_name']}` / `{item['bucket']}` "
            f"({item['instruction_count']} instructions, example "
            f"`{item['example_instruction']}`): {item['rationale']}"
        )

    lines.extend(
        [
            "",
            "## RDNA4 Delta Family Focus",
            "",
        ]
    )
    for item in plan["rdna4_delta_family_focus"]:
        lines.append(
            f"- `{item['family_name']}` / `{item['bucket']}` "
            f"({item['instruction_count']} instructions, example "
            f"`{item['example_instruction']}`): {item['rationale']}"
        )

    lines.extend(
        [
            "",
            "## Blockers",
            "",
        ]
    )
    for blocker in plan["blockers"]:
        lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "- This plan is derived from the imported gfx1201 inventory, "
                "the current gfx950 catalog, and the checked-in gfx950 support "
                "snapshot."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = read_json(INVENTORY_PATH)
    gfx950_catalog = read_json(GFX950_CATALOG_PATH)
    gfx950_support = read_json(GFX950_SUPPORT_PATH)

    known_instruction_names = {
        instruction["instruction_name"] for instruction in inventory["instructions"]
    }
    gfx950_names = {
        instruction["instruction_name"] for instruction in gfx950_catalog["instructions"]
    }
    support_by_name = {
        item["instruction_name"]: item for item in gfx950_support["instructions"]
    }

    bucket_counts = Counter()
    family_counts_by_bucket: dict[str, Counter] = defaultdict(Counter)
    encoding_counts_by_bucket: dict[str, Counter] = defaultdict(Counter)
    encoding_counts: dict[str, set[str]] = defaultdict(set)

    for instruction in inventory["instructions"]:
        name = instruction["instruction_name"]
        bucket = classify_bucket(name, gfx950_names, support_by_name)
        bucket_counts[bucket] += 1
        family_counts_by_bucket[bucket][family_for_instruction(name)] += 1

        seen_encodings = set()
        for encoding in instruction["encodings"]:
            encoding_name = encoding["encoding_name"]
            encoding_counts[encoding_name].add(name)
            if encoding_name in seen_encodings:
                continue
            encoding_counts_by_bucket[bucket][encoding_name] += 1
            seen_encodings.add(encoding_name)

    support_buckets = []
    for bucket in BUCKET_ORDER:
        support_buckets.append(
            {
                "bucket": bucket,
                "instruction_count": bucket_counts[bucket],
                "description": BUCKET_DESCRIPTIONS[bucket],
                "top_families": [
                    {
                        "family_name": family_name,
                        "instruction_count": count,
                    }
                    for family_name, count in family_counts_by_bucket[bucket].most_common(8)
                ],
                "top_encodings": [
                    {
                        "encoding_name": encoding_name,
                        "instruction_count": count,
                    }
                    for encoding_name, count in encoding_counts_by_bucket[bucket].most_common(10)
                ],
            }
        )

    plan = {
        "metadata": inventory["metadata"],
        "support_buckets": support_buckets,
        "phase0_decoder_focus": build_focus_rows(
            PHASE0_ENCODINGS,
            {name: len(instructions) for name, instructions in encoding_counts.items()},
            known_instruction_names,
        ),
        "phase1_decoder_focus": build_focus_rows(
            PHASE1_ENCODINGS,
            {name: len(instructions) for name, instructions in encoding_counts.items()},
            known_instruction_names,
        ),
        "carry_over_family_focus": build_family_rows(
            CARRY_OVER_FAMILIES, family_counts_by_bucket, known_instruction_names
        ),
        "rdna4_delta_family_focus": build_family_rows(
            RDNA4_DELTA_FAMILIES, family_counts_by_bucket, known_instruction_names
        ),
        "blockers": BLOCKERS,
    }

    PLAN_JSON_PATH.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    PLAN_MD_PATH.write_text(render_markdown(plan), encoding="utf-8")

    print(f"Wrote {PLAN_JSON_PATH.relative_to(MIRAGE_ROOT)}")
    print(f"Wrote {PLAN_MD_PATH.relative_to(MIRAGE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
