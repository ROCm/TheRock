#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MIRAGE_ROOT = Path(__file__).resolve().parents[3]
THEROCK_ROOT = MIRAGE_ROOT.parent
LLVM_ROOT = THEROCK_ROOT / "compiler" / "amd-llvm" / "llvm"

THIRD_PARTY_DIR = MIRAGE_ROOT / "third_party" / "amd_gpu_isa" / "gfx1201"
DATA_DIR = MIRAGE_ROOT / "tests" / "data" / "architecture_import" / "gfx1201"
REPORT_DIR = MIRAGE_ROOT / "reports" / "architectures" / "gfx1201"

PINNED_BUNDLE_URL = "https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2025_09_05.zip"
PINNED_BUNDLE_NAME = "AMD_GPU_MR_ISA_XML_2025_09_05.zip"
PINNED_XML_NAME = "amdgpu_isa_rdna4.xml"
TARGET_NAME = "gfx1201"

LLVM_GCN_PROCESSORS = LLVM_ROOT / "lib" / "Target" / "AMDGPU" / "GCNProcessors.td"
LLVM_SCAN_ROOTS = (
    LLVM_ROOT / "lib" / "Target" / "AMDGPU",
    LLVM_ROOT / "test" / "CodeGen" / "AMDGPU",
)
GFX950_CATALOG_JSON = MIRAGE_ROOT / "tests" / "data" / "gfx950_instruction_catalog.json"


@dataclass(frozen=True)
class EncodingRecord:
    encoding_name: str
    encoding_condition: str
    opcode: int
    operand_count: int


@dataclass(frozen=True)
class InstructionRecord:
    instruction_name: str
    flags: dict[str, bool]
    encodings: tuple[EncodingRecord, ...]


def _text(element: ET.Element | None, tag: str, default: str = "") -> str:
    if element is None:
        return default
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _bool_text(element: ET.Element | None, tag: str) -> bool:
    return _text(element, tag, "FALSE").upper() == "TRUE"


def _ensure_dirs() -> None:
    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _download_head(url: str) -> tuple[str, dict[str, str]]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.geturl(), headers


def _download_and_extract_xml(url: str, xml_name: str, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        bundle_path = tmpdir_path / PINNED_BUNDLE_NAME
        with urllib.request.urlopen(url) as response, bundle_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(bundle_path) as bundle:
            with bundle.open(xml_name) as xml_in, destination.open("wb") as xml_out:
                shutil.copyfileobj(xml_in, xml_out)


def _load_instruction_records(source_xml: Path) -> tuple[dict[str, str], list[InstructionRecord]]:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    isa = root.find("ISA")
    if isa is None:
        raise ValueError("missing ISA node")

    document = root.find("Document")
    metadata = {
        "gfx_target": TARGET_NAME,
        "architecture_name": _text(isa.find("Architecture"), "ArchitectureName"),
        "release_date": _text(document, "ReleaseDate"),
        "schema_version": _text(document, "SchemaVersion"),
        "source_xml": source_xml.name,
    }

    instructions_node = isa.find("Instructions")
    if instructions_node is None:
        raise ValueError("missing Instructions node")

    records: list[InstructionRecord] = []
    for instruction in instructions_node.findall("Instruction"):
        flags_node = instruction.find("InstructionFlags")
        encodings: list[EncodingRecord] = []
        encodings_node = instruction.find("InstructionEncodings")
        if encodings_node is None:
            raise ValueError("instruction missing InstructionEncodings")
        for encoding in encodings_node.findall("InstructionEncoding"):
            operands = encoding.find("Operands")
            operand_count = 0 if operands is None else len(operands.findall("Operand"))
            opcode_text = _text(encoding, "Opcode", "0")
            encodings.append(
                EncodingRecord(
                    encoding_name=_text(encoding, "EncodingName"),
                    encoding_condition=_text(encoding, "EncodingCondition", "default"),
                    opcode=int(opcode_text, 10),
                    operand_count=operand_count,
                )
            )

        records.append(
            InstructionRecord(
                instruction_name=_text(instruction, "InstructionName"),
                flags={
                    "is_branch": _bool_text(flags_node, "IsBranch"),
                    "is_conditional_branch": _bool_text(flags_node, "IsConditionalBranch"),
                    "is_indirect_branch": _bool_text(flags_node, "IsIndirectBranch"),
                    "is_program_terminator": _bool_text(flags_node, "IsProgramTerminator"),
                    "is_immediately_executed": _bool_text(flags_node, "IsImmediatelyExecuted"),
                },
                encodings=tuple(encodings),
            )
        )

    return metadata, records


def _render_inventory(metadata: dict[str, str], records: list[InstructionRecord]) -> dict:
    return {
        "metadata": {
            **metadata,
            "instruction_count": len(records),
            "encoding_count": sum(len(record.encodings) for record in records),
        },
        "instructions": [
            {
                "instruction_name": record.instruction_name,
                "flags": record.flags,
                "encodings": [
                    {
                        "encoding_name": encoding.encoding_name,
                        "encoding_condition": encoding.encoding_condition,
                        "opcode": encoding.opcode,
                        "operand_count": encoding.operand_count,
                    }
                    for encoding in record.encodings
                ],
            }
            for record in records
        ],
    }


def _scan_llvm_references() -> dict:
    processor_lines: list[dict[str, object]] = []
    gcn_lines = LLVM_GCN_PROCESSORS.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(gcn_lines, start=1):
        if f'"{TARGET_NAME}"' not in line:
            continue
        snippet = [line]
        if index < len(gcn_lines):
            snippet.append(gcn_lines[index])
        processor_lines.append(
            {
                "path": str(LLVM_GCN_PROCESSORS.relative_to(THEROCK_ROOT)),
                "line": index,
                "snippet": "\n".join(snippet),
            }
        )

    matches: list[dict[str, object]] = []
    total_occurrences = 0
    for root in LLVM_SCAN_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            occurrences = text.count(TARGET_NAME)
            if occurrences == 0:
                continue
            total_occurrences += occurrences
            matches.append(
                {
                    "path": str(path.relative_to(THEROCK_ROOT)),
                    "occurrences": occurrences,
                }
            )

    matches.sort(key=lambda item: (-(item["occurrences"]), item["path"]))
    return {
        "target": TARGET_NAME,
        "processor_model_entries": processor_lines,
        "match_count": len(matches),
        "total_occurrences": total_occurrences,
        "matching_files": matches,
    }


def _family_for_instruction(name: str) -> str:
    for prefix, family in (
        ("V_WMMA_", "wmma"),
        ("V_SWMMAC_", "wmma"),
        ("V_PK_", "packed"),
        ("BUFFER_", "buffer"),
        ("IMAGE_", "image"),
        ("EXP_", "export"),
        ("DS_", "lds"),
        ("FLAT_", "flat"),
        ("GLOBAL_", "global"),
        ("SCRATCH_", "scratch"),
        ("MIMG_", "image"),
        ("MTBUF_", "buffer"),
        ("MUBUF_", "buffer"),
    ):
        if name.startswith(prefix):
            return family
    if "_" not in name:
        return name.lower()
    return name.split("_", 1)[0].lower()


def _top_counts(names: set[str]) -> list[dict[str, object]]:
    counts = Counter(_family_for_instruction(name) for name in names)
    return [
        {"family": family, "count": count}
        for family, count in counts.most_common()
    ]


def _encoding_families(instructions: list[dict]) -> set[str]:
    families: set[str] = set()
    for instruction in instructions:
        for encoding in instruction["encodings"]:
            families.add(encoding["encoding_name"])
    return families


def _build_delta(inventory_payload: dict, llvm_inventory: dict) -> tuple[dict, str]:
    gfx1201_names = {
        instruction["instruction_name"] for instruction in inventory_payload["instructions"]
    }
    gfx950_catalog = json.loads(GFX950_CATALOG_JSON.read_text(encoding="utf-8"))
    gfx950_names = {
        instruction["instruction_name"] for instruction in gfx950_catalog["instructions"]
    }

    common = gfx1201_names & gfx950_names
    gfx1201_only = gfx1201_names - gfx950_names
    gfx950_only = gfx950_names - gfx1201_names

    gfx1201_encoding_families = _encoding_families(inventory_payload["instructions"])
    gfx950_encoding_families = _encoding_families(gfx950_catalog["instructions"])

    delta_payload = {
        "metadata": {
            "target": TARGET_NAME,
            "official_architecture_name": inventory_payload["metadata"]["architecture_name"],
            "compared_to": "gfx950",
            "llvm_target_hits": llvm_inventory["match_count"],
        },
        "counts": {
            "gfx1201_instruction_count": len(gfx1201_names),
            "gfx950_instruction_count": len(gfx950_names),
            "common_instruction_count": len(common),
            "gfx1201_only_instruction_count": len(gfx1201_only),
            "gfx950_only_instruction_count": len(gfx950_only),
        },
        "encoding_families": {
            "gfx1201_only": sorted(gfx1201_encoding_families - gfx950_encoding_families),
            "gfx950_only": sorted(gfx950_encoding_families - gfx1201_encoding_families),
            "common": sorted(gfx1201_encoding_families & gfx950_encoding_families),
        },
        "family_deltas": {
            "gfx1201_only_top": _top_counts(gfx1201_only),
            "gfx950_only_top": _top_counts(gfx950_only),
        },
        "instruction_sets": {
            "gfx1201_only": sorted(gfx1201_only),
            "gfx950_only": sorted(gfx950_only),
        },
    }

    highlights = [
        "# gfx1201 Ingest Report",
        "",
        f"- Official ISA source: `{PINNED_XML_NAME}` from `{PINNED_BUNDLE_NAME}`",
        f"- Official architecture name: `{inventory_payload['metadata']['architecture_name']}`",
        f"- LLVM processor model hits for `{TARGET_NAME}`: {llvm_inventory['match_count']}",
        f"- RDNA4 / gfx1201 instruction inventory: {len(gfx1201_names)}",
        f"- Current gfx950 instruction inventory: {len(gfx950_names)}",
        f"- Common instructions: {len(common)}",
        f"- gfx1201-only instructions: {len(gfx1201_only)}",
        f"- gfx950-only instructions: {len(gfx950_only)}",
        "",
        "## Major Family Deltas",
        "",
        "### gfx1201-only top families",
        "",
    ]
    for item in delta_payload["family_deltas"]["gfx1201_only_top"][:10]:
        highlights.append(f"- `{item['family']}`: {item['count']}")
    highlights.extend(["", "### gfx950-only top families", ""])
    for item in delta_payload["family_deltas"]["gfx950_only_top"][:10]:
        highlights.append(f"- `{item['family']}`: {item['count']}")
    highlights.extend(
        [
            "",
            "## Encoding Family Deltas",
            "",
            f"- gfx1201-only encoding families: {', '.join(delta_payload['encoding_families']['gfx1201_only'][:20]) or 'None'}",
            f"- gfx950-only encoding families: {', '.join(delta_payload['encoding_families']['gfx950_only'][:20]) or 'None'}",
            "",
            "## Simulator Implications",
            "",
            "- gfx1201 is sourced from the architecture-level RDNA4 XML, so the simulator needs a target-to-architecture mapping layer instead of assuming one XML per gpu target.",
            "- gfx1201 inherits the generic GFX12 feature set in local LLVM, unlike `gfx1250` which has a separate `FeatureISAVersion12_50` model. That suggests a shared RDNA4 baseline with target-specific deltas layered on later.",
            "- Compared with the current gfx950/CDNA4 baseline, gfx1201 shifts the bring-up priority toward RDNA-style graphics, buffer/image, export, and packed/vector families rather than CDNA matrix and accelerator-heavy paths.",
        ]
    )
    return delta_payload, "\n".join(highlights) + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    resolved_url, headers = _download_head(PINNED_BUNDLE_URL)
    xml_path = THIRD_PARTY_DIR / PINNED_XML_NAME
    _download_and_extract_xml(resolved_url, PINNED_XML_NAME, xml_path)

    metadata, records = _load_instruction_records(xml_path)
    inventory_payload = _render_inventory(metadata, records)
    llvm_inventory = _scan_llvm_references()

    manifest_payload = {
        "target": TARGET_NAME,
        "preferred_source": "official_gpuopen_machine_readable_isa",
        "bundle": {
            "requested_url": PINNED_BUNDLE_URL,
            "resolved_url": resolved_url,
            "filename": PINNED_BUNDLE_NAME,
            "etag": headers.get("etag", ""),
            "last_modified": headers.get("last-modified", ""),
            "content_length": headers.get("content-length", ""),
        },
        "source_xml": {
            "filename": PINNED_XML_NAME,
            "local_path": str(xml_path.relative_to(MIRAGE_ROOT)),
            "architecture_name": metadata["architecture_name"],
            "release_date": metadata["release_date"],
            "schema_version": metadata["schema_version"],
        },
        "llvm_mapping": {
            "processor_model_file": str(LLVM_GCN_PROCESSORS.relative_to(THEROCK_ROOT)),
            "processor_model_entries": llvm_inventory["processor_model_entries"],
            "match_count": llvm_inventory["match_count"],
        },
    }

    delta_payload, markdown_report = _build_delta(inventory_payload, llvm_inventory)

    _write_json(THIRD_PARTY_DIR / "source_manifest.json", manifest_payload)
    (THIRD_PARTY_DIR / "PROVENANCE.md").write_text(
        "\n".join(
            [
                "# gfx1201 Source Provenance",
                "",
                f"- Target: `{TARGET_NAME}`",
                f"- Preferred official source: `{PINNED_BUNDLE_NAME}` -> `{PINNED_XML_NAME}`",
                f"- Resolved bundle URL: `{resolved_url}`",
                f"- Architecture name in XML: `{metadata['architecture_name']}`",
                f"- XML release date: `{metadata['release_date']}`",
                f"- XML schema version: `{metadata['schema_version']}`",
                f"- Local LLVM processor-model anchor: `{LLVM_GCN_PROCESSORS.relative_to(THEROCK_ROOT)}`",
                "",
                "This ingest treats `gfx1201` as a target that maps onto the official architecture-level RDNA 4 machine-readable ISA, with local LLVM used as the target-to-architecture provenance layer.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(DATA_DIR / "gfx1201_instruction_inventory.json", inventory_payload)
    _write_json(DATA_DIR / "gfx1201_llvm_inventory.json", llvm_inventory)
    _write_json(DATA_DIR / "gfx1201_vs_gfx950.json", delta_payload)
    (REPORT_DIR / "gfx1201_vs_gfx950.md").write_text(markdown_report, encoding="utf-8")

    print(f"Wrote {xml_path}")
    print(f"Wrote {THIRD_PARTY_DIR / 'source_manifest.json'}")
    print(f"Wrote {THIRD_PARTY_DIR / 'PROVENANCE.md'}")
    print(f"Wrote {DATA_DIR / 'gfx1201_instruction_inventory.json'}")
    print(f"Wrote {DATA_DIR / 'gfx1201_llvm_inventory.json'}")
    print(f"Wrote {DATA_DIR / 'gfx1201_vs_gfx950.json'}")
    print(f"Wrote {REPORT_DIR / 'gfx1201_vs_gfx950.md'}")
    print(
        "Instruction counts:",
        inventory_payload["metadata"]["instruction_count"],
        "vs gfx950",
        delta_payload["counts"]["gfx950_instruction_count"],
    )


if __name__ == "__main__":
    main()
