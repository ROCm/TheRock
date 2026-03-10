#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


GPUOPEN_BUNDLE_URL = "https://gpuopen.com/download/machine-readable-isa/latest/"
RDNA4_XML_NAME = "amdgpu_isa_rdna4.xml"

LLVM_RELATIVE_FILES = [
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/GCNProcessors.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/AMDGPU.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/SISchedule.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/SOPInstructions.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/VOP1Instructions.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/VOP3Instructions.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/VOP3PInstructions.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/VOPInstructions.td",
    "compiler/amd-llvm/llvm/lib/Target/AMDGPU/MIMGInstructions.td",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the official RDNA4 XML from GPUOpen before generating artifacts.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_bool(text: str | None) -> bool:
    return (text or "").strip().upper() == "TRUE"


def parse_opcode(opcode_element: ET.Element | None) -> int | None:
    if opcode_element is None or opcode_element.text is None:
        return None
    text = opcode_element.text.strip()
    radix = opcode_element.get("Radix", "10")
    base = 10 if radix == "10" else 2 if radix == "2" else 16
    return int(text, base)


def parse_rdna4_inventory(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    document = root.find("Document")
    isa = root.find("ISA")
    if document is None or isa is None:
        raise RuntimeError("unexpected RDNA4 XML structure")

    architecture = isa.find("Architecture")
    metadata = {
        "architecture_name": architecture.findtext("ArchitectureName", default="").strip(),
        "architecture_id": architecture.findtext("ArchitectureId", default="").strip(),
        "release_date": document.findtext("ReleaseDate", default="").strip(),
        "schema_version": document.findtext("SchemaVersion", default="").strip(),
        "license": document.findtext("License", default="").strip(),
        "sensitivity": document.findtext("Sensitivity", default="").strip(),
        "source_xml": RDNA4_XML_NAME,
    }

    instructions_node = isa.find("Instructions")
    if instructions_node is None:
        raise RuntimeError("RDNA4 XML missing instruction list")

    instructions = []
    encoding_families: dict[str, int] = {}
    for instruction_node in instructions_node.findall("Instruction"):
        instruction_name = instruction_node.findtext("InstructionName", default="").strip()
        alias_node = instruction_node.find("AliasedInstructionNames")
        aliases = []
        if alias_node is not None:
            aliases = [
                alias.text.strip()
                for alias in alias_node.findall("InstructionName")
                if alias.text
            ]
        flags_node = instruction_node.find("InstructionFlags")
        flags = {
            "is_branch": parse_bool(flags_node.findtext("IsBranch") if flags_node is not None else None),
            "is_conditional_branch": parse_bool(
                flags_node.findtext("IsConditionalBranch") if flags_node is not None else None
            ),
            "is_indirect_branch": parse_bool(
                flags_node.findtext("IsIndirectBranch") if flags_node is not None else None
            ),
            "is_program_terminator": parse_bool(
                flags_node.findtext("IsProgramTerminator") if flags_node is not None else None
            ),
            "is_immediately_executed": parse_bool(
                flags_node.findtext("IsImmediatelyExecuted") if flags_node is not None else None
            ),
        }
        encodings = []
        instruction_encodings = instruction_node.find("InstructionEncodings")
        if instruction_encodings is not None:
            for encoding_node in instruction_encodings.findall("InstructionEncoding"):
                encoding_name = encoding_node.findtext("EncodingName", default="").strip()
                encoding_condition = encoding_node.findtext("EncodingCondition", default="default").strip()
                encodings.append(
                    {
                        "encoding_name": encoding_name,
                        "encoding_condition": encoding_condition,
                        "opcode": parse_opcode(encoding_node.find("Opcode")),
                        "operand_count": len(encoding_node.findall("./Operands/Operand")),
                    }
                )
                encoding_families[encoding_name] = encoding_families.get(encoding_name, 0) + 1
        instructions.append(
            {
                "instruction_name": instruction_name,
                "aliases": aliases,
                "flags": flags,
                "encodings": encodings,
            }
        )

    metadata["instruction_count"] = len(instructions)
    metadata["encoding_family_count"] = len(encoding_families)
    metadata["encoding_families"] = sorted(encoding_families.items())
    return {"metadata": metadata, "instructions": instructions}


SYMBOL_RE = re.compile(r"^\s*defm?\s+([^\s:]+)")
INSTRUCTION_PREFIXES = ("V_", "S_", "TENSOR_")


def normalize_llvm_symbol(symbol: str) -> str:
    normalized = symbol
    for suffix in (
        "_gfx1250_fake16_e64",
        "_gfx1250_t16_e64",
        "_gfx1250_fake16",
        "_gfx1250_t16",
        "_gfx1250_e64",
        "_gfx1250",
        "_fake16_e64",
        "_t16_e64",
        "_fake16",
        "_t16",
        "_e64",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def classify_symbol(symbol: str, file_name: str) -> list[str]:
    categories = []
    if "VOP3P" in file_name or symbol.startswith("V_PK_"):
        categories.append("vop3p")
    if any(token in symbol for token in ("WMMA", "SWMMAC", "TENSOR_", "LD_SCALE_PAIRED")):
        categories.append("wmma")
    if any(token in symbol for token in ("FP8", "BF8", "FP6", "BF6", "FP4", "F4")):
        categories.append("fp8_bf8")
    if any(token in symbol for token in ("SCALE", "PAIRED")):
        categories.append("scale_paired")
    if "VOP3" in file_name:
        categories.append("vop3")
    return sorted(set(categories))


def build_llvm_inventory(repo_root: Path) -> dict:
    entries = []
    processor_model = None
    feature_markers = []
    for relative_path in LLVM_RELATIVE_FILES:
        path = repo_root / relative_path
        lines = read_text(path).splitlines()
        for line_number, line in enumerate(lines, start=1):
            if "gfx1250" not in line and "FeatureISAVersion12_50" not in line and "FeatureGFX1250Insts" not in line:
                continue

            if 'ProcessorModel<"gfx1250"' in line:
                processor_model = {
                    "file": relative_path,
                    "line": line_number,
                    "text": line.strip(),
                }
                continue

            if "FeatureISAVersion12_50" in line or "FeatureGFX1250Insts" in line:
                feature_markers.append(
                    {
                        "file": relative_path,
                        "line": line_number,
                        "text": line.strip(),
                    }
                )

            match = SYMBOL_RE.match(line)
            if not match:
                continue
            symbol = match.group(1)
            if symbol.startswith("_") or symbol == "NAME":
                continue
            if not symbol.startswith(INSTRUCTION_PREFIXES):
                continue
            entries.append(
                {
                    "file": relative_path,
                    "line": line_number,
                    "symbol": symbol,
                    "normalized_symbol": normalize_llvm_symbol(symbol),
                    "categories": classify_symbol(symbol, Path(relative_path).name),
                    "text": line.strip(),
                }
            )

    unique_symbols = sorted({entry["symbol"] for entry in entries})
    normalized_symbols = sorted({entry["normalized_symbol"] for entry in entries})

    categorized: dict[str, list[str]] = {}
    for entry in entries:
        for category in entry["categories"]:
            categorized.setdefault(category, set()).add(entry["normalized_symbol"])
    categorized_lists = {
        category: sorted(symbols) for category, symbols in sorted(categorized.items())
    }

    return {
        "processor_model": processor_model,
        "feature_markers": feature_markers,
        "entry_count": len(entries),
        "unique_symbol_count": len(unique_symbols),
        "normalized_symbol_count": len(normalized_symbols),
        "entries": entries,
        "unique_symbols": unique_symbols,
        "normalized_symbols": normalized_symbols,
        "categories": categorized_lists,
    }


def build_delta_vs_gfx950(mirage_root: Path, rdna4_inventory: dict, llvm_inventory: dict) -> dict:
    gfx950_catalog = json.loads(
        read_text(mirage_root / "tests" / "data" / "gfx950_instruction_catalog.json")
    )
    gfx950_names = {
        instruction["instruction_name"] for instruction in gfx950_catalog["instructions"]
    }
    rdna4_names = {
        instruction["instruction_name"] for instruction in rdna4_inventory["instructions"]
    }
    llvm_names = set(llvm_inventory["normalized_symbols"])

    shared = sorted(rdna4_names & gfx950_names)
    rdna4_only = sorted(rdna4_names - gfx950_names)
    gfx950_only = sorted(gfx950_names - rdna4_names)
    llvm_target_specific = sorted(llvm_names - rdna4_names)

    category_membership = {}
    for category, symbols in llvm_inventory["categories"].items():
        category_membership[category] = {
            "count": len(symbols),
            "in_rdna4_xml": sorted(set(symbols) & rdna4_names),
            "not_in_rdna4_xml": sorted(set(symbols) - rdna4_names),
        }

    return {
        "gfx950_instruction_count": len(gfx950_names),
        "rdna4_instruction_count": len(rdna4_names),
        "shared_instruction_count": len(shared),
        "rdna4_only_count": len(rdna4_only),
        "gfx950_only_count": len(gfx950_only),
        "llvm_target_specific_count": len(llvm_target_specific),
        "shared_sample": shared[:50],
        "rdna4_only_sample": rdna4_only[:80],
        "gfx950_only_sample": gfx950_only[:80],
        "llvm_target_specific_sample": llvm_target_specific[:80],
        "llvm_category_membership": category_membership,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, manifest: dict, rdna4_inventory: dict, llvm_inventory: dict, delta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# gfx1250 Architecture Ingest",
        "",
        "## Source Resolution",
        "",
        f"- Official machine-readable source: `{manifest['official_source']['xml_name']}`",
        f"- Architecture mapping: `{manifest['target_mapping']['target_name']}` -> `{manifest['official_source']['architecture_name']}`",
        f"- LLVM processor model: `{llvm_inventory['processor_model']['text'] if llvm_inventory['processor_model'] else 'missing'}`",
        "",
        "## Counts",
        "",
        f"- RDNA4 XML instruction count: `{rdna4_inventory['metadata']['instruction_count']}`",
        f"- gfx950 catalog instruction count: `{delta['gfx950_instruction_count']}`",
        f"- Shared instruction names: `{delta['shared_instruction_count']}`",
        f"- RDNA4-only instruction names vs gfx950: `{delta['rdna4_only_count']}`",
        f"- gfx950-only instruction names vs RDNA4: `{delta['gfx950_only_count']}`",
        f"- LLVM gfx1250 normalized symbols: `{llvm_inventory['normalized_symbol_count']}`",
        f"- LLVM gfx1250 symbols not present verbatim in RDNA4 XML: `{delta['llvm_target_specific_count']}`",
        "",
        "## Focus Areas",
        "",
    ]

    for category in ("vop3p", "wmma", "fp8_bf8", "scale_paired"):
        membership = delta["llvm_category_membership"].get(category, {"count": 0, "in_rdna4_xml": [], "not_in_rdna4_xml": []})
        lines.append(f"- `{category}` normalized symbols: `{membership['count']}`")
        if membership["in_rdna4_xml"]:
            lines.append(f"  - In RDNA4 XML sample: `{', '.join(membership['in_rdna4_xml'][:10])}`")
        if membership["not_in_rdna4_xml"]:
            lines.append(
                f"  - LLVM-only naming sample: `{', '.join(membership['not_in_rdna4_xml'][:10])}`"
            )

    lines.extend(
        [
            "",
            "## Provenance Notes",
            "",
            "- GPUOpen publishes the source XML at the architecture level as `AMD RDNA 4`, not as target-tagged `gfx1250` XML.",
            "- `gfx1250` is anchored locally through LLVM `ProcessorModel<\"gfx1250\", ... FeatureISAVersion12_50 ...>` and the associated `gfx1250`-specific defs.",
            "- The high-value simulator deltas visible in LLVM center on VOP3P, WMMA/SWMMAC, FP8/BF8/FP6/BF6/FP4 forms, and scale/paired operations.",
            "",
            "## Recommended Next Slice",
            "",
            "- Generate a reusable RDNA4 catalog in Mirage, then add a gfx12 target selector that overlays LLVM-derived `gfx1250` target deltas on top of the architecture-level RDNA4 inventory.",
            "- Prioritize decoder/catalog plumbing for VOP3P, WMMA/SWMMAC, FP8/BF8 conversion ops, and scale/paired forms before generic long-tail RDNA4 coverage.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_or_load_xml(mirage_root: Path, refresh: bool) -> tuple[bytes, dict]:
    third_party_dir = mirage_root / "third_party" / "amd_gpu_isa" / "gfx1250"
    xml_path = third_party_dir / RDNA4_XML_NAME
    manifest_path = third_party_dir / "source_manifest.json"
    third_party_dir.mkdir(parents=True, exist_ok=True)

    if xml_path.exists() and not refresh:
        xml_bytes = xml_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return xml_bytes, manifest

    with urllib.request.urlopen(GPUOPEN_BUNDLE_URL) as response:
        bundle_bytes = response.read()
        effective_url = response.geturl()
    bundle_sha256 = sha256_bytes(bundle_bytes)
    bundle = zipfile.ZipFile(io.BytesIO(bundle_bytes))
    xml_bytes = bundle.read(RDNA4_XML_NAME)
    xml_sha256 = sha256_bytes(xml_bytes)
    xml_path.write_bytes(xml_bytes)

    inventory = parse_rdna4_inventory(xml_bytes)
    manifest = {
        "official_source": {
            "bundle_request_url": GPUOPEN_BUNDLE_URL,
            "bundle_effective_url": effective_url,
            "bundle_sha256": bundle_sha256,
            "xml_name": RDNA4_XML_NAME,
            "xml_sha256": xml_sha256,
            "architecture_name": inventory["metadata"]["architecture_name"],
            "release_date": inventory["metadata"]["release_date"],
            "schema_version": inventory["metadata"]["schema_version"],
            "license": inventory["metadata"]["license"],
        },
        "target_mapping": {
            "target_name": "gfx1250",
            "mapping_note": (
                "GPUOpen publishes a single RDNA4 architecture XML; gfx1250 target binding "
                "comes from the local LLVM gfx1250 processor model and feature set."
            ),
        },
    }
    return xml_bytes, manifest


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    mirage_root = script_path.parents[3]
    repo_root = mirage_root.parent

    xml_bytes, manifest = fetch_or_load_xml(mirage_root, args.refresh)
    rdna4_inventory = parse_rdna4_inventory(xml_bytes)
    llvm_inventory = build_llvm_inventory(repo_root)
    delta = build_delta_vs_gfx950(mirage_root, rdna4_inventory, llvm_inventory)

    if "official_source" not in manifest:
        manifest = {
            "official_source": {
                "bundle_request_url": GPUOPEN_BUNDLE_URL,
                "bundle_effective_url": None,
                "bundle_sha256": None,
                "xml_name": RDNA4_XML_NAME,
                "xml_sha256": sha256_bytes(xml_bytes),
                "architecture_name": rdna4_inventory["metadata"]["architecture_name"],
                "release_date": rdna4_inventory["metadata"]["release_date"],
                "schema_version": rdna4_inventory["metadata"]["schema_version"],
                "license": rdna4_inventory["metadata"]["license"],
            },
            "target_mapping": {
                "target_name": "gfx1250",
                "mapping_note": (
                    "GPUOpen publishes a single RDNA4 architecture XML; gfx1250 target binding "
                    "comes from the local LLVM gfx1250 processor model and feature set."
                ),
            },
        }

    manifest["llvm_mapping"] = {
        "processor_model": llvm_inventory["processor_model"],
        "feature_markers": llvm_inventory["feature_markers"],
    }
    manifest["generated_artifacts"] = {
        "rdna4_inventory": "tests/data/architecture_import/gfx1250/rdna4_instruction_inventory.json",
        "llvm_inventory": "tests/data/architecture_import/gfx1250/gfx1250_llvm_inventory.json",
        "delta_vs_gfx950": "tests/data/architecture_import/gfx1250/gfx1250_delta_vs_gfx950.json",
        "report": "reports/architectures/gfx1250/gfx1250_ingest_report.md",
    }

    third_party_dir = mirage_root / "third_party" / "amd_gpu_isa" / "gfx1250"
    tests_data_dir = mirage_root / "tests" / "data" / "architecture_import" / "gfx1250"
    reports_dir = mirage_root / "reports" / "architectures" / "gfx1250"

    write_json(third_party_dir / "source_manifest.json", manifest)
    write_json(tests_data_dir / "rdna4_instruction_inventory.json", rdna4_inventory)
    write_json(tests_data_dir / "gfx1250_llvm_inventory.json", llvm_inventory)
    write_json(tests_data_dir / "gfx1250_delta_vs_gfx950.json", delta)
    write_report(reports_dir / "gfx1250_ingest_report.md", manifest, rdna4_inventory, llvm_inventory, delta)

    print("Wrote", third_party_dir / "source_manifest.json")
    print("Wrote", third_party_dir / RDNA4_XML_NAME)
    print("Wrote", tests_data_dir / "rdna4_instruction_inventory.json")
    print("Wrote", tests_data_dir / "gfx1250_llvm_inventory.json")
    print("Wrote", tests_data_dir / "gfx1250_delta_vs_gfx950.json")
    print("Wrote", reports_dir / "gfx1250_ingest_report.md")
    print("RDNA4 instructions:", rdna4_inventory["metadata"]["instruction_count"])
    print("gfx1250 LLVM normalized symbols:", llvm_inventory["normalized_symbol_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
