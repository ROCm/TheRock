#!/usr/bin/env python3

from __future__ import annotations

import json
import textwrap
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = ROOT / "third_party" / "amd_gpu_isa" / "amdgpu_isa_cdna4.xml"
GENERATED_CC = ROOT / "native" / "generated" / "gfx950_instruction_catalog.cc"
GENERATED_JSON = ROOT / "tests" / "data" / "gfx950_instruction_catalog.json"


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


def _cxx_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def load_records() -> tuple[dict[str, str], list[InstructionRecord]]:
    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()
    isa = root.find("ISA")
    if isa is None:
        raise ValueError("missing ISA node")

    document = root.find("Document")
    metadata = {
        "gfx_target": "gfx950",
        "architecture_name": _text(isa.find("Architecture"), "ArchitectureName"),
        "release_date": _text(document, "ReleaseDate"),
        "schema_version": _text(document, "SchemaVersion"),
        "source_xml": SOURCE_XML.name,
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
                    "is_conditional_branch": _bool_text(
                        flags_node, "IsConditionalBranch"
                    ),
                    "is_indirect_branch": _bool_text(flags_node, "IsIndirectBranch"),
                    "is_program_terminator": _bool_text(
                        flags_node, "IsProgramTerminator"
                    ),
                    "is_immediately_executed": _bool_text(
                        flags_node, "IsImmediatelyExecuted"
                    ),
                },
                encodings=tuple(encodings),
            )
        )

    return metadata, records


def _render_metadata_cpp(metadata: dict[str, str], records: list[InstructionRecord]) -> str:
    encoding_count = sum(len(record.encodings) for record in records)
    return textwrap.dedent(
        f"""\
        const InstructionCatalogMetadata kMetadata{{
            {_cxx_string(metadata["gfx_target"])},
            {_cxx_string(metadata["architecture_name"])},
            {_cxx_string(metadata["release_date"])},
            {_cxx_string(metadata["schema_version"])},
            {_cxx_string(metadata["source_xml"])},
            {len(records)}u,
            {encoding_count}u,
        }};
        """
    )


def _iter_encoding_rows(records: Iterable[InstructionRecord]) -> Iterable[str]:
    for record in records:
        for encoding in record.encodings:
            yield (
                "    {"  # noqa: ISC003
                f"{_cxx_string(encoding.encoding_name)}, "
                f"{_cxx_string(encoding.encoding_condition)}, "
                f"{encoding.opcode}u, "
                f"{encoding.operand_count}u"
                "},"
            )


def _iter_instruction_rows(records: list[InstructionRecord]) -> Iterable[str]:
    encoding_offset = 0
    for record in records:
        encoding_count = len(record.encodings)
        yield (
            "    {"  # noqa: ISC003
            f"{_cxx_string(record.instruction_name)}, "
            "{"
            f"{str(record.flags['is_branch']).lower()}, "
            f"{str(record.flags['is_conditional_branch']).lower()}, "
            f"{str(record.flags['is_indirect_branch']).lower()}, "
            f"{str(record.flags['is_program_terminator']).lower()}, "
            f"{str(record.flags['is_immediately_executed']).lower()}"
            "}, "
            f"{encoding_offset}u, "
            f"{encoding_count}u"
            "},"
        )
        encoding_offset += encoding_count


def render_cpp(metadata: dict[str, str], records: list[InstructionRecord]) -> str:
    encoding_rows = "\n".join(_iter_encoding_rows(records))
    instruction_rows = "\n".join(_iter_instruction_rows(records))
    metadata_cpp = _render_metadata_cpp(metadata, records)
    return textwrap.dedent(
        f"""\
        #include "lib/sim/isa/instruction_catalog.h"

        #include <array>

        namespace mirage::sim::isa {{
        namespace {{

        {metadata_cpp}
        constexpr std::array<InstructionEncodingSpec, {sum(len(r.encodings) for r in records)}> kEncodingSpecs{{{{
        {encoding_rows}
        }}}};

        constexpr std::array<InstructionSpec, {len(records)}> kInstructionSpecs{{{{
        {instruction_rows}
        }}}};

        }}  // namespace

        const InstructionCatalogMetadata& GetGfx950InstructionCatalogMetadata() {{
          return kMetadata;
        }}

        std::span<const InstructionSpec> GetGfx950InstructionSpecs() {{
          return std::span<const InstructionSpec>(kInstructionSpecs.data(),
                                                  kInstructionSpecs.size());
        }}

        std::span<const InstructionEncodingSpec> GetGfx950InstructionEncodingSpecs() {{
          return std::span<const InstructionEncodingSpec>(kEncodingSpecs.data(),
                                                          kEncodingSpecs.size());
        }}

        const InstructionSpec* FindGfx950Instruction(std::string_view instruction_name) {{
          for (const InstructionSpec& instruction : kInstructionSpecs) {{
            if (instruction.instruction_name == instruction_name) {{
              return &instruction;
            }}
          }}
          return nullptr;
        }}

        std::span<const InstructionEncodingSpec> GetEncodings(
            const InstructionSpec& instruction) {{
          return std::span<const InstructionEncodingSpec>(
              kEncodingSpecs.data() + instruction.encoding_begin,
              instruction.encoding_count);
        }}

        }}  // namespace mirage::sim::isa
        """
    )


def render_json(metadata: dict[str, str], records: list[InstructionRecord]) -> str:
    payload = {
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
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> None:
    metadata, records = load_records()
    GENERATED_CC.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_JSON.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_CC.write_text(render_cpp(metadata, records), encoding="utf-8")
    GENERATED_JSON.write_text(render_json(metadata, records), encoding="utf-8")


if __name__ == "__main__":
    main()
