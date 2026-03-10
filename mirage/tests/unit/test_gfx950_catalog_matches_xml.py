from __future__ import annotations

import json
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_XML = ROOT / "third_party" / "amd_gpu_isa" / "amdgpu_isa_cdna4.xml"
GENERATED_JSON = ROOT / "tests" / "data" / "gfx950_instruction_catalog.json"
GENERATED_CC = ROOT / "native" / "generated" / "gfx950_instruction_catalog.cc"
GENERATOR = ROOT / "tools" / "generate_gfx950_isa_catalog.py"


def _load_xml_catalog() -> tuple[dict[str, object], dict[str, object]]:
    root = ET.parse(SOURCE_XML).getroot()
    document = root.find("Document")
    isa = root.find("ISA")
    assert document is not None
    assert isa is not None

    instructions: dict[str, object] = {}
    instructions_node = isa.find("Instructions")
    assert instructions_node is not None

    encoding_count = 0
    for instruction in instructions_node.findall("Instruction"):
        name = instruction.findtext("InstructionName", default="").strip()
        flags_node = instruction.find("InstructionFlags")
        encodings = []
        encodings_node = instruction.find("InstructionEncodings")
        assert encodings_node is not None
        for encoding in encodings_node.findall("InstructionEncoding"):
            operands = encoding.find("Operands")
            operand_count = 0 if operands is None else len(operands.findall("Operand"))
            encodings.append(
                {
                    "encoding_name": encoding.findtext("EncodingName", default="").strip(),
                    "encoding_condition": encoding.findtext(
                        "EncodingCondition", default="default"
                    ).strip(),
                    "opcode": int(encoding.findtext("Opcode", default="0").strip(), 10),
                    "operand_count": operand_count,
                }
            )
        encoding_count += len(encodings)
        instructions[name] = {
            "instruction_name": name,
            "flags": {
                "is_branch": flags_node.findtext("IsBranch", default="FALSE").strip()
                == "TRUE",
                "is_conditional_branch": flags_node.findtext(
                    "IsConditionalBranch", default="FALSE"
                ).strip()
                == "TRUE",
                "is_indirect_branch": flags_node.findtext(
                    "IsIndirectBranch", default="FALSE"
                ).strip()
                == "TRUE",
                "is_program_terminator": flags_node.findtext(
                    "IsProgramTerminator", default="FALSE"
                ).strip()
                == "TRUE",
                "is_immediately_executed": flags_node.findtext(
                    "IsImmediatelyExecuted", default="FALSE"
                ).strip()
                == "TRUE",
            },
            "encodings": encodings,
        }

    metadata = {
        "gfx_target": "gfx950",
        "architecture_name": isa.find("Architecture").findtext(
            "ArchitectureName", default=""
        ).strip(),
        "release_date": document.findtext("ReleaseDate", default="").strip(),
        "schema_version": document.findtext("SchemaVersion", default="").strip(),
        "source_xml": SOURCE_XML.name,
        "instruction_count": len(instructions),
        "encoding_count": encoding_count,
    }
    return metadata, instructions


def test_gfx950_generated_catalog_matches_vendored_xml():
    xml_metadata, xml_instructions = _load_xml_catalog()
    generated = json.loads(GENERATED_JSON.read_text(encoding="utf-8"))

    assert generated["metadata"] == xml_metadata

    generated_instructions = {
        instruction["instruction_name"]: instruction
        for instruction in generated["instructions"]
    }

    assert set(generated_instructions) == set(xml_instructions)
    for name, expected in xml_instructions.items():
        assert generated_instructions[name] == expected


def test_gfx950_generated_outputs_are_up_to_date():
    spec = importlib.util.spec_from_file_location("generate_gfx950_isa_catalog", GENERATOR)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    metadata, records = module.load_records()
    assert GENERATED_JSON.read_text(encoding="utf-8") == module.render_json(
        metadata, records
    )
    assert GENERATED_CC.read_text(encoding="utf-8") == module.render_cpp(
        metadata, records
    )
