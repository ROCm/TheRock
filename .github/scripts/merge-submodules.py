import os
import configparser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # Assuming script is in .github/scripts/
OUTPUT_FILE = ROOT_DIR / ".gitmodules"
MODULE_FILES = list(ROOT_DIR.glob("*/.gitmodules")) + list(
    ROOT_DIR.glob("*/.github/.gitmodules")
)

combined = configparser.ConfigParser()
combined.optionxform = str  # Preserve case sensitivity

for module_file in MODULE_FILES:
    subdir = module_file.parent.name
    local_config = configparser.ConfigParser()
    local_config.optionxform = str
    local_config.read(module_file)

    for section in local_config.sections():
        if section.startswith("submodule "):
            name = section.split('"')[1]
            new_name = f"{subdir}/{name}"
            new_section = f'submodule "{new_name}"'

            combined[new_section] = {}
            for key, value in local_config[section].items():
                if key == "path":
                    value = f"{subdir}/{value}"
                combined[new_section][key] = value

# Write combined .gitmodules
with OUTPUT_FILE.open("w") as f:
    for section in combined.sections():
        f.write(f"[{section}]\n")
        for key, value in combined[section].items():
            f.write(f"\t{key} = {value}\n")
        f.write("\n")
