import os
from pathlib import Path

# Determine monorepo root and output CODEOWNERS path
monorepo_root = Path(__file__).resolve().parents[2]
output_path = monorepo_root / ".github" / "CODEOWNERS"

merged_entries = []

# Walk top-level directories (excluding .github/.git/etc.)
for subdir in monorepo_root.iterdir():
    if subdir.name.startswith(".") or not subdir.is_dir():
        continue

    # Look for CODEOWNERS in root or .github directory of the submodule
    candidates = [subdir / "CODEOWNERS", subdir / ".github" / "CODEOWNERS"]

    for codeowners_file in candidates:
        if codeowners_file.is_file():
            with codeowners_file.open("r") as f:
                for line in f:
                    stripped = line.strip()

                    # Skip empty lines or comments
                    if not stripped or stripped.startswith("#"):
                        continue

                    parts = stripped.split()
                    if not parts:
                        continue

                    original_path = parts[0]
                    owners = " ".join(parts[1:])

                    # Ensure prefixed path starts with a single slash
                    prefixed_path = (
                        f"/{subdir.name.rstrip('/')}{original_path}"
                        if original_path.startswith("/")
                        else f"/{subdir.name}/{original_path}"
                    )

                    merged_entries.append(f"{prefixed_path} {owners}")

# Sort for consistency
merged_entries.sort()

# Write merged CODEOWNERS file
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as out:
    out.write("# Auto-generated CODEOWNERS file\n\n")
    out.write("\n".join(merged_entries))

print(f"✅ Merged CODEOWNERS written to {output_path}")
