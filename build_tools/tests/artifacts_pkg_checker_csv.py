#!/usr/bin/env python3
import sys
import json
import glob
import toml
import csv
from pathlib import Path
from collections import defaultdict


def detailed_component_analysis_csv(output_file="artifact_package_coverage.csv"):
    # Load package.json
    with open("build_tools/packaging/linux/package.json") as f:
        package_data = json.load(f)

    # Build comprehensive mapping
    # Use lowercase keys for case-insensitive matching
    packages_map = defaultdict(lambda: defaultdict(list))
    # Keep original names for display
    original_names = {}

    all_packages = []
    for item in package_data:
        if "Artifactory" in item:
            all_packages.extend(item["Artifactory"])

    for pkg in all_packages:
        artifact = pkg.get("Artifact")

        # Check if there's Artifact_Subdir (nested structure)
        if "Artifact_Subdir" in pkg:
            for subdir in pkg["Artifact_Subdir"]:
                # Use the Name from Artifact_Subdir as the key for matching
                subdir_name = subdir.get("Name")
                components = subdir.get("Components", ["run"])

                if isinstance(components, str):
                    components = [components]

                # Filter out "dbg" component
                components = [c for c in components if c != "dbg"]

                # Use lowercase for matching, but store original name
                key = subdir_name.lower()
                original_names[key] = subdir_name

                # Map using lowercase subdir_name for case-insensitive matching
                for component in components:
                    packages_map[key][component].append(subdir_name)
        else:
            # Old structure without Artifact_Subdir
            components = pkg.get("Components", ["run"])
            if isinstance(components, str):
                components = [components]

            # Filter out "dbg" component
            components = [c for c in components if c != "dbg"]

            pkg_name = pkg.get("Name", artifact)
            key = artifact.lower() if artifact else pkg_name.lower()
            original_names[key] = artifact or pkg_name

            for component in components:
                packages_map[key][component].append(pkg_name)

        # Also map the Artifact field itself (e.g., "host-blas")
        if artifact:
            artifact_key = artifact.lower()
            if artifact_key not in original_names:
                original_names[artifact_key] = artifact

            # If Artifact_Subdir exists, map artifact to same components
            if "Artifact_Subdir" in pkg:
                for subdir in pkg["Artifact_Subdir"]:
                    components = subdir.get("Components", ["run"])
                    if isinstance(components, str):
                        components = [components]

                    # Filter out "dbg" component
                    components = [c for c in components if c != "dbg"]

                    subdir_name = subdir.get("Name")
                    for component in components:
                        if subdir_name not in packages_map[artifact_key][component]:
                            packages_map[artifact_key][component].append(subdir_name)

    # Scan TOML files
    artifact_files = glob.glob("**/artifact-*.toml", recursive=True)

    # Get list of artifacts actually in the artifacts directory
    artifacts_in_directory = set()
    if Path("artifacts").exists():
        for item in Path("artifacts").iterdir():
            if item.is_dir():
                artifacts_in_directory.add(item.name.lower())

    # Prepare CSV data
    csv_rows = []

    for artifact_file in artifact_files:
        # First try: use filename
        artifact_name_from_file = Path(artifact_file).stem.replace("artifact-", "")

        try:
            # Open as text file
            with open(artifact_file, "r", encoding="utf-8") as f:
                artifact_config = toml.load(f)

            components = artifact_config.get("components", {})

            # Skip if no components defined
            if not components:
                continue

            # Try to find artifact names from the TOML content
            # Look for artifact names in the component paths
            artifact_names_in_toml = set()
            for component_type, component_data in components.items():
                # Skip "dbg" component
                if component_type == "dbg":
                    continue

                if isinstance(component_data, dict):
                    for path_key in component_data.keys():
                        # Extract artifact name from paths like "host-blas/stage" or "core/rocrtst/stage"
                        parts = path_key.split("/")
                        if parts:
                            # Could be "host-blas" or "core/rocrtst" - take the last meaningful part
                            if len(parts) >= 2:
                                artifact_names_in_toml.add(
                                    parts[-2]
                                )  # e.g., "rocrtst" from "core/rocrtst/stage"
                            artifact_names_in_toml.add(
                                parts[0]
                            )  # e.g., "host-blas" from "host-blas/stage"

            # Try filename first, then names found in TOML
            artifact_candidates = [artifact_name_from_file] + list(
                artifact_names_in_toml
            )

            # Find which candidate has package coverage
            best_match = None
            best_match_coverage = 0

            for candidate in artifact_candidates:
                candidate_key = candidate.lower()
                if candidate_key in packages_map:
                    coverage = len(packages_map[candidate_key])
                    if coverage > best_match_coverage:
                        best_match = candidate
                        best_match_coverage = coverage

            # Use best match or fall back to filename
            artifact_name = best_match if best_match else artifact_name_from_file
            artifact_key = artifact_name.lower()

            # Check each component type (skip "dbg")
            for component_type in components.keys():
                # Skip "dbg" component
                if component_type == "dbg":
                    continue

                has_package = component_type in packages_map.get(artifact_key, {})
                package_names = packages_map.get(artifact_key, {}).get(
                    component_type, []
                )

                # Determine status: check if artifact is in directory
                if artifact_key in artifacts_in_directory:
                    status = "COVERED" if has_package else "MISSING"
                else:
                    status = "NOT IN ARTIFACTS" if not has_package else "COVERED"

                csv_rows.append(
                    {
                        "Artifact": artifact_name,
                        "Artifact_From_File": artifact_name_from_file,
                        "Component": component_type,
                        "Status": status,
                        "Package Names": (
                            ", ".join(package_names) if package_names else ""
                        ),
                        "TOML File": artifact_file,
                    }
                )

        except Exception as e:
            csv_rows.append(
                {
                    "Artifact": artifact_name_from_file,
                    "Artifact_From_File": artifact_name_from_file,
                    "Component": "ERROR",
                    "Status": "ERROR",
                    "Package Names": "",
                    "TOML File": artifact_file,
                    "Error": str(e),
                }
            )

    # Write to CSV
    if csv_rows:
        fieldnames = [
            "Artifact",
            "Artifact_From_File",
            "Component",
            "Status",
            "Package Names",
            "TOML File",
        ]
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\n✓ CSV report written to: {output_file}")
        print(f"✓ Total rows processed: {len(csv_rows)}")

        # Count statuses
        status_counts = defaultdict(int)
        for row in csv_rows:
            status_counts[row["Status"]] += 1

        print("\n--- Status Summary ---")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")

        # Print artifacts with "NOT IN ARTIFACTS" status
        not_in_artifacts = [
            row for row in csv_rows if row["Status"] == "NOT IN ARTIFACTS"
        ]
        if not_in_artifacts:
            print("\n--- Artifacts NOT IN ARTIFACTS Directory ---")
            for row in not_in_artifacts:
                print(
                    f"  • {row['Artifact']} ({row['Component']}) - {row['TOML File']}"
                )

        # Print artifacts with "MISSING" status
        missing = [row for row in csv_rows if row["Status"] == "MISSING"]
        if missing:
            print("\n--- Artifacts MISSING from Package Mapping ---")
            for row in missing:
                print(
                    f"  • {row['Artifact']} ({row['Component']}) - {row['TOML File']}"
                )

        # Print artifacts with "ERROR" status
        errors = [row for row in csv_rows if row["Status"] == "ERROR"]
        if errors:
            print("\n--- Processing Errors ---")
            for row in errors:
                error_msg = row.get("Error", "Unknown error")
                print(f"  • {row['TOML File']}: {error_msg}")

        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total Component Entries:     {len(csv_rows)}")
        print(f" ✅ Covered:                 {status_counts.get('COVERED', 0)}")
        print(f" ❌ Missing:                 {status_counts.get('MISSING', 0)}")
        print(
            f" ⚠️ Not In Artifacts:         {status_counts.get('NOT IN ARTIFACTS', 0)}"
        )
        print(f" ⚠️ Errors:                   {status_counts.get('ERROR', 0)}")
        print("=" * 60)
    else:
        print("No artifact TOML files found or no components defined.")


if __name__ == "__main__":
    detailed_component_analysis_csv()
