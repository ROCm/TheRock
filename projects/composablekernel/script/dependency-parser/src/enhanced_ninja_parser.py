#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Enhanced Ninja Dependency Parser

This script combines ninja build file parsing with ninja -t deps to create a comprehensive
mapping that includes both source files AND header files, and properly handles files
used by multiple executables.
"""

import re
import os
import sys
import subprocess
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class EnhancedNinjaDependencyParser:
    def __init__(self, build_file_path, ninja_executable="ninja"):
        self.build_file_path = build_file_path
        self.build_dir = os.path.dirname(build_file_path)
        self.ninja_executable = ninja_executable

        # Core data structures
        self.executable_to_objects = {}  # exe -> [object_files]
        self.object_to_source = {}  # object -> primary_source
        self.object_to_all_deps = {}  # object -> [all_dependencies]
        self.file_to_executables = defaultdict(set)  # file -> {executables}

        # Thread safety
        self.lock = threading.Lock()

    def parse_dependencies(self):
        """Main method to parse all dependencies."""
        print(f"Parsing ninja dependencies from: {self.build_file_path}")

        # Step 1: Parse build file for executable -> object mappings
        self._parse_build_file()

        # Step 2: Get all object files and their dependencies
        print(f"Found {len(self.object_to_source)} object files")
        print("Extracting detailed dependencies for all object files...")
        self._extract_object_dependencies()

        # Step 3: Build the final file -> executables mapping
        self._build_file_to_executable_mapping()

    def _parse_build_file(self):
        """Parse the ninja build file to extract executable -> object mappings."""
        print("Parsing ninja build file...")

        with open(self.build_file_path, "r") as f:
            content = f.read()
        # Parse executable build rules
        exe_pattern = r"^build (bin/[^:]+):\s+\S+\s+([^|]+)"
        obj_pattern = r"^build ([^:]+\.(?:cpp|cu|hip)\.o):\s+\S+\s+([^\s|]+)"

        lines = content.split("\n")

        for line in lines:
            # Match executable rules
            exe_match = re.match(exe_pattern, line)
            if exe_match and (
                "EXECUTABLE" in line
                or "test_" in exe_match.group(1)
                or "example_" in exe_match.group(1)
            ):
                exe = exe_match.group(1)
                deps_part = exe_match.group(2).strip()

                object_files = []
                for dep in deps_part.split():
                    if dep.endswith(".o") and not dep.startswith("/"):
                        object_files.append(dep)

                self.executable_to_objects[exe] = object_files
                continue

            # Match object compilation rules
            obj_match = re.match(obj_pattern, line)
            if obj_match:
                object_file = obj_match.group(1)
                source_file = obj_match.group(2)
                self.object_to_source[object_file] = source_file

        print(f"Found {len(self.executable_to_objects)} executables")
        print(f"Found {len(self.object_to_source)} object-to-source mappings")

    def _extract_object_dependencies(self):
        """Extract detailed dependencies for all object files using ninja -t deps."""
        object_files = list(self.object_to_source.keys())
        # Process object files in parallel for better performance
        if not object_files:
            print("No object files found - skipping dependency extraction")
            return

        max_workers = min(128, len(object_files))  # Limit concurrent processes

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all object files for processing
            future_to_obj = {
                executor.submit(self._get_object_dependencies, obj): obj
                for obj in object_files
            }
            # Process completed futures
            completed = 0
            for future in as_completed(future_to_obj):
                obj_file = future_to_obj[future]
                try:
                    dependencies = future.result()
                    with self.lock:
                        self.object_to_all_deps[obj_file] = dependencies
                        completed += 1
                        if completed % 100 == 0:
                            print(
                                f"Processed {completed}/{len(object_files)} object files..."
                            )
                except Exception as e:
                    print(f"Error processing {obj_file}: {e}")

        print(
            f"Completed dependency extraction for {len(self.object_to_all_deps)} object files"
        )

    def _get_object_dependencies(self, object_file):
        """Get all dependencies for a single object file using ninja -t deps."""
        try:
            # Run ninja -t deps for this object file
            cmd = [self.ninja_executable, "-t", "deps", object_file]
            result = subprocess.run(
                cmd, cwd=self.build_dir, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return []

            dependencies = []
            lines = result.stdout.strip().split("\n")

            for line in lines[1:]:  # Skip first line with metadata
                line = line.strip()
                if line and not line.startswith("#"):
                    # Convert absolute paths to relative paths from workspace root
                    dep_file = line
                    ws_root = getattr(self, "workspace_root", "..")
                    ws_prefix = ws_root.rstrip("/") + "/"
                    if dep_file.startswith(ws_prefix):
                        dep_file = dep_file[len(ws_prefix) :]
                    dependencies.append(dep_file)

            return dependencies

        except Exception as e:
            print(f"Error getting dependencies for {object_file}: {e}")
            return []

    def _build_file_to_executable_mapping(self):
        """Build the final mapping from files to executables."""
        print("Building file-to-executable mapping...")

        for exe, object_files in self.executable_to_objects.items():
            for obj_file in object_files:
                # Add all dependencies of this object file
                if obj_file in self.object_to_all_deps:
                    for dep_file in self.object_to_all_deps[obj_file]:
                        # Filter out system files and focus on project files
                        if self._is_project_file(dep_file):
                            self.file_to_executables[dep_file].add(exe)

        print(f"Built mapping for {len(self.file_to_executables)} files")

        # Show statistics
        multi_exe_files = {
            f: exes for f, exes in self.file_to_executables.items() if len(exes) > 1
        }
        print(f"Files used by multiple executables: {len(multi_exe_files)}")

        if multi_exe_files:
            print("Sample files with multiple dependencies:")
            for f, exes in sorted(multi_exe_files.items())[:5]:
                print(f"  {f}: {len(exes)} executables")

    def _is_project_file(self, file_path):
        """Determine if a file is part of the project (not system files)."""
        # Include files that are clearly part of the project
        if any(
            file_path.startswith(prefix)
            for prefix in [
                "include/",
                "library/",
                "test/",
                "example/",
                "src/",
                "profiler/",
                "build/include/",
                "build/_deps/gtest",
                "client_example",
                "codegen",
                "tile_engine",
            ]
        ):
            return True

        # Exclude system files
        if any(
            file_path.startswith(prefix)
            for prefix in ["/usr/", "/opt/rocm", "/lib/", "/system/", "/local/"]
        ):
            return False

        # Include files with common source/header extensions
        if file_path.endswith(
            (".cpp", ".hpp", ".h", ".c", ".cc", ".cxx", ".cu", ".hip", ".inc")
        ):
            return True

        return False

    def export_to_csv(self, output_file):
        """Export the file-to-executable mapping to CSV with proper comma separation."""
        print(f"Exporting mapping to {output_file}")

        with open(output_file, "w") as f:
            f.write("source_file,executables\n")
            for file_path in sorted(self.file_to_executables.keys()):
                executables = sorted(self.file_to_executables[file_path])
                # Use semicolon to separate multiple executables within the field
                exe_list = ";".join(executables)
                f.write(f'"{file_path}","{exe_list}"\n')

    def export_to_json(self, output_file):
        """Export the complete mapping to JSON."""
        print(f"Exporting complete mapping to {output_file}")

        # Build reverse mapping (executable -> files)
        exe_to_files = defaultdict(set)
        for file_path, exes in self.file_to_executables.items():
            for exe in exes:
                exe_to_files[exe].add(file_path)

        mapping_data = {
            "file_to_executables": {
                file_path: list(exes)
                for file_path, exes in self.file_to_executables.items()
            },
            "executable_to_files": {
                exe: sorted(files) for exe, files in exe_to_files.items()
            },
            "statistics": {
                "total_files": len(self.file_to_executables),
                "total_executables": len(self.executable_to_objects),
                "total_object_files": len(self.object_to_source),
                "files_with_multiple_executables": len(
                    [f for f, exes in self.file_to_executables.items() if len(exes) > 1]
                ),
            },
        }

        with open(output_file, "w") as f:
            json.dump(mapping_data, f, indent=2)

    def print_summary(self):
        """Print a summary of the parsed dependencies."""
        print("\n=== Enhanced Dependency Mapping Summary ===")
        print(f"Total executables: {len(self.executable_to_objects)}")
        print(f"Total files mapped: {len(self.file_to_executables)}")
        print(f"Total object files processed: {len(self.object_to_all_deps)}")

        # Files by type
        cpp_files = sum(
            1 for f in self.file_to_executables.keys() if f.endswith(".cpp")
        )
        hpp_files = sum(
            1 for f in self.file_to_executables.keys() if f.endswith(".hpp")
        )
        h_files = sum(1 for f in self.file_to_executables.keys() if f.endswith(".h"))

        print("\nFile types:")
        print(f"  .cpp files: {cpp_files}")
        print(f"  .hpp files: {hpp_files}")
        print(f"  .h files: {h_files}")

        # Multi-executable files
        multi_exe_files = {
            f: exes for f, exes in self.file_to_executables.items() if len(exes) > 1
        }
        print(f"\nFiles used by multiple executables: {len(multi_exe_files)}")

        if multi_exe_files:
            print("\nTop files with most dependencies:")
            sorted_multi = sorted(
                multi_exe_files.items(), key=lambda x: len(x[1]), reverse=True
            )
            for file_path, exes in sorted_multi[:10]:
                print(f"  {file_path}: {len(exes)} executables")


def main():
    # Accept: build_file, ninja_path, workspace_root
    default_workspace_root = ".."
    if len(sys.argv) > 3:
        build_file = sys.argv[1]
        ninja_path = sys.argv[2]
        workspace_root = sys.argv[3]
    elif len(sys.argv) > 2:
        build_file = sys.argv[1]
        ninja_path = sys.argv[2]
        workspace_root = default_workspace_root
    elif len(sys.argv) > 1:
        build_file = sys.argv[1]
        ninja_path = "ninja"
        workspace_root = default_workspace_root
    else:
        build_file = f"{default_workspace_root}/build/build.ninja"
        ninja_path = "ninja"
        workspace_root = default_workspace_root

    if not os.path.exists(build_file):
        print(f"Error: Build file not found: {build_file}")
        sys.exit(1)

    try:
        subprocess.run([ninja_path, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Error: ninja executable not found: {ninja_path}")
        sys.exit(1)

    parser = EnhancedNinjaDependencyParser(build_file, ninja_path)
    parser.workspace_root = workspace_root  # Attach for use in _get_object_dependencies
    parser.parse_dependencies()
    parser.print_summary()

    # Export results
    output_dir = os.path.dirname(build_file)
    csv_file = os.path.join(output_dir, "enhanced_file_executable_mapping.csv")
    json_file = os.path.join(output_dir, "enhanced_dependency_mapping.json")

    parser.export_to_csv(csv_file)
    parser.export_to_json(json_file)

    print("\nResults exported to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")


if __name__ == "__main__":
    main()
