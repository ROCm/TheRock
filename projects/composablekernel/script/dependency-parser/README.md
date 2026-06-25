# Dependency-based Selective Test Filtering using Static Analysis of Ninja Builds for C++ Projects

## Overview

This tool provides advanced dependency-based selective test filtering and build optimization for large C++ monorepos using static parsing of Ninja build files. By analyzing both source and header dependencies, it enables precise identification of which tests and executables are affected by code changes, allowing for efficient CI/CD workflows and faster incremental builds.

The parser:
- Identifies all executables in the Ninja build.
- Maps object files to their source and header dependencies using `ninja -t deps`.
- Constructs a reverse mapping from each file to all dependent executables.
- Handles multi-executable dependencies and supports parallel processing for scalability.
- Exports results in CSV and JSON formats for integration with other tools.

## Features

- **Comprehensive Dependency Tracking**: Captures direct source file dependencies and, critically, all included header files via `ninja -t deps`.
- **Executable to Object Mapping**: Parses the `build.ninja` file to understand how executables are linked from object files.
- **Object to Source/Header Mapping**: Uses `ninja -t deps` for each object file to get a complete list of its dependencies.
- **File to Executable Inversion**: Inverts the dependency graph to map each file to the set of executables that depend on it.
- **Parallel Processing**: Utilizes a `ThreadPoolExecutor` to run `ninja -t deps` commands in parallel, significantly speeding up analysis for projects with many object files.
- **Filtering**: Option to filter out system files and focus on project-specific dependencies.
- **Multiple Output Formats**:
    - **CSV**: `enhanced_file_executable_mapping.csv` - A comma-separated values file where each row lists a file and a semicolon-separated list of executables that depend on it.
    - **JSON**: `enhanced_dependency_mapping.json` - A JSON file representing a dictionary where keys are file paths and values are lists of dependent executables.
- **Robust Error Handling**: Includes error handling for missing files and failed subprocess commands.

## Prerequisites

- **Python 3.7+**
- **Ninja build system**: The `ninja` executable must be in the system's PATH or its path provided as an argument.
- A **Ninja build directory** containing a `build.ninja` file and the compiled object files. The project should have been built at least once.

## Using CMake with Ninja

To use this tool effectively, your C++ project should be configured with CMake to generate Ninja build files and dependency information. Follow these steps:

1. **Configure CMake to use Ninja and generate dependencies:**
    ```bash
    cmake -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release /path/to/your/source
    ```
    - The `-G Ninja` flag tells CMake to generate Ninja build files.
    - `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is optional but useful for other tooling.
    - Ensure your CMakeLists.txt uses `target_include_directories` and proper dependency declarations for accurate results.

2. **Build your project with Ninja:**
    ```bash
    ninja
    ```
    - This step is required to generate all object files and dependency information (`.d` files) that the parser relies on.

3. **Run the dependency parser tool:**
    ```bash
    python main.py parse /path/to/build.ninja --workspace-root /path/to/your/workspace
    ```

**Note:** Always run Ninja to ensure all dependencies are up to date before invoking the parser. If you change source files or headers, re-run Ninja first.

## Usage

All features are available via the unified main.py CLI:

```bash
# Dependency parsing (now supports --workspace-root)
python main.py parse examples/build-ninja/build.ninja --workspace-root /path/to/your/workspace

# Selective test filtering
python main.py select enhanced_dependency_mapping.json <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]

# Code auditing
python main.py audit enhanced_dependency_mapping.json

# Build optimization
python main.py optimize enhanced_dependency_mapping.json <changed_file1> [<changed_file2> ...]
```

**Arguments:**

1.  `<path_to_build.ninja>`: (Required) The full path to the `build.ninja` file within your Ninja build directory.
2.  `[--workspace-root <workspace_root>]`: (Optional, recommended) The root directory of your workspace.
3.  `[path_to_ninja_executable]`: (Optional) The path to the `ninja` executable if it's not in your system's PATH. Defaults to `ninja`.

**Example:**

```bash
# Assuming your build directory is 'build-ninja' and it contains 'build.ninja'
python src/enhanced_ninja_parser.py build-ninja/build.ninja

# With custom workspace root
python src/enhanced_ninja_parser.py build-ninja/build.ninja ninja /path/to/your/workspace

# If ninja is installed in a custom location
python src/enhanced_ninja_parser.py /path/to/project/build/build.ninja /usr/local/bin/ninja
```

## How It Works

1.  **Initialization**:
    *   Takes the path to `build.ninja` and optionally the `ninja` executable.
    *   Sets up internal data structures to store mappings.

2.  **Build File Parsing (`_parse_build_file`)**:
    *   Reads the `build.ninja` file.
    *   Uses regular expressions to identify rules for linking executables (e.g., `build my_exe: link main.o utils.o`) and compiling object files (e.g., `build main.o: cxx ../src/main.cpp`).
    *   Populates `executable_to_objects` (mapping an executable name to a list of its .o files) and `object_to_source` (mapping an object file to its primary source file).

3.  **Object Dependency Extraction (`_extract_all_object_dependencies`)**:
    *   Iterates through all unique object files identified in the previous step.
    *   For each object file, it calls `_get_object_dependencies`.
    *   This process is parallelized using `ThreadPoolExecutor` for efficiency. Each call to `ninja -t deps` runs in a separate thread.

4.  **Individual Object Dependencies (`_get_object_dependencies`)**:
    *   For a given object file (e.g., `main.o`), it runs the command: `ninja -t deps main.o` in the build directory.
    *   This command outputs a list of all files that `main.o` depends on, including its primary source (`main.cpp`) and all headers (`*.h`, `*.hpp`) it includes directly or indirectly.
    *   The output is parsed, cleaned, and returned as a list of file paths.

5.  **Building Final File-to-Executable Mapping (`_build_file_to_executable_mapping`)**:
    *   This is the core inversion step. It iterates through each executable and its associated object files.
    *   For each object file, it looks up the full list of its dependencies (source and headers) obtained in step 3 & 4.
    *   For every dependent file found, it adds the current executable to that file's entry in the `file_to_executables` dictionary.
    *   If `filter_project_files` is enabled, it checks each dependency against a list of common system paths (e.g., `/usr/include`, `_deps/`) and excludes them if they match.

6.  **Filtering (`_is_project_file`)**:
    *   A helper function to determine if a given file path is likely a project file or a system/external library file. This helps in focusing the dependency map on the user's own codebase.

7.  **Output Generation**:
    *   **`export_to_csv(csv_file)`**: Writes the `file_to_executables` mapping to a CSV file. Each row contains a file path and a semicolon-delimited string of executable names.
    *   **`export_to_json(json_file)`**: Dumps the `file_to_executables` mapping (where the set of executables is converted to a list) into a JSON file.
    *   **`print_summary()`**: Prints a summary of the findings, including the number of executables, object files, source files, and header files mapped.

## Output Files

Running the script will generate two files in the same directory as the input `build.ninja` file:

-   **`enhanced_file_executable_mapping.csv`**:
    ```csv
    File,Executables
    /path/to/project/src/main.cpp,my_exe_1;my_exe_2
    /path/to/project/include/utils.h,my_exe_1;another_test
    ...
    ```

-   **`enhanced_dependency_mapping.json`**:
    ```json
    {
      "/path/to/project/src/main.cpp": ["my_exe_1", "my_exe_2"],
      "/path/to/project/include/utils.h": ["my_exe_1", "another_test"],
      ...
    }
    ```

## Use Cases

-   **Impact Analysis**: Determine which executables (especially tests) need to be rebuilt or re-run when a specific source or header file changes.
-   **Build Optimization**: Understand the dependency structure to potentially optimize build times.
-   **Code Auditing**: Get a clear overview of how files are used across different executables.
-   **Selective Testing**: Integrate with CI/CD systems to run only the tests affected by a given set of changes.

## Limitations

-   Relies on the accuracy of Ninja's dependency information (`ninja -t deps`). If the build system doesn't correctly generate `.d` (dependency) files, the header information might be incomplete.
-   The definition of "project file" vs. "system file" is based on a simple path-based heuristic and might need adjustment for specific project structures.
-   Performance for extremely large projects (tens of thousands of object files) might still be a consideration, though parallelization helps significantly.

## Troubleshooting

-   **"ninja: command not found"**: Ensure `ninja` is installed and in your PATH, or provide the full path to the executable as the second argument.
-   **"build.ninja not found"**: Double-check the path to your `build.ninja` file.
-   **Empty or Incomplete Output**:
    *   Make sure the project has been successfully built at least once. `ninja -t deps` relies on information generated during the build.
    *   Verify that your CMake (or other meta-build system) is configured to generate dependency files for Ninja.
-   **Slow Performance**: For very large projects, the number of `ninja -t deps` calls can be substantial. While parallelized, it can still take time. Consider if all object files truly need to be analyzed or if a subset is sufficient for your needs.

This tool provides a powerful way to gain deep insights into your Ninja project's dependency structure, enabling more intelligent build and test workflows.
