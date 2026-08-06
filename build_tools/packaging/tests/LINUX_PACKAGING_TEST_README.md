# ROCm Packaging Tests

This directory contains comprehensive unit tests for the ROCm packaging tools with **96% code coverage**.

## Test Files

- `test_linux_packaging_utils.py` - 54 tests for `packaging_utils.py` (99% coverage)
- `test_linux_build_package.py` - 36 tests for `build_package.py` (94% coverage)

## Running Tests

### Run All Tests

To run all tests in the tests directory:

```bash
cd TheRock/build_tools/packaging/tests
python3 -m unittest discover -v
```

### Run Tests for a Specific Module

To run tests for `packaging_utils.py`:

```bash
python3 -m unittest test_linux_packaging_utils -v
```

To run tests for `build_package.py`:

```bash
python3 -m unittest test_linux_build_package -v
```

### Run a Specific Test Class

To test a specific function (e.g., `version_to_str`):

```bash
python3 -m unittest test_linux_packaging_utils.TestVersionToStr -v
```

To test DEB package creation functions:

```bash
python3 -m unittest test_linux_build_package.TestCreateDebPackage -v
```

### Run a Single Test Method

To run just one specific test case:

```bash
python3 -m unittest test_linux_packaging_utils.TestVersionToStr.test_version_three_parts -v
```

## Test Coverage

### packaging_utils.py Tests

The `test_linux_packaging_utils.py` file includes tests for:

- `print_function_name()` - Function name printing
- `read_package_json_file()` - JSON file reading
- `is_key_defined()` - Key validation with various true/false values
- `is_postinstallscripts_available()` - Postinstall script checking
- `is_meta_package()` - Meta package identification
- `is_composite_package()` - Composite package checking
- `is_rpm_stripping_disabled()` - RPM stripping configuration
- `is_debug_package_disabled()` - Debug package configuration
- `is_packaging_disabled()` - Package disabling check
- `is_gfxarch_package()` - Graphics architecture package check
- `get_package_info()` - Package metadata retrieval
- `get_package_list()` - Package list generation
- `remove_dir()` - Directory removal
- `version_to_str()` - Version string conversion
- `update_package_name()` - Package name updates with versions/gfx
- `debian_replace_devel_name()` - Debian package name conversion
- `convert_to_versiondependency()` - Dependency versioning
- `append_version_suffix()` - Version suffix appending
- `move_packages_to_destination()` - Package file movement
- `filter_components_fromartifactory()` - Artifact filtering
- `PackageConfig` dataclass - Configuration object

### build_package.py Tests

The `test_linux_build_package.py` file includes tests for:

**File Operations:**

- `copy_package_contents()` - File/directory copying with symlink support

**Debian Package Generation:**

- `generate_changelog_file()` - Debian changelog generation
- `generate_install_file()` - Debian install file generation
- `generate_rules_file()` - Debian rules file generation
- `generate_control_file()` - Debian control file generation (with Provides/Replaces/Conflicts)
- `generate_debian_postscripts()` - Post-installation script generation
- `create_nonversioned_deb_package()` - Non-versioned DEB meta packages
- `create_versioned_deb_package()` - Versioned DEB packages with artifacts
- `package_with_dpkg_build()` - DEB package building with dpkg-buildpackage

**RPM Package Generation:**

- `generate_spec_file()` - RPM spec file generation
- `generate_rpm_postscripts()` - RPM post-installation script sections
- `create_nonversioned_rpm_package()` - Non-versioned RPM meta packages
- `create_versioned_rpm_package()` - Versioned RPM packages
- `package_with_rpmbuild()` - RPM package building with rpmbuild

**Orchestration Functions:**

- `parse_input_package_list()` - Package list parsing with filtering
- `clean_package_build_dir()` - Build directory cleanup
- `create_deb_package()` - Complete DEB package orchestration
- `create_rpm_package()` - Complete RPM package orchestration
- `run()` - Main execution function (single/both package types)
- `main()` - Argument parsing and entry point

## Test Features

### Mocking

Tests use Python's `unittest.mock` module to:

- Mock file system operations
- Mock subprocess calls (dpkg-buildpackage, rpmbuild)
- Mock external dependencies
- Isolate units under test

### Edge Cases

Tests cover various edge cases including:

- Non-existent files and directories
- Invalid version formats
- Empty package lists
- Disabled packages
- Meta packages (no artifacts)
- Non-meta packages with empty artifacts (error handling)
- Graphics architecture overrides
- Symlink handling
- Process failures (dpkg-buildpackage, rpmbuild)
- RPATH conversion paths
- Provides/Replaces/Conflicts in control files
- Both versioned and non-versioned package creation
- Creating both DEB and RPM when package type not specified

## Debugging Tests

### Verbose Output

Use the `-v` flag for detailed test output:

```bash
python3 -m unittest test_linux_packaging_utils -v
```

### Run Specific Failing Test

If a test fails, run only that test to debug:

```bash
python3 -m unittest test_linux_packaging_utils.TestVersionToStr.test_version_three_parts -v
```

### Check Test Output

The `-v` flag will show:

- Which tests passed (✓)
- Which tests failed (✗)
- Error messages and tracebacks
- Total test count and timing

## Requirements

The tests require:

- Python 3.6+
- Standard library modules: `unittest`, `tempfile`, `pathlib`, `json`, `os`, `shutil`
- The modules being tested: `packaging_utils.py` and `build_package.py`

No additional packages like pytest are required.

## Contributing

When adding new functions to `packaging_utils.py` or `build_package.py`:

1. Create a corresponding test class in the appropriate test file
1. Test both success and failure cases
1. Test edge cases and error conditions
1. Use mocks to avoid external dependencies
1. Run all tests to ensure no regressions

## Example Test Output

```
$ python3 -m unittest test_linux_packaging_utils.TestVersionToStr -v
test_version_four_parts (test_linux_packaging_utils.TestVersionToStr) ... ok
test_version_large_numbers (test_linux_packaging_utils.TestVersionToStr) ... ok
test_version_one_part (test_linux_packaging_utils.TestVersionToStr) ... ok
test_version_three_parts (test_linux_packaging_utils.TestVersionToStr) ... ok
test_version_two_parts (test_linux_packaging_utils.TestVersionToStr) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.002s

OK
```
