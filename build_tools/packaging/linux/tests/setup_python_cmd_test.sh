#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Test suite for setup_python_cmd.sh
# Run with: bash build_tools/packaging/linux/tests/setup_python_cmd_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="$SCRIPT_DIR/../setup_python_cmd.sh"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Test helper functions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local test_name="$3"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [[ "$expected" == "$actual" ]]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected: $expected"
        echo "  Actual: $actual"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

assert_contains() {
    local substring="$1"
    local string="$2"
    local test_name="$3"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [[ "$string" == *"$substring"* ]]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected to contain: $substring"
        echo "  Actual: $string"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Helper: run script with --output-format github using a temp GITHUB_ENV file.
# Returns the content written to GITHUB_ENV.
run_github_format() {
    local tmp_env
    tmp_env=$(mktemp)
    GITHUB_ENV="$tmp_env" bash "$SETUP_SCRIPT" "$@" --output-format github
    local result
    result=$(cat "$tmp_env")
    rm -f "$tmp_env"
    echo "$result"
}

# Test resolve_python_cmd logic (output format tests)
test_ubuntu_profiles() {
    local output
    output=$(run_github_format --os-profile ubuntu2404)
    assert_equals "PYTHON_CMD=python3.12" "$output" "Ubuntu 24.04 resolves to python3.12"

    output=$(run_github_format --os-profile ubuntu2204)
    assert_equals "PYTHON_CMD=python3.12" "$output" "Ubuntu 22.04 resolves to python3.12"

    output=$(run_github_format --os-profile ubuntu2004)
    assert_equals "PYTHON_CMD=python3.12" "$output" "Ubuntu 20.04 resolves to python3.12"
}

test_debian_profiles() {
    local output
    output=$(run_github_format --os-profile debian12)
    assert_equals "PYTHON_CMD=python3.12" "$output" "Debian 12 resolves to python3.12"

    output=$(run_github_format --os-profile debian11)
    assert_equals "PYTHON_CMD=python3.12" "$output" "Debian 11 resolves to python3.12"
}

test_sles_profiles() {
    local output
    output=$(run_github_format --os-profile sles16)
    assert_equals "PYTHON_CMD=python3.13" "$output" "SLES 16 resolves to python3.13"

    output=$(run_github_format --os-profile sles15)
    assert_equals "PYTHON_CMD=python3.13" "$output" "SLES 15 resolves to python3.13"
}

test_rhel_profiles() {
    local output
    output=$(run_github_format --os-profile rhel10)
    assert_equals "PYTHON_CMD=python3.12" "$output" "RHEL 10 resolves to python3.12"

    output=$(run_github_format --os-profile rhel9)
    assert_equals "PYTHON_CMD=python3.12" "$output" "RHEL 9 resolves to python3.12"
}

test_centos_profiles() {
    local output
    output=$(run_github_format --os-profile centos9)
    assert_equals "PYTHON_CMD=python3.12" "$output" "CentOS 9 resolves to python3.12 (default case)"
}

# Test output formats
test_json_output() {
    local output=$(bash "$SETUP_SCRIPT" --os-profile ubuntu2404 --output-format json)
    assert_equals '{"python_cmd": "python3.12"}' "$output" "JSON output format for Ubuntu"

    output=$(bash "$SETUP_SCRIPT" --os-profile sles16 --output-format json)
    assert_equals '{"python_cmd": "python3.13"}' "$output" "JSON output format for SLES"
}

test_github_output() {
    local output
    output=$(run_github_format --os-profile rhel10)
    assert_equals "PYTHON_CMD=python3.12" "$output" "GitHub output format writes to GITHUB_ENV"
}

test_env_output() {
    local output=$(bash "$SETUP_SCRIPT" --os-profile ubuntu2404 --output-format env)
    assert_equals "export PYTHON_CMD=python3.12" "$output" "Env output format for Ubuntu"

    output=$(bash "$SETUP_SCRIPT" --os-profile sles16 --output-format env)
    assert_equals "export PYTHON_CMD=python3.13" "$output" "Env output format for SLES"
}

# Test error handling
test_missing_os_profile() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if bash "$SETUP_SCRIPT" --output-format github 2>/dev/null; then
        echo -e "${RED}✗${NC} Missing --os-profile should fail"
        echo "  Expected: error exit code"
        echo "  Actual: success"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    else
        echo -e "${GREEN}✓${NC} Missing --os-profile should fail"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
}

test_invalid_output_format() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if bash "$SETUP_SCRIPT" --os-profile ubuntu2404 --output-format invalid 2>/dev/null; then
        echo -e "${RED}✗${NC} Invalid --output-format should fail"
        echo "  Expected: error exit code"
        echo "  Actual: success"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    else
        echo -e "${GREEN}✓${NC} Invalid --output-format should fail"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
}

# Run all tests
echo "Running setup_python_cmd.sh tests..."
echo ""

test_ubuntu_profiles
test_debian_profiles
test_sles_profiles
test_rhel_profiles
test_centos_profiles
test_json_output
test_github_output
test_env_output
test_missing_os_profile
test_invalid_output_format

# Print summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Tests run: $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    exit 1
else
    echo -e "Tests failed: $TESTS_FAILED"
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
