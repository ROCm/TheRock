#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Get the directory where the script is located
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go one level up to PACKAGE_HOME
PACKAGE_HOME="$(dirname "$BUILD_DIR")"

SCRIPT_DIR="$PACKAGE_HOME/script/"

# Search for build.ninja under PACKAGE_HOME
BUILD_NINJA_FILE="$PACKAGE_HOME/build/build.ninja"

if [ -z "$BUILD_NINJA_FILE" ]; then
    echo "Error: build.ninja not found under $PACKAGE_HOME"
    exit 1
fi

python3 "$SCRIPT_DIR/dependency-parser/main.py" parse "$BUILD_NINJA_FILE" --workspace-root "$PACKAGE_HOME"

# Get the directory containing build.ninja
BUILD_DIR=$(dirname "$BUILD_NINJA_FILE")

# Path to enhanced_dependency_mapping.json in the same directory
JSON_FILE="$BUILD_DIR/enhanced_dependency_mapping.json"

# Check if the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: $JSON_FILE not found."
    exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)

# Run the command
python3 "$SCRIPT_DIR/dependency-parser/main.py" select "$JSON_FILE" origin/develop $branch

# Path to tests_to_run.json in the same directory
TEST_FILE="tests_to_run.json"

# Configuration: Adjust these defaults as needed
# Number of tests per ctest command (can be overridden with CTEST_CHUNK_SIZE env var)
DEFAULT_CHUNK_SIZE=10
# Whether to stop on first failure (can be overridden with CTEST_FAIL_FAST env var)
DEFAULT_FAIL_FAST=false

# Split tests into chunks and run multiple ctest commands
# Export variables so Python subprocess can access them
export CHUNK_SIZE=${CTEST_CHUNK_SIZE:-$DEFAULT_CHUNK_SIZE}
export FAIL_FAST=${CTEST_FAIL_FAST:-$DEFAULT_FAIL_FAST}

python3 -c "
import json
import os
import sys
import subprocess

CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '10'))
FAIL_FAST = os.environ.get('FAIL_FAST', 'false').lower() == 'true'

with open('$TEST_FILE', 'r') as f:
    data = json.load(f)
    tests = data.get('tests_to_run', [])
    
    if not tests:
        print('# No tests to run')
        sys.exit(0)
    
    # Extract just the filename after the last '/'
    clean_tests = [os.path.basename(test) for test in tests]
    
    total_tests = len(clean_tests)
    total_chunks = (total_tests + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f'# Total tests to run: {total_tests}')
    print(f'# Running in {total_chunks} chunk(s) of up to {CHUNK_SIZE} tests each')
    print(f'# Fail-fast mode: {FAIL_FAST}')
    print()
    
    failed_chunks = []
    
    # Split into chunks
    for i in range(0, total_tests, CHUNK_SIZE):
        chunk = clean_tests[i:i+CHUNK_SIZE]
        chunk_num = (i // CHUNK_SIZE) + 1
        
        print(f'Running test chunk {chunk_num}/{total_chunks} ({len(chunk)} tests)...')
        sys.stdout.flush()
        
        # Run ctest command, don't raise exception on failure
        cmd = ['ctest', '--output-on-failure', '-R', '|'.join(chunk)]
        try:
            result = subprocess.run(cmd, cwd='$BUILD_DIR', check=False)
            
            if result.returncode != 0:
                failed_chunks.append(chunk_num)
                print(f'WARNING: Chunk {chunk_num} had test failures (exit code: {result.returncode})')
                
                # If fail-fast is enabled, exit immediately
                if FAIL_FAST:
                    print(f'FAIL-FAST: Stopping at chunk {chunk_num} due to failures')
                    sys.exit(1)
        except Exception as e:
            print(f'ERROR: Failed to run chunk {chunk_num}: {e}')
            failed_chunks.append(chunk_num)
            if FAIL_FAST:
                sys.exit(1)
        
        print()
        sys.stdout.flush()
    
    # Print summary
    print('=' * 60)
    if failed_chunks:
        print(f'SUMMARY: {len(failed_chunks)} of {total_chunks} chunk(s) had failures: {failed_chunks}')
        print('=' * 60)
        sys.exit(1)
    else:
        print(f'SUMMARY: All {total_chunks} chunk(s) passed successfully!')
        print('=' * 60)
        sys.exit(0)
" 
PYTHON_EXIT=$?

exit $PYTHON_EXIT
