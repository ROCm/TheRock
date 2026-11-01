#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest wrapper for rocshmem functional tests
# This script handles conditional test skipping based on backend type
# and other runtime conditions, matching the behavior of driver.sh

# CTest SKIP return code
SKIP_CODE=125

# Extract test name (first argument)
TEST_NAME=$1
shift

# Detect launcher mode: MPI or SLR
# - If ROCSHMEM_SLR_NP is set, we're in SLR mode (direct execution)
# - Otherwise, we're in MPI mode (command contains mpirun/mpiexec)
if [[ -n "${ROCSHMEM_SLR_NP}" ]]; then
    LAUNCHER_MODE="SLR"
else
    LAUNCHER_MODE="MPI"
fi

# Find the test executable path from remaining arguments
# For MPI: The command is: mpirun/mpiexec [mpi_args] <executable> [test_args]
# For SLR: The command is: <executable> [test_args]
# We need to find the rocshmem test executable (not mpirun)
TEST_EXECUTABLE=""
SKIP_NEXT=false
for arg in "$@"; do
    # Skip argument values that follow MPI flags
    if $SKIP_NEXT; then
        SKIP_NEXT=false
        continue
    fi
    # MPI flags that take arguments - skip their values
    if [[ "$arg" == "-n" || "$arg" == "-np" || "$arg" == "-mca" || "$arg" == "-x" || "$arg" == "--timeout" || "$arg" == "--map-by" ]]; then
        SKIP_NEXT=true
        continue
    fi
    # Skip other MPI flags
    if [[ "$arg" == -* ]]; then
        continue
    fi
    # Look for rocshmem test executable (not mpirun/mpiexec)
    if [[ "$arg" == *"rocshmem"* ]] && [[ -x "$arg" ]]; then
        TEST_EXECUTABLE="$arg"
        break
    fi
done

# Get backend type from environment or rocshmem_info
# Strategy:
# 1. If ROCSHMEM_BACKEND is explicitly set, use that value
# 2. Otherwise, check rocshmem_info:
#    a. If exactly ONE backend is compiled, use that backend
#    b. If MULTIPLE backends are compiled, don't set backend (let rocshmem decide)
if [[ -n "$ROCSHMEM_BACKEND" ]]; then
    # User explicitly requested a specific backend
    BACKEND="$ROCSHMEM_BACKEND"
    echo "Using explicitly set backend: $BACKEND"
elif [[ -n "$ROCSHMEM_BACKEND_TYPE" ]]; then
    # Legacy compatibility
    BACKEND="$ROCSHMEM_BACKEND_TYPE"
    echo "Using ROCSHMEM_BACKEND_TYPE: $BACKEND"
elif [[ -n "$TEST" ]]; then
    # For compatibility with driver.sh TEST variable (e.g., "ro", "gda")
    BACKEND="$TEST"
    echo "Using TEST variable: $BACKEND"
else
    # Auto-detect from rocshmem_info
    # Find rocshmem_info (use test executable path)
    if [[ -n "$TEST_EXECUTABLE" ]]; then
        ROCSHMEM_INFO="$(dirname "$TEST_EXECUTABLE")/rocshmem_info"
        if [[ ! -x "$ROCSHMEM_INFO" ]]; then
            # Try alternate location (builddir case)
            ROCSHMEM_INFO="$(dirname "$TEST_EXECUTABLE")/../../tools/rocshmem_info"
        fi
        if [[ -x "$ROCSHMEM_INFO" ]]; then
            # Check which backends are compiled in
            # Format: # USE_RO                      : ON                                             #
            INFO_OUTPUT=$("$ROCSHMEM_INFO")
            USE_RO=$(echo "$INFO_OUTPUT" | grep "USE_RO" | awk -F ':' '{print $2}' | awk '{print $1}')
            USE_IPC=$(echo "$INFO_OUTPUT" | grep "USE_IPC" | awk -F ':' '{print $2}' | awk '{print $1}')
            USE_GDA=$(echo "$INFO_OUTPUT" | grep "USE_GDA" | awk -F ':' '{print $2}' | awk '{print $1}')

            # Count how many backends are enabled
            BACKEND_COUNT=0
            AVAILABLE_BACKEND=""
            if [[ "$USE_RO" == "ON" ]]; then
                BACKEND_COUNT=$((BACKEND_COUNT + 1))
                AVAILABLE_BACKEND="ro"
            fi
            if [[ "$USE_IPC" == "ON" ]]; then
                BACKEND_COUNT=$((BACKEND_COUNT + 1))
                AVAILABLE_BACKEND="ipc"
            fi
            if [[ "$USE_GDA" == "ON" ]]; then
                BACKEND_COUNT=$((BACKEND_COUNT + 1))
                AVAILABLE_BACKEND="gda"
            fi

            if [[ $BACKEND_COUNT -eq 1 ]]; then
                # Exactly one backend compiled - use it and apply skip logic
                BACKEND="$AVAILABLE_BACKEND"
                echo "Single backend detected: $BACKEND (will apply backend-specific skip logic)"
            elif [[ $BACKEND_COUNT -gt 1 ]]; then
                # Multiple backends compiled - let rocshmem decide, don't skip tests
                BACKEND="multi"
                echo "Multiple backends detected (RO=$USE_RO, IPC=$USE_IPC, GDA=$USE_GDA) - letting rocshmem choose, no skip logic"
            else
                # No backends found (shouldn't happen)
                BACKEND="unknown"
                echo "Warning: No backends detected in rocshmem_info"
            fi
        else
            BACKEND="unknown"
            echo "Warning: rocshmem_info not found at $ROCSHMEM_INFO"
        fi
    else
        BACKEND="unknown"
        echo "Warning: Could not find test executable"
    fi
fi

echo "Test: $TEST_NAME (Launcher: $LAUNCHER_MODE, Backend: $BACKEND, Executable: ${TEST_EXECUTABLE:-<not found>})"

# Apply skip conditions based on backend type and known issues
# These match the skip logic from driver.sh
# Note: Skip logic only applies when backend is known (not "multi" or "unknown")

# Host non-MPI IPC tests require IPC backend
if [[ "$BACKEND" == "ro" || "$BACKEND" == "gda" ]]; then
    case "$TEST_NAME" in
        host_putmem_*|host_getmem_*|host_amo_*|host_ctx_*|host_int_*)
            echo "Skip: $TEST_NAME (host non-MPI IPC tests require IPC backend)"
            exit $SKIP_CODE
            ;;
    esac
fi

# Symmetric user-buffer registration is unsupported by the RO backend
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        buffer_register_symmetric_*)
            echo "Skip: $TEST_NAME (RO does not support symmetric buffer registration)"
            exit $SKIP_CODE
            ;;
    esac
fi

# The vmm_posix allocator aborts during rocSHMEM initialization on unsupported
# hardware. Check VMM support first so CTest can report a proper skip instead.
case "$TEST_NAME" in
    buffer_register_symmetric_*)
        if ! command -v rocminfo >/dev/null 2>&1 ||
           ! rocminfo 2>/dev/null |
             awk -F ':' '
               /^[[:space:]]*VMM Support[[:space:]]*:/ {
                 gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                 if (toupper($2) == "YES") found = 1
               }
               END { exit !found }
             '; then
            echo "Skip: $TEST_NAME (HIP VMM is unavailable)"
            exit $SKIP_CODE
        fi

        if [[ "$BACKEND" == "gda" ]]; then
            export ROCSHMEM_DISABLE_MIXED_IPC=1
        fi
        ;;
esac

# AIROCSHMEM-120: RO get tests abort
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        get_*|getnbi_*|defaultctxget_*|defaultctxgetnbi_*|teamctxget_*|teamctxgetnbi_*|wgget_*|wggetnbi_*|waveget_*|wavegetnbi_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-120: RO get tests abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-162: GDA _g not implemented
if [[ "$BACKEND" == "gda" ]]; then
    case "$TEST_NAME" in
        g_*|defaultctxg_*|flood_g_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-162: GDA _g not implemented)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-211: RO AMO operations abort
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        amo_add_*|amo_fadd_*|amo_inc_*|amo_finc_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-211: RO amo abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-217: RO putmem_signal_on_stream sometimes abort
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        putmem_signal_on_stream_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-217: RO sometimes abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-324: RO flood tests fail in UCX
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        flood_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-324: RO flood tests fail in UCX)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-418: fence tests not supported on RO
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        fence_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-418: fence tests not supported on RO)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-418: wave tests not supported on RO
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        *_wave)
            echo "Skip: $TEST_NAME (AIROCSHMEM-409: wave tests not supported on RO)"
            exit $SKIP_CODE
            ;;
    esac
fi

# Check GPU availability
if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null; then
    NUM_GPUS=$(amd-smi list | grep GPU | wc -l)
elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null; then
    NUM_GPUS=$(rocm-smi --showserial | grep GPU | wc -l)
else
    NUM_GPUS=0
fi
NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS : 0))

# Extract number of ranks from test name (format: testname_n<ranks>_w<wg>_z<threads>)
# Or from ROCSHMEM_SLR_NP if in SLR mode
if [[ "$LAUNCHER_MODE" == "SLR" ]]; then
    NUM_RANKS=${ROCSHMEM_SLR_NP}
elif [[ "$TEST_NAME" =~ _n([0-9]+)_ ]]; then
    NUM_RANKS=${BASH_REMATCH[1]}
fi

# Skip if not enough GPUs
if [[ -n "$NUM_RANKS" ]]; then
    # For SLR mode, always check GPU availability (no hostfile support)
    if [[ "$LAUNCHER_MODE" == "SLR" ]] && [[ $NUM_GPUS -lt $NUM_RANKS ]]; then
        echo "Skip: $TEST_NAME (SLR requires $NUM_RANKS GPUs, only $NUM_GPUS available)"
        exit $SKIP_CODE
    fi
    # For MPI mode, skip only if no hostfile and insufficient GPUs
    if [[ "$LAUNCHER_MODE" == "MPI" ]] && [[ -z "$HOSTFILE" ]] && [[ $NUM_GPUS -lt $NUM_RANKS ]]; then
        echo "Skip: $TEST_NAME ($NUM_RANKS ranks required but only $NUM_GPUS GPUs available)"
        exit $SKIP_CODE
    fi
fi

# Setup log directory and file (matching driver.sh behavior)
# Use environment variable LOG_DIR if set, otherwise use current directory
LOG_DIR=${ROCSHMEM_TEST_LOG_DIR:-${LOG_DIR:-.}}
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/$TEST_NAME.log"

# Print command for debugging (matching driver.sh)
echo "# $@" > "$LOG_FILE"

# Execute the actual test command and capture output
"$@" >> "$LOG_FILE" 2>&1
TEST_EXIT_CODE=$?

# If test failed, show the log content (for CTest output)
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Test failed - see log: $LOG_FILE"
    cat "$LOG_FILE"
fi

exit $TEST_EXIT_CODE
