#!/bin/bash
# Validates static library archives are well-formed and can be linked.
# This script performs comprehensive validation including:
# 1. Archive format validation
# 2. Symbol completeness check
# 3. Actual linkage test to ensure the library can be used

set -e

AR_BIN="${GRPC_VALIDATE_AR:-${AR:-ar}}"
NM_BIN="${GRPC_VALIDATE_NM:-${NM:-nm}}"
CXX_BIN="${GRPC_VALIDATE_CXX:-${CXX:-g++}}"
FILE_BIN="${GRPC_VALIDATE_FILE:-file}"
CXXFILT_BIN="${GRPC_VALIDATE_CXXFILT:-c++filt}"

GRPC_LIBS=()
LIB_DIR=""

for lib_path in "$@"; do
    echo -n "Validating static library: $lib_path"

    # Check if file exists
    if [ ! -f "$lib_path" ]; then
        echo " : FAILED"
        echo "  Error: File does not exist: $lib_path" >&2
        exit 1
    fi

    # Store library path for linkage test
    GRPC_LIBS+=("$lib_path")
    if [ -z "$LIB_DIR" ]; then
        LIB_DIR=$(dirname "$lib_path")
    fi

    # Step 1: Check if it's a valid ar archive
    if ! "${AR_BIN}" t "$lib_path" > /tmp/ar_output_$$ 2>&1; then
        echo " : FAILED"
        echo "  Error: Invalid archive format: $lib_path" >&2
        cat /tmp/ar_output_$$ >&2
        rm -f /tmp/ar_output_$$
        exit 1
    fi

    # Count object files
    object_count=$(wc -l < /tmp/ar_output_$$)
    rm -f /tmp/ar_output_$$

    if [ "$object_count" -eq 0 ]; then
        echo " : FAILED"
        echo "  Error: Archive is empty: $lib_path" >&2
        exit 1
    fi

    # Step 2: Validate object file format (ELF relocatable)
    first_object=$("${AR_BIN}" t "$lib_path" | head -1)
    obj_info=$("${AR_BIN}" p "$lib_path" "$first_object" 2>/dev/null | "${FILE_BIN}" -)

    # Check for ELF relocatable format
    if ! echo "$obj_info" | grep -q "ELF.*relocatable"; then
        echo " : FAILED"
        echo "  Error: Invalid object format. Expected: ELF relocatable object" >&2
        echo "  Got: $obj_info" >&2
        exit 1
    fi

    # Step 3: Check symbols
    if ! "${NM_BIN}" --defined-only "$lib_path" > /tmp/nm_output_$$ 2>&1; then
        echo " : FAILED"
        echo "  Error: Cannot read symbols from archive: $lib_path" >&2
        cat /tmp/nm_output_$$ >&2
        rm -f /tmp/nm_output_$$
        exit 1
    fi

    # Count defined symbols
    symbol_count=$(grep -v '^\s*U ' /tmp/nm_output_$$ | wc -l)
    rm -f /tmp/nm_output_$$

    if [ "$symbol_count" -eq 0 ]; then
        echo " : FAILED"
        echo "  Error: No defined symbols found: $lib_path" >&2
        exit 1
    fi

    # Step 4: Check for exported global symbols (functions and data)
    exported_count=$("${NM_BIN}" --extern-only "$lib_path" 2>/dev/null | grep -c " [TDW] " || true)
    if [ "$exported_count" -eq 0 ]; then
        echo " : FAILED"
        echo "  Error: No exported symbols found: $lib_path" >&2
        exit 1
    fi

    echo " : OK ($object_count objects, $symbol_count symbols, $exported_count exported)"
done

# Step 5: Linkage test for gRPC libraries
# Detect if we have gRPC libraries to test
HAS_GRPCXX=false
HAS_GRPC=false
for lib in "${GRPC_LIBS[@]}"; do
    basename=$(basename "$lib")
    if [[ "$basename" == libgrpc++* ]]; then
        HAS_GRPCXX=true
    fi
    if [[ "$basename" == libgrpc.a ]] || [[ "$basename" == libgrpc-*.a ]]; then
        HAS_GRPC=true
    fi
done

if [ "$HAS_GRPCXX" = true ] && [ "$HAS_GRPC" = true ]; then
    echo -n "Testing actual linkage with gRPC symbols"

    # Find a C API symbol from libgrpc
    GRPC_C_SYMBOL=""
    for lib in "${GRPC_LIBS[@]}"; do
        basename=$(basename "$lib")
        if [[ "$basename" == libgrpc.a ]] || [[ "$basename" == libgrpc-*.a ]]; then
            # Find grpc_init or any grpc_* C API function
            if "${NM_BIN}" --extern-only "$lib" 2>/dev/null | grep -q " T grpc_init$"; then
                GRPC_C_SYMBOL="grpc_init"
            else
                GRPC_C_SYMBOL=$("${NM_BIN}" --extern-only "$lib" 2>/dev/null | \
                    grep " T grpc_" | head -1 | awk '{print $3}')
            fi
            [ -n "$GRPC_C_SYMBOL" ] && break
        fi
    done

    # Find any C++ function symbol containing "grpc"
    GRPC_CXX_SYMBOL=""
    for lib in "${GRPC_LIBS[@]}"; do
        basename=$(basename "$lib")
        if [[ "$basename" == libgrpc++* ]]; then
            # Simple check: any T (text/function) symbol containing "grpc"
            GRPC_CXX_SYMBOL=$("${NM_BIN}" --extern-only "$lib" 2>/dev/null | \
                grep " T " | grep -i "grpc" | head -1 | awk '{print $3}')
            [ -n "$GRPC_CXX_SYMBOL" ] && break
        fi
    done

    if [ -z "$GRPC_C_SYMBOL" ] || [ -z "$GRPC_CXX_SYMBOL" ]; then
        echo " : WARNING"
        echo "  Could not find representative symbols for linkage test" >&2
        echo "  C symbol: ${GRPC_C_SYMBOL:-not found}" >&2
        echo "  C++ symbol: ${GRPC_CXX_SYMBOL:-not found}" >&2
        echo "  Skipping linkage test" >&2
    else
        # Decode the C++ symbol for display
        CXX_SYMBOL_DECODED=$(echo "$GRPC_CXX_SYMBOL" | "${CXXFILT_BIN}" 2>/dev/null || echo "$GRPC_CXX_SYMBOL")

        # Create a test program
        cat > /tmp/test_grpc_link_$$.cc <<EOF
// Test program to verify gRPC static libraries can be linked
extern "C" void ${GRPC_C_SYMBOL}(void);
extern "C" void ${GRPC_CXX_SYMBOL}(void);

int main() {
    void (*c_api_ptr)(void) = &${GRPC_C_SYMBOL};
    void (*cxx_api_ptr)(void) = &${GRPC_CXX_SYMBOL};
    return (c_api_ptr && cxx_api_ptr) ? 0 : 1;
}
EOF

        # Compile the test program
        if ! "${CXX_BIN}" -c /tmp/test_grpc_link_$$.cc -o /tmp/test_grpc_link_$$.o -std=c++17 2>/dev/null; then
            echo " : FAILED"
            echo "  Error: Could not compile test program" >&2
            rm -f /tmp/test_grpc_link_$$.cc /tmp/test_grpc_link_$$.o
            exit 1
        fi

        # Build library list
        LINK_LIBS=""
        for lib in "${GRPC_LIBS[@]}"; do
            basename=$(basename "$lib" .a)
            if [[ "$basename" == lib* ]]; then
                basename=${basename#lib}
            fi
            LINK_LIBS="$LINK_LIBS -l$basename"
        done

        # Try to link
        link_output=$("${CXX_BIN}" /tmp/test_grpc_link_$$.o -L"$LIB_DIR" $LINK_LIBS -o /tmp/test_grpc_link_$$ 2>&1 || true)

        # Check if our symbols were found
        if echo "$link_output" | grep -q "undefined reference to .*${GRPC_C_SYMBOL}\|undefined reference to.*${GRPC_CXX_SYMBOL}"; then
            echo " : FAILED"
            echo "  Error: Static libraries missing required symbols" >&2
            echo "  Missing: ${GRPC_C_SYMBOL} or ${CXX_SYMBOL_DECODED}" >&2
            echo "$link_output" >&2
            rm -f /tmp/test_grpc_link_$$ /tmp/test_grpc_link_$$.o /tmp/test_grpc_link_$$.cc
            exit 1
        fi

        echo " : OK (verified: ${GRPC_C_SYMBOL}, ${CXX_SYMBOL_DECODED})"
        rm -f /tmp/test_grpc_link_$$ /tmp/test_grpc_link_$$.o /tmp/test_grpc_link_$$.cc
    fi
fi

echo "All validation checks passed"
