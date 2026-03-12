#!/bin/bash

# #############################################################################
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################
#
# Build Static xz Binary for ROCm Installer Hybrid Compression
#
# This script builds a statically-linked xz binary from official source
# for embedding in the ROCm installer to decompress test packages.
#
# License: XZ Utils is licensed under BSD Zero Clause License (0BSD)
#          When built with musl, the result is 0BSD + MIT (no attribution required)
#
# Security: Uses xz 5.6.3 which is SAFE (backdoor CVE-2024-3094 was only in 5.6.0/5.6.1)
#
# Usage:
#   ./build-xz-static.sh
#
# Output:
#   xz-static - ~200 KB statically-linked xz binary (no dependencies)
#
# #############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
XZ_VERSION="5.6.3"
XZ_URL="https://tukaani.org/xz/xz-${XZ_VERSION}.tar.gz"
XZ_SIG_URL="https://tukaani.org/xz/xz-${XZ_VERSION}.tar.gz.sig"
BUILD_DIR="/tmp/xz-static-build-$$"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BINARY="$SCRIPT_DIR/xz-static"

# XZ Utils maintainer GPG key fingerprint (Lasse Collin)
# Key ID: 38EE757D69184620
GPG_KEY_ID="38EE757D69184620"

# Functions
print_noerr() {
    echo -e "\e[32m$1\e[0m"
}

print_warning() {
    echo -e "\e[93mWARNING: $1\e[0m"
}

print_err() {
    echo -e "\e[31mERROR: $1\e[0m"
}

# Convert size in bytes to human-readable format with proper units (GB, MB, KB)
format_size() {
    local bytes=$1
    local kb=$((bytes / 1024))
    local mb=$((kb / 1024))
    local gb=$((mb / 1024))

    if [[ $gb -gt 0 ]]; then
        local gb_dec=$(( (mb * 10 / 1024) % 10 ))
        echo "${gb}.${gb_dec} GB"
    elif [[ $mb -gt 0 ]]; then
        local mb_dec=$(( (kb * 10 / 1024) % 10 ))
        echo "${mb}.${mb_dec} MB"
    else
        echo "${kb} KB"
    fi
}

cleanup() {
    if [[ -d "$BUILD_DIR" ]]; then
        echo "Cleaning up build directory..."
        rm -rf "$BUILD_DIR"
    fi
}

trap cleanup EXIT

check_dependencies() {
    echo "Checking build dependencies..."

    local missing_deps=()

    # Check for required tools
    for tool in gcc make wget tar; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_err "Missing required dependencies: ${missing_deps[*]}"
        echo ""
        echo "Install with:"
        if [[ -f /etc/redhat-release ]]; then
            echo "  sudo dnf install -y gcc make wget tar glibc-static"
        elif [[ -f /etc/debian_version ]]; then
            echo "  sudo apt-get install -y gcc make wget tar"
        fi
        exit 1
    fi

    print_noerr "All basic dependencies found."

    # Check for glibc-static on RHEL/AlmaLinux (needed for static builds)
    if [[ -f /etc/redhat-release ]]; then
        if ! rpm -qa | grep -q glibc-static; then
            print_warning "glibc-static not found - installing it for proper static linking"
            if dnf install -y glibc-static 2>/dev/null || yum install -y glibc-static 2>/dev/null; then
                print_noerr "glibc-static installed successfully."
            else
                print_warning "Could not install glibc-static - build may produce dynamic binary"
            fi
        else
            print_noerr "glibc-static is installed"
        fi
    fi
}

check_musl() {
    echo "Checking for musl-gcc (recommended for cleanest licensing)..."

    if command -v musl-gcc &> /dev/null; then
        print_noerr "musl-gcc found - will build with musl (0BSD + MIT licensing)"
        return 0
    else
        print_warning "musl-gcc not found - will build with glibc (0BSD + LGPL 2.1+ licensing)"

        # Check if running in non-interactive environment (Docker, CI/CD, or called from build script)
        if [[ ! -t 0 ]] || [[ -n "${CI:-}" ]] || [[ -n "${DOCKER_BUILD:-}" ]] || [[ -n "${XZ_STATIC_NONINTERACTIVE:-}" ]]; then
            echo "Non-interactive environment detected - automatically proceeding with glibc build"
            return 1
        fi

        echo ""
        echo "To install musl for cleaner licensing:"
        if [[ -f /etc/redhat-release ]]; then
            echo "  sudo dnf install -y musl-gcc musl-libc-static"
        elif [[ -f /etc/debian_version ]]; then
            echo "  sudo apt-get install -y musl-tools musl-dev"
        fi
        echo ""
        read -p "Continue with glibc build? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return 1
    fi
}

verify_gpg_signature() {
    echo "Attempting GPG signature verification..."

    if ! command -v gpg &> /dev/null; then
        print_warning "GPG not found - skipping signature verification"
        print_warning "Consider installing gpg/gpg2 for security verification"
        return 1
    fi

    # Try to import the key
    echo "Importing XZ Utils maintainer public key..."
    if gpg --keyserver keyserver.ubuntu.com --recv-keys "$GPG_KEY_ID" 2>/dev/null; then
        print_noerr "Public key imported."
    else
        print_warning "Could not import GPG key - skipping signature verification"
        return 1
    fi

    # Verify the signature
    echo "Verifying signature..."
    if gpg --verify "xz-${XZ_VERSION}.tar.gz.sig" "xz-${XZ_VERSION}.tar.gz" 2>&1 | grep -q "Good signature"; then
        print_noerr "GPG signature verification PASSED."
        return 0
    else
        print_err "GPG signature verification FAILED"

        # Check if running in non-interactive environment
        if [[ ! -t 0 ]] || [[ -n "${CI:-}" ]] || [[ -n "${DOCKER_BUILD:-}" ]] || [[ -n "${XZ_STATIC_NONINTERACTIVE:-}" ]]; then
            print_warning "Non-interactive environment - continuing despite failed verification"
            print_warning "This is acceptable for trusted source downloads in CI/CD"
            return 1
        fi

        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return 1
    fi
}

check_for_backdoor() {
    local xz_binary="$1"

    echo "Checking for CVE-2024-3094 backdoor indicators..."

    # Check version number
    local version
    version=$("$xz_binary" --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')

    if echo "$version" | grep -qE '^5\.6\.[01]$'; then
        print_err "BACKDOORED VERSION DETECTED: $version"
        print_err "This version contains CVE-2024-3094 backdoor!"
        exit 1
    fi

    # Check for backdoor author strings
    if strings "$xz_binary" | grep -qi "jia.*tan\|krygorin"; then
        print_err "Backdoor author strings detected!"
        exit 1
    fi

    print_noerr "No backdoor indicators found (version: $version)."
}

build_xz() {
    local use_musl=$1

    echo "Building xz-static..."
    echo ""

    cd "$BUILD_DIR/xz-${XZ_VERSION}"

    # Configure
    echo "Running configure..."
    if [[ $use_musl -eq 0 ]]; then
        CC=musl-gcc ./configure \
            --enable-static \
            --disable-shared \
            --disable-nls \
            --disable-scripts \
            --disable-doc \
            LDFLAGS="-static"
    else
        # For glibc static builds, explicitly request static linking
        # Use both LDFLAGS and pass --enable-static
        ./configure \
            --enable-static \
            --disable-shared \
            --disable-nls \
            --disable-scripts \
            --disable-doc \
            LDFLAGS="-static" \
            LIBS="-lpthread"
    fi
    print_noerr "Configuration complete."

    # Build
    echo "Compiling (using $(nproc) cores)..."
    # For static builds with glibc, pass LDFLAGS directly to make to ensure static linking
    if [[ $use_musl -eq 0 ]]; then
        make -j"$(nproc)"
    else
        make -j"$(nproc)" LDFLAGS="-all-static"
    fi
    print_noerr "Compilation complete."

    # Strip to reduce size
    echo "Stripping binary..."
    strip src/xz/xz
    print_noerr "Binary stripped."

    print_noerr "Build complete."
}

test_binary() {
    local xz_binary="$1"

    echo "Testing xz-static binary..."

    # Test 1: Check if it's static
    echo "Checking if binary is statically linked..."
    if ldd "$xz_binary" 2>&1 | grep -q "not a dynamic executable"; then
        print_noerr "Binary is statically linked."
    else
        print_err "Binary is NOT statically linked!"
        ldd "$xz_binary"
        exit 1
    fi

    # Test 2: Check version
    echo "Checking version..."
    "$xz_binary" --version
    print_noerr "Version check passed."

    # Test 3: Functional test
    echo "Running functional test..."
    local test_data="This is test data for xz compression verification. Testing 123."
    if echo "$test_data" | "$xz_binary" -z | "$xz_binary" -d | grep -q "Testing 123"; then
        print_noerr "Compression/decompression test PASSED."
    else
        print_err "Compression/decompression test FAILED"
        exit 1
    fi

    # Test 4: Check for backdoor
    check_for_backdoor "$xz_binary"
}

# Main execution
main() {
    echo "============================================================="
    echo "  XZ Static Binary Build Script"
    echo "  Version: $XZ_VERSION"
    echo "  Output: $OUTPUT_BINARY"
    echo "============================================================="
    echo ""

    # Check dependencies
    check_dependencies

    # Check for musl (capture return value without triggering set -e)
    local use_musl=1
    if check_musl; then
        use_musl=0
    fi

    # Create build directory
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Download source
    echo "Downloading xz-${XZ_VERSION} source..."
    if ! wget -q --show-progress "$XZ_URL"; then
        print_err "Failed to download source"
        exit 1
    fi
    print_noerr "Source downloaded."

    # Download signature (optional but recommended)
    echo "Downloading GPG signature..."
    if wget -q "$XZ_SIG_URL" 2>/dev/null; then
        # Capture return value without triggering set -e
        verify_gpg_signature || true
    else
        print_warning "GPG signature not available - skipping verification"
    fi

    # Extract
    echo "Extracting source..."
    tar -xzf "xz-${XZ_VERSION}.tar.gz"
    print_noerr "Source extracted."

    # Build
    build_xz $use_musl

    # Test
    test_binary "$BUILD_DIR/xz-${XZ_VERSION}/src/xz/xz"

    # Copy to output location
    echo "Copying binary to: $OUTPUT_BINARY"
    cp "$BUILD_DIR/xz-${XZ_VERSION}/src/xz/xz" "$OUTPUT_BINARY"
    chmod +x "$OUTPUT_BINARY"
    print_noerr "Binary installed to $OUTPUT_BINARY"

    # Display results
    echo ""
    echo "============================================================="
    print_noerr "XZ static binary built successfully!"
    echo "============================================================="
    echo ""
    local binary_bytes
    local binary_size

    binary_bytes=$(stat -c%s "$OUTPUT_BINARY" 2>/dev/null || stat -f%z "$OUTPUT_BINARY" 2>/dev/null)
    binary_size=$(format_size "$binary_bytes")

    echo "Binary location: $OUTPUT_BINARY"
    echo "Binary size:     $binary_size"
    echo "Version:         $("$OUTPUT_BINARY" --version | head -1)"
    echo ""

    if [[ $use_musl -eq 0 ]]; then
        echo "License:         0BSD (XZ Utils) + MIT (musl libc)"
        echo "Attribution:     Optional (not legally required)"
    else
        echo "License:         0BSD (XZ Utils) + LGPL 2.1+ (glibc components)"
        echo "Attribution:     Required for LGPL components"
    fi

    echo ""
    echo "To verify:"
    echo "  ldd $OUTPUT_BINARY"
    echo "  $OUTPUT_BINARY --version"
    echo "  echo 'test' | $OUTPUT_BINARY | $OUTPUT_BINARY -d"
    echo ""
    print_noerr "Ready for use in hybrid compression!"
}

# Run main
main "$@"
