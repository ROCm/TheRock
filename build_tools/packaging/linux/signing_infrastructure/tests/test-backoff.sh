#!/bin/bash
# Test exponential backoff behavior

set -e

export GNUPGHOME=$(pwd)/.gnupg

echo "=== Testing Exponential Backoff ==="
echo ""

# Check if we have the test key
if ! gpg --list-secret-keys signer@example.com >/dev/null 2>&1; then
    echo "⚠ Test GPG key not found"
    echo "  Run: make setup-gpg"
    exit 1
fi

KEY_ID=$(gpg --list-secret-keys --keyid-format LONG signer@example.com | grep sec | awk '{print $2}' | cut -d'/' -f2)

# Start a server with very low thread limit to force 503 responses
echo "Starting server with max 2 threads..."
export MAX_THREADS=2
./signing-server.py --port 8888 --max-threads 2 --key "$KEY_ID" > /tmp/backoff-server.log 2>&1 &
SERVER_PID=$!
sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "✗ Server failed to start"
    cat /tmp/backoff-server.log
    exit 1
fi

echo "✓ Server started (PID: $SERVER_PID)"
echo ""

# Set up environment for testing
export GPG_SIGNING_SERVER='http://localhost:8888/sign'
export GPG_SHIM_DEBUG=1
export GPG_MAX_RETRIES=3
export GPG_INITIAL_BACKOFF=0.5

echo "Configuration:"
echo "  Server max threads: 2"
echo "  Shim max retries: 3"
echo "  Initial backoff: 0.5s"
echo ""

# Send many concurrent requests to force 503 responses
echo "Sending 10 concurrent requests (server can only handle 2)..."
echo "Expected: Some requests will retry with backoff"
echo ""

TEMP_DIR=$(mktemp -d)
trap "kill $SERVER_PID 2>/dev/null || true; rm -rf $TEMP_DIR /tmp/backoff-server.log" EXIT

START=$(date +%s)

for i in $(seq 1 10); do
    (
        echo "Request $i" > /tmp/backoff-input-$i
        RESULT=$(echo "Request $i data" | ./gpgshim --detach-sign -u "$KEY_ID" 2>&1)
        EXIT_CODE=$?
        END=$(date +%s)
        DURATION=$((END - START))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$DURATION] Request $i: SUCCESS" >> "$TEMP_DIR/results.log"
        else
            echo "[$DURATION] Request $i: FAILED" >> "$TEMP_DIR/results.log"
            echo "$RESULT" | grep -E "(busy|retry|503)" >> "$TEMP_DIR/retries.log" 2>/dev/null || true
        fi
    ) &
done

# Wait for all requests
wait

echo ""
echo "Results:"
cat "$TEMP_DIR/results.log" 2>/dev/null | sort -n || echo "No results"

echo ""
if [ -f "$TEMP_DIR/retries.log" ]; then
    RETRY_COUNT=$(wc -l < "$TEMP_DIR/retries.log")
    echo "Retry/backoff events detected: $RETRY_COUNT"
    echo ""
    echo "Sample retry messages:"
    head -5 "$TEMP_DIR/retries.log"
    echo ""
    echo "✓ Backoff mechanism is working"
else
    echo "⚠ No retry events detected (all requests may have succeeded)"
fi

echo ""
echo "Server log (last 20 lines):"
tail -20 /tmp/backoff-server.log

# Cleanup
kill $SERVER_PID 2>/dev/null || true
rm -f /tmp/backoff-input-*
