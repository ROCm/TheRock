#!/bin/bash
# Benchmark signing performance

set -e

SERVER_URL="${GPG_SIGNING_SERVER:-http://localhost:8080/sign}"
NUM_REQUESTS="${1:-100}"

echo "=== GPG Signing Performance Benchmark ==="
echo ""
echo "Server: $SERVER_URL"
echo "Number of requests: $NUM_REQUESTS"
echo ""

# Check if server is running
if ! curl -s -o /dev/null "$SERVER_URL" 2>/dev/null; then
    echo "⚠ Server not responding at $SERVER_URL"
    echo "  Start it with: make start-server-bg"
    exit 1
fi

# Check if we have the test key
export GNUPGHOME=$(pwd)/.gnupg
if ! gpg --list-secret-keys signer@example.com >/dev/null 2>&1; then
    echo "⚠ Test GPG key not found"
    echo "  Run: make setup-gpg"
    exit 1
fi

KEY_ID=$(gpg --list-secret-keys --keyid-format LONG signer@example.com | grep sec | awk '{print $2}' | cut -d'/' -f2)

echo "Warming up..."
echo "test" | ./gpgshim --detach-sign -u "$KEY_ID" > /dev/null 2>&1

echo ""
echo "Running sequential benchmark (1 at a time)..."
echo "----------------------------------------------"

START=$(date +%s.%N)
for i in $(seq 1 $NUM_REQUESTS); do
    echo "Request $i" | ./gpgshim --detach-sign -u "$KEY_ID" > /dev/null 2>&1
    if [ $((i % 10)) -eq 0 ]; then
        echo -n "."
    fi
done
echo ""
END=$(date +%s.%N)

DURATION=$(echo "$END - $START" | bc)
REQUESTS_PER_SEC=$(echo "scale=2; $NUM_REQUESTS / $DURATION" | bc)
AVG_TIME=$(echo "scale=3; $DURATION / $NUM_REQUESTS" | bc)

echo ""
echo "Sequential Results:"
echo "  Total time: ${DURATION}s"
echo "  Requests/sec: $REQUESTS_PER_SEC"
echo "  Avg time per request: ${AVG_TIME}s"
echo ""

# Concurrent benchmark
echo "Running concurrent benchmark (10 at a time)..."
echo "-----------------------------------------------"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

START=$(date +%s.%N)
BATCH_SIZE=10
for batch_start in $(seq 1 $BATCH_SIZE $NUM_REQUESTS); do
    batch_end=$((batch_start + BATCH_SIZE - 1))
    if [ $batch_end -gt $NUM_REQUESTS ]; then
        batch_end=$NUM_REQUESTS
    fi

    for i in $(seq $batch_start $batch_end); do
        (
            echo "Request $i" | ./gpgshim --detach-sign -u "$KEY_ID" > /dev/null 2>&1
            echo "$i" > "$TEMP_DIR/done.$i"
        ) &
    done

    # Wait for this batch to complete
    wait
    echo -n "."
done
echo ""
END=$(date +%s.%N)

DURATION=$(echo "$END - $START" | bc)
REQUESTS_PER_SEC=$(echo "scale=2; $NUM_REQUESTS / $DURATION" | bc)
AVG_TIME=$(echo "scale=3; $DURATION / $NUM_REQUESTS" | bc)

echo ""
echo "Concurrent Results (batches of $BATCH_SIZE):"
echo "  Total time: ${DURATION}s"
echo "  Requests/sec: $REQUESTS_PER_SEC"
echo "  Avg time per request: ${AVG_TIME}s"
echo ""

# Calculate theoretical max throughput
MAX_THREADS=$(curl -s "$SERVER_URL/../quit" 2>&1 | grep -o "Max concurrent threads: [0-9]*" || echo "Max concurrent threads: 10")
MAX_THREADS=$(echo "$MAX_THREADS" | grep -o "[0-9]*")
if [ -z "$MAX_THREADS" ]; then
    MAX_THREADS=10
fi

THEORETICAL_MAX=$(echo "scale=2; $MAX_THREADS / $AVG_TIME" | bc)

echo "Performance Summary:"
echo "--------------------"
echo "Sequential throughput:  $REQUESTS_PER_SEC req/s"
echo "Concurrent throughput:  $REQUESTS_PER_SEC req/s (batches of 10)"
echo "Server max threads:     $MAX_THREADS"
echo "Theoretical max:        ~${THEORETICAL_MAX} req/s"
echo ""
echo "Note: Actual throughput depends on:"
echo "  - GPG key type and size (RSA 2048 vs 4096)"
echo "  - CPU performance"
echo "  - Disk I/O for GPG keyring access"
echo "  - Network latency (if remote)"
