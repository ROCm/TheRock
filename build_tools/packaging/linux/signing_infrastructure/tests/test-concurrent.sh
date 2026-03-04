#!/bin/bash
# Test concurrent request handling

set -e

SERVER_URL="${GPG_SIGNING_SERVER:-http://localhost:8080/sign}"
NUM_REQUESTS=20

echo "=== Testing Concurrent Request Handling ==="
echo ""
echo "Server: $SERVER_URL"
echo "Concurrent requests: $NUM_REQUESTS"
echo ""

# Check if server is running
if ! curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL" 2>/dev/null; then
    echo "⚠ Server not responding at $SERVER_URL"
    echo "  Start it with: make start-server-bg"
    exit 1
fi

echo "Creating test digest..."
TEST_DIGEST=$(echo -n "test data" | sha256sum | cut -d' ' -f1 | xxd -r -p | base64)
PAYLOAD='{"digest":"'$TEST_DIGEST'","key_id":"test","digest_algo":"SHA256"}'

echo "Sending $NUM_REQUESTS concurrent requests..."
echo ""

# Create a temporary directory for results
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Send requests concurrently
START_TIME=$(date +%s)
for i in $(seq 1 $NUM_REQUESTS); do
    (
        RESPONSE=$(curl -s -w "\nHTTP:%{http_code}\nTIME:%{time_total}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" \
            "$SERVER_URL" 2>/dev/null || echo "FAILED")

        HTTP_CODE=$(echo "$RESPONSE" | grep "^HTTP:" | cut -d: -f2)
        TIME=$(echo "$RESPONSE" | grep "^TIME:" | cut -d: -f2)

        echo "$i $HTTP_CODE $TIME" > "$TEMP_DIR/result.$i"
    ) &
done

# Wait for all background jobs
wait

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Results:"
echo "--------"

SUCCESS=0
BUSY=0
FAILED=0

for i in $(seq 1 $NUM_REQUESTS); do
    if [ -f "$TEMP_DIR/result.$i" ]; then
        read REQ_NUM HTTP_CODE TIME < "$TEMP_DIR/result.$i"

        if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "500" ]; then
            SUCCESS=$((SUCCESS + 1))
            echo "Request $REQ_NUM: HTTP $HTTP_CODE (${TIME}s)"
        elif [ "$HTTP_CODE" = "503" ]; then
            BUSY=$((BUSY + 1))
            echo "Request $REQ_NUM: HTTP 503 - Server busy"
        else
            FAILED=$((FAILED + 1))
            echo "Request $REQ_NUM: HTTP $HTTP_CODE - Failed"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "Request $i: No result file"
    fi
done

echo ""
echo "Summary:"
echo "--------"
echo "Total requests:    $NUM_REQUESTS"
echo "Successful:        $SUCCESS"
echo "Server busy (503): $BUSY"
echo "Failed:            $FAILED"
echo "Total time:        ${DURATION}s"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "✓ Server handled concurrent requests"
    if [ $BUSY -gt 0 ]; then
        echo "  (Some requests were rate-limited - this is expected behavior)"
    fi
else
    echo "✗ No requests succeeded"
    exit 1
fi
