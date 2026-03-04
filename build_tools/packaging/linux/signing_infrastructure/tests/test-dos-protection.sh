#!/bin/bash
# Test DoS protection features of signing-server

set -e

echo "=== Testing DoS Protection ==="
echo ""

SERVER_URL="${GPG_SIGNING_SERVER:-http://localhost:8080}"

echo "Server: $SERVER_URL"
echo ""

# Test 1: Oversized request
echo "Test 1: Oversized request (should be rejected)"
echo "-----------------------------------------------"
# Create a request larger than 10KB
LARGE_PAYLOAD=$(python3 -c "import json, base64; print(json.dumps({'digest': base64.b64encode(b'x' * 15000).decode(), 'key_id': 'test'}))")
RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d "$LARGE_PAYLOAD" \
  "$SERVER_URL/sign" 2>/dev/null || echo "FAILED")

HTTP_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
if [ "$HTTP_CODE" = "413" ]; then
  echo "✓ Oversized request rejected with HTTP 413"
else
  echo "✗ Expected HTTP 413, got: $HTTP_CODE"
  echo "$RESPONSE"
fi
echo ""

# Test 2: Normal sized request
echo "Test 2: Normal request (should succeed or fail on signature)"
echo "--------------------------------------------------------------"
NORMAL_PAYLOAD='{"digest":"'$(echo -n "test" | sha256sum | cut -d' ' -f1 | xxd -r -p | base64)'","key_id":"test","digest_algo":"SHA256"}'
RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d "$NORMAL_PAYLOAD" \
  "$SERVER_URL/sign" 2>/dev/null || echo "FAILED")

HTTP_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
PAYLOAD_SIZE=${#NORMAL_PAYLOAD}
echo "Request size: $PAYLOAD_SIZE bytes"
if [ "$HTTP_CODE" != "413" ]; then
  echo "✓ Normal request accepted (HTTP $HTTP_CODE)"
  echo "  (May fail on signing if key doesn't exist, but wasn't rejected for size)"
else
  echo "✗ Normal request rejected for size (should not happen)"
fi
echo ""

# Test 3: Slow request (if timeout is short)
echo "Test 3: Slow request simulation"
echo "--------------------------------"
echo "Note: This test requires manual setup with low --read-timeout"
echo "To test: Start server with --read-timeout 2"
echo "Then run: (echo -n '{\"digest\":\"'; sleep 5; echo 'aaaa\"}') | curl -X POST -d @- $SERVER_URL/sign"
echo ""

echo "=== DoS Protection Tests Complete ==="
