#!/bin/bash
# Integration tests for authentication and authorization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_test() {
    echo -e "\n${YELLOW}TEST: $1${NC}"
}

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check prerequisites
if [ ! -f "auth.py" ]; then
    echo "Error: auth.py not found"
    exit 1
fi

if [ ! -f "signing-server.py" ]; then
    echo "Error: signing-server.py not found"
    exit 1
fi

# Create test GPG key if needed
if [ ! -d ".gnupg" ]; then
    echo "Setting up test GPG key..."
    make setup-gpg 2>&1 > /dev/null || {
        echo "Error: Failed to setup GPG key"
        exit 1
    }
fi

# Get test key ID
export GNUPGHOME="$(pwd)/.gnupg"
TEST_KEY=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep '^sec' | head -1 | awk '{print $2}' | cut -d'/' -f2)
if [ -z "$TEST_KEY" ]; then
    echo "Error: No test key found"
    exit 1
fi
echo "Using test key: $TEST_KEY"

# Update authorization config to include test key
cat > config/authorization.json <<EOF
{
  "roles": {
    "test": {
      "allowed_keys": ["$TEST_KEY"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10,
      "description": "Testing role"
    },
    "restricted": {
      "allowed_keys": ["fake-key-not-available"],
      "allowed_digest_algos": ["SHA256"],
      "max_requests_per_hour": 10,
      "description": "Restricted role for testing denial"
    }
  }
}
EOF

# Generate test tokens
print_test "Generating test tokens"
TEST_TOKEN=$(python3 -c "
from auth import generate_jwt_token
print(generate_jwt_token('test-client', 'test', 'test-secret-for-local-testing', 1))
")

RESTRICTED_TOKEN=$(python3 -c "
from auth import generate_jwt_token
print(generate_jwt_token('restricted-client', 'restricted', 'restricted-secret-for-testing', 1))
")

INVALID_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"

echo "Test token: ${TEST_TOKEN:0:50}..."
echo "Restricted token: ${RESTRICTED_TOKEN:0:50}..."

# Start server with authentication
print_test "Starting authenticated server"
python3 signing-server.py \
    --port 18080 \
    --keyring .gnupg \
    --enable-auth \
    --secrets-file config/secrets.json \
    --authz-config config/authorization.json \
    --audit-log test-audit.log \
    > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start"
    exit 1
fi

echo "Server started (PID: $SERVER_PID)"

# Cleanup function
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    rm -f test-audit.log
}
trap cleanup EXIT

# Helper function to sign with token
sign_with_token() {
    local token="$1"
    local key_id="$2"
    local expect_success="$3"

    response=$(echo "test data" | python3 -c "
import sys
import json
import base64
import hashlib
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# Read data
data = sys.stdin.buffer.read()

# Compute SHA256 digest
digest = hashlib.sha256(data).digest()

# Prepare request
payload = {
    'digest': base64.b64encode(digest).decode('ascii'),
    'key_id': '$key_id',
    'digest_algo': 'SHA256',
    'armor': False
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer $token'
}

request = Request(
    'http://localhost:18080/sign',
    data=json.dumps(payload).encode('utf-8'),
    headers=headers,
    method='POST'
)

try:
    with urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode('utf-8'))
        print('SUCCESS')
        sys.exit(0)
except HTTPError as e:
    print('ERROR:' + str(e.code))
    sys.exit(1)
except Exception as e:
    print('EXCEPTION:' + str(e))
    sys.exit(1)
" 2>&1)

    echo "$response"
}

# Test 1: No token should fail with 401
print_test "Request without token should fail (401)"
response=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:18080/sign \
    -H "Content-Type: application/json" \
    -d '{"digest": "dGVzdA==", "key_id": "test", "digest_algo": "SHA256"}')
if [ "$response" = "401" ]; then
    pass "No token rejected with 401"
else
    fail "Expected 401, got $response"
fi

# Test 2: Invalid token should fail with 401
print_test "Request with invalid token should fail (401)"
response=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:18080/sign \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $INVALID_TOKEN" \
    -d '{"digest": "dGVzdA==", "key_id": "test", "digest_algo": "SHA256"}')
if [ "$response" = "401" ]; then
    pass "Invalid token rejected with 401"
else
    fail "Expected 401, got $response"
fi

# Test 3: Valid token with correct key should succeed
print_test "Valid token with authorized key should succeed"
result=$(sign_with_token "$TEST_TOKEN" "$TEST_KEY" "true")
if echo "$result" | grep -q "SUCCESS"; then
    pass "Authorized request succeeded"
else
    fail "Authorized request failed: $result"
fi

# Test 4: Valid token with unauthorized key should fail (403)
print_test "Valid token with unauthorized key should fail (403)"
result=$(sign_with_token "$RESTRICTED_TOKEN" "$TEST_KEY" "false")
if echo "$result" | grep -q "ERROR:403"; then
    pass "Unauthorized key rejected with 403"
else
    fail "Expected 403, got: $result"
fi

# Test 5: Rate limiting
print_test "Rate limiting (max 10 requests per hour)"
# Make 10 successful requests
for i in {1..10}; do
    result=$(sign_with_token "$TEST_TOKEN" "$TEST_KEY" "true" 2>&1)
    if ! echo "$result" | grep -q "SUCCESS"; then
        fail "Request $i failed unexpectedly"
        break
    fi
done

# 11th request should be rate limited (429)
result=$(sign_with_token "$TEST_TOKEN" "$TEST_KEY" "false")
if echo "$result" | grep -q "ERROR:429"; then
    pass "Rate limit enforced (11th request rejected with 429)"
else
    fail "Rate limit not enforced, got: $result"
fi

# Test 6: Check audit log
print_test "Verify audit log entries"
if [ -f "test-audit.log" ]; then
    # Should have AUTH_FAILED entries
    auth_failed=$(grep -c '"action": "AUTH_FAILED"' test-audit.log || echo 0)
    # Should have SIGNED entries
    signed=$(grep -c '"action": "SIGNED"' test-audit.log || echo 0)
    # Should have DENIED entry
    denied=$(grep -c '"action": "DENIED"' test-audit.log || echo 0)
    # Should have RATE_LIMITED entry
    rate_limited=$(grep -c '"action": "RATE_LIMITED"' test-audit.log || echo 0)

    if [ "$auth_failed" -gt 0 ] && [ "$signed" -gt 0 ] && [ "$denied" -gt 0 ] && [ "$rate_limited" -gt 0 ]; then
        pass "Audit log contains expected entries (AUTH_FAILED: $auth_failed, SIGNED: $signed, DENIED: $denied, RATE_LIMITED: $rate_limited)"
    else
        fail "Audit log missing entries (AUTH_FAILED: $auth_failed, SIGNED: $signed, DENIED: $denied, RATE_LIMITED: $rate_limited)"
    fi
else
    fail "Audit log file not found"
fi

# Print summary
echo ""
echo "========================================"
echo "Test Results:"
echo "  Passed: $TESTS_PASSED"
echo "  Failed: $TESTS_FAILED"
echo "========================================"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
