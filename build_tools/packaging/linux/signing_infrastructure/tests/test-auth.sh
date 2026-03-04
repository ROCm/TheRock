#!/bin/bash
# Comprehensive authentication and authorization test suite
# Tests all authentication scenarios including token validation, authorization, and rate limiting

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test directory
cd "$(dirname "$0")"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Authentication Test Suite${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Helper functions
pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_RUN=$((TESTS_RUN + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_RUN=$((TESTS_RUN + 1))
}

skip() {
    echo -e "${YELLOW}○ SKIP${NC}: $1"
}

section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Cleanup function
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo -e "\n${YELLOW}Cleaning up...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    rm -f test-auth-audit.log server-test-auth.log
}
trap cleanup EXIT

# Prerequisites check
section "Prerequisites Check"

if [ ! -f "auth.py" ]; then
    echo -e "${RED}Error: auth.py not found${NC}"
    exit 1
fi
pass "auth.py exists"

if [ ! -f "signing-server.py" ]; then
    echo -e "${RED}Error: signing-server.py not found${NC}"
    exit 1
fi
pass "signing-server.py exists"

if [ ! -f "generate-token.py" ]; then
    echo -e "${RED}Error: generate-token.py not found${NC}"
    exit 1
fi
pass "generate-token.py exists"

# Setup GPG if needed
if [ ! -d ".gnupg" ]; then
    echo "Setting up GPG key..."
    make setup-gpg > /dev/null 2>&1 || {
        echo -e "${RED}Error: Failed to setup GPG${NC}"
        exit 1
    }
fi
pass "GPG keyring available"

export GNUPGHOME="$(pwd)/.gnupg"
TEST_KEY=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep '^sec' | head -1 | awk '{print $2}' | cut -d'/' -f2)
if [ -z "$TEST_KEY" ]; then
    echo -e "${RED}Error: No GPG key found${NC}"
    exit 1
fi
echo "Test key: $TEST_KEY"
pass "GPG key available: $TEST_KEY"

# Create test configuration files
section "Configuration Setup"

cat > config/authorization.json << EOF
{
  "roles": {
    "test-valid": {
      "allowed_keys": ["$TEST_KEY"],
      "allowed_digest_algos": ["SHA256"],
      "max_requests_per_hour": 10,
      "description": "Valid test role"
    },
    "test-restricted": {
      "allowed_keys": ["FAKE-KEY-NOT-AVAILABLE"],
      "allowed_digest_algos": ["SHA256"],
      "max_requests_per_hour": 10,
      "description": "Restricted role for testing denial"
    },
    "test-no-rate-limit": {
      "allowed_keys": ["$TEST_KEY"],
      "allowed_digest_algos": ["SHA256"],
      "max_requests_per_hour": 0,
      "description": "No rate limit"
    }
  }
}
EOF
pass "Created authorization.json"

cat > config/secrets.json << EOF
{
  "test-valid-client": {
    "secret": "test-secret-valid",
    "description": "Valid test client"
  },
  "test-restricted-client": {
    "secret": "test-secret-restricted",
    "description": "Restricted test client"
  }
}
EOF
chmod 600 config/secrets.json
pass "Created secrets.json"

# Generate test tokens
section "Token Generation"

VALID_TOKEN=$(python3 -c "
from auth import generate_jwt_token
print(generate_jwt_token('test-valid-client', 'test-valid', 'test-secret-valid', 1))
")
pass "Generated valid token"

RESTRICTED_TOKEN=$(python3 -c "
from auth import generate_jwt_token
print(generate_jwt_token('test-restricted-client', 'test-restricted', 'test-secret-restricted', 1))
")
pass "Generated restricted token"

EXPIRED_TOKEN=$(python3 -c "
from auth import generate_jwt_token
print(generate_jwt_token('test-valid-client', 'test-valid', 'test-secret-valid', -1))
")
pass "Generated expired token"

INVALID_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
pass "Created invalid token"

# Start authenticated server
section "Server Startup"

python3 signing-server.py \
    --port 18082 \
    --keyring .gnupg \
    --enable-auth \
    --secrets-file config/secrets.json \
    --authz-config config/authorization.json \
    --audit-log test-auth-audit.log \
    > server-test-auth.log 2>&1 &
SERVER_PID=$!

sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Error: Server failed to start${NC}"
    cat server-test-auth.log
    exit 1
fi
pass "Server started (PID: $SERVER_PID)"

# Helper function to make signing request
sign_request() {
    local token="$1"
    local key_id="$2"

    python3 << ENDPY
import sys
import json
import hashlib
import base64
from urllib.request import Request, urlopen
from urllib.error import HTTPError

data = b"test data for signing"
digest = hashlib.sha256(data).digest()

payload = {
    'digest': base64.b64encode(digest).decode('ascii'),
    'key_id': '$key_id',
    'digest_algo': 'SHA256',
    'armor': False
}

headers = {
    'Content-Type': 'application/json'
}

if '$token':
    headers['Authorization'] = 'Bearer $token'

request = Request(
    'http://localhost:18082/sign',
    data=json.dumps(payload).encode('utf-8'),
    headers=headers,
    method='POST'
)

try:
    with urlopen(request, timeout=10) as response:
        print(response.status)
        sys.exit(0)
except HTTPError as e:
    print(e.code)
    sys.exit(1)
except Exception as e:
    print("ERROR: " + str(e))
    sys.exit(2)
ENDPY
}

# Test 1: No token
section "Test 1: No Authentication Token"
TESTS_RUN=$((TESTS_RUN + 1))
response=$(sign_request "" "$TEST_KEY" 2>&1 || true)
if echo "$response" | grep -q "401"; then
    pass "Request without token rejected (401)"
else
    fail "Expected 401, got: $response"
fi

# Test 2: Invalid token
section "Test 2: Invalid Token"
TESTS_RUN=$((TESTS_RUN + 1))
response=$(sign_request "$INVALID_TOKEN" "$TEST_KEY" 2>&1 || true)
if echo "$response" | grep -q "401"; then
    pass "Invalid token rejected (401)"
else
    fail "Expected 401, got: $response"
fi

# Test 3: Expired token
section "Test 3: Expired Token"
TESTS_RUN=$((TESTS_RUN + 1))
response=$(sign_request "$EXPIRED_TOKEN" "$TEST_KEY" 2>&1 || true)
if echo "$response" | grep -q "401"; then
    pass "Expired token rejected (401)"
else
    fail "Expected 401, got: $response"
fi

# Test 4: Valid token, wrong key (authorization failure)
section "Test 4: Valid Token, Unauthorized Key"
TESTS_RUN=$((TESTS_RUN + 1))
response=$(sign_request "$RESTRICTED_TOKEN" "$TEST_KEY" 2>&1 || true)
if echo "$response" | grep -q "403"; then
    pass "Unauthorized key rejected (403)"
else
    fail "Expected 403, got: $response"
fi

# Test 5: Valid token, correct key (success)
section "Test 5: Valid Token, Authorized Key"
TESTS_RUN=$((TESTS_RUN + 1))
response=$(sign_request "$VALID_TOKEN" "$TEST_KEY" 2>&1 || true)
if echo "$response" | grep -q "200"; then
    pass "Authorized request succeeded (200)"
else
    fail "Expected 200, got: $response"
fi

# Test 6: Rate limiting
section "Test 6: Rate Limiting"
echo "Making 10 requests (limit is 10/hour)..."
for i in {1..10}; do
    sign_request "$VALID_TOKEN" "$TEST_KEY" > /dev/null 2>&1 || true
done

TESTS_RUN=$((TESTS_RUN + 1))
response=$(sign_request "$VALID_TOKEN" "$TEST_KEY" 2>&1 || true)
if echo "$response" | grep -q "429"; then
    pass "Rate limit enforced (429 after 10 requests)"
else
    fail "Expected 429 (rate limited), got: $response"
fi

# Test 7: Audit log verification
section "Test 7: Audit Log Verification"

if [ ! -f "test-auth-audit.log" ]; then
    fail "Audit log not created"
else
    pass "Audit log created"

    # Check for different action types
    TESTS_RUN=$((TESTS_RUN + 1))
    if grep -q '"action": "AUTH_FAILED"' test-auth-audit.log; then
        pass "Audit log contains AUTH_FAILED entries"
    else
        fail "Audit log missing AUTH_FAILED entries"
    fi

    TESTS_RUN=$((TESTS_RUN + 1))
    if grep -q '"action": "DENIED"' test-auth-audit.log; then
        pass "Audit log contains DENIED entries"
    else
        fail "Audit log missing DENIED entries"
    fi

    TESTS_RUN=$((TESTS_RUN + 1))
    if grep -q '"action": "SIGNED"' test-auth-audit.log; then
        pass "Audit log contains SIGNED entries"
    else
        fail "Audit log missing SIGNED entries"
    fi

    TESTS_RUN=$((TESTS_RUN + 1))
    if grep -q '"action": "RATE_LIMITED"' test-auth-audit.log; then
        pass "Audit log contains RATE_LIMITED entries"
    else
        fail "Audit log missing RATE_LIMITED entries"
    fi

    # Display sample entries
    echo ""
    echo "Sample audit log entries:"
    echo "  AUTH_FAILED: $(grep -c '"action": "AUTH_FAILED"' test-auth-audit.log || echo 0)"
    echo "  DENIED:      $(grep -c '"action": "DENIED"' test-auth-audit.log || echo 0)"
    echo "  SIGNED:      $(grep -c '"action": "SIGNED"' test-auth-audit.log || echo 0)"
    echo "  RATE_LIMITED: $(grep -c '"action": "RATE_LIMITED"' test-auth-audit.log || echo 0)"
fi

# Summary
section "Test Summary"
echo ""
echo "Tests Run:    $TESTS_RUN"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
else
    echo "Tests Failed: $TESTS_FAILED"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All authentication tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some tests failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
