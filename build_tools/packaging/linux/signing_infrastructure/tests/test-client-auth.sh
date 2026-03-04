#!/bin/bash
# Test gpgshim client authentication support

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Testing gpgshim client authentication support"
echo "=============================================="

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check prerequisites
if [ ! -f "gpgshim" ]; then
    echo "Error: gpgshim not found"
    exit 1
fi

if [ ! -f "auth.py" ]; then
    echo "Error: auth.py not found"
    exit 1
fi

# Setup GPG if needed
if [ ! -d ".gnupg" ]; then
    echo "Setting up test GPG key..."
    make setup-gpg > /dev/null 2>&1
fi

export GNUPGHOME="$(pwd)/.gnupg"
TEST_KEY=$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null | grep '^sec' | head -1 | awk '{print $2}' | cut -d'/' -f2)

echo -e "${YELLOW}Test key: $TEST_KEY${NC}"

# Create test config
cat > config/authorization.json <<EOF
{
  "roles": {
    "test": {
      "allowed_keys": ["$TEST_KEY"],
      "allowed_digest_algos": ["SHA256"],
      "max_requests_per_hour": 100
    }
  }
}
EOF

# Generate test token
echo -e "\n${YELLOW}Generating test token...${NC}"
TEST_TOKEN=$(python3 -c "
from auth import generate_jwt_token
print(generate_jwt_token('test-client', 'test', 'test-secret-for-local-testing', 1))
")
echo "Token: ${TEST_TOKEN:0:50}..."

# Start authenticated server
echo -e "\n${YELLOW}Starting authenticated server...${NC}"
python3 signing-server.py \
    --port 18081 \
    --keyring .gnupg \
    --enable-auth \
    --secrets-file config/secrets.json \
    --authz-config config/authorization.json \
    > server-test.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Error: Server failed to start${NC}"
    cat server-test.log
    exit 1
fi

echo "Server started (PID: $SERVER_PID)"

# Cleanup function
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo -e "\n${YELLOW}Stopping server...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    rm -f server-test.log test-client.sig
}
trap cleanup EXIT

# Test 1: Without token (should fail)
echo -e "\n${YELLOW}TEST 1: Request without token${NC}"
export GPG_SIGNING_SERVER='http://localhost:18081/sign'
unset GPG_SERVER_TOKEN
if echo "test data" | ./gpgshim --detach-sign > test-client.sig 2>&1; then
    echo -e "${RED}✗ FAIL: Should have failed without token${NC}"
else
    echo -e "${GREEN}✓ PASS: Correctly rejected request without token${NC}"
fi

# Test 2: With valid token (should succeed)
echo -e "\n${YELLOW}TEST 2: Request with valid token${NC}"
export GPG_SERVER_TOKEN="$TEST_TOKEN"
if echo "test data" | ./gpgshim --detach-sign -u "$TEST_KEY" > test-client.sig 2>&1; then
    if [ -f "test-client.sig" ] && [ -s "test-client.sig" ]; then
        echo -e "${GREEN}✓ PASS: Successfully signed with valid token${NC}"
        echo "  Signature size: $(stat -f%z test-client.sig 2>/dev/null || stat -c%s test-client.sig) bytes"
    else
        echo -e "${RED}✗ FAIL: No signature generated${NC}"
    fi
else
    echo -e "${RED}✗ FAIL: Should have succeeded with valid token${NC}"
fi

# Test 3: Verify signature
echo -e "\n${YELLOW}TEST 3: Verify signature${NC}"
if echo "test data" | gpg --verify test-client.sig - 2>/dev/null; then
    echo -e "${GREEN}✓ PASS: Signature is valid${NC}"
else
    echo -e "${RED}✗ FAIL: Signature verification failed${NC}"
fi

echo -e "\n${GREEN}Client authentication tests complete!${NC}"
