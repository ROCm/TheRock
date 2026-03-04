#!/bin/bash
# Debug script to verify key passing and show what rpmsign sends to gpg

set -e

GNUPGHOME=$(pwd)/.gnupg
export GNUPGHOME

echo "=== Debug Test: Verify Key Passing ==="
echo ""

# Get key ID
KEY_ID=$(gpg --list-secret-keys --keyid-format LONG signer@example.com | grep sec | awk '{print $2}' | cut -d'/' -f2)

echo "Test Key ID: $KEY_ID"
echo "Test Key Email: signer@example.com"
echo ""

# Test 1: Direct shim test with -u
echo "Test 1: Direct shim test with -u flag"
echo "---------------------------------------"
export GPG_SHIM_DEBUG=1
export GPG_SIGNING_SERVER=http://localhost:8080/sign
echo "test data" | ./gpgshim -u "$KEY_ID" --detach-sign --armor > /tmp/test-debug.sig 2>&1 | head -10
echo ""

# Test 2: What does rpmsign actually call?
echo "Test 2: Intercept rpmsign's gpg call"
echo "---------------------------------------"
echo "Creating a logging wrapper to see what rpmsign sends..."

cat > /tmp/gpg-logger <<'EOF'
#!/bin/bash
echo "gpg called with: $@" >> /tmp/gpg-calls.log
echo "stdin data size: $(wc -c | awk '{print $1}') bytes" >> /tmp/gpg-calls.log
echo "---" >> /tmp/gpg-calls.log
EOF

chmod +x /tmp/gpg-logger

echo "" > /tmp/gpg-calls.log

echo "Note: This will fail (no actual signing) but shows the arguments"
cp test-package.rpm /tmp/test-debug.rpm

# Try to sign with our logger
rpmsign --define "_gpg_path /tmp/gpg-logger" \
        --define "_gpg_name $KEY_ID" \
        --addsign /tmp/test-debug.rpm 2>/dev/null || true

echo ""
echo "rpmsign called gpg with these arguments:"
cat /tmp/gpg-calls.log

# Cleanup
rm -f /tmp/gpg-logger /tmp/gpg-calls.log /tmp/test-debug.rpm

echo ""
echo "Test 3: Verify key is in payload to signing server"
echo "----------------------------------------------------"
echo "Start the signing server with debug logging to verify key_id in request"
