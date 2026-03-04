# GPG Shim for Remote RPM Signing

A lightweight Python-based shim that intercepts `gpg` calls from `rpmsign` and forwards signing requests to a remote signing server via REST API.

## Project Structure

```
/
  gpgshim                    # Main client executable (drop-in gpg replacement)
  server/
    signing-server.py        # Production signing server
    auth.py                  # Authentication/authorization module
    server-example.py        # Simple reference implementation
  tests/
    test-*.sh                # Test scripts
    benchmark.sh             # Performance benchmarking
  tools/
    setup.sh                 # Installation script
    generate-token.py        # JWT token generation utility
  config/
    authorization.json       # Authorization rules example
    secrets.json.example     # JWT secrets template
  docs/                      # Comprehensive guides
  Makefile                   # Test automation
  README.md                  # This file
```

## Features

- **Zero dependencies**: Uses only Python 3.6+ standard library
- **Minimal data transfer**: Sends RPM header (~4KB) to server, reuses signature for full RPM via ppid cache
- **Deduplication**: Caches signatures for identical data within a session
- **Drop-in replacement**: Compatible with rpmsign's gpg invocation
- **Fast startup**: No virtual environment needed for basic operation

## Problem Statement

`rpmsign` calls `gpg` twice to sign RPMs: first with the header (~4KB), then with the full RPM. For large RPMs (e.g., 1GB LLVM package), sending the full file over the network is impractical.

This solution minimizes network transfer using **ppid-based caching**:
1. First call: shim sends RPM header (~4KB) to server, receives signature, caches it by ppid
2. Second call: shim reads full RPM (1GB) from stdin but uses cached signature instead of sending to server
3. Remote server only processes the ~4KB header, returns signature (~256 bytes)

**Network savings:** Instead of 1GB+, we transfer only ~4KB total (just the header).

## Installation

```bash
./tools/setup.sh
# Or specify custom installation directory:
# ./tools/setup.sh /usr/local/bin
```

## Configuration

Set environment variables before using:

```bash
export GPG_SIGNING_SERVER='http://your-signing-server:8080/sign'
export GPG_KEY_ID='your-key-id'  # Optional
export GPG_TIMEOUT='30'  # Optional, default 30 seconds

# Retry configuration (for handling server busy responses)
export GPG_MAX_RETRIES='5'         # Default 5 retries
export GPG_INITIAL_BACKOFF='0.1'   # Default 0.1s initial backoff
export GPG_MAX_BACKOFF='10.0'      # Default 10s max backoff
```

### Retry Behavior

The shim implements **exponential backoff** when the server returns HTTP 503 (busy):
- Initial retry delay: 0.1s (configurable)
- Each retry doubles the delay (exponential)
- Max delay capped at 10s (configurable)
- Adds random jitter (0-10% of delay) to prevent thundering herd
- Default: up to 5 retries before giving up

Example retry sequence:
1. First 503: wait 0.1s
2. Second 503: wait 0.2s
3. Third 503: wait 0.4s
4. Fourth 503: wait 0.8s
5. Fifth 503: wait 1.6s
6. Give up after 5th retry

## Usage

### Option 1: RPM macro
```bash
rpmsign --define '_gpg_path /path/to/gpgshim' --addsign mypackage.rpm
```

### Option 2: RPM macro
```bash
echo '%_gpg_path /path/to/gpgsign' >> ~/.rpmmacros
rpmsign /path/to/gpgshim' --addsign mypackage.rpm
```

## Remote Server API

The signing server must implement a REST endpoint that accepts POST requests:

### Request Format
```json
{
  "data": "base64-encoded-data",
  "key_id": "signing-key-identifier",
  "digest_algo": "SHA256",
  "armor": false
}
```

**Note:** The `data` field contains base64-encoded data (RPM header on first call, ~4KB typically). The server computes the digest and signs it.

**Required:** The `key_id` field must be provided in every request. The server will use this to select which signing key to use from its keyring.

### Response Format
```json
{
  "signature": "base64-encoded-signature",
  "key_id": "actual-key-id-used"
}
```

### Error Response
```json
{
  "error": "error message"
}
```

## How It Works

1. `rpmsign` invokes the shim instead of `gpg` (twice: header, then full RPM)
2. **First call (header):**
   - Shim reads RPM header from stdin (~4KB)
   - Sends header to remote server via REST API
   - Server computes digest and signs it using GPG
   - Server returns signature (~256 bytes)
   - Shim caches signature by ppid to `/tmp/gpgshim-cache-{ppid}.sig`
3. **Second call (full RPM):**
   - Shim reads full RPM from stdin (up to 1GB+) to consume rpmsign's pipe
   - Uses cached signature from first call (doesn't send data to server)
   - Deletes cache file after use
4. Shim writes signature to stdout for both calls
5. `rpmsign` embeds the signatures in the RPM

**Key optimization:** A 1GB RPM results in only ~4KB network transfer (just the header from first call). The ppid-based cache prevents sending the full RPM on the second call.

## Python 3.6 Compatibility

The code uses only features available in Python 3.6:
- Standard library modules (urllib, json, hashlib)
- Compatible string formatting
- No f-strings or other modern syntax

## Security Considerations

- **TLS**: Use HTTPS for the signing server in production
- **Authentication**: Add API key/token authentication to the server
- **Audit**: Log all signing requests on the server side
- **Network**: Run on a secure network or use VPN/tunnel

## Testing

### Quick Start with Makefile

```bash
# One-time setup: create test GPG key
make setup-gpg

# Terminal 1: Start signing server
make start-server

# Terminal 2: Run tests
make test-shim        # Test with small data
make test-large       # Test with 100MB data (simulates 1GB RPM)
make test-rpm         # Sign actual RPM with rpmsign

# Or run all tests automatically
make test-all         # Starts server, runs tests, stops server

# Stop server
make stop-server      # Uses /quit endpoint
```

The repository includes `test-package.rpm` (11KB) for testing actual RPM signing. The test automatically:
- Strips any existing signatures from the test copy
- Signs with your test key via the shim
- Verifies the shim was actually invoked (not system gpg)
- Confirms the signature matches your test key

### Manual Testing

```bash
# Start the signing server
python3 server/signing-server.py --port 8080 --key YOURKEYID

# With security options
python3 server/signing-server.py --port 8080 --key YOURKEYID \
  --max-request-size 8192 --read-timeout 5

# In another terminal
echo "test data" | GPG_SIGNING_SERVER='http://localhost:8080/sign' ./gpgshim --detach-sign --armor > signature.asc

# Test with large file (100MB simulating 1GB RPM)
dd if=/dev/urandom bs=1M count=100 | \
  GPG_SIGNING_SERVER='http://localhost:8080/sign' \
  ./gpgshim --detach-sign > signature.sig

# Stop server via /quit endpoint
curl -X POST http://localhost:8080/quit
```

**Security features:**
- Multi-threaded server handles concurrent requests
- Request size limit (default: 10KB) - prevents DoS via large payloads
- Read timeout (default: 10 seconds) - prevents slowloris attacks
- Thread limit (default: 10) - prevents resource exhaustion
- All requests logged with byte counts for auditing

### Test with actual RPM
```bash
export GPG_SIGNING_SERVER='http://localhost:8080/sign'
export GPG_KEY_ID='YOURKEYID'
rpmsign --addsign your-package.rpm
```

## Troubleshooting

### Debug Mode

Enable debug logging to see what arguments rpmsign passes:

```bash
export GPG_SHIM_DEBUG=1
make test-rpm
```

Or run the debug test suite:

```bash
make test-debug
```

### Common Issues

- **No response**: Check `GPG_SIGNING_SERVER` is set and accessible
- **Timeout**: Increase `GPG_TIMEOUT` for slow networks
- **Format errors**: Verify server returns JSON with base64-encoded signature
- **rpmsign errors**: Use `rpmsign -vv` for verbose output
- **Wrong key**: Check that `--define "_gpg_name KEYID"` matches your signing key
- **Signature verification fails**: Ensure GNUPGHOME is set correctly for rpm tools

## License

Public domain / Unlicense

## Authentication and Authorization

The signing server supports JWT-based authentication with role-based authorization for production deployments.

### Quick Setup

**For Production:**
```bash
# 1. Generate client secret
./tools/generate-token.py --generate-secret --client-id github-actions-prod

# 2. Add to config/secrets.json
# 3. Configure roles in config/authorization.json

# 4. Generate token (30 days)
./tools/generate-token.py --generate \
    --client-id github-actions-prod \
    --role production \
    --secret-file config/secrets.json \
    --expires-hours 720

# 5. Use token in client
export GPG_SERVER_TOKEN='eyJhbGci...'
export GPG_SIGNING_SERVER='https://signing-server:8443/sign'
```

### Architecture

- **Authentication**: HMAC-SHA256 signed JWT tokens
- **Authorization**: Role-based key access control
- **Rate Limiting**: Per-client request throttling
- **Audit Logging**: All requests logged with client_id, role, key_id, timestamp
- **TLS/HTTPS**: Optional TLS encryption for transport security

### Server Configuration

Start server with authentication:

```bash
./server/signing-server.py \
    --port 8443 \
    --keyring /path/to/.gnupg \
    --enable-auth \
    --secrets-file config/secrets.json \
    --authz-config config/authorization.json \
    --enable-tls \
    --cert-file server.crt \
    --key-file server.key
```

### Client Configuration

Configure gpgshim with authentication:

```bash
export GPG_SIGNING_SERVER='https://signing-server:8443/sign'
export GPG_SERVER_TOKEN='eyJhbGci...'  # JWT token
export GPG_SERVER_CA_CERT='/etc/ssl/certs/company-ca.crt'  # Optional
export GPG_VERIFY_SSL='true'  # Default: true

# Use with rpmsign
rpmsign --addsign mypackage.rpm
```

### Role-Based Access Control

Define roles in `config/authorization.json`:

```json
{
  "roles": {
    "production": {
      "allowed_keys": ["prod-key@company.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000
    },
    "development": {
      "allowed_keys": ["dev-key@company.com"],
      "allowed_digest_algos": ["SHA256"],
      "max_requests_per_hour": 5000
    },
    "admin": {
      "allowed_keys": ["prod-key@company.com", "dev-key@company.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 100
    }
  }
}
```

### Token Management

```bash
# Generate token
./tools/generate-token.py --generate \
    --client-id my-client \
    --role production \
    --secret-file config/secrets.json \
    --expires-hours 24

# Validate token
./tools/generate-token.py --validate 'eyJhbGci...' \
    --secret-file config/secrets.json

# Decode token (debugging)
./tools/generate-token.py --decode 'eyJhbGci...'

# Generate new secret
./tools/generate-token.py --generate-secret --client-id new-client
```

### Manual Signing

For administrators who need to manually sign RPMs:

**Don't copy the production key!** Use a personal admin token instead:

```bash
# 1. Get admin token (24 hours)
./tools/generate-token.py --generate \
    --client-id admin-yourname \
    --role admin \
    --secret-file config/secrets.json \
    --expires-hours 24

# 2. Configure environment
export GPG_SERVER_TOKEN='your-admin-token'
export GPG_SIGNING_SERVER='https://signing-server:8443/sign'

# 3. Sign RPMs
rpmsign --addsign mypackage.rpm
```

See [docs/manual-signing.md](docs/manual-signing.md) for detailed workflow.

### TLS/HTTPS Setup

**Self-Signed Certificate (Testing):**
```bash
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt \
    -days 365 -nodes -subj "/CN=signing-server.local"

./server/signing-server.py --enable-tls \
    --cert-file server.crt --key-file server.key
```

**Production with Let's Encrypt:**
```bash
# Use certbot to obtain certificate
certbot certonly --standalone -d signing-server.example.com

./server/signing-server.py --enable-tls \
    --cert-file /etc/letsencrypt/live/signing-server.example.com/fullchain.pem \
    --key-file /etc/letsencrypt/live/signing-server.example.com/privkey.pem
```

**AWS with ALB:**
- Configure ALB for TLS termination
- ALB → HTTP → signing-server (internal network)
- Manage certificates via AWS Certificate Manager

### GitHub Actions Integration

**Setup GitHub Secrets:**
```yaml
# .github/workflows/build.yml
name: Build and Sign RPM

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build RPM
        run: make build-rpm
      
      - name: Sign RPM on Azure
        env:
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
        run: |
          # SSH to Azure build server and sign
          ssh azure-build-server "
            export GPG_SERVER_TOKEN='$GPG_SERVER_TOKEN'
            export GPG_SIGNING_SERVER='$GPG_SIGNING_SERVER'
            rpmsign --addsign /path/to/package.rpm
          "
```

**GitHub Secrets to configure:**
- `GPG_SERVER_TOKEN_PROD` - Production signing token
- `GPG_SERVER_TOKEN_DEV` - Development signing token
- `GPG_SIGNING_SERVER` - Server URL (https://signing-server:8443/sign)

### Audit Logging

All signing requests are logged:

```bash
# View audit log
tail -f /var/log/gpg-signing/audit.log

# Sample entry
{
  "timestamp": "2026-03-02T14:30:45Z",
  "action": "SIGNED",
  "client_id": "github-actions-prod",
  "role": "production",
  "key_id": "prod-key@company.com",
  "digest_algo": "SHA256",
  "client_ip": "10.20.30.40",
  "success": true
}

# Query audit log
grep '"action": "DENIED"' /var/log/gpg-signing/audit.log
grep '"client_id": "admin-alice"' /var/log/gpg-signing/audit.log
```

### Security Best Practices

1. **Always use HTTPS/TLS in production**
2. **Rotate tokens quarterly** or when compromised
3. **Use short-lived tokens** (4-24 hours) for manual signing
4. **Store secrets securely** (Vault, AWS Secrets Manager, GitHub Secrets)
5. **Monitor audit logs** for unauthorized access attempts
6. **Never commit** `config/secrets.json` to version control
7. **Set restrictive file permissions**: `chmod 600 config/secrets.json`
8. **Use role-based access** - don't give everyone admin access
9. **Enable rate limiting** to prevent abuse
10. **Backup secrets encrypted** - you can't recover lost secrets

### Documentation

- [Manual Signing Guide](docs/manual-signing.md) - How to manually sign RPMs without copying keys
- [Configuration Guide](config/README.md) - Detailed configuration file documentation
- [Authentication Architecture](docs/authentication.md) - Token format and security model
- [AWS Deployment](docs/deployment-aws.md) - Deploying to AWS with ALB/EC2
- [GitHub Actions Integration](docs/github-actions-integration.md) - CI/CD workflow examples

