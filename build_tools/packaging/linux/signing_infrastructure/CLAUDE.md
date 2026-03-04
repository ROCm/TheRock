# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPG shim for remote RPM signing - a lightweight Python 3.6+ tool that intercepts `gpg` calls from `rpmsign` and forwards signing requests to a remote server via REST API.

**Design Goals:**
- Zero dependencies (standard library only)
- Python 3.6 compatibility for older RHEL/CentOS systems
- Minimal data transfer to remote server
- Fast startup without virtual environments

## Key Files

- `gpgshim` - Main executable that acts as gpg replacement
- `setup.sh` - Installation script
- `server-example.py` - Reference implementation of signing server API
- `README.md` - User documentation

## Architecture

### Data Flow
1. `rpmsign` calls `gpgshim` instead of `gpg`, passing data on stdin
2. Shim reads data from stdin (could be 50KB header or 1GB full RPM)
3. **Shim computes digest locally** using specified algorithm (SHA256/SHA512/etc)
4. Shim checks cache for this digest to avoid redundant requests
5. Sends only the digest (32 bytes for SHA256) to remote server via HTTP POST
6. If server returns 503 (busy), **retry with exponential backoff**
   - Initial delay: 0.1s, doubles each retry, max 10s
   - Random jitter added to prevent thundering herd
   - Up to 5 retries by default
7. Server signs the digest using GPG and returns signature
8. Shim caches result and writes signature to stdout for rpmsign

**Key insight:** rpmsign may call gpg multiple times:
- First: Sign just the RPM header (~50KB)
- Second: Sign the full RPM (could be 1GB+)

By computing the digest locally, we only send 32 bytes per call instead of the full data size.

### Deduplication Strategy
rpmsign makes multiple gpg calls with different data:
- Call 1: RPM header (different digest)
- Call 2: Full RPM including header (different digest)

The shim handles this by:
- Computing digest of the actual input data
- Caching by digest hash (not input data hash)
- Only identical digests are deduplicated (rare in practice)
- Primary optimization is sending digest instead of full data, not deduplication

### Remote Server API

**Endpoint:** `POST /sign`

**Request:**
```json
{
  "digest": "base64-encoded-digest",
  "key_id": "string",
  "digest_algo": "SHA256|SHA512|etc",
  "armor": boolean
}
```

The `digest` field contains the pre-computed hash (32 bytes for SHA256, 64 for SHA512, etc), NOT the original data.

**Response:**
```json
{
  "signature": "base64-encoded-signature",
  "key_id": "actual-key-used"
}
```

## Performance

### Benchmarking
```bash
make start-server-bg
make benchmark
```

Typical performance (RSA 2048-bit key):
- **Sequential**: 20-40 signatures/second
- **Concurrent (10 threads)**: 100-200 signatures/second
- **Per-signature latency**: 25-50ms

Factors affecting performance:
- GPG key type/size (RSA 2048 vs 4096, vs Ed25519)
- CPU performance
- Disk I/O (keyring access)
- Network latency (if remote server)

With 10 concurrent threads and 50ms per signature:
- **Theoretical max**: 200 req/s
- **Practical max**: 150-180 req/s (with overhead)

## Development Commands

### Quick Testing with Makefile
```bash
# Setup test environment (one-time)
make setup-gpg              # Creates test GPG key in .gnupg/

# Start server
make start-server           # Foreground
make start-server-bg        # Background

# Run tests
make test-shim              # Small data test
make test-large             # 100MB test (simulates 1GB RPM)
make test-rpm               # Actual RPM signing test
make test-all               # All tests (auto start/stop server)

# Cleanup
make stop-server            # Stop via /quit endpoint
make clean                  # Remove test artifacts
make clean-gpg              # Remove test GPG keys
```

### Verifying rpmsign uses the shim
The `test-rpm` target verifies the shim is actually called by:
1. Stripping existing signatures with `rpmsign --delsign`
2. Setting `GPGSHIM_MARKER` environment variable
3. Calling rpmsign with `--define "_gpg_path $(pwd)/gpgshim"`
4. Checking that the marker file was created (proves shim was invoked)
5. Verifying the signature in the RPM matches our test key

### Manual Testing
```bash
# Create test key
make setup-gpg

# Start server (terminal 1)
export GNUPGHOME=$(pwd)/.gnupg
./signing-server.py --port 8080 --key $(make get-key-id)

# Test (terminal 2)
export GNUPGHOME=$(pwd)/.gnupg
export GPG_SIGNING_SERVER='http://localhost:8080/sign'
echo "test" | ./gpgshim --detach-sign --armor > test.sig

# Test large file - only ~300 bytes transferred despite 100MB input!
dd if=/dev/urandom bs=1M count=100 | ./gpgshim -b > test.sig
```

### Installation
```bash
./setup.sh  # Installs to ~/.local/bin
./setup.sh /custom/path  # Custom location
```

### Usage with rpmsign
```bash
export GPG_SIGNING_SERVER='http://signing-server:8080/sign'
export PATH="$HOME/.local/bin:$PATH"
rpmsign --addsign mypackage.rpm
```

## Code Constraints

### Python 3.6 Compatibility
- No f-strings (use `.format()` or `%` formatting)
- No type hints (Python 3.6 has limited support)
- No `|` dict merge operator (Python 3.9+)
- Use `urllib.request` not `requests` for HTTP (standard library)

### Standard Library Only
The shim uses only:
- `urllib.request`, `urllib.error` - HTTP client
- `json` - JSON parsing
- `base64` - Encoding/decoding
- `hashlib` - SHA256 for deduplication
- `sys`, `os` - System interaction

No external dependencies to keep startup fast and avoid pip/venv in cloud environments.

## Security Features

### Server-side Protection
- **Multi-threaded architecture**: Uses `ThreadingMixIn` for concurrent request handling
  - Each signing request handled in separate thread
  - Prevents one slow request from blocking others
- **Thread limit**: Default 10 concurrent threads (configurable via `--max-threads`)
  - Uses semaphore to limit concurrent signing operations
  - Returns HTTP 503 if server is at capacity
  - Prevents resource exhaustion from too many concurrent requests
- **Request size limit**: Default 10KB (configurable via `--max-request-size`)
  - Typical request: ~500 bytes (32-byte digest + metadata)
  - Prevents DoS attacks via oversized payloads
- **Read timeout**: Default 10 seconds (configurable via `--read-timeout`)
  - Prevents slowloris-style attacks where client sends data slowly
- **Request logging**: All requests logged with byte count for audit trail
- Socket timeout enforced during read to prevent slow-walking attacks

## Common Modifications

### Adding Authentication
Add API key to request headers in `sign_remote()`:
```python
headers={
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}
```

### Supporting HTTPS with Custom CA
```python
import ssl
context = ssl.create_default_context(cafile='/path/to/ca.crt')
urlopen(request, timeout=self.timeout, context=context)
```

### Adding Retry Logic
Wrap the `urlopen` call with retry logic for transient failures.

### Logging
Add logging to `/var/log/gpgshim.log` or syslog for audit trail.
