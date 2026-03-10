# POC Communication Protocol

**Complete HTTP API specification for gpgshim ↔ signing-server communication**

---

## Overview

The POC uses a simple **HTTP POST JSON API** for all signing requests:

```
┌─────────────────┐                           ┌─────────────────┐
│   gpgshim       │   HTTPS POST /sign        │ signing-server  │
│   (client)      │ ─────────────────────────> │   (server)      │
│                 │                            │                 │
│                 │ <───────────────────────── │                 │
│                 │   200 OK + signature       │                 │
└─────────────────┘                           └─────────────────┘
```

**All communication:**
- Protocol: HTTPS (TLS 1.2+)
- Method: POST
- Endpoint: `/sign`
- Content-Type: `application/json`
- Authentication: JWT Bearer token in Authorization header

---

## 1. Signing Request (RPM Header - First Call)

### Client Request (gpgshim)

```http
POST /sign HTTP/1.1
Host: signing-server.example.com:8443
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRfaWQiOiJnaXRodWItYWN0aW9ucy1kZXYiLCJyb2xlIjoiZGV2ZWxvcG1lbnQiLCJpYXQiOjE3MDk0NzIwMDAsImV4cCI6MTcwOTU1ODQwMH0.signature
Content-Type: application/json
User-Agent: gpgshim/2.0
Content-Length: 5432

{
  "data": "mQINBGXxY2ABEADKj8... [base64-encoded RPM header ~4KB]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256",
  "armor": false
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | string | ✅ Yes | Base64-encoded data to sign (RPM header, metadata file, etc.) |
| `key_id` | string | ✅ Yes | GPG key identifier (email or key ID) |
| `digest_algo` | string | No | Hash algorithm: `SHA256`, `SHA512`, `SHA384`, `SHA1` (default: SHA256) |
| `armor` | boolean | No | ASCII-armor the signature (default: false for RPM, true for metadata) |

**Example Data Size:**
- RPM header (first call): ~4KB base64 = ~5.3KB in JSON
- Full RPM (second call): **NOT SENT** - cached signature reused
- DEB Release file: ~2KB base64 = ~2.6KB in JSON
- RPM repomd.xml: ~1KB base64 = ~1.3KB in JSON

### Server Response (signing-server)

```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 412

{
  "signature": "iQIzBAABCAAdFiEE... [base64-encoded GPG signature ~256 bytes]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `signature` | string | Base64-encoded GPG signature (binary or ASCII-armored) |
| `key_id` | string | Key ID that was used for signing (echoed back) |
| `digest_algo` | string | Digest algorithm used (echoed back) |

**Signature Size:**
- Binary signature: ~256 bytes base64 = ~340 bytes in JSON
- ASCII-armored: ~500 bytes base64 = ~670 bytes in JSON

---

## 2. Authentication Flow

### JWT Token Structure

The JWT token in the `Authorization: Bearer <token>` header has three parts:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9  ← Header (base64url)
.
eyJjbGllbnRfaWQiOiJnaXRodWItYWN0aW9ucy1kZXYiLCJyb2xlIjoiZGV2ZWxvcG1lbnQiLCJpYXQiOjE3MDk0NzIwMDAsImV4cCI6MTcwOTU1ODQwMH0  ← Payload
.
dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk  ← Signature (HMAC-SHA256)
```

**Decoded Header:**
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Decoded Payload:**
```json
{
  "client_id": "github-actions-dev",
  "role": "development",
  "iat": 1709472000,
  "exp": 1709558400
}
```

**Signature:**
- Algorithm: HMAC-SHA256
- Secret: Per-client secret from `config/secrets.json`
- Validates: Token hasn't been tampered with

### Token Generation

```bash
# Using tools/generate-token.py
./tools/generate-token.py --generate \
  --client-id "github-actions-dev" \
  --role "development" \
  --secret-file config/secrets.json \
  --expires-hours 24

# Output:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRfaWQiOiJnaXRodWItYWN0aW9ucy1kZXYiLCJyb2xlIjoiZGV2ZWxvcG1lbnQiLCJpYXQiOjE3MDk0NzIwMDAsImV4cCI6MTcwOTU1ODQwMH0.signature
```

### Authentication Process

1. **Client includes token** in `Authorization: Bearer <token>` header
2. **Server extracts token** from header
3. **Server decodes payload** to get `client_id`
4. **Server looks up secret** for `client_id` in `config/secrets.json`
5. **Server verifies signature** using HMAC-SHA256 with client secret
6. **Server checks expiration** (`exp` < current time)
7. **Server extracts role** for authorization checks

---

## 3. Authorization Checks

After authentication, server performs role-based authorization:

### Step 1: Check Role Permissions

```json
// config/authorization.json
{
  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000
    }
  }
}
```

**Checks performed:**
1. ✅ Is `key_id` in `allowed_keys` for this role?
2. ✅ Is `digest_algo` in `allowed_digest_algos` for this role?
3. ✅ Has client exceeded `max_requests_per_hour`?

### Step 2: Rate Limiting

Server tracks requests per client:

```python
# In-memory tracking (lost on restart)
{
  "github-actions-dev": [
    1709472000,  # timestamp of request 1
    1709472015,  # timestamp of request 2
    1709472030,  # timestamp of request 3
    ...
  ]
}
```

**Algorithm:** Sliding window (last 3600 seconds)
- Remove timestamps older than 1 hour
- Count remaining timestamps
- If count >= `max_requests_per_hour`, return 429 Too Many Requests

---

## 4. Error Responses

### 401 Unauthorized (Invalid/Missing Token)

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "error": "Unauthorized: Invalid or missing token"
}
```

**Causes:**
- No `Authorization` header
- Invalid JWT format (not 3 parts)
- Invalid signature (token tampered with)
- Token expired (`exp` < current time)
- Unknown `client_id` (not in secrets.json)

### 403 Forbidden (Authorization Failed)

```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{
  "error": "Forbidden: Key 'therock-release@amd.com' not allowed for role 'development'"
}
```

**Causes:**
- `key_id` not in `allowed_keys` for role
- `digest_algo` not in `allowed_digest_algos` for role
- Role doesn't exist in authorization.json

### 429 Too Many Requests (Rate Limit)

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "error": "Rate limit exceeded"
}
```

**Causes:**
- Client has exceeded `max_requests_per_hour` for their role

### 413 Payload Too Large

```http
HTTP/1.1 413 Payload Too Large
Content-Type: application/json

{
  "error": "Request too large: 50000 bytes (max 10240 bytes)"
}
```

**Causes:**
- `Content-Length` exceeds `MAX_REQUEST_SIZE` (default 10KB)
- Prevents DoS attacks via large payloads

### 408 Request Timeout

```http
HTTP/1.1 408 Request Timeout
Content-Type: application/json

{
  "error": "Request timeout"
}
```

**Causes:**
- Client takes too long to send request body (slowloris attack prevention)
- Read timeout exceeded (default 10 seconds)

### 503 Service Unavailable (Server Busy)

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": "Server busy, try again later"
}
```

**Causes:**
- All signing threads are busy (max concurrent requests reached)
- Client should retry with exponential backoff

**Client retry logic (gpgshim):**
```python
# Exponential backoff for 503
retry_delays = [0.1s, 0.2s, 0.4s, 0.8s, 1.6s]
for attempt in range(5):
    response = post_to_server()
    if response.status == 503:
        sleep(retry_delays[attempt] + random(0, 0.1))
    else:
        break
```

### 500 Internal Server Error

```http
HTTP/1.1 500 Internal Server Error
Content-Type: application/json

{
  "error": "Signing failed"
}
```

**Causes:**
- GPG signing command failed (wrong key, missing key, GPG error)
- Server internal error

---

## 5. Complete Example: RPM Signing Flow

### Step 1: rpmsign calls gpgshim (first time - header)

```bash
# rpmsign invokes gpgshim as GPG replacement
rpmsign --define "_gpg_path /path/to/gpgshim" --addsign package.rpm

# gpgshim receives RPM header on stdin (~4KB)
```

### Step 2: gpgshim sends header to signing server

```http
POST /sign HTTP/1.1
Host: signing.example.com:8443
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
  "data": "mQINBGX... [4KB base64]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256",
  "armor": false
}
```

### Step 3: Server validates and signs

```
Server Process:
1. Extract JWT token → Verify signature → Check expiration
2. Decode payload → Get client_id="github-actions-dev", role="development"
3. Check authorization → Is therock-dev@amd.com allowed for development? YES
4. Check rate limit → 47 requests in last hour < 10000? YES
5. Decode base64 data → Write to temp file
6. Run GPG: gpg --batch --no-tty --digest-algo SHA256 --local-user therock-dev@amd.com --detach-sign temp_file
7. Read signature from GPG output → Encode to base64
8. Log audit entry → Return response
```

### Step 4: Server responds with signature

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "signature": "iQIzBAABCA... [256 bytes base64]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256"
}
```

### Step 5: gpgshim caches signature by ppid

```bash
# Cache signature for second rpmsign call
echo "iQIzBAABCA..." | base64 -d > /tmp/gpgshim-cache-12345.sig
# (12345 = parent process ID of rpmsign)
```

### Step 6: rpmsign calls gpgshim again (second time - full RPM)

```bash
# rpmsign invokes gpgshim again with full RPM (~1GB)
# gpgshim receives 1GB RPM on stdin
```

### Step 7: gpgshim uses cached signature (NO SERVER CALL!)

```bash
# Check for cached signature
if [ -f /tmp/gpgshim-cache-12345.sig ]; then
    # Use cached signature (don't send 1GB to server!)
    cat /tmp/gpgshim-cache-12345.sig
    rm /tmp/gpgshim-cache-12345.sig
fi
```

### Step 8: rpmsign embeds signature in RPM

```bash
# rpmsign uses signature from both calls to sign package
# Package is now signed!
```

**Total network transfer:**
- First call: 4KB (header) → server
- Response: 256 bytes (signature) ← server
- Second call: **0 bytes** (cached)
- **Total: ~4.3KB instead of 1GB+**

---

## 6. Complete Example: DEB Metadata Signing

### Step 1: upload_package_repo.py generates Release file

```bash
# Create Release file with checksums
MD5Sum:
 d41d8cd98f00b204e9800998ecf8427e 1234 pool/main/rocm-hip_1.0_amd64.deb
SHA256:
 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 1234 pool/main/rocm-hip_1.0_amd64.deb
```

### Step 2: Call signing server for InRelease (clearsigned)

```http
POST /sign HTTP/1.1
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
  "data": "TUQ1U3VtOgogZDQxZDh... [base64-encoded Release file ~2KB]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256",
  "armor": true,
  "clearsign": true
}
```

**Note:** `clearsign: true` creates inline signature (InRelease format)

### Step 3: Server response (clearsigned)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "signature": "LS0tLS1CRUdJTiBQR1AgU0lHTkVE... [clearsigned Release with signature]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256"
}
```

Decoded signature (InRelease format):
```
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

MD5Sum:
 d41d8cd98f00b204e9800998ecf8427e 1234 pool/main/rocm-hip_1.0_amd64.deb
SHA256:
 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 1234 pool/main/rocm-hip_1.0_amd64.deb
-----BEGIN PGP SIGNATURE-----

iQIzBAEBCAAdFiEE...signature...
-----END PGP SIGNATURE-----
```

### Step 4: Call signing server for Release.gpg (detached)

```http
POST /sign HTTP/1.1
Authorization: Bearer eyJhbGci...
Content-Type: application/json

{
  "data": "TUQ1U3VtOgogZDQxZDh... [same Release file]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256",
  "armor": true,
  "clearsign": false
}
```

**Note:** `clearsign: false`, `armor: true` creates detached ASCII signature

### Step 5: Server response (detached signature)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "signature": "LS0tLS1CRUdJTiBQR1AgU0lHTkFUVVJF... [detached ASCII signature]",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256"
}
```

Decoded signature (Release.gpg format):
```
-----BEGIN PGP SIGNATURE-----

iQIzBAABCAAdFiEE...signature...
-----END PGP SIGNATURE-----
```

---

## 7. Server Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/sign` | POST | Sign data (main endpoint) |
| `/quit` | POST | Gracefully shutdown server (testing only) |
| `/health` | GET | Health check (optional, not in POC) |
| `/metrics` | GET | Prometheus metrics (optional, not in POC) |

**Production endpoints to add:**
```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'version': '1.0.0'}

@app.route('/metrics')
def metrics():
    # Return Prometheus metrics
    pass
```

---

## 8. Configuration Files

### config/secrets.json

```json
{
  "clients": {
    "github-actions-dev": {
      "secret": "AzKjX9... [base64-encoded 32-byte secret]",
      "description": "GitHub Actions - development builds",
      "created": "2026-03-02"
    },
    "github-actions-nightly": {
      "secret": "BqL8Y2...",
      "description": "GitHub Actions - nightly builds",
      "created": "2026-03-02"
    }
  }
}
```

**Secret generation:**
```bash
# Generate cryptographically secure secret
python3 -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"
```

### config/authorization.json

```json
{
  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000
    },
    "nightly": {
      "allowed_keys": ["therock-nightly@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 5000
    },
    "release": {
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000
    }
  }
}
```

---

## 9. Testing the Protocol

### Using curl

```bash
# Generate test token
TOKEN=$(./tools/generate-token.py --generate \
  --client-id "github-actions-dev" \
  --role "development" \
  --secret-file config/secrets.json \
  --expires-hours 1)

# Test signing request
echo "test data" | base64 | jq -R '{data: ., key_id: "therock-dev@amd.com", digest_algo: "SHA256", armor: true}' | \
  curl -X POST https://signing-server.example.com/sign \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d @-

# Expected response:
{
  "signature": "LS0tLS1CRUdJTi...",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256"
}
```

### Using Python

```python
import requests
import base64
import json

# Read data to sign
with open('package.rpm', 'rb') as f:
    data = f.read()

# Prepare request
payload = {
    'data': base64.b64encode(data).decode('ascii'),
    'key_id': 'therock-dev@amd.com',
    'digest_algo': 'SHA256',
    'armor': False
}

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

# Send request
response = requests.post(
    'https://signing-server.example.com/sign',
    json=payload,
    headers=headers,
    timeout=60
)

# Parse response
result = response.json()
signature = base64.b64decode(result['signature'])

# Save signature
with open('package.rpm.sig', 'wb') as f:
    f.write(signature)
```

---

## Summary

**Communication is dead simple:**
1. Client sends base64-encoded data + JWT token
2. Server verifies token, checks authorization
3. Server signs data with GPG
4. Server returns base64-encoded signature
5. **Optimization:** gpgshim caches signature by ppid to avoid sending full RPM

**Key advantages:**
- ✅ Standard HTTP/JSON (easy to debug)
- ✅ JWT authentication (standard, secure)
- ✅ Zero dependencies (stdlib only)
- ✅ Works with any HTTP client
- ✅ Network transfer: 4KB instead of 1GB+ (250x reduction)
