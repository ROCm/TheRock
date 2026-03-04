# Authentication Architecture

This document provides detailed technical information about the authentication and authorization system for the GPG remote signing server.

## Overview

The authentication system uses JWT (JSON Web Tokens) with HMAC-SHA256 signatures to provide:
- **Authentication**: Verify client identity using shared secrets
- **Authorization**: Role-based access control (RBAC) for GPG keys and digest algorithms
- **Rate Limiting**: Per-role request quotas with rolling time windows
- **Audit Logging**: JSON-formatted logs of all authentication events

## Token Format Specification

### JWT Structure

Tokens consist of three Base64url-encoded components separated by periods:

```
{header}.{payload}.{signature}
```

### Header
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

- `alg`: Always "HS256" (HMAC with SHA-256)
- `typ`: Always "JWT"

### Payload
```json
{
  "client_id": "github-actions-prod",
  "role": "production",
  "iat": 1709337600,
  "exp": 1709352000
}
```

- `client_id` (string): Unique identifier for the client
- `role` (string): Role name (must exist in authorization.json)
- `iat` (integer): Issued-at timestamp (Unix epoch seconds)
- `exp` (integer): Expiration timestamp (Unix epoch seconds)

### Signature

```
HMAC-SHA256(
  base64url(header) + "." + base64url(payload),
  secret_key
)
```

The signature is computed using the client's secret key from `secrets.json`.

## Authentication Flow

### 1. Token Generation (Client-side)

```bash
./generate-token.py --generate \
    --client-id github-actions-prod \
    --role production \
    --secret-file config/secrets.json \
    --expires-hours 720
```

This produces a JWT token that the client includes in HTTP requests.

### 2. Request Authentication (Server-side)

```
Client Request
    |
    v
Extract "Authorization: Bearer {token}" header
    |
    v
Decode JWT (Base64url decode header + payload)
    |
    v
Load client secret from secrets.json using payload.client_id
    |
    v
Recompute signature using secret
    |
    v
Compare computed signature with token signature
    |
    +-- Match --> Check expiration (exp > current_time)
    |                 |
    |                 +-- Valid --> Proceed to Authorization
    |                 |
    |                 +-- Expired --> HTTP 401 "Token expired"
    |
    +-- No Match --> HTTP 401 "Invalid signature"
```

**Key Security Properties:**
- Server never stores tokens (stateless)
- Only clients with correct secret can generate valid tokens
- Tokens cannot be forged or tampered with
- Expired tokens are rejected even if signature is valid

### 3. Authorization Check

```
Authenticated Request (has valid token with client_id + role)
    |
    v
Load authorization.json
    |
    v
Find role definition (e.g., "production")
    |
    v
Check if requested key_id is in role.allowed_keys
    |
    +-- Yes --> Check if digest_algo is in role.allowed_digest_algos
    |              |
    |              +-- Yes --> Proceed to Rate Limiting
    |              |
    |              +-- No --> HTTP 403 "Digest algorithm not allowed"
    |
    +-- No --> HTTP 403 "Not authorized for this key"
```

**Special Case: Empty allowed_keys List**
- `"allowed_keys": []` means "allow ANY key"
- Use only for emergency/admin roles
- Should have very restrictive rate limits

### 4. Rate Limiting

```
Authorized Request
    |
    v
Get role.max_requests_per_hour (e.g., 1000)
    |
    +-- 0 --> No rate limit, proceed to signing
    |
    +-- > 0 --> Check request history for this client_id
                    |
                    v
                Remove requests older than 1 hour from history
                    |
                    v
                Count remaining requests in past hour
                    |
                    +-- < max_requests_per_hour --> Allow request
                    |                                Add to history
                    |                                Proceed to signing
                    |
                    +-- >= max_requests_per_hour --> HTTP 429 "Rate limit exceeded"
                                                      Log RATE_LIMITED event
```

**Implementation Details:**
- Rolling 1-hour window (not fixed hourly buckets)
- History stored in memory using `collections.deque`
- Each entry: `(client_id, timestamp)`
- Old entries purged on each request check

### 5. Audit Logging

Every authentication event is logged in JSON format:

```json
{
  "timestamp": "2026-03-02T14:30:45Z",
  "action": "SIGNED",
  "client_id": "github-actions-prod",
  "role": "production",
  "key_id": "ABCD1234ABCD1234",
  "digest_algo": "SHA256",
  "client_ip": "192.0.2.15",
  "success": true
}
```

**Action Types:**
- `AUTH_FAILED`: Invalid/missing/expired token
- `DENIED`: Valid token but not authorized for requested key/algo
- `RATE_LIMITED`: Exceeded rate limit
- `SIGNED`: Successful signature operation
- `ERROR`: Server-side error during signing

## Security Best Practices

### Secret Management

**DO:**
- Generate secrets with cryptographic RNG (use `generate-token.py --generate-secret`)
- Use minimum 256-bit (32-byte) secrets
- Store secrets.json with `chmod 600` permissions
- Use separate secrets per environment (prod/dev/test)
- Use secrets manager in production (AWS Secrets Manager, HashiCorp Vault)
- Rotate secrets quarterly or when compromised

**DON'T:**
- Commit secrets.json to version control
- Reuse secrets across clients
- Share secrets via email/chat
- Use predictable secrets (words, dates, etc.)

### Token Lifetime

**Recommended Lifetimes:**
- **CI/CD automation**: 7-30 days (long-lived)
- **Admin tokens**: 4-24 hours (short-lived)
- **Emergency tokens**: 1-4 hours (very short-lived)

**Rationale:**
- Longer lifetimes reduce operational overhead (fewer rotations)
- Shorter lifetimes reduce compromise window
- Balance based on risk and usage patterns

### Role Design

**Principle of Least Privilege:**
- Create narrow roles for specific purposes
- Don't grant access to keys unless necessary
- Use separate roles for prod/dev environments
- Restrict rate limits based on actual usage patterns

**Example: Production CI/CD Role**
```json
{
  "production": {
    "allowed_keys": ["prod-signing-key@company.com"],
    "allowed_digest_algos": ["SHA256"],
    "max_requests_per_hour": 1000,
    "description": "GitHub Actions production builds only"
  }
}
```

**Example: Emergency Break-Glass Role**
```json
{
  "emergency": {
    "allowed_keys": [],
    "allowed_digest_algos": ["SHA256", "SHA384", "SHA512"],
    "max_requests_per_hour": 50,
    "description": "Emergency access - allows any key, heavily rate limited"
  }
}
```

### TLS/HTTPS

**Always use TLS in production:**
- Protects tokens in transit
- Prevents man-in-the-middle attacks
- Validates server identity

**Certificate Options:**
1. **AWS ALB with ACM** (recommended for AWS deployments)
2. **Let's Encrypt** (free, auto-renewal)
3. **Corporate CA** (for internal deployments)

**Client Verification:**
```bash
# Verify server certificate
export GPG_VERIFY_SSL=true

# Custom CA certificate
export GPG_SERVER_CA_CERT=/path/to/ca.crt
```

### Network Security

**Firewall Rules:**
- Allow HTTPS (443) from known client IPs only
- Deny all other inbound traffic
- Use security groups (AWS) or iptables (on-prem)

**Example AWS Security Group:**
```
Inbound:
  - Port 443, Source: GitHub Actions IP ranges
  - Port 443, Source: Azure build server IPs
  - Port 22 (SSH), Source: Admin VPN only

Outbound:
  - Allow all (for APT updates, etc.)
```

## Credential Rotation Procedures

### Quarterly Rotation (Planned)

Rotate secrets every 90 days as preventive measure.

**Steps:**
1. **Generate new secret:**
   ```bash
   ./generate-token.py --generate-secret --client-id github-actions-prod
   ```

2. **Add to secrets.json (keep old secret temporarily):**
   ```json
   {
     "github-actions-prod": {
       "secret": "NEW-SECRET-HERE",
       "secret_old": "OLD-SECRET-HERE",
       "description": "Production GitHub Actions",
       "rotated": "2026-03-02"
     }
   }
   ```

3. **Generate new tokens with new secret**
4. **Update GitHub Secrets / CI/CD environment**
5. **Monitor for old token usage in audit logs**
6. **After 24 hours, remove old secret from secrets.json**

### Emergency Rotation (Compromise Suspected)

Rotate immediately if secret may have been compromised.

**Steps:**
1. **Revoke immediately** - Remove client from secrets.json
2. **Generate new client ID** (don't reuse old one)
3. **Generate new secret**
4. **Update authorization.json** if needed
5. **Issue new tokens**
6. **Review audit logs** for suspicious activity with old client_id

## Threat Model and Mitigations

### Threat: Token Theft

**Attack:** Attacker steals JWT token from environment variable, logs, or network traffic.

**Impact:** Attacker can sign RPMs until token expires.

**Mitigations:**
- Short token lifetimes (hours for admin, days for CI/CD)
- TLS encryption for all traffic
- Don't log tokens (only log client_id/role)
- Use secrets managers with access controls
- Monitor audit logs for suspicious IPs

**Detection:**
- Unexpected client_ip in audit logs
- Unusual signing patterns (time of day, volume)
- Requests for unauthorized keys (attempted privilege escalation)

### Threat: Secret Compromise

**Attack:** Attacker gains access to secrets.json file.

**Impact:** Attacker can generate tokens with arbitrary expiration.

**Mitigations:**
- File permissions: `chmod 600 secrets.json`
- Store in secrets manager (AWS Secrets Manager, Vault)
- Encrypt at rest
- Regular rotation
- Limit access to signing server (no shell access for CI/CD)

**Detection:**
- Unauthorized file access (monitor with auditd)
- Tokens with unexpected expiration times
- New tokens for existing client_id that weren't generated by ops team

### Threat: Replay Attacks

**Attack:** Attacker captures valid signing request and replays it.

**Impact:** Minimal - replaying same digest just produces same signature again.

**Mitigations:**
- Not strictly necessary (idempotent operation)
- Rate limiting prevents abuse
- Audit logs track all requests

### Threat: Denial of Service

**Attack:** Attacker floods server with signing requests.

**Impact:** Server becomes unavailable for legitimate clients.

**Mitigations:**
- Rate limiting per role (enforced in application layer)
- Thread limits (default 10 concurrent requests)
- Request size limits (10KB max)
- Read timeout (10 seconds)
- Network-level rate limiting (AWS WAF, iptables)

**Detection:**
- RATE_LIMITED events in audit log
- HTTP 503 responses (server at capacity)
- High CPU usage on signing server

### Threat: Privilege Escalation

**Attack:** Client with limited role attempts to use unauthorized key.

**Impact:** Blocked by authorization system.

**Mitigations:**
- Role-based access control enforced server-side
- Empty allowed_keys list only for emergency roles
- Audit logging of all DENIED attempts

**Detection:**
- DENIED events in audit log
- Requests for keys not in role.allowed_keys
- Pattern of authorization failures from single client_id

### Threat: Time-Based Attacks

**Attack:** Attacker manipulates system clock to extend token validity.

**Impact:** None - expiration checked against server time, not client time.

**Mitigations:**
- Server validates exp field using server's clock
- Use NTP to keep server time accurate
- Client clock is irrelevant

## Token Debugging

### Decode Token (Without Validation)

```bash
./generate-token.py decode --token "$TOKEN"
```

Output:
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "client_id": "github-actions-prod",
    "role": "production",
    "iat": 1709337600,
    "exp": 1709352000
  }
}
```

### Validate Token

```bash
./generate-token.py validate \
    --token "$TOKEN" \
    --secret-file config/secrets.json
```

### Common Issues

**"Invalid signature"**
- Wrong secret in secrets.json
- Token corrupted (check for extra whitespace)
- Token generated with different secret

**"Token expired"**
- Check system time: `date -u`
- Regenerate token with longer expiration
- For debugging, use --expires-hours 1

**"Unknown role"**
- Role in token doesn't exist in authorization.json
- Typo in role name
- Server hasn't reloaded config (restart server)

## Performance Considerations

### Token Validation Cost

JWT validation is fast (~0.1ms per token):
1. Base64 decode header + payload (CPU-bound)
2. HMAC-SHA256 computation (CPU-bound)
3. String comparison (constant-time)

**Caching Strategy:**
- Secrets file cached in memory on first load
- Authorization config cached in memory on first load
- No per-token caching needed (validation is cheap)

### Rate Limiting Cost

Rate limit check is O(n) where n = requests in past hour:
- Typical: n < 1000 per client
- Deque operations: O(1) append, O(1) popleft
- Timestamp comparison: O(n) worst case

**Optimization:**
- Purge old entries only once per request
- Use deque for efficient insertion/removal
- Store in memory (not database)

### Audit Logging Cost

Logging is I/O bound:
- JSON serialization: ~0.01ms
- Disk write: ~1-10ms (depends on storage)

**Best Practices:**
- Use fast SSD storage
- Log to separate disk if high throughput
- Rotate logs daily (avoid huge files)
- Consider async logging for high volume

## Migration from Unauthenticated Setup

### Step 1: Add Authentication Files (No Impact)

```bash
# Add auth.py module (not used yet)
# Create config/secrets.json
# Create config/authorization.json
```

Server runs without authentication at this point.

### Step 2: Start Server with Authentication (Breaking Change)

```bash
python3 signing-server.py \
    --enable-auth \
    --secrets-file config/secrets.json \
    --authz-config config/authorization.json
```

**All unauthenticated requests now return HTTP 401.**

### Step 3: Update Clients

```bash
# Generate tokens
./generate-token.py --generate \
    --client-id build-server-01 \
    --role production \
    --secret-file config/secrets.json

# Update client environment
export GPG_SERVER_TOKEN="<token-from-above>"
```

### Gradual Migration Strategy

Not supported - authentication is all-or-nothing. Options:

1. **Blue-Green Deployment:**
   - Run two servers (one with auth, one without)
   - Migrate clients gradually
   - Shut down unauthenticated server when done

2. **Feature Flag in Code:**
   - Add `if AUTH_ENABLED:` checks around auth code
   - Default to disabled, enable after testing

3. **Scheduled Maintenance Window:**
   - Coordinate with all teams
   - Enable authentication at specific time
   - Update all clients in same window

## Appendix: Implementation Details

### Python 3.6 Compatibility

The implementation uses only Python standard library:
- `json` - JSON parsing
- `base64` - Base64url encoding/decoding
- `hmac` - HMAC-SHA256 signing
- `hashlib` - SHA256 hashing
- `time` - Timestamp handling
- `collections.deque` - Rate limiting history

No external dependencies (no `pyjwt`, no `cryptography`).

### Base64url Encoding

JWT uses Base64url (not standard Base64):
- Replace `+` with `-`
- Replace `/` with `_`
- Remove `=` padding

Python implementation:
```python
def base64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')

def base64url_decode(data):
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data)
```

### Constant-Time String Comparison

To prevent timing attacks on signature validation:
```python
import hmac
hmac.compare_digest(computed_sig, provided_sig)
```

This prevents attackers from using response time to guess signature bytes.
