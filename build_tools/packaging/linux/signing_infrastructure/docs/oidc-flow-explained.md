# OIDC Authentication Flow - Step by Step

## Quick Answer

**Q: How does the OIDC token get from GitHub Actions to the signing server?**

**A:** Via environment variables!

1. GitHub Actions generates OIDC token
2. Token stored in workflow step output: `steps.oidc.outputs.token`
3. Token passed as environment variable: `GPG_SERVER_TOKEN=${{ steps.oidc.outputs.token }}`
4. **gpgshim reads `GPG_SERVER_TOKEN` environment variable** (line 40 in gpgshim)
5. gpgshim adds token to HTTP header: `Authorization: Bearer <token>`
6. Signing server receives token, validates it, and signs package

---

## Detailed Flow

### 1. GitHub Actions: Generate OIDC Token

**Workflow file:**
```yaml
permissions:
  id-token: write  # ← Required to request OIDC tokens

steps:
  - name: Get OIDC token for signing
    id: oidc
    run: |
      # Request token from GitHub OIDC provider
      OIDC_TOKEN=$(curl -sSL \
        -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
        "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" \
        | jq -r '.value')

      # Store token in step output (masked in logs)
      echo "::add-mask::$OIDC_TOKEN"
      echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT
```

**What happens:**
- GitHub Actions makes request to its internal OIDC provider
- Provider generates short-lived (10 min) RS256-signed JWT token
- Token contains workflow metadata: repository, branch, actor, workflow file
- Token stored in step output: `${{ steps.oidc.outputs.token }}`

**Token payload example:**
```json
{
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/build_native_linux_packages.yml",
  "actor": "username",
  "run_id": "123456789",
  "exp": 1709737200  // Expires in 10 minutes
}
```

---

### 2. GitHub Actions: Pass Token to gpgshim

**Workflow file:**
```yaml
- name: Sign RPM packages
  env:
    GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
    GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}  # ← OIDC token here
  run: |
    rpmsign --addsign --define "_gpg_path gpgshim" package.rpm
```

**What happens:**
- Environment variable `GPG_SERVER_TOKEN` is set to the OIDC token
- When `rpmsign` runs, it calls `gpgshim` (because of `_gpg_path` macro)
- gpgshim inherits environment variables from the shell

**Key point:** gpgshim doesn't know it's an OIDC token - it just reads `GPG_SERVER_TOKEN` env var!

---

### 3. gpgshim: Read Token from Environment

**File:** `build_tools/packaging/linux/gpgshim` (line 40)

```python
class GPGShim:
    def __init__(self):
        # Read configuration from environment variables
        self.signing_server = os.environ.get('GPG_SIGNING_SERVER', 'http://localhost:8080/sign')
        self.auth_token = os.environ.get('GPG_SERVER_TOKEN', '')  # ← Reads OIDC token here
```

**What happens:**
- gpgshim reads `GPG_SERVER_TOKEN` environment variable
- Value is the OIDC token (could also be JWT - gpgshim doesn't care)
- Token stored in `self.auth_token`

---

### 4. gpgshim: Send Token to Server

**File:** `build_tools/packaging/linux/gpgshim` (lines 248-249)

```python
def sign_data(self, data, args):
    # ... prepare request ...

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'gpgshim/1.0'
    }

    # Add authentication token if available
    if self.auth_token:
        headers['Authorization'] = 'Bearer {}'.format(self.auth_token)  # ← OIDC token sent here

    # Send HTTP POST request to signing server
    request = Request(self.signing_server, data=json_data, headers=headers)
    response = urlopen(request, timeout=self.timeout)
```

**HTTP Request:**
```http
POST /sign HTTP/1.1
Host: signing.yourdomain.com
Content-Type: application/json
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjEifQ...
                      ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                      This is the OIDC token!

{
  "data": "base64-encoded-rpm-header",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256",
  "armor": false
}
```

**What happens:**
- gpgshim constructs HTTP POST request
- Adds token as `Authorization: Bearer <token>` header
- Sends request to signing server

---

### 5. Signing Server: Extract Token

**File:** `server/signing-server.py` (authenticate_request method)

```python
def authenticate_request(self):
    """Extract and validate token from Authorization header."""
    auth_header = self.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None, None

    token = auth_header[7:]  # Remove "Bearer " prefix ← OIDC token extracted here

    # Try OIDC validation first
    if OIDC_AVAILABLE:
        oidc_payload = validate_github_oidc_token(token, audience='amd-signing-service')
        if oidc_payload:
            return oidc_payload, 'oidc'  # ← OIDC token validated!

    # Fall back to JWT if OIDC fails
    jwt_payload = validate_jwt_token(token, secrets)
    if jwt_payload:
        return jwt_payload, 'jwt'

    return None, None  # Invalid token
```

**What happens:**
- Server extracts token from `Authorization` header
- Tries OIDC validation first (if PyJWT installed)
- Falls back to JWT validation if OIDC fails
- Returns token payload and type

---

### 6. Signing Server: Validate OIDC Token

**File:** `server/auth.py` (validate_github_oidc_token function)

```python
def validate_github_oidc_token(token, audience="amd-signing-service"):
    """Validate GitHub OIDC token using RS256 signature."""

    # Initialize JWKS client (fetches GitHub's public keys)
    jwks_client = PyJWKClient("https://token.actions.githubusercontent.com/.well-known/jwks")

    # Get signing key from GitHub's JWKS
    signing_key = jwks_client.get_signing_key_from_jwt(token)

    # Verify and decode token (checks signature, expiration, audience, issuer)
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=audience,  # Must match "amd-signing-service"
        issuer="https://token.actions.githubusercontent.com"
    )

    return payload  # Token is valid!
```

**What happens:**
1. Server fetches GitHub's public keys (JWKS) from `https://token.actions.githubusercontent.com/.well-known/jwks`
2. Uses public key to verify token signature (proves GitHub signed it)
3. Checks token expiration (not expired)
4. Checks audience (must be "amd-signing-service")
5. Checks issuer (must be GitHub)
6. Returns decoded payload with workflow metadata

**Cryptographic proof:** Only GitHub could have signed this token with the private key matching the public key in JWKS.

---

### 7. Signing Server: Authorize Request

**File:** `server/auth.py` (authorize_oidc_request function)

```python
def authorize_oidc_request(payload, key_id, digest_algo, authz_config):
    """Authorize based on OIDC token claims."""

    # Extract workflow metadata from token
    repository = payload.get('repository')  # "ROCm/TheRock"
    ref = payload.get('ref')  # "refs/heads/main"
    workflow = payload.get('workflow')  # ".github/workflows/build_native_linux_packages.yml"

    # Check repository restriction
    if repository not in allowed_repositories:
        return False, f"Repository '{repository}' not authorized"

    # Check branch restriction
    if not matches(ref, allowed_refs):
        return False, f"Branch '{ref}' not authorized"

    # Check workflow restriction
    if workflow not in allowed_workflows:
        return False, f"Workflow '{workflow}' not authorized"

    return True, "Authorized"
```

**What happens:**
- Server checks if repository is in `allowed_repositories` list
- Checks if branch matches `allowed_refs` patterns
- Checks if workflow is in `allowed_workflows` list
- All checks must pass for authorization

**Example authorization config:**
```json
{
  "allowed_repositories": ["ROCm/TheRock", "ROCm/rockrel"],
  "allowed_refs": ["refs/heads/main", "refs/heads/release/*"],
  "allowed_workflows": [".github/workflows/build_native_linux_packages.yml"]
}
```

---

### 8. Signing Server: Sign Package

**File:** `server/signing-server.py`

```python
# If authorization passes, sign the package
semaphore.acquire()
try:
    gpg_result = subprocess.run([
        'gpg', '--detach-sign',
        '--local-user', key_id,
        '--digest-algo', digest_algo,
        '--armor' if armor else '--no-armor'
    ], input=data, capture_output=True, timeout=30)

    signature = gpg_result.stdout

    # Return signature to gpgshim
    return {'signature': base64.b64encode(signature).decode()}
finally:
    semaphore.release()
```

**What happens:**
- Server signs package with GPG
- Returns signature to gpgshim
- gpgshim writes signature to output
- rpmsign embeds signature into RPM package

---

## Summary: Token Journey

```
GitHub Actions (generates OIDC token)
    ↓ stores in step output
${{ steps.oidc.outputs.token }}
    ↓ set as env var
GPG_SERVER_TOKEN=eyJhbGciOiJS...
    ↓ inherited by shell
rpmsign calls gpgshim
    ↓ reads env var (line 40)
gpgshim: self.auth_token = os.environ.get('GPG_SERVER_TOKEN')
    ↓ adds to HTTP header (lines 248-249)
Authorization: Bearer eyJhbGciOiJS...
    ↓ HTTP POST request
Signing Server receives token
    ↓ extracts from header
token = auth_header[7:]
    ↓ validates signature
validate_github_oidc_token(token)
    ↓ checks authorization
authorize_oidc_request(payload)
    ↓ signs package
GPG signing
    ↓ returns signature
gpgshim receives signature
    ↓ writes to stdout
rpmsign embeds in package
```

---

## Key Takeaways

1. **Environment variable is the bridge:** `GPG_SERVER_TOKEN` is how the token gets from workflow to gpgshim

2. **gpgshim is token-agnostic:** It doesn't know or care if it's JWT or OIDC - just reads env var and sends as Bearer token

3. **Server detects token type:** Server tries OIDC first (RS256), falls back to JWT (HMAC-SHA256)

4. **No code changes needed:** Switching from JWT to OIDC only requires changing how the token is generated in the workflow

5. **Backward compatible:** Existing JWT workflows continue to work without changes

---

## Common Questions

### Q: Does gpgshim need to be updated for OIDC?
**A:** No! gpgshim just reads `GPG_SERVER_TOKEN` env var - it works with any token type.

### Q: How does the server know it's an OIDC token?
**A:** It tries OIDC validation first (RS256 signature check). If that fails, tries JWT validation (HMAC-SHA256).

### Q: Can I use both JWT and OIDC at the same time?
**A:** Yes! Different workflows can use different token types. Server handles both automatically.

### Q: What if PyJWT is not installed?
**A:** Server falls back to JWT-only mode. OIDC tokens will be rejected.

### Q: How do I debug token issues?
**A:** Check server logs (`/var/log/gpg-signing/audit.log`) - shows token type and validation errors.

---

## References

- **gpgshim source:** `build_tools/packaging/linux/gpgshim` (lines 40, 248-249)
- **Server auth:** `server/auth.py` (validate_github_oidc_token, authorize_oidc_request)
- **Deployment guide:** `DEPLOYMENT_GUIDE.md` (Step 13: GitHub Actions integration)
- **Token reference:** `docs/oidc-token-reference.md`
