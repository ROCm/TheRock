# GitHub OIDC Migration Guide

**Migrating from JWT Tokens in Secrets to GitHub OIDC**

---

## Overview

### Current Architecture (JWT Tokens in Secrets)

```
GitHub Actions
    ↓
Long-lived JWT token (stored in GitHub Secrets)
    ↓
Signing Server validates HMAC-SHA256 signature
    ↓
Signs package
```

**Problem:** JWT tokens are long-lived (24 hours) and stored in GitHub Secrets

### Target Architecture (GitHub OIDC)

```
GitHub Actions
    ↓
Request short-lived OIDC token from GitHub (automatic)
    ↓
Signing Server validates token with GitHub's public key (JWKS)
    ↓
Signs package
```

**Benefit:** No long-lived secrets, tokens expire in ~10 minutes, auto-rotated

---

## Benefits of GitHub OIDC

| Aspect | Current (JWT Secrets) | With GitHub OIDC | Improvement |
|--------|----------------------|------------------|-------------|
| **Secret Storage** | Tokens in GitHub Secrets | No secrets needed | ✅ Zero secrets |
| **Token Lifetime** | 24 hours | ~10 minutes | ✅ 144x shorter |
| **Rotation** | Manual (90 days) | Automatic (every run) | ✅ Fully automated |
| **Exfiltration Risk** | Can reuse stolen token | Token tied to specific run | ✅ Cannot reuse |
| **Scope** | Role-based (dev/nightly/release) | Repo + workflow + branch | ✅ More granular |
| **Audit Trail** | Client ID + role | Repo + workflow + branch + run ID | ✅ Better traceability |

**Result:** Eliminates all security concerns with storing tokens in GitHub Secrets

---

## Changes Required

### 1. Signing Server Changes

#### A. Add JWKS Fetching (GitHub's Public Keys)

**New file:** `signing_service/github_oidc.py`

```python
#!/usr/bin/env python3
"""
GitHub OIDC token validation.

Fetches GitHub's public keys (JWKS) and validates OIDC tokens.
"""

import json
import time
import jwt
from urllib.request import urlopen
from datetime import datetime, timedelta

# GitHub OIDC endpoints
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_JWKS_URL = f"{GITHUB_OIDC_ISSUER}/.well-known/jwks"

# Cache for JWKS (avoid fetching every request)
_jwks_cache = {
    'keys': None,
    'expires_at': 0
}

def fetch_github_jwks():
    """
    Fetch GitHub's JWKS (JSON Web Key Set) for token validation.

    Caches keys for 1 hour to avoid excessive requests.
    """
    now = time.time()

    # Return cached keys if still valid
    if _jwks_cache['keys'] and now < _jwks_cache['expires_at']:
        return _jwks_cache['keys']

    # Fetch fresh keys
    try:
        with urlopen(GITHUB_JWKS_URL, timeout=10) as response:
            jwks = json.loads(response.read().decode('utf-8'))

        # Cache for 1 hour
        _jwks_cache['keys'] = jwks
        _jwks_cache['expires_at'] = now + 3600

        return jwks

    except Exception as e:
        # If fetch fails and cache exists, use stale cache
        if _jwks_cache['keys']:
            return _jwks_cache['keys']
        raise Exception(f"Failed to fetch GitHub JWKS: {e}")


def validate_github_oidc_token(token, expected_audience="amd-signing-service"):
    """
    Validate GitHub OIDC token.

    Args:
        token: OIDC token from GitHub Actions
        expected_audience: Expected audience claim (configure in workflow)

    Returns:
        Decoded token payload if valid, None otherwise

    Example payload:
        {
            "iss": "https://token.actions.githubusercontent.com",
            "sub": "repo:ROCm/TheRock:ref:refs/heads/main",
            "aud": "amd-signing-service",
            "repository": "ROCm/TheRock",
            "repository_owner": "ROCm",
            "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
            "workflow_ref": "ROCm/TheRock/.github/workflows/multi_arch_build_native_linux_packages.yml@refs/heads/main",
            "ref": "refs/heads/main",
            "ref_type": "branch",
            "actor": "octocat",
            "run_id": "1234567890",
            "run_number": "42",
            "run_attempt": "1",
            "iat": 1709472000,
            "exp": 1709472600,  # Expires in ~10 minutes
            "nbf": 1709472000
        }
    """
    if not token:
        return None

    try:
        # Fetch GitHub's public keys
        jwks = fetch_github_jwks()

        # Decode header to get key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')

        # Find matching public key
        public_key = None
        for key in jwks.get('keys', []):
            if key.get('kid') == kid:
                public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
                break

        if not public_key:
            return None

        # Validate token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=['RS256'],
            audience=expected_audience,
            issuer=GITHUB_OIDC_ISSUER,
            options={
                'verify_signature': True,
                'verify_exp': True,
                'verify_nbf': True,
                'verify_iat': True,
                'verify_aud': True,
                'verify_iss': True,
            }
        )

        return payload

    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token
    except Exception:
        return None  # Other error


def authorize_github_oidc(payload, authz_config):
    """
    Authorize GitHub OIDC token based on repository, workflow, and branch.

    Args:
        payload: Decoded OIDC token payload
        authz_config: Authorization configuration

    Returns:
        (authorized, role, reason) tuple

    Example authorization config:
        {
            "github_oidc": {
                "ROCm/TheRock": {
                    "workflows": {
                        ".github/workflows/multi_arch_build_native_linux_packages.yml": {
                            "branches": {
                                "main": {
                                    "role": "release",
                                    "allowed_keys": ["therock-release@amd.com"]
                                },
                                "develop": {
                                    "role": "development",
                                    "allowed_keys": ["therock-dev@amd.com"]
                                }
                            },
                            "tags": {
                                "v*": {
                                    "role": "release",
                                    "allowed_keys": ["therock-release@amd.com"]
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    repository = payload.get('repository', '')
    workflow = payload.get('workflow', '')
    ref = payload.get('ref', '')
    ref_type = payload.get('ref_type', '')

    # Get repository config
    github_oidc_config = authz_config.get('github_oidc', {})
    repo_config = github_oidc_config.get(repository, {})

    if not repo_config:
        return False, None, f"Repository '{repository}' not authorized"

    # Get workflow config
    workflows = repo_config.get('workflows', {})
    workflow_config = workflows.get(workflow, {})

    if not workflow_config:
        return False, None, f"Workflow '{workflow}' not authorized for repository '{repository}'"

    # Check branch or tag
    if ref_type == 'branch':
        branches = workflow_config.get('branches', {})

        # Extract branch name from ref (refs/heads/main -> main)
        branch_name = ref.replace('refs/heads/', '')

        # Check exact match or wildcard
        for pattern, config in branches.items():
            if branch_name == pattern or _match_pattern(branch_name, pattern):
                role = config.get('role')
                allowed_keys = config.get('allowed_keys', [])
                return True, role, None

        return False, None, f"Branch '{branch_name}' not authorized for workflow '{workflow}'"

    elif ref_type == 'tag':
        tags = workflow_config.get('tags', {})

        # Extract tag name from ref (refs/tags/v1.0.0 -> v1.0.0)
        tag_name = ref.replace('refs/tags/', '')

        # Check exact match or wildcard
        for pattern, config in tags.items():
            if tag_name == pattern or _match_pattern(tag_name, pattern):
                role = config.get('role')
                allowed_keys = config.get('allowed_keys', [])
                return True, role, None

        return False, None, f"Tag '{tag_name}' not authorized for workflow '{workflow}'"

    return False, None, f"Unknown ref type '{ref_type}'"


def _match_pattern(value, pattern):
    """Simple wildcard matching (* and ?)."""
    import re

    # Convert wildcard pattern to regex
    regex_pattern = pattern.replace('.', '\\.').replace('*', '.*').replace('?', '.')
    regex_pattern = f"^{regex_pattern}$"

    return bool(re.match(regex_pattern, value))
```

**Dependencies needed:**
```bash
pip install PyJWT cryptography
```

**Note:** This adds 2 dependencies (PyJWT, cryptography), but they're widely used and well-maintained.

---

#### B. Update Signing Server to Support OIDC

**Modify:** `signing_service/signing-server.py`

```python
# Add import
from github_oidc import validate_github_oidc_token, authorize_github_oidc

class SigningHandler(BaseHTTPRequestHandler):

    def authenticate_request(self):
        """
        Authenticate request - support both JWT and GitHub OIDC.

        Returns:
            (auth_type, payload) tuple where:
            - auth_type: 'jwt' or 'oidc'
            - payload: decoded token payload
        """
        auth_header = self.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None, None

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Try GitHub OIDC first (check issuer claim)
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
            issuer = unverified.get('iss', '')

            if issuer == 'https://token.actions.githubusercontent.com':
                # GitHub OIDC token
                payload = validate_github_oidc_token(token, expected_audience='amd-signing-service')
                if payload:
                    return 'oidc', payload
                else:
                    return None, None
        except:
            pass

        # Fall back to legacy JWT (for backward compatibility)
        if self.AUTH_ENABLED and AUTH_AVAILABLE:
            if self._secrets_cache is None:
                self.__class__._secrets_cache = load_secrets(self.SECRETS_FILE)

            payload = validate_jwt_token(token, self._secrets_cache)
            if payload:
                return 'jwt', payload

        return None, None

    def do_POST(self):
        """Handle POST requests (updated for OIDC)."""
        if self.path != '/sign':
            self.send_error(404, "Not Found")
            return

        try:
            # ... read request body ...

            # AUTHENTICATION
            auth_type, payload = self.authenticate_request()
            if not payload:
                self.send_json_error(401, "Unauthorized: Invalid or missing token")
                return

            # AUTHORIZATION
            if auth_type == 'oidc':
                # GitHub OIDC authorization
                if self._authz_cache is None:
                    self.__class__._authz_cache = load_authorization_config(self.AUTHZ_CONFIG_FILE)

                authorized, role, reason = authorize_github_oidc(payload, self._authz_cache)
                if not authorized:
                    self.send_json_error(403, f"Forbidden: {reason}")
                    return

                client_id = f"{payload['repository']}:{payload['workflow_ref']}"

            elif auth_type == 'jwt':
                # Legacy JWT authorization (existing code)
                role = payload.get('role', '')
                client_id = payload.get('client_id', '')

                # ... existing RBAC checks ...

            # ... rest of signing logic ...
```

---

#### C. Update Authorization Configuration

**New format:** `config/authorization.json`

```json
{
  "github_oidc": {
    "ROCm/TheRock": {
      "workflows": {
        ".github/workflows/multi_arch_build_native_linux_packages.yml": {
          "branches": {
            "main": {
              "role": "release",
              "allowed_keys": ["therock-release@amd.com"],
              "allowed_digest_algos": ["SHA256", "SHA512"],
              "max_requests_per_hour": 1000
            },
            "develop": {
              "role": "development",
              "allowed_keys": ["therock-dev@amd.com"],
              "allowed_digest_algos": ["SHA256", "SHA512"],
              "max_requests_per_hour": 10000
            }
          },
          "tags": {
            "v*": {
              "role": "release",
              "allowed_keys": ["therock-release@amd.com"],
              "allowed_digest_algos": ["SHA256", "SHA512"],
              "max_requests_per_hour": 1000
            }
          }
        }
      }
    }
  },

  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000
    },
    "release": {
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000
    }
  }
}
```

**Migration:** Keep legacy `roles` section for backward compatibility during transition.

---

### 2. GitHub Actions Workflow Changes

#### A. Add OIDC Permission

**Modify:** `.github/workflows/multi_arch_build_native_linux_packages.yml`

```yaml
name: Build Native Linux Packages

# Add permissions for OIDC token
permissions:
  id-token: write  # Required for OIDC token
  contents: read

jobs:
  build_native_packages:
    runs-on: ubuntu-24.04

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # NEW: Get OIDC token from GitHub
      - name: Get GitHub OIDC token
        id: oidc
        run: |
          # Request OIDC token from GitHub with custom audience
          OIDC_TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')

          echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT

          # Debug: Show token claims (base64 decode payload)
          echo "Token claims:"
          echo "$OIDC_TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null | jq . || true

      # Build packages (existing)
      - name: Build Packages
        run: |
          python ./build_tools/packaging/linux/build_package.py ...

      # UPDATED: Use OIDC token instead of secret
      - name: Sign RPM packages
        if: inputs.native_package_type == 'rpm'
        env:
          GPG_SIGNING_SERVER: https://signing-server.example.com/sign
          GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}  # OIDC token, not secret
        run: |
          # Install gpgshim
          cp build_tools/packaging/linux/gpgshim ~/.local/bin/
          chmod +x ~/.local/bin/gpgshim

          # Sign RPMs (gpgshim uses GPG_SERVER_TOKEN from env)
          find dist/rpm -name "*.rpm" -exec \
            rpmsign --define "_gpg_path $HOME/.local/bin/gpgshim" --addsign {} \;
```

**Key Changes:**
1. ✅ Add `permissions: id-token: write`
2. ✅ Add step to request OIDC token from GitHub
3. ✅ Use OIDC token instead of `${{ secrets.GPG_SERVER_TOKEN_DEV }}`
4. ✅ Remove all `GPG_SERVER_TOKEN_*` GitHub Secrets (can delete them!)

---

#### B. Remove Secrets Configuration Step

**BEFORE (JWT):**
```yaml
- name: Configure GPG signing environment
  run: |
    case "${{ env.RELEASE_TYPE }}" in
      dev)
        echo "GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_DEV }}" >> $GITHUB_ENV
        ;;
      nightly)
        echo "GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_NIGHTLY }}" >> $GITHUB_ENV
        ;;
      release)
        echo "GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_RELEASE }}" >> $GITHUB_ENV
        ;;
    esac
```

**AFTER (OIDC):**
```yaml
# No configuration needed - token automatically scoped by branch/workflow
- name: Get OIDC token
  id: oidc
  run: |
    OIDC_TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
      "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')
    echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT
```

**Benefit:** Automatic role selection based on branch:
- `refs/heads/main` → release role (configured in authorization.json)
- `refs/heads/develop` → development role
- No manual token selection needed!

---

### 3. No Changes Needed for gpgshim

**Good news:** gpgshim doesn't need any changes!

It already reads `GPG_SERVER_TOKEN` environment variable and sends it as `Authorization: Bearer` header. It doesn't care whether it's a JWT or OIDC token.

```python
# gpgshim (no changes needed)
self.auth_token = os.environ.get('GPG_SERVER_TOKEN', '')

headers = {
    'Authorization': f'Bearer {self.auth_token}'  # Works for both JWT and OIDC
}
```

---

## Migration Plan

### Phase 1: Add OIDC Support (Week 1)

**Server-side:**
1. Add `github_oidc.py` module
2. Install PyJWT and cryptography dependencies
3. Update `signing-server.py` to support both JWT and OIDC
4. Update `authorization.json` with GitHub OIDC config
5. Test OIDC validation locally

**Effort:** 2-3 days

---

### Phase 2: Parallel Operation (Week 2)

**Both JWT and OIDC work simultaneously:**

```python
# Server supports both auth methods
if issuer == 'https://token.actions.githubusercontent.com':
    # New: GitHub OIDC
    return 'oidc', validate_github_oidc_token(token)
else:
    # Legacy: JWT from secrets
    return 'jwt', validate_jwt_token(token, secrets)
```

**Deploy to dev environment:**
1. Deploy updated server to dev signing server
2. Test with OIDC token from dev workflow
3. Verify legacy JWT tokens still work

**Effort:** 1-2 days

---

### Phase 3: Migrate Workflows (Week 3)

**Migrate workflows one at a time:**

1. **Dev builds first:**
   ```yaml
   # Update multi_arch_build_native_linux_packages.yml for dev
   permissions:
     id-token: write

   steps:
     - name: Get OIDC token
       id: oidc
       run: |
         curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
           "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service"
   ```

2. **Nightly builds:**
   - Same changes as dev
   - Test thoroughly

3. **Release builds (last):**
   - Same changes
   - Most critical, test extensively

**Effort:** 3-5 days (including testing)

---

### Phase 4: Cleanup (Week 4)

**Remove legacy JWT support:**

1. Verify all workflows using OIDC
2. Delete GitHub Secrets:
   - `GPG_SERVER_TOKEN_DEV`
   - `GPG_SERVER_TOKEN_NIGHTLY`
   - `GPG_SERVER_TOKEN_RELEASE`
3. Remove JWT validation code from server (optional - can keep for other clients)
4. Remove `config/secrets.json` (optional - can keep for other use cases)

**Effort:** 1 day

---

## Comparison: Before and After

### Before (JWT in Secrets)

**GitHub Secrets:**
```
GPG_SIGNING_SERVER = https://signing-server.example.com/sign
GPG_SERVER_TOKEN_DEV = eyJhbGci... (long-lived, 24 hours)
GPG_SERVER_TOKEN_NIGHTLY = eyJhbGci... (long-lived, 24 hours)
GPG_SERVER_TOKEN_RELEASE = eyJhbGci... (long-lived, 24 hours)
```

**Workflow:**
```yaml
env:
  GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_DEV }}
```

**Security:**
- ⚠️ Token valid for 24 hours
- ⚠️ Manual rotation every 90 days
- ⚠️ Can be exfiltrated and reused

---

### After (GitHub OIDC)

**GitHub Secrets:**
```
GPG_SIGNING_SERVER = https://signing-server.example.com/sign
(No tokens!)
```

**Workflow:**
```yaml
permissions:
  id-token: write

steps:
  - name: Get OIDC token
    id: oidc
    run: |
      curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
        "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service"

  - name: Sign packages
    env:
      GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}
```

**Security:**
- ✅ Token valid for ~10 minutes
- ✅ Automatic rotation (every workflow run)
- ✅ Cannot be reused (tied to specific workflow run)
- ✅ More granular scope (repo + workflow + branch + run ID)

---

## Implementation Effort Summary

| Phase | Task | Effort | Risk |
|-------|------|--------|------|
| 1 | Add OIDC support to server | 2-3 days | 🟢 Low |
| 2 | Deploy and test in dev | 1-2 days | 🟢 Low |
| 3 | Migrate workflows | 3-5 days | 🟡 Medium |
| 4 | Cleanup legacy JWT | 1 day | 🟢 Low |
| **Total** | **End-to-end migration** | **7-11 days** | **🟢 Low** |

**Recommendation:** Implement after POC is proven in production (Q2 2026)

---

## Testing the OIDC Integration

### Local Testing (Without GitHub Actions)

**1. Get a real OIDC token from GitHub:**

```bash
# Trigger a test workflow that prints the token
# .github/workflows/test-oidc.yml
name: Test OIDC
on: workflow_dispatch

permissions:
  id-token: write

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Get OIDC token
        run: |
          TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')
          echo "::add-mask::$TOKEN"
          echo "TOKEN=$TOKEN" >> $GITHUB_OUTPUT

      - name: Decode token (debugging)
        run: |
          echo "${{ steps.get-token.outputs.TOKEN }}" | cut -d. -f2 | base64 -d | jq .
```

**2. Test server locally:**

```bash
# Start server
python3 server/signing-server.py --port 8080 --enable-oidc

# Test with OIDC token
TOKEN="<paste token from GitHub Actions>"

curl -X POST http://localhost:8080/sign \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "'$(echo "test" | base64)'",
    "key_id": "therock-dev@amd.com",
    "digest_algo": "SHA256"
  }'
```

---

## Rollback Plan

If OIDC implementation has issues:

**Immediate rollback:**
1. Keep legacy JWT support in server (don't remove)
2. Re-create GitHub Secrets with JWT tokens
3. Revert workflow changes (remove OIDC token step)

**Duration:** < 1 hour

**Why this is safe:**
- Server supports both JWT and OIDC simultaneously
- No breaking changes if JWT support retained
- Can switch back and forth easily

---

## Recommendation

### For POC (Current): Use JWT in Secrets ✅

**Reasons:**
1. ✅ Faster to implement (already designed)
2. ✅ Fewer dependencies (no PyJWT needed)
3. ✅ Simpler for initial deployment
4. ✅ Good enough security with mitigations

**Timeline:** Ready to deploy now

---

### For v2 (Future): Migrate to OIDC ✅

**Reasons:**
1. ✅ Better security (no long-lived secrets)
2. ✅ Automatic rotation (zero maintenance)
3. ✅ More granular authorization (repo + workflow + branch)
4. ✅ Industry best practice

**Timeline:** Q2 2026 (after POC proven in production)

**Effort:** ~2 weeks (including testing and migration)

---

## Summary: What Changes Are Needed?

### Signing Server (2-3 days)

1. ✅ Add `github_oidc.py` module (~200 lines)
2. ✅ Install PyJWT + cryptography dependencies
3. ✅ Update `signing-server.py` to support OIDC (~50 lines changed)
4. ✅ Update `authorization.json` format (add GitHub OIDC section)
5. ✅ Test JWKS fetching and validation

### GitHub Actions Workflows (3-5 days)

1. ✅ Add `permissions: id-token: write` to workflows
2. ✅ Add step to request OIDC token from GitHub
3. ✅ Replace `${{ secrets.GPG_SERVER_TOKEN_* }}` with OIDC token
4. ✅ Remove token selection logic (automatic based on branch)
5. ✅ Test all workflows (dev, nightly, release)

### GitHub Secrets (1 day)

1. ✅ Delete `GPG_SERVER_TOKEN_DEV` (optional)
2. ✅ Delete `GPG_SERVER_TOKEN_NIGHTLY` (optional)
3. ✅ Delete `GPG_SERVER_TOKEN_RELEASE` (optional)
4. ✅ Keep only `GPG_SIGNING_SERVER` URL

### gpgshim Client

1. ✅ **No changes needed!** (already compatible)

---

**Total Implementation: ~7-11 days (1.5-2 weeks)**

**Risk: Low** (can run both JWT and OIDC in parallel during migration)
