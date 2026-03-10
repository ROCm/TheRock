# JWT vs OIDC: Workflow Restriction Capabilities

## Question
Can we restrict signing workflow invocation by checking:
1. Repository (e.g., only ROCm/TheRock)
2. Branch (e.g., only main)
3. Invoking job/workflow

Using both **JWT** and **OIDC** approaches?

## TL;DR Answer

| Feature | JWT (Current POC) | GitHub OIDC |
|---------|-------------------|-------------|
| **Restrict by repository** | ⚠️ Requires custom implementation | ✅ Native support via `repository` claim |
| **Restrict by branch** | ⚠️ Requires custom implementation | ✅ Native support via `ref` claim |
| **Restrict by workflow** | ⚠️ Requires custom implementation | ✅ Native support via `workflow` claim |
| **Implementation effort** | Medium (custom GitHub Actions context passing) | Low (built-in claims) |
| **Security** | Depends on token generation process | High (claims are cryptographically signed by GitHub) |

**Recommendation:** Both can achieve workflow restriction, but **OIDC is superior** as it provides these controls natively and cryptographically verifies all claims.

---

## Detailed Comparison

### Option 1: JWT (Current POC) - Custom Implementation Required

**Current JWT Token Payload:**
```json
{
  "client_id": "github-actions-dev",
  "role": "development",
  "exp": 1709740800
}
```

**Problem:** Current JWT tokens do NOT contain repository, branch, or workflow information.

**Solution A: Pass Context in GitHub Actions (Recommended for JWT)**

Modify the token generation to embed GitHub Actions context:

**Modified Token Payload:**
```json
{
  "client_id": "github-actions-dev",
  "role": "development",
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
  "run_id": "123456789",
  "actor": "username",
  "exp": 1709740800
}
```

**Implementation:**

**File: `build_tools/packaging/linux/signing_infrastructure/tools/generate-token.py`**

```python
#!/usr/bin/env python3
"""
Generate JWT token with GitHub Actions context for workflow restriction
"""

import jwt
import json
import sys
import os
from datetime import datetime, timedelta

def generate_token_with_context(client_id, role, secret, expires_hours=24,
                                  repository=None, ref=None, workflow=None,
                                  run_id=None, actor=None):
    """Generate JWT token with GitHub Actions context"""

    now = datetime.utcnow()
    expiration = now + timedelta(hours=expires_hours)

    payload = {
        'client_id': client_id,
        'role': role,
        'exp': int(expiration.timestamp())
    }

    # Add GitHub Actions context if provided
    if repository:
        payload['repository'] = repository
    if ref:
        payload['ref'] = ref
    if workflow:
        payload['workflow'] = workflow
    if run_id:
        payload['run_id'] = run_id
    if actor:
        payload['actor'] = actor

    # Sign token with HMAC-SHA256
    token = jwt.encode(payload, secret, algorithm='HS256')
    return token

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--secret-file', required=True)
    parser.add_argument('--expires-hours', type=int, default=24)

    # GitHub Actions context
    parser.add_argument('--repository', help='GitHub repository (e.g., ROCm/TheRock)')
    parser.add_argument('--ref', help='Git ref (e.g., refs/heads/main)')
    parser.add_argument('--workflow', help='Workflow file path')
    parser.add_argument('--run-id', help='GitHub Actions run ID')
    parser.add_argument('--actor', help='GitHub username who triggered workflow')

    args = parser.parse_args()

    # Load secret from file
    with open(args.secret_file) as f:
        secrets = json.load(f)

    secret = secrets['clients'][args.client_id]['secret']

    token = generate_token_with_context(
        args.client_id,
        args.role,
        secret,
        args.expires_hours,
        args.repository,
        args.ref,
        args.workflow,
        args.run_id,
        args.actor
    )

    print(token)
```

**File: `.github/workflows/multi_arch_build_native_linux_packages.yml`**

```yaml
- name: Generate JWT token with workflow context
  id: generate_token
  run: |
    # Generate token with GitHub Actions context embedded
    TOKEN=$(python3 build_tools/packaging/linux/signing_infrastructure/tools/generate-token.py \
      --client-id github-actions-dev \
      --role development \
      --secret-file config/secrets.json \
      --expires-hours 1 \
      --repository "${{ github.repository }}" \
      --ref "${{ github.ref }}" \
      --workflow "${{ github.workflow }}" \
      --run-id "${{ github.run_id }}" \
      --actor "${{ github.actor }}")

    echo "::add-mask::$TOKEN"
    echo "token=$TOKEN" >> $GITHUB_OUTPUT

- name: Sign RPMs with context-aware token
  env:
    GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
    GPG_SERVER_TOKEN: ${{ steps.generate_token.outputs.token }}
  run: |
    # gpgshim uses the token with embedded context
    rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" dist/*.rpm
```

**File: `build_tools/packaging/linux/signing_infrastructure/server/auth.py`**

```python
def authorize_jwt_with_workflow_checks(payload, authz_config):
    """
    Authorize JWT token with workflow restriction checks

    Args:
        payload: JWT token payload (already validated signature)
        authz_config: Authorization configuration

    Returns:
        (authorized, error_message)
    """

    role = payload.get('role')
    if not role:
        return False, "Missing 'role' in token"

    role_config = authz_config['roles'].get(role)
    if not role_config:
        return False, f"Unknown role: {role}"

    # Check repository restriction
    allowed_repos = role_config.get('allowed_repositories', [])
    if allowed_repos:
        repository = payload.get('repository')
        if not repository:
            return False, "Token missing 'repository' claim (required for this role)"
        if repository not in allowed_repos:
            return False, f"Repository '{repository}' not authorized for role '{role}'"

    # Check branch restriction
    allowed_refs = role_config.get('allowed_refs', [])
    if allowed_refs:
        ref = payload.get('ref')
        if not ref:
            return False, "Token missing 'ref' claim (required for this role)"

        # Support patterns like "refs/heads/main" or "refs/heads/release/*"
        if not any(match_ref_pattern(ref, pattern) for pattern in allowed_refs):
            return False, f"Branch '{ref}' not authorized for role '{role}'"

    # Check workflow restriction
    allowed_workflows = role_config.get('allowed_workflows', [])
    if allowed_workflows:
        workflow = payload.get('workflow')
        if not workflow:
            return False, "Token missing 'workflow' claim (required for this role)"
        if workflow not in allowed_workflows:
            return False, f"Workflow '{workflow}' not authorized for role '{role}'"

    return True, None

def match_ref_pattern(ref, pattern):
    """Match git ref against pattern (supports wildcards)"""
    import fnmatch
    return fnmatch.fnmatch(ref, pattern)
```

**File: `build_tools/packaging/linux/signing_infrastructure/config/authorization.json`**

```json
{
  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000,

      "allowed_repositories": ["ROCm/TheRock"],
      "allowed_refs": ["refs/heads/*"],
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml",
        ".github/workflows/build_native_linux_packages.yml"
      ]
    },

    "release": {
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000,

      "allowed_repositories": ["ROCm/TheRock"],
      "allowed_refs": ["refs/heads/main", "refs/heads/release/*"],
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml"
      ]
    }
  }
}
```

**Pros of JWT with custom context:**
- ✅ Can restrict by repo, branch, workflow
- ✅ Reuses existing JWT infrastructure
- ✅ Fine-grained control via authorization.json

**Cons of JWT with custom context:**
- ❌ Requires custom implementation (GitHub Actions context passing)
- ❌ Token generation happens in workflow (trust boundary)
- ❌ More complex than OIDC
- ❌ GitHub Actions context must be manually passed
- ❌ No cryptographic proof that claims came from GitHub (only HMAC from secret)

---

### Option 2: GitHub OIDC (Recommended) - Native Support

**OIDC Token Payload (provided by GitHub):**
```json
{
  "jti": "example-id",
  "sub": "repo:ROCm/TheRock:ref:refs/heads/main",
  "aud": "amd-signing-service",
  "ref": "refs/heads/main",
  "sha": "example-sha",
  "repository": "ROCm/TheRock",
  "repository_owner": "ROCm",
  "repository_owner_id": "123456",
  "run_id": "123456789",
  "run_number": "42",
  "run_attempt": "1",
  "actor": "username",
  "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
  "head_ref": "",
  "base_ref": "",
  "event_name": "workflow_dispatch",
  "ref_type": "branch",
  "job_workflow_ref": "ROCm/TheRock/.github/workflows/multi_arch_build_native_linux_packages.yml@refs/heads/main",
  "iss": "https://token.actions.githubusercontent.com",
  "nbf": 1709736000,
  "exp": 1709737200,
  "iat": 1709736300
}
```

**Implementation:**

**File: `build_tools/packaging/linux/signing_infrastructure/server/auth.py`**

```python
import jwt
import requests
from jwt import PyJWKClient

# GitHub OIDC configuration
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_OIDC_JWKS_URL = f"{GITHUB_OIDC_ISSUER}/.well-known/jwks"

# Cache JWKS client
jwks_client = PyJWKClient(GITHUB_OIDC_JWKS_URL, cache_keys=True)

def validate_github_oidc_token(token, audience):
    """
    Validate GitHub OIDC token

    Args:
        token: OIDC token from GitHub Actions
        audience: Expected audience (configured in workflow)

    Returns:
        Decoded payload if valid, None if invalid
    """
    try:
        # Get signing key from GitHub's JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Verify and decode token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=GITHUB_OIDC_ISSUER
        )

        return payload

    except jwt.InvalidTokenError as e:
        print(f"OIDC token validation failed: {e}")
        return None

def authorize_github_oidc(payload, authz_config):
    """
    Authorize GitHub OIDC token with workflow restrictions

    Args:
        payload: OIDC token payload (already validated)
        authz_config: Authorization configuration

    Returns:
        (role, authorized, error_message)
    """

    # Extract claims
    repository = payload.get('repository')
    ref = payload.get('ref')
    workflow = payload.get('workflow')
    actor = payload.get('actor')

    # Determine role based on ref (branch)
    if ref == 'refs/heads/main':
        role = 'release'
    elif ref.startswith('refs/heads/release/'):
        role = 'release'
    elif ref.startswith('refs/heads/'):
        role = 'development'
    else:
        return None, False, f"Unknown ref type: {ref}"

    role_config = authz_config['roles'].get(role)
    if not role_config:
        return None, False, f"Unknown role: {role}"

    # Check repository restriction
    allowed_repos = role_config.get('allowed_repositories', [])
    if allowed_repos and repository not in allowed_repos:
        return role, False, f"Repository '{repository}' not authorized for role '{role}'"

    # Check branch restriction
    allowed_refs = role_config.get('allowed_refs', [])
    if allowed_refs:
        if not any(match_ref_pattern(ref, pattern) for pattern in allowed_refs):
            return role, False, f"Branch '{ref}' not authorized for role '{role}'"

    # Check workflow restriction
    allowed_workflows = role_config.get('allowed_workflows', [])
    if allowed_workflows and workflow not in allowed_workflows:
        return role, False, f"Workflow '{workflow}' not authorized for role '{role}'"

    return role, True, None

def match_ref_pattern(ref, pattern):
    """Match git ref against pattern (supports wildcards)"""
    import fnmatch
    return fnmatch.fnmatch(ref, pattern)
```

**File: `.github/workflows/multi_arch_build_native_linux_packages.yml`**

```yaml
jobs:
  build_native_packages:
    name: Build ${{ inputs.native_package_type }} packages
    runs-on: ubuntu-24.04

    # Enable OIDC token generation
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Get OIDC token from GitHub
        id: oidc
        run: |
          # Request OIDC token from GitHub
          OIDC_TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')

          echo "::add-mask::$OIDC_TOKEN"
          echo "oidc_token=$OIDC_TOKEN" >> $GITHUB_OUTPUT

      - name: Sign RPMs with OIDC token
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.oidc_token }}
        run: |
          # gpgshim uses OIDC token (contains repo, branch, workflow)
          rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" dist/*.rpm
```

**File: `build_tools/packaging/linux/signing_infrastructure/config/authorization.json`**

```json
{
  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000,

      "allowed_repositories": ["ROCm/TheRock"],
      "allowed_refs": ["refs/heads/*", "refs/pull/*"],
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml",
        ".github/workflows/build_native_linux_packages.yml"
      ]
    },

    "release": {
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000,

      "allowed_repositories": ["ROCm/TheRock"],
      "allowed_refs": ["refs/heads/main", "refs/heads/release/*"],
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml"
      ]
    }
  }
}
```

**Pros of OIDC:**
- ✅ **Native support** - repo, branch, workflow are built-in claims
- ✅ **Cryptographically signed by GitHub** - tamper-proof
- ✅ **No secrets stored** - tokens are short-lived (10 min)
- ✅ **Automatic expiration** - no manual rotation needed
- ✅ **Rich context** - includes actor, run_id, event_name, etc.
- ✅ **Industry standard** - OpenID Connect protocol

**Cons of OIDC:**
- ❌ Requires PyJWT library (external dependency)
- ❌ More complex initial setup (JWKS fetching)
- ❌ Requires GitHub Actions `id-token: write` permission

---

## Comparison Table: Workflow Restriction Implementation

| Aspect | JWT (Custom) | GitHub OIDC |
|--------|--------------|-------------|
| **Repository restriction** | ✅ Add `repository` to token payload manually | ✅ Native `repository` claim |
| **Branch restriction** | ✅ Add `ref` to token payload manually | ✅ Native `ref` claim |
| **Workflow restriction** | ✅ Add `workflow` to token payload manually | ✅ Native `workflow` claim |
| **Actor tracking** | ✅ Add `actor` to token payload manually | ✅ Native `actor` claim |
| **Run ID tracking** | ✅ Add `run_id` to token payload manually | ✅ Native `run_id` claim |
| **Token generation** | Manual in workflow (requires secret access) | Automatic by GitHub (no secrets) |
| **Cryptographic proof** | HMAC-SHA256 (symmetric key) | RS256 (asymmetric, signed by GitHub) |
| **Trust boundary** | Workflow can forge claims (has access to secret) | Claims cannot be forged (GitHub-signed) |
| **External dependencies** | None (stdlib only) | PyJWT library |
| **Implementation effort** | Medium (custom token generation) | Low (built-in claims) |
| **Security** | Good (if secret protected) | Excellent (no secrets, GitHub-signed) |

---

## Recommendation

### For Immediate Implementation (Phase 1)
**Use JWT with custom context** (Option 1) if:
- You want to keep the POC's zero-dependency approach
- You're comfortable with token generation in GitHub Actions
- You can protect the JWT secret adequately in GitHub Secrets

**Implementation steps:**
1. Add repository, ref, workflow to token generation
2. Update authorization.json with allowed_repositories, allowed_refs, allowed_workflows
3. Update auth.py to validate these claims
4. Estimated effort: **2-3 days**

### For Long-Term Production (Phase 2)
**Migrate to GitHub OIDC** (Option 2) because:
- ✅ Superior security (no secrets, GitHub-signed tokens)
- ✅ Native workflow restriction (no custom implementation)
- ✅ Tamper-proof claims
- ✅ Automatic token expiration
- ✅ Industry standard

**Migration effort:** 7-11 days (see GITHUB_OIDC_MIGRATION_GUIDE.md)

---

## Example: Blocking Unauthorized Workflow

### Scenario
Attacker with write access to ROCm/TheRock creates malicious workflow:

**File: `.github/workflows/malicious.yml`**
```yaml
name: Malicious Workflow
on: workflow_dispatch

jobs:
  steal_signatures:
    runs-on: ubuntu-latest
    steps:
      - name: Attempt signing
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_DEV }}  # Stolen token
        run: |
          # Try to sign malicious package
          curl -X POST $GPG_SIGNING_SERVER/sign \
            -H "Authorization: Bearer $GPG_SERVER_TOKEN" \
            -d '{"data": "...", "key_id": "therock-dev@amd.com"}'
```

### Protection with JWT (Custom Context)

**Token payload from malicious workflow:**
```json
{
  "client_id": "github-actions-dev",
  "role": "development",
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/attacker-branch",
  "workflow": ".github/workflows/malicious.yml",  ← NOT in allowed list!
  "exp": 1709740800
}
```

**Server response:**
```
HTTP 403 Forbidden
{
  "error": "Workflow '.github/workflows/malicious.yml' not authorized for role 'development'"
}
```

**Audit log:**
```json
{
  "timestamp": "2024-03-06T10:30:00Z",
  "client_id": "github-actions-dev",
  "role": "development",
  "repository": "ROCm/TheRock",
  "workflow": ".github/workflows/malicious.yml",
  "status": "REJECTED",
  "reason": "Workflow not in allowed list"
}
```

### Protection with OIDC

**OIDC token payload:**
```json
{
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/attacker-branch",
  "workflow": ".github/workflows/malicious.yml",  ← Cryptographically signed by GitHub
  "iss": "https://token.actions.githubusercontent.com"
}
```

**Server validation:**
1. Verify token signature using GitHub's JWKS (tamper-proof)
2. Check workflow against allowed list
3. **REJECT** - malicious.yml not in authorized workflows

**Advantage:** Even if attacker modifies workflow to claim it's the authorized workflow, the signature verification will fail because GitHub signed the token with the actual workflow name.

---

## Audit Trail Comparison

### JWT Audit Log
```json
{
  "timestamp": "2024-03-06T10:30:00Z",
  "client_id": "github-actions-dev",
  "role": "development",
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
  "run_id": "123456789",
  "actor": "john.doe",
  "package_name": "rocm-hip-runtime.rpm",
  "package_hash": "abc123...",
  "key_id": "therock-dev@amd.com",
  "status": "SUCCESS",
  "duration_ms": 42
}
```

### OIDC Audit Log (richer context)
```json
{
  "timestamp": "2024-03-06T10:30:00Z",
  "repository": "ROCm/TheRock",
  "repository_owner": "ROCm",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
  "run_id": "123456789",
  "run_number": "42",
  "run_attempt": "1",
  "actor": "john.doe",
  "event_name": "workflow_dispatch",
  "job_workflow_ref": "ROCm/TheRock/.github/workflows/...@refs/heads/main",
  "package_name": "rocm-hip-runtime.rpm",
  "package_hash": "abc123...",
  "key_id": "therock-dev@amd.com",
  "status": "SUCCESS",
  "duration_ms": 42,
  "oidc_sub": "repo:ROCm/TheRock:ref:refs/heads/main"
}
```

**OIDC provides:**
- ✅ `run_number` - detect repeated attempts
- ✅ `run_attempt` - detect retries
- ✅ `event_name` - distinguish workflow_dispatch vs push vs pull_request
- ✅ `job_workflow_ref` - full workflow + commit reference
- ✅ `oidc_sub` - unique identifier for authorization policies

---

## Summary

**Question:** Can we restrict workflow invocation by repo, branch, and workflow?

**Answer:**
- **JWT (Current POC):** ✅ Yes, but requires custom implementation (add GitHub context to token)
- **GitHub OIDC:** ✅ Yes, natively supported with cryptographic proof

**Recommendation:**
1. **Phase 1 (Now):** Implement JWT with custom workflow context (2-3 days effort)
2. **Phase 2 (Later):** Migrate to GitHub OIDC for superior security (7-11 days effort)

Both approaches provide the workflow restriction you need. OIDC is the long-term best practice.
