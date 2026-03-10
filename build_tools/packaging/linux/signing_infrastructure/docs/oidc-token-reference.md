# GitHub OIDC Token Reference

## Standard OIDC Token Payload

This document describes the structure of GitHub Actions OIDC tokens used for signing server authentication.

## Full Token Payload

```json
{
  "jti": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "sub": "repo:ROCm/TheRock:ref:refs/heads/main",
  "aud": "amd-signing-service",
  "ref": "refs/heads/main",
  "sha": "abc123def456789...",
  "repository": "ROCm/TheRock",
  "repository_owner": "ROCm",
  "repository_owner_id": "123456",
  "repository_id": "789012",
  "run_id": "123456789",
  "run_number": "42",
  "run_attempt": "1",
  "runner_environment": "github-hosted",
  "actor": "username",
  "actor_id": "345678",
  "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
  "workflow_ref": "ROCm/TheRock/.github/workflows/multi_arch_build_native_linux_packages.yml@refs/heads/main",
  "workflow_sha": "abc123def456...",
  "head_ref": "",
  "base_ref": "",
  "event_name": "workflow_dispatch",
  "ref_type": "branch",
  "ref_protected": true,
  "repository_visibility": "public",
  "job_workflow_ref": "ROCm/TheRock/.github/workflows/multi_arch_build_native_linux_packages.yml@refs/heads/main",
  "job_workflow_sha": "abc123def456...",
  "iss": "https://token.actions.githubusercontent.com",
  "nbf": 1709736000,
  "exp": 1709737200,
  "iat": 1709736300
}
```

## Field Descriptions

### Standard OIDC Claims

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `jti` | String | JWT ID - unique identifier for this token | `"a1b2c3d4-e5f6-7890-abcd-1234567890ab"` |
| `sub` | String | Subject - identifies the workflow execution | `"repo:ROCm/TheRock:ref:refs/heads/main"` |
| `aud` | String | Audience - identifies the intended recipient | `"amd-signing-service"` |
| `iss` | String | Issuer - GitHub Actions OIDC provider | `"https://token.actions.githubusercontent.com"` |
| `iat` | Number | Issued at - timestamp when token was created | `1709736300` |
| `nbf` | Number | Not before - timestamp when token becomes valid | `1709736000` |
| `exp` | Number | Expiration - timestamp when token expires | `1709737200` (usually ~10 min from iat) |

### GitHub-Specific Claims

#### Repository Information

| Field | Type | Description | Example | Used For |
|-------|------|-------------|---------|----------|
| `repository` | String | Full repository name | `"ROCm/TheRock"` | Repository restriction |
| `repository_owner` | String | Repository owner (org or user) | `"ROCm"` | Organization-level policies |
| `repository_owner_id` | String | GitHub ID of repository owner | `"123456"` | Owner verification |
| `repository_id` | String | GitHub ID of repository | `"789012"` | Repository verification |
| `repository_visibility` | String | Repository visibility | `"public"` or `"private"` | Visibility-based policies |

#### Branch/Ref Information

| Field | Type | Description | Example | Used For |
|-------|------|-------------|---------|----------|
| `ref` | String | Git ref that triggered the workflow | `"refs/heads/main"` | Branch restriction |
| `ref_type` | String | Type of ref | `"branch"` or `"tag"` | Ref type filtering |
| `ref_protected` | Boolean | Whether ref is protected | `true` | Protected branch enforcement |
| `sha` | String | Git commit SHA | `"abc123def456..."` | Commit verification |
| `head_ref` | String | Head ref for pull requests | `""` (empty for non-PR) | PR source branch |
| `base_ref` | String | Base ref for pull requests | `""` (empty for non-PR) | PR target branch |

#### Workflow Information

| Field | Type | Description | Example | Used For |
|-------|------|-------------|---------|----------|
| `workflow` | String | Workflow file path | `".github/workflows/build.yml"` | Workflow restriction |
| `workflow_ref` | String | Workflow file with ref | `"ROCm/TheRock/.github/workflows/build.yml@refs/heads/main"` | Full workflow identification |
| `workflow_sha` | String | Commit SHA of workflow file | `"abc123def456..."` | Workflow integrity verification |
| `job_workflow_ref` | String | Job workflow reference | Same as `workflow_ref` | Job-level workflow tracking |
| `job_workflow_sha` | String | Job workflow SHA | Same as `workflow_sha` | Job-level integrity |
| `event_name` | String | Event that triggered workflow | `"workflow_dispatch"`, `"push"`, `"pull_request"` | Event-based policies |

#### Run Information

| Field | Type | Description | Example | Used For |
|-------|------|-------------|---------|----------|
| `run_id` | String | Workflow run ID | `"123456789"` | Run tracking and audit |
| `run_number` | String | Sequential run number | `"42"` | Human-readable tracking |
| `run_attempt` | String | Retry attempt number | `"1"` | Retry detection |
| `runner_environment` | String | Runner environment | `"github-hosted"` | Runner type policies |

#### Actor Information

| Field | Type | Description | Example | Used For |
|-------|------|-------------|---------|----------|
| `actor` | String | User who triggered the workflow | `"john.doe"` | User attribution and audit |
| `actor_id` | String | GitHub ID of actor | `"345678"` | User verification |

## Authorization Use Cases

### Use Case 1: Repository Restriction

**Goal:** Only allow signing from `ROCm/TheRock` repository

**Configuration:**
```json
{
  "roles": {
    "release": {
      "allowed_repositories": ["ROCm/TheRock"]
    }
  }
}
```

**Token claims checked:**
- `repository` must equal `"ROCm/TheRock"`

---

### Use Case 2: Branch Restriction

**Goal:** Only allow release key usage from main and release branches

**Configuration:**
```json
{
  "roles": {
    "release": {
      "allowed_refs": [
        "refs/heads/main",
        "refs/heads/release/*"
      ]
    }
  }
}
```

**Token claims checked:**
- `ref` must match one of the patterns (supports wildcards via fnmatch)

---

### Use Case 3: Workflow Restriction

**Goal:** Only allow official build workflow to sign packages

**Configuration:**
```json
{
  "roles": {
    "release": {
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml"
      ]
    }
  }
}
```

**Token claims checked:**
- `workflow` must match exactly

**Security benefit:** Prevents malicious workflows from signing packages even if they have access to the repository.

---

### Use Case 4: Protected Branch Enforcement

**Goal:** Only allow signing from protected branches

**Implementation:** (requires custom code)
```python
def authorize_protected_branch_only(payload):
    if not payload.get('ref_protected', False):
        return False, "Only protected branches can sign with this key"
    return True, "Authorized"
```

**Token claims checked:**
- `ref_protected` must be `true`

---

### Use Case 5: Event-Based Policies

**Goal:** Prevent signing from pull request workflows

**Configuration:**
```json
{
  "roles": {
    "release": {
      "denied_event_names": ["pull_request", "pull_request_target"]
    }
  }
}
```

**Token claims checked:**
- `event_name` must NOT be in denied list

**Security benefit:** Prevents PR authors from signing packages with release keys.

---

### Use Case 6: Dynamic Role Assignment

**Goal:** Automatically determine role based on branch

**Configuration:**
```json
{
  "oidc_role_mapping": {
    "refs/heads/main": "release",
    "refs/heads/release/*": "release",
    "refs/heads/*": "development",
    "refs/pull/*": "development",
    "refs/tags/v*": "release"
  }
}
```

**Token claims used:**
- `ref` determines the role
- Role then determines which key is used

**Workflow:** No need to specify role in workflow - server determines it automatically.

---

## Token Validation Process

### 1. Signature Verification

```python
# Server fetches GitHub's public keys (JWKS)
jwks_client = PyJWKClient("https://token.actions.githubusercontent.com/.well-known/jwks")

# Get signing key for this token
signing_key = jwks_client.get_signing_key_from_jwt(token)

# Verify signature
payload = jwt.decode(token, signing_key.key, algorithms=["RS256"])
```

**What this proves:**
- Token was issued by GitHub Actions (not forged)
- Token content hasn't been tampered with
- All claims are authentic

### 2. Standard Claims Validation

```python
jwt.decode(
    token,
    signing_key.key,
    algorithms=["RS256"],
    audience="amd-signing-service",  # Must match
    issuer="https://token.actions.githubusercontent.com"  # Must match
)
```

**What this proves:**
- Token is intended for this signing server (audience check)
- Token was issued by GitHub (issuer check)
- Token hasn't expired (exp check)
- Token is valid now (nbf check)

### 3. Custom Claims Authorization

```python
# Check repository
if payload['repository'] not in allowed_repositories:
    return 403, "Repository not authorized"

# Check branch
if not any(fnmatch(payload['ref'], pattern) for pattern in allowed_refs):
    return 403, "Branch not authorized"

# Check workflow
if payload['workflow'] not in allowed_workflows:
    return 403, "Workflow not authorized"
```

**What this proves:**
- Request comes from authorized repository
- Request comes from authorized branch
- Request comes from authorized workflow file

---

## Audit Log Example

When using OIDC tokens, audit logs include full context:

```json
{
  "timestamp": "2024-03-06T10:30:00.123Z",
  "auth_type": "oidc",
  "action": "SIGNED",
  "success": true,

  "repository": "ROCm/TheRock",
  "repository_owner": "ROCm",
  "ref": "refs/heads/main",
  "sha": "abc123def456...",

  "workflow": ".github/workflows/multi_arch_build_native_linux_packages.yml",
  "job_workflow_ref": "ROCm/TheRock/.github/workflows/multi_arch_build_native_linux_packages.yml@refs/heads/main",

  "actor": "john.doe",
  "run_id": "123456789",
  "run_number": "42",
  "run_attempt": "1",
  "event_name": "workflow_dispatch",

  "role": "release",
  "key_id": "therock-release@amd.com",
  "digest_algo": "SHA256",
  "package_name": "rocm-hip-runtime-8.0.0.rpm",
  "package_hash": "abc123...",

  "client_ip": "140.82.112.50"
}
```

This enables answering questions like:
- Who signed this package? → `actor`
- From which workflow? → `workflow`
- From which commit? → `sha`
- From which branch? → `ref`
- Which run? → `run_id` (link to GitHub Actions UI)

---

## Security Considerations

### What OIDC Tokens Prove

✅ **Proven cryptographically:**
- Token was issued by GitHub Actions
- All claims are authentic (signed by GitHub)
- Token was intended for this audience
- Token is within validity period

✅ **Enforceable by signing server:**
- Repository is authorized
- Branch/ref is authorized
- Workflow file is authorized
- Rate limits per repository/user

### What OIDC Tokens Don't Prove

❌ **Not proven:**
- Workflow file content is correct (could be malicious code in workflow)
- Package content is legitimate (server signs what it receives)
- User authorization within organization (GitHub controls `actor` claim)

**Mitigation:**
- Use `allowed_workflows` to restrict to reviewed workflow files
- Use `ref_protected` check to ensure workflow on protected branches
- Use code review process for workflow changes
- Monitor audit logs for anomalies

---

## Token Lifetime

OIDC tokens have short lifetime:
- **Issued at (iat):** Current time
- **Not before (nbf):** Usually same as iat
- **Expiration (exp):** Usually iat + 10 minutes

**Implications:**
- Token must be used within 10 minutes
- Replay attacks have 10-minute window
- No manual revocation needed (auto-expires)
- No token rotation required

---

## Testing

### Inspect Token Locally

```bash
# Get token from GitHub Actions
OIDC_TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
  "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')

# Decode token (note: this doesn't verify signature)
echo $OIDC_TOKEN | cut -d. -f2 | base64 -d | jq .
```

### Verify Token with Server

```bash
# Send test signing request
curl -X POST https://signing-server/sign \
  -H "Authorization: Bearer $OIDC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "'$(echo "test" | base64)'",
    "key_id": "therock-dev@amd.com",
    "digest_algo": "SHA256",
    "armor": true
  }'
```

---

## References

- [GitHub OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [GitHub OIDC Token Claims](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect#understanding-the-oidc-token)
- [OpenID Connect Core Specification](https://openid.net/specs/openid-connect-core-1_0.html)
- [JWT RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519)
