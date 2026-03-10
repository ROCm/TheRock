# Package Signing Infrastructure Proposal
## Secure ROCm Package Distribution

---

## SLIDE 1: Title Slide

**Package Signing Infrastructure for TheRock**

Secure, Scalable, and Efficient GPG Signing Solution

*ROCm Platform Team*
*March 2026*

---

## SLIDE 2: Executive Summary

**Challenge:**
- TheRock builds native Linux packages (DEB, RPM) but does NOT sign them
- Users cannot verify package authenticity
- Security risk for enterprise deployments

**Solution:**
- Implement dedicated signing server with remote GPG signing
- Use innovative gpgshim architecture for 250x data reduction
- Support for dev, nightly, and release builds with separate keys

**Timeline:** 6 weeks to production deployment

---

## SLIDE 3: Latest Updates (March 2026)

**OIDC Authentication Implemented** ✅
- No secrets stored in GitHub (OIDC auto-generated)
- RS256 cryptographic verification by GitHub
- Workflow/repository/branch restriction
- 10-minute token expiration (vs 24 hours for JWT)
- Backward compatible with JWT

**Integrated Metadata Signing** ✅
- DEB Release file signing built into `upload_package_repo.py`
- RPM repomd.xml signing built into `upload_package_repo.py`
- Correct order: Sign AFTER generation, BEFORE upload
- Optional parameters (backwards compatible)

**What Gets Signed:**
| Item | Signed? | Where |
|------|---------|-------|
| RPM packages | ✅ Yes | Workflow (gpgshim + rpmsign) |
| RPM repomd.xml | ✅ Yes | upload_package_repo.py |
| DEB Release file | ✅ Yes | upload_package_repo.py |
| DEB packages | ❌ No | N/A (not standard for DEB) |

---

## SLIDE 4: Proposal Comparison

### Proposal 1: Azure Runners with Private Keys ❌ REJECTED

**Architecture:**
- Store GPG private keys in GitHub Secrets
- GitHub Actions runners import keys during build
- Sign packages directly on build runner

**Security Concerns:**
| Risk | Impact | Severity |
|------|--------|----------|
| Collaborator Compromise | Modify workflows to exfiltrate keys | 🔴 HIGH |
| Self-Hosted Runner Persistence | Keys leaked between jobs via /tmp | 🔴 VERY HIGH |
| Third-Party Actions | Compromised actions can steal secrets | 🟡 MEDIUM |

**Verdict:** ❌ **Rejected due to unacceptable security risks**

---

## SLIDE 4: Proposal Comparison (cont.)

### Proposal 2: POC - Dedicated Signing Server ✅ RECOMMENDED

**Architecture:**
- Dedicated on-prem signing server with GPG keys in isolated keyring
- GitHub Actions runners use gpgshim client to call signing API
- Zero private keys stored in GitHub infrastructure
- JWT Bearer token authentication with RBAC

**Security Advantages:**
- ✅ Keys never leave secure signing server
- ✅ Role-based access control per release type
- ✅ Comprehensive audit logging
- ✅ Rate limiting prevents abuse
- ✅ Zero external dependencies (Python stdlib only)

**Verdict:** ✅ **Approved - Production Ready**

---

## SLIDE 5: POC Architecture Overview (Updated with OIDC)

```
┌─────────────────────────────────────────────────────────────────┐
│                  GitHub Actions Runner                          │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐       │
│  │ Build RPM    │  │  gpgshim    │  │ upload_package_  │       │
│  │ Build DEB    │─>│  (client)   │  │ repo.py          │       │
│  └──────────────┘  └──────┬──────┘  └────────┬─────────┘       │
│                           │                  │                  │
│  ┌────────────────────────┴──────────────────┘                  │
│  │ Get OIDC Token from GitHub (10 min expiry)                   │
│  │ audience=amd-signing-service                                 │
│  └───────────────────────┬───────────────────────────────────┐  │
└──────────────────────────┼───────────────────────────────────┼──┘
                           │                                   │
               HTTPS POST /sign (OIDC)          HTTPS POST /sign (OIDC)
               4KB RPM header                   Release/repomd.xml
                           │                                   │
                           ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│            On-Prem Signing Server (Isolated)                    │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐       │
│  │   Python    │  │  Auth Module     │  │  GPG Keyring │       │
│  │   Server    │──│  OIDC (RS256) +  │──│   (Vault)    │       │
│  │  (stdlib)   │  │  JWT (fallback)  │  │              │       │
│  └─────────────┘  └──────────────────┘  └──────────────┘       │
│                                                                  │
│  Security:                                                       │
│  • TLS 1.2+                  • RBAC + Workflow Restriction       │
│  • Rate Limiting             • Audit Logging                     │
│  • OIDC Signature Verify     • No Secrets in GitHub              │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │ 256 bytes GPG signature
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│        Signed Packages + Metadata → S3 Bucket                   │
│  • RPM packages (signed)     • repomd.xml.asc (signed)          │
│  • DEB packages (unsigned)   • InRelease + Release.gpg (signed) │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **gpgshim:** Drop-in GPG replacement (signs RPM packages)
- **upload_package_repo.py:** Integrated metadata signing (DEB/RPM)
- **Signing Server:** Python stdlib HTTP server (zero dependencies)
- **Auth Module:** OIDC (primary) + JWT (fallback) + workflow restriction
- **GPG Keyring:** Isolated on signing server only

---

## SLIDE 6: Detailed Architecture Blocks

### Client Side (GitHub Actions)
```
┌──────────────────────────────────────┐
│         gpgshim Client               │
├──────────────────────────────────────┤
│ • Drop-in GPG replacement            │
│ • Python 3.6+ stdlib only            │
│ • Zero external dependencies         │
│ • JWT Bearer token auth              │
│ • ppid-based signature caching       │
│ • Exponential backoff retry          │
│ • TLS certificate validation         │
└──────────────────────────────────────┘
```

### Server Side (On-Prem)
```
┌──────────────────────────────────────┐
│      Signing Server (signing-server.py)           │
├──────────────────────────────────────┤
│ • Multi-threaded HTTP server         │
│ • JWT token validation               │
│ • Role-based authorization           │
│ • Rate limiting (requests/hour)      │
│ • Request size limits (10KB)         │
│ • Read timeout (slowloris defense)   │
│ • Thread semaphore (max concurrent)  │
│ • Audit logging (JSON)               │
│ • TLS 1.2+ support                   │
└──────────────────────────────────────┘
```

---

## SLIDE 7: Communication Flow - RPM Signing

### Step-by-Step Process

**1. rpmsign invokes gpgshim (First Call - Header)**
```
rpmsign → gpgshim reads RPM header (~4KB) from stdin
```

**2. gpgshim sends to signing server**
```http
POST /sign
Authorization: Bearer <JWT>
{
  "data": "base64(header)",  # 4KB
  "key_id": "therock-dev@amd.com"
}
```

**3. Server validates and signs**
```
• Verify JWT → Check role → Check rate limit
• Run GPG with isolated keyring
• Return signature (~256 bytes)
```

**4. gpgshim caches signature**
```bash
/tmp/gpgshim-cache-{ppid}.sig
```

**5. rpmsign calls gpgshim again (Second Call - Full RPM 1GB)**
```
• gpgshim finds cached signature by ppid
• NO SERVER CALL - reuses cached signature
• Returns signature to rpmsign
```

**6. RPM signed with minimal network transfer**
```
Total: 4KB + 256 bytes = ~4.3KB (instead of 1GB+)
```

---

## SLIDE 8: Communication Flow Diagram

```
┌─────────────┐
│   rpmsign   │
│  (parent)   │
└──────┬──────┘
       │
       ├────── Call 1: Sign Header (4KB) ──────┐
       │                                        │
       │  ┌──────────────┐       HTTPS         ▼
       │  │   gpgshim    │ ─────────────> ┌──────────┐
       │  │  (pid=123)   │                │  Signing │
       │  └──────┬───────┘ <───────────── │  Server  │
       │         │            Signature    └──────────┘
       │         │              (256B)
       │         ▼
       │    Cache: /tmp/gpgshim-cache-ppid.sig
       │
       ├────── Call 2: Sign Full RPM (1GB) ────┐
       │                                        │
       │  ┌──────────────┐                     │
       │  │   gpgshim    │  Check cache        │
       │  │  (pid=124)   │  ✅ Found!          │
       │  └──────┬───────┘  NO SERVER CALL     │
       │         │         Use cached sig       │
       │         ▼                              │
       │    Return signature                    │
       │                                        │
       ▼
   Signed RPM
```

**Network Transfer:** 4.3KB instead of 1GB+ = **250x reduction**

---

## SLIDE 9: Security Architecture

### Authentication: GitHub OIDC (Primary) + JWT (Fallback)

**OIDC Token Structure (RS256 - Cryptographically Stronger):**
```
Header.Payload.GitHub-RS256-Signature

Payload: {
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/build_native_linux_packages.yml",
  "actor": "username",
  "run_id": "123456789",
  "aud": "amd-signing-service",
  "exp": 1709472600  // 10 minutes
}
```

**Security Features:**
- ✅ **RS256 signed by GitHub** (cryptographically verified, tamper-proof)
- ✅ **No secrets in GitHub** (OIDC token auto-generated)
- ✅ **Short expiration** (10 minutes - minimal exposure window)
- ✅ **Workflow restriction** (only authorized repos/branches/workflows)
- ✅ **Fallback to JWT** for backward compatibility

**Token Generation:**
```yaml
# Automatic in GitHub Actions (no secrets needed!)
OIDC_TOKEN=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
  "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" | jq -r '.value')
```

---

## SLIDE 10: Security Architecture (cont.)

### Authorization: Role-Based Access Control + OIDC Workflow Restriction

**Configuration:** `config/authorization.json`
```json
{
  "oidc_role_mapping": {
    "refs/heads/main": "release",
    "refs/heads/release/*": "release",
    "refs/heads/*": "development"
  },
  "roles": {
    "development": {
      "allowed_repositories": ["ROCm/TheRock", "ROCm/rockrel"],
      "allowed_refs": ["refs/heads/*", "refs/pull/*"],
      "allowed_workflows": [
        ".github/workflows/build_native_linux_packages.yml",
        ".github/workflows/multi_arch_build_native_linux_packages.yml"
      ],
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000
    },
    "release": {
      "allowed_repositories": ["ROCm/TheRock", "ROCm/rockrel"],
      "allowed_refs": ["refs/heads/main", "refs/heads/release/*"],
      "allowed_workflows": [".github/workflows/build_native_linux_packages.yml"],
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000
    }
  }
}
```

**OIDC Authorization Checks:**
1. ✅ Is `repository` in allowed list? (ROCm/TheRock, ROCm/rockrel)
2. ✅ Is `ref` (branch) allowed? (main, release/*, etc.)
3. ✅ Is `workflow` in allowed list?
4. ✅ Is `key_id` allowed for this role?
5. ✅ Is `digest_algo` allowed for this role?
6. ✅ Has client exceeded rate limit?

---

## SLIDE 11: Security Features Summary

| Layer | Feature | Implementation |
|-------|---------|----------------|
| **Transport** | TLS 1.2+ | SSL context with minimum version enforcement |
| **Authentication** | JWT Tokens | HMAC-SHA256 signed, per-client secrets |
| **Authorization** | RBAC | Per-role key restrictions, digest algorithm restrictions |
| **Rate Limiting** | Per-client quotas | Sliding window (requests/hour) |
| **DoS Protection** | Request limits | 10KB max request size, 10s read timeout |
| **Concurrency** | Thread semaphore | Max concurrent signing operations |
| **Audit** | JSON logging | All requests logged with timestamp, client, key, IP |
| **Input Validation** | Regex + sanitization | Prevents injection, directory traversal |
| **Key Isolation** | GNUPGHOME | Keys never exposed to GitHub Actions |

**Result:** Defense-in-depth security with zero keys in GitHub infrastructure

---

## SLIDE 12: Data Transfer Improvements

### Problem: Naive Approach
```
┌──────────────┐
│  1GB RPM     │ ────────────> Signing Server
└──────────────┘   1GB upload

Total network: 1GB+ per RPM
```

### Solution: gpgshim ppid Caching
```
Call 1 (Header):
┌──────────────┐
│  4KB Header  │ ────────────> Signing Server
└──────────────┘   4KB upload  │
                               │ GPG signs
                               ▼
                           256B signature ──> Cache by ppid

Call 2 (Full RPM):
┌──────────────┐
│  1GB RPM     │ ──> Read stdin ──> Use cached signature
└──────────────┘     (consumed)     (NO SERVER CALL)

Total network: 4KB + 256B = ~4.3KB per RPM
```

**Improvement: 250x reduction in network transfer**

---

## SLIDE 13: Performance Comparison

### Network Transfer Savings

| Package Size | Naive Approach | gpgshim Approach | Savings |
|--------------|----------------|------------------|---------|
| 100 MB RPM | 100 MB | 4.3 KB | **23,255x** |
| 500 MB RPM | 500 MB | 4.3 KB | **116,279x** |
| 1 GB RPM | 1 GB | 4.3 KB | **250,000x** |
| **LLVM RPM (largest)** | **1.2 GB** | **4.3 KB** | **🔥 300,000x** |

### Time Savings (assuming 1 Gbps network)

| Package | Upload Time (Naive) | Upload Time (gpgshim) | Time Saved |
|---------|---------------------|----------------------|------------|
| 100 MB | 0.8 seconds | **0.00003 seconds** | 0.8s |
| 1 GB | 8 seconds | **0.00003 seconds** | **~8s** |
| 10 packages | 80 seconds | **0.0003 seconds** | **~80s** |

**Impact:** Multi-arch builds with 50+ packages complete signing in **seconds** instead of **minutes**

---

## SLIDE 14: Why gpgshim Works - RPM v4 Signing

### RPM Signature Format (v4)

RPM packages have **two signatures**:
1. **Header signature** - Signs RPM header metadata (~4KB)
2. **Payload signature** - Signs entire RPM (header + payload)

### rpmsign Behavior

`rpmsign` calls `gpg` **twice**:
```bash
# Call 1: Sign header only
gpg --detach-sign header_data  # ~4KB

# Call 2: Sign full RPM
gpg --detach-sign full_rpm  # 1GB+
```

**Both calls produce the SAME signature** (because header contains hash of payload)

### gpgshim Optimization

- **Call 1:** Send header to server → Get signature → Cache by ppid
- **Call 2:** Reuse cached signature (don't send full RPM)
- **Result:** Only 4KB sent to server, not 1GB+

**Key Insight:** Exploit rpmsign's dual-call pattern with ppid-based caching

---

## SLIDE 15: DEB vs RPM Signing

### RPM Signing (Individual Packages)
```
┌─────────────┐
│ rocm-hip.rpm│ ──> gpgshim ──> Signing Server ──> Signed RPM
└─────────────┘

Each RPM has embedded GPG signature
yum/dnf verifies signature during installation
```

### DEB Signing (Repository Metadata Only)
```
┌──────────────────┐
│ upload_package_  │ ──> Generates Release file
│ repo.py          │      (contains SHA256 of all .deb)
└────────┬─────────┘              │
         │                        ▼
         │               ┌─────────────────┐
         │               │ sign_deb_       │ ──> Signing Server
         │               │ release_file()  │           │
         │               └─────────────────┘           ▼
         │                                      InRelease
         │                                      Release.gpg
         └──────────────> Upload to S3

Individual .deb files: NOT SIGNED
apt verifies: Release signature → Package checksums
```

**Difference:**
- RPM: Sign every package individually + repomd.xml metadata
- DEB: Sign only Release file (contains checksums of all packages)

**New:** Metadata signing integrated into `upload_package_repo.py`

---

## SLIDE 16: Integrated Metadata Signing Flow

### Repository Metadata Signing (New Implementation)

**Previous approach:** Separate workflow steps for metadata signing
**New approach:** Integrated into `upload_package_repo.py`

**Flow:**
```python
# upload_package_repo.py
def regenerate_deb_metadata_from_s3(..., gpg_signing_server="", gpg_server_token=""):
    # 1. Upload packages to S3
    upload_to_s3(package_dir, bucket, prefix)

    # 2. Generate Release file (merge with existing S3 metadata)
    generate_release_file_with_checksums(release_file, job_type, dists_dir)

    # 3. Sign Release file if credentials provided
    if gpg_signing_server and gpg_server_token:
        sign_deb_release_file(release_file, gpg_signing_server, gpg_server_token)
        # Creates: InRelease (clearsigned) + Release.gpg (detached)

    # 4. Upload signed metadata to S3
    upload_deb_metadata_to_s3(s3, bucket, prefix, dists_dir, release_file)
```

**For RPM:**
```python
def regenerate_rpm_metadata_from_s3(..., gpg_signing_server="", gpg_server_token=""):
    # 1. Upload packages to S3
    # 2. Generate repomd.xml (mergerepo_c)
    mergerepo_c --repo old_repo --repo new_repo --outputdir merged_repo

    # 3. Sign repomd.xml if credentials provided
    if gpg_signing_server and gpg_server_token:
        sign_rpm_repomd_files(merged_arch_dir, gpg_signing_server, gpg_server_token)
        # Creates: repomd.xml.asc (detached signature)

    # 4. Upload signed metadata to S3
```

**Benefits:**
- ✅ Correct order: Sign AFTER metadata creation, BEFORE upload
- ✅ Cleaner workflow (fewer steps)
- ✅ Optional parameters (backwards compatible)
- ✅ Reusable functions for both DEB and RPM

---

## SLIDE 17: Implementation Timeline

| Week | Phase | Deliverables | Status |
|------|-------|-------------|--------|
| **1-2** | **Signing Server Deployment** | • Production server running<br>• Keys configured<br>• OIDC + JWT auth setup<br>• TLS certificates | ✅ **READY** |
| **2-3** | **RPM Package Signing** | • gpgshim integrated into workflow<br>• RPM packages signed<br>• Public keys on S3 | ✅ **COMPLETE** |
| **3** | **Repository Metadata Signing** | • Integrated into upload_package_repo.py<br>• DEB Release file signing<br>• RPM repomd.xml signing<br>• Metadata uploaded to S3 | ✅ **COMPLETE** |
| **4** | **Testing & Validation** | • All tests passing<br>• End-to-end verification<br>• User installation testing | 🔄 **NEXT** |
| **5** | **Production Hardening** | • Monitoring setup<br>• Alerting configured<br>• Token rotation procedure | 📋 **PENDING** |
| **6** | **Documentation** | • User docs<br>• Admin runbooks<br>• Internal documentation | ✅ **COMPLETE** |

**Total: 6 weeks (~1.5 months)**
**Current Progress: Week 3 - 50% Complete**

---

## SLIDE 17: Further Improvements - Short Term

### 1. Monitoring & Observability

**Add Prometheus metrics:**
```python
@app.route('/metrics')
def metrics():
    return prometheus_metrics_export()
```

**Metrics to track:**
- `signing_requests_total{role, status}` - Total requests by role/status
- `signing_duration_seconds` - Signing operation latency
- `rate_limit_exceeded_total{client_id}` - Rate limit violations
- `authentication_failures_total` - Auth failures (security monitoring)

**Integration:**
- Prometheus → Grafana dashboards
- CloudWatch for AWS deployment
- PagerDuty alerts for failures

---

## SLIDE 18: Further Improvements - Mid Term

### 2. Distributed Deployment

**Current:** Single signing server
**Future:** Multiple servers with shared state

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Signing    │     │  Signing    │     │  Signing    │
│  Server 1   │     │  Server 2   │     │  Server 3   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                    ┌──────▼──────┐
                    │    Redis    │
                    │ (rate limit │
                    │  + cache)   │
                    └─────────────┘
```

**Benefits:**
- Load balancing across multiple servers
- High availability (redundancy)
- Shared rate limiting via Redis
- Horizontal scaling

---

## SLIDE 19: Further Improvements - Long Term

### 3. Hardware Security Module (HSM)

**Current:** GPG keys in file-based keyring
**Future:** Keys in HSM (YubiHSM2, AWS CloudHSM)

**Benefits:**
- Keys never extractable from HSM
- FIPS 140-2 Level 3 compliance
- Tamper-evident hardware
- Key backup/recovery procedures

### 4. Advanced Audit Features

**Current:** JSON log files
**Future:** Database-backed audit trail

**Features:**
- SQL queries for audit analysis
- Compliance reporting (SOC2, ISO 27001)
- Automatic anomaly detection
- SIEM integration (Splunk, ELK)

---

## SLIDE 20: Further Improvements - Advanced

### 5. Automated Key Rotation

**Implement automated rotation every 90 days:**

```python
# Scheduled task (cron/GitHub Actions)
def rotate_signing_key(release_type):
    # 1. Generate new GPG key
    new_key = generate_gpg_key(f"therock-{release_type}-v2")

    # 2. Update signing server keyring
    import_to_server(new_key)

    # 3. Update authorization.json
    add_allowed_key(new_key.key_id)

    # 4. Keep old key for verification
    # (allow verification of previously signed packages)

    # 5. Update GitHub Secrets with transition period
    # Both old and new tokens valid for 1 week
```

**Benefits:**
- Reduced blast radius if key compromised
- Compliance with key rotation policies
- Automatic process (no manual intervention)

---

## SLIDE 21: Migration from POC to FastAPI (Future)

### When to Migrate?

**Stay with POC if:**
- ✅ Request volume < 500/sec
- ✅ Single signing server sufficient
- ✅ Team comfortable with stdlib code
- ✅ Zero dependencies requirement

**Migrate to FastAPI if:**
- 🔴 Request volume > 1000/sec (async needed)
- 🔴 Need distributed deployment (multiple servers)
- 🔴 Want OpenAPI documentation for external consumers
- 🔴 Need Prometheus metrics built-in
- 🔴 Want modern Python async patterns

### Migration Path

**Phase 1:** Continue with POC (2026 Q1-Q2)
**Phase 2:** Evaluate usage patterns (2026 Q3)
**Phase 3:** Migrate if needed (2026 Q4)

**Estimated effort:** 2-3 weeks for FastAPI migration

---

## SLIDE 22: Comparison with Industry Standards

### How POC Compares

| Feature | POC | Sigstore (Cosign) | AWS KMS | Enterprise PKI |
|---------|-----|-------------------|---------|----------------|
| **Zero dependencies** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **On-prem deployment** | ✅ Yes | ⚠️ Possible | ❌ Cloud only | ✅ Yes |
| **Cost** | ✅ Free | ✅ Free | 💰 $1/key/mo | 💰 $$$$ |
| **Setup complexity** | ✅ Low | ⚠️ Medium | ⚠️ Medium | 🔴 High |
| **apt/yum auto-verify** | ✅ Yes (GPG) | ❌ Manual | ❌ N/A | ✅ Yes |
| **Network efficiency** | ✅ 250x better | ❌ Standard | ❌ Standard | ❌ Standard |
| **FIPS compliance** | ⚠️ GPG | ✅ Yes | ✅ Yes | ✅ Yes |

**Verdict:** POC is optimal for AMD's requirements (on-prem, zero deps, GPG standard)

---

## SLIDE 23: Risk Assessment

### Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Server outage** | Low | High | • Systemd auto-restart<br>• Monitoring/alerting<br>• Backup server |
| **Key compromise** | Very Low | Critical | • Keys isolated on server<br>• Audit logging<br>• Key rotation |
| **Token leakage** | Low | Medium | • Short expiration (24h)<br>• Rate limiting<br>• Revocation procedure |
| **DoS attack** | Medium | Medium | • Request size limits<br>• Read timeouts<br>• Rate limiting |
| **Network failure** | Low | Medium | • Retry logic in gpgshim<br>• Exponential backoff |

### Security Posture

**Before:** ❌ No package signatures (trust based on HTTPS only)
**After:** ✅ Cryptographic signatures with isolated signing infrastructure

**Security improvement:** 🔒 **Significant** (from 0% to enterprise-grade)

---

## SLIDE 24: Cost-Benefit Analysis

### Implementation Costs

| Item | Cost | Notes |
|------|------|-------|
| **Server hardware** | $0 | Use existing AMD on-prem infrastructure |
| **Development time** | 6 weeks | 1 engineer @ 6 weeks |
| **Maintenance** | ~2 hrs/month | Token rotation, monitoring |
| **HSM (optional)** | $500-2000 | YubiHSM2 or CloudHSM (future) |

**Total upfront cost:** Minimal (engineering time only)

### Benefits

| Benefit | Value |
|---------|-------|
| **Security** | ✅ Cryptographic verification of all packages |
| **Compliance** | ✅ Meets enterprise security requirements |
| **Trust** | ✅ Users can verify package authenticity |
| **Network** | ✅ 250x reduction in signing traffic |
| **Automation** | ✅ Fully automated signing pipeline |

**ROI:** High (security + compliance benefits significantly outweigh costs)

---

## SLIDE 25: Success Metrics

### Key Performance Indicators

**Technical Metrics:**
- ✅ 100% of packages signed before S3 upload
- ✅ < 5 seconds signing time per package
- ✅ 99.9% signing server uptime
- ✅ 0 key compromise incidents
- ✅ < 10MB network transfer per build (vs 10GB+ before)

**Security Metrics:**
- ✅ 100% audit log coverage
- ✅ 0 unauthorized signing attempts succeed
- ✅ 100% rate limit compliance
- ✅ All tokens rotated within 90 days

**User Metrics:**
- ✅ Users can verify signatures via `rpm --checksig` / `apt verify`
- ✅ 0 user-reported signature failures
- ✅ Public key availability 99.9% (S3 uptime)

---

## SLIDE 26: Next Steps

### Immediate Actions (Week 1)

1. **Approval:** ✅ Get stakeholder sign-off on POC approach
2. **Infrastructure:** 🔧 Provision on-prem server for signing service
3. **Keys:** 🔑 Generate GPG signing keys (dev, nightly, release)
4. **Secrets:** 🔐 Generate JWT secrets and store securely

### Week 2-6 Actions

1. **Deploy:** 🚀 Deploy signing server to production
2. **Integrate:** 🔗 Modify GitHub Actions workflows
3. **Test:** 🧪 End-to-end testing (dev → nightly → release)
4. **Document:** 📝 User documentation and runbooks
5. **Monitor:** 📊 Set up monitoring and alerting
6. **Launch:** 🎉 Enable signing for all builds

### Post-Launch

1. **Monitor:** Track metrics and audit logs
2. **Iterate:** Implement improvements based on usage
3. **Evaluate:** Review performance after 3 months

---

## SLIDE 27: Questions & Discussion

**Key Decisions Needed:**

1. ✅ **Approval of POC approach** (vs rejected Azure runner approach)
2. 🔧 **Server infrastructure** - Which on-prem environment?
3. 🔑 **Key management** - File-based or HSM? (recommend file-based for v1)
4. ⏰ **Token expiration** - 24 hours or 7 days? (recommend 24h)
5. 📊 **Monitoring** - CloudWatch, Prometheus, or both? (recommend both)

**Resources Required:**

- 1 senior engineer (6 weeks)
- On-prem server (existing infrastructure)
- Security team review (2-3 hours)

**Timeline to Production:** 6 weeks

---

## SLIDE 28: Thank You

**Contact:**

- **Technical Questions:** ROCm Platform Team
- **Security Review:** Information Security Team
- **Infrastructure:** IT Operations

**Documentation:**

- POC Analysis: `signing_infrastructure/POC_ANALYSIS_AND_IMPLEMENTATION_PLAN.md`
- Security Comparison: `signing_infrastructure/SECURITY_COMPARISON.md`
- Communication Protocol: `signing_infrastructure/COMMUNICATION_PROTOCOL.md`

**POC Code:**

- Branch: `users/isparry/remote_signing`
- Commit: `83ed5f3616c8f76a9b00b4571be63fa8be3ea73d`

---

# APPENDIX SLIDES

---

## APPENDIX A: Technical Deep Dive - JWT Token

### Token Structure
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9    ← Header (base64url)
.
eyJjbGllbnRfaWQi...                     ← Payload (base64url)
.
dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1...   ← Signature (HMAC-SHA256)
```

### Decoded Header
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

### Decoded Payload
```json
{
  "client_id": "github-actions-dev",
  "role": "development",
  "iat": 1709472000,
  "exp": 1709558400
}
```

### Signature Verification
```python
message = base64url(header) + "." + base64url(payload)
signature = HMAC-SHA256(message, client_secret)
```

---

## APPENDIX B: Error Handling

### Client-Side (gpgshim) Retry Logic

```python
for attempt in range(max_retries):
    try:
        response = post_to_server(payload)
        if response.status == 200:
            return response.json()['signature']
        elif response.status == 503:
            # Server busy - exponential backoff
            sleep(backoff * (2 ** attempt) + random(0, 0.1))
        elif response.status in [401, 403, 429]:
            # Auth/authz/rate limit - don't retry
            raise SigningError(response.json()['error'])
        else:
            # Other error - retry
            sleep(backoff * (2 ** attempt))
    except NetworkError:
        # Network error - retry
        sleep(backoff * (2 ** attempt))
```

### Server-Side Error Responses

| Status | Error | Client Action |
|--------|-------|---------------|
| 401 | Invalid token | Check token, regenerate if expired |
| 403 | Forbidden | Check role permissions |
| 413 | Request too large | Package > 10KB (shouldn't happen) |
| 429 | Rate limit | Wait, reduce request rate |
| 503 | Server busy | Retry with exponential backoff |
| 500 | Internal error | Check server logs, retry |

---

## APPENDIX C: Deployment Checklist

### Pre-Deployment

- [ ] Server provisioned and accessible
- [ ] TLS certificates obtained (Let's Encrypt or internal CA)
- [ ] GPG keys generated for all release types
- [ ] JWT secrets generated and stored securely
- [ ] Authorization config reviewed and approved
- [ ] Firewall rules configured (allow GitHub Actions IP ranges)
- [ ] Monitoring/logging infrastructure ready

### Deployment

- [ ] Deploy signing server code to production server
- [ ] Import GPG keys to server keyring
- [ ] Configure systemd service
- [ ] Start signing server
- [ ] Verify health check endpoint
- [ ] Test signing with curl (all release types)
- [ ] Add GitHub Secrets (tokens, server URL)
- [ ] Update GitHub Actions workflows
- [ ] Test dev build end-to-end
- [ ] Test nightly build end-to-end
- [ ] Test release build end-to-end

### Post-Deployment

- [ ] Monitor audit logs for 48 hours
- [ ] Verify signature verification works for users
- [ ] Set up alerting for failures
- [ ] Document runbook for common issues
- [ ] Train team on token rotation procedure
- [ ] Schedule first token rotation (90 days)

---

## APPENDIX D: Troubleshooting Guide

### Common Issues

**1. gpgshim returns 401 Unauthorized**
- **Cause:** Token expired or invalid
- **Solution:** Regenerate token with `generate-token.py`, update GitHub Secret

**2. Server returns 403 Forbidden**
- **Cause:** Key not allowed for role, or wrong digest algorithm
- **Solution:** Check `config/authorization.json`, verify `allowed_keys` includes requested key

**3. Signing is slow (> 10 seconds)**
- **Cause:** Network latency or server overload
- **Solution:** Check server CPU/memory, verify network connectivity, increase thread limit

**4. Rate limit exceeded (429)**
- **Cause:** Too many requests in last hour
- **Solution:** Increase `max_requests_per_hour` in authorization config, or investigate unusual activity

**5. rpm --checksig fails to verify**
- **Cause:** Public key not imported, or wrong key used
- **Solution:** Import public key from S3, verify key ID matches signed package

### Debug Mode

Enable gpgshim debug logging:
```bash
export GPG_SHIM_DEBUG=/tmp/gpgshim.log
```

View server audit log:
```bash
tail -f /var/log/gpg-signing/audit.log | jq .
```

---

# END OF PRESENTATION
