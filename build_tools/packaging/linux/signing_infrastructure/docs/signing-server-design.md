# Signing Server — Detailed Design Document

**Project:** AMD ROCm Build System  
**Component:** Remote GPG Signing Service  
**Status:** Draft — v0.1

---

## 1. Introduction

### 1.1 Background

TheRock is an open-source CMake super-project that builds HIP and ROCm from source and publishes native Linux packages (RPM and DEB) to an S3-hosted package repository. These packages are installed by end users and automated systems on production hardware.

Enterprise Linux package distribution requires cryptographic signing. GPG-signed packages allow `rpm` and `apt` to verify that:
- The package originated from AMD/ROCm
- The package has not been modified after publication

Without signing, package managers either refuse to install the packages or present warnings that discourage adoption.

### 1.2 Problem

GPG signing requires access to a private key. In a CI/CD pipeline, the naive approaches are:

| Approach | Problem |
|----------|---------|
| Store private key in GitHub Secrets and import into build runner | Key is exposed in the build environment; any compromise of the runner exposes the key permanently |
| Sign packages manually offline | Blocks automated release pipelines; does not scale |
| Skip signing | Packages cannot be distributed through standard package manager channels that require signed metadata |

A remote signing server solves this by keeping the private key in an isolated, air-gapped environment. Build runners send only the data to be signed and receive back a signature — the key itself never leaves the signing server.

### 1.3 Scope

This document covers the design of the remote GPG signing service for ROCm Linux packages. It is scoped to:
- GPG signing of RPM packages and RPM/DEB repository metadata
- Two authorized caller types: TheRock automated CI builds and authorized operators
- AWS-hosted infrastructure in a single account

It does not cover: code signing, Windows packages, ELF binary attestation, or cross-account signing.

---

## 2. What We Are Building

A remote HTTP signing service running in an AWS private subnet, accepting signed data from authorized build runners and returning GPG signatures. The service is air-gapped — no internet egress — and access is controlled at the network layer by VPC Security Groups.

### 2.1 What Gets Signed

| Artifact | Signature Format | Produced By | Phase |
|----------|-----------------|-------------|-------|
| RPM packages (`.rpm`) | Embedded GPG signature (via `rpmsign`) | `gpgshim` intercepting `rpmsign` on build runner | 1 |
| RPM repo metadata (`repomd.xml`) | Detached ASCII signature (`repomd.xml.asc`) | `upload_package_repo.py` → `POST /sign` direct | 1 |
| DEB repo metadata (`Release`) | Clearsigned `InRelease` + detached `Release.gpg` | `upload_package_repo.py` → `POST /sign` direct | 1 |
| RPM packages, ad-hoc (no gpgshim) | Embedded GPG signature (server-side `rpmsign`) | `POST /sign-rpm` — full RPM uploaded and returned signed | 2 |

DEB package files themselves are not signed — the repository metadata signature is sufficient for `apt`.

**Why two RPM signing paths?** `gpgshim` (Phase 1) is efficient — it sends only ~4 KB regardless of RPM size — but requires `rpmsign` and `gpgshim` installed on the caller's machine. `POST /sign-rpm` (Phase 2) requires only an HTTP client, making it accessible to operators and external callers who cannot install `gpgshim`, at the cost of transferring the full RPM over the network.

### 2.2 Callers

| Caller | Mechanism | Use case |
|--------|-----------|----------|
| TheRock CI build (GitHub Actions, self-hosted EC2) | `gpgshim` for RPM signing; `upload_package_repo.py` for metadata | Automated release pipeline |
| Authorized operator (workstation via VPN) | `sign-file` CLI tool | One-off signing, key verification, emergency re-signing |

---

## 3. Architecture

### 3.1 Component Diagram

```
  TheRock CI Build Runner              Authorized Operator
  (Self-hosted EC2, build-runner SG)   (Workstation via VPN)
         │                                      │
         │  HTTP POST /sign                     │  HTTP POST /sign
         │  (private IP, port 443)              │  (via VPN to private IP)
         │                                      │
         └──────────────────┬───────────────────┘
                            │
                ┌───────────▼────────────┐
                │    Security Group      │  ← Only build-runner SG
                │    Enforcement (VPC)   │    and operator VPN range
                └───────────┬────────────┘    allowed on port 443
                            │
                ┌───────────▼────────────────────────────────┐
                │  Signing Server  (private subnet, no egress)│
                │                                            │
                │  signing-server.py                         │
                │  ┌──────────────┐  ┌─────────────────────┐│
                │  │  auth.py     │  │  GPG keyring (tmpfs) ││
                │  │  app-layer   │  │  Key: Secrets Mgr    ││
                │  │  token check │  │  at startup          ││
                │  └──────────────┘  └─────────────────────┘│
                │                                            │
                │  POST /sign → gpg --detach-sign → response │
                │  Audit log → CloudWatch Logs (VPC endpoint)│
                └────────────────────────────────────────────┘
```

**Phase 2 extends this with:**
```
         Build Runner                    Operator
              │                             │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Internal ALB      │  ← Routes to primary, fails
              │   (health checks)   │    over to secondary
              └──────┬──────┬───────┘
                     │      │
              Primary │      │ Secondary
              Server  │      │ Server
                     ▼      ▼
              (scheduled key sync from Secrets Manager)
```

### 3.2 Request Flow — RPM Package Signing

```
rpmsign                    gpgshim                  Signing Server
   │                          │                          │
   │  gpg --detach-sign        │                          │
   │  (call 1: header ~4 KB)  │                          │
   ├─────────────────────────►│                          │
   │                          │  check ppid cache        │
   │                          │  → cache miss            │
   │                          │                          │
   │                          │  POST /sign              │
   │                          │  {data, key_id, algo}    │
   │                          ├─────────────────────────►│
   │                          │                          │  gpg --detach-sign
   │                          │                          │  using tmpfs keyring
   │                          │  {signature: base64}     │
   │                          │◄─────────────────────────┤
   │                          │                          │
   │                          │  write to cache file     │
   │  256-byte signature       │  /tmp/gpgshim-<ppid>.sig│
   │◄─────────────────────────┤                          │
   │                          │                          │
   │  gpg --detach-sign        │                          │
   │  (call 2: full RPM 1GB+) │                          │
   ├─────────────────────────►│                          │
   │                          │  check ppid cache        │
   │                          │  → cache HIT             │
   │                          │  read + delete cache     │
   │  256-byte signature       │  (no network call)       │
   │◄─────────────────────────┤                          │
```

**Key efficiency:** A 1 GB RPM generates exactly one ~4 KB network request. The second `rpmsign` call costs zero bytes on the wire.

### 3.3 Request Flow — Metadata Signing (Direct API)

```
upload_package_repo.py              Signing Server
         │                               │
         │  POST /sign                   │
         │  {data: base64(Release),      │
         │   key_id, armor: true,        │
         │   clearsign: true}            │
         ├──────────────────────────────►│
         │                              │  gpg --clearsign
         │  {signature: base64}          │
         │◄──────────────────────────────┤
         │                               │
         │  write InRelease file         │
         │                               │
         │  POST /sign                   │
         │  {data: base64(Release),      │
         │   key_id, armor: true}        │
         ├──────────────────────────────►│
         │  {signature: base64}          │
         │◄──────────────────────────────┤
         │                               │
         │  write Release.gpg            │
```

---

## 4. Key Design Decisions

### 4.1 VPC Security Groups as Primary Access Control (not SigV4)

Access to the signing server is controlled at the VPC network layer. Only EC2 instances in the designated build runner security group, and the operator VPN IP range, can reach port 443 on the signing server.

**Why not SigV4:** SigV4 is designed for public-facing AWS API endpoints. For a server already air-gapped in a private subnet, Security Groups enforce the same perimeter more simply — with no per-request signing overhead in `gpgshim`, no `botocore` dependency, and no IMDS credential fetching.

**App-layer auth (Phase 2):** A lightweight pre-shared token is added on top of Security Groups as a second layer, primarily for audit traceability (distinguishing CI build calls from operator calls in logs) rather than as a security boundary.

**Trade-off acknowledged:** Security Groups control which *instances* can reach the server, not which *processes* on those instances. This is acceptable because the build runner EC2 instances are dedicated to the TheRock CI pipeline.

### 4.2 Primary + Secondary with Scheduled Key Sync

Two signing server instances run in separate Availability Zones: a primary and a hot standby secondary. Both hold identical GPG keys, fetched independently from AWS Secrets Manager on a schedule (not server-to-server sync). There is no sync channel between them, which preserves the air-gap on each instance.

In Phase 1, clients connect to the primary directly by private IP. In Phase 2, an internal ALB routes traffic and fails over automatically to the secondary when the primary fails health checks.

### 4.3 GPG Key Storage — Options Considered and Decision

Three approaches were evaluated for storing and protecting the GPG private key in AWS. This section documents all three, their trade-offs, and the rationale for the chosen approach.

---

#### Option 1 — KMS Asymmetric Key (private key lives inside KMS hardware)

AWS KMS supports native asymmetric signing using RSA or ECC keys. The private key is generated inside KMS hardware and **never leaves the HSM** — not even AWS can extract it. Signing calls go directly to KMS, which performs the operation internally and returns a signature.

```
Build Runner
     │  kms:Sign(data, key-id)
     ▼
AWS KMS HSM  ← private key never leaves
     │  Returns raw RSA/ECDSA signature bytes
     ▼
Build Runner  →  raw signature bytes
```

| Aspect | Detail |
|--------|--------|
| **Private key ever on disk** | Never |
| **Private key ever in RAM** | Never |
| **GPG / OpenPGP compatible** | ❌ No — KMS returns raw RSA/ECDSA bytes, not OpenPGP packet format |
| **Works with rpm --checksig** | ❌ Fails — RPM expects OpenPGP signature packets |
| **Works with apt verify** | ❌ Fails — same reason |
| **Operational complexity** | Low |
| **Monthly cost** | Low — pay per API call |
| **FIPS 140-2 Level 3** | ✅ Yes |
| **Instant revocation** | ✅ Disable CMK |

**Why rejected:** KMS asymmetric signing produces raw cryptographic signatures. `rpm --checksig`, `gpg --verify`, and `apt` all require OpenPGP-format signatures (`-----BEGIN PGP SIGNATURE-----`). Bridging the gap would require reimplementing the OpenPGP packet format — effectively rewriting part of GPG. Not viable for standard package distribution.

---

#### Option 2 — AWS CloudHSM + PKCS#11 GPG Engine

AWS CloudHSM provides a dedicated hardware security module. GPG can be configured with a PKCS#11 engine that delegates signing operations to CloudHSM. The private key is generated inside and never leaves the HSM, but the output is a valid OpenPGP-format signature — GPG handles the packet wrapping, CloudHSM provides the raw cryptographic operation.

```
Signing Server
     │  gpg --detach-sign (via PKCS#11 engine)
     ▼
CloudHSM Cluster
     │  Private key in HSM hardware
     │  Raw RSA operation happens inside HSM
     │  Returns raw signature bytes to GPG
     ▼
GPG wraps into OpenPGP packet format
     │
     ▼
Valid .asc signature  ✅
```

| Aspect | Detail |
|--------|--------|
| **Private key ever on disk** | Never |
| **Private key ever in RAM** | Never (HSM signs internally) |
| **GPG / OpenPGP compatible** | ✅ Yes — PKCS#11 engine + GPG produces standard OpenPGP |
| **Works with rpm --checksig** | ✅ Yes |
| **Works with apt verify** | ✅ Yes |
| **Operational complexity** | High — PKCS#11 driver, CloudHSM cluster management, custom AMI |
| **Monthly cost** | ~$1,100+ (2× HSM at $1.50/hour for HA, minimum 2 required) |
| **FIPS 140-2 Level 3** | ✅ Yes — certified |
| **Instant revocation** | ✅ Delete key from HSM |

**Why not chosen (now):** CloudHSM is the only approach that gives HSM-level protection AND GPG compatibility. However, the cost ($1,100+/month) and operational complexity (PKCS#11 setup, cluster management, custom AMI) are not justified by the current threat model. **Captured as the upgrade path if a FIPS 140-2 Level 3 compliance requirement is introduced.**

---

#### Option 3 — AWS Secrets Manager + KMS CMK (chosen)

The GPG private key is stored in AWS Secrets Manager, with Secrets Manager configured to use a Customer Managed KMS Key (CMK) for encryption at rest. Secrets Manager internally uses envelope encryption — it generates a data key, encrypts the GPG private key with it, and wraps the data key with the CMK. The GPG private key exists on AWS-managed encrypted storage as ciphertext only.

At server startup, the signing server calls `GetSecretValue`. Secrets Manager transparently calls `kms:Decrypt` on the data key and returns the plaintext GPG key in the API response. The server imports it into a `tmpfs`-backed keyring and immediately discards the plaintext.

```
Key provisioning (offline, once per key):
  1. Generate GPG key pair on isolated machine
  2. aws secretsmanager create-secret \
       --kms-key-id alias/amd-signing-gpg-key \
       --secret-string file://private.asc
     ↳ Secrets Manager calls kms:GenerateDataKey internally
     ↳ Encrypts private.asc with data key, stores ciphertext
     ↳ Wraps data key with CMK, stores encrypted data key
     ↳ Plaintext data key discarded — never persisted
  3. shred -u private.asc  ← plaintext gone

Server startup (every restart):
  1. secretsmanager:GetSecretValue
     ↳ Secrets Manager calls kms:Decrypt on data key internally
     ↳ Returns plaintext GPG key in API response
  2. echo "$KEY" | gpg --import  (into tmpfs GNUPGHOME)
  3. unset KEY  ← plaintext gone from memory
```

| Aspect | Detail |
|--------|--------|
| **Private key ever on disk** | As AES-256 ciphertext only — useless without KMS CMK access |
| **Private key ever in RAM** | Yes — briefly during `GetSecretValue` response and `gpg --import` |
| **GPG / OpenPGP compatible** | ✅ Yes — standard GPG, no special drivers |
| **Works with rpm --checksig** | ✅ Yes |
| **Works with apt verify** | ✅ Yes |
| **Operational complexity** | Low — standard `boto3` + `gpg --import` |
| **Monthly cost** | ~$2 (Secrets Manager $0.40/secret + KMS $1/CMK + API calls) |
| **FIPS 140-2 Level 3** | ❌ Not certified |
| **Instant revocation** | ✅ Disable CMK — all future `GetSecretValue` calls fail immediately |

---

#### Comparison Summary

| | KMS Asymmetric | CloudHSM + PKCS#11 | **Secrets Manager + CMK** |
|--|---------------|-------------------|--------------------------|
| **Private key ever on disk** | Never | Never | Encrypted ciphertext only |
| **Private key ever in RAM** | Never | Never | Yes — during startup import |
| **GPG / OpenPGP compatible** | ❌ | ✅ | ✅ |
| **Works with rpm / apt** | ❌ | ✅ | ✅ |
| **Operational complexity** | Low | High | **Low** |
| **Monthly cost (approx.)** | Low | ~$1,100+ | **~$2** |
| **FIPS 140-2 Level 3** | ✅ | ✅ | ❌ |
| **Instant revocation** | ✅ | ✅ | **✅** |
| **Audit trail** | CloudTrail | CloudHSM logs + CloudTrail | **CloudTrail** |

---

#### Decision: Secrets Manager + KMS CMK

**Chosen for Phase 1 and Phase 2.** It is the only approach that is simultaneously GPG-compatible, low-cost, and low-complexity. The key protection level is appropriate for the current threat model: the real attack surface is IAM credential theft and EC2 instance compromise — not raw disk access to Secrets Manager storage. Both are mitigated by Security Groups, no public IPs, and CloudTrail alerting regardless of key storage approach.

**Upgrade path to CloudHSM:** If a FIPS 140-2 Level 3 compliance requirement is introduced, only the key loading code in `signing-server.py` changes — the rest of the pipeline (gpgshim, upload_package_repo.py, GitHub Actions workflow) is completely unaffected.

---

#### Threat Coverage with Chosen Approach

Security controls operate at two distinct layers. It is important to understand which layer each control protects — KMS and Secrets Manager protect the key **before it reaches the server**; network and OS controls protect the key **while it is on the running server**. Neither layer substitutes for the other.

**Layer 1 — Key at rest and in distribution (KMS + Secrets Manager)**

| Threat | What attacker gets | Protected by |
|--------|-------------------|-------------|
| Raw access to AWS Secrets Manager storage | AES-256 ciphertext — useless without CMK | KMS CMK |
| Secrets Manager API call without server role | `GetSecretValue` denied by resource policy | IAM resource policy |
| IAM credential theft (non-server role) | Cannot call `GetSecretValue` | IAM resource policy |
| IAM credential theft (server role) | Gets plaintext key via `GetSecretValue` | CloudTrail alarm — detected, not prevented |
| EBS snapshot of signing server volume | tmpfs is RAM-only — not in EBS snapshot | tmpfs mount |
| AWS insider accesses Secrets Manager storage | AES-256 ciphertext only | KMS CMK |
| Key distributed via insecure channel (scp, ansible) | N/A — Secrets Manager is the only distribution path | SM architecture |

**Layer 2 — Key on the running server (network + OS controls)**

KMS and IAM do NOT protect against OS-level access to the signing server. Anyone with a shell on the instance can read the tmpfs keyring directly. The controls below are therefore the primary defence for the running key — they must be treated as mandatory, not optional hardening:

| Threat | What attacker gets | Protected by |
|--------|-------------------|-------------|
| SSH access to signing server | Full access to tmpfs keyring and process memory | **No SSH rule in `sg-signing-server`** — port 22 not open to anyone |
| AWS SSM Session Manager access | Same as SSH — full shell | **`ssm:StartSession` and `ssm:SendCommand` explicitly denied** in `role-signing-server` IAM policy |
| SSRF attack stealing instance credentials | IAM role credentials via IMDS | **IMDSv2 enforced** — `--http-tokens required` on instance metadata |
| Unauthorised process on signing server | Can call `/sign` or read keyring | Signing server is single-purpose; no other processes should run |
| Shell access via application exploit | Code execution within signing server process | Minimal Python dependencies; input validation on all request fields |
| Shell access (if somehow obtained) | Can read `/var/gpg-keyring` | This IS a compromise — incident response required; revoke CMK |

**What no software control can prevent:**
If an attacker obtains OS-level access to the running signing server, the GPG private key in the tmpfs keyring is readable. This is true of every key management approach — CloudHSM, KMS, HSM cards — because any running signing process must have access to key material to perform signing operations. The goal is to make OS-level access impossible, not to protect against it after the fact.

---

#### In-Memory GPG Keyring (tmpfs)

After the key is fetched and imported, it lives in a `tmpfs`-backed GNUPGHOME directory. `tmpfs` is RAM-backed — it does not survive a reboot and nothing is written to the EBS volume. Secrets Manager + KMS is the authoritative store; the tmpfs keyring is a working copy valid only for the current server session.

### 4.4 Rate Limiting (Per-Instance, Sliding Window)

The current implementation uses an in-memory sliding window counter (per client, per process) from `auth.py`. This operates independently on each server instance.

- **Normal operation (primary only):** rate limit enforced per-instance correctly
- **During failover (both instances serving):** effective limit is doubled transiently — this is acceptable as it is a brief window and the limits are sized conservatively
- **Rate limit purpose:** safety valve against runaway jobs, not a hard organizational policy

### 4.5 gpgshim — Two-Call Optimization

`rpmsign` calls `gpg` twice per package: once for the header (~4 KB), once for the header+payload (up to 1 GB+). `gpgshim` intercepts both calls transparently:

- **Call 1:** sends data to signing server, caches signature in `/tmp/gpgshim-cache-<ppid>.sig`
- **Call 2:** reads and returns the cached signature, deletes the cache file — no network call

This means signing a 1 GB RPM costs the same network transfer as signing a 4 KB file.

---

### 4.6 Operational Lifecycle — What Requires Manual Intervention vs What Is Automatic

A key goal of this design is that the signing pipeline runs without human involvement. This section documents exactly what is automated, what is one-time setup, and what remains a manual operational task.

#### One-Time Server Setup (manual, done once per EC2 instance)

These steps are performed when the signing server EC2 instance is first provisioned — never again unless the instance is replaced:

```
1. Mount tmpfs for GPG keyring (added to /etc/fstab — auto-mounts on every reboot):
     echo "tmpfs /var/gpg-keyring tmpfs size=64m,mode=0700 0 0" >> /etc/fstab
     mount /var/gpg-keyring

2. Install dependencies:
     apt install gnupg2 python3 python3-boto3
     (or equivalent for Amazon Linux)

3. Deploy signing-server.py as a systemd service:
     systemctl enable signing-server
     systemctl start signing-server

4. Verify /health returns 200:
     curl -k https://localhost/health
```

After this, the instance is self-managing — systemd restarts the server on failure, tmpfs is remounted on reboot, and the key is reloaded automatically.

#### Every Server Start or Restart (fully automatic — no human involvement)

Triggered by: instance reboot, systemd restart after crash, or manual `systemctl restart signing-server`.

```
systemd starts signing-server.py
         │
         ▼
  [~200ms] boto3: secretsmanager.get_secret_value('signing/gpg/therock-release')
           → Secrets Manager calls kms:Decrypt internally
           → Returns plaintext GPG private key in API response
         │
         ▼
  [~50ms]  gpg --import (GNUPGHOME=/var/gpg-keyring)
           → Key loaded into tmpfs keyring
           → Plaintext variable immediately unset from memory
         │
         ▼
  Server begins listening on HTTPS :443
  GET /health → 200 OK  (key confirmed present in keyring)
```

Secrets Manager is called **once per server startup** — not per signing request.

#### Every Signing Request (fully automatic — ~50-100ms per request)

Triggered by: `rpmsign` invoking `gpgshim`, or `upload_package_repo.py` calling `POST /sign` directly.

```
Client sends POST /sign
  {data, key_id, digest_algo, armor}
         │
         ▼
  [~5ms]   Validate request (key_id regex, size limit, rate limit)
         │
         ▼
  [~50ms]  gpg --detach-sign
           using GNUPGHOME=/var/gpg-keyring
           (key already in memory — no Secrets Manager call)
         │
         ▼
  Return {signature: base64} to client
  Write audit log entry to CloudWatch
```

No AWS API calls happen during signing — the key is already in the tmpfs keyring from startup.

#### Phase 2 — Scheduled Key Sync (fully automatic — every 6 hours)

A background thread re-fetches the key from Secrets Manager on a schedule and atomically reloads the keyring — handles key rotation without a server restart:

```
Background thread wakes (every 6 hours)
         │
         ▼
  Fetch new key from Secrets Manager (same as startup sequence)
         │
         ▼
  Import into a temporary GNUPGHOME directory
  Verify at least one valid key is present
         │
         ▼
  Atomically swap: replace live GNUPGHOME with new one
  Delete old GNUPGHOME
         │
         ▼
  Log: "Key reloaded successfully"
  (Server continues serving requests throughout — no downtime)
```

#### Administrative Operations — No Shell Access Required

A key design goal is that all routine administration is performed via AWS APIs, not via shell access to the signing server. The following table covers every administrative scenario and how it is handled without SSH or SSM:

| Operation | Who | How (no shell access needed) |
|-----------|-----|------------------------------|
| **GPG key rotation** | Key provisioner | `aws secretsmanager put-secret-value` with new key from an isolated machine → Phase 2: server picks up automatically on next scheduled sync. Phase 1: reboot instance via `aws ec2 reboot-instances` — systemd restarts server, new key loaded from Secrets Manager |
| **KMS CMK rotation** | AWS admin | AWS KMS re-encrypts the Secrets Manager data key automatically — no server interaction needed |
| **Emergency key revocation** | AWS admin | `aws kms disable-key` → all future `GetSecretValue` calls fail immediately → server cannot reload key on next restart |
| **signing-server.py code update** | DevOps | Build new AMI with updated code → launch new EC2 instance (same IAM role, same SG, same fstab) → verify `/health` → terminate old instance. Never patch a running instance |
| **OS security patches** | DevOps | Same as code update — replace instance from freshly patched AMI |
| **TLS certificate renewal** | DevOps | Store certificate in Secrets Manager or ACM → instance replacement picks it up automatically |
| **Configuration change** (rate limits, auth config) | DevOps | Update config in S3 or bake into new AMI → instance replacement |
| **One-off operator signing** | Authorized operator | `sign-file` CLI tool via VPN — calls `POST /sign` directly, no server access needed |
| **Server instance replacement** | DevOps | Launch new EC2 (same IAM role + fstab) → systemd starts automatically → key loads from Secrets Manager → old instance terminated |

---

#### Break-Glass — Emergency Shell Access

For genuine emergencies where the server is in a bad state requiring direct investigation before replacement (e.g., diagnosing a signing failure that cannot be reproduced on a fresh instance), a **controlled break-glass procedure** provides time-limited, fully audited shell access:

```
Break-glass procedure:

1. Approval
   Requires sign-off from 2 authorized personnel (4-eyes principle)
   Incident ticket created before access is granted

2. Enable access (audited in CloudTrail)
   Admin temporarily modifies role-signing-server IAM policy
   to add ssm:StartSession for a specific session
   (The IAM deny is a policy — not permanent; can be modified
    by an IAM admin when genuinely needed)

3. Session
   AWS SSM Session Manager used — no SSH key, no port 22
   Session is fully recorded to CloudWatch Logs / S3
   Time-limited: access reverted after session ends

4. Key protection during session
   If key compromise is suspected: disable KMS CMK before
   granting access — server cannot be used for signing
   during the investigation

5. Revert
   IAM policy deny on SSM restored immediately after session
   Incident ticket updated with findings
   Instance typically replaced after investigation
```

The critical point: **removing SSH from the Security Group and adding an IAM deny on SSM does not make the server permanently inaccessible** — it makes access an explicit, audited, approved act rather than a routine convenience. Every step of the break-glass procedure appears in CloudTrail.

---

## 5. Improvements Over Baseline

### 5.1 Current Baseline

The existing signing process works as follows:
- Build runners build packages and upload unsigned artifacts to S3
- Only **release builds** are signed — dev and nightly builds are not signed at all
- Signing is performed **manually** by an authorized engineer after the build completes, using GPG installed on an **in-house signing server** with no external network exposure
- The signed packages are then manually uploaded to the S3 repository
- The signing server is a self-managed internal machine — no cloud infrastructure, no automation, no audit trail

This process works but does not scale with the build pipeline and introduces manual steps that can delay releases and create inconsistency between builds.

---

### 5.2 Improvements

| Area | Current Baseline | After |
|------|-----------------|-------|
| **Signing trigger** | Manual — engineer runs signing after build completes | **Autonomous** — signing happens automatically as part of the CI/CD pipeline, no human interaction required |
| **Build types signed** | Release builds only — dev and nightly are unsigned | **All configured build tiers** (dev, nightly, release) signed automatically when `release_type` is set |
| **Release pipeline speed** | Signing is a manual gate — release blocked until an engineer is available | **No manual gate** — signing completes within the CI run; S3 upload follows immediately |
| **Signing server hosting** | In-house server, manually maintained, no HA | **AWS-managed EC2** in a private VPC subnet; Phase 2 adds primary + secondary with automatic ALB failover |
| **Key storage** | GPG private key on the in-house signing server's disk | **AWS Secrets Manager** encrypted with KMS CMK; plaintext exists only in RAM during `gpg --import` at startup |
| **Key access control** | Physical/network access to the in-house server | **IAM resource policy** on Secrets Manager secret — only `role-signing-server` can retrieve the key; enforced by AWS |
| **Key audit trail** | No record of when the key was used or by whom | **CloudTrail** records every `secretsmanager:GetSecretValue` and `kms:Decrypt` call — timestamp, caller identity, key used |
| **Key revocation** | Physically remove or overwrite key on the server | **Disable KMS CMK** — all future decrypts fail immediately across all server instances simultaneously |
| **Signing audit trail** | No record of which packages were signed, when, or by which build | **Structured JSON audit log** per signing request → CloudWatch Logs; includes source IP, key used, digest algo, latency |
| **Network exposure** | In-house server — no external exposure (same as new design) | **AWS private subnet** — no internet gateway, no public IP; Security Groups restrict access to build runner IPs only |
| **One-off / operator signing** | Engineer manually runs GPG on the in-house server | **sign-file CLI tool** — authorized operator signs a specific file via VPN without needing access to the signing server host |
| **Scalability** | Single server; one engineer can sign at a time | **Concurrent signing** — thread semaphore allows up to 10 parallel signing operations; Phase 2 adds a second server |
| **Network transfer (RPM)** | Full signing toolchain runs locally on in-house server | **gpgshim** sends only the ~4 KB RPM header to the signing server; 250× reduction vs full RPM transfer |

---

### 5.3 What Does Not Change

| Area | Note |
|------|------|
| **Network isolation of signing server** | The in-house server has no external exposure; the AWS signing server is also in a private subnet with no internet access — same posture |
| **GPG toolchain** | `gpg` and `rpmsign` are still used; the signing server runs standard GnuPG 2.x |
| **Signature format** | OpenPGP format signatures — fully compatible with existing `rpm --checksig` and `apt` verification |
| **Public key distribution** | How end users obtain the public key to verify packages is unchanged — same keyserver or static URL process |

---

## 6. HTTP API

### `POST /sign`

Request body:
```json
{
  "data":        "<base64-encoded bytes to sign>",
  "key_id":      "therock-release@amd.com",
  "digest_algo": "SHA256",
  "armor":       true,
  "clearsign":   false
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `data` | Yes | Base64-encoded bytes. For RPM: the header section piped by rpmsign. For metadata: the full Release or repomd.xml file. |
| `key_id` | Yes | GPG key identifier (email or hex key ID). Must match a key in the server keyring and be permitted for the caller's role. |
| `digest_algo` | No | Hash algorithm. Default: `SHA256`. Supported: `SHA256`, `SHA512`. |
| `armor` | No | Return ASCII-armored signature. Default: `false`. Required for metadata signatures. |
| `clearsign` | No | Produce clearsigned output (data + signature in one block). Default: `false`. Required for DEB `InRelease`. |

Response (success `200`):
```json
{
  "signature":   "<base64-encoded signature>",
  "key_id":      "therock-release@amd.com",
  "digest_algo": "SHA256"
}
```

Error codes: `400` bad request, `401` missing/invalid app token (Phase 2), `403` key not permitted for caller, `429` rate limit exceeded, `503` server busy (retry).

### `GET /health`

Returns `200 OK` with `{"status": "ok"}` when the server is running and the GPG keyring is loaded. Used by ALB health checks (Phase 2) and monitoring.

---

## 7. Sample Communication Flows

These examples show the full request path for each caller type, including the HTTPS wire format. Both examples use Phase 2 app-layer tokens (`Authorization: Bearer`). In Phase 1, the `Authorization` header is omitted — Security Groups provide the only access control.

---

### 7.1 RPM Package Signing via gpgshim

This flow is triggered automatically when the GitHub Actions workflow runs `rpmsign`. The developer or workflow author never calls the signing server directly — `gpgshim` handles it transparently.

#### Environment setup (in GitHub Actions workflow step)

```bash
# Workflow sets these before invoking rpmsign
export GPG_SIGNING_SERVER="https://10.0.2.45/sign"   # signing server private IP (Phase 1)
                                                       # or ALB DNS name (Phase 2)
export GPG_KEY_ID="therock-release@amd.com"
export GPG_SERVER_TOKEN="ey..."                        # pre-shared token from Secrets Manager
export GPG_VERIFY_SSL="false"                          # Phase 1: self-signed cert
export GPG_TIMEOUT="30"
export GPG_MAX_RETRIES="5"

# Install gpgshim as the gpg binary for this session
cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
chmod +x ~/.local/bin/gpgshim
export PATH="$HOME/.local/bin:$PATH"
```

#### rpmsign invocation (workflow step)

```bash
rpmsign --addsign \
  --define "%__gpg $HOME/.local/bin/gpgshim" \
  --define "_gpg_name therock-release@amd.com" \
  rocm-6.4.0-1.x86_64.rpm
```

#### What gpgshim sends — Call 1 (RPM header, ~4 KB)

`rpmsign` pipes the RPM header section to `gpgshim` via stdin. `gpgshim` reads it, checks the ppid cache (miss), and sends:

```
POST /sign HTTP/1.1
Host: 10.0.2.45
Content-Type: application/json
Authorization: Bearer ey...
User-Agent: gpgshim/2.0
Content-Length: 312

{
  "data":        "AAAABA...(base64-encoded RPM header, ~4 KB)...",
  "key_id":      "therock-release@amd.com",
  "digest_algo": "SHA256",
  "armor":       false
}
```

#### Signing server response — Call 1

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 398

{
  "signature":   "iQEzBAABCAAdFiEE...(base64-encoded 256-byte GPG signature)...",
  "key_id":      "therock-release@amd.com",
  "digest_algo": "SHA256"
}
```

`gpgshim` decodes the signature, writes it to the output file, and caches it to `/tmp/gpgshim-cache-<ppid>.sig`.

#### What gpgshim does — Call 2 (full RPM, 1 GB+)

`rpmsign` pipes the full RPM body (~1 GB) to `gpgshim` stdin. `gpgshim` reads and discards it (must consume stdin), checks the ppid cache — **hit**. Reads the cached signature, deletes the cache file, writes the signature to the output file.

**No network call is made on Call 2.** The signing server never sees the full RPM body.

#### Audit log entry written by signing server (CloudWatch)

```json
{
  "timestamp":    "2026-06-16T10:23:41Z",
  "action":       "SIGNED",
  "caller":       "therock-release",
  "source_ip":    "10.0.1.12",
  "key_id":       "therock-release@amd.com",
  "digest_algo":  "SHA256",
  "armor":        false,
  "clearsign":    false,
  "status":       200,
  "latency_ms":   87,
  "data_size_bytes": 4096
}
```

---

### 7.2 Ad-hoc Signing Request (Operator, Direct HTTPS)

An authorized operator re-signs a specific metadata file outside of the CI pipeline. The operator is on VPN, has AWS SSO credentials, and has fetched their token from Secrets Manager.

#### Fetch the operator token (one-time per session)

```bash
# Operator fetches their pre-shared token from Secrets Manager
# Requires AWS SSO login and VPN connection
SIGNING_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id signing/tokens/operator \
  --query SecretString \
  --output text)
```

#### Option A — Using the sign-file CLI tool

```bash
# Sign a DEB Release file, producing both InRelease (clearsigned) and Release.gpg (detached)

# Clearsigned InRelease
python signing_infrastructure/tools/sign-file \
  --server  https://10.0.2.45 \
  --key-id  therock-release@amd.com \
  --file    /tmp/repo/dists/jammy/Release \
  --clearsign \
  --output  /tmp/repo/dists/jammy/InRelease \
  --token   "$SIGNING_TOKEN"

# Detached ASCII signature Release.gpg
python signing_infrastructure/tools/sign-file \
  --server  https://10.0.2.45 \
  --key-id  therock-release@amd.com \
  --file    /tmp/repo/dists/jammy/Release \
  --armor \
  --output  /tmp/repo/dists/jammy/Release.gpg \
  --token   "$SIGNING_TOKEN"
```

#### Option B — Raw curl (for debugging or one-liners)

```bash
# Encode the file to base64
DATA_B64=$(base64 -w0 /tmp/repo/dists/jammy/Release)

# Send signing request
curl --silent --fail \
  --request POST \
  --url "https://10.0.2.45/sign" \
  --insecure \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $SIGNING_TOKEN" \
  --data "{
    \"data\":        \"$DATA_B64\",
    \"key_id\":      \"therock-release@amd.com\",
    \"digest_algo\": \"SHA256\",
    \"armor\":       true,
    \"clearsign\":   true
  }" \
| python3 -c "
import sys, json, base64
resp = json.load(sys.stdin)
sys.stdout.buffer.write(base64.b64decode(resp['signature']))
" > /tmp/repo/dists/jammy/InRelease

echo "Signed. Verifying..."
gpg --verify /tmp/repo/dists/jammy/InRelease
```

#### Wire request — ad-hoc signing

```
POST /sign HTTP/1.1
Host: 10.0.2.45
Content-Type: application/json
Authorization: Bearer ey...
User-Agent: curl/7.88.1
Content-Length: 2847

{
  "data":        "T2JqZWN0OiBkZWIu...(base64-encoded Release file, ~2 KB)...",
  "key_id":      "therock-release@amd.com",
  "digest_algo": "SHA256",
  "armor":       true,
  "clearsign":   true
}
```

#### Server response — ad-hoc signing

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1423

{
  "signature":   "LS0tLS1CRUdJTiBQR1AgU0lHTkVEIE1FU1NBR0UtLS0tLQo...(base64-encoded ASCII-armored clearsigned block)...",
  "key_id":      "therock-release@amd.com",
  "digest_algo": "SHA256"
}
```

Decoded, the `signature` field contains a complete `-----BEGIN PGP SIGNED MESSAGE-----` block ready to write as `InRelease`.

#### Audit log entry — ad-hoc signing

```json
{
  "timestamp":    "2026-06-16T14:05:12Z",
  "action":       "SIGNED",
  "caller":       "operator",
  "source_ip":    "10.10.0.55",
  "key_id":       "therock-release@amd.com",
  "digest_algo":  "SHA256",
  "armor":        true,
  "clearsign":    true,
  "status":       200,
  "latency_ms":   124,
  "data_size_bytes": 2048
}
```

---

### 7.3 Common Error Responses

```bash
# 401 — missing or invalid token
HTTP/1.1 401 Unauthorized
{"error": "Unauthorized: missing or invalid token"}

# 403 — valid token but key_id not permitted for this caller
HTTP/1.1 403 Forbidden
{"error": "Forbidden: caller 'therock-dev' is not authorized for key 'therock-release@amd.com'"}

# 429 — rate limit exceeded
HTTP/1.1 429 Too Many Requests
{"error": "Rate limit exceeded: caller 'operator' limit is 100 requests/hour"}

# 503 — server busy (all signing threads occupied); gpgshim will retry with backoff
HTTP/1.1 503 Service Unavailable
{"error": "Server busy, try again later"}
```

---

## 8. Security Considerations

Security for this system operates at three distinct layers. Understanding which layer each control belongs to is important — controls at one layer do not substitute for controls at another.

---

### 8.1 Layer 1 — Network Perimeter (who can reach the server)

These controls determine whether a request ever reaches the signing server. They are enforced by AWS infrastructure, not by server code.

| Threat | Control | Enforced by |
|--------|---------|-------------|
| Unauthorised host calling `/sign` | `sg-signing-server` allows port 443 inbound from `sg-build-runner` and operator VPN CIDR only | AWS Security Group |
| Internet-facing attack | No public IP, no internet gateway, no NAT on signing server subnet | VPC routing |
| Signing server exfiltrating data | No outbound except to VPC endpoints (SM, KMS, CloudWatch) — Security Group outbound rules | AWS Security Group |
| Build runner calling signing server from wrong process | Security Groups are instance-level — any process on the build runner EC2 can reach the server; mitigated by dedicated single-purpose runner instances | Operational policy |

---

### 8.2 Layer 2 — Key at Rest and in Distribution (protecting the key before it reaches the server)

These controls protect the GPG private key while it is stored in AWS and during the fetch-to-import path at server startup. KMS and Secrets Manager operate entirely at this layer.

| Threat | Control | Enforced by |
|--------|---------|-------------|
| GPG private key stored on any disk in plaintext | Key stored in Secrets Manager only; fetched into tmpfs RAM at startup; never written to EBS | Architecture |
| Secrets Manager access by unauthorized AWS principal | Resource-based policy on each secret restricts `GetSecretValue` to `role-signing-server` | IAM resource policy |
| EBS snapshot of signing server exposes key | tmpfs is RAM-backed — not captured in EBS snapshots | tmpfs |
| AWS insider reads Secrets Manager storage | Encrypted with KMS CMK — ciphertext only on AWS-managed storage | KMS CMK |
| IAM credential theft (server role) | Attacker can call `GetSecretValue` and get plaintext key — detected but not prevented | CloudTrail alarm |

**Important limitation:** KMS and Secrets Manager protect the key *before* it reaches the server. Once the server has fetched the key into tmpfs at startup, KMS and IAM provide no further protection. OS-level access to the server bypasses them entirely.

---

### 8.3 Layer 3 — Server Instance Hardening (protecting the key while it is on the running server)

This is the most critical layer and the one most commonly underestimated. Once the GPG key is in the tmpfs keyring, **anyone with OS-level access to the signing server can read it** — regardless of KMS policies, IAM roles, or Secrets Manager configurations. The controls below must be treated as mandatory requirements, not optional hardening.

| Threat | Control | How to enforce |
|--------|---------|---------------|
| SSH access to signing server | No port 22 inbound rule in `sg-signing-server` — not open to any principal, including admins | Security Group |
| AWS SSM Session Manager shell access | Explicitly deny `ssm:StartSession` and `ssm:SendCommand` in `role-signing-server` IAM policy | IAM deny policy |
| SSRF attack stealing instance metadata credentials | IMDSv2 enforced: `aws ec2 modify-instance-metadata-options --http-tokens required` | Instance metadata config |
| Persistent attacker installing backdoor | Immutable infrastructure — updates replace the instance from a new AMI, never patch in place | Operational policy |
| Unnecessary attack surface from extra services | Signing server runs only `signing-server.py` — no other services, no package manager access in prod | AMI hardening |
| Code execution via application exploit | Input validation on all request fields; `key_id` validated against strict regex before GPG subprocess | Application code |

**Fundamental limit — applicable to all key management approaches:**  
If an attacker achieves OS-level access to a running signing server, the GPG private key in tmpfs is accessible. This is not unique to this design — it applies equally to CloudHSM (the PKCS#11 handle is in the process), hardware tokens, and any other approach where a live process must perform signing operations. The correct response to an OS-level compromise is: disable the KMS CMK immediately, rotate the GPG key pair, and distribute the new public key to package consumers.

---

### 8.4 Application-Level Controls

| Threat | Control |
|--------|---------|
| `key_id` command injection into GPG subprocess | Strict regex validation: `[a-zA-Z0-9@.\-_ <>]+`, max 256 chars — request rejected on mismatch |
| Runaway build job exhausting signing capacity | Thread semaphore (max 10 concurrent) + per-source-IP rate limiting (sliding window) |
| Slow-read / slowloris attack | Socket read timeout: 10 seconds |
| Oversized request payload (memory exhaustion) | Request body size limit: 10 KB — rejected with `413` |
| Replay of a captured signature | GPG signatures are bound to specific data — a replayed signature for different data fails `gpg --verify` |
| Audit trail gaps | Every request (success and failure) written to CloudWatch Logs — source IP, key used, digest algo, latency, HTTP status |
