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

### 3.1 Component Diagram — Full System View (All Phases)

This diagram shows the complete signing microservice as it looks when all phases are implemented. Phase-specific elements are annotated. Subsections below describe what each phase adds.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  EXTERNAL — Outside the signing service boundary                             ║
║                                                                              ║
║   ┌─────────────────────────────┐    ┌──────────────────────────────────┐   ║
║   │  TheRock CI Build Runner    │    │  Authorized Operator             │   ║
║   │  (Self-hosted EC2)          │    │  (Corporate workstation + VPN)   │   ║
║   │                             │    │                                  │   ║
║   │  Clients:                   │    │  Clients:                        │   ║
║   │  • gpgshim (RPM signing)    │    │  • sign-file CLI (one-off)       │   ║
║   │  • upload_package_repo.py   │    │  • curl / direct HTTP (Phase 2)  │   ║
║   │    (metadata signing)       │    │                                  │   ║
║   │  • App token in header [P2] │    │  • App token in header [P2]      │   ║
║   │                             │    │                                  │   ║
║   │  IAM Role: build-runner     │    │  IAM Role: operator (assumed)    │   ║
║   └──────────────┬──────────────┘    └───────────────┬──────────────────┘   ║
║                  │                                   │                      ║
║                  │  [EXT-1] HTTPS POST /sign         │  [EXT-2] HTTPS POST  ║
║                  │  TLS 1.2+, port 443               │  /sign or /sign-rpm  ║
║                  │  {tier, artifact, data}            │  port 443, via VPN   ║
╚══════════════════│═══════════════════════════════════│══════════════════════╝
                   │                                   │
╔══════════════════│═══════════════════════════════════│══════════════════════╗
║  AWS VPC — Trust boundary enforced by Security Groups + ALB [Phase 2]       ║
║                  │                                   │                      ║
║                  └──────────────┬────────────────────┘                      ║
║                                 │                                           ║
║              ┌──────────────────▼──────────────────────┐                   ║
║              │  sg-signing-server (Security Group)      │  ← Phase 1        ║
║              │  Inbound: TCP 443 from sg-build-runner   │    primary control ║
║              │           TCP 443 from operator VPN CIDR │                   ║
║              │  Outbound: TCP 443 to VPC endpoints only │                   ║
║              │  No SSH. No SSM. No internet. No NAT.   │                   ║
║              └──────────────────┬──────────────────────┘                   ║
║                                 │                                           ║
║         ┌───────────────────────▼─────────────────────────┐                ║
║         │  Internal ALB  [Phase 2]                         │                ║
║         │  HTTPS :443, ACM cert, internal DNS only         │                ║
║         │  Health check: GET /health (every 30s)           │                ║
║         │  Routes to healthy instance, auto-failover       │                ║
║         └────────────────┬──────────────┬─────────────────┘                ║
║                          │              │                                   ║
║           ┌──────────────▼──┐    ┌──────▼──────────────┐                   ║
║           │  PRIMARY EC2    │    │  SECONDARY EC2 [P2] │                   ║
║           │  AZ-1           │    │  AZ-2               │                   ║
║           │  (Phase 1+2)    │    │  (Phase 2 only)     │                   ║
║           │                 │    │                     │                   ║
║  ┌────────┴─────────────────┴──────────────────────┐  │                   ║
║  │  SIGNING SERVER  (private subnet, no egress)     │  │  ← same on both   ║
║  │                                                  │  │    instances       ║
║  │  ┌─────────────────────────────────────────────┐ │  │                   ║
║  │  │  signing-server.py  (HTTPS :443, TLS 1.2+) │ │  │                   ║
║  │  │                                             │ │  │                   ║
║  │  │  GET  /health     keyring loaded check      │ │  │                   ║
║  │  │  POST /sign       sign data → signature     │ │  │                   ║
║  │  │  POST /sign-rpm   sign full RPM [Phase 2]   │ │  │                   ║
║  │  │                                             │ │  │                   ║
║  │  │  ┌──────────────┐  ┌─────────────────────┐ │ │  │                   ║
║  │  │  │  auth.py      │  │  GPG Keyring (tmpfs)│ │ │  │                   ║
║  │  │  │  rate limit   │  │  /var/gpg-keyring   │ │ │  │                   ║
║  │  │  │  [P1: by IP]  │  │  RAM only, not EBS  │ │ │  │                   ║
║  │  │  │  [P2: by token│  │  Loaded from SM     │ │ │  │                   ║
║  │  │  │   + authz]    │  │  at startup + sync  │ │ │  │                   ║
║  │  │  └──────────────┘  └─────────────────────┘ │ │  │                   ║
║  │  │                                             │ │  │                   ║
║  │  │  authorization.json:                        │ │  │                   ║
║  │  │    key_aliases:       tier → GPG email      │ │  │                   ║
║  │  │    artifact_profiles: artifact → GPG params │ │  │                   ║
║  │  │    roles:             rate limits per tier  │ │  │                   ║
║  │  └─────────────────────────────────────────────┘ │  │                   ║
║  │                                                  │  │                   ║
║  │  IAM: role-signing-server                        │  │                   ║
║  │  SM:GetSecretValue, KMS:Decrypt, CW:PutLogs      │  │                   ║
║  └──────────────────────────────────────────────────┘  │                   ║
║                  │              │              │         │                   ║
║      [INT-1]     │  [INT-2]     │  [INT-3]    │  [INT-4]│  [Phase 2]        ║
║  SM:GetSecret    │  KMS:Decrypt │  CW:PutLogs │  Scheduled key sync        ║
║  (startup)       │  (via SM)    │  (per req)  │  (every 6h, both servers)  ║
║                  │              │              │                            ║
║  ┌───────────────▼─┐ ┌──────────▼───┐ ┌───────▼──────────────────────┐   ║
║  │ Secrets Manager  │ │  AWS KMS     │ │  AWS CloudWatch Logs         │   ║
║  │                  │ │              │ │                              │   ║
║  │ signing/gpg/dev  │ │ CMK:         │ │  /amd/signing-server/audit   │   ║
║  │ signing/gpg/     │ │ alias/amd-   │ │                              │   ║
║  │   nightly        │ │ signing-     │ │  Structured JSON per request │   ║
║  │ signing/gpg/     │ │ gpg-key      │ │  source_ip, tier, artifact   │   ║
║  │   release        │ │              │ │  latency_ms, success         │   ║
║  │                  │ │ Encrypts SM  │ │  90-day retention            │   ║
║  │ Tokens [P2]:     │ │ data key at  │ │                              │   ║
║  │ signing/tokens/  │ │ rest. Decrypt│ │  CloudWatch Alarms [Phase 2] │   ║
║  │   dev/nightly/   │ │ by SM on     │ │  • error rate > 10%          │   ║
║  │   release/       │ │ GetSecret    │ │  • rate limit hits           │   ║
║  │   operator       │ │              │ │  • health check failures      │   ║
║  └──────────────────┘ └──────────────┘ └──────────────────────────────┘   ║
║                                                                              ║
║  All outbound traffic via VPC Interface Endpoints (PrivateLink).            ║
║  No NAT. No internet gateway. Traffic never leaves the AWS network.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

### 3.1a Interface Summary

| ID | Interface | Type | Direction | Protocol | Auth | Phase |
|----|-----------|------|-----------|----------|------|-------|
| EXT-1 | Build runner → Signing server | External inbound | Inbound | HTTPS POST `/sign` | VPC Security Group (sg-build-runner) | 1 |
| EXT-2 | Operator → Signing server | External inbound | Inbound | HTTPS POST `/sign` or `/sign-rpm` | VPC Security Group (operator VPN CIDR) | 1 |
| EXT-1/2 | Any caller → ALB → Signing server | External inbound | Inbound | HTTPS POST, ACM cert | SG + app token header | 2 |
| INT-1 | Signing server → Secrets Manager | Internal outbound | Outbound | HTTPS via VPC endpoint | IAM `secretsmanager:GetSecretValue` | 1 |
| INT-2 | Signing server → KMS | Internal outbound | Outbound | HTTPS via VPC endpoint | IAM `kms:Decrypt` (invoked by SM) | 1 |
| INT-3 | Signing server → CloudWatch Logs | Internal outbound | Outbound | HTTPS via VPC endpoint | IAM `logs:PutLogEvents` | 1 |
| INT-4 | Both servers → SM (scheduled sync) | Internal outbound | Outbound | HTTPS via VPC endpoint | IAM `secretsmanager:GetSecretValue` | 2 |

**No external outbound interfaces.** The signing server has no internet egress, no public IP, and no access to S3, GitHub, or any service outside the VPC endpoints above.

---

### 3.1b What Each Phase Delivers

#### Phase 1 — Working signing pipeline, single server

| Component | What's delivered |
|-----------|-----------------|
| Signing server EC2 | Single instance in private subnet, self-signed TLS cert |
| Access control | VPC Security Groups only — no app-layer auth token |
| GPG key storage | Secrets Manager + KMS CMK, tmpfs keyring at startup |
| API | `POST /sign` with tier+artifact (simplified) or legacy key_id params |
| Clients | `gpgshim` for RPM, `upload_package_repo.py` for metadata, `sign-file` for operators |
| Rate limiting | Sliding window per source IP |
| Audit | Structured JSON to stdout → systemd journal (local only) |
| Observability | `GET /health`, manual `journalctl` inspection |
| High availability | None — single server; outage blocks signing |

#### Phase 2 — Production hardening

| Component | What's added |
|-----------|-------------|
| Secondary server | Second EC2 in a different AZ, identical config |
| ALB | Internal ALB with health checks, auto-failover to secondary |
| TLS | Self-signed cert replaced by ACM cert on ALB (trusted, no `--no-verify-ssl`) |
| App-layer auth | Pre-shared token per caller tier stored in Secrets Manager; `Authorization: Bearer` header required |
| Rate limiting | Keyed by token identifier instead of source IP (accurate per-tier limits) |
| Scheduled key sync | Background thread re-fetches SM every 6 hours; atomic keyring reload without restart |
| `POST /sign-rpm` | Server-side RPM signing — accept full RPM, run `rpmsign`, return signed RPM (no gpgshim needed on caller) |
| CloudWatch | Log agent forwards journal to CloudWatch; alarms on error rate, rate limit hits, health failures |
| Tokens in SM | `signing/tokens/{dev,nightly,release,operator}` secrets added alongside GPG keys |

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

### 4.5 gpgshim — What It Is and the Two-Call Optimization

#### What is gpgshim?

`gpgshim` is a lightweight Python script deployed on **build runners** (not the signing server) that acts as a drop-in replacement for the `gpg` binary. It exists for one specific reason: `rpmsign` calls `gpg` as a subprocess to produce signatures and embed them in RPM files. There is no way to redirect this subprocess call to an HTTP endpoint without intercepting it at the binary level.

`gpgshim` pretends to be `gpg`. When `rpmsign` calls it, `gpgshim`:
1. Reads the data piped from `rpmsign` via stdin
2. Forwards it to the signing server via `POST /sign`
3. Returns the signature bytes to `rpmsign` via the output file

`rpmsign` never knows it talked to a remote server — it sees a binary that behaves like `gpg`. The signing server never needs `rpmsign` installed — it only runs `gpg --detach-sign` directly.

`gpgshim` is only needed for RPM package signing. For repository metadata (`repomd.xml`, `Release`), `upload_package_repo.py` calls `POST /sign` directly — no `gpgshim` involved.

#### Two-Call Optimization

`rpmsign` calls `gpg` (and therefore `gpgshim`) **twice** per package:
- **Call 1:** pipes the RPM header section (~4 KB) for signing
- **Call 2:** pipes the full RPM body (up to 1 GB+) for signing

`gpgshim` intercepts both calls transparently using a per-process cache:

- **Call 1:** sends ~4 KB to signing server, receives signature, writes to output file, caches signature in `/tmp/gpgshim-cache-<ppid>.sig`
- **Call 2:** reads and discards the full RPM body from stdin (must consume it), returns the cached signature from Call 1, deletes cache file — **no network call**

This means signing a 1 GB RPM costs exactly one ~4 KB network request to the signing server, regardless of package size.

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

This section summarises the security posture by threat layer. Detailed design rationale for each decision — including the KMS key storage options comparison, why SSH is disabled, and the break-glass procedure — is in §4 (Key Design Decisions). The implementation of each control is in the operations runbook (`operations-runbook.md`).

Security operates at three distinct layers. Controls at one layer do not substitute for controls at another.

---

### 8.1 Layer 1 — Network Perimeter

**Purpose:** Prevent unauthorized hosts from reaching the signing server at all.  
**Enforced by:** AWS VPC Security Groups and routing — not application code.

| Control | What it does |
|---------|-------------|
| `sg-signing-server` inbound | Allows TCP 443 from `sg-build-runner` and operator VPN CIDR only. All other sources silently dropped. |
| No public IP / internet gateway | Server has no route to the internet — unreachable from outside the VPC |
| Outbound restricted to VPC endpoints | Server can only reach Secrets Manager, KMS, and CloudWatch via PrivateLink — no other outbound traffic permitted |

See §4.1 for the full rationale on why VPC Security Groups are used instead of per-request SigV4 authentication.

---

### 8.2 Layer 2 — Key at Rest and in Distribution

**Purpose:** Protect the GPG private key while stored in AWS and during server startup fetch.  
**Enforced by:** KMS CMK + Secrets Manager resource policy.

| Control | What it does |
|---------|-------------|
| Secrets Manager + KMS CMK | GPG private key stored as KMS-encrypted ciphertext — plaintext never persists on any disk |
| IAM resource policy on secrets | Only `role-signing-server` can call `GetSecretValue` — no other AWS principal |
| tmpfs keyring | After startup import, key exists in RAM only — EBS snapshots cannot capture it |
| CloudTrail on KMS | Every `GetSecretValue` triggers a `kms:Decrypt` — logged with caller identity and timestamp |

**Important limit:** KMS and Secrets Manager protect the key *before* it reaches the server. Once the key is in tmpfs, these controls provide no further protection. OS-level access bypasses them entirely — see Layer 3.

See §4.3 for the full options analysis (KMS Asymmetric vs CloudHSM vs Secrets Manager + CMK) and the rationale for the chosen approach.

---

### 8.3 Layer 3 — Server Instance Hardening

**Purpose:** Make OS-level access to the signing server impossible under normal operations.  
**Why this matters:** Once the GPG key is in the tmpfs keyring, anyone with a shell on the instance can read it — regardless of KMS, IAM, or any other AWS control. This layer is therefore the most critical.

| Control | What it does | Enforced by |
|---------|-------------|-------------|
| No SSH inbound | Port 22 not open to anyone — not even admins or bastion hosts | Security Group |
| SSM Session Manager denied | `ssm:StartSession` and `ssm:SendCommand` explicitly denied in `role-signing-server` IAM policy | IAM deny policy |
| IMDSv2 enforced | `--http-tokens required` prevents SSRF attacks stealing instance IAM credentials | EC2 instance config |
| Immutable infrastructure | Server is never patched in place — updates replace the instance from a new AMI | Operational policy |
| Single-purpose server | Only `signing-server.py` runs — no other services, no package manager in prod | AMI hardening |
| Input validation | `key_id` validated against strict regex before passing to GPG subprocess — prevents command injection | Application code |

**Fundamental limit:** If an attacker achieves OS-level access to a running signing server, the GPG private key in tmpfs is readable. This is true of every key management approach — CloudHSM, hardware tokens, KMS asymmetric — because any running signing process must have access to key material. The correct response is: disable the KMS CMK immediately, which stops all future key fetches, then rotate the GPG key pair.

#### Operations without SSH — how it works in practice

Removing SSH does not mean the server is unmanageable. All routine operations are done via AWS APIs, not shell access:

| Operation | How (no SSH needed) |
|-----------|-------------------|
| **GPG key rotation** | `aws secretsmanager put-secret-value` from provisioner workstation → server picks up on next restart or scheduled sync |
| **Code update** | Build new AMI → launch new EC2 with same IAM role → verify `/health` → terminate old instance |
| **OS patching** | Same as code update — replace instance from freshly patched AMI |
| **View logs** | `aws logs filter-log-events` from workstation, or `journalctl` via CloudWatch |
| **Check server health** | `curl -k https://<server-ip>/health` from build runner or operator via VPN |
| **Emergency access** | Break-glass procedure: temporarily enable SSM via IAM policy change (audited in CloudTrail), conduct investigation, revert immediately |

For the full break-glass procedure and all operational commands, see **`operations-runbook.md`**.

---

### 8.4 Application-Level Controls

**Purpose:** Protect against malformed requests, resource exhaustion, and audit gaps.

| Threat | Control |
|--------|---------|
| `key_id` command injection into GPG subprocess | Strict regex: `[a-zA-Z0-9@.\-_ <>]+`, max 256 chars — rejected on mismatch |
| Runaway job exhausting signing capacity | Thread semaphore (max 10 concurrent) + per-source-IP rate limiting (sliding window) |
| Slow-read / slowloris attack | Socket read timeout: 10 seconds |
| Oversized request payload | Body size limit: 10 KB — rejected with HTTP 413 |
| Replay of a captured signature | GPG signatures bind to specific data — replayed signature for different data fails `gpg --verify` |
| Audit gaps | Every request (success and failure) written to stdout → CloudWatch Logs: source IP, tier, artifact, latency, status |

---

### 8.5 Operational References

The security controls described in this section are implemented and operated according to:

| Topic | Reference |
|-------|-----------|
| Full provisioning sequence (IAM → KMS → SM → EC2) | `operations-runbook.md` §1 |
| Day-to-day operations (status, logs, health check) | `operations-runbook.md` §2 |
| GPG key rotation procedure | `operations-runbook.md` §3 |
| Emergency key revocation | `operations-runbook.md` §4.1 |
| Break-glass shell access procedure | `operations-runbook.md` §4.2 |
| Server instance replacement | `operations-runbook.md` §4.3 |
| Troubleshooting common errors | `operations-runbook.md` §5 |
