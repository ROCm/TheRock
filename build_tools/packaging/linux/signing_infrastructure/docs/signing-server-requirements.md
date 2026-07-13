# Signing Server — Implementation Plan and Requirements

**Project:** AMD ROCm Build System  
**Component:** Remote GPG Signing Service  
**Status:** Draft — v0.4  
**Branch:** `users/nunnikri/signing-serverver-test`

> For architecture rationale, component design, and security analysis see:  
> `signing-server-design.md`

---

## 1. Phased Approach

Implementation is split into two phases to deliver a working signing pipeline quickly (Phase 1) and then harden it with redundancy and application-layer auth (Phase 2).

| | Phase 1 | Phase 2 |
|--|---------|---------|
| **Goal** | Working end-to-end signing in CI | Production-hardened with HA and audit traceability |
| **Connectivity** | Direct HTTPS to primary server private IP | Internal ALB routing to primary + secondary |
| **Auth** | VPC Security Groups only | Security Groups + app-layer pre-shared token |
| **Redundancy** | Single signing server | Primary + secondary with scheduled key sync and ALB failover |
| **gpgshim** | Current implementation (stdlib only) | Updated with fallback URL and Phase 2 auth header |
| **Operator access** | `sign-file` CLI (basic) | `sign-file` with app-layer token support |

---

## 2. Phase 1 Requirements

### 2.1 Infrastructure — AWS and VPC

#### Compute

| ID | Requirement |
|----|-------------|
| P1-INF-1 | One EC2 instance shall be provisioned as the signing server in a dedicated private subnet with no internet gateway and no NAT gateway attached |
| P1-INF-2 | The signing server instance type shall support at least 2 vCPUs and 4 GB RAM (e.g., `t3.medium` or equivalent) |
| P1-INF-3 | The signing server root volume shall be encrypted EBS; no sensitive data is written to it (GPG keyring is on tmpfs) |
| P1-INF-4 | A `tmpfs` volume shall be mounted at a dedicated path (e.g., `/var/gpg-keyring`) for the in-memory GPG keyring — this mount shall not survive instance reboot; it shall be configured in `/etc/fstab` so it is automatically re-mounted on every boot |
| P1-INF-5 | The signing server operating system shall be Amazon Linux 2023 or Ubuntu 22.04 LTS |
| P1-INF-6 | IMDSv2 (Instance Metadata Service v2) shall be enforced on the signing server EC2 instance with `--http-tokens required` — this prevents SSRF attacks from stealing instance IAM credentials via the metadata endpoint |
| P1-INF-7 | The signing server EC2 instance shall run no services other than `signing-server.py` — no web servers, no package managers accessible at runtime, no cron jobs beyond the Phase 2 key sync |

#### Networking — VPC

| ID | Requirement |
|----|-------------|
| P1-NET-1 | A dedicated VPC (or an existing shared build VPC) shall contain all signing infrastructure components |
| P1-NET-2 | The signing server shall reside in a **private subnet** with no route to an internet gateway |
| P1-NET-3 | The build runners (self-hosted EC2 GitHub Actions runners) shall reside in a subnet routable to the signing server private subnet — either the same VPC or via VPC peering |
| P1-NET-4 | The signing server shall be reachable from build runners using its **private IP address** directly — no load balancer, no DNS alias required in Phase 1 |
| P1-NET-5 | Interface VPC endpoints (AWS PrivateLink) shall be provisioned for the following services so the signing server can reach them without NAT: |
| | &nbsp;&nbsp;- `com.amazonaws.<region>.secretsmanager` — GPG key ciphertext fetch |
| | &nbsp;&nbsp;- `com.amazonaws.<region>.kms` — KMS decrypt of GPG private key |
| | &nbsp;&nbsp;- `com.amazonaws.<region>.logs` — CloudWatch audit log |
| P1-NET-6 | VPC endpoint security groups shall allow HTTPS (443) inbound only from the signing server security group |

#### Security Groups

| ID | Requirement |
|----|-------------|
| P1-SG-1 | A security group `sg-signing-server` shall be attached to the signing server EC2 instance |
| P1-SG-2 | `sg-signing-server` inbound rules: |
| | &nbsp;&nbsp;- HTTPS (TCP 443) from `sg-build-runner` (TheRock EC2 build runner security group) |
| | &nbsp;&nbsp;- HTTPS (TCP 443) from the operator VPN CIDR range |
| | &nbsp;&nbsp;- **No SSH (TCP 22) inbound from any source** — including admins and bastion hosts |
| P1-SG-3 | `sg-signing-server` outbound rules: |
| | &nbsp;&nbsp;- HTTPS (TCP 443) to VPC endpoint security groups (Secrets Manager, KMS, CloudWatch Logs) |
| | &nbsp;&nbsp;- No other outbound traffic permitted |
| P1-SG-4 | `sg-build-runner` shall be the security group attached to all TheRock self-hosted EC2 runner instances — membership in this group is the sole network-layer access control for signing |
| P1-SG-5 | No inbound rule shall permit traffic from `0.0.0.0/0` (no public access) on any port |

#### IAM

Two IAM principals are required for the signing infrastructure. A third is noted for completeness but requires no signing-specific permissions.

**ARN 1 — Signing Server Role (`role-signing-server`)**
Attached as an EC2 instance profile to the signing server. Used at runtime on every server startup when the GPG key is fetched.

| ID | Requirement |
|----|-------------|
| P1-IAM-1 | `role-signing-server` shall have the following permissions and no others: |
| | &nbsp;&nbsp;- `secretsmanager:GetSecretValue` on `signing/gpg/*` secret ARNs |
| | &nbsp;&nbsp;- `kms:Decrypt` on the CMK — called internally by Secrets Manager during `GetSecretValue` |
| | &nbsp;&nbsp;- `logs:PutLogEvents`, `logs:CreateLogStream` on the signing audit log group ARN |
| P1-IAM-2 | `role-signing-server` shall explicitly NOT have `kms:GenerateDataKey`, `kms:Encrypt`, `secretsmanager:CreateSecret`, or `secretsmanager:PutSecretValue` — the signing server reads keys only, never provisions them |
| P1-IAM-2a | `role-signing-server` shall explicitly DENY `ssm:StartSession`, `ssm:SendCommand`, and `ssm:StartSSHSession` — AWS SSM Session Manager must not be usable as a backdoor shell into the signing server |

**ARN 2 — Key Provisioning Principal (`role-key-provisioner`)**
Used once per key during initial setup to store GPG private keys into Secrets Manager. Not attached to any EC2 instance — assumed by an authorized operator via AWS SSO or explicit role assumption.

| ID | Requirement |
|----|-------------|
| P1-IAM-3 | `role-key-provisioner` shall have the following permissions and no others: |
| | &nbsp;&nbsp;- `secretsmanager:CreateSecret`, `secretsmanager:PutSecretValue` on `signing/gpg/*` ARNs |
| | &nbsp;&nbsp;- `kms:GenerateDataKey` on the CMK — called internally by Secrets Manager when storing a new secret |
| P1-IAM-4 | `role-key-provisioner` shall NOT have `kms:Decrypt` or `secretsmanager:GetSecretValue` — it writes keys, it cannot read them back |
| P1-IAM-5 | `role-key-provisioner` shall not be attached to any EC2 instance profile — it must only be usable via explicit role assumption by an authorized operator |

**ARN 3 — Build Runner Role (no signing-specific permissions needed)**

| ID | Requirement |
|----|-------------|
| P1-IAM-6 | Build runner EC2 instances require no IAM permissions related to the signing server — access is controlled entirely by VPC Security Groups. The build runner role is listed here only to make explicit that no IAM configuration is needed on the client side |

#### GPG Key Storage Approach

> Three approaches were evaluated — KMS Asymmetric Keys, AWS CloudHSM + PKCS#11, and Secrets Manager + KMS CMK. See `signing-server-design.md` §4.3 for the full comparison and rationale. **Secrets Manager + KMS CMK was chosen** as the only approach that is simultaneously GPG-compatible, low-cost, and low-complexity. CloudHSM remains the documented upgrade path if FIPS 140-2 Level 3 is required.

#### AWS KMS — Customer Managed Key (CMK)

| ID | Requirement |
|----|-------------|
| P1-KMS-1 | A Customer Managed Key (CMK) shall be created in AWS KMS exclusively for GPG signing key protection — key alias: `alias/amd-signing-gpg-key` |
| P1-KMS-2 | The CMK key policy shall grant `kms:Decrypt` to `role-signing-server` only — Secrets Manager calls this internally when the server calls `GetSecretValue` |
| P1-KMS-3 | The CMK key policy shall grant `kms:GenerateDataKey` to `role-key-provisioner` only — Secrets Manager calls this internally when the provisioner stores a new secret |
| P1-KMS-5 | The CMK key policy shall grant `kms:*` to the AWS account root for administration (required by KMS to prevent policy lockout) |
| P1-KMS-6 | The CMK shall have key deletion protection enabled (minimum 30-day waiting period before deletion) |
| P1-KMS-7 | All KMS API calls shall be captured in AWS CloudTrail — every `GetSecretValue` call triggers a `kms:Decrypt` internally, providing a full audit record of key use |

#### AWS Secrets Manager

| ID | Requirement |
|----|-------------|
| P1-SEC-1 | GPG private keys shall be stored in AWS Secrets Manager, one secret per signing tier, with the CMK specified as the encryption key — Secrets Manager handles envelope encryption internally: |
| | &nbsp;&nbsp;- `signing/gpg/therock-dev` |
| | &nbsp;&nbsp;- `signing/gpg/therock-nightly` |
| | &nbsp;&nbsp;- `signing/gpg/therock-release` |
| P1-SEC-2 | Each secret shall be created with `--kms-key-id alias/amd-signing-gpg-key` — this ensures Secrets Manager uses the CMK for all encryption/decryption operations rather than the default AWS-managed key |
| P1-SEC-3 | Each Secrets Manager secret shall have a resource-based policy restricting `secretsmanager:GetSecretValue` to `role-signing-server` only |
| P1-SEC-4 | Secrets Manager secret rotation shall not be configured to auto-rotate — key rotation is a manual, coordinated process requiring public key re-distribution to all package consumers |
| P1-SEC-5 | If the GPG private key has a passphrase, the passphrase shall be stored as a separate Secrets Manager secret (`signing/gpg/therock-<tier>-passphrase`), also using the CMK |

#### Key Provisioning Process (Offline, One-Time per Key)

The following steps are performed offline by an authorized operator — not by the signing server itself. Secrets Manager handles KMS envelope encryption internally; no manual `kms encrypt` call is needed.

| Step | Action |
|------|--------|
| 1 | Generate GPG key pair on an isolated machine: `gpg --batch --gen-key` |
| 2 | Export the private key to a temp file: `gpg --export-secret-keys --armor <key-id> > /tmp/private.asc` |
| 3 | Store in Secrets Manager with CMK: `aws secretsmanager create-secret --name signing/gpg/therock-release --kms-key-id alias/amd-signing-gpg-key --secret-string file:///tmp/private.asc` |
| 4 | Securely delete the plaintext file: `shred -vzu /tmp/private.asc` |
| 5 | Verify what Secrets Manager holds starts with `-----BEGIN PGP` — if it starts with `AQI` the wrong step was taken (manual kms encrypt was called instead of letting Secrets Manager handle it) |
| 6 | Export and distribute the **public key** to the ROCm package repository keyserver and RPM/APT tooling |

### 2.2 Software — Signing Server

#### Runtime

| ID | Requirement |
|----|-------------|
| P1-SRV-1 | The signing server shall run `signing-server.py` as a systemd service, automatically restarted on failure |
| P1-SRV-2 | Python 3.9 or later shall be installed on the signing server |
| P1-SRV-3 | `boto3` shall be installed on the signing server (for Secrets Manager key fetch at startup) |
| P1-SRV-4 | `gpg` (GnuPG 2.x) shall be installed on the signing server — this is the only machine that requires GPG |
| P1-SRV-5 | The server shall listen on HTTPS port 443 with TLS 1.2 minimum using a self-signed certificate (internal only; no CA validation required from clients in Phase 1) |
| P1-SRV-6 | TLS certificate and private key shall be stored on the signing server's EBS volume (not in the tmpfs keyring) |

#### Key Loading

| ID | Requirement |
|----|-------------|
| P1-KEY-1 | At process startup, the signing server shall perform the following key loading sequence for each configured signing tier: |
| | &nbsp;&nbsp;1. Call `secretsmanager:GetSecretValue` → Secrets Manager internally calls `kms:Decrypt`, returns plaintext GPG private key (PEM-armored) in the API response |
| | &nbsp;&nbsp;2. Pipe plaintext key directly to `gpg --import` using the tmpfs-backed GNUPGHOME — key goes from API response to GPG without being written to any file |
| | &nbsp;&nbsp;3. Immediately unset and delete the plaintext key variable from memory |
| P1-KEY-2 | The tmpfs GNUPGHOME directory shall be created at server startup with mode `0700`, owned by the signing server process user, mounted on tmpfs so it does not survive a reboot |
| P1-KEY-3 | The plaintext GPG private key shall exist in process memory only for the duration of the `gpg --import` call — it shall not be written to any file, log, or variable that persists beyond that call |
| P1-KEY-4 | If either the Secrets Manager fetch or the KMS decrypt call fails at startup, the server shall exit with a non-zero code — systemd shall restart it |
| P1-KEY-5 | The signing server shall expose a `/health` endpoint returning `200 OK` only when the GPG keyring is loaded and at least one valid key is confirmed present via `gpg --list-secret-keys` |

#### Signing Operations

| ID | Requirement |
|----|-------------|
| P1-SRV-7 | The server shall accept `POST /sign` with JSON body: `data` (base64), `key_id`, `digest_algo`, `armor`, `clearsign` |
| P1-SRV-8 | The `key_id` field shall be validated against a regex allowlist (`[a-zA-Z0-9@.\-_ <>]+`, max 256 chars) before being passed to the GPG subprocess — to prevent command injection |
| P1-SRV-9 | The server shall invoke `gpg --detach-sign` (or `--clearsign`) using the isolated tmpfs GNUPGHOME via the `GNUPGHOME` environment variable |
| P1-SRV-10 | Concurrent signing operations shall be limited by a thread semaphore (default: 10 threads). Requests exceeding the limit receive `503 Server Busy` |
| P1-SRV-11 | Maximum request body size: 10 KB. Requests exceeding this are rejected with `413` |
| P1-SRV-12 | Socket read timeout: 10 seconds. Prevents slow-read attacks |

#### Authorization

| ID | Requirement |
|----|-------------|
| P1-AUTH-1 | In Phase 1, no application-layer token is required — network access via Security Groups is the sole auth control |
| P1-AUTH-2 | The server shall maintain an `authorization.json` config mapping IAM role ARNs or caller identifiers to permitted `key_id` values and rate limits (structure preserved for Phase 2 compatibility) |
| P1-AUTH-3 | A request for a `key_id` not in the server's loaded keyring shall be rejected with `403` |

#### Rate Limiting

| ID | Requirement |
|----|-------------|
| P1-RL-1 | The server shall implement a per-client sliding window rate limiter (in-memory, per process) using the existing `check_rate_limit()` implementation in `auth.py` |
| P1-RL-2 | In Phase 1, the client identifier for rate limiting shall be the caller's source IP address (since no app-layer token is present) |
| P1-RL-3 | Default rate limits shall be configured in `authorization.json`: |
| | &nbsp;&nbsp;- Build runner IP range: 10,000 requests/hour |
| | &nbsp;&nbsp;- Operator VPN range: 100 requests/hour |
| P1-RL-4 | Requests exceeding the rate limit shall receive `429 Too Many Requests`; `gpgshim` shall treat `429` as a non-retryable error |

#### Observability

| ID | Requirement |
|----|-------------|
| P1-OBS-1 | Every signing request (success and failure) shall produce a structured JSON log entry containing: timestamp, source IP, `key_id` requested, `digest_algo`, HTTP status, signing latency (ms) |
| P1-OBS-2 | Log entries shall be written to stdout (captured by systemd journal) and forwarded to CloudWatch Logs via the CloudWatch agent |
| P1-OBS-3 | A CloudWatch log group `/amd/signing-server/audit` shall be created with a 90-day retention policy |
| P1-OBS-4 | A CloudWatch alarm shall trigger if the error rate (4xx + 5xx responses) exceeds 10% of requests in any 5-minute window |

### 2.3 Software — gpgshim (TheRock Build Runner)

| ID | Requirement |
|----|-------------|
| P1-SHIM-1 | `gpgshim` shall be a self-contained Python script with no external dependencies beyond the Python standard library — no `botocore`, no `boto3`, no `requests` |
| P1-SHIM-2 | `gpgshim` shall send `POST /sign` requests to the signing server URL specified by `GPG_SIGNING_SERVER` environment variable using `urllib` |
| P1-SHIM-3 | In Phase 1, `gpgshim` shall not include any `Authorization` header — Security Groups provide access control |
| P1-SHIM-4 | `gpgshim` shall implement ppid-based signature caching: write `/tmp/gpgshim-cache-<ppid>.sig` after the first signing call; read and delete it on the second call within the same `rpmsign` process |
| P1-SHIM-5 | `gpgshim` shall retry on `503 Server Busy` using exponential backoff with jitter (default: 5 retries, starting at 100 ms, capped at 10 s) |
| P1-SHIM-6 | `gpgshim` shall not retry on `401`, `403`, `413`, or `429` — these are permanent failures |
| P1-SHIM-7 | `gpgshim` shall be configurable via environment variables: `GPG_SIGNING_SERVER` (full URL), `GPG_KEY_ID` (default key), `GPG_TIMEOUT` (seconds, default 30), `GPG_MAX_RETRIES` (default 5) |
| P1-SHIM-8 | `gpgshim` shall support HTTPS with optional CA certificate verification disabled for self-signed certs: `GPG_VERIFY_SSL=false` |

### 2.4 Software — sign-file CLI Tool (Operator)

| ID | Requirement |
|----|-------------|
| P1-CLI-1 | A `sign-file` Python script shall be provided in `signing_infrastructure/tools/sign-file` |
| P1-CLI-2 | `sign-file` shall accept arguments: `--server` (URL), `--key-id`, `--file` (input file path), `--armor` (flag), `--clearsign` (flag), `--output` (optional output path) |
| P1-CLI-3 | `sign-file` shall base64-encode the input file, POST to `/sign`, and write the returned signature to `<file>.asc` (armored) or `<file>.sig` (binary) |
| P1-CLI-4 | `sign-file` shall depend only on the Python standard library — no external packages |
| P1-CLI-5 | `sign-file` shall print a human-readable error message on `403`, `429`, and `503` responses |

### 2.5 CI/CD Integration (TheRock GitHub Actions)

| ID | Requirement |
|----|-------------|
| P1-CI-1 | Signing shall be activated only when the workflow `release_type` input is non-empty (`dev`, `nightly`, `prerelease`) |
| P1-CI-2 | If `GPG_SIGNING_SERVER` secret is not set, the build shall complete without signing — no error |
| P1-CI-3 | The workflow shall install `gpgshim` to `~/.local/bin/gpgshim` and invoke `rpmsign` with `--define "_gpg_path ~/.local/bin/gpgshim"` |
| P1-CI-4 | RPM packages shall be signed before repository metadata (`repomd.xml`) is generated, so signed package checksums are captured in the metadata |
| P1-CI-5 | `upload_package_repo.py` shall call `POST /sign` directly for repo metadata (`repomd.xml`, `Release`) — no `gpgshim` involved for this step |
| P1-CI-6 | `GPG_SIGNING_SERVER` (the server private IP and port) shall be stored as a GitHub Actions repository secret — not hardcoded in workflow YAML |

---

## 3. Phase 2 Requirements

Phase 2 adds redundancy (primary + secondary with ALB failover), application-layer authentication for audit traceability, and server-side RPM signing for callers who cannot use `gpgshim`. Phase 1 infrastructure remains in place; Phase 2 extends it.

### 3.1 Infrastructure — High Availability

#### Secondary Server

| ID | Requirement |
|----|-------------|
| P2-INF-1 | A second EC2 signing server instance shall be provisioned in a **different Availability Zone** from the primary, in the same private subnet tier |
| P2-INF-2 | The secondary shall be identical in configuration to the primary (same AMI, same systemd service, same IAM role, same tmpfs keyring setup) |
| P2-INF-3 | Both primary and secondary shall fetch GPG key ciphertexts from Secrets Manager and decrypt via KMS independently — no server-to-server key transfer |

#### Scheduled Key Sync

| ID | Requirement |
|----|-------------|
| P2-SYNC-1 | Both signing servers shall periodically re-fetch GPG keys from Secrets Manager and reload the in-memory keyring without restarting the process |
| P2-SYNC-2 | The sync interval shall be configurable (default: every 6 hours) via an environment variable or config file |
| P2-SYNC-3 | If a sync fails (Secrets Manager unreachable), the server shall continue operating with the currently loaded key and log a warning — it shall not stop serving requests |
| P2-SYNC-4 | Key reload shall be atomic: the new key shall be imported into a temporary GNUPGHOME, verified, then swapped in — the server shall never be in a state with no valid key |

#### Internal Application Load Balancer

| ID | Requirement |
|----|-------------|
| P2-ALB-1 | An internal (non-internet-facing) Application Load Balancer shall be provisioned in the same VPC |
| P2-ALB-2 | The ALB listener shall be HTTPS (port 443) with TLS 1.2 minimum, using an ACM certificate for the internal DNS name |
| P2-ALB-3 | The ALB target group shall include both primary and secondary signing server instances |
| P2-ALB-4 | The ALB shall use `GET /health` as the health check endpoint, with: interval 30 s, healthy threshold 2, unhealthy threshold 3 |
| P2-ALB-5 | If the primary fails health checks, the ALB shall route all traffic to the secondary automatically — no client changes required |
| P2-ALB-6 | The ALB DNS name (internal Route 53 alias) shall replace the primary server's private IP as the value of `GPG_SIGNING_SERVER` in the workflow |

#### Security Groups (additions to Phase 1)

| ID | Requirement |
|----|-------------|
| P2-SG-1 | An `sg-alb-signing` security group shall be created for the ALB |
| P2-SG-2 | `sg-alb-signing` inbound: HTTPS (443) from `sg-build-runner` and operator VPN CIDR |
| P2-SG-3 | `sg-alb-signing` outbound: HTTPS (443) to `sg-signing-server` |
| P2-SG-4 | `sg-signing-server` inbound shall be updated: accept HTTPS (443) from `sg-alb-signing` only (build runners and operators now go through the ALB, not directly to the instance) |

### 3.2 Application-Layer Authentication

| ID | Requirement |
|----|-------------|
| P2-AUTH-1 | The signing server shall require an `Authorization: Bearer <token>` header on all `POST /sign` requests |
| P2-AUTH-2 | Pre-shared tokens shall be defined per caller type, stored in Secrets Manager: |
| | &nbsp;&nbsp;- `signing/tokens/therock-dev` → used by dev/PR build runners |
| | &nbsp;&nbsp;- `signing/tokens/therock-nightly` → used by nightly build runners |
| | &nbsp;&nbsp;- `signing/tokens/therock-release` → used by release build runners |
| | &nbsp;&nbsp;- `signing/tokens/operator` → used by the `sign-file` CLI tool |
| P2-AUTH-3 | The signing server shall load all caller tokens from Secrets Manager at startup and reload them on the same schedule as the GPG key sync |
| P2-AUTH-4 | Token validation shall use constant-time comparison (`hmac.compare_digest`) to prevent timing attacks |
| P2-AUTH-5 | A request with a missing or invalid token shall receive `401 Unauthorized` and be logged with the source IP |
| P2-AUTH-6 | The `authorization.json` config shall map each token identifier (caller name) to its permitted `key_id` values and rate limit, replacing the source-IP-based mapping used in Phase 1 |
| P2-AUTH-7 | Per-caller rate limiting shall use the token identifier as the client key (replacing source IP from Phase 1), giving accurate per-caller limits even when multiple build runners share the same NAT IP |

### 3.3 Software — gpgshim Updates

| ID | Requirement |
|----|-------------|
| P2-SHIM-1 | `gpgshim` shall read a token from the `GPG_SERVER_TOKEN` environment variable and include it as `Authorization: Bearer <token>` on all requests |
| P2-SHIM-2 | The token shall be fetched by the GitHub Actions workflow from Secrets Manager (or passed via a repository secret) before invoking `rpmsign` — `gpgshim` itself does not fetch from Secrets Manager |
| P2-SHIM-3 | `gpgshim` shall treat `401 Unauthorized` as a non-retryable failure and exit with a non-zero code |

### 3.4 Software — sign-file CLI Updates

| ID | Requirement |
|----|-------------|
| P2-CLI-1 | `sign-file` shall accept a `--token` argument (or read from `SIGNING_TOKEN` environment variable) and include it as an `Authorization` header |
| P2-CLI-2 | `sign-file` shall print a clear error on `401` indicating the token is missing or invalid |

### 3.5 Server-Side RPM Signing (POST /sign-rpm)

**Context:** Phase 1 RPM package signing requires `gpgshim` to be installed on the calling machine, because `rpmsign` must run client-side to embed the signature into the RPM binary format. For ad-hoc or operator use cases where installing `gpgshim` is not practical, the signing server shall provide a new endpoint that accepts a full RPM file, runs `rpmsign` server-side, and returns the complete signed RPM.

**Trade-offs vs Phase 1 approach:**

| | Phase 1 (gpgshim) | Phase 2 (POST /sign-rpm) |
|--|------------------|--------------------------|
| **Network transfer** | ~4 KB (header only) | Full RPM (can be 1 GB+) |
| **Client requirement** | `gpgshim` + `rpmsign` installed | HTTP client only (curl, Python) |
| **Use case** | Automated CI builds | Ad-hoc operator, external callers |
| **Server requirement** | `gpg` only | `gpg` + `rpmsign` installed on server |

| ID | Requirement |
|----|-------------|
| P2-RPM-1 | The signing server shall expose a new endpoint `POST /sign-rpm` that accepts a complete RPM file as a binary upload and returns the signed RPM as a binary download |
| P2-RPM-2 | The request shall use `multipart/form-data` or `application/octet-stream` content type; `key_id` and `digest_algo` passed as query parameters or a JSON envelope |
| P2-RPM-3 | The server shall invoke `rpmsign --addsign` on the uploaded RPM using a local `gpgshim` or direct GPG configuration against the tmpfs keyring |
| P2-RPM-4 | The maximum request size limit for `POST /sign-rpm` shall be configurable separately from `POST /sign` — default 512 MB to accommodate large RPM packages |
| P2-RPM-5 | The uploaded RPM shall be written to a temporary file on tmpfs (not EBS), signed in place, streamed back to the caller, and immediately deleted — the RPM shall not persist on the server |
| P2-RPM-6 | `POST /sign-rpm` shall be subject to the same auth (Phase 2 app token) and rate limiting as `POST /sign` |
| P2-RPM-7 | `rpmsign` shall be installed on the signing server as an additional dependency for Phase 2 |
| P2-RPM-8 | The `sign-file` CLI shall be extended with a `--rpm` flag that uses `POST /sign-rpm` instead of `POST /sign` when signing RPM files directly |

**Example usage (Phase 2):**

```bash
# Operator signing an RPM without gpgshim installed
python3 sign-file \
  --server https://signing.internal.amd.com \
  --key-id therock-release@amd.com \
  --file mypackage.rpm \
  --rpm \
  --token "$SIGNING_TOKEN"
# Output: mypackage.rpm (overwritten in place with embedded signature)

# Or raw curl
curl -k -X POST https://signing.internal.amd.com/sign-rpm \
  -H "Authorization: Bearer $SIGNING_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  -F "key_id=therock-release@amd.com" \
  --data-binary @mypackage.rpm \
  -o mypackage-signed.rpm
```

---

## 4. Dependency Summary

### Phase 1 — What Must Exist Before Implementation Starts

| Dependency | Type | Notes |
|------------|------|-------|
| AWS account with VPC | Infrastructure | Signing server and build runners must be in routable subnets |
| Self-hosted EC2 GitHub Actions runners | Infrastructure | Must have a defined security group (`sg-build-runner`) |
| KMS CMK (`alias/amd-signing-gpg-key`) | AWS / Cryptographic | Must be created and key policy set before key provisioning |
| GPG key pairs (dev, nightly, release) | Cryptographic | Generated offline; stored directly in Secrets Manager with `--kms-key-id` — Secrets Manager handles envelope encryption |
| GPG private keys in Secrets Manager | Configuration | One secret per tier (`signing/gpg/therock-*`); Secrets Manager stores encrypted, decrypts transparently on `GetSecretValue` |
| GPG public keys published | Configuration | Must be distributed to end users and configured in RPM/APT tooling before signed packages are consumed |
| `gpg` 2.x installed on signing server | Software | Package signing requires local GPG on the server only |
| Python 3.9+ on signing server | Software | For `signing-server.py` |
| `boto3` on signing server | Software | For Secrets Manager fetch + KMS decrypt at startup |
| TLS certificate + key for signing server | Cryptographic | Self-signed acceptable for Phase 1; stored on EBS (not tmpfs) |
| CloudWatch log group `/amd/signing-server/audit` | AWS | Must be created before server starts logging |
| CloudWatch agent on signing server | Software | For forwarding systemd journal to CloudWatch |
| VPC endpoint for `com.amazonaws.<region>.kms` | AWS | Required so signing server can call KMS decrypt without internet egress |

### Phase 2 — Additional Prerequisites

| Dependency | Type | Notes |
|------------|------|-------|
| Second EC2 instance (secondary server) | Infrastructure | Different AZ from primary |
| ACM certificate for internal ALB DNS name | Cryptographic | Required for ALB HTTPS listener |
| Internal Route 53 hosted zone | AWS | For ALB DNS alias record |
| Pre-shared tokens generated and stored in Secrets Manager | Configuration | One token per caller type |

---

## 5. Open Questions

| # | Question | Impact |
|---|----------|--------|
| Q-1 | Should the signing server be a long-running EC2 instance or ECS Fargate? Fargate simplifies deployment but adds cold-start latency (key fetch from Secrets Manager on each task start) | Medium — affects operational model |
| Q-2 | Is there a FIPS 140-2 requirement for GPG key storage? Secrets Manager is not FIPS 140-2 certified; AWS CloudHSM would be required | Medium — would significantly change key management |
| Q-3 | What is the expected peak signing volume per release across all callers? Informs thread pool size and whether one instance is sufficient | Low — informs capacity planning |
| Q-4 | Who owns GPG key generation and rotation? This is an operational process outside the signing server itself but must be defined before Phase 1 goes live | High — blocking for Phase 1 go-live |
| Q-5 | How will the GPG public keys be distributed to end users? (e.g., keyserver, static URL, bundled in package manager config) | High — required for users to verify signed packages |
