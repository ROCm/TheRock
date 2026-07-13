# Signing Server — Use Case Flows

**Component:** Remote GPG Signing Service  
**Related docs:** `signing-server-design.md`, `signing-server-requirements.md`

This document describes the end-to-end flow for each supported use case from the actor's perspective. For the corresponding code execution path, see `code-flows.md`.

---

## Actors

| Actor | Description |
|-------|-------------|
| **TheRock CI build** | GitHub Actions workflow running on a self-hosted EC2 build runner |
| **Authorized operator** | AMD engineer performing manual one-off signing via VPN |
| **Signing server** | EC2 instance running `signing-server.py` in a private subnet |
| **AWS** | Secrets Manager, KMS, CloudWatch — managed services |

---

## UC-1: Server Startup — Load GPG Keys

**Actor:** Signing server (automatic on every start/restart)  
**Trigger:** `systemctl start signing-server` or reboot  
**Preconditions:** GPG private key stored in Secrets Manager; KMS CMK accessible via VPC endpoint

```
systemd
  │  starts signing-server.py
  ▼
Signing Server
  │  1. Reads --secrets-manager-secret arg (e.g. signing/gpg/therock-release)
  │
  │  2. Calls AWS Secrets Manager → GetSecretValue
  │     (Secrets Manager internally calls KMS → Decrypt to get the data key,
  │      decrypts the stored GPG key, returns plaintext in API response)
  │
  │  3. Pipes plaintext GPG key to: gpg --batch --import
  │     Key is loaded into /var/gpg-keyring (tmpfs — RAM only)
  │
  │  4. Unsets the plaintext key variable from memory
  │
  │  5. Runs: gpg --list-secret-keys
  │     Verifies at least one key is present → sets _keyring_ready = True
  │
  │  6. Begins listening on HTTPS :443
  │
  ▼
GET /health → 200 OK  {"status": "ok"}
```

**Outcome:** Server is ready to accept signing requests. GPG key in RAM only.  
**Failure:** If Secrets Manager or KMS unreachable → server exits → systemd restarts.

---

## UC-2: RPM Package Signing (TheRock CI Build)

**Actor:** TheRock GitHub Actions workflow on self-hosted EC2 build runner  
**Trigger:** `release_type` input is set (dev, nightly, or prerelease)  
**Preconditions:** Server is healthy; build runner in `sg-build-runner` security group

```
GitHub Actions Workflow
  │  1. Installs gpgshim to ~/.local/bin/gpgshim
  │  2. Sets environment variables:
  │       GPG_SIGNING_SERVER=https://<server-ip>/sign
  │       GPG_KEY_ID=therock-release@amd.com
  │       GPG_VERIFY_SSL=false  (Phase 1 self-signed cert)
  │
  │  3. Finds all .rpm files in package output directory
  │
  │  For each RPM:
  │  4. Calls: rpmsign --addsign \
  │              --define "%__gpg ~/.local/bin/gpgshim" \
  │              package.rpm
  │
  ▼
rpmsign
  │  Makes two gpg subprocess calls per package:
  │
  │  Call 1: gpgshim --detach-sign (RPM header, ~4 KB piped via stdin)
  │
  ▼
gpgshim (Call 1)
  │  Reads ~4 KB header from stdin
  │  Checks /tmp/gpgshim-cache-<ppid>.sig → MISS
  │  Sends POST /sign to signing server
  │  Receives signature → writes to output file
  │  Saves signature to /tmp/gpgshim-cache-<ppid>.sig
  │
  ▼
Signing Server
  │  Security Group validates: source IP in sg-build-runner → allowed
  │  Validates request: key_id, data not empty, size < 10 KB
  │  Runs: gpg --batch --detach-sign --local-user therock-release@amd.com
  │  Returns: {"signature": "<base64>", "key_id": "...", "digest_algo": "SHA256"}
  │  Writes audit log entry to stdout → CloudWatch Logs
  │
  ▼
gpgshim (Call 2)
  │  rpmsign calls gpgshim again (full RPM body, up to 1 GB+ piped via stdin)
  │  gpgshim reads and discards stdin (must consume it)
  │  Checks /tmp/gpgshim-cache-<ppid>.sig → HIT
  │  Returns cached signature to rpmsign
  │  Deletes cache file
  │  (No network call — zero bytes sent to server)
  │
  ▼
rpmsign
  │  Embeds both signatures into the RPM header
  │
  ▼
GitHub Actions Workflow
  │  5. Verifies signature: rpm --checksig package.rpm
  │  6. Repeats for all RPMs
  │  7. Proceeds to repository metadata generation and upload
```

**Outcome:** All RPMs signed; each package costs one ~4 KB network request regardless of size.  
**Failure:** If server returns 503 → gpgshim retries with exponential backoff (up to 5 retries).

---

## UC-3: Repository Metadata Signing (TheRock CI Build)

**Actor:** TheRock GitHub Actions workflow — `upload_package_repo.py`  
**Trigger:** After RPM/DEB packages uploaded; before S3 metadata publish  
**Preconditions:** Repository metadata (repomd.xml or Release file) already generated

### UC-3a: RPM repomd.xml signing

```
upload_package_repo.py
  │  Finds all repodata/repomd.xml files (one per architecture)
  │
  │  For each repomd.xml:
  │  1. Reads file content (~2 KB)
  │  2. Base64-encodes content
  │  3. Sends POST /sign:
  │       { data: base64, key_id: ..., armor: true, clearsign: false }
  │
  ▼
Signing Server
  │  Signs with gpg --armor --detach-sign
  │  Returns ASCII-armored detached signature
  │
  ▼
upload_package_repo.py
  │  4. Writes signature to repomd.xml.asc alongside repomd.xml
  │  5. Uploads both repomd.xml and repomd.xml.asc to S3
```

### UC-3b: DEB Release file signing

```
upload_package_repo.py
  │  Reads the generated Release file
  │
  │  Request 1 — InRelease (clearsigned):
  │  1. Sends POST /sign:
  │       { data: base64(Release), armor: true, clearsign: true }
  │
  ▼
Signing Server
  │  Signs with gpg --clearsign
  │  Returns complete PGP signed message block
  │  (data + signature combined)
  │
  ▼
upload_package_repo.py
  │  2. Writes output to InRelease
  │
  │  Request 2 — Release.gpg (detached):
  │  3. Sends POST /sign:
  │       { data: base64(Release), armor: true, clearsign: false }
  │
  ▼
Signing Server
  │  Signs with gpg --armor --detach-sign
  │  Returns detached ASCII signature
  │
  ▼
upload_package_repo.py
  │  4. Writes output to Release.gpg
  │  5. Uploads Release, InRelease, Release.gpg to S3
```

**Outcome:** Repository metadata signed; `apt` and `rpm` clients can verify package authenticity.

---

## UC-4: One-Off Manual Signing (Operator)

**Actor:** Authorized operator on corporate workstation via VPN  
**Trigger:** Need to re-sign a specific artifact outside the CI pipeline  
**Preconditions:** Operator has VPN access; workstation IP in operator VPN CIDR range

```
Operator Workstation (via VPN)
  │
  │  1. Confirm VPN is connected
  │
  │  2. Run sign-file CLI:
  │       python sign-file \
  │         --server https://10.0.2.45 \
  │         --key-id therock-release@amd.com \
  │         --file repomd.xml \
  │         --armor \
  │         --no-verify-ssl
  │
  ▼
sign-file script
  │  Reads repomd.xml from disk
  │  Base64-encodes content
  │  Sends POST /sign to server
  │
  ▼
Signing Server
  │  Security Group validates: source IP in operator VPN CIDR → allowed
  │  Signs with gpg --armor --detach-sign
  │  Returns signature
  │  Writes audit log entry (source_ip, key_id, latency_ms)
  │
  ▼
sign-file script
  │  Writes repomd.xml.asc
  │  Prints: "Signature written to: repomd.xml.asc (288 bytes)"
  │
  ▼
Operator
  │  3. Verifies: gpg --verify repomd.xml.asc repomd.xml
  │  4. Uploads signed artifact to S3 manually
```

**Outcome:** Single file signed without triggering a full CI pipeline run.  
**Failure modes:**
- Connection refused → VPN not connected or server IP incorrect
- 403 Forbidden → operator IP not in allowed CIDR range (check Security Group)

---

## UC-5: Health Check (Load Balancer / Monitoring)

**Actor:** AWS ALB target group health check (Phase 2) or monitoring script (Phase 1)  
**Trigger:** Periodic (every 30 seconds in Phase 2)

```
ALB / Monitoring
  │  GET /health
  │
  ▼
Signing Server
  │  If _keyring_ready = True  → 200 {"status": "ok"}
  │  If _keyring_ready = False → 503 {"status": "unavailable",
  │                                   "reason": "keyring not loaded"}
```

**Phase 2 behaviour:** ALB stops routing traffic to an instance that returns 503 after 3 consecutive failures, routing all requests to the secondary server.

---

## UC-6: Signing Request Rejected (Error Cases)

### UC-6a: Oversized request (DoS protection)

```
Any caller
  │  POST /sign with Content-Length > 10 KB
  │
  ▼
Signing Server
  │  Reads Content-Length header only — does not read body
  │  Returns: 413 {"error": "Request too large: N bytes (max 10240 bytes)"}
  │  Writes audit log entry
```

### UC-6b: Server at capacity

```
11th concurrent request arrives (thread semaphore at limit)
  │
  ▼
Signing Server
  │  Returns: 503 {"error": "Server busy, try again later"}
  │
  ▼
gpgshim
  │  Retries with exponential backoff:
  │  Wait 100ms → retry → wait 200ms → retry → ... up to 5 retries
  │  If still failing after 5 retries → fails with error
```

### UC-6c: Rate limit exceeded

```
Client exceeds max_requests_per_hour for their role
  │
  ▼
Signing Server
  │  Returns: 429 {"error": "Rate limit exceeded"}
  │  Writes audit log entry: action=RATE_LIMITED
  │
  ▼
gpgshim
  │  Does NOT retry (429 is a permanent error in current session)
  │  Exits with error code
```

---

## UC-7: Emergency Key Revocation

**Actor:** AWS admin  
**Trigger:** GPG private key compromise detected

```
AWS Admin
  │  1. Disable KMS CMK immediately:
  │       aws kms disable-key --key-id alias/amd-signing-gpg-key
  │
  │     Effect: All future secretsmanager:GetSecretValue calls fail
  │     (Secrets Manager cannot decrypt the data key without the CMK)
  │
  │  2. Signing server continues operating with key already in RAM
  │     (does NOT call Secrets Manager per request)
  │
  │  3. Restart signing server to force key reload:
  │       aws ec2 reboot-instances --instance-ids <id>
  │
  │     Server startup fails to fetch key from SM → exits → systemd restarts
  │     → exits again → signing server is effectively down
  │
  │  4. Rotate GPG key pair (generate new, store in SM)
  │  5. Re-enable or replace CMK
  │  6. Restart server → loads new key
  │  7. Distribute new public key to package consumers
```

**Outcome:** Signing stops within one server restart cycle (~30 seconds); new key provisioned and signing resumes.

---

## UC-8: GPG Key Rotation (Planned)

**Actor:** Key provisioner  
**Trigger:** Key expiry, scheduled rotation, or policy requirement

```
Key Provisioner (from isolated workstation)
  │
  │  1. Generate new GPG key pair:
  │       gpg --batch --gen-key
  │
  │  2. Export private key:
  │       gpg --export-secret-keys --armor new-key@amd.com > /tmp/new.asc
  │
  │  3. Update Secrets Manager secret:
  │       aws secretsmanager put-secret-value \
  │         --secret-id signing/gpg/therock-release \
  │         --secret-string file:///tmp/new.asc
  │
  │  4. Delete local copy:
  │       shred -vzu /tmp/new.asc
  │
  │  5. Export and distribute new PUBLIC key to:
  │       - ROCm package repository keyserver
  │       - RPM/APT client configuration
  │
  │  Phase 1: Restart server to pick up new key:
  │       aws ec2 reboot-instances --instance-ids <id>
  │
  │  Phase 2: Server picks up automatically on next scheduled sync
  │       (every 6 hours, no restart needed)
```

**Outcome:** New key active; old packages remain verifiable with old public key; new packages signed with new key.
