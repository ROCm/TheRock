# CLAUDE.md

This file provides guidance to Claude Code when working with code in this directory.

## Project Overview

Remote GPG signing service for TheRock's RPM/DEB packages and repository metadata. `gpgshim` (one directory up, `build_tools/packaging/linux/gpgshim`) is a zero-dependency Python 3.6+ shim that intercepts `gpg` calls from `rpmsign` and forwards them to the signing server in this directory (`server/signing-server.py`).

**Authoritative docs — read these before making design changes, not just this file:**

- `docs/signing-server-design.md` — architecture, security model, rationale
- `docs/signing-server-requirements.md` — phased requirements (`P1-*`, `P1b-*`, `P2-*`)
- `docs/operations-runbook.md` — provisioning, key rotation, incident response

**Design goals (gpgshim only):**

- Zero dependencies — standard library only, no `boto3`/`requests`/`PyJWT`
- Python 3.6 compatibility (older RHEL/CentOS build runners)
- Minimal data transfer to the remote server
- Fast startup — no virtualenv needed

## Key Files

- `../gpgshim` — client shim, drop-in `gpg` replacement (lives one directory above this folder)
- `server/signing-server.py` — the signing server (threaded `http.server`)
- `server/auth.py` — authentication/authorization module, all phases
- `tools/sign-file` — operator CLI for manual signing over VPN
- `tools/setup-server.sh` — server provisioning script
- `tools/generate-token.py` — generates Phase 2 legacy JWT tokens
- `config/authorization.json` — `key_aliases`, `artifact_profiles`, `roles`, `clients` (Phase 1b)

## Architecture

### Data Flow (gpgshim → signing server)

1. `rpmsign` calls `gpgshim` instead of `gpg`, passing data on stdin (twice: RPM header ~4-100KB, then full RPM up to 1GB+)
1. `gpgshim` reads the data and sends it — base64-encoded, in full, not just a digest — to the server via `POST /sign`
1. **ppid-based caching is what limits network transfer, not digest-only transmission:** the shim keys a signature cache file (`/tmp/gpgshim-cache-<ppid>.sig`) by `rpmsign`'s parent process ID. The first call (small header) is sent to the server and its signature cached; the second call (full RPM) is read from stdin (rpmsign requires this) but never sent over the network — the cached signature from the first call is reused instead
1. The server signs using GPG (via the tmpfs-backed keyring) and returns the signature; `gpgshim` writes it to stdout for `rpmsign`

### Request Formats

**Simplified (preferred) — tier + artifact, resolved server-side via `config/authorization.json`:**

```json
{"data": "<base64>", "tier": "nightly", "artifact": "rpm"}
```

`gpgshim` uses this form when `GPG_TIER` is set. `tier` → GPG key via `key_aliases`; `artifact` → armor/clearsign/digest_algo via `artifact_profiles`.

**Legacy (explicit) — still supported, used by operator tooling and older integrations:**

```json
{"data": "<base64>", "key_id": "signer@example.com", "digest_algo": "SHA256", "armor": false}
```

**Response:**

```json
{"signature": "<base64>", "key_id": "actual-key-used", "digest_algo": "SHA256"}
```

### Authentication (three mechanisms, by phase — see design doc §4.1/§4.1a/§4.1b)

| Env var / flag                                                                | Mechanism                                                                                 | Phase |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----- |
| (none — network only)                                                         | VPC Security Groups                                                                       | 1     |
| `GPG_SIGNING_API_URL` set on the client; `--trust-apigw-header` on the server | AWS SigV4 through API Gateway, IAM role resolved via `authorization.json`'s `clients` map | 1b    |
| `GPG_SERVER_TOKEN` / `--enable-auth`                                          | Pre-shared app token (primary) / GitHub OIDC / legacy JWT                                 | 2     |

`gpgshim` picks its transport based on which client env var is set: `GPG_SIGNING_API_URL` (SigV4, Phase 1b) takes priority over `GPG_SIGNING_SERVER` (direct HTTP, Phase 1/operator path). Only one of `--enable-auth` / `--trust-apigw-header` may be active on the server at a time.

## Development Commands

There is no Makefile — run the server and shim directly.

### Manual local testing

```bash
# Terminal 1: start the server against a test keyring
export GNUPGHOME=/path/to/test/.gnupg
python3 server/signing-server.py --port 8080 --keyring "$GNUPGHOME"

# Terminal 2: sign something through gpgshim
export GPG_TIER=nightly
export GPG_SIGNING_SERVER='http://localhost:8080/sign'
echo "test data" | ../gpgshim --detach-sign --armor > test.sig

# Large-file check: only the ppid cache should prevent a second network call,
# not any digest-only optimization
dd if=/dev/urandom bs=1M count=100 | ../gpgshim -b > test.sig
```

### Existing tests (`tests/`)

These target the current `signing-server.py` (confirmed — they reference it by name, not the old `server-example.py`), but predate the Phase 1/1b/2 redesign and have not been re-verified against it. Treat failures as "needs updating for the current CLI flags/API," not necessarily as real regressions, until they've been reviewed.

## Code Constraints

### Python 3.6 Compatibility (gpgshim only — the server can use 3.9+)

- No f-strings in `gpgshim` (use `.format()` or `%` formatting)
- No type hints in `gpgshim`
- No `|` dict merge operator
- `urllib.request`, not `requests`

### Standard Library Only (gpgshim only)

`hashlib`, `hmac`, `json`, `base64`, `urllib.request`/`urllib.parse`, `datetime`, `sys`, `os` — including the Phase 1b AWS SigV4 signing implementation, which is hand-rolled against stdlib rather than depending on `boto3`/`botocore`.

## Security Features (server)

- Multi-threaded (`ThreadingMixIn`), semaphore-limited concurrent signing (default 10 threads, `--max-threads`)
- Request size limit: 512 KB default (`--max-request-size`) — RPM headers run 50-100 KB, metadata files are smaller
- Read timeout: 10s default (`--read-timeout`), guards against slow-write attacks
- Every request logged (`auth.py::audit_log()`) with `auth_type`, role, key_id, digest_algo, source IP, latency; OIDC/client_role entries additionally carry repo/ref/workflow context
- `TRUST_APIGW_HEADER` (Phase 1b) defaults off and must only be enabled on a server reachable exclusively through the API Gateway VPC Link — see design doc §4.1b before touching this

## Common Modifications

- **Adding a new signing tier/client:** edit `config/authorization.json` (`key_aliases`, `roles`, and — for Phase 1b — `clients`), not code. See `operations-runbook.md` §7.2 for the full onboarding procedure.
- **Changing SigV4 behavior:** the implementation lives in `gpgshim::_sigv4_headers()` — keep it stdlib-only; do not introduce a `boto3` dependency.
- **Changing auth logic:** `server/auth.py` has one `authorize_*_request()` function per mechanism (`authorize_request`, `authorize_oidc_request`, `authorize_client_role_request`) — keep new mechanisms structurally parallel rather than branching deep inside `signing-server.py`'s `do_POST`.
