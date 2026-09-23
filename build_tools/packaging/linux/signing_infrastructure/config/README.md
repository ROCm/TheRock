# Configuration Files

This directory holds the signing server's configuration. See `../docs/signing-server-design.md` §4.1/§4.1a/§4.1b for the full rationale behind each mechanism below — this file is a reference for the config *shape*, not the design.

## Files

| File                   | Committed?          | Purpose                                                                                            |
| ---------------------- | ------------------- | -------------------------------------------------------------------------------------------------- |
| `authorization.json`   | Yes                 | `key_aliases`, `artifact_profiles`, `roles`, and `clients` (Phase 1b) — no secrets, safe to commit |
| `secrets.json.example` | Yes (template only) | Shape of the legacy JWT shared-secret file (Phase 2 fallback mechanism, not the primary one)       |
| `secrets.json`         | **Never**           | Real JWT shared secrets, if the legacy mechanism is in use — gitignored                            |

There is no `authorization.json.example` — `authorization.json` itself contains no secrets and is the single source of truth; edit it directly and track changes through git history rather than copying from a template.

## `authorization.json` Structure

```json
{
  "key_aliases": {
    "dev": "dev-key@example.com",
    "nightly": "nightly-key@example.com"
  },
  "artifact_profiles": {
    "rpm":      {"armor": false, "clearsign": false, "digest_algo": "SHA256"},
    "repodata": {"armor": true,  "clearsign": false, "digest_algo": "SHA256"},
    "deb-release":   {"armor": true, "clearsign": false, "digest_algo": "SHA256"},
    "deb-inrelease": {"armor": true, "clearsign": true,  "digest_algo": "SHA256"}
  },
  "clients": {
    "therock-nightly": {"role": "therock-nightly"}
  },
  "roles": {
    "therock-nightly": {
      "allowed_keys": ["nightly-key@example.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 5000
    }
  }
}
```

### `key_aliases` and `artifact_profiles` — the simplified API

Callers send `{"tier": "nightly", "artifact": "rpm"}` instead of raw GPG parameters. `key_aliases` resolves `tier` → a GPG key identity; `artifact_profiles` resolves `artifact` → the armor/clearsign/digest_algo combination that artifact type needs. Add a new tier or artifact type here, not in server code.

### `roles` — what each authenticated identity may do

Rate limits and key/algorithm restrictions, keyed by role name. Current roles:

| Role              | Used by                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `default`         | Phase 1 catch-all (no auth) — keyed by source IP                                                   |
| `operator`        | `sign-file` CLI, manual signing over VPN                                                           |
| `therock-dev`     | Dev/PR builds                                                                                      |
| `therock-nightly` | Nightly builds                                                                                     |
| `therock-release` | Release builds — **not** reachable via the Phase 1b cross-cloud path; release signing stays manual |

`allowed_keys: []` means unrestricted (not yet locked down) — populate this per environment before relying on it as a security boundary.

### `clients` — Phase 1b cross-cloud identity mapping

Maps an AWS IAM role name (surfaced to the server via the `X-Signing-Client-Role` header, itself only ever set by API Gateway's own integration mapping — see design doc §4.1b) to an entry in `roles` above:

```json
"clients": {
  "therock-nightly": {"role": "therock-nightly"}
}
```

Only populated/consulted when the server runs with `--trust-apigw-header`. See `../docs/operations-runbook.md` §7.2 for the full new-client onboarding procedure (IAM role, resource policy, then this map).

**Validate JSON syntax after any edit:**

```bash
python3 -m json.tool authorization.json > /dev/null && echo "Valid JSON"
```

## `secrets.json` — Legacy JWT Mechanism (Phase 2 fallback, not primary)

Phase 2's *primary* mechanism is the pre-shared app token (typically sourced from AWS Secrets Manager — see design doc §4.1a); JWT with a locally-stored shared secret is the older fallback, kept for existing integrations. Don't build new integrations against it.

```bash
cp secrets.json.example secrets.json
chmod 600 secrets.json
./generate-token.py --generate-secret --client-id some-client
# paste the output into secrets.json
```

**Never commit `secrets.json`.** It's gitignored; only `secrets.json.example` (no real values) belongs in version control.

## Troubleshooting

| Symptom                    | Cause                                                             | Fix                                                                  |
| -------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------- |
| `Unknown role`             | The caller's resolved role isn't a key in `roles`                 | Add the role, or fix the `clients`/token mapping that resolved to it |
| `Unknown signing client`   | (Phase 1b) IAM role name has no entry in `clients`                | Add an entry — see operations-runbook.md §7.2                        |
| `Not authorized for key`   | `roles[role].allowed_keys` doesn't include the requested key/tier | Add the key, or use a different tier                                 |
| Server doesn't see an edit | Server caches config in memory (`_authz_cache`)                   | Restart the server — there's no separate reload signal               |

## See Also

- `../docs/signing-server-design.md` §4.1/§4.1a/§4.1b — why each mechanism exists
- `../docs/operations-runbook.md` §3 (key rotation), §7 (Phase 1b provisioning + client onboarding)
- `./generate-token.py --help` — legacy JWT token utility usage
