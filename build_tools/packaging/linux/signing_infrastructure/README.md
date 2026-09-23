# Signing Infrastructure — Remote GPG Signing Service

Remote GPG signing service for TheRock's RPM/DEB packages and repository metadata. Build runners never hold a GPG private key — they send data to be signed to this service and receive back a signature.

**Start here:** [`docs/signing-server-design.md`](docs/signing-server-design.md) (architecture, rationale) and [`docs/signing-server-requirements.md`](docs/signing-server-requirements.md) (phased requirements) are the authoritative design documents. This README is an orientation map, not a design spec — if anything here conflicts with those two documents, they win.

## Phases

| Phase  | What it covers                                                                                                                                                                                            |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**  | Working signing pipeline, single server, VPC-Security-Group-only access control (no application-layer auth)                                                                                               |
| **1b** | Cross-cloud client access — build runners outside AWS (e.g. Azure) authenticate via AWS SigV4 through an API Gateway front door, reusing the GitHub OIDC → AWS IAM federation already used for S3 uploads |
| **2**  | Production hardening — primary/secondary with ALB failover, pre-shared app-token auth, server-side RPM signing                                                                                            |

See `signing-server-requirements.md` §1 / §2A / §3 for the full breakdown of each phase.

## Project Structure

```
build_tools/packaging/linux/
├── gpgshim                          # Client shim — lives OUTSIDE this folder, one level up.
│                                     # Drop-in `gpg` replacement invoked by rpmsign.
└── signing_infrastructure/
    ├── server/
    │   ├── signing-server.py        # The signing server (http.server, threaded)
    │   └── auth.py                  # Auth/authorization module (all phases)
    ├── tools/
    │   ├── sign-file                # Operator CLI (manual signing over VPN)
    │   ├── setup-server.sh          # Server provisioning script
    │   └── generate-token.py        # Generates Phase 2 legacy JWT tokens
    ├── config/
    │   ├── authorization.json       # key_aliases, artifact_profiles, roles, clients (Phase 1b)
    │   └── secrets.json.example     # Template for the legacy JWT shared-secret mechanism
    ├── tests/                       # Test scripts — predate the Phase 1/1b/2 design;
    │                                 # target the current signing-server.py but may need
    │                                 # updating/verifying against it, not yet re-validated
    └── docs/
        ├── signing-server-design.md         # Architecture, security model, rationale
        ├── signing-server-requirements.md   # Phased requirements (P1-*, P1b-*, P2-*)
        ├── operations-runbook.md            # Provisioning, key rotation, incident response
        ├── use-case-flows.md                # End-to-end flows per actor
        └── code-flows.md                    # Corresponding code-level execution paths
```

## Quick Start

**Provision the server** (see `docs/operations-runbook.md` §1 for the full sequence — IAM role, KMS key, Secrets Manager, then the server itself):

```bash
./tools/setup-server.sh
```

**Sign something as an operator** (over VPN, Phase 1 direct path):

```bash
./tools/sign-file --server https://<signing-server-ip>/sign --tier nightly --artifact rpm --file mypackage.rpm
```

**CI build runners** don't call the server directly — they go through `gpgshim` (RPM packages) or call `POST /sign` directly (repo metadata), configured via environment variables set by the workflow. See `docs/use-case-flows.md` for the exact sequence, and `docs/signing-server-design.md` §4.1b if the runner is outside AWS (cross-cloud/SigV4 path).

## The Simplified Signing API

Callers specify *what* they want signed, not raw GPG parameters — the server resolves the rest from `config/authorization.json`:

```json
{"data": "<base64>", "tier": "nightly", "artifact": "rpm"}
```

`tier` resolves to a GPG key via `key_aliases`; `artifact` resolves to armor/clearsign/digest_algo via `artifact_profiles`. The legacy explicit form (`key_id`, `digest_algo`, `armor`, `clearsign`) is still accepted for backward compatibility and operator tooling.

## Authentication Model

Three independent mechanisms, layered by phase — see `docs/signing-server-design.md` §4.1/§4.1a/§4.1b for full detail:

| Phase | Mechanism                                                                                      | Used by                                               |
| ----- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 1     | VPC Security Groups only                                                                       | Same-cloud (AWS) build runners, operator VPN          |
| 1b    | AWS SigV4 + API Gateway, IAM-role identity mapped through `authorization.json`'s `clients` map | Cross-cloud (e.g. Azure) build runners                |
| 2     | Pre-shared app token (primary) / GitHub OIDC / legacy JWT                                      | Any caller, added on top of the network-layer control |

`therock-release` (the most sensitive key) is intentionally excluded from the Phase 1b automated path — release signing stays a manual, operator-driven process.

## Security

Do not read this README for the security model — see `docs/signing-server-design.md` §8 (three-layer model: network perimeter, key-at-rest/in-distribution, instance hardening) and the corresponding `operations-runbook.md` procedures. Key points that matter regardless of phase:

- The GPG private key lives only in a tmpfs-backed keyring, loaded from AWS Secrets Manager (KMS-encrypted) at startup — never written to disk.
- The signing server has no internet egress and no public IP in any phase, including 1b (only a new AWS-managed API Gateway front door is internet-reachable; the server itself is unchanged).
- Never commit `config/secrets.json` (real secrets) — only `config/secrets.json.example` (template) belongs in version control.
