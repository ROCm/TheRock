# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Readable CLI results on stderr; stdout remains a JSON report."""

import sys

from .diagnostics import record_message


def emit(message):
    # Keep each record on one line even if a diagnostic contains newlines.
    message = message.replace("\r", "\\r").replace("\n", "\\n")
    print(f"[artifact-cache] {message}", file=sys.stderr, flush=True)
    record_message(message)


def log_progress(status, artifact_identity=None, **details):
    """Flush an operation event immediately; it is not a final success report."""
    fields = []
    for name, value in details.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        fields.append(str(value) if name == "reason" else f"{name}={value}")
    identity = f" {artifact_identity}" if artifact_identity is not None else ""
    suffix = f" ({', '.join(fields)})" if fields else ""
    emit(f"{status}{identity}{suffix}")


def log_build_progress(status, artifact_identity=None, **details):
    """Show operation phases, keeping all per-artifact outcomes for the end."""
    if artifact_identity is not None:
        return
    messages = {
        "SCANNING": "CONFIGURE: refreshing artifact fingerprints",
        "VERIFYING_CACHE": "CACHE CHECK: looking up keys and verifying cached files",
        "CONFIGURING": "CHECK: validating inputs before the build (configure output saved to logs)",
        "BUILD_START": "BUILD: running the default incremental build",
        "BUILD_PASSED": "BUILD PASSED: validating inputs after the build (configure output saved to logs)",
        "BUILD_FAILED": "BUILD FAILED: outputs will not be published",
        "SAVING_OUTPUTS": "PUBLISH: saving outputs and recording verified inventories",
        "RECHECKING": "CHECK: validating final artifact keys (configure output saved to logs)",
        "NO_BUILD": "BUILD NOT RUN: no cacheable misses; see the final report for skipped artifacts",
    }
    if status in messages:
        emit(messages[status])


def log_report(report, *, include_artifacts=True):
    """Log completed cache/publication/restore results, never metadata hits.

    Call only after the operation returns, including its final input checks.
    REUSED is the display label for verified_cached: verified in temporary
    storage, not installed into a consumer build or used to skip a CI stage.
    Only the cache operation performs a lookup before publishing its misses;
    direct publication must not claim a preceding MISS.
    """
    operation = report.get("operation")
    if operation not in ("cache", "build-publish", "restore"):
        return

    artifacts = report["published"] if operation == "build-publish" else report["artifacts"]
    for artifact in artifacts if include_artifacts else ():
        status = "stored" if operation == "build-publish" else artifact["outcome"]
        display_status = status.upper()
        if operation == "cache":
            if status == "stored":
                display_status = "MISS -> STORED"
                reason = "no usable cache entry; saved after successful build"
            elif status == "verified_cached":
                display_status = "REUSED"
                reason = "matching cached files verified"
            elif status == "uncacheable":
                # Only an explicitly missing sidecar means NO_FINGERPRINT.
                # Preserve other diagnostics (invalid contents, unreadable file).
                reason = artifact.get("reason") or "missing or invalid fingerprint"
                if artifact.get("fingerprint_state") == "missing" or reason == "expected fingerprint file is missing":
                    display_status = "NO_FINGERPRINT"
                    reason = (reason if artifact.get("components") else ".fprint file missing")
                else:
                    display_status = "CACHE_SKIPPED"
                reason += "; cache restore and storage skipped"
            else:
                reason = artifact.get("reason")
            if artifact.get("components"):
                from .inventory import parse_identity
                components = [parse_identity(name)[1] if parse_identity(name) else name
                              for name in artifact["components"]]
                reason = f"components={','.join(components)}; " + (reason or "")
            suffix = f" ({reason})" if reason else ""
            emit(f"{display_status} {artifact['artifact_identity']}{suffix}")
            continue
        details = []
        epoch = artifact.get("epoch", report.get("epoch"))
        if epoch is not None:
            details.append(f"epoch={epoch}")
        if artifact.get("cache_entry_id") is not None:
            details.append(f"entry={artifact['cache_entry_id']}")
        if artifact.get("payloads_verified") is True:
            details.append("payloads_verified=true")
        if artifact.get("reason"):
            details.append(artifact["reason"])
        suffix = f" ({', '.join(details)})" if details else ""
        emit(f"{display_status} {artifact['artifact_identity']}{suffix}")

    if operation == "cache":
        summary = report["summary"]
        emit(f"SUMMARY stored={summary['stored']} "
             f"reused={summary['verified_cached']} "
             f"cache_skipped={summary['uncacheable']}"
             + (" units=bundles" if report.get("granularity") == "bundle" else ""))
    elif operation == "build-publish":
        emit(f"SUMMARY stored={len(artifacts)} run_id={report['run_id']} build_executed=true")
    else:
        emit(f"SUMMARY restored={report['restored']} needs_build={report['needs_build']} "
             f"destination={report['destination']}")
