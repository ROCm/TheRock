# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Point-in-time reuse estimates over imported evidence, not measured savings."""

from dataclasses import dataclass

from .cache_db import ReadPolicy, timestamp
from .manifest import TrustLevel, ValidationState


@dataclass
class ReplayReport:
    results: list[dict]

    def to_dict(self):
        units = sum(r["units"] for r in self.results)
        hits = sum(r["potential_hits"] for r in self.results)
        baseline = sum(r["same_commit_hits"] for r in self.results)
        return {"scope": "history_model", "payloads_verified": False,
                "inventory_scope": "recorded_outputs_only", "runs": len(self.results),
                "units": units, "potential_hits": hits, "same_commit_hits": baseline,
                "potential_hit_rate": hits / units if units else 0,
                "same_commit_hit_rate": baseline / units if units else 0,
                "per_run": self.results}

    def render(self):
        import json
        return json.dumps(self.to_dict(), indent=2)


def replay(scenarios, *, read_policy=None, topology=None):
    """Only completed earlier productions can satisfy a run at its start.

    Counts only outputs actually listed. Missing expected outputs cannot be
    inferred from run history, so this does not claim complete stage skipping.
    Corrections to the same run attempt use the latest imported snapshot.
    """
    policy = read_policy or ReadPolicy(min_trust=TrustLevel.UNTRUSTED)
    unique = {}
    for run in scenarios:
        if run.get("schema_version") != 1:
            raise ValueError("unsupported history evidence version")
        timestamp(run["started_at"])
        if timestamp(run["available_at"]) < timestamp(run["started_at"]):
            raise ValueError("run availability predates start")
        key = (run["repository"], run["platform"], run["build_variant"], run["run_id"], run["run_attempt"])
        unique[key] = run
    runs = sorted(unique.values(), key=lambda r: (timestamp(r["started_at"]), r["run_id"], r["run_attempt"]))
    results = []
    for run in runs:
        start = timestamp(run["started_at"])
        cutoff = min(start, timestamp(policy.created_before)) if policy.created_before else start
        eligible = [p for p in runs if p is not run
                    and timestamp(p["available_at"]) < cutoff
                    and all(p[k] == run[k] for k in ("repository", "platform", "build_variant"))
                    and TrustLevel(p["trust"]).meets(policy.min_trust)
                    and ValidationState(p["validation"]).meets(ValidationState.BUILD_PASSED)
                    and ValidationState(p["validation"]).meets(policy.min_validation)]
        candidates = {(u["artifact_identity"], u["fingerprint_kind"], u["fingerprint"])
                      for p in eligible for u in p["units"]}
        same_commit = {u["artifact_identity"] for p in eligible if p["commit"] == run["commit"] for u in p["units"]}
        decisions, seen = [], set()
        for unit in run["units"]:
            identity = unit["artifact_identity"]
            if identity in seen:
                raise ValueError("duplicate component in history evidence")
            seen.add(identity)
            hit = (identity, unit["fingerprint_kind"], unit["fingerprint"]) in candidates
            decisions.append({"artifact_identity": identity,
                              "stage": topology.stage_for(identity) if topology else None,
                              "potential_hit": hit, "same_commit_hit": identity in same_commit,
                              "fingerprint_kind": unit["fingerprint_kind"]})
        results.append({"run_id": run["run_id"], "run_attempt": run["run_attempt"],
                        "repository": run["repository"], "started_at": run["started_at"],
                        "units": len(decisions), "potential_hits": sum(d["potential_hit"] for d in decisions),
                        "same_commit_hits": sum(d["same_commit_hit"] for d in decisions),
                        "decisions": decisions})
    return ReplayReport(results)
