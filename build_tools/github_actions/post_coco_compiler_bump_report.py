#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Posts RCCL CoCo results from Compiler Daily Bump runs onto TheRock PRs
that ship the same compiler.

CoCo only runs inside compiler_bump.yml, which is not tied to any PR, while
compiler promotion PRs are opened separately (often hours later, and with
the LLVM commits re-created so their hashes differ). This script joins the
two by source tree rather than commit hash:

  collect        In compiler_bump.yml, after CoCo: gathers each cluster's
                 CoCo summary plus the bump branch's compiler submodule
                 commits and trees into a report directory, uploaded as the
                 artifact `coco-compiler-bump-report-<amd-llvm tree>`.
  post-open-prs  In compiler_bump.yml, after upload: posts the report onto
                 every open PR whose compiler submodules have the same trees.
  post-pr        On a PR that changes a compiler submodule: finds the newest
                 matching report artifact and posts it, or posts a status
                 comment saying no CoCo run matches this compiler yet.

Each cluster's report is a sticky comment (see report_marker()); reports
from an older compiler are marked outdated rather than deleted.

Example usage:
  python build_tools/github_actions/post_coco_compiler_bump_report.py \\
      post-pr --pr 10069 --dry_run
"""

import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen
import zipfile

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
if str(THIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_SCRIPT_DIR))

from github_actions_api import GitHubAPI, GitHubAPIError, gha_set_output

THEROCK_REPO = "ROCm/TheRock"
API = "https://api.github.com"

# Submodules that define "the compiler". A report applies to a PR only when
# all three have identical source trees.
COMPILER_SUBMODULES = {
    "compiler/amd-llvm": "ROCm/llvm-project",
    "compiler/hipify": "ROCm/HIPIFY",
    "compiler/spirv-llvm-translator": "ROCm/SPIRV-LLVM-Translator",
}
ROCM_SYSTEMS_PATH = "rocm-systems"

ARTIFACT_PREFIX = "coco-compiler-bump-report-"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = 1

MARKER_PREFIX = "<!-- coco-compiler-bump-report:"
STATUS_MARKER = f"{MARKER_PREFIX}status -->"
OUTDATED_NOTE = (
    "> [!WARNING]\n"
    "> Outdated: this PR's compiler submodules changed after this report was "
    "posted, so it no longer describes this PR.\n\n"
)
COMMENT_AUTHOR = "github-actions[bot]"

# rccl-coco-run.yml uploads coco-run-<job>-<run id>-<attempt>-output.
_COCO_OUTPUT_RE = re.compile(r"^coco-run-(?P<job>.+)-\d+-(?P<attempt>\d+)-output$")
_MARKER_RE = re.compile(re.escape(MARKER_PREFIX) + r"(?P<key>[^ ]+) -->")

# GitHub rejects comment bodies over 65536 characters.
_MAX_COMMENT_CHARS = 60000
_COMMENT_PAGES = 10


def report_marker(job: str) -> str:
    return f"{MARKER_PREFIX}{job} -->"


def artifact_name(components: dict) -> str:
    return ARTIFACT_PREFIX + components["compiler/amd-llvm"]["tree"]


# --- Resolving what a ref builds ---


def get_submodule_commit(ref: str, path: str, api: GitHubAPI) -> str:
    """Commit that TheRock `ref` pins submodule `path` to."""
    entry = api.send_request(f"{API}/repos/{THEROCK_REPO}/contents/{path}?ref={ref}")
    if not isinstance(entry, dict) or entry.get("type") != "submodule":
        raise GitHubAPIError(f"{path} at {ref} is not a submodule")
    return entry["sha"]


def get_tree(repo: str, commit: str, api: GitHubAPI) -> str:
    """Root tree of `commit`: equal trees mean byte-identical source."""
    response = api.send_request(f"{API}/repos/{repo}/git/commits/{commit}")
    return response["tree"]["sha"]


def resolve_components(ref: str, api: GitHubAPI) -> dict:
    """Compiler submodule commits and trees pinned by TheRock `ref`."""
    components = {}
    for path, repo in COMPILER_SUBMODULES.items():
        commit = get_submodule_commit(ref, path, api)
        components[path] = {
            "repo": repo,
            "commit": commit,
            "tree": get_tree(repo, commit, api),
        }
    return components


def components_match(a: dict, b: dict) -> bool:
    return all(a[path]["tree"] == b[path]["tree"] for path in COMPILER_SUBMODULES)


# --- Report directory: manifest.json + reports/<job>.md ---


def collect_reports(coco_output_dir: Path) -> dict[str, str]:
    """Maps each CoCo job to its markdown summary, from downloaded
    `coco-run-*-output` artifacts (one subdirectory per artifact)."""
    latest: dict[str, tuple[int, str]] = {}
    for artifact_dir in (p for p in coco_output_dir.iterdir() if p.is_dir()):
        match = _COCO_OUTPUT_RE.match(artifact_dir.name)
        if not match:
            continue
        summaries = sorted(artifact_dir.glob("reports/*/*_summary.md"))
        if not summaries:
            print(f"[WARN] No CoCo summary in {artifact_dir.name}")
            continue
        job, attempt = match.group("job"), int(match.group("attempt"))
        if job in latest and latest[job][0] > attempt:
            continue
        text = "\n\n".join(s.read_text(encoding="utf-8").strip() for s in summaries)
        latest[job] = (attempt, text)
    return {job: text for job, (_, text) in latest.items()}


def write_report_dir(out_dir: Path, manifest: dict, reports: dict[str, str]) -> None:
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    for job, text in reports.items():
        (out_dir / "reports" / f"{job}.md").write_text(
            text.rstrip("\n") + "\n", encoding="utf-8"
        )
    manifest = dict(manifest, jobs=sorted(reports))
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def read_report_dir(report_dir: Path) -> tuple[dict, dict[str, str]]:
    manifest = json.loads((report_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported report schema {manifest.get('schema')!r}")
    reports = {
        job: (report_dir / "reports" / f"{job}.md").read_text(encoding="utf-8")
        for job in manifest["jobs"]
    }
    return manifest, reports


# --- Report artifacts ---


def download_artifact(artifact_id: int, dest: Path) -> None:
    path = f"/repos/{THEROCK_REPO}/actions/artifacts/{artifact_id}/zip"
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        request = Request(API + path, headers={"Accept": "application/vnd.github+json"})
        # Unredirected so the token is not forwarded to the blob-storage redirect.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
        with urlopen(request, timeout=300) as response:
            data = response.read()
    else:
        data = subprocess.run(
            ["gh", "api", path], capture_output=True, check=True, timeout=300
        ).stdout
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)


def find_report(
    components: dict, api: GitHubAPI, work_dir: Path
) -> tuple[dict, dict[str, str]] | None:
    """Newest unexpired report artifact whose compiler matches `components`."""
    name = artifact_name(components)
    response = api.send_request(
        f"{API}/repos/{THEROCK_REPO}/actions/artifacts?name={name}&per_page=100"
    )
    artifacts = sorted(
        (a for a in response.get("artifacts", []) if not a.get("expired")),
        key=lambda a: a["created_at"],
        reverse=True,
    )
    for artifact in artifacts:
        dest = work_dir / str(artifact["id"])
        download_artifact(artifact["id"], dest)
        manifest, reports = read_report_dir(dest)
        if components_match(manifest["components"], components):
            return manifest, reports
        print(f"[WARN] Artifact {artifact['id']} is named {name} but does not match")
    return None


# --- Comment bodies ---


def _short(sha: str) -> str:
    return f"`{sha[:12]}`"


def build_provenance(manifest: dict, components: dict, rocm_systems: str) -> str:
    run = f"[run {manifest['run_id']}]({manifest['run_url']})"
    branch = manifest["branch"]
    lines = [
        f"> CoCo results for this PR's compiler, from Compiler Daily Bump {run} "
        f"(branch [`{branch}`](https://github.com/{THEROCK_REPO}/tree/{branch})). "
        "That run built the same source as this PR for "
        + ", ".join(f"`{path}`" for path in COMPILER_SUBMODULES)
        + "."
    ]
    for path in COMPILER_SUBMODULES:
        theirs = manifest["components"][path]["commit"]
        ours = components[path]["commit"]
        if theirs != ours:
            lines.append(
                f"> `{path}` was built as {_short(theirs)}, which has the same "
                f"source tree as this PR's {_short(ours)}."
            )
    if manifest.get("rocm_systems") and manifest["rocm_systems"] != rocm_systems:
        lines.append(
            f"> `{ROCM_SYSTEMS_PATH}` differs: {_short(manifest['rocm_systems'])} "
            f"there, {_short(rocm_systems)} here."
        )
    return "\n>\n".join(lines)


def build_report_body(
    job: str, report: str, manifest: dict, components: dict, rocm_systems: str
) -> str:
    body = (
        f"{report_marker(job)}\n"
        f"{build_provenance(manifest, components, rocm_systems)}\n\n"
        f"{report.strip()}\n"
    )
    if len(body) > _MAX_COMMENT_CHARS:
        body = (
            body[:_MAX_COMMENT_CHARS]
            + f"\n\n_Truncated; full report in {manifest['run_url']}._\n"
        )
    return body


def build_status_body(components: dict) -> str:
    rows = "\n".join(
        f"| `{path}` | [{_short(c['commit'])}](https://github.com/{c['repo']}/commit/{c['commit']}) |"
        for path, c in components.items()
    )
    llvm = components["compiler/amd-llvm"]["commit"]
    hipify = components["compiler/hipify"]["commit"]
    spirv = components["compiler/spirv-llvm-translator"]["commit"]
    return (
        f"{STATUS_MARKER}\n"
        "### RCCL CoCo: no results for this compiler yet\n\n"
        "No Compiler Daily Bump run with CoCo results has built this PR's "
        "compiler source:\n\n"
        "| Submodule | Commit |\n|---|---|\n"
        f"{rows}\n\n"
        "To get results, run [Compiler Daily Bump]"
        f"(https://github.com/{THEROCK_REPO}/actions/workflows/compiler_bump.yml) "
        f"with `llvm_ref={llvm}`, `hipify_ref={hipify}`, `spirv_ref={spirv}` and "
        "`test_type=standard`. They are posted here when its CoCo jobs finish.\n"
    )


# --- PR comments ---


def list_report_comments(pr: int, api: GitHubAPI) -> dict[str, dict]:
    """This script's comments on `pr`, keyed by marker key (job or "status")."""
    found = {}
    url = f"{API}/repos/{THEROCK_REPO}/issues/{pr}/comments?per_page=100"
    for page in range(1, _COMMENT_PAGES + 1):
        comments = api.send_request(f"{url}&page={page}")
        for comment in comments:
            if comment.get("user", {}).get("login") != COMMENT_AUTHOR:
                continue
            match = _MARKER_RE.search(comment.get("body", ""))
            if match:
                found.setdefault(match.group("key"), comment)
        if len(comments) < 100:
            break
    return found


def _upsert(pr: int, existing: dict | None, body: str, api: GitHubAPI, dry_run: bool):
    if existing and existing["body"] == body:
        return
    if dry_run:
        action = f"update comment {existing['id']}" if existing else "post"
        print(f"[DRY RUN] Would {action} on #{pr}:\n{body}")
        return
    if existing:
        api.send_request(
            f"{API}/repos/{THEROCK_REPO}/issues/comments/{existing['id']}",
            method="PATCH",
            body={"body": body},
        )
    else:
        api.send_request(
            f"{API}/repos/{THEROCK_REPO}/issues/{pr}/comments",
            method="POST",
            body={"body": body},
        )


def _delete(pr: int, comment: dict, api: GitHubAPI, dry_run: bool):
    if dry_run:
        print(f"[DRY RUN] Would delete comment {comment['id']} on #{pr}")
        return
    api.send_request(
        f"{API}/repos/{THEROCK_REPO}/issues/comments/{comment['id']}",
        method="DELETE",
    )


def _mark_outdated(pr: int, comment: dict, api: GitHubAPI, dry_run: bool):
    body = comment["body"]
    if OUTDATED_NOTE in body:
        return
    marker = _MARKER_RE.search(body).group(0)
    rest = body.replace(marker, "", 1).lstrip("\n")
    _upsert(pr, comment, f"{marker}\n{OUTDATED_NOTE}{rest}", api, dry_run)


def sync_pr_comments(
    pr: int,
    components: dict,
    rocm_systems: str,
    found: tuple[dict, dict[str, str]] | None,
    api: GitHubAPI,
    dry_run: bool = False,
) -> None:
    """Makes `pr`'s report comments reflect `found` (None: no matching run)."""
    existing = list_report_comments(pr, api)
    status = existing.pop("status", None)
    current_jobs = set()
    if found:
        manifest, reports = found
        for job, report in sorted(reports.items()):
            body = build_report_body(job, report, manifest, components, rocm_systems)
            _upsert(pr, existing.get(job), body, api, dry_run)
            current_jobs.add(job)
        if status:
            _delete(pr, status, api, dry_run)
    else:
        _upsert(pr, status, build_status_body(components), api, dry_run)
    for job, comment in existing.items():
        if job not in current_jobs:
            _mark_outdated(pr, comment, api, dry_run)


# --- Open PRs ---

_OPEN_PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { number headRefOid files(first: 100) { nodes { path } } }
    }
  }
}
"""


def list_open_compiler_prs(api: GitHubAPI) -> list[tuple[int, str]]:
    """(number, head SHA) of open PRs that change a compiler submodule."""
    owner, name = THEROCK_REPO.split("/")
    prs, cursor = [], None
    while True:
        response = api.send_request(
            f"{API}/graphql",
            method="POST",
            body={
                "query": _OPEN_PRS_QUERY,
                "variables": {"owner": owner, "name": name, "cursor": cursor},
            },
        )
        page = response["data"]["repository"]["pullRequests"]
        for node in page["nodes"]:
            paths = {f["path"] for f in (node["files"] or {}).get("nodes", [])}
            if paths & COMPILER_SUBMODULES.keys():
                prs.append((node["number"], node["headRefOid"]))
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]


# --- Subcommands ---


def cmd_collect(args, api: GitHubAPI) -> None:
    reports = collect_reports(Path(args.coco_output_dir))
    if not reports:
        print("[WARN] No CoCo summaries found; nothing to report.")
        gha_set_output({"artifact_name": ""})
        return
    components = resolve_components(args.branch, api)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": args.run_id,
        "run_url": f"https://github.com/{THEROCK_REPO}/actions/runs/{args.run_id}",
        "branch": args.branch,
        "components": components,
        "rocm_systems": get_submodule_commit(args.branch, ROCM_SYSTEMS_PATH, api),
    }
    write_report_dir(Path(args.out_dir), manifest, reports)
    name = artifact_name(components)
    print(f"[INFO] Collected {sorted(reports)} as {name}")
    gha_set_output({"artifact_name": name})


def cmd_post_open_prs(args, api: GitHubAPI) -> None:
    manifest, reports = read_report_dir(Path(args.report_dir))
    llvm_tree = manifest["components"]["compiler/amd-llvm"]["tree"]
    failures = []
    for pr, head in list_open_compiler_prs(api):
        try:
            llvm = get_submodule_commit(head, "compiler/amd-llvm", api)
            if (
                get_tree(COMPILER_SUBMODULES["compiler/amd-llvm"], llvm, api)
                != llvm_tree
            ):
                continue
            components = resolve_components(head, api)
            if not components_match(manifest["components"], components):
                continue
            print(f"[INFO] #{pr} matches run {manifest['run_id']}; posting")
            rocm_systems = get_submodule_commit(head, ROCM_SYSTEMS_PATH, api)
            sync_pr_comments(
                pr, components, rocm_systems, (manifest, reports), api, args.dry_run
            )
        except GitHubAPIError as e:
            print(f"[WARN] Skipping #{pr}: {e}")
            failures.append(pr)
    if failures:
        raise RuntimeError(f"Failed to check PR(s) {failures}")


def cmd_post_pr(args, api: GitHubAPI) -> None:
    head = (
        args.head_sha
        or api.send_request(f"{API}/repos/{THEROCK_REPO}/pulls/{args.pr}")["head"][
            "sha"
        ]
    )
    components = resolve_components(head, api)
    rocm_systems = get_submodule_commit(head, ROCM_SYSTEMS_PATH, api)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    found = find_report(components, api, work_dir)
    if found:
        print(f"[INFO] #{args.pr} matches run {found[0]['run_id']}")
    else:
        print(f"[INFO] No CoCo report matches #{args.pr}'s compiler")
    sync_pr_comments(args.pr, components, rocm_systems, found, api, args.dry_run)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    posting = argparse.ArgumentParser(add_help=False)
    posting.add_argument(
        "--dry_run",
        action="store_true",
        help="Print comment changes instead of making them.",
    )

    collect = sub.add_parser("collect")
    collect.add_argument("--run_id", required=True)
    collect.add_argument("--branch", required=True)
    collect.add_argument("--coco_output_dir", required=True)
    collect.add_argument("--out_dir", required=True)

    post_open = sub.add_parser("post-open-prs", parents=[posting])
    post_open.add_argument("--report_dir", required=True)

    post_pr = sub.add_parser("post-pr", parents=[posting])
    post_pr.add_argument("--pr", type=int, required=True)
    post_pr.add_argument("--head_sha", default="")
    post_pr.add_argument("--work_dir", default="coco-report-artifacts")

    args = parser.parse_args(argv)
    api = GitHubAPI()
    {
        "collect": cmd_collect,
        "post-open-prs": cmd_post_open_prs,
        "post-pr": cmd_post_pr,
    }[args.command](args, api)


if __name__ == "__main__":
    main()
