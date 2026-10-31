# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Common controls for the opt-in normal two-family CI comparison.

No product optimization is applied. Commands execute only inside TheRock Actions.
Unit fixtures may call the pure validation/configuration functions directly.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normal_ci_input_identity import (
    declared_sources,
    generated_inputs,
    loader_dependencies,
    observed_provider_identities,
    query_loader,
    runtime_cache_state,
    sdk_inputs,
)

PRODUCT_SHA = "5b7cb88373362ab80dc0f19605915e534f4158aa"
VERSION_ORIGIN_SHA = "82d5e8c176ce0cc00b5fd1cf631ce7790be8992b"
CONFIG_SHA = "7624a5e209cec569bbed38dad7071fa7393cb52b"
GITLINKS = {
    "base/half": "207ee58595a64b5c4a70df221f1e6e704b807811",
    "base/rocm-cmake": "10155d7272ea1bf79f6b5a9dbc339657af1aa372",
    "compiler/amd-llvm": "e095f1b3becf0c1bfdc5e7647003d60976a2460b",
    "compiler/hipify": "d364ef407b2c3e20c6ceb3575bd852f0c7cebb8c",
    "compiler/spirv-llvm-translator": "d50d9db05ea340453d2b4abea4c546c93b9494e4",
    "debug-tools/rocgdb/source": "003062db3007894472038e57ac223e81d10420df",
    "math-libs/libhipcxx": "88f23f01fd7000c1ad25492311693e8e7ccabe31",
    "rocm-docs": "9bcf25db1e77f58b435fe162baaa99566bac0189",
    "rocm-libraries": "f4e24d58b3f313f1a31068def1e35376e43c2f49",
    "rocm-systems": "a10dca087065a82f15ea89dc7d6fdd3c6bb2c246",
    "third-party/sysdeps/common/mesa-fork": "4b0d67e19365c50d1a60e8ada444323c4c82c458",
}
PACKAGE_SUFFIX = "41147801272"
CI_CACHE_REMOTE = "http://bazelremote-svc.bazelremote-ns.svc.cluster.local:8080|layout=bazel|connect-timeout=50"
VERSIONS = {
    "rocm_package_version": f"10.3.0.dev0+{VERSION_ORIGIN_SHA}",
    "rocm_deb_package_version": "10.3.0~dev20261031",
    "rocm_rpm_package_version": "10.3.0~20261031g82d5e8c1",
    "rocm_msi_package_version": "10.3.3865.0",
}
INSTALL_IMAGES = {
    "ubuntu2404": "ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:a13e8370a2d1d33a910e47669cdf075dbe9554685e78592f22935ec7c7c35e39",
    "rhel8": "registry.access.redhat.com/ubi8/ubi@sha256:c322aa68f2d630b36b9faab64094357f62720be2f56f7099376ca2d6c4fe389b",
    "rhel10": "registry.access.redhat.com/ubi10/ubi@sha256:15f0a6e2b448ac7aa24425c9ab541e6402dc32b782844a285ccaeffb1c938fe1",
    "sles16": "registry.suse.com/bci/bci-base@sha256:38bb07164a0b64c429a285e877277d618fa7531eccd0d31d34375b9ea247c843",
}


def validate_context(env, event):
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
        or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
    ):
        raise ValueError("normal validation executes exclusively in TheRock Actions")
    inputs = event.get("inputs", {})
    if str(inputs.get("profile_normal_validation", "")).lower() != "true":
        raise ValueError("explicit normal validation mode required")
    families = [
        s.strip().lower()
        for s in inputs.get("linux_amdgpu_families", "").split(",")
        if s.strip()
    ]
    if len(families) != 2 or set(families) != {"gfx94x", "gfx103x"}:
        raise ValueError("this comparison requires exactly gfx94x and gfx103x")
    required = {
        "windows_amdgpu_families": "",
        "build_stages": "",
        "prebuilt_stages": "",
        "baseline_run_id": "",
        "stage_reuse_mode": "off",
        "changed_projects": "",
        "pr_labels": "test_filter:comprehensive",
        "build_python_packages": "true",
        "build_native_linux": "true",
        "build_native_windows": "false",
        "build_pytorch": "false",
        "build_jax": "false",
    }
    for key, value in required.items():
        if str(inputs.get(key, "")).lower() != value:
            raise ValueError(f"normal validation input mismatch: {key}")
    if inputs.get("profile_normal_case", "baseline") not in ["baseline", "candidate"]:
        raise ValueError("unknown case label")
    if inputs.get("profile_normal_cache_regime", "ci-readonly") not in [
        "ci-readonly",
        "private-cold",
    ]:
        raise ValueError("unknown comparison cache regime")
    if env.get("RELEASE_TYPE", "ci") != "ci":
        raise ValueError("release destinations forbidden in normal validation")
    return inputs


def guard():
    with Path(os.environ["GITHUB_EVENT_PATH"]).open(encoding="utf-8") as stream:
        event = json.load(stream)
    return validate_context(os.environ, event)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def output(values):
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            if "\n" in str(value) or "\r" in str(value):
                raise ValueError("multiline output rejected")
            stream.write(f"{key}={value}\n")


def fixed_runner_config(config):
    """Keep GPU policy unchanged; freeze only normal Linux build pool choices."""
    result = json.loads(json.dumps(config))
    for size, label in {
        "default": "aws-linux-scale-rocm-prod",
        "small": "aws-linux-scale-rocm-small",
        "medium": "aws-linux-scale-rocm-medium",
    }.items():
        result["build_runners"]["linux"][size] = [{"label": label, "weight": 1.0}]
    return result


def controlled_cache_text(text, private_dir, regime):
    if regime not in ["ci-readonly", "private-cold"]:
        raise ValueError("unknown comparison cache regime")
    original = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            original[key.strip()] = value.strip()
    if regime == "ci-readonly" and original.get("remote_storage") != CI_CACHE_REMOTE:
        raise ValueError(
            "unexpected CI cache backend; trust domains must stay separate"
        )
    if original.get("namespace") != "therock-v4":
        raise ValueError("unexpected compiler cache namespace")
    overrides = {
        "cache_dir": str(private_dir),
        "remote_storage": (
            CI_CACHE_REMOTE + "|read-only=true" if regime == "ci-readonly" else ""
        ),
        "remote_only": "false",
        "read_only": "true" if regime == "ci-readonly" else "false",
        "read_only_direct": "false",
        "recache": "false",
        "disable": "false",
    }
    retained = [
        line
        for line in text.splitlines()
        if line.split("=", 1)[0].strip() not in overrides
    ]
    retained.extend(f"{key} = {value}" for key, value in overrides.items())
    return "\n".join(retained) + "\n"


def child_environment(env):
    result = {key: value for key, value in env.items() if not key.startswith("CCACHE_")}
    config = env.get("CCACHE_CONFIGPATH")
    if not config:
        raise ValueError("controlled compiler cache config missing")
    result["CCACHE_CONFIGPATH"] = config
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def setup(args):
    rows = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], text=True)
    observed_links = {}
    for row in rows.splitlines():
        metadata, path = row.split("\t", 1)
        mode, _, sha = metadata.split()
        if mode == "160000":
            observed_links[path] = sha
    if observed_links != GITLINKS:
        raise ValueError("product source gitlinks differ from recorded HEAD")
    config_root = args.config.resolve()
    observed = subprocess.check_output(
        ["git", "-C", str(config_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != CONFIG_SHA:
        raise ValueError("external runner configuration revision differs")
    config_path = config_root / "runner-config-v2.json"
    before = digest(config_path)
    with config_path.open(encoding="utf-8") as stream:
        original = json.load(stream)
    updated = fixed_runner_config(original)
    save(config_path, updated)
    save(
        args.output,
        dict(
            product_sha=PRODUCT_SHA,
            version_origin_sha=VERSION_ORIGIN_SHA,
            config_sha=observed,
            original_config_sha256=before,
            controlled_config_sha256=digest(config_path),
            versions=VERSIONS,
            package_build_suffix=PACKAGE_SUFFIX,
            installed_image_pins=INSTALL_IMAGES,
            source_patch_policy="No product patch applied; preserve submodule HEAD/history",
            recorded_source_gitlinks=GITLINKS,
            cache_policy="explicit ci-readonly preferred or distinct private-cold; no shared writes",
        ),
    )
    output(VERSIONS)


def cache(args, inputs):
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    configured = Path(os.environ["CCACHE_CONFIGPATH"])
    config = configured.resolve()
    private = workspace / ".normal-validation-cache"
    if (
        configured.is_symlink()
        or workspace not in config.parents
        or private.is_symlink()
    ):
        raise ValueError("comparison cache/config must be owned workspace paths")
    if private.exists() and any(private.iterdir()):
        raise ValueError("private comparison cache must start empty")
    private.mkdir(exist_ok=True)
    config.write_text(
        controlled_cache_text(
            config.read_text(encoding="utf-8"),
            private,
            inputs.get("profile_normal_cache_regime", "ci-readonly"),
        ),
        encoding="utf-8",
    )
    save(
        args.output,
        dict(
            cache_config_sha256=digest(config),
            private_directory=str(private),
            initial_entries=0,
            namespace="therock-v4",
            regime=inputs.get("profile_normal_cache_regime", "ci-readonly"),
            remote_storage=(
                "CI backend read-only"
                if inputs.get("profile_normal_cache_regime", "ci-readonly")
                == "ci-readonly"
                else "disabled"
            ),
            external_mutation_gap="remote cache contents may change independently; compare actual critical-command keys/hit outcomes",
            environment_policy="run wrapper removes inherited CCACHE_* except configpath",
        ),
    )


def capture(args, inputs):
    start = time.monotonic()
    captured_hashes = {}

    def fingerprint(path):
        resolved = path.resolve()
        stat = resolved.stat()
        key = (
            (stat.st_dev, stat.st_ino) if stat.st_ino else str(resolved),
            stat.st_size,
            stat.st_mtime_ns,
        )
        if key not in captured_hashes:
            captured_hashes[key] = digest(resolved)
            after = resolved.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError("tool/source input changed during fingerprint capture")
        return captured_hashes[key]

    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    data = {
        "product_sha": PRODUCT_SHA,
        "case": inputs.get("profile_normal_case", "baseline"),
        "candidate_patch": "none integrated; case is a label only",
        "ci": {
            key: os.environ.get(key)
            for key in [
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "RUNNER_NAME",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "NORMAL_IMAGE_PIN",
                "NORMAL_REQUESTED_RUNNER",
            ]
        },
        "tools": {},
        "path_tools_scope": "ambient PATH only; active generated CMake selections captured separately",
        "source": {},
    }
    image = os.environ.get("NORMAL_IMAGE_PIN", "")
    if image and "@sha256:" not in image and not image.startswith("native-"):
        raise ValueError("normal comparison container reference must be immutable")
    data["allocation"] = dict(
        logical_cpus=os.cpu_count(),
        affinity=(
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    )
    for path in [
        "/etc/os-release",
        "/proc/self/cgroup",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
    ]:
        p = Path(path)
        if p.is_file():
            data["allocation"][path] = p.read_text()
    for name in [
        "cmake",
        "ninja",
        "ccache",
        "clang",
        "clang++",
        "ld.lld",
        "patchelf",
        "readelf",
        "dpkg-deb",
        "rpm",
    ]:
        resolved = shutil.which(name)
        if not resolved:
            continue
        p = Path(resolved).resolve()
        version = subprocess.run([str(p), "--version"], capture_output=True, text=True)
        data["tools"][name] = dict(
            path=str(p),
            sha256=fingerprint(p),
            version=version.stdout[:8192],
            returncode=version.returncode,
        )
    for name, command in {
        "checkout_sha": ["git", "rev-parse", "HEAD"],
        "working_changes": ["git", "status", "--porcelain", "--untracked-files=no"],
        "submodule_status": ["git", "submodule", "status", "--recursive"],
        "checkout_tree": ["git", "ls-tree", "-r", "HEAD"],
    }.items():
        result = subprocess.run(command, cwd=workspace, capture_output=True, text=True)
        data["source"][name] = dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
    data["source"]["recorded_product_gitlinks"] = GITLINKS
    data["source"]["repositories"] = []
    for relative in ["."] + list(GITLINKS):
        repository = workspace / relative
        if not (repository / ".git").exists():
            continue
        head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(repository), "diff", "--binary", "HEAD"]
        )
        names = (
            subprocess.check_output(
                ["git", "-C", str(repository), "diff", "--name-only", "-z", "HEAD"]
            )
            .decode()
            .split("\0")
        )
        files = []
        for name in names:
            path = repository / name
            if name and path.is_file() and not path.is_symlink():
                files.append(
                    dict(path=name, sha256=fingerprint(path), bytes=path.stat().st_size)
                )
        data["source"]["repositories"].append(
            dict(
                path=relative,
                head=head,
                diff_sha256=hashlib.sha256(diff).hexdigest(),
                changed_files=files,
            )
        )
        history = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
        )
        data["source"]["repositories"][-1]["is_shallow"] = history.stdout.strip()
        if relative == "rocm-systems":
            version_commit = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "log",
                    "-1",
                    "--format=%H",
                    "--",
                    "projects/clr",
                ],
                capture_output=True,
                text=True,
            )
            data["source"]["repositories"][-1]["clr_last_directory_commit"] = dict(
                returncode=version_commit.returncode,
                stdout=version_commit.stdout.strip(),
            )
    data["python_packages"] = sorted(
        (d.metadata.get("Name", ""), d.version)
        for d in importlib.metadata.distributions()
    )
    for executable, command in {
        "dpkg-query": ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"],
        "rpm": ["rpm", "-qa", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\n"],
    }.items():
        if shutil.which(executable):
            result = subprocess.run(command, capture_output=True, text=True)
            data[executable + "_inventory"] = dict(
                returncode=result.returncode, stdout=result.stdout
            )
    build = workspace / "build"
    data["generated_build_inputs"] = generated_inputs(build, fingerprint)
    data["sdk_runtime_inputs"] = sdk_inputs(build, fingerprint)
    loader_candidates = set()
    for row in data["generated_build_inputs"]["fields"]:
        identity = row.get("identity", {})
        if row["role"] == "generated-compiler" and identity.get("status") == "file":
            loader_candidates.add(identity["resolved_path"])
    for row in data["sdk_runtime_inputs"]["files"]:
        if row.get("status") == "file" and any(
            key in Path(row["path"]).name
            for key in [
                "libamd_comgr",
                "libhiprtc",
                "libamdhip64",
                "libhsa-runtime64",
                "rocfft_rtc_helper",
            ]
        ):
            loader_candidates.add(row["resolved_path"])
    data["resolved_loader_providers"] = [
        loader_dependencies(Path(binary), fingerprint, query_loader)
        for binary in sorted(loader_candidates)
    ]
    data["observed_runtime_provider_inputs"] = observed_provider_identities(
        build / "logs/profile/samples.jsonl", fingerprint
    )
    data["runtime_loader_environment"] = {
        key: os.environ.get(key) for key in ["LD_LIBRARY_PATH", "LD_PRELOAD"]
    }
    data["rtc_comgr_cache_snapshot"] = runtime_cache_state(
        workspace, os.environ, sys.platform
    )
    data["rtc_comgr_cache_snapshot"]["capture_phase"] = args.output.stem
    data["declared_source_inputs"] = declared_sources(
        workspace,
        Path(__file__).with_name("normal_ci_declared_inputs.json"),
        inputs.get("profile_normal_case", "baseline"),
        fingerprint,
        stage=os.environ.get("STAGE_NAME"),
    )
    data["capture_wall_s"] = time.monotonic() - start
    save(args.output, data)
    if not data["declared_source_inputs"]["valid"]:
        raise ValueError("declared source/patch input mismatch; receipt retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("setup")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ["cache", "capture"]:
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("preconfigure")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    commands.add_parser("guard")
    p = commands.add_parser("install-image")
    p.add_argument("--os-profile", choices=INSTALL_IMAGES, required=True)
    args = parser.parse_args()
    inputs = guard()
    if args.command == "setup":
        setup(args)
    elif args.command == "cache":
        cache(args, inputs)
    elif args.command == "capture":
        capture(args, inputs)
    elif args.command == "preconfigure":
        # Deliberate extension point: later reviewed product patches belong here,
        # after stock source fetch and before configure, without advancing source HEAD.
        # This common-control commit applies none and does not change compile flags.
        save(
            args.output,
            dict(
                case=inputs.get("profile_normal_case", "baseline"),
                product_patches=[],
                compiler_option_changes=[],
                source_git_commit_mutations=False,
            ),
        )
    elif args.command == "install-image":
        output({"container_image": INSTALL_IMAGES[args.os_profile]})
    elif args.command == "run":
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not argv:
            raise ValueError("child command required")
        return subprocess.call(argv, env=child_environment(os.environ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
