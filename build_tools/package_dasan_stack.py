#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Package this run's complete DASan stack and validate the extracted SDK.

Uses the standard release tarball implementation. Archives contain bin/, lib/,
include/ and share/ at their root. GPU execution is a separate workflow gate.
"""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

from _therock_utils.artifacts import ArtifactName
from _therock_utils.build_topology import get_topology

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_NAME = "dasan-stack-sdk-metadata-gfx950"

# BUILD_TOPOLOGY includes dependency/configuration nodes as well as archives.
# Keep this list source-backed and explicit; never infer completeness from
# whichever files happened to upload. OpenMPI is a real archive and stays
# required (third-party/openmpi/CMakeLists.txt: therock_provide_artifact).
LINUX_NON_ARCHIVE_NODES = {
    "core-hipinfo": "core/CMakeLists.txt declares this artifact only inside if(WIN32)",
    "fmt": "third-party/fmt/CMakeLists.txt declares a build dependency, no artifact",
    "grpc": "third-party/grpc/CMakeLists.txt declares a build dependency, no artifact",
    "spdlog": "third-party/spdlog/CMakeLists.txt declares a build dependency, no artifact",
    "sysdeps-libmnl": "third-party/sysdeps/linux/CMakeLists.txt bundles therock-libmnl in sysdeps",
    "sysdeps-libnl": "third-party/sysdeps/linux/CMakeLists.txt bundles therock-libnl in sysdeps",
}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(argv: list[str | Path], log: Path, env: dict | None = None) -> str:
    command = [str(arg) for arg in argv]
    print("++", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.stdout


def required_artifacts(topology) -> set[str]:
    """All Linux/x86_64 topology artifacts, including WSL support."""
    return {
        name
        for name, artifact in topology.artifacts.items()
        if not topology.is_artifact_disabled_on_platform(artifact, "linux")
        and not topology.is_artifact_disabled_on_processor(artifact, "x86_64")
        and name not in LINUX_NON_ARCHIVE_NODES
    }


def check_inventory(filenames: list[str], expected: set[str], family: str) -> list[str]:
    selected = {}
    present = set()
    for filename in sorted(filenames):
        artifact = ArtifactName.from_filename(filename)
        if artifact and artifact.target_family in {"generic", family, "gfx950"}:
            # Match artifact_manager's preference when both encodings exist.
            if artifact not in selected or filename.endswith(".tar.zst"):
                selected[artifact] = filename
            present.add(artifact.name)
    missing = expected - present
    if missing:
        raise RuntimeError(f"Incomplete same-run artifact inventory: {sorted(missing)}")
    return sorted(selected.values())


def check_digest(path: Path, sidecar: Path) -> str:
    values = sidecar.read_text(encoding="utf-8").split()
    if not values or not re.fullmatch(r"[0-9a-fA-F]{64}", values[0]):
        raise RuntimeError(f"Invalid SHA256 sidecar: {sidecar.name}")
    digest = sha256(path)
    if digest != values[0].lower():
        raise RuntimeError(f"Artifact checksum mismatch: {path.name}")
    return digest


def check_portable_links(root: Path) -> int:
    count = 0
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        if path.readlink().is_absolute() or not path.resolve().is_relative_to(
            root.resolve()
        ):
            raise RuntimeError(
                f"Nonportable installation link: {path.relative_to(root)}"
            )
        if not path.exists():
            raise RuntimeError(f"Dangling installation link: {path.relative_to(root)}")
        count += 1
    for alias, target in {"llvm": "lib/llvm", "amdgcn": "lib/llvm/amdgcn"}.items():
        path = root / alias
        if not path.is_symlink() or path.resolve() != (root / target).resolve():
            raise RuntimeError(f"Missing portable SDK alias: {alias} -> {target}")
    return count


def extract_sdk(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Archive validation needs a clean directory: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(destination, filter="tar")


def validate_run_identity(run_id: str, repository: str) -> None:
    if not run_id.isdecimal():
        raise RuntimeError("A numeric current workflow run ID is required")
    if os.environ.get("GITHUB_RUN_ID", run_id) != run_id:
        raise RuntimeError("Packaging must fetch artifacts from its own workflow run")
    if repository != "ROCm/TheRock":
        raise RuntimeError("This DASan stack must use ROCm/TheRock artifacts")


def validate_sdk(rocm: Path, logs: Path, kpack_python: Path) -> dict:
    rocm = rocm.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    for relative in (
        "lib/llvm/bin/clang",
        "bin/hipcc",
        "include/hip/hip_runtime.h",
        "include/hsa/hsa.h",
        "lib/libamdhip64.so",
        "lib/libhsa-runtime64.so",
        "lib/librccl.so",
        "lib/librocblas.so",
        "lib/librocfft.so",
        "lib/librocsolver.so",
        "lib/librocsparse.so",
        "share/therock/therock_manifest.json",
    ):
        path = rocm / relative
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing or empty SDK payload: {relative}")
    report = {"portable_symlinks": check_portable_links(rocm)}
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(rocm / "bin"), str(rocm / "lib/llvm/bin"), env.get("PATH", "")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(rocm / "lib"),
            str(rocm / "lib/llvm/lib"),
            str(rocm / "lib/rocm_sysdeps/lib"),
        ]
    )
    env.update(ROCM_PATH=str(rocm), HIP_PATH=str(rocm), HIP_PLATFORM="amd")
    clang = rocm / "lib/llvm/bin/clang"
    report["clang_version"] = run([clang, "--version"], logs / "clang-version.log", env)
    report["hipcc_version"] = run(
        [rocm / "bin/hipcc", "--version"], logs / "hipcc-version.log", env
    )
    resource = Path(
        run([clang, "-print-resource-dir"], logs / "resource-dir.log", env).strip()
    ).resolve()
    if not resource.is_relative_to(rocm):
        raise RuntimeError("Packaged compiler resource directory escapes the SDK")
    runtimes = list(resource.rglob("libclang_rt.dasan*"))
    triples = {"amdgpu-amd-amdhsa", "amdgcn-amd-amdhsa"}
    host = [p for p in runtimes if not triples.intersection(p.parts)]
    device = [p for p in runtimes if triples.intersection(p.parts)]
    for name, paths in {
        "host static": [p for p in host if p.name == "libclang_rt.dasan.a"],
        "host shared": [p for p in host if p.name == "libclang_rt.dasan.so"],
        "host preinit": [p for p in host if p.name == "libclang_rt.dasan-preinit.a"],
        "device static": [p for p in device if p.name == "libclang_rt.dasan.a"],
    }.items():
        if not paths or not all(p.is_file() and p.stat().st_size for p in paths):
            raise RuntimeError(f"Missing packaged DASan runtime: {name}")
    report["dasan_runtimes"] = [str(p.relative_to(rocm)) for p in runtimes]
    source = logs / "initialize.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    shared = next(p for p in host if p.name == "libclang_rt.dasan.so")
    env["LD_LIBRARY_PATH"] = str(shared.parent) + os.pathsep + env["LD_LIBRARY_PATH"]
    for linkage in ("static", "shared"):
        output = logs / f"initialize-{linkage}"
        flags = (
            ["-shared-libsan", f"-Wl,-rpath,{shared.parent}"]
            if linkage == "shared"
            else []
        )
        run(
            [clang, "-fsanitize=daddress", source, *flags, "-o", output],
            logs / f"initialize-{linkage}-compile.log",
            env,
        )
        text = run(
            [output],
            logs / f"initialize-{linkage}-run.log",
            {**env, "DASAN_OPTIONS": "verbosity=1"},
        )
        if "DeviceAddressSanitizer: initialized" not in text:
            raise RuntimeError(f"Packaged {linkage} DASan runtime did not initialize")
    hip = logs / "smoke.hip"
    hip.write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void write_element(int *p, int i) { p[i] = 1; }\n"
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    run(
        [
            rocm / "bin/hipcc",
            "--offload-arch=gfx950",
            "-fsanitize=daddress",
            "-shared-libsan",
            "-O1",
            hip,
            "-o",
            logs / "hipcc-smoke",
        ],
        logs / "hipcc-compile-link.log",
        env,
    )
    run(
        [
            sys.executable,
            REPO / "test_tools/dasan_stack_cpu.py",
            "--rocm-path",
            rocm,
            "--output-dir",
            logs / "hip-runtime",
        ],
        logs / "hip-runtime.log",
        env,
    )
    ir = logs / "smoke.ll"
    run(
        [
            clang,
            "-xhip",
            "--offload-arch=gfx950",
            "-fsanitize=daddress",
            "-O1",
            "--offload-device-only",
            "-S",
            "-emit-llvm",
            hip,
            "-o",
            ir,
        ],
        logs / "device-instrumentation.log",
        env,
    )
    if "__dasan_report_store" not in ir.read_text(encoding="utf-8"):
        raise RuntimeError("Packaged compiler omitted gfx950 DASan instrumentation")
    run(
        [
            sys.executable,
            REPO / "build_tools/audit_dasan_rccl.py",
            "--rocm-path",
            rocm,
            "--llvm-bin",
            clang.parent,
            "--kpack-python",
            kpack_python,
            "--output-dir",
            logs / "rccl-device",
            "--arch",
            "gfx950",
        ],
        logs / "rccl-device.log",
        env,
    )
    report.update(
        status="passed",
        gpu_execution="separate workflow gate",
        hip_host_runtime_linkage="shared (-shared-libsan)",
    )
    return report


def package(args: argparse.Namespace) -> None:
    validate_run_identity(args.run_id, args.run_github_repo)
    # Lazy imports keep validation helpers usable without storage/compression deps.
    from _therock_utils.artifact_backend import create_backend_from_env

    sys.path.insert(0, str(REPO / "build_tools/packaging/archives"))
    from build_tarballs import (
        DEFAULT_EXCLUDED_ARTIFACTS,
        DEFAULT_EXCLUDED_COMPONENTS,
        compress_tarball,
        fetch_and_flatten,
    )

    if not re.fullmatch(r"[A-Za-z0-9.+_-]+", args.package_version):
        raise RuntimeError("Invalid package version for archive filename")
    output = args.output_dir.resolve()
    deliver = output / "deliverables"
    deliver.mkdir(parents=True, exist_ok=True)
    logs = output / "validation"
    logs.mkdir(exist_ok=True)
    backend = create_backend_from_env(
        run_id=args.run_id, github_repository=args.run_github_repo, platform="linux"
    )
    available = backend.list_artifacts()
    write_json(logs / "available-artifacts.json", available)
    selected = check_inventory(
        available, required_artifacts(get_topology()), args.amdgpu_family
    )
    cache = output / "download-cache"
    archives = []
    archive_metadata = []
    for tests in (True, False):
        prefix = output / ("full-tests-install" if tests else "runtime-install")
        fetch_and_flatten(
            run_id=args.run_id,
            run_github_repo=args.run_github_repo,
            amdgpu_families=[args.amdgpu_family],
            platform="linux",
            output_dir=prefix,
            download_cache_dir=cache,
            extraction_cache_dir=output / "extraction-cache",
            exclude_components=None if tests else DEFAULT_EXCLUDED_COMPONENTS,
            exclude_artifacts=None if tests else DEFAULT_EXCLUDED_ARTIFACTS,
        )
        if tests:
            inventory = []
            for name in selected:
                path = cache / name
                if not path.is_file():
                    raise RuntimeError(
                        f"Expected same-run artifact was not fetched: {name}"
                    )
                sidecar = cache / (name + ".sha256sum")
                backend.download_artifact(name + ".sha256sum", sidecar)
                inventory.append(
                    {
                        "name": name,
                        "bytes": path.stat().st_size,
                        "sha256": check_digest(path, sidecar),
                    }
                )
            write_json(deliver / "artifact-index.json", inventory)
            run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_artifact_structure.py",
                    "-v",
                ],
                logs / "artifact-structure.log",
                {
                    **os.environ,
                    "THEROCK_ARTIFACTS_DIR": str(cache),
                    "PLATFORM": "linux",
                },
            )
        suffix = "-tests" if tests else ""
        archive = deliver / (
            f"therock-dist-linux-{args.amdgpu_family}{suffix}-{args.package_version}.tar.gz"
        )
        compress_tarball(source_dir=prefix, tarball_path=archive)
        archives.append(archive)
        archive_metadata.append(
            {
                "name": archive.name,
                "bytes": archive.stat().st_size,
                "unpacked_bytes": sum(
                    p.stat().st_size
                    for p in prefix.rglob("*")
                    if p.is_file() and not p.is_symlink()
                ),
                "sha256": sha256(archive),
            }
        )
        shutil.rmtree(prefix)
    shutil.rmtree(output / "extraction-cache")
    shutil.rmtree(cache)
    restored = output / "extracted-sdk"
    extract_sdk(archives[0], restored)
    report = validate_sdk(restored, logs, args.kpack_python.resolve())
    write_json(deliver / "usability.json", report)
    shutil.copy2(
        restored / "share/therock/therock_manifest.json",
        deliver / "therock_manifest.json",
    )
    shutil.copy2(REPO / "BUILD_TOPOLOGY.toml", deliver / "BUILD_TOPOLOGY.toml")
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "HEAD"], cwd=REPO, text=True
    )
    manifest = {
        "run_id": args.run_id,
        "repository": args.run_github_repo,
        "run_url": f"https://github.com/{args.run_github_repo}/actions/runs/{args.run_id}",
        "therock_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_gitlinks": [
            line for line in tree.splitlines() if line.startswith("160000 ")
        ],
        "patch_sha256": {
            str(p.relative_to(REPO)): sha256(p)
            for p in sorted((REPO / "patches").rglob("*.patch"))
        },
        "package_version": args.package_version,
        "family": args.amdgpu_family,
        "artifact_name": ARTIFACT_NAME,
        "installation_archive": archives[0].name,
        "runtime_archive": archives[1].name,
        "archives": archive_metadata,
        "archive_transport": "Separate raw single-file GitHub artifacts; metadata is a small zipped artifact",
        "archive_layout": "installation prefix at archive root: bin/, lib/, include/, share/",
        "wsl": "Support artifact uses the release-package preset; no DASan device code",
        "gpu_validation": "Separate GPU workflow uses this archive and the unchanged lit bundle",
        "gpu_runtime_linkage": "Shared host runtime; do not mix a static DASan host runtime with sanitized HIP/ROCr shared libraries",
        "raw_artifacts": str(backend.base_uri),
        "non_archive_topology_nodes": LINUX_NON_ARCHIVE_NODES,
    }
    write_json(deliver / "source-config-manifest.json", manifest)
    (deliver / "README.txt").write_text(
        f"DASan complete gfx950 SDK, run {args.run_id}\n\n"
        f"Download metadata ({ARTIFACT_NAME}) and one archive artifact from {manifest['run_url']}.\n"
        "The -tests- archive is the full SDK including installed test payloads.\n"
        "The other archive omits tests. Choose one; do not overlay them.\n\n"
        "Each .tar.gz is a separate raw artifact, named exactly as the archive.\n"
        "The master SHA256SUMS covers both archives and every metadata file.\n"
        "Download both archives to check the entire master file, or verify the exact chosen archive entry.\n"
        "mkdir rocm-dasan\n"
        f"tar -xzf {archives[0].name} -C rocm-dasan\n"
        'export ROCM_PATH="$PWD/rocm-dasan"\n'
        'export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"\n'
        'export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"\n'
        'DASAN_RUNTIME="$(clang -print-file-name=libclang_rt.dasan.so)"\n'
        'export LD_LIBRARY_PATH="$(dirname "$DASAN_RUNTIME"):$LD_LIBRARY_PATH"\n'
        "hipcc --offload-arch=gfx950 -fsanitize=daddress -shared-libsan app.hip -o app\n\n"
        "Use shared host DASan when linking this sanitized HIP/ROCr SDK.\n"
        "A default static DASan host runtime mixed with these shared libraries is unsupported.\n"
        "Linux x86_64; gfx950 GPU required for device execution.\n"
        "Extract with tar to preserve executable permissions and relative links.\n"
        "GitHub artifact retention: 30 days. Raw same-run components are in S3.\n"
        "Host and RCCL device audit: usability.json. GPU results are separate.\n",
        encoding="utf-8",
    )
    files = sorted(
        p for p in deliver.iterdir() if p.is_file() and p.name != "SHA256SUMS"
    )
    (deliver / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8"
    )
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(
                f"installation_archive={archives[0].name}\nartifact_name={ARTIFACT_NAME}\n"
                f"runtime_archive={archives[1].name}\n"
            )


def selected_download(metadata: Path) -> dict:
    """Check every metadata checksum and select exactly the full-tests archive."""
    manifest = json.loads((metadata / "source-config-manifest.json").read_text())
    validate_run_identity(str(manifest["run_id"]), manifest["repository"])
    entries = {}
    for line in (metadata / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9.+_-]+)", line)
        if not match or match[2] in entries or match[2] in {".", ".."}:
            raise RuntimeError("Invalid or duplicate master checksum entry")
        entries[match[2]] = match[1]
    archives = {item["name"]: item for item in manifest["archives"]}
    if len(archives) != 2 or set(archives) != {
        manifest["installation_archive"],
        manifest["runtime_archive"],
    }:
        raise RuntimeError("Delivery must describe exactly both SDK archives")
    for name, digest in entries.items():
        if name in archives:
            if archives[name]["sha256"] != digest:
                raise RuntimeError(f"Archive manifest/checksum mismatch: {name}")
        elif not (metadata / name).is_file() or sha256(metadata / name) != digest:
            raise RuntimeError(f"Metadata missing or checksum mismatch: {name}")
    if (
        not set(archives).issubset(entries)
        or "source-config-manifest.json" not in entries
    ):
        raise RuntimeError("Missing exact archive or provenance checksum")
    selected = archives[manifest["installation_archive"]]
    if "-tests-" not in selected["name"] or not selected["name"].endswith(".tar.gz"):
        raise RuntimeError("GPU validation must download the full-tests SDK")
    if any(
        not isinstance(selected[key], int) or selected[key] <= 0
        for key in ("bytes", "unpacked_bytes")
    ):
        raise RuntimeError("Missing archive disk-size metadata")
    return {**selected, "run_id": str(manifest["run_id"])}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def github_artifact_request(artifact_id: str, token: str, *, location: bool = False):
    """Send credentials only to api.github.com; never follow its redirect."""
    if not artifact_id.isdecimal():
        raise RuntimeError("Invalid artifact ID")
    suffix = "/zip" if location else ""
    request = urllib.request.Request(
        f"https://api.github.com/repos/ROCm/TheRock/actions/artifacts/{artifact_id}{suffix}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(
            request, timeout=30
        ) as response:
            if location:
                raise RuntimeError("Artifact API did not return a download redirect")
            return json.load(response)
    except urllib.error.HTTPError as error:
        if location and error.code == 302:
            url = error.headers["Location"]
            if not url.startswith("https://"):
                raise RuntimeError("Artifact redirect must use HTTPS") from None
            return url
        raise RuntimeError(f"GitHub artifact API returned HTTP {error.code}") from None


def download_raw_archive(
    selected: dict,
    directory: Path,
    artifact_id: str,
    token: str,
    timeout: int = 9000,
    margin: int = 15 * 1024**3,
) -> Path:
    artifact = github_artifact_request(artifact_id, token)
    if (
        artifact["name"] != selected["name"]
        or artifact["size_in_bytes"] != selected["bytes"]
        or str(artifact["workflow_run"]["id"]) != selected["run_id"]
        or artifact.get("expired")
    ):
        raise RuntimeError(
            "Raw artifact identity/size does not match this run's selected archive"
        )
    target = directory / selected["name"]
    partial = target.with_name(target.name + ".part")
    state = partial.with_name(partial.name + ".json")
    identity = {
        "artifact_id": artifact_id,
        "sha256": selected["sha256"],
        "bytes": selected["bytes"],
    }
    if partial.exists() and (
        not state.exists() or json.loads(state.read_text()) != identity
    ):
        raise RuntimeError("Partial download belongs to a different artifact")
    write_json(state, identity)
    deadline = time.monotonic() + timeout
    last_progress = time.monotonic()
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > selected["bytes"]:
            raise RuntimeError("Partial download exceeds expected archive size")
        needed = selected["bytes"] - offset + selected["unpacked_bytes"] + margin
        if shutil.disk_usage(directory).free < needed:
            raise RuntimeError(
                f"Insufficient disk: need {needed} bytes for archive, extraction and scratch"
            )
        if offset == selected["bytes"]:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("SDK download exceeded its transfer time budget")
        print(
            f"SDK transfer attempt {attempt}: {offset}/{selected['bytes']} bytes",
            flush=True,
        )
        try:
            # The signed URL stays in memory, with no Authorization on blob requests.
            url = github_artifact_request(artifact_id, token, location=True)
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(
                request, timeout=min(30, remaining)
            ) as response:
                if response.status == 206:
                    expected = (
                        f"bytes {offset}-{selected['bytes'] - 1}/{selected['bytes']}"
                    )
                    if response.headers.get("Content-Range") != expected:
                        raise RuntimeError(
                            "Blob returned an inconsistent Content-Range"
                        )
                elif response.status != 200 or offset:
                    raise RuntimeError(
                        "Blob rejected the required resumable byte range"
                    )
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != selected["bytes"] - offset:
                    raise RuntimeError(
                        "Blob length does not match expected raw archive"
                    )
                with partial.open("ab") as stream:
                    while offset < selected["bytes"]:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "SDK download exceeded its transfer time budget"
                            )
                        # A trickling response must not hide the deadline.
                        read = getattr(response, "read1", response.read)
                        chunk = read(min(1024**2, selected["bytes"] - offset))
                        if not chunk:
                            raise OSError("Artifact stream ended before expected size")
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_progress >= 30:
                            print(
                                f"SDK transfer: {offset}/{selected['bytes']} bytes ({offset / selected['bytes']:.1%})",
                                flush=True,
                            )
                            last_progress = time.monotonic()
            break
        except RuntimeError:
            raise
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            print(
                f"SDK transfer interrupted ({type(error).__name__}); retaining verified-identity partial file",
                flush=True,
            )
            if attempt == 8:
                raise RuntimeError("SDK transfer exhausted eight attempts") from None
            time.sleep(min(5 * attempt, 30, max(0, deadline - time.monotonic())))
    if (
        not partial.is_file()
        or partial.stat().st_size != selected["bytes"]
        or sha256(partial) != selected["sha256"]
    ):
        raise RuntimeError("Downloaded archive size or SHA256 mismatch")
    partial.replace(target)
    state.unlink()
    (directory / "SELECTED_SHA256SUMS").write_text(
        f"{selected['sha256']}  {selected['name']}\n"
    )
    print(
        f"SDK transfer complete: {selected['bytes']} bytes; SHA256 verified", flush=True
    )
    return target


def download_main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only this run's raw full-tests SDK with progress and resume"
    )
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=9000)
    args = parser.parse_args(sys.argv[2:])
    selected = selected_download(args.metadata_dir)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for artifact download")
    download_raw_archive(
        selected, args.metadata_dir, args.artifact_id, token, args.timeout_seconds
    )


def main() -> None:
    if sys.argv[1:2] == ["download"]:
        download_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-github-repo", default="ROCm/TheRock")
    parser.add_argument(
        "--amdgpu-family", choices=["gfx950-dcgpu"], default="gfx950-dcgpu"
    )
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--kpack-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    status = {"status": "running", "run_id": args.run_id}
    try:
        package(args)
        status["status"] = "passed"
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output_dir / "validation/status.json", status)


if __name__ == "__main__":
    main()
