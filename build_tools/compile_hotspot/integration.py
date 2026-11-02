# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the real CK archive and MIOpen consumer; emit a minimal GPU test bundle."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

import profile as diagnostic
from benchmark import arguments, common_flags, entries, reconfigure_ck

HERE = Path(__file__).resolve().parent
ROOT, BUILD, OUT, CK = diagnostic.ROOT, diagnostic.BUILD, diagnostic.OUT, diagnostic.CK
LIBRARIES = ROOT / "rocm-libraries"
CK_SOURCE = LIBRARIES / "projects/composablekernel"
MIOPEN = BUILD / "ml-libs/MIOpen/build"
STAGE = BUILD / "math-libs/composable_kernel/stage"
CACHE = ROOT / "integration-download-cache"
BUNDLE = ROOT / "integration-bundle"
LIBRARIES_SHA = "f4e24d58b3f313f1a31068def1e35376e43c2f49"
BASELINES = {"release": "41149779244", "asan-debug": "41211496307"}
REBUILT = {"composable-kernel", "miopen"}
# Native MIOpen helpers and the prebuilt rocprofiler-sdk's runtime closure are
# outside the pinned math-stage artifact payloads. nlohmann-json's optional
# artifact is not inbound; the others have no artifact provider at this pin.
# Their source URLs/hashes remain pinned by TheRock.
HELPER_TARGETS = {"therock-googletest": "googletest", "therock-eigen": "eigen",
                  "therock-frugally-deep": "frugally-deep",
                  "therock-FunctionalPlus": "FunctionalPlus", "therock-nlohmann-json": "nlohmann-json",
                  "therock-fmt": "fmt", "therock-yaml-cpp": "yaml-cpp"}
LOGICAL_ONLY_ARTIFACTS = {
    "core-hipinfo": "Pinned core/CMakeLists.txt publishes this only under WIN32",
    "spdlog": "Pinned third-party/spdlog/CMakeLists.txt has no artifact provider; not a CK/MIOpen dependency",
    "sysdeps-libmnl": "Published within sysdeps dev/lib; verify its actual stage root",
    "sysdeps-libnl": "Published within sysdeps dev/lib; verify its actual stage root",
}
BUNDLED_SYSDEPS_ROOTS = {"third-party/sysdeps/linux/libmnl/build/stage",
                         "third-party/sysdeps/linux/libnl/build/stage"}
SOURCE_DIR = "projects/composablekernel/library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/nhwgc/"
HELPER_SUFFIXES = ("kernel_group1", "nongrouped_match")
PRIVATE_HEADER = diagnostic.SOURCE.replace("_instance.cpp", "_kernel_groups.hpp")
NAMES = {diagnostic.SOURCE, *(diagnostic.SOURCE.replace("_instance.cpp", "_" + suffix + "_instance.cpp")
                             for suffix in HELPER_SUFFIXES)}


def save(name, value):
    diagnostic.save("integration-" + name, value)


def run(argv, label, cwd=ROOT):
    save(label + "-command.json", {"argv": argv, "cwd": str(cwd)})
    diagnostic.run(argv, cwd=cwd, log="integration-" + label + ".log")


def measured(argv, label, cwd=ROOT):
    diagnostic.measure(argv, cwd, "integration-" + label)
    result = json.loads((OUT / ("integration-" + label + "-measurement.json")).read_text())
    result.pop("samples", None)
    report = (OUT / ("integration-" + label + "-time.txt")).read_text()
    for key, text in [("user_cpu_s", "User time (seconds)"), ("system_cpu_s", "System time (seconds)"),
                      ("maximum_process_rss_kib", "Maximum resident set size (kbytes)")]:
        result[key] = float(re.search(re.escape(text) + r":\s*([\d.]+)", report)[1])
    return result


def cache_value(build, key):
    match = re.search(r"^" + re.escape(key) + r":[^=]*=(.*)$", (build / "CMakeCache.txt").read_text(), re.M)
    if not match:
        raise RuntimeError(f"Missing CMake cache value {key}")
    return match[1]


def patch_paths(text):
    paths = re.findall(r"^\+\+\+ b/(.*)$", text, re.M)
    expected = {SOURCE_DIR + "CMakeLists.txt", SOURCE_DIR + "xdl/" + PRIVATE_HEADER,
                *(SOURCE_DIR + "xdl/" + name for name in NAMES)}
    if len(paths) != len(expected) or set(paths) != expected:
        raise RuntimeError(f"Unexpected product patch paths: {paths}")
    return paths


def check_markers(roots):
    from _therock_utils.artifacts import prebuilt_marker_relpath
    for root in roots:
        if not (BUILD / prebuilt_marker_relpath(root)).is_file():
            raise RuntimeError(f"Missing imported prebuilt marker: {root}")
    markers = [str(p.relative_to(BUILD)) for p in BUILD.rglob("*.prebuilt")]
    if any("composable_kernel" in p or "/MIOpen/" in p for p in markers):
        raise RuntimeError("Changed component unexpectedly imported as prebuilt")
    save("prebuilt-markers.json", markers)


def fetch_dependencies(args):
    sys.path.insert(0, str(ROOT / "build_tools"))
    from _therock_utils.build_topology import BuildTopology
    from _therock_utils.artifacts import ArtifactName
    from _therock_utils.archive_util import open_archive_for_read
    from _therock_utils.artifact_backend import create_backend_from_env
    topology = BuildTopology(str(ROOT / "BUILD_TOPOLOGY.toml"))
    logical = (topology.get_inbound_artifacts("math-libs") | topology.get_produced_artifacts("math-libs")) - REBUILT
    if len(logical) != 37 or not set(LOGICAL_ONLY_ARTIFACTS).issubset(logical):
        raise RuntimeError("Pinned logical dependency closure changed")
    imported = logical - set(LOGICAL_ONLY_ARTIFACTS)
    excluded = set(topology.artifacts) - imported
    if len(imported) != 33 or not REBUILT.issubset(excluded):
        raise RuntimeError("Pinned dependency closure changed")
    backend = create_backend_from_env(run_id=args.baseline_run_id,
                                      github_repository="ROCm/TheRock", platform="linux")
    published = backend.list_artifacts()
    available = {entry.name for filename in published
                 if (entry := ArtifactName.from_filename(filename)) is not None
                 and entry.target_family in {"generic", "gfx950"}
                 and entry.component not in {"dbg", "doc"}}
    save("dependency-plan.json", {"baseline_uri": backend.base_uri, "published_archives": published,
                                 "requested_logical_artifacts": sorted(logical),
                                 "imported_artifacts": sorted(imported),
                                 "logical_only_artifacts": LOGICAL_ONLY_ARTIFACTS})
    if imported - available:
        raise RuntimeError(f"Required published artifacts absent before fetch: {sorted(imported - available)}")
    if set(LOGICAL_ONLY_ARTIFACTS) & available:
        raise RuntimeError("Pinned logical-only artifact assumptions changed; reinspect published metadata")
    run([sys.executable, "build_tools/artifact_manager.py", "fetch", "--stage=all",
         "--run-id=" + args.baseline_run_id, "--run-github-repo=ROCm/TheRock", "--platform=linux",
         "--amdgpu-families=gfx950", "--amdgpu-targets=gfx950", "--bootstrap",
         "--exclude-artifacts=" + ",".join(sorted(excluded)), "--exclude-components=dbg,doc",
         "--output-dir=" + str(BUILD), "--download-cache-dir=" + str(CACHE)], "fetch-dependencies")
    archives, found, roots = [], set(), set()
    for path in sorted(CACHE.rglob("*.tar.*")):
        name = ArtifactName.from_filename(path.name)
        if name is None or name.name not in imported:
            raise RuntimeError(f"Unexpected dependency archive: {path}")
        with open_archive_for_read(path) as archive:
            first = next(iter(archive))
            if first.name != "artifact_manifest.txt":
                raise RuntimeError(f"Unexpected archive manifest: {first.name}")
            manifest = archive.extractfile(first).read().decode().splitlines()
        for root in filter(None, manifest):
            if "composable_kernel" in root or "/MIOpen/" in root:
                raise RuntimeError(f"Changed component in dependency payload: {root}")
            roots.add(root)
        found.add(name.name)
        archives.append({**diagnostic.fingerprint(path), "artifact": name.name, "roots": manifest})
    if found != imported:
        raise RuntimeError(f"Missing imported artifacts: {sorted(imported - found)}")
    if not BUNDLED_SYSDEPS_ROOTS.issubset(roots):
        raise RuntimeError("Published sysdeps payload omitted required bundled libmnl/libnl roots")
    check_markers(roots)
    save("dependencies.json", {"imported_artifacts": sorted(imported), "archives": archives})
    return sorted(imported), archives, roots


def check_planned_builds():
    commands = diagnostic.run(["ninja", "-C", str(BUILD), "-t", "commands", "composable_kernel", "MIOpen+configure"])
    (OUT / "integration-planned-commands.txt").write_text(commands)
    unexpected = []
    allowed = {(BUILD / "math-libs/composable_kernel/build").resolve(), MIOPEN.resolve()}
    allowed.update((BUILD / "third-party" / name / "build").resolve() for name in HELPER_TARGETS.values())
    for command in commands.splitlines():
        tokens = shlex.split(command)
        for index, token in enumerate(tokens[:-1]):
            if token in {"--build", "--install", "-B"}:
                path = Path(tokens[index + 1])
                path = (path if path.is_absolute() else BUILD / path).resolve()
                if path not in allowed:
                    unexpected.append({"command": command, "path": str(path)})
    save("unexpected-builds.json", unexpected)
    if unexpected:
        raise RuntimeError("Imported dependency would be rebuilt; see planned commands")


def configure(args):
    OUT.mkdir(exist_ok=True)
    if diagnostic.run(["git", "rev-parse", "HEAD"]).strip() != diagnostic.SOURCE_SHA:
        raise RuntimeError("Product source does not match pinned revision")
    if args.baseline_run_id != BASELINES[args.variant]:
        raise RuntimeError("Variant does not match pinned dependency baseline")
    for directory in (BUILD, CACHE, BUNDLE):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Expected fresh output directory: {directory}")
    context = {"source_sha": diagnostic.SOURCE_SHA, "instrumentation_sha": os.getenv("GITHUB_SHA"),
               "producer_run_id": os.getenv("GITHUB_RUN_ID"), "baseline_run_id": args.baseline_run_id,
               "variant": args.variant, "runner_name": os.getenv("RUNNER_NAME"), "platform": platform.platform(),
               "logical_cpus": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
               "cache_contract": "CCACHE_DISABLE=1; fresh build outputs; filesystem caches uncontrolled"}
    save("context.json", context)
    imported, archives, roots = fetch_dependencies(args)
    run([sys.executable, "build_tools/fetch_sources.py", "--stage", "math-libs", "--jobs", "12", "--depth", "1"], "fetch-sources")
    if diagnostic.run(["git", "rev-parse", "HEAD"], cwd=LIBRARIES).strip() != LIBRARIES_SHA:
        raise RuntimeError("Unexpected rocm-libraries source revision")
    if diagnostic.run(["git", "status", "--porcelain"], cwd=LIBRARIES).strip():
        raise RuntimeError("Expected pristine rocm-libraries sources")
    arg_file = OUT / "integration-configure-args.txt"
    run([sys.executable, "build_tools/configure_stage.py", "--stage=math-libs", "--amdgpu-families=gfx950",
         "--dist-amdgpu-families=gfx950", "--manylinux", "--output-cmake-args=" + str(arg_file)], "stage-configuration")
    preset = ["--preset=linux-release-asan-debug"] if args.variant == "asan-debug" else []
    run(["cmake", "-S", str(ROOT), "-B", str(BUILD), "-GNinja", *preset,
         *diagnostic.stage_arguments(arg_file.read_text()), "-DTHEROCK_BACKGROUND_BUILD_JOBS=9",
         "-DLLVM_PARALLEL_LINK_JOBS=4", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
         "-Dcomposable_kernel_CMAKE_ARGS=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
         "-DMIOpen_CMAKE_ARGS=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], "configure")
    check_markers(roots)
    check_planned_builds()
    # CK+configure stages rocprofiler-sdk's runtime dependencies even when that
    # SDK is prebuilt. Measure these helpers before the configure edge can build
    # them implicitly, keeping their setup out of the affected-target timing.
    helper_setup = measured(["cmake", "--build", str(BUILD), "--target", *HELPER_TARGETS], "helper-setup")
    run(["cmake", "--build", str(BUILD), "--target", "composable_kernel+configure"], "ck-configure")
    baseline = entries({diagnostic.SOURCE})[diagnostic.SOURCE]
    generated_headers = {str(p.relative_to(CK)): diagnostic.fingerprint(p)["sha256"]
                         for p in (CK / "include").rglob("*") if p.is_file()}
    context["compiler_sha256"] = diagnostic.fingerprint(Path(arguments(baseline["command"])[0]))["sha256"]
    context["harness_source_sha256"] = diagnostic.fingerprint(HERE / "ck_f32_contract.cpp")["sha256"]
    context["rocm_libraries_sha"] = LIBRARIES_SHA
    context["unimported_build_helpers"] = {
        target: {"build_directory": str(BUILD / "third-party" / name / "build"),
                 "source_declaration": diagnostic.fingerprint(ROOT / "third-party" / name / "CMakeLists.txt")}
        for target, name in HELPER_TARGETS.items()}
    patch = HERE / "candidate.patch"
    paths = patch_paths(patch.read_text())
    run(["git", "apply", "--check", str(patch)], "patch-check", LIBRARIES)
    run(["git", "apply", str(patch)], "patch", LIBRARIES)
    reconfigure_ck()
    candidate = entries(NAMES)
    candidate_headers = {str(p.relative_to(CK)): diagnostic.fingerprint(p)["sha256"]
                         for p in (CK / "include").rglob("*") if p.is_file()}
    if generated_headers != candidate_headers:
        raise RuntimeError("Candidate regeneration changed generated CK headers")
    for name, entry in candidate.items():
        if common_flags(arguments(entry["command"])) != common_flags(arguments(baseline["command"])):
            raise RuntimeError(f"Compiler settings changed for {name}")
    save("source-contract.json", {"baseline": baseline, "candidate": candidate,
                                 "generated_headers": generated_headers,
                                 "source_files": [diagnostic.fingerprint(LIBRARIES / p) for p in paths]})
    return context, imported, archives, candidate, helper_setup


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target = os.readlink(source)
        if Path(target).is_absolute():
            raise RuntimeError(f"Absolute output symlink: {source}")
        destination.symlink_to(target)
    else:
        shutil.copy2(source, destination)


def archive_proof(archive, candidate):
    ar, nm = cache_value(CK, "CMAKE_AR"), cache_value(CK, "CMAKE_NM")
    members = diagnostic.run([ar, "t", str(archive)]).splitlines()
    native_archive = CK / "lib/libdevice_conv_operations.a"
    native_members = diagnostic.run([ar, "t", str(native_archive)]).splitlines()
    obsolete = diagnostic.SOURCE.replace("_instance.cpp", "_noshuffle_instance.cpp") + ".o"
    if obsolete in members or obsolete in native_members:
        raise RuntimeError("Old three-TU noshuffle member survived in candidate archive")
    proof_dir = OUT / "integration-archive-proof"
    proof_dir.mkdir()
    proof = []
    for name, entry in candidate.items():
        path = Path(entry["output"])
        if not path.is_absolute():
            path = Path(entry["directory"]) / path
        member = path.name
        if members.count(member) != 1 or native_members.count(member) != 1:
            raise RuntimeError(f"Expected unique archive member {member}")
        extracted = proof_dir / member
        with extracted.open("wb") as stream:
            subprocess.run([ar, "p", str(archive), member], stdout=stream, check=True)
        actual, archived = diagnostic.fingerprint(path), diagnostic.fingerprint(extracted)
        native_object = proof_dir / (member + ".native")
        with native_object.open("wb") as stream:
            subprocess.run([ar, "p", str(native_archive), member], stdout=stream, check=True)
        native = diagnostic.fingerprint(native_object)
        if actual["sha256"] != native["sha256"]:
            raise RuntimeError(f"Native archive member differs from compiled object {name}")
        symbols = diagnostic.run([nm, "--demangle", str(extracted)])
        (proof_dir / (member + ".symbols.txt")).write_text(symbols)
        proof.append({"source": name, "object": actual, "native_archive_member": native,
                      "staged_archive_member": archived,
                      "install_changed_bytes": actual["sha256"] != archived["sha256"]})
    public = (proof_dir / (diagnostic.SOURCE + ".o.symbols.txt")).read_text()
    for suffix in HELPER_SUFFIXES:
        symbol = "detail::add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_" + suffix + "_instances"
        if not any(" U " in line and symbol in line for line in public.splitlines()):
            raise RuntimeError(f"Public archive member does not reference {symbol}")
        helper = diagnostic.SOURCE.replace("_instance.cpp", "_" + suffix + "_instance.cpp")
        symbols = (proof_dir / (helper + ".o.symbols.txt")).read_text()
        if not any(re.search(r"\s[TW]\s", line) and symbol in line for line in symbols.splitlines()):
            raise RuntimeError(f"Helper archive member does not define {symbol}")
    result = {"archive": diagnostic.fingerprint(archive), "native_archive": diagnostic.fingerprint(native_archive),
              "members": members, "changed_members": proof,
              "install_note": "The normal stage install may strip debug information; byte identity is required between compiled objects and native archive members, then helper reachability is checked in the installed archive."}
    save("archive-proof.json", result)
    return result


def link_contract(args, candidate, archive, utility):
    flags = [arg for arg in common_flags(arguments(candidate[diagnostic.SOURCE]["command"])) if arg != "-MD"]
    output = OUT / "integration-contract.o"
    measured([*flags, "-c", str(HERE / "ck_f32_contract.cpp"), "-o", str(output)], "contract-compile", CK)
    link_flags, index = [], 0
    while index < len(flags):
        if flags[index] == "-x":
            index += 2
        else:
            link_flags.append(flags[index])
            index += 1
    binary = BUNDLE / "contract/ck_f32_contract"
    binary.parent.mkdir(parents=True, exist_ok=True)
    argv = [*link_flags, "--hip-link", str(output), str(archive), str(utility),
            "-L" + str(BUILD / "core/clr/dist/lib"), "-lamdhip64",
            "-Wl,-rpath,$ORIGIN/../sdk/lib", "-o", str(binary)]
    if args.variant == "asan-debug":
        argv.insert(1, "-shared-libsan")
    measured(argv, "contract-link", CK)
    command = {"argv": argv, "cwd": str(CK), "archive_sha256": diagnostic.fingerprint(archive)["sha256"]}
    (BUNDLE / "contract-command.json").write_text(json.dumps(command, indent=2) + "\n")
    (OUT / "integration-contract-dynamic.txt").write_text(diagnostic.run(["readelf", "-d", str(binary)]))
    return {"binary": str(binary.relative_to(BUNDLE)), "binary_sha256": diagnostic.fingerprint(binary)["sha256"],
            "archive_sha256": command["archive_sha256"], "link_command_file": "contract-command.json"}


def package(args, context, imported, archives, candidate, measurements):
    archive = STAGE / "lib/libdevice_conv_operations.a"
    proof = archive_proof(archive, candidate)
    consumer_archive = BUILD / "math-libs/composable_kernel/dist/lib/libdevice_conv_operations.a"
    if diagnostic.fingerprint(consumer_archive)["sha256"] != diagnostic.fingerprint(archive)["sha256"]:
        raise RuntimeError("CK dist archive consumed by MIOpen differs from the staged archive")
    utilities = list(CK.rglob("libutility.a"))
    if len(utilities) != 1:
        raise RuntimeError(f"Expected one CK utility archive, got {utilities}")
    utility = utilities[0]
    copy_file(archive, BUNDLE / "ck/lib/libdevice_conv_operations.a")
    copy_file(utility, BUNDLE / "ck/lib/libutility.a")
    shutil.copytree(STAGE / "include/ck", BUNDLE / "ck/include/ck", symlinks=True)
    libdir = MIOPEN / cache_value(MIOPEN, "CMAKE_INSTALL_LIBDIR")
    plugin = libdir / "libMIOpenCKGroupedConv_gfx950.so"
    miopen = libdir / "libMIOpen.so"
    if not plugin.is_file() or not miopen.is_file():
        raise RuntimeError("Native MIOpen target did not produce expected libraries")
    for path in libdir.glob("*.so*"):
        if path.is_file() or path.is_symlink():
            copy_file(path, BUNDLE / "sdk/lib" / path.name)
    commands = diagnostic.run(["ninja", "-t", "commands", "MIOpenCKGroupedConv_gfx950"], cwd=MIOPEN)
    links = [line for line in commands.splitlines() if "-shared" in line and plugin.name in line]
    if len(links) != 1:
        raise RuntimeError(f"Expected one plugin link command, got {len(links)}")
    link = links[0]
    responses = {}
    for token in shlex.split(link):
        if token.startswith("@"):
            path = Path(token[1:])
            path = path if path.is_absolute() else MIOPEN / path
            responses[str(path)] = path.read_text()
    expanded = link + "\n" + "\n".join(responses.values())
    if str(consumer_archive) not in expanded:
        raise RuntimeError("Consumer link command does not name the actual rebuilt CK archive")
    save("plugin-link.json", {"command": link, "response_files": responses,
                              "consumer_archive": str(consumer_archive),
                              "archive_sha256": diagnostic.fingerprint(archive)["sha256"]})
    (OUT / "integration-plugin-dynamic.txt").write_text(diagnostic.run(["readelf", "-d", str(plugin)]))
    contract = link_contract(args, candidate, archive, utility)
    files, symlinks = {}, {}
    for path in sorted(BUNDLE.rglob("*")):
        relative = str(path.relative_to(BUNDLE))
        if path.is_symlink():
            if not path.resolve().is_relative_to(BUNDLE.resolve()) or not path.exists():
                raise RuntimeError(f"Uncontained or broken bundle symlink: {path}")
            symlinks[relative] = os.readlink(path)
        elif path.is_file():
            files[relative] = diagnostic.fingerprint(path)["sha256"]
    manifest = {**context, "libraries_sha": LIBRARIES_SHA, "imported_artifacts": imported,
                "requested_logical_artifacts": sorted(set(imported) | set(LOGICAL_ONLY_ARTIFACTS)),
                "logical_only_artifacts": LOGICAL_ONLY_ARTIFACTS,
                "dependency_archives": archives, "files": files, "symlinks": symlinks,
                "plugin_path": "lib/" + plugin.name, "miopen_library_path": "lib/" + miopen.name,
                "ck_archive_path": "ck/lib/libdevice_conv_operations.a", "contract": contract,
                "patch_sha256": diagnostic.fingerprint(HERE / "candidate.patch")["sha256"],
                "measurements": measurements, "archive_proof": proof,
                "status": "producer complete; numerical and consumer tests pending",
                "test_layout": "Consumer imports untouched dependencies and baseline MIOpen run/test/data; overlay only sdk/lib from this bundle"}
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save("bundle-manifest-fingerprint.json", diagnostic.fingerprint(BUNDLE / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=BASELINES, default="release")
    parser.add_argument("--baseline-run-id", required=True)
    args = parser.parse_args()
    if os.getenv("GITHUB_ACTIONS") != "true":
        raise RuntimeError("This producer requires the pinned CI container")
    os.environ["CCACHE_DISABLE"] = "1"
    context, imported, archives, candidate, helper_setup = configure(args)
    measurements = {"unimported_helper_setup": helper_setup}
    measurements["ck_archive_and_stage"] = measured(["cmake", "--build", str(BUILD), "--target", "composable_kernel"], "ck-build")
    measurements["miopen_configure"] = measured(["cmake", "--build", str(BUILD), "--target", "MIOpen+configure"], "miopen-configure")
    measurements["miopen_native_consumer"] = measured(["cmake", "--build", str(MIOPEN), "--target",
            "MIOpenCKGroupedConv_gfx950", "MIOpen", "--", "-d", "keeprsp"], "miopen-build")
    package(args, context, imported, archives, candidate, measurements)


if __name__ == "__main__":
    main()
