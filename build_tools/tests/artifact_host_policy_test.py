# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host artifact policy propagation through topology, CMake, and flattening."""

import io
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from _therock_utils.build_topology import BuildTopology
from topology_to_cmake import generate_validation_metadata

ROOT = Path(__file__).resolve().parents[2]


def test_topology_host_policy():
    topology = BuildTopology(str(ROOT / "BUILD_TOPOLOGY.toml"))
    artifact = topology.artifacts["miopen"]
    assert artifact.target_specific_host_components == ["test"]
    out = io.StringIO()
    generate_validation_metadata(topology, out)
    assert (
        'set(THEROCK_ARTIFACT_TARGET_SPECIFIC_HOST_COMPONENTS_miopen "test")'
        in out.getvalue()
    )
    assert not topology.validate_topology()
    artifact.target_specific_host_components = ["typo"]
    assert any(
        "target_specific_host_components" in e for e in topology.validate_topology()
    )
    artifact.target_specific_host_components = ["test"]
    artifact.type = "target-neutral"
    assert any("requires target-specific" in e for e in topology.validate_topology())


@pytest.mark.parametrize(
    "specific,enabled", [(False, False), (True, False), (True, True)]
)
def test_cmake_tracks_and_flattens_host_manifest(tmp_path, specific, enabled):
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "build"
    payload = build / "payload/stage/bin/test_gemm"
    payload.parent.mkdir(parents=True)
    payload.write_text("current host")
    descriptor = source / "artifact.toml"
    descriptor.write_text('[components.test."payload/stage"]\ninclude = ["bin/**"]\n')
    tools = source / "systems/shared/kpack/python/rocm_kpack"
    (tools / "tools").mkdir(parents=True)
    (tools / "database_handlers.py").touch()
    (tools / "artifact_splitter.py").touch()
    # Exercise the generated command without requiring a rocm-systems checkout.
    (tools / "tools/split_artifacts.py").write_text(
        textwrap.dedent(
            """\
        import argparse
        from pathlib import Path
        import shutil
        p = argparse.ArgumentParser()
        p.add_argument("--artifact-dir", type=Path)
        p.add_argument("--output-dir", type=Path)
        p.add_argument("--artifact-prefix")
        p.add_argument("--host-artifact-target", default="generic")
        p.add_argument("--clang-offload-bundler")
        p.add_argument("--gpu-targets", nargs="+")
        a = p.parse_args()
        shutil.copytree(a.artifact_dir, a.output_dir / f"{a.artifact_prefix}_{a.host_artifact_target}", dirs_exist_ok=True)
        """
        )
    )
    stale = build / "artifacts/miopen_test_generic"
    if specific and enabled:
        (stale / "payload/stage/bin").mkdir(parents=True)
        (stale / "artifact_manifest.txt").write_text("payload/stage\n")
        (stale / "payload/stage/bin/test_gemm").write_text("obsolete host")
        for suffix in [
            ".tar.zst",
            ".tar.zst.sha256sum",
            ".tar.xz",
            ".tar.xz.sha256sum",
        ]:
            Path(str(stale) + suffix).write_text("obsolete archive")
    (source / "CMakeLists.txt").write_text(
        textwrap.dedent(
            f"""\
        cmake_minimum_required(VERSION 3.25)
        project(ArtifactHostPolicy NONE)
        set(THEROCK_SOURCE_DIR "{ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${{CMAKE_SOURCE_DIR}}/systems")
        set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
        set(THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS ON)
        set(THEROCK_FLAG_KPACK_TARGET_SPECIFIC_HOSTS {'ON' if enabled else 'OFF'})
        set(THEROCK_ARTIFACT_TYPE_miopen target-specific)
        set(THEROCK_ARTIFACT_TARGET_SPECIFIC_HOST_COMPONENTS_miopen "{'test' if specific else ''}")
        set(THEROCK_AMDGPU_DIST_BUNDLE_NAME gfx94X-dcgpu)
        set(THEROCK_AMDGPU_TARGETS gfx942)
        foreach(name therock-dist therock-artifacts therock-expunge)
          add_custom_target(${{name}})
        endforeach()
        function(_therock_cmake_subproject_deps_to_stamp output)
          set(${{output}} "" PARENT_SCOPE)
        endfunction()
        include("{ROOT.as_posix()}/cmake/therock_artifacts.cmake")
        therock_provide_artifact(miopen COMPONENTS test DESCRIPTOR artifact.toml DISTRIBUTION rocm)
        """
        )
    )

    def run(*args):
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout

    run("cmake", "-S", str(source), "-B", str(build), "-G", "Ninja")
    run("cmake", "--build", str(build), "--target", "artifact-miopen")
    host_target = "gfx94X-dcgpu" if specific and enabled else "generic"
    manifest = build / f"artifacts/miopen_test_{host_target}/artifact_manifest.txt"
    assert manifest.read_text().strip() == "payload/stage"
    assert (build / "dist/rocm/bin/test_gemm").read_text() == "current host"
    if specific and enabled:
        assert not list(stale.parent.glob("miopen_test_generic*"))
    assert "no work to do" in run(
        "cmake", "--build", str(build), "--target", "artifact-miopen"
    )
