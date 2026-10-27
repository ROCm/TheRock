# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configure_stage import get_topology
from dasan_component_build import (
    STAGES,
    assembly_device_fixture,
    audit_gpu_targets,
    audit_instrumentation,
    has_daddress,
    import_dependencies,
    select_artifacts,
    shell_commands,
    source_sets,
    unsupported_device_fixture,
)


class SelectionTest(unittest.TestCase):
    def test_scopes_do_not_enable_unrelated_math_or_compiler_sources(self):
        topology = get_topology()
        for stage in STAGES:
            with self.subTest(stage=stage):
                primary, closure = select_artifacts(topology, stage)
                self.assertTrue(primary)
                self.assertLessEqual(primary, closure)
                self.assertNotIn("compilers", source_sets(topology, closure))
                self.assertFalse(
                    {"blas", "solver", "miopen", "miopenprovider"} & closure
                )
        primary, closure = select_artifacts(topology, "hipdnn")
        self.assertEqual(primary, {"hipdnn"})
        self.assertIn("spdlog", closure)
        _, closure = select_artifacts(topology, "comm-libs")
        self.assertTrue({"rocjitsu", "hipify", "rocprofiler-sdk"} <= closure)
        _, closure = select_artifacts(topology, "profiler-apps")
        self.assertNotIn("rocjpeg", closure)


class ImportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()

    def artifact(self, name, relative, content="payload"):
        artifact = self.artifacts / name
        (artifact / relative).mkdir(parents=True)
        (artifact / "artifact_manifest.txt").write_text(relative + "\n")
        (artifact / relative / "library").write_text(content)

    def test_only_imports_dependencies_and_resume_checks_identity(self):
        self.artifact("sysdeps_lib_generic", "third-party/zlib/stage")
        self.artifact("sysdeps_dev_generic", "third-party/zlib/stage", "headers")
        self.artifact("rccl_lib_gfx950", "comm-libs/rccl/stage")
        self.artifact("core-hip_lib_generic", "core/clr/stage")
        self.artifact("sysdeps_lib_gfx942", "third-party/wrong/stage")
        primary, closure = {"rccl"}, {"rccl", "sysdeps", "core-hip"}
        result = import_dependencies(self.artifacts, self.output, primary, closure)
        self.assertEqual(len(result["selected_artifacts"]), 2)
        self.assertTrue(
            (self.output / "build/third-party/zlib/stage.prebuilt").is_file()
        )
        self.assertFalse((self.output / "build/comm-libs").exists())
        self.assertFalse((self.output / "build/core").exists())
        self.assertEqual(
            import_dependencies(self.artifacts, self.output, primary, closure), result
        )
        with patch.dict("os.environ", {"DEPENDENCIES_RUN_ID": "different"}):
            with self.assertRaisesRegex(RuntimeError, "selection changed"):
                import_dependencies(self.artifacts, self.output, primary, closure)

    def test_rejects_existing_stage_and_protected_manifest(self):
        self.artifact("sysdeps_lib_generic", "third-party/zlib/stage")
        (self.output / "build/third-party/zlib/stage").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            import_dependencies(self.artifacts, self.output, set(), {"sysdeps"})
        manifest = self.artifacts / "sysdeps_lib_generic/artifact_manifest.txt"
        manifest.write_text("compiler/amd-llvm/stage\n")
        with self.assertRaisesRegex(RuntimeError, "protected"):
            import_dependencies(self.artifacts, self.output, set(), {"sysdeps"})
        manifest.write_text("../outside\n")
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            import_dependencies(self.artifacts, self.output, set(), {"sysdeps"})


class AuditTest(unittest.TestCase):
    def test_effective_test_toolchain_cannot_widen_requested_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            project = output / "build/core/hip-tests"
            project.mkdir(parents=True)
            (project / "_init.cmake").write_text("# generated\n")
            toolchain = project / "_toolchain.cmake"
            toolchain.write_text(
                'set(CMAKE_HIP_ARCHITECTURES gfx950;gfx1200 CACHE STRING "From super-project" FORCE)\n'
            )
            with self.assertRaisesRegex(RuntimeError, "gfx1200"):
                audit_gpu_targets(output)
            report = json.loads((output / "target-selection.json").read_text())
            self.assertTrue(report["failures"])
            toolchain.write_text(
                'set(CMAKE_HIP_ARCHITECTURES gfx950 CACHE STRING "From super-project" FORCE)\n'
            )
            audit_gpu_targets(output)
            report = json.loads((output / "target-selection.json").read_text())
            self.assertEqual(
                report["generated_toolchains"]["core/hip-tests/_toolchain.cmake"][
                    "CMAKE_HIP_ARCHITECTURES"
                ],
                ["gfx950"],
            )

    def test_custom_commands_are_independent_and_assembly_inputs_explicit(self):
        commands = shell_commands(
            "cd build && clang -x hip -fsanitize=daddress a.hip && clang -x hip b.hip"
        )
        self.assertEqual(len(commands), 3)
        self.assertTrue(has_daddress(commands[1]))
        self.assertFalse(has_daddress(commands[2]))
        self.assertTrue(
            assembly_device_fixture("clang -target amdgcn trap.S -o trap.o")
        )
        self.assertTrue(
            assembly_device_fixture(
                "clang -target amdgcn -x assembler fixture -o trap.o"
            )
        )
        self.assertFalse(assembly_device_fixture("clang -x hip kernel.hip -o kernel.s"))

    def test_flags_and_specific_spirv_exception(self):
        self.assertTrue(has_daddress("clang -fsanitize=daddress -c a.cpp"))
        self.assertTrue(has_daddress("clang -fsanitize=undefined,daddress -c a.cpp"))
        self.assertFalse(has_daddress("clang -fsanitize=address -c a.cpp"))
        self.assertFalse(has_daddress("clang -fsanitize=daddress-other -c a.cpp"))
        self.assertTrue(
            unsupported_device_fixture(
                "clang --offload-arch=spirv64 -x hip fixture.cpp"
            )
        )
        self.assertTrue(
            unsupported_device_fixture(
                "clang --offload-arch=amdgcnspirv -x hip fixture.cpp"
            )
        )
        self.assertFalse(
            unsupported_device_fixture(
                "clang --offload-arch=gfx950 -x hip spirv_named.cpp"
            )
        )

    def test_fails_uninstrumented_native_commands_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            build = output / "build"
            build.mkdir()
            (build / "artifact_subprojects.json").write_text(
                json.dumps({"rccl": ["rccl"], "base": ["rocm-core"]})
            )
            fragment = build / "compile_commands_fragment_rccl.json"
            row = {
                "file": "kernel.hip",
                "directory": str(build),
                "command": "clang --offload-arch=gfx950 -c kernel.hip",
            }
            fragment.write_text(json.dumps([row]))
            with self.assertRaisesRegex(RuntimeError, "omit daddress"):
                audit_instrumentation(output, {"rccl"}, {"rccl", "base"}, complete=True)
            report = json.loads((output / "instrumentation-audit.json").read_text())
            self.assertEqual(report["components"]["rccl"]["daddress_commands"], 0)
            row["command"] += " -fsanitize=daddress"
            fragment.write_text(json.dumps([row]))
            audit_instrumentation(output, {"rccl"}, {"rccl", "base"}, complete=True)
            fragment.write_text("")
            audit_instrumentation(output, {"rccl"}, {"rccl", "base"}, complete=False)
            with self.assertRaisesRegex(RuntimeError, "missing native"):
                audit_instrumentation(output, {"rccl"}, {"rccl", "base"}, complete=True)


if __name__ == "__main__":
    unittest.main()
