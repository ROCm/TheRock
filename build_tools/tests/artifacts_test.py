# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import tempfile
import textwrap
import unittest
import sys

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifacts import ArtifactName
import _therock_utils.artifact_builder as builder


class TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        override_temp = os.getenv("TEST_TMPDIR")
        if override_temp is not None:
            self.temp_context = None
            self.temp_dir = Path(override_temp)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_context = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self.temp_context.name)

    def tearDown(self):
        if self.temp_context:
            self.temp_context.cleanup()

    def write_indented(self, relpath: Path, contents: str):
        contents = textwrap.dedent(contents)
        p = self.temp_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)

    def touch(self, relpath: Path):
        p = self.temp_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


class ArtifactNameTest(TmpDirTestCase):
    def testFromPath(self):
        p1 = Path(self.temp_dir / "dir" / "name_component_generic")
        p1.mkdir(parents=True, exist_ok=True)
        p2 = Path(self.temp_dir / "other" / "name_component_generic.tar.xz")
        p2.parent.mkdir(parents=True, exist_ok=True)
        p2.touch()

        an1 = ArtifactName.from_path(p1)
        an2 = ArtifactName.from_path(p2)
        self.assertEqual(an1.name, "name")
        self.assertEqual(an1.component, "component")
        self.assertEqual(an2.target_family, "generic")
        self.assertEqual(an1, an2)
        self.assertEqual(hash(an1), hash(an2))

    def testFromFilename(self):
        f1 = "name_component_generic.tar.xz"
        an1 = ArtifactName.from_filename(f1)
        self.assertEqual(an1.name, "name")
        self.assertEqual(an1.component, "component")
        self.assertEqual(an1.target_family, "generic")

        f_invalid1 = "invalid_name.zip"
        an_invalid1 = ArtifactName.from_filename(f_invalid1)
        self.assertIsNone(an_invalid1)

        # Component names containing one underscore could be misinterpretted
        # as [name]_[component]_[target family]_suffix with each group shifted
        # over one. See https://github.com/ROCm/TheRock/issues/935.
        f_invalid2 = "underscore_name_component_generic.tar.xz"
        an_invalid2 = ArtifactName.from_filename(f_invalid2)
        self.assertIsNone(an_invalid2)


class ArtifactDescriptorTomlValidationTest(TmpDirTestCase):
    def testTopLevel(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        foobar = 1
        """,
        )
        with self.assertRaisesRegex(ValueError, "illegal key: 'foobar'"):
            builder.ArtifactDescriptor.load_toml_file(
                self.temp_dir / "descriptor.toml", artifact_name="test"
            )

    def testComponentExtendsDefault(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.run]
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        self.assertListEqual(d.components["run"].extends, ["lib"])

    def testComponentExtends(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib]
        extends = ["extras"]
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        self.assertListEqual(d.components["lib"].extends, ["extras"])

    def testComponentExtendsStr(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib]
        extends = "extras"
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        self.assertListEqual(d.components["lib"].extends, ["extras"])

    def testBasedirUnrecognized(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."stage/somedir"]
        foobar = 1
        """,
        )
        with self.assertRaisesRegex(ValueError, "illegal key: 'foobar'"):
            builder.ArtifactDescriptor.load_toml_file(
                self.temp_dir / "descriptor.toml", artifact_name="test"
            )

    def testBasedirDefaults(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."stage/somedir"]
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]
        # Includes should have lib defaults only
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.includes],
            builder.ComponentDefaults.ALL["lib"].includes,
        )
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.excludes],
            builder.ComponentDefaults.ALL["lib"].excludes,
        )
        # Kpack patterns should be in force_includes
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.force_includes],
            [".kpack/test_lib.kpm", ".kpack/test_lib_*.kpack"],
        )
        self.assertFalse(bd.optional)

    def testBasedirPlatformOptional(self):
        self.write_indented(
            "descriptor.toml",
            rf"""
        [components.lib."stage/somedir"]
        optional = "{platform.system().upper()}"
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]
        self.assertTrue(bd.optional)

    def testBasedirPlatformNotOptional(self):
        self.write_indented(
            "descriptor.toml",
            rf"""
        [components.lib."stage/somedir"]
        optional = "not{platform.system().upper()}"
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]
        self.assertFalse(bd.optional)

    def testBasedirNoDefaults(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."stage/somedir"]
        default_patterns = false
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]
        # With default_patterns=false, includes should be empty
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.includes],
            [],
        )
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.excludes],
            [],
        )
        # Kpack patterns should still be in force_includes
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.force_includes],
            [".kpack/test_lib.kpm", ".kpack/test_lib_*.kpack"],
        )

    def testBasedirExplicitLists(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."stage/somedir"]
        include = ["**/abc"]
        exclude = ["**/def"]
        force_include = ["**/xyz"]
        optional = true
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]
        expected_includes = ["**/abc"] + builder.ComponentDefaults.ALL["lib"].includes
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.includes],
            expected_includes,
        )
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.excludes],
            ["**/def"],
        )
        # Kpack patterns added to force_includes along with explicit ones
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.force_includes],
            ["**/xyz", ".kpack/test_lib.kpm", ".kpack/test_lib_*.kpack"],
        )
        self.assertTrue(bd.optional)

    def testBasedirExplicitStrs(self):
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."stage/somedir"]
        include = "**/abc"
        exclude = "**/def"
        force_include = "**/xyz"
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]
        expected_includes = ["**/abc"] + builder.ComponentDefaults.ALL["lib"].includes
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.includes],
            expected_includes,
        )
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.excludes],
            ["**/def"],
        )
        # Kpack patterns added to force_includes along with explicit ones
        self.assertEqual(
            [pattern.glob for pattern in bd.predicate.force_includes],
            ["**/xyz", ".kpack/test_lib.kpm", ".kpack/test_lib_*.kpack"],
        )

    def testEmptyComponentMatchesEverything(self):
        """Regression test: empty includes (no patterns) should match all files.

        This is a critical invariant - when a component has no include patterns
        (and default_patterns=false), it should match everything. Previously,
        adding kpack patterns to 'includes' broke this because non-empty includes
        require files to match at least one pattern. Kpack patterns are now in
        force_includes to preserve this behavior.
        """
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."stage/somedir"]
        default_patterns = false
        """,
        )
        d = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        bd = d.components["lib"].basedirs["stage/somedir"]

        # Verify includes is empty (the critical invariant)
        self.assertEqual([pattern.glob for pattern in bd.predicate.includes], [])

        # Create test files of various types
        stage_dir = self.temp_dir / "src" / "stage" / "somedir"
        stage_dir.mkdir(parents=True)
        (stage_dir / "bin").mkdir()
        (stage_dir / "bin" / "executable").write_text("exe")
        (stage_dir / "lib").mkdir()
        (stage_dir / "lib" / "libfoo.so").write_text("lib")
        (stage_dir / "random.txt").write_text("text")

        # Use PatternMatcher to verify all files match
        from _therock_utils.pattern_match import PatternMatcher

        pm = PatternMatcher(
            includes=[p.glob for p in bd.predicate.includes],
            excludes=[p.glob for p in bd.predicate.excludes],
            force_includes=[p.glob for p in bd.predicate.force_includes],
        )
        pm.add_basedir(stage_dir)

        matched_files = set(relpath for relpath, _ in pm.matches())
        # All files should match when includes is empty
        self.assertIn("bin/executable", matched_files)
        self.assertIn("lib/libfoo.so", matched_files)
        self.assertIn("random.txt", matched_files)


class ComponentScannerTest(TmpDirTestCase):
    def testNoRootDirNoop(self):
        self.write_indented(
            "descriptor.toml",
            # Note: in reverse extends order, this ensures that the worklist traverses
            # properly.
            r"""
        [components.doc."a/stage"]
        [components.dev."a/stage"]
        [components.dbg."a/stage"]
        [components.lib."a/stage"]
        [components.run."a/stage"]
        """,
        )
        ad = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        scanner = builder.ComponentScanner(self.temp_dir / "src", ad)
        self.assertSetEqual(scanner.matched_relpaths, set())

    def testNoMatches(self):
        self.write_indented(
            "descriptor.toml",
            # Note: in reverse extends order, this ensures that the worklist traverses
            # properly.
            r"""
        [components.doc."a/stage"]
        [components.dev."a/stage"]
        [components.dbg."a/stage"]
        [components.lib."a/stage"]
        [components.run."a/stage"]
        """,
        )
        (self.temp_dir / "src").mkdir()
        ad = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        scanner = builder.ComponentScanner(self.temp_dir / "src", ad)
        self.assertSetEqual(scanner.matched_relpaths, set())

    def testHappyPath(self):
        self.write_indented(
            "descriptor.toml",
            # Note: in reverse extends order, this ensures that the worklist traverses
            # properly.
            r"""
        [options]
        unmatched_exclude = ["bin/xyz.exe"]

        [components.doc."a/stage"]
        include = [
            "share/doc/README.md",
            # Should not match because it will already have been consumed by lib
            "lib/libfoo*",
            # Should match because lib excluded it and it won't have been consumed
            # yet.
            "bin/def.exe",
        ]
        [components.dev."a/stage"]
        [components.dbg."a/stage"]
        [components.lib."a/stage"]
        [components.run."a/stage"]
        include = "bin/*.exe"
        exclude = [
            "bin/def.exe",
            "bin/xyz.exe",
        ]
        [components.lib."b/stage"]
        """,
        )
        ad = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        self.touch("src/a/stage/lib/libfoo.so.1")
        self.touch("src/a/stage/lib/libfoo.a")
        self.touch("src/a/stage/bin/abc.exe")
        self.touch("src/a/stage/bin/def.exe")
        self.touch("src/a/stage/bin/xyz.exe")
        self.touch("src/a/stage/.build-id/999999999/88888.debug")
        self.touch("src/a/stage/share/doc/README.md")
        self.touch("src/b/stage/lib/libbar.so.1")

        scanner = builder.ComponentScanner(self.temp_dir / "src", ad)
        scanner.verify()
        # Ensure aggregate contents are correct.
        self.assertSetEqual(
            scanner.matched_relpaths,
            set(
                [
                    "bin/abc.exe",
                    "bin/def.exe",
                    "lib/libbar.so.1",
                    "lib/libfoo.a",
                    "lib/libfoo.so.1",
                    "share/doc/README.md",
                    "share/doc",
                    ".build-id/999999999/88888.debug",
                ]
            ),
        )
        self.assertSetEqual(
            set(relpath for relpath, _ in scanner.unmatched_files),
            set(
                [
                    "bin/xyz.exe",
                ]
            ),
        )

        # Ensure component contents are correct.
        lib_comp = scanner.components["lib"]
        pm_a = lib_comp.basedir_contents["a/stage"]
        pm_b = lib_comp.basedir_contents["b/stage"]
        self.assertSetEqual(set(pm_a.all.keys()), set(["lib/libfoo.so.1"]))
        self.assertSetEqual(set(pm_b.all.keys()), set(["lib/libbar.so.1"]))

        # Write the artifact and verify.
        lib_comp.write_artifact(self.temp_dir / "out")
        manifest = (self.temp_dir / "out" / "artifact_manifest.txt").read_text()
        self.assertSetEqual(set(manifest.splitlines()), set(["a/stage", "b/stage"]))
        self.assertTrue(
            (self.temp_dir / "out" / "a" / "stage" / "lib" / "libfoo.so.1").exists()
        )
        self.assertTrue(
            (self.temp_dir / "out" / "b" / "stage" / "lib" / "libbar.so.1").exists()
        )
        # Artifact files must be copies, not hardlinks to the source.
        # Hardlinks cause race conditions when the split pipeline modifies
        # files still referenced by stage/dist (#3447).
        self.assertFalse(
            os.path.samefile(
                self.temp_dir / "src" / "a" / "stage" / "lib" / "libfoo.so.1",
                self.temp_dir / "out" / "a" / "stage" / "lib" / "libfoo.so.1",
            ),
            "artifact should be a copy, not a hardlink",
        )

    def testNonOptionalNotExists(self):
        self.write_indented(
            "descriptor.toml",
            # Note: in reverse extends order, this ensures that the worklist traverses
            # properly.
            r"""
        [components.doc."a/stage"]
        """,
        )
        ad = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )

        scanner = builder.ComponentScanner(self.temp_dir / "src", ad)
        with self.assertRaisesRegex(
            ValueError,
            "Directory a/stage of doc: marked non-optional but does not exist",
        ):
            scanner.verify()

    def testOptionalNotExists(self):
        self.write_indented(
            "descriptor.toml",
            # Note: in reverse extends order, this ensures that the worklist traverses
            # properly.
            r"""
        [components.doc."a/stage"]
        optional = true
        """,
        )
        ad = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )

        scanner = builder.ComponentScanner(self.temp_dir / "src", ad)
        scanner.verify()

    def testUnmatchedUndeclared(self):
        self.write_indented(
            "descriptor.toml",
            # Note: in reverse extends order, this ensures that the worklist traverses
            # properly.
            r"""
        [components.doc."a/stage"]
        """,
        )
        ad = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="test"
        )
        self.touch("src/a/stage/not/default/dir/README.md")

        scanner = builder.ComponentScanner(self.temp_dir / "src", ad)
        with self.assertRaisesRegex(ValueError, "Unmatched artifact files"):
            scanner.verify()

    def testKpackFileInclusion(self):
        """Test that kpack files are automatically included based on artifact/component naming."""
        # Create directory structure with .kpack files
        stage_dir = self.temp_dir / "src" / "myproject" / "stage"
        kpack_dir = stage_dir / ".kpack"
        kpack_dir.mkdir(parents=True)

        # Create kpack files matching artifact "myartifact" components "lib" and "test"
        (kpack_dir / "myartifact_lib.kpm").write_text("manifest")
        (kpack_dir / "myartifact_lib_gfx1100.kpack").write_text("lib code")
        (kpack_dir / "myartifact_lib_gfx1101.kpack").write_text("lib code")
        (kpack_dir / "myartifact_test.kpm").write_text("test manifest")
        (kpack_dir / "myartifact_test_gfx1100.kpack").write_text("test code")
        # Also create a kpack file for different artifact (should NOT be included)
        (kpack_dir / "other_lib.kpm").write_text("other manifest")

        # Create a regular file for lib component
        (stage_dir / "lib").mkdir()
        (stage_dir / "lib" / "libfoo.so").write_text("library")

        # Write descriptor
        self.write_indented(
            "descriptor.toml",
            r"""
        [options]
        unmatched_exclude = [".kpack/other_lib.kpm", ".kpack/myartifact_test*"]
        [components.lib."myproject/stage"]
        """,
        )

        # Load descriptor with artifact_name (patterns added at descriptor level)
        descriptor = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="myartifact"
        )
        scanner = builder.ComponentScanner(self.temp_dir / "src", descriptor)

        # Verify lib component has lib kpack files
        lib_contents = scanner.components["lib"]
        lib_files = set()
        for bd, pm in lib_contents.basedir_contents.items():
            for relpath, _ in pm.matches():
                lib_files.add(relpath)

        self.assertIn(".kpack/myartifact_lib.kpm", lib_files)
        self.assertIn(".kpack/myartifact_lib_gfx1100.kpack", lib_files)
        self.assertIn(".kpack/myartifact_lib_gfx1101.kpack", lib_files)
        self.assertNotIn(
            ".kpack/myartifact_test.kpm", lib_files
        )  # test files not in lib
        self.assertNotIn(".kpack/other_lib.kpm", lib_files)  # different artifact

    def testKpackNoMatchingFiles(self):
        """Test that scanner works when .kpack dir exists but has no matching files."""
        stage_dir = self.temp_dir / "src" / "myproject" / "stage"
        kpack_dir = stage_dir / ".kpack"
        kpack_dir.mkdir(parents=True)

        # Only create kpack files for a different artifact
        (kpack_dir / "other_lib.kpm").write_text("other manifest")

        self.write_indented(
            "descriptor.toml",
            r"""
        [options]
        unmatched_exclude = [".kpack/other_lib.kpm"]
        [components.lib."myproject/stage"]
        """,
        )

        descriptor = builder.ArtifactDescriptor.load_toml_file(
            self.temp_dir / "descriptor.toml", artifact_name="myartifact"
        )
        scanner = builder.ComponentScanner(self.temp_dir / "src", descriptor)

        # Should not fail, just have no kpack files
        lib_contents = scanner.components["lib"]
        lib_files = set()
        for bd, pm in lib_contents.basedir_contents.items():
            for relpath, _ in pm.matches():
                lib_files.add(relpath)

        self.assertNotIn(".kpack/other_lib.kpm", lib_files)

    def testArtifactNameRequired(self):
        """Test that artifact_name is required."""
        self.write_indented(
            "descriptor.toml",
            r"""
        [components.lib."myproject/stage"]
        """,
        )

        # Test that load_toml_file requires artifact_name
        with self.assertRaises(TypeError):
            builder.ArtifactDescriptor.load_toml_file(self.temp_dir / "descriptor.toml")

        # Test that empty artifact_name raises error
        with self.assertRaisesRegex(ValueError, "artifact_name is required"):
            builder.ArtifactDescriptor.load_toml_file(
                self.temp_dir / "descriptor.toml", artifact_name=""
            )


class ArtifactNameEdgeCaseTest(TmpDirTestCase):
    """Additional edge case tests for ArtifactName parsing."""

    def testFromPathNonDir(self):
        """Test from_path with an archive file."""
        p = Path(self.temp_dir / "name_component_generic.tar.zst")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        an = ArtifactName.from_path(p)
        self.assertIsNotNone(an)
        self.assertEqual(an.name, "name")
        self.assertEqual(an.component, "component")
        self.assertEqual(an.target_family, "generic")

    def testFromPathInvalidDir(self):
        """Test from_path returns None for directories with wrong format."""
        # Needs only two segments or more than three to be invalid
        p = Path(self.temp_dir / "onlytwoparts")
        p.mkdir(parents=True, exist_ok=True)
        an = ArtifactName.from_path(p)
        self.assertIsNone(an)

    def testFromFilenameZst(self):
        """Test from_filename with .tar.zst extension."""
        an = ArtifactName.from_filename("name_component_gfx1100.tar.zst")
        self.assertIsNotNone(an)
        self.assertEqual(an.name, "name")
        self.assertEqual(an.component, "component")
        self.assertEqual(an.target_family, "gfx1100")

    def testFromFilenameInvalidExtension(self):
        """Test from_filename returns None for unsupported extensions."""
        self.assertIsNone(ArtifactName.from_filename("name_component_generic.tar.gz"))
        self.assertIsNone(ArtifactName.from_filename("name_component_generic.zip"))

    def testRepr(self):
        """Test string representation."""
        an = ArtifactName("rocblas", "lib", "gfx1100")
        self.assertEqual(repr(an), "Artifact(rocblas[lib:gfx1100])")

    def testEqualityDifferentComponent(self):
        """Test inequality for different components."""
        an1 = ArtifactName("name", "lib", "generic")
        an2 = ArtifactName("name", "dev", "generic")
        self.assertNotEqual(an1, an2)

    def testEqualityDifferentType(self):
        """Test equality returns NotImplemented for non-ArtifactName."""
        an = ArtifactName("name", "lib", "generic")
        result = an.__eq__("not an ArtifactName")
        self.assertIs(result, NotImplemented)

    def testHashConsistency(self):
        """Test that equal ArtifactNames have same hash."""
        an1 = ArtifactName("a", "b", "c")
        an2 = ArtifactName("a", "b", "c")
        an3 = ArtifactName("a", "b", "d")
        self.assertEqual(hash(an1), hash(an2))
        # Different names should (usually) have different hashes
        self.assertNotEqual(hash(an1), hash(an3))

    def testCanBeUsedInSet(self):
        """Test that ArtifactNames work in sets."""
        an1 = ArtifactName("a", "b", "c")
        an2 = ArtifactName("a", "b", "c")
        an3 = ArtifactName("x", "y", "z")
        s = {an1, an2, an3}
        self.assertEqual(len(s), 2)


class OpenArchiveForReadTest(TmpDirTestCase):
    """Tests for _open_archive_for_read function."""

    def test_unknown_format_raises(self):
        """Test that unknown archive format raises ValueError."""
        from _therock_utils.artifacts import _open_archive_for_read

        p = Path(self.temp_dir / "file.tar.gz")
        p.touch()
        with self.assertRaises(ValueError) as ctx:
            _open_archive_for_read(p)
        self.assertIn("Unknown archive format", str(ctx.exception))

    def test_xz_archive(self):
        """Test opening a .tar.xz archive."""
        import tarfile

        from _therock_utils.artifacts import _open_archive_for_read

        # Create a valid tar.xz archive
        archive_path = Path(self.temp_dir / "test_archive.tar.xz")
        test_file = Path(self.temp_dir / "test_content.txt")
        test_file.write_text("test content")

        with tarfile.open(archive_path, "w:xz") as tf:
            tf.add(test_file, arcname="test_content.txt")

        with _open_archive_for_read(archive_path) as tf:
            members = tf.getnames()
            self.assertIn("test_content.txt", members)

    def test_zst_archive(self):
        """Test opening a .tar.zst archive."""
        import tarfile

        import pyzstd
        from _therock_utils.artifacts import _open_archive_for_read

        # Create a valid tar.zst archive
        archive_path = Path(self.temp_dir / "test_archive.tar.zst")
        test_file = Path(self.temp_dir / "test_content.txt")
        test_file.write_text("test content")

        # Create tar archive in memory, then compress with zstd
        tar_path = Path(self.temp_dir / "temp.tar")
        with tarfile.open(tar_path, "w") as tf:
            tf.add(test_file, arcname="test_content.txt")

        with open(tar_path, "rb") as f_in:
            data = f_in.read()
        compressed = pyzstd.compress(data)
        archive_path.write_bytes(compressed)

        with _open_archive_for_read(archive_path) as tf:
            members = tf.getnames()
            self.assertIn("test_content.txt", members)


class ArtifactCatalogTest(TmpDirTestCase):
    """Tests for ArtifactCatalog class."""

    def _create_artifact_dir(self, name: str, manifest_lines: list[str]):
        """Create an artifact directory with manifest and files."""
        artifact_dir = self.temp_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest = artifact_dir / "artifact_manifest.txt"
        manifest.write_text("\n".join(manifest_lines) + "\n")

        # Create directories referenced by manifest
        for line in manifest_lines:
            if line:
                (artifact_dir / line).mkdir(parents=True, exist_ok=True)

        return artifact_dir

    def test_scans_artifact_directories(self):
        """Test that ArtifactCatalog scans valid artifact directories."""
        from _therock_utils.artifacts import ArtifactCatalog

        # Create artifact dirs in the expected format
        self._create_artifact_dir("mylib_lib_generic", ["stage/rocm"])
        # Add a file inside the manifested dir
        (self.temp_dir / "mylib_lib_generic" / "stage/rocm" / "libfoo.so").write_text(
            "content"
        )

        catalog = ArtifactCatalog(self.temp_dir)
        self.assertEqual(len(catalog.artifact_basedirs), 1)
        names = catalog.artifact_names
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].name, "mylib")
        self.assertEqual(names[0].component, "lib")

    def test_filter_artifacts(self):
        """Test that ArtifactCatalog respects the filter function."""
        from _therock_utils.artifacts import ArtifactCatalog

        self._create_artifact_dir("mylib_lib_generic", ["stage/rocm"])
        (self.temp_dir / "mylib_lib_generic" / "stage/rocm" / "lib.so").write_text("")

        self._create_artifact_dir("mylib_dev_generic", ["stage/rocm"])
        (self.temp_dir / "mylib_dev_generic" / "stage/rocm" / "foo.h").write_text("")

        # Filter to only lib component
        catalog = ArtifactCatalog(
            self.temp_dir, filter=lambda an: an.component == "lib"
        )
        names = catalog.artifact_names
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].component, "lib")

    def test_all_target_families(self):
        """Test all_target_families property."""
        from _therock_utils.artifacts import ArtifactCatalog

        self._create_artifact_dir("mylib_lib_gfx1100", ["stage/rocm"])
        (self.temp_dir / "mylib_lib_gfx1100" / "stage/rocm" / "lib.so").write_text("")

        self._create_artifact_dir("mylib_lib_generic", ["stage/rocm"])
        (self.temp_dir / "mylib_lib_generic" / "stage/rocm" / "lib.so").write_text("")

        catalog = ArtifactCatalog(self.temp_dir)
        families = catalog.all_target_families
        # 'generic' should be excluded
        self.assertNotIn("generic", families)
        self.assertIn("gfx1100", families)

    def test_skips_non_artifact_directories(self):
        """Test that non-artifact directories are skipped."""
        from _therock_utils.artifacts import ArtifactCatalog

        # Directory without proper name format
        (self.temp_dir / "not_an_artifact").mkdir()
        # File (not directory)
        (self.temp_dir / "file.txt").write_text("")
        # Directory without manifest
        (self.temp_dir / "valid_name_generic").mkdir()

        catalog = ArtifactCatalog(self.temp_dir)
        self.assertEqual(len(catalog.artifact_basedirs), 0)


class ArtifactPopulatorTest(TmpDirTestCase):
    """Tests for ArtifactPopulator class."""

    def _create_artifact_tree(self, name: str, files: dict[str, str]):
        """Create an artifact directory with manifest and files.

        Args:
            name: Artifact directory name
            files: Dict mapping relative paths (under stage/rocm) to content
        """
        artifact_dir = self.temp_dir / "artifacts" / name
        manifest_paths = ["stage/rocm"]

        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = artifact_dir / "artifact_manifest.txt"
        manifest.write_text("\n".join(manifest_paths))

        for relpath, content in files.items():
            full_path = artifact_dir / "stage/rocm" / relpath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        return artifact_dir

    def test_populate_from_dir(self):
        """Test populating from an exploded artifact directory."""
        from _therock_utils.artifacts import ArtifactPopulator

        artifact_dir = self._create_artifact_tree(
            "mylib_lib_generic",
            {"lib/libfoo.so": "library content", "bin/tool": "tool content"},
        )

        output = self.temp_dir / "output"
        populator = ArtifactPopulator(output_path=output)
        populator(artifact_dir)

        self.assertTrue((output / "stage/rocm" / "lib" / "libfoo.so").exists())
        self.assertTrue((output / "stage/rocm" / "bin" / "tool").exists())
        self.assertEqual(
            (output / "stage/rocm" / "lib" / "libfoo.so").read_text(),
            "library content",
        )

    def test_populate_from_dir_flatten(self):
        """Test populating with flatten=True."""
        from _therock_utils.artifacts import ArtifactPopulator

        artifact_dir = self._create_artifact_tree(
            "mylib_lib_generic",
            {"lib/libfoo.so": "library content"},
        )

        output = self.temp_dir / "output"
        populator = ArtifactPopulator(output_path=output, flatten=True)
        populator(artifact_dir)

        # With flatten, files go directly to output without the relpath prefix
        self.assertTrue((output / "lib" / "libfoo.so").exists())
        self.assertFalse((output / "stage/rocm" / "lib" / "libfoo.so").exists())

    def test_on_relpath_callback(self):
        """Test that on_relpath is called for each relpath."""
        from _therock_utils.artifacts import ArtifactPopulator

        artifact_dir = self._create_artifact_tree(
            "mylib_lib_generic", {"file.txt": "content"}
        )

        output = self.temp_dir / "output"
        populator = ArtifactPopulator(output_path=output)
        populator(artifact_dir)

        self.assertIn("stage/rocm", populator.relpaths)

    def test_populate_from_tar_xz_archive(self):
        """Test populating from a .tar.xz archive."""
        import tarfile

        from _therock_utils.artifacts import ArtifactPopulator

        archive_path = self.temp_dir / "mylib_lib_generic.tar.xz"

        with tarfile.open(archive_path, "w:xz") as tf:
            # Add manifest as first member
            manifest_content = b"stage/rocm\n"
            import io

            manifest_info = tarfile.TarInfo(name="artifact_manifest.txt")
            manifest_info.size = len(manifest_content)
            tf.addfile(manifest_info, io.BytesIO(manifest_content))

            # Add a file
            file_content = b"library content"
            file_info = tarfile.TarInfo(name="stage/rocm/lib/libfoo.so")
            file_info.size = len(file_content)
            file_info.mode = 0o755
            tf.addfile(file_info, io.BytesIO(file_content))

        output = self.temp_dir / "output"
        populator = ArtifactPopulator(output_path=output)
        populator(archive_path)

        self.assertTrue((output / "stage/rocm" / "lib" / "libfoo.so").exists())
        self.assertEqual(
            (output / "stage/rocm" / "lib" / "libfoo.so").read_text(),
            "library content",
        )

    def test_archive_without_manifest_raises(self):
        """Test that archive without manifest as first member raises error."""
        import io
        import tarfile

        from _therock_utils.artifacts import ArtifactPopulator

        archive_path = self.temp_dir / "bad_archive.tar.xz"

        with tarfile.open(archive_path, "w:xz") as tf:
            # Add a file instead of manifest as first member
            content = b"not a manifest"
            info = tarfile.TarInfo(name="some_file.txt")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))

        output = self.temp_dir / "output"
        populator = ArtifactPopulator(output_path=output)
        with self.assertRaises(IOError) as ctx:
            populator(archive_path)
        self.assertIn("artifact_manifest.txt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
