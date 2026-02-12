import unittest

from configure_stage import get_platform_cmake_args


class GetPlatformCmakeArgsTest(unittest.TestCase):
    def test_linux_returns_empty(self):
        result = get_platform_cmake_args(platform_name="linux")
        self.assertEqual(result, [])

    def test_windows_with_vctools(self):
        import os

        old = os.environ.get("VCToolsInstallDir")
        try:
            os.environ["VCToolsInstallDir"] = "C:/Program Files/MSVC/14.40"
            result = get_platform_cmake_args(platform_name="windows")
            self.assertIn(
                "-DCMAKE_C_COMPILER=C:/Program Files/MSVC/14.40/bin/Hostx64/x64/cl.exe",
                result,
            )
            self.assertIn(
                "-DCMAKE_CXX_COMPILER=C:/Program Files/MSVC/14.40/bin/Hostx64/x64/cl.exe",
                result,
            )
            self.assertIn(
                "-DCMAKE_LINKER=C:/Program Files/MSVC/14.40/bin/Hostx64/x64/link.exe",
                result,
            )
            self.assertIn("-DTHEROCK_BACKGROUND_BUILD_JOBS=4", result)
        finally:
            if old is None:
                os.environ.pop("VCToolsInstallDir", None)
            else:
                os.environ["VCToolsInstallDir"] = old

    def test_windows_without_vctools_raises(self):
        import os

        old = os.environ.pop("VCToolsInstallDir", None)
        try:
            with self.assertRaises(RuntimeError):
                get_platform_cmake_args(platform_name="windows")
        finally:
            if old is not None:
                os.environ["VCToolsInstallDir"] = old

    def test_windows_backslash_normalization(self):
        import os

        old = os.environ.get("VCToolsInstallDir")
        try:
            os.environ["VCToolsInstallDir"] = "C:\\Program Files\\MSVC\\14.40\\"
            result = get_platform_cmake_args(platform_name="windows")
            # All paths should use forward slashes, no trailing slash
            compiler_flags = [a for a in result if "CMAKE_" in a]
            for flag in compiler_flags:
                self.assertNotIn("\\", flag)
                self.assertNotIn("//bin", flag)
        finally:
            if old is None:
                os.environ.pop("VCToolsInstallDir", None)
            else:
                os.environ["VCToolsInstallDir"] = old

    def test_comments_mode(self):
        result = get_platform_cmake_args(platform_name="windows", include_comments=True)
        comment_lines = [a for a in result if a.startswith("#")]
        self.assertTrue(len(comment_lines) > 0)
