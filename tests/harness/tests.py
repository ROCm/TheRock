# TODO: add filter options


class TestRock:
    """This is an Pytest Test Suite Class to test various components of TheRock"""

    def test_hipcub(self, orch, therock_path, result):
        """A Test case to verify hipcub"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/hipcub")
        assert result.testVerdict

    def test_rocprim(self, orch, therock_path, result):
        """A Test case to verify rocprim"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/rocprim")
        assert result.testVerdict

    def test_rocrand(self, orch, therock_path, result):
        """A Test case to verify rocrand"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/rocRAND")
        assert result.testVerdict

    def test_rocthrust(self, orch, therock_path, result):
        """A Test case to verify rocthrust"""
        result.testVerdict = orch.runCtest(cwd=f"{therock_path}/bin/rocthrust")
        assert result.testVerdict
