"""
Custom exceptions for test execution and system operations.
"""


class FrameworkException(Exception):
    """Base exception for complete framework custom exceptions."""

    pass


class ConfigurationError(FrameworkException):
    """Configuration file errors (missing, invalid YAML, validation failures)."""

    pass


class HardwareDetectionError(FrameworkException):
    """Hardware detection failures (CPU, GPU, initialization errors)."""

    pass


class ROCmNotFoundError(FrameworkException):
    """ROCm not found or version cannot be determined."""

    pass


class ROCmVersionError(FrameworkException):
    """ROCm version incompatibility or requirement not met."""

    pass


class TestExecutionError(FrameworkException):
    """Test execution failures (script not found, timeout, critical errors)."""

    pass


class ValidationError(FrameworkException):
    """Data or input validation failures."""

    pass


class RequirementNotMetError(FrameworkException):
    """System requirements not met (GPU, ROCm, minimum specs)."""

    pass


class BenchmarkExecutionError(FrameworkException):
    """Benchmark execution or parsing failure (infrastructure/setup issue).

    Raised when benchmarks fail to run or results cannot be parsed.

    Common causes:
        - Binary not found or crashed during execution
        - Log parsing failed (missing CSV headers/data)
        - File I/O errors (permissions, disk space)
        - Configuration errors

    This is NOT a performance issue - the benchmark itself failed to execute properly.
    """

    pass


class BenchmarkResultError(FrameworkException):
    """Benchmark result validation failure (test/performance issue).

    Raised when benchmarks execute successfully but results show failures.

    Common causes:
        - Performance regression detected (FAIL status)
        - Missing baseline data (UNKNOWN status)
        - Result validation errors
        - Missing expected result columns

    The benchmark executed successfully - this indicates a test failure or missing data.
    """

    pass
