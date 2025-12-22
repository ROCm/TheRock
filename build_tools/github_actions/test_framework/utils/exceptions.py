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
