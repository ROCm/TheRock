"""
Custom exceptions for test execution and system operations.
"""

from typing import Type, List, Dict


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


# Exception helper functions
def raise_with_solution(exception_class: Type, message: str, solutions: List) -> None:
    """Raise exception with formatted error message and suggested solutions.
    
    Args:
        exception_class: Exception class to raise
        message: Error message
        solutions: List of suggested solution strings
    """
    solution_text = "\n".join(f"  {i+1}. {sol}" for i, sol in enumerate(solutions))
    full_message = f"{message}\n\nSuggested solutions:\n{solution_text}"
    raise exception_class(full_message)


def format_error_with_context(error: Exception, context: Dict) -> str:
    """Format error message with additional context information.
    
    Args:
        error: Original exception
        context: Context dict (e.g., test_id, script_path)
        
    Returns:
        str: Formatted error message with context
    """
    context_str = "\n".join(f"  {k}: {v}" for k, v in context.items())
    return f"{str(error)}\n\nContext:\n{context_str}"
