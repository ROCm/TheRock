"""Utils package for test execution with system detection and result reporting.

Provides TestClient API for collecting system info (OS, hardware, ROCm) and
uploading test results to API endpoints.

Organization:
    - config/: Configuration loading and validation
    - system/: Platform, hardware, and ROCm detection
    - results/: Results collection and API submission

"""

__version__ = "1.0.0"
__all__ = [
    # Core modules (at root)
    'logger',
    'constants',
    'exceptions',
    # Main API
    'TestClient',
    # Commonly used exports
    'SystemContext',
    'SystemDetector',
    'ConfigHelper',
    'ConfigParser',
    'ConfigValidator',
    'ResultsHandler',
    'ResultsAPI',
    'HardwareDetector',
    'PlatformDetector',
    'ROCmDetector',
]

# Import main API
from .test_client import TestClient

# Export commonly used classes from subdirectories
from .system import SystemContext, SystemDetector, HardwareDetector, PlatformDetector, ROCmDetector
from .config import ConfigHelper, ConfigParser, ConfigValidator
from .results import ResultsHandler, ResultsAPI
