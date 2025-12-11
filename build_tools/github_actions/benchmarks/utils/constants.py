"""Centralized constants for test execution, API, and system configuration."""

from pathlib import Path


class Constants:
    """Application-wide constants for paths, timeouts, status codes, and config values."""
    
    # File Paths
    DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs/config.yml"
    DEFAULT_LOG_DIR = "./logs"
    DEFAULT_RESULTS_DIR = "./results"
    
    # File Extensions
    EXT_JSON = ".json"
    EXT_YAML = ".yaml"
    EXT_YML = ".yml"
    EXT_LOG = ".log"
    
    # Log Levels
    LOG_LEVEL_DEBUG = "DEBUG"
    LOG_LEVEL_INFO = "INFO"
    LOG_LEVEL_WARNING = "WARNING"
    LOG_LEVEL_ERROR = "ERROR"
    LOG_LEVEL_CRITICAL = "CRITICAL"
    
    # Default Values
    DEFAULT_LOG_LEVEL = LOG_LEVEL_INFO
    DEFAULT_LOG_MAX_SIZE_MB = 10
    DEFAULT_LOG_BACKUP_COUNT = 5
    DEFAULT_TIMEOUT = 3600
    DEFAULT_API_TIMEOUT = 30
    DEFAULT_API_MAX_RETRIES = 3
    DEFAULT_API_RETRY_DELAY = 5
    
    # Test Status
    TEST_STATUS_PASS = "PASS"
    TEST_STATUS_FAIL = "FAIL"
    TEST_STATUS_SKIP = "SKIP"
    TEST_STATUS_ERROR = "ERROR"
    TEST_STATUS_TIMEOUT = "TIMEOUT"
    
    # Test Environment
    TEST_ENV_BARE_METAL = "bm"
    TEST_ENV_VM = "vm"
    TEST_ENV_DOCKER = "docker"
    
    # Exit Codes
    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1
    EXIT_CONFIG_ERROR = 2
    EXIT_REQUIREMENT_ERROR = 3
    EXIT_TEST_FAILURE = 4
    
    # API Endpoints
    API_ENDPOINT_RESULTS = "/api/v1/results"
    API_ENDPOINT_HEALTH = "/api/v1/health"
    
    # HTTP Status Codes
    HTTP_OK = 200
    HTTP_CREATED = 201
    HTTP_BAD_REQUEST = 400
    HTTP_UNAUTHORIZED = 401
    HTTP_FORBIDDEN = 403
    HTTP_NOT_FOUND = 404
    HTTP_INTERNAL_ERROR = 500
    HTTP_SERVICE_UNAVAILABLE = 503
    
    # Timeouts (seconds)
    TIMEOUT_SHORT = 5
    TIMEOUT_MEDIUM = 30
    TIMEOUT_LONG = 60
    TIMEOUT_VERY_LONG = 300
    TIMEOUT_TEST_DEFAULT = 3600
    
    # Retry Settings
    RETRY_MAX_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 5
    RETRY_BACKOFF_MULTIPLIER = 2
    
    # Display Separators
    SEPARATOR_WIDTH = 80
    SEPARATOR_CHAR = "="
    SEPARATOR_LINE = SEPARATOR_CHAR * SEPARATOR_WIDTH

# Convenience aliases (backward compatibility)
SEPARATOR_LINE = Constants.SEPARATOR_LINE
DEFAULT_TIMEOUT = Constants.DEFAULT_TIMEOUT