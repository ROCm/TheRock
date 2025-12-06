"""Configuration helper for loading, validation, and logging setup."""

from pathlib import Path
from typing import Optional

from ..logger import log, set_log_level
from .config_parser import ConfigParser
from ..constants import Constants


class ConfigHelper:
    """Configuration helper for file discovery, loading, validation, and logging setup."""
    
    @staticmethod
    def find_config_file(config_file: Optional[str] = None,
                        default_name: str = Constants.DEFAULT_CONFIG_FILE) -> str:
        """Find configuration file in standard locations.
        
        Search order:
        1. Provided config_file path (if given)
        2. Executor directory (framework parent)
        3. Current working directory
        4. Return default name (will be created if doesn't exist)
        
        Args:
            config_file: Explicit config file path (optional)
            default_name: Default config filename to use
        
        Returns:
            Path to configuration file
        """
        # Use provided path if given
        if config_file:
            return config_file
        
        # Try executor directory
        executor_dir = Path(__file__).parent.parent
        config_path = executor_dir / default_name
        if config_path.exists():
            return str(config_path)
        
        # Try current directory
        config_path = Path.cwd() / default_name
        if config_path.exists():
            return str(config_path)
        
        # Return default name (will be created if doesn't exist)
        return default_name
    
    @staticmethod
    def load_config(config_file: str,
                   validate: bool = True,
                   required: bool = False) -> Optional[ConfigParser]:
        """Load configuration file with error handling.
        
        Args:
            config_file: Path to configuration file
            validate: Validate configuration against schema
            required: Raise exception if config file not found
        
        Returns:
            ConfigParser instance or None if file not found and not required
            
        Raises:
            FileNotFoundError: If config file not found and required=True
        """
        try:
            if Path(config_file).exists():
                config = ConfigParser(config_file, validate=validate)
                log.debug(f"Loaded config: {config_file}")
                return config
            else:
                if required:
                    raise FileNotFoundError(f"Config file not found: {config_file}")
                log.warning(f"Config file not found: {config_file}")
                log.warning("Using default configuration")
                return None
        except Exception as e:
            if required:
                raise
            log.warning(f"Failed to load config: {e}")
            log.warning("Using default configuration")
            return None
    
    @staticmethod
    def configure_logging(config: Optional[ConfigParser],
                         default_level: str = 'INFO'):
        """Configure logging from configuration.
        
        Sets log level and optionally enables file logging with rotation.
        
        Args:
            config: ConfigParser instance (None = use defaults)
            default_level: Default log level if config not available
        """
        if not config:
            set_log_level(default_level)
            return
        
        try:
            config_data = config.getConfig()
            core_config = config_data.get('Config', {}).get('Core', {})
            
            # Set log level
            log_level = core_config.get('LogLevel', default_level)
            set_log_level(log_level)
            log.debug(f"Log level set to: {log_level}")
            
            # Enable file logging if configured
            log_to_file = core_config.get('LogToFile', False)
            if log_to_file:
                from datetime import datetime
                import os
                
                log_dir = core_config.get('LogDirectory', './logs')
                max_size_mb = core_config.get('LogMaxSizeMB', 10)
                backup_count = core_config.get('LogBackupCount', 5)
                
                # Create log filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = os.path.join(log_dir, f'test_executor_{timestamp}.log')
                
                # Enable file logging
                log.enable_file_logging(
                    log_file,
                    max_bytes=max_size_mb * 1024 * 1024,
                    backup_count=backup_count
                )
                log.info(f"File logging enabled: {log_file}")
                
        except Exception as e:
            log.warning(f"Failed to configure logging: {e}")
            log.warning("Using default logging configuration")
    
    @staticmethod
    def get_api_config(config: Optional[ConfigParser]) -> dict:
        """Get API configuration from config.
        
        Args:
            config: ConfigParser instance
        
        Returns:
            Dictionary with API configuration:
            - enabled: bool
            - url: str
            - api_key: str
            - timeout: int
            - max_retries: int
            - retry_delay: int
        """
        if not config:
            return {
                'enabled': False,
                'url': '',
                'fallback_url': '',
                'api_key': '',
                'timeout': Constants.DEFAULT_API_TIMEOUT,
                'max_retries': Constants.DEFAULT_API_MAX_RETRIES,
                'retry_delay': Constants.DEFAULT_API_RETRY_DELAY
            }
        
        try:
            config_data = config.getConfig()
            core_config = config_data.get('Config', {}).get('Core', {})
            api_config = core_config.get('ResultsAPI', {})
            
            return {
                'enabled': core_config.get('UploadTestResultsToAPI', False),
                'url': api_config.get('URL', ''),
                'fallback_url': api_config.get('FallbackURL', ''),
                'api_key': api_config.get('APIKey', ''),
                'timeout': api_config.get('Timeout', Constants.DEFAULT_API_TIMEOUT),
                'max_retries': api_config.get('MaxRetries', Constants.DEFAULT_API_MAX_RETRIES),
                'retry_delay': api_config.get('RetryDelay', Constants.DEFAULT_API_RETRY_DELAY)
            }
        except Exception as e:
            log.debug(f"Failed to get API config: {e}")
            return {
                'enabled': False,
                'url': '',
                'fallback_url': '',
                'api_key': '',
                'timeout': Constants.DEFAULT_API_TIMEOUT,
                'max_retries': Constants.DEFAULT_API_MAX_RETRIES,
                'retry_delay': Constants.DEFAULT_API_RETRY_DELAY
            }
    
    @staticmethod
    def get_execution_label(config: Optional[ConfigParser]) -> str:
        """Get execution label from configuration.
        
        Args:
            config: ConfigParser instance
        
        Returns:
            Execution label string (empty if not configured)
        """
        if not config:
            return ''
        
        try:
            config_data = config.getConfig()
            return config_data.get('Config', {}).get('Core', {}).get('ExecutionLabel', '')
        except Exception:
            return ''
    
    @staticmethod
    def get_ci_group(config: Optional[ConfigParser]) -> str:
        """Get CI group from configuration.
        
        Args:
            config: ConfigParser instance
        
        Returns:
            CI group string (default: therock_pr)
        """
        if not config:
            return 'therock_pr'
        
        try:
            config_data = config.getConfig()
            return config_data.get('Config', {}).get('Core', {}).get('CIGroup', 'therock_pr')
        except Exception:
            return 'therock_pr'

    @staticmethod
    def get_deployed_user(config: Optional[ConfigParser]) -> str:
        """Get deployed user from configuration.
        
        Args:
            config: ConfigParser instance
        
        Returns:
            Deployed user string (empty if not configured)
        """
        if not config:
            return ''
        
        try:
            config_data = config.getConfig()
            return config_data.get('Config', {}).get('Core', {}).get('DeployedUser', '')
        except Exception:
            return ''
