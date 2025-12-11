"""YAML configuration parser with schema validation and environment variable expansion."""

import yaml
import os
import re
from pathlib import Path
from typing import Any, Dict
from .config_validator import ConfigValidator, ConfigurationError


class ConfigParser:
    """YAML configuration parser with validation and environment variable expansion."""
    
    # Pattern for environment variable substitution: ${VAR_NAME} or ${VAR_NAME:-default}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self, config_file: str, validate: bool = True, expand_env_vars: bool = True):
        """Initialize config parser.
        
        Args:
            config_file: Path to YAML configuration file
            validate: Whether to validate configuration against schema
            expand_env_vars: Whether to expand environment variables
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigurationError: If configuration is invalid
        """
        self.config_file = Path(config_file)
        self.config_data = {}
        self.root = {}  # Alias for compatibility
        self.validate_on_load = validate
        self.expand_env_vars = expand_env_vars
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        self._load()
    
    def _load(self):
        """Load configuration from file with environment variable expansion and validation."""
        try:
            with open(self.config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Check if config is empty
            if not raw_config:
                raise ConfigurationError(
                    f"Configuration file is empty: {self.config_file}\n"
                    "Please ensure the file contains valid YAML configuration."
                )
            
            # Expand environment variables if enabled
            if self.expand_env_vars:
                self.config_data = self._expand_env_vars(raw_config)
            else:
                self.config_data = raw_config
            
            self.root = self.config_data  # Alias for compatibility
            
            # Validate configuration if enabled
            if self.validate_on_load:
                try:
                    ConfigValidator.validate_config(self.config_data)
                except ValueError as e:
                    raise ConfigurationError(f"Configuration validation failed: {e}")
                    
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {e}")
        except ConfigurationError:
            raise
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def get_config(self) -> Dict:
        """Get full configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config_data
    
    def getConfig(self) -> Dict:
        """Get full configuration dictionary (camelCase alias).
        
        Returns:
            Configuration dictionary
        """
        return self.get_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'Config.Core.LogLevel')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config_data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration object.
        
        Args:
            obj: Configuration object (dict, list, str, etc.)
            
        Returns:
            Configuration with expanded environment variables
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        
        elif isinstance(obj, str):
            return self._expand_string(obj)
        
        else:
            return obj
    
    def _expand_string(self, value: str) -> str:
        """Expand environment variables in string (${VAR} or ${VAR:-default}).
        
        Args:
            value: String potentially containing environment variables
            
        Returns:
            String with expanded environment variables
            
        Raises:
            ConfigurationError: If required environment variable is not set
        """
        def replacer(match):
            var_expr = match.group(1)
            
            # Check for default value syntax: VAR:-default
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                var_name = var_name.strip()
                default = default.strip()
                return os.environ.get(var_name, default)
            else:
                var_name = var_expr.strip()
                if var_name not in os.environ:
                    raise ConfigurationError(
                        f"Required environment variable not set: {var_name}\n"
                        f"Set it with: export {var_name}=<value>"
                    )
                return os.environ[var_name]
        
        return self.ENV_VAR_PATTERN.sub(replacer, value)
