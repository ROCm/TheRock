"""Configuration validation using JSON Schema for early error detection."""

from jsonschema import validate, ValidationError, Draft7Validator
from typing import Dict, Any, Tuple, List


# Configuration schema definition
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "Config": {
            "type": "object",
            "properties": {
                "Core": {
                    "type": "object",
                    "properties": {
                        "LogLevel": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            "description": "Logging level"
                        },
                        "LogToFile": {
                            "type": "boolean",
                            "description": "Enable file logging"
                        },
                        "LogDirectory": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Directory for log files"
                        },
                        "LogMaxSizeMB": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum log file size in MB before rotation"
                        },
                        "LogBackupCount": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Number of backup log files to keep"
                        },
                        "ExecutionLabel": {
                            "type": "string",
                            "description": "Label for this test execution"
                        },
                        "UploadTestResultsToAPI": {
                            "type": "boolean",
                            "description": "Enable API result submission"
                        },
                        "ResultsAPI": {
                            "type": "object",
                            "properties": {
                                "URL": {
                                    "type": "string",
                                    "format": "uri",
                                    "description": "API endpoint URL"
                                },
                                "APIKey": {
                                    "type": "string",
                                    "description": "API authentication key"
                                },
                                "Timeout": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 300,
                                    "description": "API request timeout in seconds"
                                },
                                "MaxRetries": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 10,
                                    "description": "Maximum retry attempts"
                                },
                                "RetryDelay": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 60,
                                    "description": "Delay between retries in seconds"
                                }
                            },
                            "required": ["URL"]
                        }
                    },
                    "required": ["LogLevel"]
                },
                "Results": {
                    "type": "object",
                    "properties": {
                        "OutputDirectory": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Directory for result files"
                        },
                        "SaveJSON": {
                            "type": "boolean",
                            "description": "Save results in JSON format"
                        },
                    },
                    "required": ["OutputDirectory"]
                }
            },
            "required": ["Core", "Results"]
        }
    },
    "required": ["Config"]
}


class ConfigValidator:
    """Configuration validator with schema validation and detailed error reporting."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            validate(instance=config, schema=CONFIG_SCHEMA)
        except ValidationError as e:
            path = '.'.join(str(p) for p in e.path) if e.path else 'root'
            raise ValueError(
                f"Configuration validation failed:\n"
                f"  Location: {path}\n"
                f"  Error: {e.message}\n"
                f"  Schema: {e.schema.get('description', 'N/A')}"
            )
    
    @staticmethod
    def validate_and_report(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration and return detailed errors.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        validator = Draft7Validator(CONFIG_SCHEMA)
        errors = list(validator.iter_errors(config))
        
        if not errors:
            return True, []
        
        error_messages = []
        for error in errors:
            path = '.'.join(str(p) for p in error.path) if error.path else 'root'
            description = error.schema.get('description', '')
            desc_str = f" ({description})" if description else ""
            error_messages.append(f"{path}: {error.message}{desc_str}")
        
        return False, error_messages
    
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Get configuration schema dictionary.
        
        Returns:
            Configuration schema dictionary
        """
        return CONFIG_SCHEMA
    
    @staticmethod
    def print_schema_summary():
        """Print a human-readable summary of the schema."""
        print("Configuration Schema Summary")
        print("=" * 60)
        print()
        
        def print_properties(props: Dict, indent: int = 0):
            """Recursively print properties."""
            for key, value in props.items():
                prefix = "  " * indent
                prop_type = value.get('type', 'unknown')
                desc = value.get('description', '')
                required = value.get('required', [])
                
                print(f"{prefix}{key} ({prop_type})")
                if desc:
                    print(f"{prefix}  → {desc}")
                
                if 'enum' in value:
                    print(f"{prefix}  → Options: {', '.join(value['enum'])}")
                
                if 'minimum' in value or 'maximum' in value:
                    min_val = value.get('minimum', '-∞')
                    max_val = value.get('maximum', '∞')
                    print(f"{prefix}  → Range: {min_val} to {max_val}")
                
                if 'properties' in value:
                    if required:
                        print(f"{prefix}  → Required fields: {', '.join(required)}")
                    print_properties(value['properties'], indent + 1)
                
                print()
        
        config_props = CONFIG_SCHEMA['properties']['Config']['properties']
        print_properties(config_props)


class ConfigurationError(Exception):
    """Configuration validation error."""
    pass
