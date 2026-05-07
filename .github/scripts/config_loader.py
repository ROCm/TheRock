import json
import sys
import logging
from typing import List
from repo_config_model import RepoConfig, RepoEntry

logger = logging.getLogger(__name__)


def load_repo_config(config_path: str) -> List[RepoEntry]:
    """Load and validate repository config from JSON using Pydantic."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = RepoConfig(**data)
        return config.repositories
    except Exception as e:
        logger.error(f"Failed to load or validate config file '{config_path}': {e}")
        sys.exit(1)
