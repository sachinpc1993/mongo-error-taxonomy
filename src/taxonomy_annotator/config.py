"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Loads and validates YAML configuration for the annotation runner."""

    SUPPORTED_PROVIDERS = {"ollama", "bedrock"}

    @staticmethod
    def load(path: str | Path = "config.yaml") -> Dict[str, Any]:
        """Load a YAML configuration file and validate required fields."""
        load_dotenv()
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        provider = str(config.get("provider", "")).strip().lower()
        if provider not in ConfigLoader.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Expected one of {sorted(ConfigLoader.SUPPORTED_PROVIDERS)}."
            )

        if provider not in config:
            raise ValueError(f"Missing '{provider}' configuration section.")

        return config
