"""
Configuration module for Agent School

Loads and validates configuration from environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """
    Centralized configuration for Agent School.

    This class provides type-safe access to all configuration values
    and validates that required API keys are present.
    """

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    LUMA_API_KEY: Optional[str] = os.getenv("LUMA_API_KEY")

    # LLM Configuration
    DEFAULT_LLM_PROVIDER: Literal["anthropic", "openai"] = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")  # type: ignore
    DEFAULT_MODEL: Optional[str] = os.getenv("DEFAULT_MODEL")

    # Paths
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./generated_functions"))

    # Debug
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required configuration is missing
        """
        errors = []

        # Check that at least one LLM API key is present
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            errors.append("At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")

        # Check that the default provider has a key
        if cls.DEFAULT_LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY must be set when DEFAULT_LLM_PROVIDER is 'openai'")
        elif cls.DEFAULT_LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY must be set when DEFAULT_LLM_PROVIDER is 'anthropic'")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        return True

    @classmethod
    def get_api_key(cls, provider: Literal["anthropic", "openai"]) -> str:
        """
        Get API key for a specific provider.

        Args:
            provider: The LLM provider

        Returns:
            The API key

        Raises:
            ValueError: If the API key is not set
        """
        if provider == "anthropic":
            if not cls.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set in environment")
            return cls.ANTHROPIC_API_KEY
        elif provider == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in environment")
            return cls.OPENAI_API_KEY
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def ensure_cache_dir(cls) -> Path:
        """Ensure the cache directory exists and return its path."""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls.CACHE_DIR


# Validate configuration on import
try:
    Config.validate()
    if Config.DEBUG:
        print("[OK] Configuration loaded successfully")
except ValueError as e:
    print(f"[WARNING] Configuration Warning: {e}")
    print("         Please check your .env file")
