"""
Tests for Configuration Module
"""

import pytest
import os
from agent_school.config import Config


class TestConfig:
    """Test configuration loading and validation"""

    def test_config_has_api_keys(self):
        """Test that API keys are loaded from environment"""
        # At least one should be set (from .env file)
        assert Config.OPENAI_API_KEY or Config.ANTHROPIC_API_KEY

    def test_config_default_provider(self):
        """Test default LLM provider is set"""
        assert Config.DEFAULT_LLM_PROVIDER in ["anthropic", "openai"]

    def test_config_cache_dir(self):
        """Test cache directory is configured"""
        assert Config.CACHE_DIR is not None
        assert isinstance(Config.CACHE_DIR, type(Config.CACHE_DIR))

    def test_config_validate(self):
        """Test configuration validation"""
        # Should not raise if .env is properly configured
        try:
            is_valid = Config.validate()
            assert is_valid is True
        except ValueError as e:
            pytest.fail(f"Config validation failed: {e}")

    def test_config_get_api_key(self):
        """Test getting API key for configured provider"""
        try:
            key = Config.get_api_key(Config.DEFAULT_LLM_PROVIDER)
            assert key is not None
            assert len(key) > 0
        except ValueError:
            pytest.fail("API key not configured for default provider")

    def test_config_ensure_cache_dir(self):
        """Test cache directory creation"""
        cache_dir = Config.ensure_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()
