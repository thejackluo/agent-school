"""
Pytest configuration and fixtures for Agent School tests
"""

import pytest
import os
from pathlib import Path


def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption(
        "--run-llm-tests",
        action="store_true",
        default=False,
        help="Run tests that make expensive LLM API calls"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def temp_cache_dir(tmp_path_factory):
    """Create a temporary cache directory for tests"""
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing without API calls"""
    return {
        "workflow_code": '''
def fetch_events():
    """Fetch events from platform"""
    return []
''',
        "plan": [
            {
                "step_number": 1,
                "action": "search",
                "description": "Search for events",
                "required_data": [],
                "output_data": ["events"],
                "method": "browser"
            }
        ]
    }
