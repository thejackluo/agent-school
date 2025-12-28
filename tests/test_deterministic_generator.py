"""
Tests for Deterministic Generator

These tests verify that workflows are generated correctly.
"""

import pytest
from agent_school.core.deterministic_generator import DeterministicGenerator
from agent_school.core.registry import Registry


class TestDeterministicGenerator:
    """Test deterministic workflow generation"""

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create temporary registry"""
        return Registry(workflows_dir=str(tmp_path / "workflows"))

    @pytest.fixture
    def generator(self, temp_registry):
        """Create generator with temp registry"""
        return DeterministicGenerator(registry=temp_registry)

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-llm-tests", default=False),
        reason="This test makes LLM API calls"
    )
    def test_generate_workflow_complete(self, generator):
        """Test complete workflow generation"""
        result = generator.generate_workflow(
            name="test_luma_scraper",
            description="Extract events from Luma",
            target_platform="lu.ma",
            constraints={}
        )

        # Check result structure
        assert "code" in result
        assert "method" in result
        assert "input_schema" in result
        assert "output_schema" in result
        assert "path" in result

        # Check code is valid Python
        assert len(result["code"]) > 0
        assert "def " in result["code"]

        # Check method is valid
        assert result["method"] in ["api", "browser"]

        # Check workflow is registered
        workflow = generator.registry.get_workflow("test_luma_scraper")
        assert workflow is not None

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-llm-tests", default=False),
        reason="This test makes LLM API calls"
    )
    def test_api_vs_browser_decision(self, generator):
        """Test that LLM correctly decides API vs browser"""
        # Test with platform that has API
        method_api = generator._decide_method("github.com", "Fetch repositories")
        # GitHub has free API, should prefer it
        assert method_api == "api"

        # Test with platform without API
        method_browser = generator._decide_method("lu.ma", "Scrape events")
        # Luma API is paid, should use browser
        assert method_browser == "browser"

    def test_extract_schemas_from_code(self, generator):
        """Test schema extraction from generated code"""
        sample_code = '''
def fetch_events(
    location: str,
    radius: int = 5,
    keywords: list = None
) -> list[dict]:
    """Fetch events"""
    return []
'''

        input_schema, output_schema = generator._extract_schemas(sample_code)

        # Check input schema
        assert "location" in input_schema
        assert input_schema["location"] == "string"
        assert "radius" in input_schema
        assert input_schema["radius"] == "int"

        # Check output schema
        assert "type" in output_schema
        assert output_schema["type"] == "list"
