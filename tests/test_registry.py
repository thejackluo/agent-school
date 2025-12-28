"""
Tests for Registry Module
"""

import pytest
import json
from pathlib import Path
from agent_school.core.registry import Registry


class TestRegistry:
    """Test registry functionality"""

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create a temporary registry for testing"""
        return Registry(workflows_dir=str(tmp_path / "workflows"))

    def test_registry_initialization(self, temp_registry):
        """Test registry initializes correctly"""
        assert temp_registry is not None
        assert temp_registry.deterministic_dir.exists()
        assert temp_registry.agent_plans_dir.exists()

    def test_register_workflow(self, temp_registry):
        """Test registering a deterministic workflow"""
        temp_registry.register_deterministic_workflow(
            name="test_workflow",
            description="Test workflow",
            input_schema={"location": "str", "radius": "int"},
            output_schema={"type": "list"},
            method="browser"
        )

        workflow = temp_registry.get_workflow("test_workflow")
        assert workflow is not None
        assert workflow["name"] == "test_workflow"
        assert workflow["method"] == "browser"
        assert workflow["input_schema"]["location"] == "str"

    def test_register_agent_plan(self, temp_registry):
        """Test registering an agent plan"""
        # First register a workflow
        temp_registry.register_deterministic_workflow(
            name="test_workflow",
            description="Test",
            input_schema={},
            output_schema={},
            method="api"
        )

        # Then register a plan that uses it
        temp_registry.register_agent_plan(
            name="test_plan",
            description="Test plan",
            uses_workflows=["test_workflow"],
            steps=[
                {"id": 1, "type": "llm", "action": "parse", "output_var": "params"},
                {"id": 2, "type": "deterministic", "workflow": "test_workflow",
                 "input_from": "params", "output_var": "result"}
            ]
        )

        plan = temp_registry.get_plan("test_plan")
        assert plan is not None
        assert plan["name"] == "test_plan"
        assert "test_workflow" in plan["uses_workflows"]

    def test_list_workflows(self, temp_registry):
        """Test listing workflows"""
        # Register some workflows
        for i in range(3):
            temp_registry.register_deterministic_workflow(
                name=f"workflow_{i}",
                description=f"Workflow {i}",
                input_schema={},
                output_schema={},
                method="api"
            )

        workflows = temp_registry.list_workflows()
        assert len(workflows) == 3

    def test_search_workflows(self, temp_registry):
        """Test searching workflows"""
        temp_registry.register_deterministic_workflow(
            name="luma_scraper",
            description="Scrape events from Luma",
            input_schema={},
            output_schema={},
            method="browser"
        )

        temp_registry.register_deterministic_workflow(
            name="twitter_api",
            description="Fetch tweets from Twitter API",
            input_schema={},
            output_schema={},
            method="api"
        )

        # Search for "luma"
        results = temp_registry.search_workflows("luma")
        assert len(results) == 1
        assert results[0]["name"] == "luma_scraper"

        # Search for "api"
        results = temp_registry.search_workflows("api")
        assert len(results) == 1
        assert results[0]["name"] == "twitter_api"

    def test_unregister_workflow(self, temp_registry):
        """Test unregistering a workflow"""
        temp_registry.register_deterministic_workflow(
            name="temp_workflow",
            description="Temporary",
            input_schema={},
            output_schema={},
            method="api"
        )

        assert temp_registry.get_workflow("temp_workflow") is not None

        temp_registry.unregister_workflow("temp_workflow")

        assert temp_registry.get_workflow("temp_workflow") is None

    def test_stats(self, temp_registry):
        """Test registry statistics"""
        # Register some items
        temp_registry.register_deterministic_workflow(
            name="workflow_1",
            description="Test",
            input_schema={},
            output_schema={},
            method="api"
        )

        temp_registry.register_agent_plan(
            name="plan_1",
            description="Test",
            uses_workflows=["workflow_1"],
            steps=[]
        )

        stats = temp_registry.stats()
        assert stats["deterministic_workflows"] == 1
        assert stats["agent_plans"] == 1
