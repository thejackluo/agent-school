"""
Integration Tests

Tests the full three-layer architecture working together.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_school.core.registry import Registry
from agent_school.core.executor import Executor
from agent_school.core.router import Router


class TestIntegration:
    """Test full system integration"""

    @pytest.fixture
    def workflows_dir(self):
        """Get workflows directory"""
        return project_root / "workflows"

    @pytest.fixture
    def registry(self, workflows_dir):
        """Create registry with real workflows"""
        return Registry(workflows_dir=str(workflows_dir))

    @pytest.fixture
    def executor(self, registry):
        """Create executor"""
        return Executor(registry=registry)

    @pytest.fixture
    def router(self, registry):
        """Create router"""
        return Router(registry=registry)

    def test_luma_workflow_exists(self, registry):
        """Test that Luma scraper workflow is registered"""
        workflow = registry.get_workflow("luma_scraper")
        assert workflow is not None, "Luma scraper workflow should exist"
        assert workflow["name"] == "luma_scraper"
        assert workflow["method"] == "browser"
        assert workflow["type"] == "deterministic"

    def test_luma_workflow_has_schemas(self, workflows_dir):
        """Test that Luma workflow has all required files"""
        luma_dir = workflows_dir / "deterministic" / "luma_scraper"
        assert luma_dir.exists(), "Luma scraper directory should exist"

        required_files = ["workflow.py", "input_schema.json", "output_schema.json", "metadata.json"]
        for filename in required_files:
            file_path = luma_dir / filename
            assert file_path.exists(), f"{filename} should exist"

    def test_agent_plan_exists(self, registry):
        """Test that personalized event finder plan exists"""
        plan = registry.get_plan("personalized_event_finder")
        assert plan is not None, "Personalized event finder plan should exist"
        assert plan["name"] == "personalized_event_finder"
        assert "luma_scraper" in plan["uses_workflows"]
        assert len(plan["steps"]) == 5

    def test_agent_plan_structure(self, registry):
        """Test agent plan has correct structure"""
        plan = registry.get_plan("personalized_event_finder")

        # Check steps
        steps = plan["steps"]
        assert len(steps) == 5

        # Step 1: LLM parse
        assert steps[0]["type"] == "llm"
        assert steps[0]["output_var"] == "search_params"

        # Step 2: Deterministic workflow
        assert steps[1]["type"] == "deterministic"
        assert steps[1]["workflow"] == "luma_scraper"
        assert steps[1]["output_var"] == "raw_events"

        # Step 3-5: LLM steps
        assert steps[2]["type"] == "llm"
        assert steps[3]["type"] == "llm"
        assert steps[4]["type"] == "llm"
        assert steps[4]["output_var"] == "final_response"

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-llm-tests", default=False),
        reason="This test makes LLM API calls and browser automation"
    )
    def test_full_workflow_execution(self, executor):
        """Test executing the full agent plan (requires LLM + browser)"""
        # This is an expensive test - requires:
        # 1. Browser automation (Playwright)
        # 2. LLM API calls (4 calls)
        # Only run with --run-llm-tests flag

        result = executor.execute_plan(
            plan_name="personalized_event_finder",
            user_input="Find tech startup events in San Francisco"
        )

        # Check result structure
        assert "final_response" in result
        assert isinstance(result["final_response"], str)
        assert len(result["final_response"]) > 0

        # Check intermediate results exist
        assert "search_params" in result
        assert "raw_events" in result
        assert "filtered_events" in result
        assert "ranked_events" in result

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-llm-tests", default=False),
        reason="This test makes LLM API calls"
    )
    def test_router_integration(self, router):
        """Test router can detect and route to agent plan"""
        # Test intent detection
        result = router.route("Find hip-hop parties in SF")

        assert result is not None
        assert "intent" in result or "response" in result

    def test_registry_lists_all_workflows(self, registry):
        """Test registry can list all workflows"""
        workflows = registry.list_workflows()
        assert len(workflows) > 0

        # Check that our example workflow is in the list
        workflow_names = [w["name"] for w in workflows]
        assert "luma_scraper" in workflow_names

    def test_registry_lists_all_plans(self, registry):
        """Test registry can list all agent plans"""
        plans = registry.list_plans()
        assert len(plans) > 0

        # Check that our example plan is in the list
        plan_names = [p["name"] for p in plans]
        assert "personalized_event_finder" in plan_names

    def test_registry_search(self, registry):
        """Test registry search functionality"""
        # Search for luma
        results = registry.search_workflows("luma")
        assert len(results) > 0
        assert results[0]["name"] == "luma_scraper"

        # Search for events
        results = registry.search_workflows("events")
        assert len(results) > 0

    def test_workflow_metadata(self, registry):
        """Test workflow metadata is complete"""
        workflow = registry.get_workflow("luma_scraper")

        required_fields = ["name", "type", "method", "input_schema", "output_schema"]
        for field in required_fields:
            assert field in workflow, f"Workflow should have {field} field"

    def test_plan_metadata(self, registry):
        """Test agent plan metadata is complete"""
        plan = registry.get_plan("personalized_event_finder")

        required_fields = ["name", "type", "uses_workflows", "steps"]
        for field in required_fields:
            assert field in plan, f"Plan should have {field} field"


class TestLayerSeparation:
    """Test that layers are properly separated"""

    def test_layer1_has_no_llm_calls(self):
        """Test that Layer 1 workflow has no LLM calls"""
        workflow_path = project_root / "workflows" / "deterministic" / "luma_scraper" / "workflow.py"
        assert workflow_path.exists()

        with open(workflow_path, "r") as f:
            code = f.read()

        # Check that there are no LLM-related imports or calls
        forbidden_patterns = [
            "from openai",
            "import openai",
            "from anthropic",
            "import anthropic",
            "ChatCompletion",
            "claude",
            "gpt-",
        ]

        for pattern in forbidden_patterns:
            assert pattern not in code, f"Layer 1 should not contain '{pattern}'"

    def test_layer2_uses_layer1(self):
        """Test that Layer 2 (agent plan) references Layer 1 (workflow)"""
        plan_path = project_root / "workflows" / "agent_plans" / "personalized_event_finder" / "plan.json"
        assert plan_path.exists()

        import json
        with open(plan_path, "r") as f:
            plan = json.load(f)

        # Check that plan uses the deterministic workflow
        assert "luma_scraper" in plan["uses_workflows"]

        # Check that at least one step is type "deterministic"
        has_deterministic_step = any(
            step.get("type") == "deterministic" for step in plan["steps"]
        )
        assert has_deterministic_step, "Agent plan should have at least one deterministic step"

    def test_layer2_has_llm_steps(self):
        """Test that Layer 2 (agent plan) has LLM orchestration"""
        plan_path = project_root / "workflows" / "agent_plans" / "personalized_event_finder" / "plan.json"

        import json
        with open(plan_path, "r") as f:
            plan = json.load(f)

        # Check that plan has LLM steps
        llm_steps = [step for step in plan["steps"] if step.get("type") == "llm"]
        assert len(llm_steps) > 0, "Agent plan should have LLM steps"

        # Check that LLM steps have prompts
        for step in llm_steps:
            assert "prompt" in step, "LLM steps should have prompts"


class TestDocumentation:
    """Test that documentation is complete"""

    def test_workflow_has_readme(self):
        """Test that workflow has README"""
        readme_path = project_root / "workflows" / "deterministic" / "luma_scraper" / "README.md"
        assert readme_path.exists(), "Workflow should have README.md"

        with open(readme_path, "r") as f:
            content = f.read()

        assert len(content) > 100, "README should have substantial content"
        assert "Deterministic Workflow" in content

    def test_plan_has_readme(self):
        """Test that agent plan has README"""
        readme_path = project_root / "workflows" / "agent_plans" / "personalized_event_finder" / "README.md"
        assert readme_path.exists(), "Agent plan should have README.md"

        with open(readme_path, "r") as f:
            content = f.read()

        assert len(content) > 100, "README should have substantial content"
        assert "Agent Plan" in content
