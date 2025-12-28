"""
Tests for Plan Generator
"""

import pytest
from agent_school.plan_generator import PlanGenerator, ExecutionPlan, Step


class TestPlanGenerator:
    """Test suite for PlanGenerator"""

    @pytest.fixture
    def plan_generator(self):
        """Create a PlanGenerator instance for testing"""
        return PlanGenerator(llm_provider="anthropic")

    def test_plan_generator_initialization(self, plan_generator):
        """Test that PlanGenerator initializes correctly"""
        assert plan_generator is not None
        assert plan_generator.llm_provider == "anthropic"
        assert plan_generator.client is not None

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-llm-tests", default=False),
        reason="LLM tests are expensive and slow"
    )
    def test_generate_plan(self, plan_generator):
        """Test plan generation for a simple query"""
        plan = plan_generator.generate_plan(
            user_query="Find events in San Francisco",
            target_platform="lu.ma",
            context={"has_api": False}
        )

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) > 0
        assert plan.task_description == "Find events in San Francisco"

    def test_execution_plan_to_dict(self):
        """Test ExecutionPlan serialization"""
        steps = [
            Step(
                step_number=1,
                action="search",
                description="Search for events",
                required_data=["query"],
                output_data=["events"],
                method="browser"
            )
        ]
        plan = ExecutionPlan(
            task_description="Test task",
            steps=steps
        )

        plan_dict = plan.to_dict()
        assert plan_dict["task_description"] == "Test task"
        assert len(plan_dict["steps"]) == 1
        assert plan_dict["steps"][0]["action"] == "search"

    def test_execution_plan_to_json(self):
        """Test ExecutionPlan JSON serialization"""
        steps = [
            Step(
                step_number=1,
                action="search",
                description="Search for events",
                required_data=[],
                output_data=["events"],
                method="browser"
            )
        ]
        plan = ExecutionPlan(
            task_description="Test task",
            steps=steps
        )

        json_str = plan.to_json()
        assert isinstance(json_str, str)
        assert "Test task" in json_str
        assert "search" in json_str


def pytest_addoption(parser):
    """Add command line options for pytest"""
    parser.addoption(
        "--run-llm-tests",
        action="store_true",
        default=False,
        help="Run tests that make LLM API calls"
    )
