"""
Plan Generator - Converts natural language queries into structured execution plans.

This module uses LLMs to parse user queries and generate step-by-step plans
for extracting event data from various sources.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.core.llm_client import LLMClient, LLMProvider

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_number: int = Field(..., description="Step sequence number")
    action: str = Field(..., description="Action to perform (e.g., 'search_events', 'filter_results')")
    method: str = Field(..., description="Execution method: 'api' or 'browser'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for this step")
    fallback: Optional[str] = Field(None, description="Fallback action if this step fails")


class ExecutionPlan(BaseModel):
    """Complete execution plan for a user query."""
    task_name: str = Field(..., description="Unique name for this task")
    intent: str = Field(..., description="User's intent/goal")
    extracted_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters (location, category, keywords, etc.)"
    )
    steps: List[PlanStep] = Field(..., description="Ordered list of execution steps")
    estimated_duration_seconds: int = Field(..., description="Estimated execution time")
    fallback_strategy: str = Field(..., description="Overall fallback if primary plan fails")
    data_source: str = Field(..., description="Primary data source (e.g., 'luma_api', 'luma_browser')")


class PlanGenerator:
    """
    Generates structured execution plans from natural language queries.

    Uses LLMs to analyze user intent and create step-by-step plans for
    extracting event data.

    Examples:
        >>> generator = PlanGenerator()
        >>> plan = await generator.generate_plan(
        ...     "Find hip-hop parties in SF within 5 miles"
        ... )
        >>> print(plan.task_name)
        'fetch_sf_hiphop_parties'
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: LLMProvider = LLMProvider.OPENAI
    ):
        """
        Initialize plan generator.

        Args:
            llm_client: Optional pre-configured LLM client
            provider: LLM provider to use if client not provided
        """
        self.llm_client = llm_client or LLMClient(provider=provider)
        logger.info("Initialized PlanGenerator")

    def _build_plan_prompt(self, query: str) -> str:
        """Build the prompt for plan generation."""
        schema = ExecutionPlan.schema_json(indent=2)

        prompt = f"""You are a workflow planning expert for web scraping and API integration.

Your task is to analyze a user query about finding events and create a detailed execution plan.

USER QUERY: "{query}"

AVAILABLE DATA SOURCES:
1. Luma API (requires API key, may not be available for all queries)
2. Luma Website (https://lu.ma/explore) - always available, requires browser automation

YOUR PLAN SHOULD INCLUDE:
1. **Intent Analysis**: What is the user trying to find?
2. **Parameter Extraction**: Extract location, radius, category, date range, keywords, filters
3. **Data Source Selection**: Choose 'api' or 'browser' based on availability
4. **Step-by-Step Workflow**: Create numbered steps to accomplish the task
5. **Fallback Strategy**: What to do if primary method fails
6. **Output Format**: Structure for results

PARAMETER EXTRACTION GUIDELINES:
- Location: City, state, address, or "current location"
- Radius: Distance in miles (default: 5 miles if not specified)
- Category: Event type (music, tech, sports, networking, etc.)
- Keywords: Specific terms to search for
- Date Range: When events should occur (default: next 30 days)
- Filters: Attendee count, price range, venue type, etc.

STEP ACTION TYPES:
- validate_location: Verify and geocode location
- search_events: Query event database/website
- apply_filters: Filter results by criteria
- extract_data: Parse event details
- format_results: Structure output data
- handle_pagination: Navigate multiple pages of results

EXECUTION METHODS:
- 'api': Use when Luma API is available (faster, more reliable)
- 'browser': Use Playwright for web scraping (always works, slower)

Generate a JSON plan following this exact schema:
{schema}

IMPORTANT: Return ONLY valid JSON, no additional text or explanation.
"""
        return prompt

    def _build_system_message(self) -> str:
        """Build system message for LLM."""
        return """You are an expert workflow planning system. You analyze user queries and generate precise, executable plans in JSON format. You always return valid JSON that matches the provided schema exactly. You are detail-oriented and consider edge cases and fallback strategies."""

    async def generate_plan(
        self,
        query: str,
        temperature: float = 0.7
    ) -> ExecutionPlan:
        """
        Generate execution plan from natural language query.

        Args:
            query: Natural language query from user
            temperature: LLM sampling temperature (0.0-1.0)

        Returns:
            ExecutionPlan object with structured plan

        Raises:
            ValueError: If query is empty or plan generation fails
            json.JSONDecodeError: If LLM returns invalid JSON
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Generating plan for query: '{query}'")

        # Generate plan using LLM
        prompt = self._build_plan_prompt(query)
        system = self._build_system_message()

        response = await self.llm_client.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=2000
        )

        # Parse JSON response
        try:
            # Try to extract JSON if LLM added extra text
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            plan_dict = json.loads(response)
            plan = ExecutionPlan(**plan_dict)

            logger.info(
                f"Generated plan '{plan.task_name}' with {len(plan.steps)} steps "
                f"using {plan.data_source}"
            )

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response[:500]}...")
            raise ValueError(f"LLM returned invalid JSON: {str(e)}")

    async def validate_plan(self, plan: ExecutionPlan) -> bool:
        """
        Validate that a plan is complete and executable.

        Args:
            plan: ExecutionPlan to validate

        Returns:
            True if plan is valid, False otherwise
        """
        # Check required fields
        if not plan.task_name or not plan.intent:
            logger.warning("Plan missing task_name or intent")
            return False

        if not plan.steps or len(plan.steps) == 0:
            logger.warning("Plan has no steps")
            return False

        # Check step numbering is sequential
        for i, step in enumerate(plan.steps, start=1):
            if step.step_number != i:
                logger.warning(f"Step numbering mismatch: expected {i}, got {step.step_number}")
                return False

            # Check method is valid
            if step.method not in ["api", "browser"]:
                logger.warning(f"Invalid method '{step.method}' in step {i}")
                return False

        logger.info(f"Plan '{plan.task_name}' validation passed")
        return True

    async def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Optimize plan by combining steps, removing redundancy, etc.

        Args:
            plan: ExecutionPlan to optimize

        Returns:
            Optimized ExecutionPlan
        """
        # For now, just return the plan as-is
        # Future: implement step merging, parallel execution detection, etc.
        logger.info(f"Plan '{plan.task_name}' optimization skipped (not implemented)")
        return plan


# Example usage
async def main():
    """Example usage of PlanGenerator."""
    import asyncio

    generator = PlanGenerator(provider=LLMProvider.OPENAI)

    # Test queries
    test_queries = [
        "Find hip-hop parties in SF within 5 miles",
        "YC co-founder events",
        "Tech conferences in SF next month under $100",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        plan = await generator.generate_plan(query)
        is_valid = await generator.validate_plan(plan)

        print(f"\nTask Name: {plan.task_name}")
        print(f"Intent: {plan.intent}")
        print(f"Data Source: {plan.data_source}")
        print(f"Estimated Duration: {plan.estimated_duration_seconds}s")
        print(f"Valid: {is_valid}")
        print(f"\nExtracted Parameters:")
        for key, value in plan.extracted_params.items():
            print(f"  {key}: {value}")
        print(f"\nSteps:")
        for step in plan.steps:
            print(f"  {step.step_number}. {step.action} ({step.method})")
            if step.params:
                print(f"     Params: {step.params}")
        print(f"\nFallback Strategy: {plan.fallback_strategy}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
