"""
Plan Generator (File 2)

This module generates step-by-step execution plans for data extraction tasks.
It analyzes user queries and creates detailed plans that break down complex
tasks into manageable steps.

For example:
- User: "Find hip-hop parties in SF with hot girls"
- Plan: [1. Search Luma for events in SF, 2. Filter by 'party' and 'hip-hop' keywords,
         3. Check event descriptions for relevant terms, 4. Return top 10 matches]
"""

import os
import json
from typing import List, Dict, Any, Literal, Optional
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Step:
    """Represents a single step in an execution plan."""

    def __init__(
        self,
        step_number: int,
        action: str,
        description: str,
        required_data: List[str],
        output_data: List[str],
        method: Literal["api", "browser", "processing"] = "processing"
    ):
        self.step_number = step_number
        self.action = action
        self.description = description
        self.required_data = required_data
        self.output_data = output_data
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "description": self.description,
            "required_data": self.required_data,
            "output_data": self.output_data,
            "method": self.method
        }


class ExecutionPlan:
    """Represents a complete execution plan for a task."""

    def __init__(self, task_description: str, steps: List[Step]):
        self.task_description = task_description
        self.steps = steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "task_description": self.task_description,
            "steps": [step.to_dict() for step in self.steps]
        }

    def to_json(self) -> str:
        """Convert plan to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """Human-readable plan representation."""
        lines = [
            f"Execution Plan: {self.task_description}",
            "=" * 80,
            ""
        ]

        for step in self.steps:
            lines.extend([
                f"Step {step.step_number}: {step.action}",
                f"  Description: {step.description}",
                f"  Method: {step.method}",
                f"  Requires: {', '.join(step.required_data) if step.required_data else 'None'}",
                f"  Outputs: {', '.join(step.output_data)}",
                ""
            ])

        return "\n".join(lines)


class PlanGenerator:
    """
    Generates step-by-step execution plans for data extraction tasks.

    This class analyzes user queries and creates detailed, actionable plans
    that can be executed by the workflow generator.
    """

    def __init__(
        self,
        llm_provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None
    ):
        """
        Initialize the plan generator.

        Args:
            llm_provider: Which LLM provider to use
            model: Specific model to use (defaults to best available)
        """
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o"

    def generate_plan(
        self,
        user_query: str,
        target_platform: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Generate an execution plan from a user query.

        Args:
            user_query: Natural language description of what to extract
            target_platform: The platform to extract from (e.g., "Luma", "Eventbrite")
            context: Additional context (available APIs, rate limits, etc.)

        Returns:
            ExecutionPlan object with detailed steps
        """
        system_prompt = self._build_plan_system_prompt()
        user_prompt = self._build_plan_user_prompt(user_query, target_platform, context or {})

        # Get the LLM response
        response = self._call_llm(system_prompt, user_prompt)

        # Parse the response into an ExecutionPlan
        plan = self._parse_plan_response(response, user_query)

        return plan

    def _build_plan_system_prompt(self) -> str:
        """Build the system prompt for plan generation."""
        return """You are an expert at breaking down data extraction tasks into detailed execution plans.

Given a user's request, you must create a step-by-step plan that:

1. **Identifies what data to extract** (events, profiles, listings, etc.)
2. **Determines the extraction method** (API call, browser automation, data processing)
3. **Specifies filtering criteria** (location, keywords, date ranges)
4. **Defines data transformation steps** (parsing, cleaning, formatting)
5. **Handles edge cases** (no results, rate limits, authentication)

For each step, provide:
- **action**: A short name for the step (e.g., "search_events", "filter_by_location")
- **description**: A detailed explanation of what this step does
- **required_data**: What data this step needs (from previous steps or user input)
- **output_data**: What data this step produces
- **method**: How to execute this step ("api", "browser", or "processing")

Output your response as a JSON array of steps. Example format:

```json
[
  {
    "step_number": 1,
    "action": "search_events",
    "description": "Search for events on Luma using browser automation with the query 'hip-hop party San Francisco'",
    "required_data": ["search_query", "location"],
    "output_data": ["raw_event_list"],
    "method": "browser"
  },
  {
    "step_number": 2,
    "action": "filter_by_distance",
    "description": "Filter events to only those within 5 miles of San Francisco downtown",
    "required_data": ["raw_event_list", "user_location", "radius"],
    "output_data": ["filtered_events"],
    "method": "processing"
  }
]
```

Respond ONLY with the JSON array, no other text."""

    def _build_plan_user_prompt(
        self,
        user_query: str,
        target_platform: str,
        context: Dict[str, Any]
    ) -> str:
        """Build the user prompt with task details."""
        prompt_parts = [
            f"User Query: {user_query}",
            f"Target Platform: {target_platform}",
            ""
        ]

        if context:
            prompt_parts.append("Context:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Generate a detailed execution plan as a JSON array of steps.",
            "Consider:",
            "1. How to access the data (API vs browser automation)",
            "2. What filters to apply (location, keywords, dates)",
            "3. How to process and format the results",
            "4. Error handling and edge cases",
            "",
            "JSON array:"
        ])

        return "\n".join(prompt_parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM to generate a plan."""
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        else:  # openai
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.2
            )
            return response.choices[0].message.content

    def _parse_plan_response(self, response: str, task_description: str) -> ExecutionPlan:
        """Parse the LLM response into an ExecutionPlan."""
        # Extract JSON from response (in case there's extra text)
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        try:
            steps_data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse plan response as JSON: {e}\n\nResponse:\n{response}")

        steps = []
        for step_data in steps_data:
            step = Step(
                step_number=step_data["step_number"],
                action=step_data["action"],
                description=step_data["description"],
                required_data=step_data["required_data"],
                output_data=step_data["output_data"],
                method=step_data.get("method", "processing")
            )
            steps.append(step)

        return ExecutionPlan(task_description=task_description, steps=steps)

    def save_plan(self, plan: ExecutionPlan, output_path: str) -> None:
        """Save the plan to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(plan.to_json())
        print(f"✓ Plan saved to: {output_path}")


def demo():
    """Demo the plan generator with Luma examples."""
    generator = PlanGenerator(llm_provider="anthropic")

    # Example 1: Party events
    print("Example 1: Hip-hop parties in SF")
    print("-" * 80)
    plan1 = generator.generate_plan(
        user_query="Find hip-hop parties in SF within 5 miles that are happening this weekend",
        target_platform="Luma",
        context={
            "has_api": False,
            "requires_login": False,
            "location": "San Francisco, CA"
        }
    )
    print(plan1)
    print()

    # Example 2: Professional events
    print("Example 2: YC Co-founder events")
    print("-" * 80)
    plan2 = generator.generate_plan(
        user_query="Find YC co-founder matching events or private networking events",
        target_platform="Luma",
        context={
            "has_api": False,
            "requires_login": True,
            "keywords": ["YC", "Y Combinator", "co-founder", "startup", "networking"]
        }
    )
    print(plan2)


if __name__ == "__main__":
    demo()
