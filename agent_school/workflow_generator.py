"""
Workflow Generator (File 1)

This module uses LLMs to generate deterministic workflows that can:
1. Call APIs directly (when API keys are available)
2. Use browser automation (when scraping is needed)
3. Handle various data extraction scenarios

The workflow generator analyzes the target (e.g., Luma events) and creates
executable code that can reliably fetch data across different devices/environments.
"""

import os
from typing import Literal, Optional, Dict, Any
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class WorkflowGenerator:
    """
    Generates executable workflow code using LLMs.

    This class takes a high-level description of a data extraction task
    and generates Python code that can be executed to perform that task
    deterministically.
    """

    def __init__(
        self,
        llm_provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None
    ):
        """
        Initialize the workflow generator.

        Args:
            llm_provider: Which LLM provider to use ("anthropic" or "openai")
            model: Specific model to use (defaults to best available)
        """
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o"

    def generate_workflow(
        self,
        task_description: str,
        target_website: str,
        has_api: bool = False,
        api_documentation: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a workflow for extracting data.

        Args:
            task_description: What data to extract (e.g., "Fetch hip-hop party events in SF")
            target_website: The target website/platform (e.g., "lu.ma")
            has_api: Whether the target has a usable API
            api_documentation: Documentation for the API (if available)
            constraints: Additional constraints (device types, rate limits, etc.)

        Returns:
            Python code as a string that implements the workflow
        """
        # Build the system prompt for workflow generation
        system_prompt = self._build_workflow_system_prompt()

        # Build the user prompt with specific task details
        user_prompt = self._build_workflow_user_prompt(
            task_description=task_description,
            target_website=target_website,
            has_api=has_api,
            api_documentation=api_documentation,
            constraints=constraints or {}
        )

        # Generate the workflow code
        workflow_code = self._call_llm(system_prompt, user_prompt)

        return workflow_code

    def _build_workflow_system_prompt(self) -> str:
        """Build the system prompt for workflow generation."""
        return """You are an expert at creating deterministic, production-ready data extraction workflows.

Your task is to generate Python code that reliably extracts data from websites or APIs.
The code you generate must be:

1. **Deterministic**: Same inputs always produce same outputs
2. **Robust**: Handle errors gracefully with retries and fallbacks
3. **Well-structured**: Use functions, type hints, and clear documentation
4. **Production-ready**: Include logging, error handling, and validation

You can use:
- requests/httpx for API calls
- playwright for browser automation (when scraping is needed)
- BeautifulSoup/lxml for HTML parsing
- Standard Python libraries

Always include:
- Proper error handling with try/except blocks
- Rate limiting to avoid overwhelming servers
- Clear docstrings explaining what the code does
- Type hints for all functions
- Example usage in comments

Generate ONLY the Python code, no markdown formatting or explanation before/after."""

    def _build_workflow_user_prompt(
        self,
        task_description: str,
        target_website: str,
        has_api: bool,
        api_documentation: Optional[str],
        constraints: Dict[str, Any]
    ) -> str:
        """Build the user prompt with specific task details."""
        prompt_parts = [
            f"Generate a Python workflow to: {task_description}",
            f"Target: {target_website}",
            ""
        ]

        if has_api and api_documentation:
            prompt_parts.extend([
                "This platform has an API. Prefer using the API over scraping.",
                "API Documentation:",
                api_documentation,
                ""
            ])
        else:
            prompt_parts.extend([
                "This platform does NOT have a public API (or it's paid/unavailable).",
                "You must use browser automation (Playwright) to extract the data.",
                ""
            ])

        if constraints:
            prompt_parts.append("Constraints:")
            for key, value in constraints.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Requirements:",
            "1. Create a main function that accepts relevant parameters",
            "2. Return structured data (list of dicts or similar)",
            "3. Include proper error handling and logging",
            "4. Add docstrings with parameter descriptions",
            "5. Make it easy to call from other Python code",
            "",
            "Generate the complete Python code now:"
        ])

        return "\n".join(prompt_parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM to generate workflow code."""
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
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
                max_tokens=4096,
                temperature=0.1  # Lower temperature for more deterministic code
            )
            return response.choices[0].message.content

    def save_workflow(self, workflow_code: str, output_path: str) -> None:
        """
        Save the generated workflow to a file.

        Args:
            workflow_code: The generated Python code
            output_path: Where to save the file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(workflow_code)
        print(f"✓ Workflow saved to: {output_path}")


def demo():
    """Demo the workflow generator with a Luma example."""
    generator = WorkflowGenerator(llm_provider="anthropic")

    workflow = generator.generate_workflow(
        task_description="Fetch events in San Francisco within a 5-mile radius that match specific keywords",
        target_website="lu.ma",
        has_api=False,  # Luma API is paid, so we'll use browser automation
        constraints={
            "location": "San Francisco, CA",
            "radius_miles": 5,
            "search_terms": "hip-hop party, YC events, private events"
        }
    )

    print("Generated Workflow:")
    print("=" * 80)
    print(workflow)
    print("=" * 80)


if __name__ == "__main__":
    demo()
