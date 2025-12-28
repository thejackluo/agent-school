"""
Documentation Generator (File 3)

This module automatically generates comprehensive documentation for workflows
and execution plans. It creates:

1. API-style documentation for generated functions
2. Usage examples with code snippets
3. Parameter descriptions and return types
4. Error handling documentation
5. Integration guides for MCP servers
"""

import os
from typing import Dict, Any, Optional, Literal
from datetime import datetime
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class DocumentationGenerator:
    """
    Generates comprehensive documentation for AI-generated workflows.

    This class analyzes generated code and plans to create user-friendly
    documentation that explains how to use the functions, what parameters
    they accept, and what to expect from them.
    """

    def __init__(
        self,
        llm_provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None
    ):
        """
        Initialize the documentation generator.

        Args:
            llm_provider: Which LLM provider to use
            model: Specific model to use
        """
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o"

    def generate_documentation(
        self,
        function_name: str,
        workflow_code: str,
        execution_plan: Optional[str] = None,
        use_case: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive documentation for a workflow function.

        Args:
            function_name: Name of the function to document
            workflow_code: The actual Python code of the workflow
            execution_plan: The execution plan (from plan_generator)
            use_case: Description of what this function is used for

        Returns:
            Markdown documentation as a string
        """
        system_prompt = self._build_doc_system_prompt()
        user_prompt = self._build_doc_user_prompt(
            function_name, workflow_code, execution_plan, use_case
        )

        documentation = self._call_llm(system_prompt, user_prompt)

        # Add metadata header
        metadata = self._generate_metadata(function_name)
        full_documentation = f"{metadata}\n\n{documentation}"

        return full_documentation

    def generate_mcp_documentation(
        self,
        function_name: str,
        function_description: str,
        parameters: Dict[str, Any],
        returns: Dict[str, Any],
        examples: list[str]
    ) -> str:
        """
        Generate MCP-specific documentation for a function.

        This creates documentation formatted for MCP server integration,
        including JSON schemas for parameters and return types.

        Args:
            function_name: Name of the MCP function
            function_description: What the function does
            parameters: Parameter schema (JSON schema format)
            returns: Return value schema
            examples: List of example usages

        Returns:
            MCP-formatted documentation
        """
        doc_parts = [
            f"# {function_name}",
            "",
            f"**Description:** {function_description}",
            "",
            "## Parameters",
            "",
            "```json",
            self._format_json_schema(parameters),
            "```",
            "",
            "## Returns",
            "",
            "```json",
            self._format_json_schema(returns),
            "```",
            "",
            "## Examples",
            ""
        ]

        for i, example in enumerate(examples, 1):
            doc_parts.extend([
                f"### Example {i}",
                "",
                "```python",
                example,
                "```",
                ""
            ])

        doc_parts.extend([
            "## Integration",
            "",
            "This function is available as an MCP tool. To use it:",
            "",
            "```python",
            "# In your MCP client",
            f"result = mcp_client.call_tool('{function_name}', parameters)",
            "```",
            ""
        ])

        return "\n".join(doc_parts)

    def _build_doc_system_prompt(self) -> str:
        """Build system prompt for documentation generation."""
        return """You are an expert technical writer specializing in API documentation.

Your task is to create clear, comprehensive documentation for AI-generated workflow functions.

The documentation should include:

1. **Overview**: What the function does in 2-3 sentences
2. **Parameters**: Each parameter with:
   - Name and type
   - Description
   - Whether it's required or optional
   - Default value (if any)
   - Example values
3. **Returns**: What the function returns:
   - Type
   - Structure (if complex)
   - Example return value
4. **Usage Examples**: 2-3 realistic examples showing how to use the function
5. **Error Handling**: What errors might occur and how to handle them
6. **Notes**: Any important considerations (rate limits, authentication, etc.)

Format the documentation in clean Markdown with code blocks for examples.
Use a professional, friendly tone that's easy for developers to understand."""

    def _build_doc_user_prompt(
        self,
        function_name: str,
        workflow_code: str,
        execution_plan: Optional[str],
        use_case: Optional[str]
    ) -> str:
        """Build user prompt for documentation generation."""
        prompt_parts = [
            f"Function Name: {function_name}",
            "",
            "Workflow Code:",
            "```python",
            workflow_code,
            "```",
            ""
        ]

        if execution_plan:
            prompt_parts.extend([
                "Execution Plan:",
                execution_plan,
                ""
            ])

        if use_case:
            prompt_parts.extend([
                f"Use Case: {use_case}",
                ""
            ])

        prompt_parts.extend([
            "Generate comprehensive documentation for this function in Markdown format.",
            "Include all sections mentioned in the system prompt."
        ])

        return "\n".join(prompt_parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM to generate documentation."""
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3072,
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
                max_tokens=3072,
                temperature=0.3
            )
            return response.choices[0].message.content

    def _generate_metadata(self, function_name: str) -> str:
        """Generate metadata header for documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""---
Function: {function_name}
Generated: {timestamp}
Generator: Agent School Documentation Generator v0.1.0
---"""

    def _format_json_schema(self, schema: Dict[str, Any]) -> str:
        """Format JSON schema for display."""
        import json
        return json.dumps(schema, indent=2)

    def save_documentation(self, documentation: str, output_path: str) -> None:
        """Save documentation to a markdown file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(documentation)
        print(f"✓ Documentation saved to: {output_path}")


class DocumentationTemplate:
    """Pre-built documentation templates for common patterns."""

    @staticmethod
    def mcp_function_template(
        name: str,
        description: str,
        parameters: list[tuple[str, str, str]]  # (name, type, description)
    ) -> str:
        """Generate a basic MCP function documentation template."""
        param_docs = []
        for param_name, param_type, param_desc in parameters:
            param_docs.append(f"- `{param_name}` ({param_type}): {param_desc}")

        return f"""# {name}

{description}

## Parameters

{chr(10).join(param_docs)}

## Example Usage

```python
result = {name}(
    # Add parameter values here
)
```

## Returns

Returns a structured response containing the extracted data.
"""

    @staticmethod
    def workflow_template(name: str, steps: list[str]) -> str:
        """Generate workflow documentation template."""
        step_docs = []
        for i, step in enumerate(steps, 1):
            step_docs.append(f"{i}. {step}")

        return f"""# {name} Workflow

## Execution Steps

{chr(10).join(step_docs)}

## Implementation

This workflow is automatically generated and executed by the Agent School system.
"""


def demo():
    """Demo the documentation generator."""
    generator = DocumentationGenerator(llm_provider="anthropic")

    # Example: Document a Luma event fetcher
    sample_code = '''
def fetch_luma_events(
    search_query: str,
    location: str = "San Francisco, CA",
    radius_miles: int = 5,
    max_results: int = 10
) -> list[dict]:
    """
    Fetch events from Luma matching search criteria.

    Args:
        search_query: Keywords to search for
        location: Location to search near
        radius_miles: Search radius
        max_results: Maximum number of results

    Returns:
        List of event dictionaries
    """
    # Implementation here
    pass
'''

    sample_plan = '''
1. Navigate to lu.ma search page
2. Enter search query and location
3. Apply radius filter
4. Extract event data from results
5. Format and return data
'''

    documentation = generator.generate_documentation(
        function_name="fetch_luma_events",
        workflow_code=sample_code,
        execution_plan=sample_plan,
        use_case="Fetch events from Luma based on user search criteria"
    )

    print("Generated Documentation:")
    print("=" * 80)
    print(documentation)
    print("=" * 80)


if __name__ == "__main__":
    demo()
