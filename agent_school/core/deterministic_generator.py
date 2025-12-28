"""
Deterministic Generator - Layer 1

Generates pure Python code (deterministic workflows) that:
- Takes structured input
- Returns structured output
- Has NO LLM calls inside
- Can run forever deterministically

The LLM decides: API vs Browser automation
"""

import os
import json
import ast
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from anthropic import Anthropic
from openai import OpenAI

from ..config import Config
from ..utils.code_validator import CodeValidator
from .registry import Registry


class DeterministicGenerator:
    """
    Generates deterministic workflow scripts using LLM.

    The LLM analyzes the target platform and decides:
    - Use API if available (and not paid/restricted)
    - Use Playwright browser automation if no API

    The generated code must be pure Python with no LLM calls.
    """

    def __init__(
        self,
        llm_provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None,
        registry: Optional[Registry] = None
    ):
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = Anthropic(api_key=Config.get_api_key("anthropic"))
            self.model = model or "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=Config.get_api_key("openai"))
            self.model = model or "gpt-4o"

        self.registry = registry or Registry()

    def generate_workflow(
        self,
        name: str,
        description: str,
        target_platform: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a deterministic workflow.

        Args:
            name: Workflow name (e.g., "luma_scraper")
            description: What this workflow does
            target_platform: Target website/API (e.g., "lu.ma")
            constraints: Additional constraints (location, etc.)

        Returns:
            {
                "code": "...",  # Generated Python code
                "method": "api" or "browser",
                "input_schema": {...},
                "output_schema": {...}
            }
        """
        print(f"[INFO] Generating deterministic workflow: {name}")
        print(f"[INFO] Target platform: {target_platform}")

        # Step 1: LLM decides API vs Browser
        method_decision = self._decide_method(target_platform, description)
        print(f"[INFO] Method decided: {method_decision}")

        # Step 2: Generate code based on method
        code = self._generate_code(
            name=name,
            description=description,
            target_platform=target_platform,
            method=method_decision,
            constraints=constraints or {}
        )

        # Step 3: Validate generated code
        validator = CodeValidator()
        validation_result = validator.validate_code(code, strict=False)

        if not validation_result["is_valid"]:
            raise ValueError(f"Generated code validation failed: {validation_result['errors']}")

        if validation_result["warnings"]:
            print(f"[WARNING] Code validation warnings:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")

        # Step 4: Extract schemas from code
        input_schema, output_schema = self._extract_schemas(code)

        # Step 5: Save to disk
        workflow_dir = self.registry.deterministic_dir / name
        workflow_dir.mkdir(parents=True, exist_ok=True)

        # Save workflow.py
        with open(workflow_dir / "workflow.py", 'w') as f:
            f.write(code)

        # Save schemas
        with open(workflow_dir / "input_schema.json", 'w') as f:
            json.dump(input_schema, f, indent=2)

        with open(workflow_dir / "output_schema.json", 'w') as f:
            json.dump(output_schema, f, indent=2)

        # Save metadata
        metadata = {
            "name": name,
            "description": description,
            "target_platform": target_platform,
            "method": method_decision,
            "constraints": constraints or {}
        }
        with open(workflow_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Register in registry
        self.registry.register_deterministic_workflow(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            method=method_decision,
            metadata=metadata
        )

        print(f"[OK] Workflow saved to: {workflow_dir}")

        return {
            "code": code,
            "method": method_decision,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "path": str(workflow_dir)
        }

    def _decide_method(self, target_platform: str, description: str) -> str:
        """
        Use LLM to decide: API or Browser automation.

        The LLM analyzes the platform and determines the best approach.
        """
        decision_prompt = f"""Analyze this platform and decide the best data extraction method.

Platform: {target_platform}
Task: {description}

Considerations:
1. Does this platform have a publicly available API?
2. Is the API free/accessible or paid/restricted?
3. Is the data available without authentication?

Respond with ONLY one word: "api" or "browser"

Examples:
- Platform: github.com → "api" (free public API)
- Platform: lu.ma → "browser" (API is paid/restricted)
- Platform: reddit.com → "api" (free API available)
- Platform: instagram.com → "browser" (API restricted)

Your response:"""

        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": decision_prompt}]
            )
            decision = response.content[0].text.strip().lower()
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": decision_prompt}],
                max_tokens=10,
                temperature=0
            )
            decision = response.choices[0].message.content.strip().lower()

        # Validate decision
        if decision not in ["api", "browser"]:
            print(f"[WARNING] LLM returned invalid decision: {decision}, defaulting to 'browser'")
            decision = "browser"

        return decision

    def _generate_code(
        self,
        name: str,
        description: str,
        target_platform: str,
        method: str,
        constraints: Dict[str, Any]
    ) -> str:
        """Generate Python code for the workflow."""
        # Use optimized prompts
        from ..optimized_prompts import get_workflow_prompt, WORKFLOW_SYSTEM_PROMPT

        user_prompt = get_workflow_prompt(
            task_description=description,
            target_website=target_platform,
            has_api=(method == "api"),
            api_documentation="",  # Could be added later
            constraints=constraints
        )

        print(f"[INFO] Generating code with {self.llm_provider}...")

        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=WORKFLOW_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            code = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": WORKFLOW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4096,
                temperature=0.1
            )
            code = response.choices[0].message.content

        # Remove markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

    def _extract_schemas(self, code: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract input and output schemas from generated code.

        Parses function signature and docstring to determine schemas.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}, {}

        input_schema = {}
        output_schema = {}

        # Find main function (first non-private function)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                # Extract input schema from parameters
                for arg in node.args.args:
                    if arg.arg == 'self':
                        continue

                    arg_name = arg.arg
                    arg_type = "string"  # default

                    # Try to get type hint
                    if arg.annotation:
                        type_hint = ast.unparse(arg.annotation)
                        if 'int' in type_hint.lower():
                            arg_type = "int"
                        elif 'float' in type_hint.lower():
                            arg_type = "float"
                        elif 'bool' in type_hint.lower():
                            arg_type = "bool"
                        elif 'list' in type_hint.lower():
                            arg_type = "list"
                        elif 'dict' in type_hint.lower():
                            arg_type = "dict"

                    input_schema[arg_name] = arg_type

                # Try to extract output schema from return annotation
                if node.returns:
                    return_type = ast.unparse(node.returns)
                    if 'list' in return_type.lower():
                        output_schema = {"type": "list", "items": "dict"}
                    elif 'dict' in return_type.lower():
                        output_schema = {"type": "dict"}
                    else:
                        output_schema = {"type": "any"}

                break  # Only process first function

        return input_schema, output_schema
