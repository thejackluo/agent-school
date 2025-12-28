"""
Executor - Runs Agent Plans

Executes agent plans step by step:
1. Load plan from disk
2. Execute each step (LLM or deterministic)
3. Maintain context (variables between steps)
4. Handle errors gracefully
5. Return final result
"""

import json
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from anthropic import Anthropic
from openai import OpenAI

from ..config import Config
from .registry import Registry
from .validator import Validator


class Executor:
    """
    Executes agent plans step by step.
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
        self.validator = Validator()

    def execute_plan(self, plan_name: str, user_input: str) -> Any:
        """
        Execute an agent plan with user input.

        Args:
            plan_name: Name of the agent plan to execute
            user_input: Natural language input from user

        Returns:
            Final result from the plan
        """
        print(f"[INFO] Executing plan: {plan_name}")
        print(f"[INFO] User input: {user_input}")

        # Load plan
        plan_entry = self.registry.get_plan(plan_name)
        if not plan_entry:
            raise ValueError(f"Plan '{plan_name}' not found in registry")

        plan_path = Path(plan_entry["path"])
        plan_file = plan_path / "plan.json"

        if not plan_file.exists():
            raise FileNotFoundError(f"Plan file not found: {plan_file}")

        with open(plan_file, 'r') as f:
            plan = json.load(f)

        # Execute steps
        context = {"user_input": user_input}
        steps = plan["steps"]

        for step in steps:
            step_id = step.get("id")
            step_type = step.get("type")
            action = step.get("action")

            print(f"[INFO] Executing step {step_id}: {action} ({step_type})")

            try:
                if step_type == "llm":
                    result = self._execute_llm_step(step, context)
                elif step_type == "deterministic":
                    result = self._execute_deterministic_step(step, context)
                else:
                    raise ValueError(f"Unknown step type: {step_type}")

                # Store result in context
                output_var = step.get("output_var")
                if output_var:
                    context[output_var] = result
                    print(f"[INFO] Stored result in context['{output_var}']")

            except Exception as e:
                print(f"[ERROR] Step {step_id} failed: {e}")
                # Try to continue with partial results
                context[step.get("output_var", f"step_{step_id}_error")] = {
                    "error": str(e),
                    "step": step_id
                }

        # Return final result (last step's output or specific variable)
        final_var = plan.get("final_output_var", steps[-1].get("output_var"))
        return context.get(final_var, context)

    def _execute_llm_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute an LLM step."""
        prompt_template = step.get("prompt", "")
        input_from = step.get("input_from")

        # Build prompt with context
        prompt = prompt_template

        # Replace variables in prompt
        if input_from:
            if isinstance(input_from, str):
                # Single input
                if input_from in context:
                    # Replace {variable} with actual value
                    prompt = prompt.replace(f"{{{input_from}}}", str(context[input_from]))
            elif isinstance(input_from, list):
                # Multiple inputs
                for var_name in input_from:
                    if var_name in context:
                        prompt = prompt.replace(f"{{{var_name}}}", str(context[var_name]))

        # Also replace {user_input} if present
        if "{user_input}" in prompt:
            prompt = prompt.replace("{user_input}", context.get("user_input", ""))

        print(f"[DEBUG] LLM prompt (first 200 chars): {prompt[:200]}...")

        # Call LLM
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3
            )
            result = response.choices[0].message.content

        # Try to parse as JSON - handle markdown code blocks and other LLM formats
        result = result.strip()
        
        # Strip markdown code blocks if present
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Try to parse as JSON
        if result.startswith("{") or result.startswith("["):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse JSON: {e}")
                # Keep as string if parsing fails

        return result

    def _execute_deterministic_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a deterministic workflow step."""
        workflow_name = step.get("workflow")

        # Load workflow from registry
        workflow_entry = self.registry.get_workflow(workflow_name)
        if not workflow_entry:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow_path = Path(workflow_entry["path"])
        workflow_file = workflow_path / "workflow.py"

        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_file}")

        # Load input parameters
        input_from = step.get("input_from")
        if not input_from:
            raise ValueError(f"Step {step.get('id')}: deterministic step requires 'input_from'")

        if input_from not in context:
            raise ValueError(f"Input variable '{input_from}' not found in context")

        input_data = context[input_from]

        # Validate input against schema
        input_schema = workflow_entry["input_schema"]
        is_valid, error = self.validator.validate_input(input_data, input_schema)

        if not is_valid:
            print(f"[WARNING] Input validation failed: {error}")
            # Try to coerce types
            input_data = self.validator.coerce_types(input_data, input_schema)
            print(f"[INFO] Coerced input types")

        # Import and execute workflow
        spec = importlib.util.spec_from_file_location("workflow_module", workflow_file)
        workflow_module = importlib.util.module_from_spec(spec)
        sys.modules["workflow_module"] = workflow_module
        spec.loader.exec_module(workflow_module)

        # Get function name from metadata (preferred) or find by convention
        metadata = workflow_entry.get("metadata", {})
        function_name = metadata.get("function_name")
        
        if function_name and hasattr(workflow_module, function_name):
            main_function = getattr(workflow_module, function_name)
        else:
            # Fallback: look for common function names
            common_names = ["fetch_luma_events", "run", "main", "execute", "fetch", "scrape"]
            main_function = None
            for name in common_names:
                if hasattr(workflow_module, name):
                    main_function = getattr(workflow_module, name)
                    break
            
            # Last resort: first function (not class) that's not private
            if not main_function:
                import inspect
                for attr_name in dir(workflow_module):
                    if attr_name.startswith('_'):
                        continue
                    attr = getattr(workflow_module, attr_name)
                    if inspect.isfunction(attr):
                        main_function = attr
                        break

        if not main_function:
            raise ValueError(f"No main function found in workflow: {workflow_name}")

        print(f"[INFO] Calling workflow function: {main_function.__name__}")

        # Filter input_data to only include keys that exist in schema
        # This handles cases where LLM adds extra fields the workflow doesn't expect
        if isinstance(input_data, dict):
            valid_keys = set(input_schema.keys())
            filtered_input = {k: v for k, v in input_data.items() if k in valid_keys}
            print(f"[DEBUG] Filtered input keys: {list(filtered_input.keys())}")
            result = main_function(**filtered_input)
        else:
            result = main_function(input_data)

        # Validate output against schema (if available)
        output_schema = workflow_entry.get("output_schema", {})
        if output_schema:
            # Basic validation - just check it's not None/empty
            if result is None:
                print(f"[WARNING] Workflow returned None")

        return result
