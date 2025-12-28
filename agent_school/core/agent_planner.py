"""
Agent Planner - Layer 2

Creates LLM orchestration plans that:
- Parse natural language → structured params (LLM)
- Call deterministic workflows with params
- Process/rank results (LLM)
- Format output (LLM)

Plans are JSON definitions of multi-step workflows.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from anthropic import Anthropic
from openai import OpenAI

from ..config import Config
from .registry import Registry
from .validator import Validator


class AgentPlanner:
    """
    Creates agent plans that orchestrate deterministic workflows with LLM steps.
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

    def create_plan(
        self,
        name: str,
        description: str,
        goal: str,
        uses_workflows: List[str],
        example_inputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create an agent plan.

        Args:
            name: Plan name (e.g., "personalized_event_finder")
            description: What this plan does
            goal: The end goal (e.g., "Find personalized events for user")
            uses_workflows: List of deterministic workflow names to use
            example_inputs: Example user inputs for this plan

        Returns:
            Complete plan definition
        """
        print(f"[INFO] Creating agent plan: {name}")
        print(f"[INFO] Uses workflows: {uses_workflows}")

        # Step 1: Validate that all workflows exist
        for workflow_name in uses_workflows:
            if not self.validator.validate_workflow_exists(workflow_name, self.registry):
                raise ValueError(f"Workflow '{workflow_name}' not found in registry")

        # Step 2: Load workflow schemas
        workflow_info = {}
        for workflow_name in uses_workflows:
            workflow = self.registry.get_workflow(workflow_name)
            workflow_info[workflow_name] = {
                "input_schema": workflow["input_schema"],
                "output_schema": workflow["output_schema"],
                "description": workflow["description"]
            }

        # Step 3: LLM generates the plan
        plan_steps = self._generate_plan_steps(
            goal=goal,
            description=description,
            workflow_info=workflow_info,
            example_inputs=example_inputs or []
        )

        # Step 4: Validate data flow
        plan_data = {
            "name": name,
            "description": description,
            "goal": goal,
            "uses_workflows": uses_workflows,
            "steps": plan_steps
        }

        is_valid, error = self.validator.validate_plan_data_flow(plan_data, self.registry)
        if not is_valid:
            raise ValueError(f"Plan data flow validation failed: {error}")

        # Step 5: Save to disk
        plan_dir = self.registry.agent_plans_dir / name
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Save plan.json
        with open(plan_dir / "plan.json", 'w') as f:
            json.dump(plan_data, f, indent=2)

        # Save metadata
        metadata = {
            "name": name,
            "description": description,
            "goal": goal,
            "example_inputs": example_inputs or []
        }
        with open(plan_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Register in registry
        self.registry.register_agent_plan(
            name=name,
            description=description,
            uses_workflows=uses_workflows,
            steps=plan_steps,
            metadata=metadata
        )

        print(f"[OK] Plan saved to: {plan_dir}")

        return plan_data

    def _generate_plan_steps(
        self,
        goal: str,
        description: str,
        workflow_info: Dict[str, Dict[str, Any]],
        example_inputs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to generate plan steps.
        """
        from ..optimized_prompts import PLAN_SYSTEM_PROMPT

        user_prompt = f"""Create an agent plan for: {goal}

Description: {description}

Available Deterministic Workflows:
"""
        for workflow_name, info in workflow_info.items():
            user_prompt += f"""
- {workflow_name}:
  Description: {info['description']}
  Input Schema: {json.dumps(info['input_schema'], indent=2)}
  Output Schema: {json.dumps(info['output_schema'], indent=2)}
"""

        if example_inputs:
            user_prompt += f"\nExample User Inputs:\n"
            for example in example_inputs:
                user_prompt += f"- \"{example}\"\n"

        user_prompt += """
Create a multi-step plan that:
1. Parses user input (LLM step)
2. Calls deterministic workflows with structured params
3. Processes/ranks results (LLM step if needed)
4. Formats final response (LLM step)

Each step must have:
- id: step number
- type: "llm" or "deterministic"
- action: short name
- description: what this step does
- input_from: variable name or list of variable names (optional)
- output_var: variable name to store output
- workflow: workflow name (only for deterministic steps)
- prompt: LLM prompt template (only for LLM steps)

Return ONLY a JSON array of steps, no other text.
"""

        print(f"[INFO] Generating plan steps with {self.llm_provider}...")

        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=PLAN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            steps_text = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.2
            )
            steps_text = response.choices[0].message.content

        # Parse JSON response
        steps_text = steps_text.strip()
        if "```json" in steps_text:
            steps_text = steps_text.split("```json")[1].split("```")[0].strip()
        elif "```" in steps_text:
            steps_text = steps_text.split("```")[1].split("```")[0].strip()

        try:
            steps = json.loads(steps_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse plan steps as JSON: {e}\n\nResponse:\n{steps_text}")

        return steps
