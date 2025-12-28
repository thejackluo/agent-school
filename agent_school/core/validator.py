"""
Validator - Schema and Data Flow Validation

Ensures:
1. Data types match at workflow boundaries
2. Required fields are present
3. Agent plans have valid data flow
4. No circular dependencies
"""

from typing import Dict, Any, List, Optional
import json


class Validator:
    """
    Validates schemas and data flow between components.
    """

    @staticmethod
    def validate_input(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate input data against schema.

        Args:
            data: Input data to validate
            schema: Expected schema (can be flat {"field": "type"} or nested {"field": {"type": "..."}}

        Returns:
            (is_valid, error_message)
        """
        if not schema:
            return True, None

        for field_name, field_spec in schema.items():
            # Handle nested schema format: {"field": {"type": "string", "required": true, ...}}
            if isinstance(field_spec, dict):
                expected_type = field_spec.get("type", "string").lower()
                is_required = field_spec.get("required", False)
            else:
                # Simple format: {"field": "string"}
                expected_type = str(field_spec).lower()
                is_required = True

            # Check if required field exists
            if field_name not in data:
                if is_required:
                    return False, f"Missing required field: {field_name}"
                continue

            # Check type (basic type checking)
            value = data[field_name]
            
            # Skip None values for optional fields
            if value is None and not is_required:
                continue

            if expected_type == "str" or expected_type == "string":
                if not isinstance(value, str):
                    return False, f"Field '{field_name}' must be string, got {type(value).__name__}"

            elif expected_type == "int" or expected_type == "integer":
                if not isinstance(value, int):
                    return False, f"Field '{field_name}' must be integer, got {type(value).__name__}"

            elif expected_type == "float" or expected_type == "number":
                if not isinstance(value, (int, float)):
                    return False, f"Field '{field_name}' must be number, got {type(value).__name__}"

            elif expected_type == "bool" or expected_type == "boolean":
                if not isinstance(value, bool):
                    return False, f"Field '{field_name}' must be boolean, got {type(value).__name__}"

            elif expected_type == "list" or expected_type == "array":
                if not isinstance(value, list):
                    return False, f"Field '{field_name}' must be list, got {type(value).__name__}"

            elif expected_type == "dict" or expected_type == "object":
                if not isinstance(value, dict):
                    return False, f"Field '{field_name}' must be dict, got {type(value).__name__}"

        return True, None

    @staticmethod
    def validate_plan_data_flow(plan: Dict[str, Any], registry) -> tuple[bool, Optional[str]]:
        """
        Validate that data flows correctly through an agent plan.

        Checks:
        1. All referenced workflows exist
        2. Variables are defined before use
        3. No circular dependencies
        """
        steps = plan.get("steps", [])
        context_vars = {}  # Track available variables

        for i, step in enumerate(steps):
            step_id = step.get("id", i + 1)
            step_type = step.get("type")

            if step_type == "deterministic":
                # Check workflow exists
                workflow_name = step.get("workflow")
                if not workflow_name:
                    return False, f"Step {step_id}: missing 'workflow' field"

                workflow = registry.get_workflow(workflow_name)
                if not workflow:
                    return False, f"Step {step_id}: workflow '{workflow_name}' not found"

                # Check input variable exists
                input_from = step.get("input_from")
                if input_from and input_from not in context_vars:
                    return False, f"Step {step_id}: input variable '{input_from}' not defined"

                # Add output variable to context
                output_var = step.get("output_var")
                if output_var:
                    context_vars[output_var] = "defined"

            elif step_type == "llm":
                # Check input variable if specified
                input_from = step.get("input_from")
                if input_from:
                    if isinstance(input_from, str):
                        if input_from not in context_vars:
                            return False, f"Step {step_id}: input variable '{input_from}' not defined"
                    elif isinstance(input_from, list):
                        for var in input_from:
                            if var not in context_vars:
                                return False, f"Step {step_id}: input variable '{var}' not defined"

                # Add output variable to context
                output_var = step.get("output_var")
                if output_var:
                    context_vars[output_var] = "defined"

            else:
                return False, f"Step {step_id}: unknown step type '{step_type}'"

        return True, None

    @staticmethod
    def coerce_types(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try to coerce data to match schema types.

        Returns coerced data.
        """
        coerced = {}

        for field_name, field_spec in schema.items():
            if field_name not in data:
                continue

            value = data[field_name]
            
            # Handle nested schema format
            if isinstance(field_spec, dict):
                expected_type = field_spec.get("type", "string").lower()
            else:
                expected_type = str(field_spec).lower()

            try:
                if expected_type in ["str", "string"]:
                    coerced[field_name] = str(value)
                elif expected_type in ["int", "integer"]:
                    coerced[field_name] = int(value)
                elif expected_type in ["float", "number"]:
                    coerced[field_name] = float(value)
                elif expected_type in ["bool", "boolean"]:
                    coerced[field_name] = bool(value)
                else:
                    coerced[field_name] = value
            except (ValueError, TypeError):
                # Coercion failed, keep original
                coerced[field_name] = value

        return coerced

    @staticmethod
    def validate_workflow_exists(workflow_name: str, registry) -> bool:
        """Check if a deterministic workflow exists."""
        return registry.get_workflow(workflow_name) is not None

    @staticmethod
    def validate_plan_exists(plan_name: str, registry) -> bool:
        """Check if an agent plan exists."""
        return registry.get_plan(plan_name) is not None
