"""
Function Creator (File 4)

This is the core orchestrator that combines the workflow generator and plan generator
to create executable, MCP-callable functions. This is the "magic" of Agent School.

Flow:
1. User provides a task description (e.g., "fetch hip-hop parties in SF")
2. Plan Generator creates a step-by-step plan
3. Workflow Generator creates executable code based on the plan
4. Function Creator wraps it all into a callable function
5. Documentation Generator creates usage docs
6. Result: A new function that LLMs can call via MCP

This enables dynamic function generation - the system creates new tools on-the-fly
based on user needs rather than having a fixed set of predefined functions.
"""

import os
import json
import inspect
from typing import Any, Dict, Optional, Callable
from pathlib import Path

from .workflow_generator import WorkflowGenerator
from .plan_generator import PlanGenerator, ExecutionPlan
from .documentation_generator import DocumentationGenerator


class GeneratedFunction:
    """
    Represents a dynamically generated function that can be called by LLMs.

    This wraps the generated workflow code and provides metadata for MCP integration.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        workflow_code: str,
        execution_plan: ExecutionPlan,
        documentation: str,
        executable_func: Optional[Callable] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.workflow_code = workflow_code
        self.execution_plan = execution_plan
        self.documentation = documentation
        self.executable_func = executable_func

    def to_mcp_schema(self) -> Dict[str, Any]:
        """
        Convert to MCP tool schema format.

        This is the format MCP servers expect for tool definitions.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": [
                    param_name for param_name, param_info in self.parameters.items()
                    if param_info.get("required", False)
                ]
            }
        }

    def execute(self, **kwargs) -> Any:
        """
        Execute the generated function with given parameters.

        Args:
            **kwargs: Parameters to pass to the function

        Returns:
            The result of executing the workflow
        """
        if self.executable_func is None:
            raise RuntimeError("Function has not been compiled yet. Call compile() first.")

        return self.executable_func(**kwargs)

    def save(self, output_dir: str) -> Dict[str, str]:
        """
        Save all artifacts (code, plan, docs) to disk.

        Args:
            output_dir: Directory to save artifacts

        Returns:
            Dictionary mapping artifact type to file path
        """
        output_path = Path(output_dir) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Save workflow code
        code_path = output_path / "workflow.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(self.workflow_code)
        artifacts["code"] = str(code_path)

        # Save execution plan
        plan_path = output_path / "plan.json"
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write(self.execution_plan.to_json())
        artifacts["plan"] = str(plan_path)

        # Save documentation
        doc_path = output_path / "README.md"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(self.documentation)
        artifacts["documentation"] = str(doc_path)

        # Save MCP schema
        schema_path = output_path / "mcp_schema.json"
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_mcp_schema(), f, indent=2)
        artifacts["mcp_schema"] = str(schema_path)

        return artifacts


class FunctionCreator:
    """
    The main orchestrator that creates executable functions from user descriptions.

    This is the heart of Agent School - it coordinates all the components to
    transform natural language task descriptions into working, callable functions.
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the function creator.

        Args:
            llm_provider: Which LLM provider to use
            model: Specific model to use
            cache_dir: Directory to cache generated functions
        """
        self.workflow_generator = WorkflowGenerator(llm_provider, model)
        self.plan_generator = PlanGenerator(llm_provider, model)
        self.doc_generator = DocumentationGenerator(llm_provider, model)

        self.cache_dir = cache_dir or "./generated_functions"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Registry of generated functions
        self.function_registry: Dict[str, GeneratedFunction] = {}

    def create_function(
        self,
        task_description: str,
        function_name: str,
        target_platform: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        compile_code: bool = True
    ) -> GeneratedFunction:
        """
        Create a new executable function from a task description.

        This is the main entry point for generating new functions.

        Args:
            task_description: Natural language description of what to do
            function_name: Name for the generated function
            target_platform: Platform to extract from (e.g., "Luma", "Eventbrite")
            parameters: Expected parameters for the function
            context: Additional context (API availability, constraints, etc.)
            compile_code: Whether to compile the code into an executable function

        Returns:
            GeneratedFunction object that can be called or saved

        Example:
            >>> creator = FunctionCreator()
            >>> func = creator.create_function(
            ...     task_description="Fetch hip-hop parties in SF",
            ...     function_name="fetch_sf_hiphop_parties",
            ...     target_platform="Luma"
            ... )
            >>> results = func.execute(radius_miles=5)
        """
        print(f"🔧 Creating function: {function_name}")
        print(f"   Task: {task_description}")
        print()

        # Step 1: Generate execution plan
        print("📋 Step 1/4: Generating execution plan...")
        execution_plan = self.plan_generator.generate_plan(
            user_query=task_description,
            target_platform=target_platform,
            context=context or {}
        )
        print(f"   ✓ Generated {len(execution_plan.steps)} steps")
        print()

        # Step 2: Generate workflow code
        print("⚙️  Step 2/4: Generating workflow code...")
        workflow_code = self.workflow_generator.generate_workflow(
            task_description=task_description,
            target_website=target_platform,
            has_api=context.get("has_api", False) if context else False,
            api_documentation=context.get("api_docs") if context else None,
            constraints=context or {}
        )
        print(f"   ✓ Generated {len(workflow_code.splitlines())} lines of code")
        print()

        # Step 3: Generate documentation
        print("📝 Step 3/4: Generating documentation...")
        documentation = self.doc_generator.generate_documentation(
            function_name=function_name,
            workflow_code=workflow_code,
            execution_plan=str(execution_plan),
            use_case=task_description
        )
        print("   ✓ Documentation generated")
        print()

        # Step 4: Create the GeneratedFunction object
        print("🎁 Step 4/4: Packaging function...")

        # Infer parameters from workflow code if not provided
        if parameters is None:
            parameters = self._infer_parameters(workflow_code, execution_plan)

        # Create the function object
        generated_func = GeneratedFunction(
            name=function_name,
            description=task_description,
            parameters=parameters,
            workflow_code=workflow_code,
            execution_plan=execution_plan,
            documentation=documentation
        )

        # Optionally compile the code into an executable function
        if compile_code:
            generated_func.executable_func = self._compile_function(workflow_code, function_name)
            print("   ✓ Code compiled and ready to execute")

        # Save to registry
        self.function_registry[function_name] = generated_func

        # Save to disk
        artifacts = generated_func.save(self.cache_dir)
        print(f"   ✓ Artifacts saved to: {self.cache_dir}/{function_name}/")
        print()

        print(f"✅ Function '{function_name}' created successfully!")
        return generated_func

    def _infer_parameters(
        self,
        workflow_code: str,
        execution_plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """
        Infer function parameters from workflow code and execution plan.

        This extracts parameter information from the generated code's function signature.
        """
        # Try to extract function signature from code
        lines = workflow_code.split('\n')
        params = {}

        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and '(' in line:
                # Found a function definition
                # Extract parameter names (basic parsing)
                func_start = line.index('(')
                # Look for the closing paren (might be on next lines)
                func_sig = line[func_start:]
                for j in range(i + 1, min(i + 10, len(lines))):
                    func_sig += ' ' + lines[j].strip()
                    if ')' in lines[j]:
                        break

                # Basic parameter extraction
                param_str = func_sig[func_sig.index('(') + 1:func_sig.index(')')]
                for param in param_str.split(','):
                    param = param.strip()
                    if not param or param == 'self':
                        continue

                    # Extract parameter name (before : or =)
                    param_name = param.split(':')[0].split('=')[0].strip()
                    if not param_name:
                        continue

                    # Extract type hint if available
                    param_type = "string"  # default
                    if ':' in param:
                        type_hint = param.split(':')[1].split('=')[0].strip()
                        if 'int' in type_hint:
                            param_type = "integer"
                        elif 'bool' in type_hint:
                            param_type = "boolean"
                        elif 'list' in type_hint or 'List' in type_hint:
                            param_type = "array"

                    # Check if required (no default value)
                    is_required = '=' not in param

                    params[param_name] = {
                        "type": param_type,
                        "description": f"Parameter {param_name}",
                        "required": is_required
                    }

                break  # Found the main function

        return params

    def _compile_function(self, workflow_code: str, function_name: str) -> Callable:
        """
        Compile the generated workflow code into an executable Python function.

        This uses exec() to turn the generated code string into a callable function.
        SECURITY NOTE: Only use this with code generated by trusted LLMs.
        """
        # Create a namespace for execution
        namespace = {}

        # Execute the code to define the function
        exec(workflow_code, namespace)

        # Find the main function (usually the first defined function)
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_'):
                return obj

        raise ValueError(f"Could not find a callable function in generated code")

    def get_function(self, function_name: str) -> Optional[GeneratedFunction]:
        """Get a previously generated function from the registry."""
        return self.function_registry.get(function_name)

    def list_functions(self) -> list[str]:
        """List all generated functions in the registry."""
        return list(self.function_registry.keys())

    def export_mcp_server_config(self, output_path: str) -> None:
        """
        Export all generated functions as an MCP server configuration.

        This creates a JSON file that MCP servers can use to expose
        all generated functions as callable tools.
        """
        tools = []
        for func_name, func in self.function_registry.items():
            tools.append(func.to_mcp_schema())

        config = {
            "name": "agent-school-functions",
            "version": "0.1.0",
            "tools": tools
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"✓ MCP server config exported to: {output_path}")


def demo():
    """Demo the function creator with Luma examples."""
    creator = FunctionCreator(llm_provider="anthropic")

    # Example 1: Create a function to fetch hip-hop parties
    print("=" * 80)
    print("DEMO 1: Creating function to fetch hip-hop parties in SF")
    print("=" * 80)
    print()

    func1 = creator.create_function(
        task_description="Fetch hip-hop parties in San Francisco within a specified radius",
        function_name="fetch_sf_hiphop_parties",
        target_platform="Luma",
        context={
            "has_api": False,
            "requires_browser": True,
            "location": "San Francisco, CA"
        }
    )

    print()
    print("Function created! Here's the MCP schema:")
    print(json.dumps(func1.to_mcp_schema(), indent=2))
    print()

    # Example 2: Create a function to fetch YC events
    print("=" * 80)
    print("DEMO 2: Creating function to fetch YC co-founder events")
    print("=" * 80)
    print()

    func2 = creator.create_function(
        task_description="Find Y Combinator co-founder matching or networking events",
        function_name="fetch_yc_events",
        target_platform="Luma",
        context={
            "has_api": False,
            "requires_browser": True,
            "keywords": ["YC", "Y Combinator", "co-founder", "startup"]
        }
    )

    print()
    print("All generated functions:")
    for func_name in creator.list_functions():
        print(f"  - {func_name}")
    print()

    # Export MCP config
    creator.export_mcp_server_config("./mcp_tools_config.json")


if __name__ == "__main__":
    demo()
