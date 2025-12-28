"""
Registry - Workflow Discovery and Management

Maintains a registry of all deterministic workflows and agent plans.
Auto-updates when workflows are created or deleted.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class Registry:
    """
    Central registry for all workflows and plans.

    Provides:
    - Discovery of available workflows
    - Schema loading
    - Metadata access
    - Search and filtering
    """

    def __init__(self, workflows_dir: str = "./workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.registry_file = self.workflows_dir / "registry.json"
        self.deterministic_dir = self.workflows_dir / "deterministic"
        self.agent_plans_dir = self.workflows_dir / "agent_plans"

        # Ensure directories exist
        self.deterministic_dir.mkdir(parents=True, exist_ok=True)
        self.agent_plans_dir.mkdir(parents=True, exist_ok=True)

        # Load or create registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk or create new one."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "deterministic_workflows": {},
                "agent_plans": {},
                "last_updated": datetime.now().isoformat()
            }

    def _save_registry(self):
        """Save registry to disk."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_deterministic_workflow(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        method: str,  # "api" or "browser"
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a new deterministic workflow."""
        workflow_path = self.deterministic_dir / name

        entry = {
            "name": name,
            "path": str(workflow_path),
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "method": method,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.registry["deterministic_workflows"][name] = entry
        self._save_registry()

        print(f"[OK] Registered deterministic workflow: {name}")

    def register_agent_plan(
        self,
        name: str,
        description: str,
        uses_workflows: List[str],
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a new agent plan."""
        plan_path = self.agent_plans_dir / name

        entry = {
            "name": name,
            "path": str(plan_path),
            "description": description,
            "uses_workflows": uses_workflows,
            "num_steps": len(steps),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.registry["agent_plans"][name] = entry
        self._save_registry()

        print(f"[OK] Registered agent plan: {name}")

    def get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """Get deterministic workflow by name."""
        return self.registry["deterministic_workflows"].get(name)

    def get_plan(self, name: str) -> Optional[Dict[str, Any]]:
        """Get agent plan by name."""
        return self.registry["agent_plans"].get(name)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all deterministic workflows."""
        return list(self.registry["deterministic_workflows"].values())

    def list_plans(self) -> List[Dict[str, Any]]:
        """List all agent plans."""
        return list(self.registry["agent_plans"].values())

    def search_workflows(self, query: str) -> List[Dict[str, Any]]:
        """Search workflows by name or description."""
        query_lower = query.lower()
        results = []

        for workflow in self.list_workflows():
            if (query_lower in workflow["name"].lower() or
                query_lower in workflow["description"].lower()):
                results.append(workflow)

        return results

    def search_plans(self, query: str) -> List[Dict[str, Any]]:
        """Search agent plans by name or description."""
        query_lower = query.lower()
        results = []

        for plan in self.list_plans():
            if (query_lower in plan["name"].lower() or
                query_lower in plan["description"].lower()):
                results.append(plan)

        return results

    def unregister_workflow(self, name: str):
        """Remove workflow from registry."""
        if name in self.registry["deterministic_workflows"]:
            del self.registry["deterministic_workflows"][name]
            self._save_registry()
            print(f"[OK] Unregistered workflow: {name}")

    def unregister_plan(self, name: str):
        """Remove plan from registry."""
        if name in self.registry["agent_plans"]:
            del self.registry["agent_plans"][name]
            self._save_registry()
            print(f"[OK] Unregistered plan: {name}")

    def refresh(self):
        """Scan filesystem and update registry."""
        print("[INFO] Refreshing registry from filesystem...")

        # Scan deterministic workflows
        if self.deterministic_dir.exists():
            for workflow_dir in self.deterministic_dir.iterdir():
                if workflow_dir.is_dir():
                    metadata_file = workflow_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        name = workflow_dir.name
                        if name not in self.registry["deterministic_workflows"]:
                            print(f"[INFO] Found new workflow: {name}")
                            # Register it (loading schemas)
                            input_schema_file = workflow_dir / "input_schema.json"
                            output_schema_file = workflow_dir / "output_schema.json"

                            input_schema = {}
                            output_schema = {}

                            if input_schema_file.exists():
                                with open(input_schema_file, 'r') as f:
                                    input_schema = json.load(f)

                            if output_schema_file.exists():
                                with open(output_schema_file, 'r') as f:
                                    output_schema = json.load(f)

                            self.register_deterministic_workflow(
                                name=name,
                                description=metadata.get("description", ""),
                                input_schema=input_schema,
                                output_schema=output_schema,
                                method=metadata.get("method", "unknown"),
                                metadata=metadata
                            )

        # Scan agent plans
        if self.agent_plans_dir.exists():
            for plan_dir in self.agent_plans_dir.iterdir():
                if plan_dir.is_dir():
                    plan_file = plan_dir / "plan.json"
                    if plan_file.exists():
                        with open(plan_file, 'r') as f:
                            plan_data = json.load(f)

                        name = plan_dir.name
                        if name not in self.registry["agent_plans"]:
                            print(f"[INFO] Found new plan: {name}")
                            self.register_agent_plan(
                                name=name,
                                description=plan_data.get("description", ""),
                                uses_workflows=plan_data.get("uses_workflows", []),
                                steps=plan_data.get("steps", []),
                                metadata=plan_data.get("metadata", {})
                            )

        print("[OK] Registry refreshed")

    def stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        return {
            "deterministic_workflows": len(self.registry["deterministic_workflows"]),
            "agent_plans": len(self.registry["agent_plans"]),
        }
