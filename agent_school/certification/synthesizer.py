"""
Certification Synthesizer - Converts exploration logs to cached certifications

This is Phase B of the certification loop: taking successful ReAct
exploration runs and synthesizing them into deterministic JSON scripts.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from .certification import (
    Certification, 
    CertStep, 
    ParamSpec, 
    ActionType, 
    SelectorStrategy
)
from .action_logger import ActionLog, Action


class CertificationSynthesizer:
    """
    Synthesizes exploration logs into parameterized certifications.
    
    Key responsibilities:
    1. Analyze action sequence from successful exploration
    2. Identify variable inputs (e.g., "Hello World" → {message})
    3. Convert to semantic selectors where possible
    4. Generate clean, parameterized certification
    """
    
    def __init__(self, llm_provider: str = "anthropic"):
        self.llm_provider = llm_provider
        self._client = None
    
    def _get_llm_client(self):
        """Get LLM client"""
        if self._client is None:
            if self.llm_provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic()
            else:
                from openai import OpenAI
                self._client = OpenAI()
        return self._client
    
    def synthesize(
        self,
        action_log: ActionLog,
        task_patterns: Optional[List[str]] = None,
    ) -> Certification:
        """
        Synthesize an action log into a certification.
        
        Args:
            action_log: Log from successful exploration
            task_patterns: Optional regex patterns to match similar tasks
            
        Returns:
            A new Certification ready for execution
        """
        if not action_log.success:
            raise ValueError("Cannot synthesize from failed exploration")
        
        if not action_log.actions:
            raise ValueError("Action log has no actions")
        
        # Use LLM to analyze and parameterize
        analysis = self._analyze_actions(action_log)
        
        # Convert actions to cert steps
        steps = self._convert_to_steps(action_log.actions, analysis)
        
        # Extract parameters
        parameters = self._extract_parameters(analysis)
        
        # Generate task patterns if not provided
        if not task_patterns:
            task_patterns = self._generate_task_patterns(action_log.task, analysis)
        
        # Create certification name from task
        name = self._generate_name(action_log.task)
        
        return Certification(
            name=name,
            domain=action_log.domain,
            task_description=action_log.task,
            task_patterns=task_patterns,
            steps=steps,
            parameters=parameters,
            source_exploration_id=action_log.exploration_id,
        )
    
    def _analyze_actions(self, action_log: ActionLog) -> Dict[str, Any]:
        """
        Use LLM to analyze actions and identify patterns/variables.
        
        Returns analysis with:
        - parameters: detected variable inputs
        - selector_improvements: suggestions for semantic selectors
        - step_descriptions: improved descriptions
        """
        client = self._get_llm_client()
        
        # Build action summary for LLM
        action_summary = []
        for i, action in enumerate(action_log.actions):
            action_summary.append({
                "step": i + 1,
                "action": action.action_type,
                "description": action.description,
                "element": action.element_text,
                "value": action.input_value,
            })
        
        prompt = f"""Analyze this browser automation sequence and identify:

1. **Parameters**: Any values that should be user-provided variables instead of hardcoded.
   - Look for text that was typed, specific content, dates, names, etc.
   - Don't parameterize UI navigation (button names, menu items)
   
2. **Selector Improvements**: Suggest semantic selectors instead of CSS.
   - Prefer: visible text, aria-labels, placeholders
   - Avoid: CSS selectors, XPaths (brittle)
   
3. **Step Descriptions**: Provide clear, concise descriptions for each step.

Task: {action_log.task}
Domain: {action_log.domain}

Actions:
{json.dumps(action_summary, indent=2)}

Respond as JSON:
{{
  "parameters": [
    {{"name": "param_name", "description": "what it's for", "type": "string", "source_step": 1}}
  ],
  "steps": [
    {{
      "step": 1,
      "description": "Clear description",
      "selector_strategy": "text|aria|placeholder|css",
      "selector_value": "Button Text or [aria-label=...]",
      "parameterized_value": "{{param_name}} or null if not input"
    }}
  ],
  "task_patterns": ["regex pattern to match similar requests"]
}}
"""
        
        try:
            if self.llm_provider == "anthropic":
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.choices[0].message.content
            
            # Parse JSON
            if "{" in result_text:
                json_start = result_text.index("{")
                json_end = result_text.rindex("}") + 1
                return json.loads(result_text[json_start:json_end])
            
        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}")
        
        # Fallback: basic analysis
        return self._basic_analysis(action_log)
    
    def _basic_analysis(self, action_log: ActionLog) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        parameters = []
        steps = []
        
        for i, action in enumerate(action_log.actions):
            step = {
                "step": i + 1,
                "description": action.description,
                "selector_strategy": "text" if action.element_text else "css",
                "selector_value": action.element_text or action.element_selector,
                "parameterized_value": None,
            }
            
            # Basic parameter detection: typed values
            if action.input_value and len(action.input_value) > 3:
                param_name = f"input_{i+1}"
                parameters.append({
                    "name": param_name,
                    "description": f"Value for step {i+1}",
                    "type": "string",
                    "source_step": i + 1,
                })
                step["parameterized_value"] = f"{{{param_name}}}"
            
            steps.append(step)
        
        return {
            "parameters": parameters,
            "steps": steps,
            "task_patterns": [re.escape(action_log.task)],
        }
    
    def _convert_to_steps(
        self,
        actions: List[Action],
        analysis: Dict[str, Any],
    ) -> List[CertStep]:
        """Convert raw actions + analysis to CertStep objects"""
        steps = []
        analyzed_steps = {s["step"]: s for s in analysis.get("steps", [])}
        
        for i, action in enumerate(actions):
            step_num = i + 1
            analyzed = analyzed_steps.get(step_num, {})
            
            # Determine action type
            action_type = self._map_action_type(action.action_type)
            
            # Determine selector strategy
            strategy_str = analyzed.get("selector_strategy", "text")
            selector_strategy = self._map_selector_strategy(strategy_str)
            
            # Get selector value
            selector_value = analyzed.get("selector_value") or action.element_text
            
            # Get value (parameterized or original)
            value = analyzed.get("parameterized_value") or action.input_value
            
            cert_step = CertStep(
                id=step_num,
                action=action_type,
                description=analyzed.get("description", action.description),
                selector_strategy=selector_strategy if selector_value else None,
                selector_value=selector_value,
                value=value,
                timeout_ms=10000,
            )
            steps.append(cert_step)
        
        return steps
    
    def _map_action_type(self, action_str: str) -> ActionType:
        """Map action string to ActionType enum"""
        mapping = {
            "click": ActionType.CLICK,
            "type": ActionType.TYPE,
            "input": ActionType.TYPE,
            "navigate": ActionType.NAVIGATE,
            "goto": ActionType.NAVIGATE,
            "scroll": ActionType.SCROLL,
            "select": ActionType.SELECT,
            "wait": ActionType.WAIT,
            "extract": ActionType.EXTRACT,
        }
        return mapping.get(action_str.lower(), ActionType.CLICK)
    
    def _map_selector_strategy(self, strategy_str: str) -> SelectorStrategy:
        """Map strategy string to SelectorStrategy enum"""
        mapping = {
            "text": SelectorStrategy.TEXT,
            "aria": SelectorStrategy.ARIA_LABEL,
            "aria-label": SelectorStrategy.ARIA_LABEL,
            "placeholder": SelectorStrategy.PLACEHOLDER,
            "role": SelectorStrategy.ROLE,
            "css": SelectorStrategy.CSS,
            "xpath": SelectorStrategy.XPATH,
        }
        return mapping.get(strategy_str.lower(), SelectorStrategy.TEXT)
    
    def _extract_parameters(self, analysis: Dict[str, Any]) -> List[ParamSpec]:
        """Extract parameter specifications from analysis"""
        params = []
        for p in analysis.get("parameters", []):
            params.append(ParamSpec(
                name=p["name"],
                description=p.get("description", ""),
                type=p.get("type", "string"),
                required=True,
            ))
        return params
    
    def _generate_task_patterns(
        self,
        task: str,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate regex patterns to match similar tasks"""
        patterns = analysis.get("task_patterns", [])
        
        if not patterns:
            # Generate basic pattern from task
            # Replace specific values with wildcards
            pattern = task.lower()
            # Common replacements
            pattern = re.sub(r'"[^"]*"', '.*', pattern)  # Quoted strings
            pattern = re.sub(r'\d+', '\\d+', pattern)  # Numbers
            patterns.append(pattern)
        
        return patterns
    
    def _generate_name(self, task: str) -> str:
        """Generate a clean name from task description"""
        # Take first few words, lowercase, underscore-separated
        words = task.lower().split()[:4]
        name = "_".join(words)
        # Remove non-alphanumeric
        name = re.sub(r'[^a-z0-9_]', '', name)
        return name or "unnamed_cert"
    
    def validate_certification(
        self,
        cert: Certification,
        test_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validate a certification by doing a dry run.
        
        TODO: Implement actual validation by running cert with test params
        """
        # Basic validation
        if not cert.steps:
            return False
        if not cert.task_patterns:
            return False
        
        # Check all referenced params exist
        param_names = {p.name for p in cert.parameters}
        for step in cert.steps:
            if step.value:
                # Find {param} references
                refs = re.findall(r'\{(\w+)\}', step.value)
                for ref in refs:
                    if ref not in param_names:
                        print(f"Warning: Step {step.id} references unknown param: {ref}")
                        return False
        
        return True
