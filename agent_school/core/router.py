"""
Router - Natural Language Intent Detection (Layer 3)

Routes user requests to appropriate actions:
- Create new workflow
- Create new agent plan
- Execute existing plan
- Show help/guidance

Acts as a conversational mentor for non-technical users.
"""

import json
from typing import Dict, Any, Literal, Optional
from anthropic import Anthropic
from openai import OpenAI

from ..config import Config
from .registry import Registry
from .executor import Executor


class Router:
    """
    Natural language router that acts as a mentor.

    Analyzes user intent and routes to:
    - Workflow creation
    - Plan creation
    - Plan execution
    - Help/guidance
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
        self.executor = Executor(llm_provider, model, registry)

    def route(self, user_input: str) -> Dict[str, Any]:
        """
        Route user input to appropriate action.

        Args:
            user_input: Natural language input from user

        Returns:
            {
                "action": "create_workflow" | "create_plan" | "execute" | "help",
                "intent": {...},  # Parsed intent
                "response": "...",  # Response to user
                "needs_confirmation": bool,  # Whether to ask user before proceeding
                "suggested_params": {...}  # Suggested parameters
            }
        """
        print(f"[INFO] Routing user input: {user_input[:100]}...")

        # Analyze intent
        intent = self._analyze_intent(user_input)

        action = intent.get("action")
        confidence = intent.get("confidence", 0)

        print(f"[INFO] Detected action: {action} (confidence: {confidence})")

        if action == "execute":
            return self._handle_execute(user_input, intent)
        elif action == "create_workflow":
            return self._handle_create_workflow(user_input, intent)
        elif action == "create_plan":
            return self._handle_create_plan(user_input, intent)
        elif action == "help":
            return self._handle_help(user_input, intent)
        elif action == "list":
            return self._handle_list(intent)
        else:
            return {
                "action": "help",
                "response": "I'm not sure what you want to do. Could you clarify? I can help you:\n- Find events or data\n- Create new workflows\n- Set up automation\n\nWhat would you like to do?"
            }

    def _analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to analyze user intent."""
        # Get available workflows and plans
        workflows = self.registry.list_workflows()
        plans = self.registry.list_plans()

        system_prompt = """You are an AI assistant that analyzes user intent for a workflow automation system.

Available actions:
1. "execute" - User wants to run an existing workflow (e.g., "find events in SF")
2. "create_workflow" - User wants to create a new data extraction workflow
3. "create_plan" - User wants to create an agent orchestration plan
4. "list" - User wants to see what's available
5. "help" - User needs guidance

Analyze the user's input and respond with JSON:
{
  "action": "execute" | "create_workflow" | "create_plan" | "list" | "help",
  "confidence": 0.0 to 1.0,
  "reasoning": "why you chose this action",
  "extracted_info": {
    // For execute: {task, platform, filters}
    // For create_workflow: {platform, description}
    // For create_plan: {goal, workflows_needed}
  }
}"""

        user_prompt = f"""User input: "{user_input}"

Available workflows: {[w['name'] for w in workflows]}
Available plans: {[p['name'] for p in plans]}

Analyze the intent and respond with JSON only."""

        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            intent_text = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.2
            )
            intent_text = response.choices[0].message.content

        # Parse JSON
        intent_text = intent_text.strip()
        if "```json" in intent_text:
            intent_text = intent_text.split("```json")[1].split("```")[0].strip()
        elif "```" in intent_text:
            intent_text = intent_text.split("```")[1].split("```")[0].strip()

        try:
            intent = json.loads(intent_text)
        except json.JSONDecodeError:
            intent = {"action": "help", "confidence": 0}

        return intent

    def _handle_execute(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execution of existing plan."""
        plans = self.registry.list_plans()

        if not plans:
            return {
                "action": "create_workflow",
                "response": "I don't have any workflows set up yet. Let's create one!\n\nWhat data would you like to extract? (e.g., events from Luma, tweets from Twitter, etc.)"
            }

        # Find matching plan
        extracted_info = intent.get("extracted_info", {})
        matching_plan = None

        # Simple matching for now (can be improved with vector search)
        for plan in plans:
            plan_desc = plan["description"].lower()
            if any(word in plan_desc for word in user_input.lower().split()):
                matching_plan = plan
                break

        if not matching_plan:
            # Use first plan or ask user
            if len(plans) == 1:
                matching_plan = plans[0]
            else:
                plans_list = "\n".join([f"{i+1}. {p['name']} - {p['description']}"
                                       for i, p in enumerate(plans)])
                return {
                    "action": "execute",
                    "response": f"I have multiple workflows available:\n\n{plans_list}\n\nWhich one would you like to use?",
                    "needs_confirmation": True,
                    "plans": plans
                }

        # Execute the plan
        print(f"[INFO] Executing plan: {matching_plan['name']}")

        try:
            result = self.executor.execute_plan(matching_plan["name"], user_input)

            return {
                "action": "execute",
                "plan_used": matching_plan["name"],
                "result": result,
                "response": f"Here's what I found:\n\n{result}"
            }
        except Exception as e:
            return {
                "action": "execute",
                "error": str(e),
                "response": f"I encountered an error: {e}\n\nWould you like me to help you fix this?"
            }

    def _handle_create_workflow(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Guide user through creating a workflow."""
        extracted_info = intent.get("extracted_info", {})

        response = """Great! I'll help you create a new workflow. Let me ask you a few questions:

1. **What platform/website** are you trying to extract data from?
   (e.g., lu.ma, eventbrite.com, twitter.com)

2. **What specific data** do you want to extract?
   (e.g., event listings, user profiles, product information)

3. **Any specific filters** or requirements?
   (e.g., location, date range, keywords)

Please provide these details, and I'll create the workflow for you!"""

        # If we already extracted some info, pre-fill
        if extracted_info.get("platform"):
            response = f"""I understand you want to extract data from {extracted_info['platform']}.

What specific data would you like to extract from this platform?"""

        return {
            "action": "create_workflow",
            "response": response,
            "needs_confirmation": True,
            "extracted_info": extracted_info
        }

    def _handle_create_plan(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Guide user through creating an agent plan."""
        workflows = self.registry.list_workflows()

        if not workflows:
            return {
                "action": "create_workflow",
                "response": "Before creating an agent plan, you need at least one workflow. Let's create a workflow first!\n\nWhat data source would you like to work with?"
            }

        workflows_list = "\n".join([f"- {w['name']}: {w['description']}"
                                   for w in workflows])

        response = f"""I'll help you create an agent plan that orchestrates workflows with AI.

Available workflows:
{workflows_list}

What goal would you like to achieve? (e.g., "Find personalized events for users", "Aggregate news from multiple sources")"""

        return {
            "action": "create_plan",
            "response": response,
            "needs_confirmation": True,
            "available_workflows": workflows
        }

    def _handle_help(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Provide help and guidance."""
        workflows = self.registry.list_workflows()
        plans = self.registry.list_plans()

        stats = self.registry.stats()

        response = f"""Welcome to Agent School! I'm here to help you automate data extraction and processing.

**What I can do:**

1. **Find Data** - Extract information from websites and APIs
   Example: "Find tech events in San Francisco"

2. **Create Workflows** - Set up automated data extraction
   Example: "Create a workflow to scrape events from Luma"

3. **Build AI Agents** - Create intelligent automation
   Example: "Build an agent that finds personalized events"

**Current System Status:**
- {stats['deterministic_workflows']} workflow(s) available
- {stats['agent_plans']} agent plan(s) available

**What would you like to do?**
Just tell me in plain English, and I'll guide you through it!"""

        return {
            "action": "help",
            "response": response,
            "stats": stats
        }

    def _handle_list(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """List available workflows and plans."""
        workflows = self.registry.list_workflows()
        plans = self.registry.list_plans()

        response = "**Available Workflows:**\n"
        if workflows:
            for w in workflows:
                response += f"- {w['name']} ({w['method']}): {w['description']}\n"
        else:
            response += "No workflows created yet.\n"

        response += "\n**Available Agent Plans:**\n"
        if plans:
            for p in plans:
                response += f"- {p['name']}: {p['description']}\n"
        else:
            response += "No agent plans created yet.\n"

        response += "\nWhat would you like to do with these?"

        return {
            "action": "list",
            "response": response,
            "workflows": workflows,
            "plans": plans
        }
