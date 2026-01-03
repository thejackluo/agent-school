"""
Router - Natural Language Intent Detection (Layer 3)

Routes user requests to appropriate actions:
- Create new workflow
- Create new agent plan
- Execute existing plan
- Execute cached certification (deterministic)
- Explore and certify new browser tasks
- Show help/guidance

Now with Agent Certification support:
- Checks for existing certifications before LLM-heavy exploration
- Executes certified workflows at near-zero cost
- Falls back to exploration when no certification exists
"""

import json
import asyncio
from typing import Dict, Any, Literal, Optional, List
from anthropic import Anthropic
from openai import OpenAI

from ..config import Config
from .registry import Registry
from .executor import Executor

# Certification module imports
try:
    from ..certification import (
        CertificationRegistry,
        CertifiedExecutor,
        ExplorationAgent,
        CertificationSynthesizer,
        DriftDetector,
        DocIngester,
    )
    CERTIFICATION_AVAILABLE = True
except ImportError:
    CERTIFICATION_AVAILABLE = False

# Entity Registry for instant name→identity lookups
try:
    from ..entities import EntityRegistry
    ENTITY_REGISTRY_AVAILABLE = True
except ImportError:
    ENTITY_REGISTRY_AVAILABLE = False

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
        
        # Initialize certification components
        if CERTIFICATION_AVAILABLE:
            self.cert_registry = CertificationRegistry()
            self.cert_executor = CertifiedExecutor(headless=True)
            self.explorer = ExplorationAgent(llm_provider=llm_provider)
            self.synthesizer = CertificationSynthesizer(llm_provider=llm_provider)
            self.drift_detector = DriftDetector(self.explorer, self.synthesizer)
        else:
            self.cert_registry = None
        
        # Initialize entity registry for name→identity lookups
        if ENTITY_REGISTRY_AVAILABLE:
            self.entity_registry = EntityRegistry()
        else:
            self.entity_registry = None

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
        elif action == "browser_task":
            return self._handle_browser_task(user_input, intent)
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
                "response": "I'm not sure what you want to do. Could you clarify? I can help you:\n- Find events or data\n- Automate browser tasks (Gmail, Google Docs, etc.)\n- Create new workflows\n- Set up automation\n\nWhat would you like to do?"
            }

    def _analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to analyze user intent."""
        # Get available workflows and plans
        workflows = self.registry.list_workflows()
        plans = self.registry.list_plans()

        system_prompt = """You are an AI assistant that analyzes user intent for a workflow automation system.

Available actions:
1. "execute" - User wants to run an existing workflow (e.g., "find events in SF")
2. "browser_task" - User wants to perform a browser-based task on a website (e.g., "compose an email", "create a Google Doc", "update a spreadsheet")
3. "create_workflow" - User wants to create a new data extraction workflow
4. "create_plan" - User wants to create an agent orchestration plan
5. "list" - User wants to see what's available
6. "help" - User needs guidance

Detect "browser_task" when users want to:
- Interact with web applications (Gmail, Google Docs, Sheets, social media, etc.)
- Perform actions like compose, send, create, edit, post, click, navigate
- Automate repetitive browser interactions

Analyze the user's input and respond with JSON:
{
  "action": "execute" | "browser_task" | "create_workflow" | "create_plan" | "list" | "help",
  "confidence": 0.0 to 1.0,
  "reasoning": "why you chose this action",
  "extracted_info": {
    // For execute: {task, platform, filters}
    // For browser_task: {task, domain, parameters}
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
    
    def _handle_browser_task(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle browser automation tasks with certification routing.
        
        The Learning Loop:
        1. Check if certification exists for this task
        2. If yes: Execute certification (near-zero cost)
        3. If no: Explore with ReAct, synthesize certification, save for future
        """
        if not CERTIFICATION_AVAILABLE or not self.cert_registry:
            return {
                "action": "help",
                "response": "Browser automation is not available. Please install browser-use with: uv add browser-use"
            }
        
        extracted_info = intent.get("extracted_info", {})
        task = extracted_info.get("task", user_input)
        domain = extracted_info.get("domain", self._infer_domain(user_input))
        
        print(f"[INFO] Browser task: {task}")
        print(f"[INFO] Domain: {domain}")
        
        # Check for existing certification - try both extracted task and full user_input
        cert = self.cert_registry.find_for_task(task, domain)
        if not cert:
            # Patterns like "send.*email.*to.*" need full user input to match
            cert = self.cert_registry.find_for_task(user_input, domain)
        
        if cert:
            # Phase C: Execute cached certification
            print(f"[INFO] Found certification: {cert.name} (v{cert.version})")
            return self._execute_certification(cert, user_input, extracted_info)
        else:
            # Phase A + B: Explore and synthesize
            print(f"[INFO] No certification found. Starting exploration...")
            return self._explore_and_certify(task, domain, user_input)
    
    def _execute_certification(
        self, 
        cert, 
        user_input: str, 
        extracted_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a cached certification with drift detection."""
        try:
            # Extract parameters from user input
            params = self.cert_executor.bind_params_from_input(cert, user_input)
            params.update(extracted_info.get("parameters", {}))
            
            # Check for missing required parameters
            missing_params = []
            for param in cert.parameters:
                if param.required and param.name not in params:
                    missing_params.append(param.name)
            
            if missing_params:
                # Try to resolve missing params from Entity Registry
                resolved_any = False
                if self.entity_registry:
                    resolved_any = self._try_resolve_entities(
                        user_input, params, missing_params, cert
                    )
                
                # Recheck after entity resolution
                if resolved_any:
                    missing_params = [
                        p.name for p in cert.parameters 
                        if p.required and p.name not in params
                    ]
                
                if missing_params:
                    # Still missing params - fall back to exploration
                    print(f"[INFO] Missing required params: {missing_params}. Falling back to exploration...")
                    return self._explore_and_certify(
                        extracted_info.get("task", user_input), 
                        extracted_info.get("domain", self._infer_domain(user_input)),
                        user_input
                    )
                else:
                    print(f"[INFO] 🚀 Resolved entities from registry - instant execution!")
            
            # Execute with self-healing
            result, updated_cert = asyncio.get_event_loop().run_until_complete(
                self.drift_detector.execute_with_healing(
                    cert, params, self.cert_executor
                )
            )
            
            # Update certification if healed
            if updated_cert:
                self.cert_registry.register(updated_cert)
                print(f"[INFO] Certification healed: v{updated_cert.version}")
            
            if result.success:
                return {
                    "action": "browser_task",
                    "certification_used": cert.name,
                    "result": result.final_result,
                    "response": f"✅ Completed: {cert.task_description}\n\n{result.final_result or 'Task executed successfully.'}",
                    "execution_cost": "$0.00 (cached certification)"
                }
            else:
                return {
                    "action": "browser_task",
                    "certification_used": cert.name,
                    "error": result.error_message,
                    "response": f"❌ Task failed: {result.error_message}\n\nWould you like me to re-learn this workflow?"
                }
                
        except KeyError as ke:
            # Missing parameter - fall back to exploration
            print(f"[INFO] Missing param {ke}. Falling back to exploration...")
            return self._explore_and_certify(
                extracted_info.get("task", user_input),
                extracted_info.get("domain", self._infer_domain(user_input)),
                user_input
            )
        except Exception as e:
            return {
                "action": "browser_task",
                "error": str(e),
                "response": f"❌ Error executing certification: {e}"
            }

    
    def _explore_and_certify(self, task: str, domain: str, user_input: str) -> Dict[str, Any]:
        """Explore a task and create a certification."""
        try:
            # Run exploration with FULL user input (not just extracted task summary)
            # The user_input contains all the details like recipients, content, etc.
            result = asyncio.get_event_loop().run_until_complete(
                self.explorer.explore(user_input, domain)  # Use user_input, not task!
            )
            
            if not result.success:
                return {
                    "action": "browser_task",
                    "exploration_id": result.exploration_id,
                    "error": result.error_message,
                    "response": f"❌ Exploration failed: {result.error_message}\n\nThis task may require additional guidance or permissions."
                }
            
            # Try to synthesize certification (may fail on complex histories)
            cert_saved = False
            try:
                if result.action_log and result.action_log.actions:
                    print(f"[INFO] Exploration successful. Synthesizing certification...")
                    cert = self.synthesizer.synthesize(result.action_log)
                    
                    # Save certification
                    cert_path = self.cert_registry.register(cert)
                    print(f"[INFO] Certification saved: {cert_path}")
                    cert_saved = True
                else:
                    print(f"[INFO] No actions logged, skipping certification synthesis")
            except Exception as synth_err:
                print(f"[WARN] Certification synthesis failed: {synth_err}")
                # Continue - task may have still succeeded
            
            # Learn entities from exploration for instant lookups next time
            self._learn_entities_from_exploration(result, user_input)
            
            response_suffix = "\n\n📚 I've learned this workflow! Next time will be instant." if cert_saved else ""
            
            return {
                "action": "browser_task",
                "exploration_id": result.exploration_id,
                "result": result.final_result,
                "response": f"✅ Completed: {task}\n\n{result.final_result or 'Task executed successfully.'}{response_suffix}",
                "execution_cost": f"~${result.cost_estimate:.2f} (exploration)"
            }
            
        except Exception as e:
            return {
                "action": "browser_task",
                "error": str(e),
                "response": f"❌ Error during exploration: {e}"
            }

    def _try_resolve_entities(
        self,
        user_input: str,
        params: Dict[str, Any],
        missing_params: List[str],
        cert: Any,
    ) -> bool:
        """
        Try to resolve missing parameters using Entity Registry and smart extraction.
        
        Looks for names in user_input and resolves them to platform identities.
        Also extracts subject/body from 'about X' patterns.
        Returns True if any params were resolved.
        """
        import re
        
        resolved_any = False
        
        # Check for email-related params
        email_param_names = ["recipient_email", "email", "to", "recipient"]
        for param_name in missing_params:
            if param_name in email_param_names:
                # Try to extract a name from the input
                # Look for patterns like "to [Name]" or "[Name] about"
                name_patterns = [
                    r'(?:to|email|send.*?to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:about|regarding)',
                ]
                
                for pattern in name_patterns:
                    match = re.search(pattern, user_input)
                    if match:
                        name = match.group(1)
                        # Try to find in registry
                        identity = self.entity_registry.get_identity(name, "gmail")
                        if identity:
                            params[param_name] = identity
                            print(f"[INFO] 📇 Resolved '{name}' → {identity} from Entity Registry")
                            resolved_any = True
                            break
                
                if param_name in params:
                    continue  # Already resolved
        
        # Extract email subject from "about X" pattern
        if "email_subject" in missing_params:
            about_match = re.search(r'about\s+(.+?)(?:\s*$|\s+and\s+)', user_input, re.IGNORECASE)
            if about_match:
                topic = about_match.group(1).strip()
                # Clean up and capitalize
                subject = topic.title() if len(topic) < 50 else topic[:50].title()
                params["email_subject"] = subject
                print(f"[INFO] 📝 Extracted subject: '{subject}'")
                resolved_any = True
        
        # Use LLM to generate email body if still missing subject or body
        if "email_body" in missing_params or "email_subject" in missing_params:
            generated = self._llm_extract_email_params(user_input, params)
            if generated:
                if "email_subject" not in params and "subject" in generated:
                    params["email_subject"] = generated["subject"]
                    print(f"[INFO] 🤖 LLM generated subject: '{generated['subject']}'")
                    resolved_any = True
                if "email_body" not in params and "body" in generated:
                    params["email_body"] = generated["body"]
                    print(f"[INFO] 🤖 LLM generated body")
                    resolved_any = True
        
        return resolved_any
    
    def _llm_extract_email_params(
        self, 
        user_input: str, 
        existing_params: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Use LLM to generate email subject and body from user request."""
        try:
            prompt = f"""Given this user request to send an email:
"{user_input}"

Generate an appropriate email subject line and body.
The recipient email is: {existing_params.get('recipient_email', 'unknown')}

Respond in JSON format:
{{"subject": "...", "body": "..."}}

Keep it natural and friendly. The body should be 2-4 sentences."""

            if self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",  # Fast model for quick extraction
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast model
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            print(f"[WARN] LLM extraction failed: {e}")
        return None
    
    def _learn_entities_from_exploration(
        self,
        result: Any,  # ExplorationResult
        user_input: str,
    ) -> None:
        """
        Learn entities from a successful exploration.
        
        Extracts discovered identities (emails, handles) and registers
        them in the Entity Registry for future instant lookups.
        """
        if not self.entity_registry:
            return
        
        if not result.success or not result.action_log:
            return
        
        import re
        
        # Look for email addresses typed during exploration
        for action in result.action_log.actions:
            if action.action_type in ["input", "type"]:
                value = action.value or ""
                
                # Check if it looks like an email
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', value)
                if email_match:
                    email = email_match.group(0)
                    
                    # Try to find the name that was searched for
                    # Look in the user_input for names
                    name_patterns = [
                        r'(?:to|email|send.*?to|message|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                    ]
                    
                    for pattern in name_patterns:
                        name_match = re.search(pattern, user_input)
                        if name_match:
                            name = name_match.group(1)
                            
                            # Register the entity
                            entity = self.entity_registry.register(
                                canonical_name=name,
                                platform="gmail",
                                identity=email,
                                metadata={"learned_from": result.exploration_id}
                            )
                            print(f"[INFO] 📚 Learned entity: {name} → {email}")
                            return  # Only learn one per exploration
    
    def _infer_domain(self, user_input: str) -> str:
        """Infer domain from user input."""
        input_lower = user_input.lower()
        
        # Common domain mappings
        domain_keywords = {
            "mail.google.com": ["gmail", "email", "mail", "inbox", "compose"],
            "docs.google.com": ["google doc", "gdoc", "document"],
            "sheets.google.com": ["spreadsheet", "sheet", "google sheet"],
            "twitter.com": ["twitter", "tweet", "x.com"],
            "linkedin.com": ["linkedin", "connection"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in input_lower for kw in keywords):
                return domain
        
        # Default to a generic domain
        return "unknown"

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
