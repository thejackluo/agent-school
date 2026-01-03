"""
Explorer - ReAct-style autonomous browser exploration using Browser Use

The exploration agent attempts tasks without a certification, using
documentation context and Browser Use's AI-native automation.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from .action_logger import ActionLogger, ActionLog
from .doc_store import DocStore


@dataclass
class ExplorationResult:
    """Result of an exploration attempt"""
    exploration_id: str
    task: str
    domain: str
    success: bool
    action_log: ActionLog
    final_result: Optional[str] = None
    error_message: Optional[str] = None
    cost_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_id": self.exploration_id,
            "task": self.task,
            "domain": self.domain,
            "success": self.success,
            "action_log": self.action_log.to_dict(),
            "final_result": self.final_result,
            "error_message": self.error_message,
            "cost_estimate": self.cost_estimate,
        }


class ExplorationAgent:
    """
    Uses Browser Use for goal-oriented browser exploration.
    
    This is Phase A of the certification loop - the agent uses
    ReAct-style reasoning to accomplish tasks when no certification exists.
    
    Features:
    - Injects documentation context for better guidance
    - Logs all actions for later synthesis
    - Tracks success/failure and costs
    """
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        doc_store: Optional[DocStore] = None,
        log_dir: str = "workflows",
        headless: bool = False,  # Show browser so user can sign in
    ):
        self.llm_provider = llm_provider
        self.doc_store = doc_store or DocStore()
        self.action_logger = ActionLogger()
        self.log_dir = log_dir
        self.headless = headless
        
        # Persistent browser profile - saves cookies/sessions between runs
        import os
        self.profile_dir = os.path.expanduser("~/.agent-school/browser_profile")
        os.makedirs(self.profile_dir, exist_ok=True)
    
    async def explore(
        self,
        task: str,
        domain: str,
        max_steps: int = 30,
        timeout_seconds: int = 300,  # 5 minutes - allows time for user auth
    ) -> ExplorationResult:
        """
        Explore and complete a task on a domain.
        
        Args:
            task: Natural language task description
            domain: Target domain (e.g., "mail.google.com")
            max_steps: Maximum browser actions to take
            timeout_seconds: Overall timeout
            
        Returns:
            ExplorationResult with success status and action log
        """
        exploration_id = str(uuid.uuid4())[:8]
        
        # Get relevant documentation for context
        doc_context = self.doc_store.retrieve_context(domain, task)
        
        # Build enhanced prompt with documentation
        enhanced_task = self._build_exploration_prompt(task, domain, doc_context)
        
        return await self._run_exploration(
            exploration_id=exploration_id,
            task=task,
            domain=domain,
            enhanced_task=enhanced_task,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
    
    async def explore_with_context(
        self,
        task: str,
        domain: str,
        doc_context: str,
        max_steps: int = 20,
    ) -> ExplorationResult:
        """
        Explore with explicitly provided documentation context.
        
        Useful when you want to override or supplement stored docs.
        """
        exploration_id = str(uuid.uuid4())[:8]
        enhanced_task = self._build_exploration_prompt(task, domain, doc_context)
        
        return await self._run_exploration(
            exploration_id=exploration_id,
            task=task,
            domain=domain,
            enhanced_task=enhanced_task,
            max_steps=max_steps,
        )
    
    def _build_exploration_prompt(
        self,
        task: str,
        domain: str,
        doc_context: str,
    ) -> str:
        """Build the exploration prompt with documentation context"""
        
        # Core auth instructions that apply to all prompts
        auth_instructions = """
CRITICAL - Authentication Handling:
- If you encounter a login/sign-in page, WAIT for the human user to complete authentication
- Do NOT try to find alternative services or workarounds
- Do NOT try to create new accounts
- The user will manually enter credentials, complete 2FA, biometrics, etc.
- After seeing "Sign in" or "Login", wait 30-60 seconds for user to authenticate
- Check periodically if you've been redirected to the authenticated interface
- Continue with the task ONLY after authentication is complete
"""
        
        # Smart contact/entity resolution
        resolution_instructions = """
SMART RESOLUTION - Finding People/Contacts:
- If the user mentions a PERSON'S NAME (not an email), you MUST find their actual email/contact
- In Gmail: Use the search bar to search for the person's name to find their email in past conversations
- In other apps: Use the app's search/contact lookup features
- Do NOT just type a name where an email is required - that will fail!
- Example: "email Jack Luo" -> Search Gmail for "Jack Luo" to find his email first, then compose
"""
        
        if doc_context and "No documentation available" not in doc_context:
            return f"""Task: {task}
Target: {domain}

Reference Documentation:
{doc_context}

{auth_instructions}

{resolution_instructions}

Instructions:
1. Use the documentation above to guide your actions
2. Navigate to the appropriate page if not already there
3. If you see a login page, WAIT for the user to sign in manually
4. If the task mentions a person's name, SEARCH to find their email/contact first
5. Complete the task step by step after authentication
6. Report success or failure with details
"""
        else:
            return f"""Task: {task}
Target: {domain}

{auth_instructions}

{resolution_instructions}

Instructions:
1. Navigate to {domain}
2. If you encounter a login page, WAIT for the user to authenticate manually
3. If the task mentions a person's name, SEARCH to find their email/contact first
4. After authentication, explore the interface to complete the task
5. Complete the task step by step
6. Report what you learned and whether you succeeded

"""

    
    async def _run_exploration(
        self,
        exploration_id: str,
        task: str,
        domain: str,
        enhanced_task: str,
        max_steps: int = 20,
        timeout_seconds: int = 120,
    ) -> ExplorationResult:
        """Run the actual exploration using Browser Use"""
        
        try:
            from browser_use import Agent, Browser
        except ImportError:
            raise ImportError("browser-use is required. Install with: uv add browser-use")
        
        start_time = datetime.now()
        success = False
        final_result = None
        error_message = None
        history = None
        browser = None
        
        try:
            # Initialize LLM based on provider
            llm = self._get_browser_use_llm()
            
            # Use persistent profile to save login sessions
            browser = Browser(
                headless=self.headless,
                user_data_dir=self.profile_dir,  # Saves cookies/sessions!
            )
            
            # Create agent with task and LLM
            agent = Agent(
                task=enhanced_task,
                browser=browser,
                llm=llm,
                max_steps=max_steps,
            )
            
            # Run with timeout
            try:
                history = await asyncio.wait_for(
                    agent.run(),
                    timeout=timeout_seconds
                )
                
                # Check for success
                if history:
                    success = self._evaluate_success(history)
                    final_result = self._extract_final_result(history)
                    
            except asyncio.TimeoutError:
                error_message = f"Exploration timed out after {timeout_seconds}s"
                
        except Exception as e:
            error_message = str(e)
            
        finally:
            try:
                if browser:
                    await browser.close()
            except:
                pass
        
        # Create action log
        action_log = self.action_logger.create_action_log(
            exploration_id=exploration_id,
            task=task,
            domain=domain,
            browser_history=history,
            success=success,
            final_result=final_result,
        )
        
        # Save the log
        self.action_logger.save_log(action_log, self.log_dir)
        
        # Estimate cost (rough: ~$0.01 per step for typical LLM)
        cost_estimate = len(action_log.actions) * 0.01
        
        return ExplorationResult(
            exploration_id=exploration_id,
            task=task,
            domain=domain,
            success=success,
            action_log=action_log,
            final_result=final_result,
            error_message=error_message,
            cost_estimate=cost_estimate,
        )
    
    def _get_browser_use_llm(self):
        """Get the appropriate LLM for Browser Use based on provider."""
        if self.llm_provider == "anthropic":
            try:
                from browser_use import ChatAnthropic
                return ChatAnthropic(model="claude-sonnet-4-20250514")
            except ImportError:
                # Fallback: try langchain
                from langchain_anthropic import ChatAnthropic as LangchainAnthropic
                return LangchainAnthropic(model="claude-sonnet-4-20250514")
        else:
            try:
                from browser_use import ChatOpenAI
                return ChatOpenAI(model="gpt-4o")
            except ImportError:
                from langchain_openai import ChatOpenAI as LangchainOpenAI
                return LangchainOpenAI(model="gpt-4o")
    
    def _evaluate_success(self, history: Any) -> bool:
        """
        Evaluate if the exploration was successful.
        
        This is a heuristic - in practice you might want
        task-specific success criteria.
        """
        if history is None:
            return False
        
        # Check if history indicates success
        if hasattr(history, 'is_done') and history.is_done:
            if hasattr(history, 'is_successful'):
                return history.is_successful
            return True
        
        # Check for error indicators
        if hasattr(history, 'errors') and history.errors:
            return False
        
        # Default: assume success if we got results
        return True
    
    def _extract_final_result(self, history: Any) -> Optional[str]:
        """Extract the final result from exploration history"""
        if history is None:
            return None
        
        if hasattr(history, 'final_result'):
            return str(history.final_result)
        
        if hasattr(history, 'extracted_content'):
            return str(history.extracted_content)
        
        # Try to get last action result
        if hasattr(history, 'history') and history.history:
            last_step = history.history[-1]
            if hasattr(last_step, 'result'):
                return str(last_step.result)
        
        return None
